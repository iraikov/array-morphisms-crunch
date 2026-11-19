/* pool-tsan.c -- a stand-alone test of crunch-thread-pool.h.
 *
 * Runs many dispatches of a simple element-wise chunk kernel with
 * varying chunk counts, both with spinning and with sleeping only, with
 * shutdowns and regrowth in between, and from a forked child, and checks
 * every result against a serial reference.  Built with
 * -fsanitize=thread by run.sh, it also lets ThreadSanitizer look for
 * data races in the pool.
 */

#include "../../crunch-thread-pool.h"
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>

typedef struct { long start, end; const float *in; float *out; } job;

static void run_job(void *p, long k) {
  job *j = (job *)p + k;
  for (long i = j->start; i < j->end; i++) j->out[i] = 2.0f * j->in[i] + 1.0f;
}

static void dispatch(long n, const float *in, float *out, long T) {
  job jobs[AM_POOL_MAX_THREADS];
  for (long k = 0; k < T; k++) {
    jobs[k].start = n * k / T; jobs[k].end = n * (k + 1) / T;
    jobs[k].in = in; jobs[k].out = out;
  }
  am_pool_run_chunks(run_job, jobs, T);
}

static int failures = 0;

static void check(const char *what, int ok) {
  if (!ok) { printf("FAIL: %s\n", what); failures++; }
}

static int run_round(long iters, long maxT) {
  enum { N = 20011 };
  static float in[N], out[N];
  for (long i = 0; i < N; i++) in[i] = (float)i * 0.5f;
  for (long r = 0; r < iters; r++) {
    long n = 1 + (r * 7919) % N;
    long T = 1 + r % maxT;
    memset(out, 0, sizeof out);
    dispatch(n, in, out, T);
    for (long i = 0; i < N; i++) {
      float want = i < n ? 2.0f * in[i] + 1.0f : 0.0f;
      if (out[i] != want) return 0;
    }
  }
  return 1;
}

int main(void) {
  check("T = 0 and T = 1 do not create workers",
        (dispatch(10, NULL, NULL, 0), am_pool_size() == 0));
  check("spinning, T up to 8", run_round(3000, 8));
  check("pool grew to 7 workers", am_pool_size() == 7);
  check("smaller T does not shrink the pool", run_round(200, 3) && am_pool_size() == 7);

  am_pool_set_spin_us(0);
  check("spin setting reads back", am_pool_spin_us() == 0);
  unsigned long w0 = am_pool_stat(AM_POOL_STAT_WAKEUPS);
  check("sleeping only, T up to 8", run_round(2000, 8));
  check("sleeping workers were woken", am_pool_stat(AM_POOL_STAT_WAKEUPS) > w0);

  check("shutdown joins 7 workers", am_pool_shutdown() == 7);
  check("shutdown again joins none", am_pool_shutdown() == 0);
  check("pool is empty", am_pool_size() == 0);

  am_pool_set_spin_us(50);
  check("regrown pool, oversubscribed T up to 64", run_round(300, 64));
  check("pool grew to 63 workers", am_pool_size() == 63);
  check("back to T up to 8 after oversubscription", run_round(1000, 8));

  fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) {
    int ok = am_pool_size() == 0 && run_round(500, 8) && am_pool_size() == 7;
    _exit(ok ? 0 : 1);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  check("forked child recreates the pool and gets correct results",
        WIFEXITED(status) && WEXITSTATUS(status) == 0);
  check("parent pool still works after fork", run_round(500, 8));

  check("final shutdown", am_pool_shutdown() == 63);
  printf("pool-tsan: %d failure(s); dispatches %lu, spin hits %lu, wakeups %lu, caller sleeps %lu\n",
         failures, am_pool_stat(AM_POOL_STAT_DISPATCHES), am_pool_stat(AM_POOL_STAT_SPIN_HITS),
         am_pool_stat(AM_POOL_STAT_WAKEUPS), am_pool_stat(AM_POOL_STAT_CALLER_SLEEPS));
  return failures != 0;
}
