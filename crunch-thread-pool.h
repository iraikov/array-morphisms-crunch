/* crunch-thread-pool.h
 * A persistent pool of POSIX worker threads for running kernels.
 *
 * am_pool_run_chunks(run, jobs, T) calls run(jobs, k) for k = 0..T-1
 * and returns when every call has finished.  Chunk 0 runs on the
 * calling thread; chunk k > 0 runs on worker k of the pool.  jobs is
 * an array of per-chunk argument records on the caller's stack, which
 * stays valid because the caller does not return before the last
 * chunk is done.
 *
 * The pool is created lazily.  It grows to the largest T requested so
 * far, up to AM_POOL_MAX_THREADS - 1 workers, and shrinks only through
 * am_pool_shutdown.  Workers are created with every signal blocked, so
 * signal handlers only run on threads that existed before the pool.
 *
 * Each worker has its own task id.  A dispatch wakes workers 1..T-1
 * by giving each a new task id.  The caller counts down the chunks
 * still running in `remaining`.
 *
 * Waiting spins first and then sleeps: a worker that has finished a
 * chunk polls its task id for a bounded time (AM_CRUNCH_SPIN_US
 * microseconds, 20 by default, 0 to disable) before sleeping on its
 * condition variable, and the caller does the same while waiting for
 * `remaining` to reach zero.  Back-to-back dispatches therefore find the
 * workers awake, and an idle pool costs no CPU once the spin time has
 * passed.  Spinning is turned off for a dispatch that asks for more
 * threads than there are processors online, where it would only take
 * time away from the threads doing the work.
 *
 * The pool assumes one dispatching thread at a time, which holds for the
 * CHICKEN runtime.  A dispatch that finds another in progress runs all
 * of its chunks on the calling thread instead of waiting.
 *
 * After fork the child has no workers; its first threaded dispatch
 * creates them again.
 */

#ifndef CRUNCH_THREAD_POOL_H
#define CRUNCH_THREAD_POOL_H

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define AM_POOL_MAX_THREADS 64
#define AM_POOL_DEFAULT_SPIN_US 20
#define AM_POOL_MAX_SPIN_US 1000000

#if defined(__x86_64__) || defined(__i386__)
#define AM_POOL_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define AM_POOL_PAUSE() __asm__ __volatile__("yield")
#else
#define AM_POOL_PAUSE() ((void)0)
#endif

typedef void (*am_pool_run)(void *jobs, long k);

/* One worker.  Aligned to a cache line so that task ids of different
   workers do not share one. */
typedef struct {
  _Alignas(64) atomic_uint go;   /* task, changed by the caller to wake */
  atomic_int sleeping;           /* 1 while blocked, or about to block, on cv */
  unsigned seen;                 /* the worker's task when it was created */
  long k;                        /* chunk index served by this worker */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  pthread_t th;
} am_pool_worker;

enum { AM_POOL_STAT_DISPATCHES, AM_POOL_STAT_SPIN_HITS, AM_POOL_STAT_WAKEUPS,
       AM_POOL_STAT_CALLER_SLEEPS, AM_POOL_NSTATS };

static struct {
  am_pool_worker w[AM_POOL_MAX_THREADS];   /* w[1] .. w[nworkers] are live */
  int nworkers;
  unsigned task_id;
  atomic_int stop;
  atomic_int busy;

  /* The current job; written before the tasks are handed out. */
  am_pool_run run;
  void *jobs;

  atomic_long remaining;
  atomic_int caller_sleeping;
  pthread_mutex_t caller_mu;
  pthread_cond_t caller_cv;

  atomic_long spin_ns;       /* setting */
  atomic_long cur_spin_ns;   /* setting as applied to the current dispatch */
  long nprocs;

  atomic_ulong stats[AM_POOL_NSTATS];
} am_pool;

static pthread_once_t am_pool_once = PTHREAD_ONCE_INIT;

static void am_pool_init_sync(void) {
  for (int k = 0; k < AM_POOL_MAX_THREADS; k++) {
    pthread_mutex_init(&am_pool.w[k].mu, NULL);
    pthread_cond_init(&am_pool.w[k].cv, NULL);
    atomic_store(&am_pool.w[k].sleeping, 0);
  }
  pthread_mutex_init(&am_pool.caller_mu, NULL);
  pthread_cond_init(&am_pool.caller_cv, NULL);
  atomic_store(&am_pool.caller_sleeping, 0);
}

/* In the child of fork only the forking thread exists.  Forget the
   workers and give every lock a fresh state, since a worker may have
   held one at the moment of the fork. */
static void am_pool_atfork_child(void) {
  am_pool_init_sync();
  am_pool.nworkers = 0;
  atomic_store(&am_pool.stop, 0);
  atomic_store(&am_pool.busy, 0);
  atomic_store(&am_pool.remaining, 0);
}

static void am_pool_init(void) {
  const char *env = getenv("AM_CRUNCH_SPIN_US");
  long us = AM_POOL_DEFAULT_SPIN_US;
  if (env && *env) {
    char *end;
    long v = strtol(env, &end, 10);
    if (*end == '\0' && v >= 0) us = v > AM_POOL_MAX_SPIN_US ? AM_POOL_MAX_SPIN_US : v;
  }
  atomic_store(&am_pool.spin_ns, us * 1000);
  am_pool.nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  if (am_pool.nprocs < 1) am_pool.nprocs = 1;
  am_pool_init_sync();
  pthread_atfork(NULL, NULL, am_pool_atfork_child);
}

static void am_pool_ensure_init(void) {
  pthread_once(&am_pool_once, am_pool_init);
}

static int64_t am_pool_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static void am_pool_count(int stat) {
  atomic_fetch_add_explicit(&am_pool.stats[stat], 1, memory_order_relaxed);
}

/* Wait until worker w's task id differs from seen; return the new task id.
   The worker sets `sleeping` and then rereads the task id, while the
   caller sets the task id and then reads `sleeping` (both sequentially
   consistent), so at least one of them sees the other's store and no
   wakeup is lost. */
static unsigned am_pool_wait_task(am_pool_worker *w, unsigned seen) {
  unsigned g;
  long spin = atomic_load_explicit(&am_pool.cur_spin_ns, memory_order_relaxed);
  if (spin > 0) {
    int64_t t0 = am_pool_now_ns();
    for (unsigned i = 1;; i++) {
      g = atomic_load_explicit(&w->go, memory_order_acquire);
      if (g != seen) { am_pool_count(AM_POOL_STAT_SPIN_HITS); return g; }
      AM_POOL_PAUSE();
      if ((i & 63) == 0 && am_pool_now_ns() - t0 >= spin) break;
    }
  }
  pthread_mutex_lock(&w->mu);
  atomic_store(&w->sleeping, 1);
  while ((g = atomic_load(&w->go)) == seen)
    pthread_cond_wait(&w->cv, &w->mu);
  atomic_store(&w->sleeping, 0);
  pthread_mutex_unlock(&w->mu);
  am_pool_count(AM_POOL_STAT_WAKEUPS);
  return g;
}

static void *am_pool_worker_main(void *p) {
  am_pool_worker *w = (am_pool_worker *)p;
  unsigned seen = w->seen;
  for (;;) {
    seen = am_pool_wait_task(w, seen);
    if (atomic_load(&am_pool.stop)) break;
    am_pool.run(am_pool.jobs, w->k);
    /* The same pairing as in am_pool_wait_task, with the caller. */
    if (atomic_fetch_sub(&am_pool.remaining, 1) == 1 &&
        atomic_load(&am_pool.caller_sleeping)) {
      pthread_mutex_lock(&am_pool.caller_mu);
      pthread_cond_signal(&am_pool.caller_cv);
      pthread_mutex_unlock(&am_pool.caller_mu);
    }
  }
  return NULL;
}

/* Grow the pool to n workers, or as close to it as pthread_create
   allows.  Returns the number of live workers. */
static int am_pool_grow(int n) {
  if (n > AM_POOL_MAX_THREADS - 1) n = AM_POOL_MAX_THREADS - 1;
  if (am_pool.nworkers < n) {
    sigset_t all, old;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &old);
    for (int k = am_pool.nworkers + 1; k <= n; k++) {
      am_pool_worker *w = &am_pool.w[k];
      w->k = k;
      w->seen = atomic_load(&w->go);
      atomic_store(&w->sleeping, 0);
      if (pthread_create(&w->th, NULL, am_pool_worker_main, w) != 0) break;
      am_pool.nworkers = k;
    }
    pthread_sigmask(SIG_SETMASK, &old, NULL);
  }
  return am_pool.nworkers;
}

static void am_pool_wake(am_pool_worker *w, unsigned task_id) {
  atomic_store(&w->go, task_id);
  if (atomic_load(&w->sleeping)) {
    pthread_mutex_lock(&w->mu);
    pthread_cond_signal(&w->cv);
    pthread_mutex_unlock(&w->mu);
  }
}

static void am_pool_run_chunks(am_pool_run run, void *jobs, long T) {
  long P, k;
  if (T <= 1) {
    if (T == 1) run(jobs, 0);
    return;
  }
  if (atomic_exchange(&am_pool.busy, 1)) {
    for (k = 0; k < T; k++) run(jobs, k);
    return;
  }
  am_pool_ensure_init();
  P = am_pool_grow((int)(T - 1));
  if (P > T - 1) P = T - 1;
  am_pool_count(AM_POOL_STAT_DISPATCHES);

  atomic_store_explicit(&am_pool.cur_spin_ns,
                        T > am_pool.nprocs ? 0 : atomic_load(&am_pool.spin_ns),
                        memory_order_relaxed);
  am_pool.run = run;
  am_pool.jobs = jobs;
  atomic_store(&am_pool.remaining, P);
  am_pool.task_id++;
  for (k = 1; k <= P; k++) am_pool_wake(&am_pool.w[k], am_pool.task_id);

  run(jobs, 0);
  for (k = P + 1; k < T; k++) run(jobs, k);   /* workers that could not be created */

  if (atomic_load_explicit(&am_pool.remaining, memory_order_acquire) > 0) {
    long spin = atomic_load_explicit(&am_pool.cur_spin_ns, memory_order_relaxed);
    int done = 0;
    if (spin > 0) {
      int64_t t0 = am_pool_now_ns();
      for (unsigned i = 1;; i++) {
        if (atomic_load_explicit(&am_pool.remaining, memory_order_acquire) == 0) {
          done = 1;
          break;
        }
        AM_POOL_PAUSE();
        if ((i & 63) == 0 && am_pool_now_ns() - t0 >= spin) break;
      }
    }
    if (!done) {
      pthread_mutex_lock(&am_pool.caller_mu);
      atomic_store(&am_pool.caller_sleeping, 1);
      while (atomic_load(&am_pool.remaining) > 0)
        pthread_cond_wait(&am_pool.caller_cv, &am_pool.caller_mu);
      atomic_store(&am_pool.caller_sleeping, 0);
      pthread_mutex_unlock(&am_pool.caller_mu);
      am_pool_count(AM_POOL_STAT_CALLER_SLEEPS);
    }
  }
  atomic_store(&am_pool.busy, 0);
}

/* Stop and join every worker.  Returns the number joined, or -1 if a
   dispatch is in progress, in which case nothing is done. */
static int am_pool_shutdown(void) {
  int n;
  if (atomic_exchange(&am_pool.busy, 1)) return -1;
  n = am_pool.nworkers;
  if (n > 0) {
    atomic_store(&am_pool.stop, 1);
    am_pool.task_id++;
    for (int k = 1; k <= n; k++) am_pool_wake(&am_pool.w[k], am_pool.task_id);
    for (int k = 1; k <= n; k++) pthread_join(am_pool.w[k].th, NULL);
    am_pool.nworkers = 0;
    atomic_store(&am_pool.stop, 0);
  }
  atomic_store(&am_pool.busy, 0);
  return n;
}

static int am_pool_size(void) { return am_pool.nworkers; }

static long am_pool_spin_us(void) {
  am_pool_ensure_init();
  return atomic_load(&am_pool.spin_ns) / 1000;
}

static void am_pool_set_spin_us(long us) {
  am_pool_ensure_init();
  if (us < 0) us = 0;
  if (us > AM_POOL_MAX_SPIN_US) us = AM_POOL_MAX_SPIN_US;
  atomic_store(&am_pool.spin_ns, us * 1000);
}

static unsigned long am_pool_stat(int i) {
  return (i >= 0 && i < AM_POOL_NSTATS) ? atomic_load(&am_pool.stats[i]) : 0;
}

#endif /* CRUNCH_THREAD_POOL_H */
