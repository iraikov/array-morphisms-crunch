;;; crunch-thread-dispatch.scm
;;; Applies a crunch chunk kernel across an array on several POSIX threads.
;;;
;;; A chunk kernel is a Crunch procedure (kernel start end in out) that
;;; processes elements [start, end) of the SRFI-4 vectors in and out.
;;;
;;; crunch-dispatch-f32 and crunch-dispatch-f64 split [0, n) into T
;;; contiguous chunks of nearly equal size, run the first chunk on the
;;; calling thread and each other chunk on a worker thread, and return
;;; when every chunk is done.  Since the chunks are disjoint and each
;;; element is computed on its own, the result is the same for every
;;; thread count.
;;;
;;; The number of threads is
;;;
;;;   T = max(1, min(threads, n div min-chunk, 64))
;;;
;;; so an array shorter than two chunks runs on the calling thread alone.
;;; threads defaults to the value of (crunch-thread-count) and min-chunk
;;; to (crunch-thread-min-chunk).
;;;
;;; crunch-dispatch-n runs the kernels that array-morphisms-crunch-codegen
;;; generates.  They take any number of numeric vectors, up to
;;; crunch-dispatch-n-max-vectors, of any element type, through the job
;;; record of crunch-dispatch-n.h, and each comes with a run procedure that
;;; passes a job's arguments to the kernel.
;;;
;;; The workers belong to a persistent pool (crunch-thread-pool.h).  The
;;; pool is created at the first dispatch with T > 1, grows to the
;;; largest T requested, and is kept until crunch-thread-pool-shutdown!
;;; or the end of the process.  Between dispatches a worker spins for
;;; (crunch-thread-pool-spin-us) microseconds, so that the next kernel of
;;; a replay finds it awake, and then sleeps.  The spin time defaults to
;;; the environment variable AM_CRUNCH_SPIN_US, or 20 microseconds.
;;;
;;; The worker threads do not touch the CHICKEN heap or call into the
;;; CHICKEN runtime.  The dispatcher describes the two vectors to the
;;; kernel with block descriptors on its own stack, pointing at the
;;; vectors' storage; the descriptors have a reference count of -1, so
;;; the threads can share them without synchronisation.  The calling
;;; thread stays inside the one foreign call until every chunk has
;;; finished, so no garbage collection can move the vectors in the
;;; mean time.
;;;
;;; All signals are blocked in the workers, so CHICKEN's signal
;;; handler only ever runs on the thread that owns the runtime.  If a
;;; worker cannot be created, its chunk runs on the calling thread
;;; instead.  A child process created by fork starts with an empty pool.
;;;
;;;   (import array-morphisms-crunch-threads)
;;;   (crunch-thread-count 8)
;;;   (crunch-dispatch-f32 n in out kernel-pointer)

(module array-morphisms-crunch-threads

  (crunch-dispatch-f32
   crunch-dispatch-f64
   crunch-dispatch4-f32
   crunch-dispatch4-f64
   crunch-dispatch-n
   crunch-dispatch-n-max-vectors
   crunch-thread-count
   crunch-thread-min-chunk
   crunch-available-processors
   crunch-max-threads
   crunch-thread-pool-size
   crunch-thread-pool-shutdown!
   crunch-thread-pool-stats
   crunch-thread-pool-spin-us
   crunch-thread-pool-spin-us-set!)

  (import scheme (chicken base) (chicken foreign) (chicken number-vector)
          (chicken process-context) (only (chicken memory) pointer?)
          (only (scheme base) make-parameter))

  (foreign-declare #<<EOC
#define CRUNCH_EMBEDDED
#include "crunch.h"
#include "crunch-thread-pool.h"
#include "crunch-dispatch-n.h"

#define AM_MAX_THREADS AM_POOL_MAX_THREADS

/* Point descriptor b at the element storage of SRFI-4 vector vec (slot 1
   of the vector structure is its bytevector). */
#define AM_DEFINE_WRAP(t, et)                                                \
static void am_wrap_ ## t(C_word vec, crunch_ ## t ## vector_block *b) {     \
  C_word bv = C_block_item(vec, 1);                                          \
  b->rc    = -1;                                                             \
  b->flags = CRUNCH_BLOCK;                                                   \
  b->size  = C_header_size(bv);                                              \
  b->len   = b->size / sizeof(et);                                           \
  b->data  = (et *)C_data_pointer(bv);                                       \
}

/* The thread count for n units: max(1, min(nthreads, n div min_chunk, 64)). */
static long am_thread_count(long n, int nthreads, long min_chunk) {
  long T = nthreads;
  if (min_chunk < 1) min_chunk = 1;
  if (T > n / min_chunk) T = n / min_chunk;
  if (T > AM_MAX_THREADS) T = AM_MAX_THREADS;
  if (T < 1) T = 1;
  return T;
}

/* am_dispatch_<t>: see the module comment.  Returns the thread count T. */
#define AM_DEFINE_DISPATCH(t, et)                                            \
typedef void (*am_chunk_ ## t)(crunch_integer, crunch_integer,              \
                               crunch_ ## t ## vector, crunch_ ## t ## vector); \
typedef struct {                                                             \
  crunch_integer start, end;                                                 \
  crunch_ ## t ## vector in, out;                                            \
  am_chunk_ ## t fn;                                                         \
} am_job_ ## t;                                                              \
                                                                             \
static void am_run_ ## t(void *p, long k) {                                  \
  am_job_ ## t *j = (am_job_ ## t *)p + k;                                   \
  j->fn(j->start, j->end, j->in, j->out);                                    \
}                                                                            \
                                                                             \
static int am_dispatch_ ## t(long n, C_word in, C_word out, void *fn,        \
                             int nthreads, long min_chunk) {                 \
  crunch_ ## t ## vector_block bin, bout;                                    \
  am_job_ ## t jobs[AM_MAX_THREADS];                                         \
  long T = am_thread_count(n, nthreads, min_chunk);                          \
  am_wrap_ ## t(in, &bin);                                                   \
  am_wrap_ ## t(out, &bout);                                                 \
  for (long k = 0; k < T; k++) {                                             \
    jobs[k].start = n * k / T;                                               \
    jobs[k].end   = n * (k + 1) / T;                                         \
    jobs[k].in    = &bin;                                                    \
    jobs[k].out   = &bout;                                                   \
    jobs[k].fn    = (am_chunk_ ## t)fn;                                      \
  }                                                                          \
  am_pool_run_chunks(am_run_ ## t, jobs, T);                                 \
  return (int)T;                                                             \
}

/* am_dispatch4_<t>: like am_dispatch_<t>, for kernels of the form
     fn(start, end, i0, i1, i2, scal, v0, v1, v2, v3)
   where i0-i2 are integers passed through unchanged, scal is an
   f64vector of scalar parameters and v0-v3 are vectors of element type
   et.  [0, n) is split into at most nthreads chunks of at least
   min_chunk units each. */
#define AM_DEFINE_DISPATCH4(t, et)                                           \
typedef void (*am_chunk4_ ## t)(crunch_integer, crunch_integer,             \
                                crunch_integer, crunch_integer, crunch_integer, \
                                crunch_f64vector,                            \
                                crunch_ ## t ## vector, crunch_ ## t ## vector, \
                                crunch_ ## t ## vector, crunch_ ## t ## vector); \
typedef struct {                                                             \
  crunch_integer start, end, i0, i1, i2;                                     \
  crunch_f64vector scal;                                                     \
  crunch_ ## t ## vector v0, v1, v2, v3;                                     \
  am_chunk4_ ## t fn;                                                        \
} am_job4_ ## t;                                                             \
                                                                             \
static void am_run4_ ## t(void *p, long k) {                                 \
  am_job4_ ## t *j = (am_job4_ ## t *)p + k;                                 \
  j->fn(j->start, j->end, j->i0, j->i1, j->i2, j->scal,                      \
        j->v0, j->v1, j->v2, j->v3);                                         \
}                                                                            \
                                                                             \
static int am_dispatch4_ ## t(long n, long i0, long i1, long i2,             \
                              C_word scal, C_word v0, C_word v1,             \
                              C_word v2, C_word v3, void *fn,                \
                              int nthreads, long min_chunk) {                \
  crunch_f64vector_block bscal;                                              \
  crunch_ ## t ## vector_block b0, b1, b2, b3;                               \
  am_job4_ ## t jobs[AM_MAX_THREADS];                                        \
  long T = am_thread_count(n, nthreads, min_chunk);                          \
  am_wrap_f64(scal, &bscal);                                                 \
  am_wrap_ ## t(v0, &b0);                                                    \
  am_wrap_ ## t(v1, &b1);                                                    \
  am_wrap_ ## t(v2, &b2);                                                    \
  am_wrap_ ## t(v3, &b3);                                                    \
  for (long k = 0; k < T; k++) {                                             \
    jobs[k].start = n * k / T;                                               \
    jobs[k].end   = n * (k + 1) / T;                                         \
    jobs[k].i0 = i0; jobs[k].i1 = i1; jobs[k].i2 = i2;                       \
    jobs[k].scal = &bscal;                                                   \
    jobs[k].v0 = &b0; jobs[k].v1 = &b1; jobs[k].v2 = &b2; jobs[k].v3 = &b3;  \
    jobs[k].fn = (am_chunk4_ ## t)fn;                                        \
  }                                                                          \
  am_pool_run_chunks(am_run4_ ## t, jobs, T);                                \
  return (int)T;                                                             \
}

AM_DEFINE_WRAP(f32, float)
AM_DEFINE_WRAP(f64, double)
AM_DEFINE_WRAP(s64, int64_t)
AM_DEFINE_DISPATCH(f32, float)
AM_DEFINE_DISPATCH(f64, double)
AM_DEFINE_DISPATCH4(f32, float)
AM_DEFINE_DISPATCH4(f64, double)

/* am_dispatch_n: see crunch-dispatch-n.  vecs is a Scheme vector of
   SRFI-4 vectors and esz an s32vector with the element size of each. */
static int am_dispatch_n(long n, C_word ip, C_word sc, C_word vecs, C_word esz,
                         void *run, int nthreads, long min_chunk) {
  crunch_s64vector_block bip;
  crunch_f64vector_block bsc;
  crunch_vector_block bv[AM_DN_MAX_VECS];
  am_jobn jobs[AM_MAX_THREADS];
  long nv = C_header_size(vecs);
  int32_t *es = (int32_t *)C_data_pointer(C_block_item(esz, 1));
  long T = am_thread_count(n, nthreads, min_chunk);
  am_wrap_s64(ip, &bip);
  am_wrap_f64(sc, &bsc);
  for (long j = 0; j < nv; j++) {
    C_word bytes = C_block_item(C_block_item(vecs, j), 1);
    bv[j].rc    = -1;
    bv[j].flags = CRUNCH_BLOCK;
    bv[j].size  = C_header_size(bytes);
    bv[j].len   = bv[j].size / es[j];
    bv[j].data  = C_data_pointer(bytes);
  }
  for (long k = 0; k < T; k++) {
    jobs[k].start = n * k / T;
    jobs[k].end   = n * (k + 1) / T;
    jobs[k].ip    = &bip;
    jobs[k].sc    = &bsc;
    for (long j = 0; j < nv; j++) jobs[k].v[j] = &bv[j];
  }
  am_pool_run_chunks((am_pool_run)run, jobs, T);
  return (int)T;
}

static int am_available_processors(void) {
  long p = sysconf(_SC_NPROCESSORS_ONLN);
  return p < 1 ? 1 : (int)p;
}
EOC
)

  (define %dispatch-f32
    (foreign-lambda int "am_dispatch_f32"
                    long scheme-object scheme-object c-pointer int long))
  (define %dispatch-f64
    (foreign-lambda int "am_dispatch_f64"
                    long scheme-object scheme-object c-pointer int long))
  (define %dispatch4-f32
    (foreign-lambda int "am_dispatch4_f32"
                    long long long long scheme-object scheme-object scheme-object
                    scheme-object scheme-object c-pointer int long))
  (define %dispatch4-f64
    (foreign-lambda int "am_dispatch4_f64"
                    long long long long scheme-object scheme-object scheme-object
                    scheme-object scheme-object c-pointer int long))

  (define %dispatch-n
    (foreign-lambda int "am_dispatch_n"
                    long scheme-object scheme-object scheme-object scheme-object
                    c-pointer int long))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Thread-count settings
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define crunch-max-threads 64)

  (define (crunch-available-processors)
    "Number of processors currently online."
    ((foreign-lambda int "am_available_processors")))

  ;; Empirical measurements showed that 16 threads were not reliably
  ;; faster than 8 threads for arrays of 0.4M to 4M elements, and
  ;; sometimes were slower.
  (define default-thread-cap 8)

  (define (positive-fixnum-guard who)
    (lambda (x)
      (unless (and (fixnum? x) (> x 0))
        (error who "value must be a positive fixnum" x))
      x))

  (define (default-thread-count)
    (let* ((env (get-environment-variable "AM_CRUNCH_THREADS"))
           (n   (and env (string->number env))))
      (if (and n (exact? n) (integer? n) (> n 0))
          (min n crunch-max-threads)
          (min (crunch-available-processors) default-thread-cap))))

  ;; Requested number of threads; the dispatcher may use fewer.
  ;; Defaults to AM_CRUNCH_THREADS when that is set to a positive integer,
  ;; otherwise to the number of online processors, at most 8.
  (define crunch-thread-count
    (make-parameter (default-thread-count)
                    (positive-fixnum-guard 'crunch-thread-count)))

  ;; Smallest number of elements worth giving to one thread.  In
  ;; wall-clock measurements of pairs of f32 kernels a second thread
  ;; pays off at about 4K elements per thread for relu and at under 1K
  ;; for sigmoid.  See tests/bench-threaded-activations.scm.
  (define crunch-thread-min-chunk
    (make-parameter 4096
                    (positive-fixnum-guard 'crunch-thread-min-chunk)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Thread pool
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (crunch-thread-pool-size)
    "Number of worker threads in the pool (0 before the first threaded
    dispatch and after crunch-thread-pool-shutdown!)."
    ((foreign-lambda int "am_pool_size")))

  (define (crunch-thread-pool-shutdown!)
    "Stop and join every worker thread.  The next threaded dispatch
    creates the pool again.  Returns the number of workers stopped."
    (let ((n ((foreign-lambda int "am_pool_shutdown"))))
      (when (< n 0)
        (error 'crunch-thread-pool-shutdown! "a dispatch is in progress"))
      n))

  (define (crunch-thread-pool-stats)
    "Counters since the start of the process, as an association list:
    dispatches (calls that used the pool), spin-hits (chunks a worker
    found while spinning), wakeups (chunks a worker found after going to
    sleep) and caller-sleeps (dispatches whose caller slept while
    waiting for the workers)."
    (let ((stat (foreign-lambda unsigned-long "am_pool_stat" int)))
      (list (cons 'dispatches    (stat (foreign-value "AM_POOL_STAT_DISPATCHES" int)))
            (cons 'spin-hits     (stat (foreign-value "AM_POOL_STAT_SPIN_HITS" int)))
            (cons 'wakeups       (stat (foreign-value "AM_POOL_STAT_WAKEUPS" int)))
            (cons 'caller-sleeps (stat (foreign-value "AM_POOL_STAT_CALLER_SLEEPS" int))))))

  (define (crunch-thread-pool-spin-us)
    "Microseconds a waiting thread spins before it sleeps."
    ((foreign-lambda long "am_pool_spin_us")))

  (define crunch-thread-pool-max-spin-us
    (foreign-value "AM_POOL_MAX_SPIN_US" long))

  (define (crunch-thread-pool-spin-us-set! us)
    "Set the spin time to us microseconds, a fixnum from 0 (sleep at
    once) to 1000000."
    (unless (and (fixnum? us) (<= 0 us crunch-thread-pool-max-spin-us))
      (error 'crunch-thread-pool-spin-us-set!
             "spin time must be a fixnum from 0 to 1000000" us))
    ((foreign-lambda void "am_pool_set_spin_us" long) us))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Dispatch
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (check-dispatch who vec? vec-length n in out kernel threads min-chunk)
    (unless (and (vec? in) (vec? out))
      (error who "vectors have the wrong type" in out))
    (unless (and (fixnum? n) (>= n 0) (<= n (vec-length in)) (<= n (vec-length out)))
      (error who "size must be a non-negative fixnum within both vectors" n))
    (unless (pointer? kernel)
      (error who "kernel must be a C function pointer" kernel))
    (unless (and (fixnum? threads) (> threads 0))
      (error who "thread count must be a positive fixnum" threads))
    (unless (and (fixnum? min-chunk) (> min-chunk 0))
      (error who "minimum chunk must be a positive fixnum" min-chunk)))

  (define (crunch-dispatch-f32 n in out kernel
                               #!optional (threads (crunch-thread-count))
                                          (min-chunk (crunch-thread-min-chunk)))
    "Apply the f32 chunk kernel at C address kernel to elements [0, n) of
    the f32vectors in and out, using up to threads threads.  Returns the
    number of threads used."
    (check-dispatch 'crunch-dispatch-f32 f32vector? f32vector-length
                    n in out kernel threads min-chunk)
    (%dispatch-f32 n in out kernel threads min-chunk))

  (define (crunch-dispatch-f64 n in out kernel
                               #!optional (threads (crunch-thread-count))
                                          (min-chunk (crunch-thread-min-chunk)))
    "Apply the f64 chunk kernel at C address kernel to elements [0, n) of
    the f64vectors in and out, using up to threads threads.  Returns the
    number of threads used."
    (check-dispatch 'crunch-dispatch-f64 f64vector? f64vector-length
                    n in out kernel threads min-chunk)
    (%dispatch-f64 n in out kernel threads min-chunk))

  (define (check-dispatch4 who vec? n scal vecs kernel threads min-chunk)
    (unless (f64vector? scal)
      (error who "scalar parameters must be an f64vector" scal))
    (for-each (lambda (v)
                (unless (vec? v) (error who "vector has the wrong type" v)))
              vecs)
    (unless (and (fixnum? n) (>= n 0))
      (error who "size must be a non-negative fixnum" n))
    (unless (pointer? kernel)
      (error who "kernel must be a C function pointer" kernel))
    (unless (and (fixnum? threads) (> threads 0))
      (error who "thread count must be a positive fixnum" threads))
    (unless (and (fixnum? min-chunk) (> min-chunk 0))
      (error who "minimum chunk must be a positive fixnum" min-chunk)))

  (define (crunch-dispatch4-f32 n i0 i1 i2 scal v0 v1 v2 v3 kernel
                                #!optional (threads (crunch-thread-count))
                                           (min-chunk (crunch-thread-min-chunk)))
    "Apply the f32 chunk kernel at C address kernel, of the form
    (start end i0 i1 i2 scal v0 v1 v2 v3), to the units [0, n), using up to
    threads threads with at least min-chunk units each.  i0-i2 are passed
    to every chunk unchanged; scal is an f64vector and v0-v3 are
    f32vectors.  Kernels run without index checks, so the caller must
    make sure that every index a kernel uses lies within its vectors.
    Returns the number of threads used."
    (check-dispatch4 'crunch-dispatch4-f32 f32vector? n scal (list v0 v1 v2 v3)
                     kernel threads min-chunk)
    (%dispatch4-f32 n i0 i1 i2 scal v0 v1 v2 v3 kernel threads min-chunk))

  (define (crunch-dispatch4-f64 n i0 i1 i2 scal v0 v1 v2 v3 kernel
                                #!optional (threads (crunch-thread-count))
                                           (min-chunk (crunch-thread-min-chunk)))
    "f64 counterpart of crunch-dispatch4-f32: v0-v3 are f64vectors."
    (check-dispatch4 'crunch-dispatch4-f64 f64vector? n scal (list v0 v1 v2 v3)
                     kernel threads min-chunk)
    (%dispatch4-f64 n i0 i1 i2 scal v0 v1 v2 v3 kernel threads min-chunk))

  (define crunch-dispatch-n-max-vectors (foreign-value "AM_DN_MAX_VECS" int))

  (define (srfi-4-vector? v)
    (or (f32vector? v) (f64vector? v) (s32vector? v) (s64vector? v)
        (u32vector? v) (u64vector? v) (s8vector? v) (u8vector? v)
        (s16vector? v) (u16vector? v)))

  (define (crunch-dispatch-n n ip sc vecs esz run threads min-chunk)
    "Run a generated kernel over the units [0, n), using up to threads
    threads with at least min-chunk units each, the first chunk on the
    calling thread.  run is the C address of the kernel's run procedure
    (see crunch-dispatch-n.h), ip an s64vector of integer parameters, sc
    an f64vector of scalar parameters, vecs a vector of at most
    crunch-dispatch-n-max-vectors SRFI-4 vectors and esz an s32vector
    holding the element size in bytes of each.  The kernel decides which
    elements it touches, without index checks, so the caller must make
    sure that every index it uses lies within its vectors.  Returns the
    number of threads used."
    (unless (and (s64vector? ip) (f64vector? sc) (vector? vecs) (s32vector? esz)
                 (<= (vector-length vecs) crunch-dispatch-n-max-vectors)
                 (= (vector-length vecs) (s32vector-length esz)))
      (error 'crunch-dispatch-n "invalid parameter vectors" ip sc vecs esz))
    (do ((j 0 (+ j 1))) ((= j (vector-length vecs)))
      (unless (srfi-4-vector? (vector-ref vecs j))
        (error 'crunch-dispatch-n "not a numeric vector" (vector-ref vecs j))))
    (unless (and (fixnum? n) (>= n 0))
      (error 'crunch-dispatch-n "size must be a non-negative fixnum" n))
    (unless (pointer? run)
      (error 'crunch-dispatch-n "run must be a C function pointer" run))
    (unless (and (fixnum? threads) (> threads 0) (fixnum? min-chunk) (> min-chunk 0))
      (error 'crunch-dispatch-n "thread count and minimum chunk must be positive fixnums"
             threads min-chunk))
    (%dispatch-n n ip sc vecs esz run threads min-chunk))

) ;; end module array-morphisms-crunch-threads
