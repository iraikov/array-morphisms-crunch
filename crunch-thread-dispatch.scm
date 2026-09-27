;;; crunch-thread-dispatch.scm
;;; Applies a crunch chunk kernel across an array on several POSIX threads.
;;;
;;; A chunk kernel is a Crunch procedure (kernel start end in out) that
;;; processes elements [start, end) of the SRFI-4 vectors in and out.
;;;
;;; crunch-dispatch-f32 and crunch-dispatch-f64 split [0, n) into T
;;; contiguous chunks of nearly equal size, run the first chunk on the
;;; calling thread and each other chunk on a new thread, and return
;;; when all threads have been joined.  Since the chunks are disjoint
;;; and each element is computed on its own, the result is the same
;;; for every thread count.
;;;
;;; The number of threads is
;;;
;;;   T = max(1, min(threads, n div min-chunk, 64))
;;;
;;; so an array shorter than two chunks runs on the calling thread alone,
;;; without the cost of creating threads.  threads defaults to the value of
;;; (crunch-thread-count) and min-chunk to (crunch-thread-min-chunk).
;;;
;;; The worker threads do not touch the CHICKEN heap or call into the
;;; CHICKEN runtime.  The dispatcher describes the two vectors to the
;;; kernel with block descriptors on its own stack, pointing at the
;;; vectors' storage; the descriptors have a reference count of -1, so
;;; the threads can share them without synchronisation.  The calling
;;; thread stays inside the one foreign call until every worker has
;;; finished, so no garbage collection can move the vectors in the
;;; mean time.
;;;
;;; All signals are blocked in the workers, so CHICKEN's signal
;;; handler only ever runs on the thread that owns the runtime.  If a
;;; thread cannot be created, its chunk runs on the calling thread
;;; instead.
;;;
;;;   (import array-morphisms-crunch-threads)
;;;   (crunch-thread-count 8)
;;;   (crunch-dispatch-f32 n in out kernel-pointer)

(module array-morphisms-crunch-threads

  (crunch-dispatch-f32
   crunch-dispatch-f64
   crunch-dispatch4-f32
   crunch-dispatch4-f64
   crunch-thread-count
   crunch-thread-min-chunk
   crunch-available-processors
   crunch-max-threads)

  (import scheme (chicken base) (chicken foreign) (chicken number-vector)
          (chicken process-context) (only (chicken memory) pointer?)
          (only (scheme base) make-parameter))

  (foreign-declare #<<EOC
#define CRUNCH_EMBEDDED
#include "crunch.h"
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#define AM_MAX_THREADS 64

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
static void *am_worker_ ## t(void *p) {                                      \
  am_job_ ## t *j = (am_job_ ## t *)p;                                       \
  j->fn(j->start, j->end, j->in, j->out);                                    \
  return NULL;                                                               \
}                                                                            \
                                                                             \
static int am_dispatch_ ## t(long n, C_word in, C_word out, void *fn,        \
                             int nthreads, long min_chunk) {                 \
  crunch_ ## t ## vector_block bin, bout;                                    \
  pthread_t th[AM_MAX_THREADS];                                              \
  am_job_ ## t jobs[AM_MAX_THREADS];                                         \
  int started[AM_MAX_THREADS];                                               \
  sigset_t all, old;                                                         \
  long T = nthreads;                                                         \
  if (min_chunk < 1) min_chunk = 1;                                          \
  if (T > n / min_chunk) T = n / min_chunk;                                  \
  if (T > AM_MAX_THREADS) T = AM_MAX_THREADS;                                \
  if (T < 1) T = 1;                                                          \
  am_wrap_ ## t(in, &bin);                                                   \
  am_wrap_ ## t(out, &bout);                                                 \
  for (long k = 0; k < T; k++) {                                             \
    jobs[k].start = n * k / T;                                               \
    jobs[k].end   = n * (k + 1) / T;                                         \
    jobs[k].in    = &bin;                                                    \
    jobs[k].out   = &bout;                                                   \
    jobs[k].fn    = (am_chunk_ ## t)fn;                                      \
  }                                                                          \
  if (T > 1) {                                                               \
    sigfillset(&all);                                                        \
    pthread_sigmask(SIG_SETMASK, &all, &old);                                \
    for (long k = 1; k < T; k++)                                             \
      started[k] = pthread_create(&th[k], NULL, am_worker_ ## t, &jobs[k]) == 0; \
    pthread_sigmask(SIG_SETMASK, &old, NULL);                                \
  }                                                                          \
  am_worker_ ## t(&jobs[0]);                                                 \
  for (long k = 1; k < T; k++) {                                             \
    if (started[k]) pthread_join(th[k], NULL);                               \
    else am_worker_ ## t(&jobs[k]);                                          \
  }                                                                          \
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
static void *am_worker4_ ## t(void *p) {                                     \
  am_job4_ ## t *j = (am_job4_ ## t *)p;                                     \
  j->fn(j->start, j->end, j->i0, j->i1, j->i2, j->scal,                      \
        j->v0, j->v1, j->v2, j->v3);                                         \
  return NULL;                                                               \
}                                                                            \
                                                                             \
static int am_dispatch4_ ## t(long n, long i0, long i1, long i2,             \
                              C_word scal, C_word v0, C_word v1,             \
                              C_word v2, C_word v3, void *fn,                \
                              int nthreads, long min_chunk) {                \
  crunch_f64vector_block bscal;                                              \
  crunch_ ## t ## vector_block b0, b1, b2, b3;                               \
  pthread_t th[AM_MAX_THREADS];                                              \
  am_job4_ ## t jobs[AM_MAX_THREADS];                                        \
  int started[AM_MAX_THREADS];                                               \
  sigset_t all, old;                                                         \
  long T = nthreads;                                                         \
  if (min_chunk < 1) min_chunk = 1;                                          \
  if (T > n / min_chunk) T = n / min_chunk;                                  \
  if (T > AM_MAX_THREADS) T = AM_MAX_THREADS;                                \
  if (T < 1) T = 1;                                                          \
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
  if (T > 1) {                                                               \
    sigfillset(&all);                                                        \
    pthread_sigmask(SIG_SETMASK, &all, &old);                                \
    for (long k = 1; k < T; k++)                                             \
      started[k] = pthread_create(&th[k], NULL, am_worker4_ ## t, &jobs[k]) == 0; \
    pthread_sigmask(SIG_SETMASK, &old, NULL);                                \
  }                                                                          \
  am_worker4_ ## t(&jobs[0]);                                                \
  for (long k = 1; k < T; k++) {                                             \
    if (started[k]) pthread_join(th[k], NULL);                               \
    else am_worker4_ ## t(&jobs[k]);                                         \
  }                                                                          \
  return (int)T;                                                             \
}

AM_DEFINE_WRAP(f32, float)
AM_DEFINE_WRAP(f64, double)
AM_DEFINE_DISPATCH(f32, float)
AM_DEFINE_DISPATCH(f64, double)
AM_DEFINE_DISPATCH4(f32, float)
AM_DEFINE_DISPATCH4(f64, double)

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

  ;; Smallest number of elements worth giving to one thread.  Creating
  ;; and joining a thread costs some tens of microseconds; in
  ;; wall-clock measurements of the f32 sigmoid, tanh and relu kernels
  ;; a second thread starts to pay off at about 16K elements per
  ;; thread.  See tests/bench-threaded-activations.scm.
  (define crunch-thread-min-chunk
    (make-parameter 16384
                    (positive-fixnum-guard 'crunch-thread-min-chunk)))

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

) ;; end module array-morphisms-crunch-threads
