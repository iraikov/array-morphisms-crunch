;;; smoke-8-threads.scm -- a crunch chunk kernel called from several POSIX
;;; threads.
;;;
;;; The kernel is defined with an underscore-only name, so its C symbol is
;;; the Scheme name unchanged, and its address is taken with foreign-value.
;;; A small C dispatcher splits [0, n) into contiguous chunks, runs chunk 0
;;; on the calling thread and the others on new threads, and joins them.
;;; The worker threads only call the crunch function on vector descriptors
;;; built on the dispatcher's stack; they never touch the CHICKEN heap.
;;; The descriptors carry rc = -1, which crunch never reference-counts.
;;;
;;; Checked: results for 1, 2 and 4 threads equal the single-threaded
;;; result, the dispatcher reports the thread count it used, and the
;;; program links against the pthread library.  Wall-clock timings are
;;; printed for information only (wall-clock, via current-jiffy).
(import scheme (chicken base) (chicken foreign) (chicken number-vector)
        (scheme time) crunch)
(include-relative "../../crunch-numvector-fix.scm")

(crunch
  (: (am_smoke_sigmoid_chunk_f32 integer integer f32vector f32vector) void)
  (define (am_smoke_sigmoid_chunk_f32 start end in out)
    (do ((i start (+ i 1))) ((= i end))
      (let ((x (f32vector-ref in i)))
        (f32vector-set! out i (/ 1.0 (+ 1.0 (exp (- x)))))))))

(foreign-declare #<<EOC
#include <pthread.h>
#include <signal.h>

typedef void (*smoke_chunk_f32)(crunch_integer, crunch_integer,
                                crunch_f32vector, crunch_f32vector);
typedef struct { crunch_integer start, end;
                 crunch_f32vector in, out; smoke_chunk_f32 fn; } smoke_job;

static void *smoke_worker(void *p) {
  smoke_job *j = (smoke_job *)p;
  j->fn(j->start, j->end, j->in, j->out);
  return NULL;
}

static void smoke_wrap(C_word vec, crunch_f32vector_block *b) {
  C_word bv = C_block_item(vec, 1);
  b->rc = -1; b->flags = CRUNCH_BLOCK;
  b->size = C_header_size(bv); b->len = b->size / sizeof(float);
  b->data = (float *)C_data_pointer(bv);
}

static int smoke_dispatch(long n, C_word in, C_word out, void *fn, int T) {
  crunch_f32vector_block bin, bout;
  pthread_t th[16]; smoke_job jobs[16]; int ok[16];
  sigset_t all, old;
  smoke_wrap(in, &bin); smoke_wrap(out, &bout);
  for (int t = 0; t < T; t++) {
    jobs[t].start = n * t / T; jobs[t].end = n * (t + 1) / T;
    jobs[t].in = &bin; jobs[t].out = &bout; jobs[t].fn = (smoke_chunk_f32)fn;
  }
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &old);
  for (int t = 1; t < T; t++)
    ok[t] = pthread_create(&th[t], NULL, smoke_worker, &jobs[t]) == 0;
  pthread_sigmask(SIG_SETMASK, &old, NULL);
  smoke_worker(&jobs[0]);
  for (int t = 1; t < T; t++) {
    if (ok[t]) pthread_join(th[t], NULL); else smoke_worker(&jobs[t]);
  }
  return T;
}
EOC
)

(define smoke-dispatch
  (foreign-lambda int "smoke_dispatch" long scheme-object scheme-object c-pointer int))

(define chunk-ptr
  (foreign-value "((void *)&am_smoke_sigmoid_chunk_f32)" c-pointer))

(define n 1000000)
(define in (make-f32vector n 0.0))
(do ((i 0 (+ i 1))) ((= i n))
  (f32vector-set! in i (* 0.00001 (- i 500000))))

(define (run threads)
  (let ((out (make-f32vector n -1.0)))
    (assert (= threads (smoke-dispatch n in out chunk-ptr threads)))
    out))

(define reference (f32vector->list (run 1)))

(for-each
 (lambda (threads)
   (assert (equal? (f32vector->list (run threads)) reference)))
 '(2 3 4 16))

;; A kernel given n = 0 writes nothing.
(let ((out (f32vector 9.0)))
  (smoke-dispatch 0 in out chunk-ptr 4)
  (assert (equal? (f32vector->list out) '(9.0))))

(define (wall-ms threads)
  (let ((out (make-f32vector n 0.0)))
    (smoke-dispatch n in out chunk-ptr threads)
    (let ((t0 (current-jiffy)))
      (do ((r 0 (+ r 1))) ((= r 20)) (smoke-dispatch n in out chunk-ptr threads))
      (/ (* 1000.0 (- (current-jiffy) t0)) (* 20.0 (jiffies-per-second))))))

(for-each (lambda (threads)
            (display "  sigmoid 1M f32, ") (display threads)
            (display " thread(s): ") (display (wall-ms threads)) (display " ms\n"))
          '(1 2 4 8))
(display "smoke-8-threads: ok\n")
