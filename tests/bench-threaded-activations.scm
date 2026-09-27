;;; tests/bench-threaded-activations.scm
;;;
;;; Wall-clock time measurements of the crunch activation kernels on
;;; one thread and on several threads, and of the Scheme combiner loop
;;; they replace.  The size sweep shows where threading starts to pay
;;; off, which is what crunch-thread-min-chunk is set from.
;;;
;;; Usage: csc -O3 bench-threaded-activations.scm -o bench && ./bench

(import scheme (chicken base) (chicken format) (scheme time) srfi-4)
(import array-morphisms-realization)
(import array-morphisms-activation-exec)
(import array-morphisms-crunch-activations)
(import array-morphisms-crunch-threads)

(define (wall-ms reps thunk)
  (thunk)
  (let ((t0 (current-jiffy)))
    (do ((r 0 (+ r 1))) ((= r reps)) (thunk))
    (/ (* 1000.0 (- (current-jiffy) t0)) (* reps (jiffies-per-second)))))

(define (input n)
  (let ((v (make-f32vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f32vector-set! v i (* 8.0 (- (/ i n) 0.5))))))

(define plain    (make-crunch-activation-backend))
(define threaded (make-crunch-threaded-activation-backend))

(define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))

(printf "processors online: ~a, default threads: ~a, default min-chunk: ~a~%~%"
        (crunch-available-processors) (crunch-thread-count) (crunch-thread-min-chunk))

;; 1. Kernel comparison at activation-tensor sizes.
(for-each
 (lambda (n)
   (let ((in (input n)) (out (make-f32vector n 0.0))
         (reps (if (> n 1000000) 10 50)))
     (for-each
      (lambda (op)
        (printf "~a f32 n=~a~%" op n)
        (when (eq? op 'sigmoid)
          (printf "  scheme combiner   ~a ms~%"
                  (wall-ms (max 1 (quotient reps 10))
                           (lambda () (execute-flat-unary-compute sigmoid in 'f32 out n 'f32)))))
        (printf "  crunch, unthreaded ~a ms~%"
                (wall-ms reps (lambda () ((lookup-activation-kernel plain op 'f32) n in out))))
        (for-each
         (lambda (threads)
           (parameterize ((crunch-thread-count threads) (crunch-thread-min-chunk 1))
             (printf "  crunch, ~a threads ~a ms~%" threads
                     (wall-ms reps (lambda () ((lookup-activation-kernel threaded op 'f32)
                                               n in out))))))
         '(1 2 4 8 16)))
      '(relu sigmoid tanh))))
 '(400000 4000000))

;; 2. Break-even sweep: per-call time of sigmoid and relu on 1 thread and on
;;    2 and 4 threads forced (min-chunk 1), for growing n.
(printf "~%break-even sweep (ms per call)~%")
(for-each
 (lambda (op)
   (for-each
    (lambda (n)
      (let ((in (input n)) (out (make-f32vector n 0.0)))
        (printf "  ~a n=~a:" op n)
        (for-each
         (lambda (threads)
           (parameterize ((crunch-thread-count threads) (crunch-thread-min-chunk 1))
             (printf "  T=~a ~a" threads
                     (wall-ms 200 (lambda () ((lookup-activation-kernel threaded op 'f32)
                                              n in out))))))
         '(1 2 4))
        (newline)))
    '(4096 16384 32768 65536 131072)))
 '(sigmoid relu))
