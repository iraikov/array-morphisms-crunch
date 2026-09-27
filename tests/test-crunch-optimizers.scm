;;; tests/test-crunch-optimizers.scm
;;; Tests for the Adam kernels of array-morphisms-crunch-optimizers.
;;;
;;; The reference below performs the kernel's documented steps in Scheme,
;;; storing each intermediate result in its vector as the kernel does.  In
;;; a portable build the kernel must match it bit for bit; a build for the
;;; host CPU may fuse multiply-adds, so there the comparison uses a tight
;;; tolerance.

(import scheme (chicken base)
        test
        (only srfi-1 iota every)
        srfi-4
        (only (chicken format) sprintf)
        (only array-morphisms-crunch-blas-backend crunch-native-build?)
        (only array-morphisms-crunch-threads crunch-thread-count crunch-thread-min-chunk)
        array-morphisms-crunch-optimizers)

(define (make-vec make n f)
  (let ((v (make n)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (if (f32vector? v)
          (f32vector-set! v i (f i))
          (f64vector-set! v i (f i))))))

(define (vref v i) (if (f32vector? v) (f32vector-ref v i) (f64vector-ref v i)))
(define (vset! v i x) (if (f32vector? v) (f32vector-set! v i x) (f64vector-set! v i x)))
(define (vlist v) (if (f32vector? v) (f32vector->list v) (f64vector->list v)))

(define (reference-adam! n hyper param grad m v)
  (let ((lr (f64vector-ref hyper 0)) (b1 (f64vector-ref hyper 1)) (b2 (f64vector-ref hyper 2))
        (eps (f64vector-ref hyper 3)) (bc1 (f64vector-ref hyper 4)) (bc2 (f64vector-ref hyper 5))
        (wd (f64vector-ref hyper 6)))
    (do ((i 0 (+ i 1))) ((= i n))
      (when (> wd 0.0) (vset! grad i (+ (vref grad i) (* wd (vref param i)))))
      (let ((g (vref grad i)))
        (vset! m i (* b1 (vref m i)))
        (vset! m i (+ (vref m i) (* (- 1.0 b1) g)))
        (vset! v i (* b2 (vref v i)))
        (vset! v i (+ (vref v i) (* (* (- 1.0 b2) g) g)))
        (let ((m-hat (/ (vref m i) bc1)) (v-hat (/ (vref v i) bc2)))
          (vset! param i (- (vref param i) (/ (* lr m-hat) (+ (sqrt v-hat) eps)))))))))

(define (hyper-for step wd)
  (f64vector 0.001 0.9 0.999 1e-8 (- 1.0 (expt 0.9 step)) (- 1.0 (expt 0.999 step)) wd))

;; State after five steps of update! on fresh vectors of the given kind.
(define (five-steps update! make n wd)
  (let ((param (make-vec make n (lambda (i) (sin (* 0.37 i)))))
        (m (make n 0.0))
        (v (make n 0.0)))
    (do ((step 1 (+ step 1))) ((> step 5))
      (let ((grad (make-vec make n (lambda (i) (cos (* 0.11 (+ i step)))))))
        (update! n (hyper-for step wd) param grad m v)))
    (append (vlist param) (vlist m) (vlist v))))

(define (lists-match? a b tol)
  (and (= (length a) (length b))
       (if crunch-native-build?
           (every (lambda (x y) (<= (abs (- x y)) (* tol (max 1.0 (abs x))))) a b)
           (equal? a b))))

(test-group "crunch Adam kernels"
  (for-each
   (lambda (dtype make kernel tol)
     (for-each
      (lambda (wd)
        (test-assert (sprintf "~A Adam matches the reference over five steps, weight decay ~A" dtype wd)
          (lists-match? (five-steps kernel make 1000 wd)
                        (five-steps reference-adam! make 1000 wd)
                        tol))
        (test-assert (sprintf "~A Adam gives identical results for 1, 3 and 8 threads, weight decay ~A" dtype wd)
          (let ((run (lambda (threads)
                       (parameterize ((crunch-thread-count threads) (crunch-thread-min-chunk 1000))
                         (five-steps kernel make 20001 wd)))))
            (let ((one (run 1)))
              (every (lambda (t) (equal? one (run t))) '(3 8))))))
      '(0.0 0.01)))
   '(f32 f64)
   (list make-f32vector make-f64vector)
   (list crunch-adam-f32 crunch-adam-f64)
   '(1e-6 1e-12))
  (test-error "Adam rejects a size beyond its vectors"
    (crunch-adam-f32 10 (hyper-for 1 0.0) (make-f32vector 10) (make-f32vector 5)
                     (make-f32vector 10) (make-f32vector 10)))
  (test-error "Adam rejects a short hyperparameter vector"
    (crunch-adam-f32 1 (f64vector 0.1) (make-f32vector 1) (make-f32vector 1)
                     (make-f32vector 1) (make-f32vector 1))))

(test-exit)
