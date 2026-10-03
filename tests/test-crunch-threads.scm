;;; tests/test-crunch-threads.scm
;;; Tests for the thread dispatcher (crunch-thread-dispatch.scm) and
;;; the threaded activation backend (make-crunch-threaded-activation-backend).
;;;
;;; Splitting an element-wise kernel across threads must not change any
;;; result, so every comparison here is exact: eqv? element by element,
;;; with NaN accepted only against NaN.  
;;;
;;; Organisation:
;;;   Group 1 - Thread-count decisions
;;;   Group 2 - Exact results for every op, dtype, size and thread count
;;;   Group 3 - Buffers: in place, prefixes
;;;   Group 4 - Repeated calls
;;;   Group 5 - Settings: parameters, defaults
;;;   Group 6 - Argument checking
;;;   Group 7 - SSA replay through the threaded backend

(import scheme (chicken base)
        test
        (only srfi-1 iota every filter-map take)
        srfi-4
        (only (chicken number-vector)
         f32vector->bytevector/shared)
        datatype
        array-morphisms-core
        array-morphisms-basic-ops
        array-morphisms-realization
        array-morphisms-context
        array-morphisms-activation-exec
        (prefix array-morphisms-grad am:)
        array-morphisms-ssa
        array-morphisms-crunch-activations
        array-morphisms-crunch-threads)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (th-same? a b) (or (and (nan? a) (nan? b)) (eqv? a b)))

(define (th-same-lists? a b)
  (and (= (length a) (length b)) (every th-same? a b)))

(define th-ops '(relu sigmoid tanh relu-deriv sigmoid-deriv tanh-deriv
                 exp negate abs))

;; Special values first, then a deterministic spread covering both signs,
;; small and large magnitudes, and exp overflow.
(define th-specials (list 0.0 -0.0 +inf.0 -inf.0 +nan.0 710.0 -710.0 400.0 -400.0))

(define (th-input n)
  (let ((v (make-f64vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f64vector-set! v i
                      (if (< i (length th-specials))
                          (list-ref th-specials i)
                          (* 12.0 (sin (* 0.37 i))))))))

(define (th-vector dtype n)
  (let ((src (th-input n)))
    (if (eq? dtype 'f32)
        (list->f32vector (f64vector->list src))
        src)))

(define (th->list v) (if (f32vector? v) (f32vector->list v) (f64vector->list v)))

(define (th-make dtype n fill)
  (if (eq? dtype 'f32) (make-f32vector n fill) (make-f64vector n fill)))

(define th-plain    (make-crunch-activation-backend))
(define th-threaded (make-crunch-threaded-activation-backend))

(define (plain-kernel op dtype)    (lookup-activation-kernel th-plain op dtype))
(define (threaded-kernel op dtype) (lookup-activation-kernel th-threaded op dtype))

(define (dispatch dtype n in out op threads min-chunk)
  ((if (eq? dtype 'f32) crunch-dispatch-f32 crunch-dispatch-f64)
   n in out (crunch-activation-chunk-kernel op dtype) threads min-chunk))

;; The Scheme combiners the kernels replace.
(define th-combiners
  `((relu . ,(lambda (x) (max 0.0 x)))
    (sigmoid . ,(lambda (x) (/ 1.0 (+ 1.0 (exp (- x))))))
    (tanh . ,(lambda (x)
               (let* ((e (exp (* -2.0 (abs x))))
                      (t (/ (- 1.0 e) (+ 1.0 e))))
                 (if (< x 0.0) (- t) t))))
    (relu-deriv . ,(lambda (x) (if (> x 0.0) 1.0 0.0)))
    (sigmoid-deriv . ,(lambda (s) (* s (- 1.0 s))))
    (tanh-deriv . ,(lambda (t) (- 1.0 (* t t))))
    (exp . ,exp)
    (negate . ,(lambda (x) (- x)))
    (abs . ,abs)))

(define (combiner-result op dtype in n)
  (let ((out (th-make dtype n 0.0)))
    (execute-flat-unary-compute (cdr (assq op th-combiners)) in dtype out n dtype)
    (th->list out)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - Thread-count decisions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - thread-count decisions"
  (define (used n threads min-chunk)
    (dispatch 'f32 n (th-vector 'f32 (max n 1)) (th-make 'f32 (max n 1) 0.0)
              'relu threads min-chunk))
  (test "fewer elements than min-chunk: one thread" 1 (used 1000 8 4096))
  (test "exactly two chunks' worth: two threads" 2 (used 8192 8 4096))
  (test "one thread requested: one thread" 1 (used 100000 1 1))
  (test "min-chunk 1, n 1000, 4 threads requested: four" 4 (used 1000 4 1))
  (test "n 3 with 16 threads requested: three" 3 (used 3 16 1))
  (test "more threads than the maximum are clamped" crunch-max-threads
        (used 10000 100 1))
  (test "n 0: one thread" 1 (used 0 8 1))
  (test "the maximum is 64" 64 crunch-max-threads))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - Exact results
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - results identical for every thread count"
  (for-each
   (lambda (dtype)
     (for-each
      (lambda (op)
        (test-assert (string-append (symbol->string op) " " (symbol->string dtype)
                                    ": all sizes and thread counts")
          (every
           (lambda (n)
             (let* ((in  (th-vector dtype (max n 1)))
                    (ref (let ((o (th-make dtype (max n 1) 0.0)))
                           ((plain-kernel op dtype) n in o)
                           (th->list o)))
                    (comb (combiner-result op dtype in n)))
               (and (th-same-lists? (take ref n) comb)
                    (every
                     (lambda (threads)
                       (let ((out (th-make dtype (max n 1) 0.0)))
                         (dispatch dtype n in out op threads 1)
                         (th-same-lists? (th->list out) ref)))
                     '(1 2 3 4 7 16)))))
           '(0 1 2 6 7 8 1000 100003))))
      th-ops))
   '(f64 f32)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - Buffers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - buffers"
  (test-assert "in place with 8 threads equals out of place"
    (let* ((n 50001)
           (buf (th-vector 'f32 n))
           (ref (let ((o (th-make 'f32 n 0.0)))
                  ((plain-kernel 'tanh 'f32) n buf o)
                  (th->list o))))
      (and (= 8 (dispatch 'f32 n buf buf 'tanh 8 1))
           (th-same-lists? (th->list buf) ref))))

  (test "a prefix leaves the rest of the output untouched"
    '(0.0 0.0 1.0 9.0 9.0)
    (let ((out (f64vector 9.0 9.0 9.0 9.0 9.0)))
      (dispatch 'f64 3 (f64vector -1.0 0.0 2.0 -3.0 4.0) out 'relu-deriv 3 1)
      (f64vector->list out)))

  (test "output longer than input: only n elements are written"
    '(0.5 7.0)
    (let ((out (f32vector 7.0 7.0)))
      (dispatch 'f32 1 (f32vector 0.0) out 'sigmoid 4 1)
      (f32vector->list out))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - Repeated calls
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - repeated calls"
  (test-assert "2000 threaded calls give the same result every time"
    ;; Each call uses 2 to 8 threads, so the chunk boundaries move from
    ;; call to call; one element is spoiled before each call and must be
    ;; rewritten by it.  Outputs are compared byte for byte through shared
    ;; bytevector views of the two f32vectors, which checks that they are
    ;; bit-identical (a NaN matches only a NaN with the same bits).
    (let* ((n 50000)
           (in  (th-vector 'f32 n))
           (ref (let ((o (th-make 'f32 n 0.0)))
                  ((plain-kernel 'sigmoid 'f32) n in o)
                  o))
           (out (th-make 'f32 n 0.0))
           (ref-bytes (f32vector->bytevector/shared ref))
           (out-bytes (f32vector->bytevector/shared out)))
      (let loop ((k 0))
        (or (= k 2000)
            (begin
              (f32vector-set! out (modulo (* k 7919) n) -1.0)
              (dispatch 'f32 n in out 'sigmoid (+ 2 (modulo k 7)) 1)
              (and (equal? out-bytes ref-bytes) (loop (+ k 1)))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5 - Settings
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - settings"
  (test-assert "the default thread count is between 1 and the maximum"
    (<= 1 (crunch-thread-count) crunch-max-threads))
  (test-assert "the default min-chunk is positive"
    (> (crunch-thread-min-chunk) 0))
  (test-assert "at least one processor is reported"
    (>= (crunch-available-processors) 1))

  (test "backend kernels follow parameterized settings at call time"
    '(1 4 2)
    (let* ((n 4096) (in (th-vector 'f64 n)) (out (th-make 'f64 n 0.0))
           (k (threaded-kernel 'sigmoid 'f64)))
      (list (parameterize ((crunch-thread-count 1)) (k n in out))
            (parameterize ((crunch-thread-count 4) (crunch-thread-min-chunk 1)) (k n in out))
            (parameterize ((crunch-thread-count 4) (crunch-thread-min-chunk 2048)) (k n in out)))))

  (test "with the default min-chunk a small array stays on one thread"
    1
    ((threaded-kernel 'relu 'f32) 100 (th-vector 'f32 100) (th-make 'f32 100 0.0)))

  (test-error "crunch-thread-count rejects 0" (crunch-thread-count 0))
  (test-error "crunch-thread-min-chunk rejects -5" (crunch-thread-min-chunk -5))

  (test "the threaded backend is named crunch-threaded"
    'crunch-threaded (activation-backend-name th-threaded))
  (test-assert "the threaded backend has f32 and f64 kernels for every op"
    (every (lambda (op) (and (threaded-kernel op 'f32) (threaded-kernel op 'f64))) th-ops))
  (test "an unknown op has no chunk kernel"
    #f (crunch-activation-chunk-kernel 'no-such-op 'f32)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6 - Argument checking
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - argument checking"
  (define k32 (crunch-activation-chunk-kernel 'relu 'f32))
  (test-error "size larger than the input"
    (crunch-dispatch-f32 5 (make-f32vector 4 0.0) (make-f32vector 5 0.0) k32 2 1))
  (test-error "size larger than the output"
    (crunch-dispatch-f32 5 (make-f32vector 5 0.0) (make-f32vector 4 0.0) k32 2 1))
  (test-error "negative size"
    (crunch-dispatch-f32 -1 (make-f32vector 4 0.0) (make-f32vector 4 0.0) k32 2 1))
  (test-error "thread count 0"
    (crunch-dispatch-f32 4 (make-f32vector 4 0.0) (make-f32vector 4 0.0) k32 0 1))
  (test-error "min-chunk 0"
    (crunch-dispatch-f32 4 (make-f32vector 4 0.0) (make-f32vector 4 0.0) k32 2 0))
  (test-error "an f64vector given to the f32 dispatcher"
    (crunch-dispatch-f32 4 (make-f64vector 4 0.0) (make-f32vector 4 0.0) k32 2 1))
  (test-error "a kernel that is not a pointer"
    (crunch-dispatch-f64 4 (make-f64vector 4 0.0) (make-f64vector 4 0.0) 'relu 2 1))
  (test-error "the threaded backend kernel checks sizes too"
    ((threaded-kernel 'tanh 'f64) 10 (make-f64vector 4 0.0) (make-f64vector 10 0.0))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7 - SSA replay through the threaded backend
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (th-array-values m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (map (lambda (i)
             (exact->inexact
              (typed-vector-ref data dtype
                                (multi-to-linear-index (linear-to-multi-index i shape)
                                                       strides offset))))
           (iota (shape-size shape))))
    (else (error "th-array-values: not concrete"))))

;; Trace, finalize and replay twice; the loss and gradients of the second
;; replay as value lists.
(define (th-replay make-loss)
  (let-values (((loss params) (make-loss)))
    (let* ((ctx    (make-morphism-context))
           (fwd    (morphism-to-ssa loss))
           (p-vals (filter-map (lambda (p) (ssa-constant-id fwd (am:var-value p))) params))
           (joint  (ssa-vjp fwd p-vals (ssa-loss-binding-val fwd))))
      (ssa-realize/ctx ctx joint)
      (finalize-context! ctx)
      (reset-context! ctx)
      (ssa-realize/ctx ctx joint)
      (reset-context! ctx)
      (map th-array-values (ssa-realize/ctx ctx joint)))))

;; h = relu(x W1 + b1); y = sigmoid(tanh(h W2)); loss = mean(y)
(define (th-mlp dtype)
  (lambda ()
    (let* ((mk  (lambda (n shape f g?)
                  (am:make-var (morph-from-list (map f (iota n)) shape dtype) g?)))
           (xv  (mk 24 #(8 3) (lambda (i) (- (* 0.29 i) 3.0)) #f))
           (W1  (mk 15 #(3 5) (lambda (i) (* 0.21 (sin (+ i 1.0)))) #t))
           (b1  (mk 5  #(5)   (lambda (i) (- (* 0.1 i) 0.2)) #t))
           (W2  (mk 10 #(5 2) (lambda (i) (* 0.3 (cos (+ i 0.5)))) #t))
           (h   (am:var-relu (am:var+ (am:var-matmul xv W1) b1)))
           (y   (am:var-sigmoid (am:var-tanh (am:var-matmul h W2)))))
      (values (am:var-mean y) (list W1 b1 W2)))))

;; A copy of the threaded backend whose kernels record the largest thread
;; count any call used.
(define (recording-backend record)
  (let ((be (make-activation-backend 'recording-threaded)))
    (for-each
     (lambda (key)
       (let ((k (lookup-activation-kernel th-threaded (car key) (cdr key))))
         (activation-backend-add-kernel!
          be (car key) (cdr key)
          (lambda (n in out)
            (let ((t (k n in out)))
              (set-car! record (max (car record) t))
              t)))))
     (activation-backend-kernels th-threaded))
    be))

(test-group "crunch threads - SSA replay equivalence"
  (for-each
   (lambda (dtype)
     (let ((record (list 0)))
       (register-activation-backend! #f)
       (let ((without (th-replay (th-mlp dtype))))
         (register-activation-backend! (recording-backend record))
         (let ((with (parameterize ((crunch-thread-count 4) (crunch-thread-min-chunk 1))
                       (th-replay (th-mlp dtype)))))
           (register-activation-backend! #f)
           (test-assert (string-append "MLP " (symbol->string dtype)
                                       ": replay used several threads")
             (> (car record) 1))
           (test-assert (string-append "MLP " (symbol->string dtype)
                                       ": loss and gradients identical to the fallback")
             (and (= (length without) (length with) 4)
                  (every th-same-lists? without with)))))))
   '(f64 f32)))

(register-activation-backend! #f)
(test-exit)
