;;; tests/test-crunch-activations.scm
;;; Test suite for array-morphisms-crunch-activations: the crunch-compiled
;;; activation kernels and the activation backend built from them.
;;;
;;; The kernels are meant to reproduce the Scheme combiners they replace
;;; exactly, so kernel outputs are compared with the combiner results
;;; element by element with eqv? (NaN is accepted only against NaN), not
;;; within a tolerance.
;;;
;;; Organisation:
;;;   Group 1 - Backend construction
;;;   Group 2 - Kernel values against the Scheme combiners (f32 and f64)
;;;   Group 3 - Sizes, prefixes, in-place use and argument checking
;;;   Group 4 - SSA replay with and without the crunch backend

(import scheme (chicken base)
        test
        (only srfi-1 iota every filter-map)
        srfi-4
        datatype
        array-morphisms-core
        array-morphisms-basic-ops
        array-morphisms-realization
        array-morphisms-context
        array-morphisms-activation-exec
        (prefix array-morphisms-grad am:)
        array-morphisms-ssa
        array-morphisms-crunch-activations
        (only array-morphisms-crunch-threads crunch-thread-count crunch-thread-min-chunk)
        (only (chicken format) sprintf))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Identical flonums: eqv? distinguishes 0.0 from -0.0; NaN matches NaN.
(define (same-flonum? a b)
  (or (and (nan? a) (nan? b)) (eqv? a b)))

(define (same-lists? a b)
  (and (= (length a) (length b)) (every same-flonum? a b)))

;; Inputs covering signed zeros, tiny and large magnitudes (exp overflow
;; and underflow), infinities and NaN.
(define activation-inputs
  (list 0.0 -0.0 1e-300 -1e-300 0.25 -0.25 0.5 -1.0 1.0 2.0 -3.5 20.0 -20.0
        88.0 -88.0 400.0 -400.0 710.0 -710.0 +inf.0 -inf.0 +nan.0))

;; Derivative inputs: sigmoid-deriv and tanh-deriv receive forward outputs,
;; relu-deriv receives forward inputs; the general inputs cover all three.
(define (list->f64 l) (list->f64vector l))
(define (list->f32 l) (list->f32vector l))

;; Values of the fallback: realize the basic-ops morphism, which evaluates
;; the combiner of basic-ops.scm per element.
(define (fallback-values morph-op lst dtype)
  (let ((m (morph-from-list lst (vector (length lst)) dtype)))
    (cases array-morphism (realize (morph-op m))
      (concrete-array (data shape strides offset dt alloc-id batch-axis)
        (if (eq? dt 'f32) (f32vector->list data) (f64vector->list data)))
      (else (error "fallback-values: not concrete")))))

;; The derivative maps exactly as ssa-vjp emits them.
(define deriv-combiners
  `((relu-deriv    . ,(lambda (xv) (if (> xv 0.0) 1.0 0.0)))
    (sigmoid-deriv . ,(lambda (sv) (* sv (- 1.0 sv))))
    (tanh-deriv    . ,(lambda (tv) (- 1.0 (* tv tv))))))

(define (combiner-values f lst dtype)
  (if (eq? dtype 'f32)
      (let* ((in  (list->f32 lst))
             (out (make-f32vector (length lst) 0.0)))
        (execute-flat-unary-compute f in 'f32 out (length lst) 'f32)
        (f32vector->list out))
      (let* ((in  (list->f64 lst))
             (out (make-f64vector (length lst) 0.0)))
        (execute-flat-unary-compute f in 'f64 out (length lst) 'f64)
        (f64vector->list out))))

(define (kernel-values kernel lst dtype)
  (if (eq? dtype 'f32)
      (let ((in (list->f32 lst)) (out (make-f32vector (length lst) 0.0)))
        (kernel (length lst) in out)
        (f32vector->list out))
      (let ((in (list->f64 lst)) (out (make-f64vector (length lst) 0.0)))
        (kernel (length lst) in out)
        (f64vector->list out))))

(define (backend-kernel op dtype)
  (lookup-activation-kernel (make-crunch-activation-backend) op dtype))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - Backend construction
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define all-ops '(relu sigmoid tanh relu-deriv sigmoid-deriv tanh-deriv))

(test-group "crunch activations - Backend construction"
  (test-assert "make-crunch-activation-backend returns an activation backend"
    (activation-backend? (make-crunch-activation-backend)))
  (test "the backend is named crunch"
    'crunch (activation-backend-name (make-crunch-activation-backend)))
  (test "crunch-activation-ops lists the six default activation ops"
    all-ops (crunch-activation-ops))
  (test-assert "every op has an f32 and an f64 kernel"
    (let ((be (make-crunch-activation-backend)))
      (every (lambda (op)
               (and (procedure? (lookup-activation-kernel be op 'f32))
                    (procedure? (lookup-activation-kernel be op 'f64))))
             all-ops)))
  (test-assert "constructing the backend leaves every op registered"
    (begin (make-crunch-activation-backend)
           (every activation-op-registered? all-ops))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - Kernel values against the Scheme combiners
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch activations - forward kernels equal basic-ops combiners"
  (for-each
   (lambda (dtype)
     (for-each
      (lambda (entry)
        (let ((op (car entry)) (morph-op (cadr entry)) (raw (caddr entry)))
          (test-assert (string-append (symbol->string op) " " (symbol->string dtype)
                                      ": backend kernel")
            (same-lists? (kernel-values (backend-kernel op dtype) activation-inputs dtype)
                         (fallback-values morph-op activation-inputs dtype)))
          (test-assert (string-append (symbol->string op) " " (symbol->string dtype)
                                      ": raw kernel")
            (same-lists? (kernel-values raw activation-inputs dtype)
                         (fallback-values morph-op activation-inputs dtype)))))
      (if (eq? dtype 'f32)
          `((relu ,morph-relu ,crunch-relu-f32)
            (sigmoid ,morph-sigmoid ,crunch-sigmoid-f32)
            (tanh ,morph-tanh-am ,crunch-tanh-f32))
          `((relu ,morph-relu ,crunch-relu-f64)
            (sigmoid ,morph-sigmoid ,crunch-sigmoid-f64)
            (tanh ,morph-tanh-am ,crunch-tanh-f64)))))
   '(f64 f32))

  (test "relu maps -0.0 and NaN to 0.0, as (max 0.0 x) does"
    '(#t #t #t #t)
    (let ((out64 (kernel-values crunch-relu-f64 '(-0.0 +nan.0) 'f64))
          (out32 (kernel-values crunch-relu-f32 '(-0.0 +nan.0) 'f32)))
      (map (lambda (v) (eqv? v 0.0)) (append out64 out32)))))

(test-group "crunch activations - derivative kernels equal ssa-vjp maps"
  (for-each
   (lambda (dtype)
     (for-each
      (lambda (op)
        (test-assert (string-append (symbol->string op) " " (symbol->string dtype))
          (same-lists? (kernel-values (backend-kernel op dtype) activation-inputs dtype)
                       (combiner-values (cdr (assq op deriv-combiners))
                                        activation-inputs dtype))))
      '(relu-deriv sigmoid-deriv tanh-deriv)))
   '(f64 f32)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - Sizes, prefixes, in-place use and argument checking
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch activations - sizes and buffers"
  (test "size 0 writes nothing"
    '(9.0 9.0)
    (let ((out (f64vector 9.0 9.0)))
      ((backend-kernel 'relu 'f64) 0 (f64vector -1.0 1.0) out)
      (f64vector->list out)))

  (test "size 1"
    '(0.0)
    (let ((out (f64vector 9.0)))
      ((backend-kernel 'relu 'f64) 1 (f64vector -1.0) out)
      (f64vector->list out)))

  (test "size n < length touches only the first n elements"
    '(0.0 2.0 9.0)
    (let ((out (f64vector 9.0 9.0 9.0)))
      ((backend-kernel 'relu 'f64) 2 (f64vector -1.0 2.0 -3.0) out)
      (f64vector->list out)))

  (test-assert "in and out may be the same vector (f64)"
    (let* ((xs  '(-2.0 -0.5 0.0 0.5 2.0))
           (buf (list->f64 xs)))
      ((backend-kernel 'sigmoid 'f64) 5 buf buf)
      (same-lists? (f64vector->list buf)
                   (fallback-values morph-sigmoid xs 'f64))))

  (test-assert "in and out may be the same vector (f32)"
    (let* ((xs  '(-2.0 -0.5 0.0 0.5 2.0))
           (buf (list->f32 xs)))
      ((backend-kernel 'tanh 'f32) 5 buf buf)
      (same-lists? (f32vector->list buf)
                   (fallback-values morph-tanh-am xs 'f32))))

  (test-assert "a large array (100000 elements)"
    (let* ((n   100000)
           (xs  (map (lambda (i) (* 0.001 (- i 50000))) (iota n)))
           (out (make-f32vector n 0.0)))
      ((backend-kernel 'sigmoid 'f32) n (list->f32 xs) out)
      (same-lists? (f32vector->list out)
                   (fallback-values morph-sigmoid xs 'f32))))

  (test-error "a size larger than the input vector is an error"
    ((backend-kernel 'relu 'f64) 3 (f64vector 1.0 2.0) (make-f64vector 3 0.0)))
  (test-error "a size larger than the output vector is an error"
    ((backend-kernel 'relu 'f32) 3 (f32vector 1.0 2.0 3.0) (make-f32vector 2 0.0)))
  (test-error "a negative size is an error"
    ((backend-kernel 'relu 'f64) -1 (f64vector 1.0) (make-f64vector 1 0.0))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - SSA replay with and without the crunch backend
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (array-values m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (map (lambda (i)
             (exact->inexact
              (typed-vector-ref data dtype
                                (multi-to-linear-index (linear-to-multi-index i shape)
                                                       strides offset))))
           (iota (shape-size shape))))
    (else (error "array-values: not concrete"))))

;; Build a fresh graph with make-loss, trace it, then replay it twice
;; (the first replay compiles the plan).  Returns (values joint results)
;; with the loss and gradients of the second replay as value lists.
(define (trace-and-replay make-loss)
  (let-values (((loss params) (make-loss)))
    (let* ((ctx    (make-morphism-context))
           (fwd    (morphism-to-ssa loss))
           (p-vals (filter-map (lambda (p) (ssa-constant-id fwd (am:var-value p)))
                               params))
           (joint  (ssa-vjp fwd p-vals (ssa-loss-binding-val fwd))))
      (ssa-realize/ctx ctx joint)
      (finalize-context! ctx)
      (reset-context! ctx)
      (ssa-realize/ctx ctx joint)
      (reset-context! ctx)
      (let ((results (ssa-realize/ctx ctx joint)))
        (values joint (map array-values results))))))

(define (activation-instruction-count joint)
  (let ((p (assq 'ri-activation-unary
                 (cdr (assq 'counts (replay-plan-stats joint))))))
    (if p (cdr p) 0)))

;; A two-layer network with every activation and its derivative:
;;   h = relu(x @ W1 + b1);  y = sigmoid(tanh(h @ W2));  loss = mean(y)
(define (make-mlp-loss dtype)
  (lambda ()
    (let* ((mk  (lambda (n shape f requires-grad)
                  (am:make-var (morph-from-list (map f (iota n)) shape dtype)
                               requires-grad)))
           (xv  (mk 12 #(4 3) (lambda (i) (- (* 0.37 i) 2.0)) #f))
           (W1  (mk 15 #(3 5) (lambda (i) (* 0.21 (sin (+ i 1.0)))) #t))
           (b1  (mk 5  #(5)   (lambda (i) (- (* 0.1 i) 0.2)) #t))
           (W2  (mk 10 #(5 2) (lambda (i) (* 0.3 (cos (+ i 0.5)))) #t))
           (h   (am:var-relu (am:var+ (am:var-matmul xv W1) b1)))
           (y   (am:var-sigmoid (am:var-tanh (am:var-matmul h W2)))))
      (values (am:var-mean y) (list W1 b1 W2)))))

(define (replay-with-and-without make-loss)
  (register-activation-backend! #f)
  (let-values (((j0 without) (trace-and-replay make-loss)))
    (register-activation-backend! (make-crunch-activation-backend))
    (let-values (((j1 with) (trace-and-replay make-loss)))
      (register-activation-backend! #f)
      (values j1 without with))))

(test-group "crunch activations - SSA replay equivalence"
  (for-each
   (lambda (dtype)
     (let-values (((joint without with) (replay-with-and-without (make-mlp-loss dtype))))
       (test-assert (string-append "MLP " (symbol->string dtype)
                                   ": plan uses activation instructions")
         (> (activation-instruction-count joint) 0))
       (test-assert (string-append "MLP " (symbol->string dtype)
                                   ": loss and gradients identical with and without crunch")
         (and (= (length without) (length with) 4)
              (every same-lists? without with)))
       (test-assert (string-append "MLP " (symbol->string dtype)
                                   ": gradients are not all zero")
         (every (lambda (g) (not (every zero? g))) (cdr with)))))
   '(f64 f32))

  (test-assert "relu gradient through the crunch backend is the analytical one"
    (let-values (((joint without with)
                  (replay-with-and-without
                   (lambda ()
                     (let ((xv (am:make-var (morph-from-list '(-2.0 -1.0 0.0 1.0 2.0)
                                                             #(5) 'f64) #t)))
                       (values (am:var-mean (am:var-relu xv)) (list xv)))))))
      (same-lists? (cadr with) '(0.0 0.0 0.0 0.2 0.2)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Binary, reduction and copy kernels
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Values that exercise signed zeros, infinities, NaN and rounding.
(define special-values
  (list 0.0 -0.0 1.0 -1.0 0.1 -2.5 1e30 -1e-30 +inf.0 -inf.0 +nan.0 3.25 7e-8))

(define (values-vector make n)
  (let ((v (make n)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (let ((x (list-ref special-values (modulo (* i 7) (length special-values)))))
        (if (f32vector? v)
            (f32vector-set! v i (+ x (* 0.001 i)))
            (f64vector-set! v i (+ x (* 0.001 i))))))))

(define (srfi4->list v) (if (f32vector? v) (f32vector->list v) (f64vector->list v)))

(define binary-combiners
  (list (cons 'add +) (cons 'sub -) (cons 'mul *) (cons 'div /)))

(test-group "crunch kernels - binary ops equal the Scheme combiners"
  (let ((be (make-crunch-threaded-activation-backend)))
    (for-each
     (lambda (entry)
       (let ((op (car entry)) (f (cdr entry)))
         (test-assert (sprintf "~A is registered" op) (binary-op-registered? op))
         (for-each
          (lambda (dtype make)
            (for-each
             (lambda (threads)
               (test-assert (sprintf "~A ~A with ~A thread(s)" op dtype threads)
                 (parameterize ((crunch-thread-count threads)
                                (crunch-thread-min-chunk 1000))
                   (let* ((n 50001)
                          (a (values-vector make n))
                          (b (let ((v (values-vector make n)))
                               ;; shift so that a and b pair different values
                               (let ((w (make n)))
                                 (do ((i 0 (+ i 1))) ((= i n) w)
                                   (if (f32vector? w)
                                       (f32vector-set! w i (f32vector-ref v (modulo (+ i 5) n)))
                                       (f64vector-set! w i (f64vector-ref v (modulo (+ i 5) n))))))))
                          (out (make n))
                          (ref (make n)))
                     ((lookup-binary-kernel be op dtype) n a b out)
                     (execute-flat-binary-compute f a dtype b dtype ref n dtype)
                     (same-lists? (srfi4->list out) (srfi4->list ref))))))
             '(1 8)))
          '(f32 f64)
          (list make-f32vector make-f64vector))))
     binary-combiners)
    (test-error "binary kernel rejects a size beyond its vectors"
      ((lookup-binary-kernel be 'add 'f32) 10 (make-f32vector 5) (make-f32vector 10) (make-f32vector 10)))))

(test-group "crunch kernels - reductions equal the Scheme fast path"
  (for-each
   (lambda (rop)
     (for-each
      (lambda (axis)
        (for-each
         (lambda (dtype make)
           (for-each
            (lambda (dims)
              (let* ((rows (car dims)) (cols (cadr dims))
                     (src (values-vector make (* rows cols)))
                     (shape (vector rows cols))
                     (out-shape (if (= axis 0) (vector cols) (vector rows)))
                     (n-out (if (= axis 0) cols rows))
                     (run (lambda ()
                            (let ((out (make n-out)))
                              (execute-reduction-morphism rop out out-shape src shape
                                                          (vector cols 1) 0 (list axis)
                                                          #f #f dtype dtype)
                              (srfi4->list out)))))
                (test-assert (sprintf "~A over axis ~A, ~A, ~Ax~A" rop axis dtype rows cols)
                  (let ((scheme (begin (register-activation-backend! #f) (run)))
                        (kernel (begin (register-activation-backend! (make-crunch-activation-backend))
                                       (run))))
                    (register-activation-backend! #f)
                    (same-lists? scheme kernel)))))
            '((1 1) (7 3) (300 16) (5 257))))
         '(f32 f64)
         (list make-f32vector make-f64vector)))
      '(0 1)))
   '(sum mean max min))
  (test-assert "reduction kernels are installed for every op, axis and dtype"
    (let ((be (make-crunch-activation-backend)))
      (every (lambda (key) (lookup-reduction-kernel be (car key) (cadr key) (caddr key)))
             (list '(sum 0 f32) '(mean 1 f64) '(max 0 f64) '(min 1 f32))))))

(test-group "crunch kernels - strided copy"
  (let ((copy (lookup-copy-kernel (make-crunch-activation-backend) 'f32)))
    ;; Reads element (i,j,...) of a view with the given shape, strides and
    ;; offset, in row-major order.
    (define (reference src off shape strides)
      (let* ((rank (vector-length shape))
             (n (let loop ((k 0) (p 1)) (if (= k rank) p (loop (+ k 1) (* p (vector-ref shape k)))))))
        (map (lambda (i)
               (let loop ((k (- rank 1)) (rem i) (phys off))
                 (if (< k 0)
                     (f32vector-ref src phys)
                     (loop (- k 1) (quotient rem (vector-ref shape k))
                           (+ phys (* (remainder rem (vector-ref shape k)) (vector-ref strides k)))))))
             (iota n))))
    (for-each
     (lambda (view)
       (let* ((shape (car view)) (strides (cadr view)) (off (caddr view))
              (src (values-vector make-f32vector 200))
              (n (let loop ((k 0) (p 1)) (if (= k (vector-length shape)) p (loop (+ k 1) (* p (vector-ref shape k))))))
              (dst (make-f32vector n)))
         (copy src off shape strides dst)
         (test-assert (sprintf "copy of shape ~A strides ~A offset ~A" shape strides off)
           (same-lists? (f32vector->list dst) (reference src off shape strides)))))
     (list (list (vector 7) (vector 3) 2)
           (list (vector 5 6) (vector 1 5) 0)        ; transpose of a 6x5 matrix
           (list (vector 3 4 5) (vector 1 3 12) 0)   ; reversed axes
           (list (vector 2 3 4 5) (vector 60 1 3 12) 1)
           (list (vector 0 4) (vector 4 1) 0)))
    (test-error "copy rejects a view reaching beyond the source"
      (copy (make-f32vector 10) 0 (vector 4 4) (vector 4 1) (make-f32vector 16)))))

(register-activation-backend! #f)
(test-exit)
