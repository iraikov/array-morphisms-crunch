;;; tests/test-crunch-fused.scm
;;; Test suite for the fused kernels of array-morphisms-crunch-activations.
;;;
;;; A fused kernel computes a whole segop body in one pass.  It must give
;;; exactly the values of the reference interpreter of array-morphisms on
;;; the same body, which are those of the separate bindings the body was
;;; fused from, so outputs are compared element by element with eqv?
;;; (NaN is accepted only against NaN).
;;;
;;; Organisation:
;;;   Group 1 - The fused kernels the backends hold
;;;   Group 2 - Kernel values against the interpreter (f32 and f64, one
;;;             thread and several), at sizes around the thread chunks
;;;   Group 3 - Argument checking
;;;   Group 4 - SSA training graphs: fusion into the kernels

(import scheme (chicken base)
        test
        (only srfi-1 iota every filter-map)
        srfi-4
        datatype
        array-morphisms-core
        array-morphisms-basic-ops
        array-morphisms-context
        array-morphisms-activation-exec
        array-morphisms-segop
        (prefix array-morphisms-grad am:)
        array-morphisms-ssa
        array-morphisms-crunch-activations
        (only array-morphisms-crunch-threads crunch-thread-count crunch-thread-min-chunk)
        (only (chicken format) sprintf))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (fz-same? a b) (or (and (nan? a) (nan? b)) (eqv? a b)))

;; The f64 form of an f32 body: its casts removed.
(define (fz-strip e)
  (cond ((not (pair? e)) e)
        ((eq? (car e) 'cast) (fz-strip (caddr e)))
        ((eq? (car e) 'in) e)
        (else (cons (car e) (map fz-strip (cdr e))))))

(define (fz-n-inputs body)
  (let walk ((e body))
    (cond ((not (pair? e)) 0)
          ((eq? (car e) 'in) (+ 1 (cadr e)))
          (else (apply max 0 (map walk (cdr e)))))))

;; Special values first, then a spread of ordinary ones.
(define fz-specials (list +nan.0 +inf.0 -inf.0 -0.0 0.0 1e30 -1e30 1e-30 -1e-30 1.0 -1.0 0.5))

(define (fz-vector dtype n seed)
  (let ((v (allocate-typed-vector dtype n)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (typed-vector-set! v dtype i
                         (if (< (modulo (+ i seed) 50) (length fz-specials))
                             (list-ref fz-specials (modulo (+ i seed) 50))
                             (* 3.0 (sin (+ (* 0.731 i) seed))))))))

;; The values the reference interpreter gives for body on the inputs ins.
(define (fz-reference body dtype ins n)
  (let* ((space (vector n))
         (sg (make-seg-map space
                           (map (lambda (k) (make-segop-operand k dtype space #(1) 0 (identity-index-map 1)))
                                (iota (length ins)))
                           (list (make-segop-operand 'out dtype space #(1) 0 (identity-index-map 1)))
                           (list body)))
         (out (allocate-typed-vector dtype n)))
    (segop-interpret sg ins (list out))
    out))

(define (fz-mismatches a b dtype n)
  (let loop ((i 0) (m 0))
    (if (= i n)
        m
        (loop (+ i 1) (if (fz-same? (typed-vector-ref a dtype i) (typed-vector-ref b dtype i)) m (+ m 1))))))

(define fz-backends
  (list (cons "one thread" (make-crunch-activation-backend))
        (cons "threaded" (make-crunch-threaded-activation-backend))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - The fused kernels the backends hold
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "fused kernels: catalogue"
  (test-assert "there are fused bodies" (pair? (crunch-fused-bodies)))
  (for-each
   (lambda (body)
     (test-assert (sprintf "~S is canonical" body)
       (equal? body (call-with-values (lambda () (canonical-scalar-body body)) (lambda (c p) c))))
     (for-each
      (lambda (b)
        (test-assert (sprintf "~A backend holds the f32 kernel of ~S" (car b) body)
          (lookup-fused-kernel (cdr b) body 'f32))
        (test-assert (sprintf "~A backend holds the f64 kernel of ~S" (car b) body)
          (lookup-fused-kernel (cdr b) (fz-strip body) 'f64)))
      fz-backends))
   (crunch-fused-bodies)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - Kernel values against the interpreter
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Sizes below, at and above multiples of the minimum chunk, so that the
;; threaded kernels split the work in several ways.
(define fz-sizes
  (let ((c (crunch-thread-min-chunk)))
    (list 1 7 (- c 1) c (+ c 1) (+ (* 2 c) 3) (+ (* 4 c) 5))))

(test-group "fused kernels: values"
  (for-each
   (lambda (body32)
     (for-each
      (lambda (dtype)
        (let* ((body (if (eq? dtype 'f32) body32 (fz-strip body32)))
               (k    (fz-n-inputs body)))
          (for-each
           (lambda (b)
             (let ((kernel (lookup-fused-kernel (cdr b) body dtype)))
               (test (sprintf "~A ~A ~S: equal to the interpreter at every size" (car b) dtype body)
                 (map (lambda (n) 0) fz-sizes)
                 (map (lambda (n)
                        (let ((ins (map (lambda (j) (fz-vector dtype n (* 7 j))) (iota k)))
                              (out (allocate-typed-vector dtype n)))
                          (kernel n ins out)
                          (fz-mismatches out (fz-reference body dtype ins n) dtype n)))
                      fz-sizes))))
           fz-backends)))
      '(f32 f64)))
   (crunch-fused-bodies))
  ;; the casts matter: x*y + z with x = y = 1 + 2^-12 and z = -1 gives
  ;; 2^-11 when the product is rounded to f32, and 2^-11 + 2^-24 without
  (let* ((body '(add (in 0) (cast f32 (mul (in 1) (in 2)))))
         (x0 (+ 1.0 (expt 2.0 -12)))
         (z (f32vector -1.0)) (x (f32vector x0)) (y (f32vector x0)) (out (f32vector 0.0)))
    ((lookup-fused-kernel (cdar fz-backends) body 'f32) 1 (list z x y) out)
    (test "the f32 kernel rounds the product before adding" (expt 2.0 -11) (f32vector-ref out 0))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - Argument checking
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "fused kernels: argument checks"
  (let ((k (lookup-fused-kernel (cdar fz-backends) '(mul (in 0) (cast f32 (relu-deriv (in 1)))) 'f32)))
    (test-error "a missing input is an error" (k 4 (list (make-f32vector 4)) (make-f32vector 4)))
    (test-error "a size beyond an input is an error"
      (k 5 (list (make-f32vector 4) (make-f32vector 5)) (make-f32vector 5)))
    (test-error "a size beyond the output is an error"
      (k 5 (list (make-f32vector 5) (make-f32vector 5)) (make-f32vector 4)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - SSA training graphs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; loss = sum(relu(x W) * c) for a parameter W: the backward pass
;; multiplies the incoming gradient by relu-deriv of the product, which
;; the fusion pass fuses into one kernel when the backend holds it.
(define (fz-loss dtype)
  (let* ((x (am:make-var (morph-from-list (map (lambda (i) (sin (* 0.37 i))) (iota 48)) #(8 6) dtype) #f))
         (W (am:make-var (morph-from-list (map (lambda (i) (cos (* 0.53 i))) (iota 24)) #(6 4) dtype) #t))
         (c (am:make-var (morph-from-list (map (lambda (i) (- (* 0.25 i) 3.0)) (iota 32)) #(8 4) dtype) #f)))
    (values (am:var-sum (am:var* (am:var-relu (am:var-matmul x W)) c)) (list W))))

;; (list trace replay tags) of the loss and gradient under fusion policy
;; fusion, with backend be active.
(define (fz-train dtype be fusion)
  (register-activation-backend! be)
  (let-values (((loss params) (fz-loss dtype)))
    (let* ((fwd (morphism-to-ssa loss))
           (pv  (filter-map (lambda (p) (ssa-constant-id fwd (am:var-value p))) params))
           (joint (parameterize ((segop-fusion fusion)) (ssa-vjp fwd pv (ssa-loss-binding-val fwd))))
           (ctx (make-morphism-context))
           (lists (lambda (rs) (map (lambda (m) (cases array-morphism m
                                                  (concrete-array (d s st off dt _ _)
                                                    (map (lambda (i) (typed-vector-ref d dt i))
                                                         (iota (shape-size s))))
                                                  (else #f)))
                                    rs)))
           (trace (lists (ssa-realize/ctx ctx joint))))
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((replay (lists (ssa-realize/ctx ctx joint))))
        (register-activation-backend! #f)
        (list trace replay (map car (cdr (assq 'counts (replay-plan-stats joint)))))))))

(test-group "fused kernels: SSA"
  (for-each
   (lambda (dtype)
     (for-each
      (lambda (b)
        (let ((on (fz-train dtype (cdr b) #t)) (off (fz-train dtype (cdr b) #f)))
          (test-assert (sprintf "~A ~A: the default fuses g * relu-deriv(x) into a kernel" (car b) dtype)
            (memq 'segop/fused-kernel (caddr on)))
          (test-assert (sprintf "~A ~A: without fusion there is no fused kernel" (car b) dtype)
            (not (memq 'segop/fused-kernel (caddr off))))
          (test-assert (sprintf "~A ~A: the trace equals the unfused one" (car b) dtype)
            (equal? (car on) (car off)))
          (test-assert (sprintf "~A ~A: the replay equals the unfused one" (car b) dtype)
            (equal? (cadr on) (cadr off)))))
      fz-backends))
   '(f32 f64)))

(test-exit)
