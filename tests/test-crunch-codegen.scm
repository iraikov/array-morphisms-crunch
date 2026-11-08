;;; tests/test-crunch-codegen.scm
;;; Test suite for the code generator array-morphisms-crunch-codegen.
;;;
;;; A generated kernel must give exactly the values that the host computes
;;; for the same segop, so outputs are compared element by element with
;;; eqv?, NaN matching any NaN.  Floating-point segops are compared with the
;;; reference interpreter of array-morphisms; segops with integer operands,
;;; whose values follow the Scheme procedures of the ops, with the
;;; interpreter running those procedures.
;;;
;;; The segops of each group are selected in one plan context, so their
;;; kernels are compiled as one module.  The kernels are compiled in a
;;; temporary cache directory.
;;;
;;; Organisation:
;;;   Group 1 - Selection: when the backend takes a segop and when not
;;;   Group 2 - seg-maps over f32 and f64 operands, flat and with special values
;;;   Group 3 - Broadcast, transposed and offset operands, ranks 0 to 4
;;;   Group 4 - Several outputs
;;;   Group 5 - Mixed and integer dtypes, and the failure fallback
;;;   Group 6 - Thread counts
;;;   Group 6b - seg-reds, against the interpreter and the host executor
;;;   Group 7 - The cache and compile failures
;;;   Group 8 - Native queries and SSA programs

(import scheme (chicken base) (chicken file) (only (scheme base) open-output-string)
        test
        (only srfi-1 iota every any filter list-index cons*)
        srfi-4
        datatype
        array-morphisms-core
        array-morphisms-basic-ops
        array-morphisms-context
        array-morphisms-segop
        array-morphisms-scalar-ops
        (prefix array-morphisms-grad am:)
        array-morphisms-ssa
        array-morphisms-crunch-codegen
        (only array-morphisms-crunch-threads crunch-thread-min-chunk)
        (only (chicken format) sprintf))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; The temporary directories the tests compile in, removed at the end.
(define temp-dirs '())
(define (temp-dir)
  (let ((d (create-temporary-directory)))
    (set! temp-dirs (cons d temp-dirs))
    d))

(crunch-codegen-cache-directory (temp-dir))

;; The code generator followed by the scheme backend.
(define (cg-backends)
  (parameterize ((segop-backends
                  (filter (lambda (b) (eq? (segop-backend-name b) 'scheme)) (segop-backends))))
    (register-crunch-codegen-backend!)
    (segop-backends)))

;; Selects every segop of sgs in one plan context and finalizes it.
;; Returns the list of (tag . executor) and the plan notes.
(define (cg-select-all sgs)
  (let ((ctx (make-segop-plan-context)))
    (let ((sels (parameterize ((segop-plan-context ctx)
                               (segop-backends (cg-backends))
                               (crunch-codegen #t))
                  (map select-segop-executor sgs))))
      (values sels (segop-plan-finalize! ctx)))))

;; The elements of an output vector as a list.
(define (vec->list v)
  (cond ((f32vector? v) (f32vector->list v)) ((f64vector? v) (f64vector->list v))
        ((s32vector? v) (s32vector->list v)) ((s64vector? v) (s64vector->list v))
        ((u32vector? v) (u32vector->list v))
        (else (error "vec->list: unsupported vector" v))))

;; Equal element by element: flonums with eqv?, which tells -0.0 from 0.0,
;; and any NaN equal to any NaN, since the sign of a NaN is not specified
;; by the arithmetic (the C compiler may compute -1.0 * x as a sign flip).
(define (same-element? a b)
  (or (and (flonum? a) (flonum? b) (nan? a) (nan? b)) (eqv? a b)))

(define (same-bits? a b)
  (let ((xs (vec->list a)) (ys (vec->list b)))
    (and (= (length xs) (length ys)) (every same-element? xs ys))))

;; A test case: a segop, the input arrays by slot, and the sizes of its
;; output buffers.
(define-record tcase sg env out-sizes)

(define (case-in-bufs c)
  (map (lambda (o) (operand-data (cdr (assq (segop-operand-slot o) (tcase-env c)))))
       (segop-inputs (tcase-sg c))))

(define (operand-data arr)
  (cases array-morphism arr (concrete-array (d _ _ _ _ _ _) d) (else #f)))

(define (case-out-bufs c)
  (map (lambda (o n) (allocate-typed-vector (segop-operand-dtype o) n))
       (segop-outputs (tcase-sg c)) (tcase-out-sizes c)))

(define (run-executor exec c)
  (let ((outs (case-out-bufs c))
        (deref (lambda (slot) (cdr (assq slot (tcase-env c))))))
    (if (null? (cdr outs)) (exec deref (car outs)) (exec deref (car outs) (cdr outs)))
    outs))

(define (run-reference c)
  (let ((outs (case-out-bufs c)))
    (segop-interpret (tcase-sg c) (case-in-bufs c) outs)
    outs))

;; For each case, the tag, and whether the executor's outputs equal the
;; reference's bit for bit; and the plan notes.
(define (check-cases cases)
  (let-values (((sels notes) (cg-select-all (map tcase-sg cases))))
    (values (map (lambda (sel c)
                   (cons (car sel)
                         (every same-bits? (run-executor (cdr sel) c) (run-reference c))))
                 sels cases)
            notes)))

(define (all-generated-and-equal? results)
  (every (lambda (r) (and (eq? (car r) 'segop/codegen-map) (cdr r))) results))

(define (codegen-note notes key)
  (let ((cg (assq 'crunch-codegen notes)))
    (and cg (cdr (assq key (cdr cg))))))

;; A concrete array of the given dtype, shape, strides and offset over a
;; buffer of n elements filled by f.
(define (array dt shape strides offset n f)
  (let ((v (allocate-typed-vector dt n)))
    (do ((i 0 (+ i 1))) ((= i n)) (typed-vector-set! v dt i (f i)))
    (concrete-array v shape strides offset dt -1 -1)))

(define (flat-array dt shape f)
  (array dt shape (compute-strides shape) 0 (shape-size shape) f))

(define (operand slot arr m)
  (cases array-morphism arr
    (concrete-array (d shape strides offset dt _ _)
      (make-segop-operand slot dt shape strides offset m))
    (else #f)))

(define (out-operand slot dt shape)
  (make-segop-operand slot dt shape (compute-strides shape) 0
                      (identity-index-map (vector-length shape))))

;; A seg-map case over space reading the arrays of env, each through its
;; broadcast map, with one output of dtype out-dt per body.  With fallback?,
;; a single-output case gets the compiled body as its fallback, which the
;; interpreter uses for integer operands.
(define (map-case space env bodies out-dts #!optional (fallback? #f))
  (let* ((rank (vector-length space))
         (ins  (map (lambda (p)
                      (operand (car p) (cdr p)
                               (broadcast-index-map (get-morphism-shape (cdr p)) rank)))
                    env))
         (outs (map (lambda (dt k) (out-operand (string->symbol (sprintf "out~a" k)) dt space))
                    out-dts (iota (length out-dts)))))
    (make-tcase (make-seg-map space ins outs bodies
                             (and fallback? (= 1 (length bodies))
                                  (scalar-expr->procedure (car bodies) (length ins))))
               env
               (map (lambda (o) (shape-size space)) outs))))

(define special-values
  (list 0.0 -0.0 1.0 -1.0 0.5 -2.75 3.0 1e-40 -1e-310 1e-310 +inf.0 -inf.0 +nan.0
        3.4e38 -1e300 1e300 12.5 -0.125 7.0e-3 88.7))

(define (value-at i) (list-ref special-values (modulo i (length special-values))))
(define (value2-at i) (list-ref special-values (modulo (* 7 (+ i 3)) (length special-values))))
(define (positive-at i) (+ 0.25 (abs (* 0.37 (- i 11)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - Selection
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "selection"
  (let ((c (map-case #(4) (list (cons 'a (flat-array 'f32 #(4) value-at))) '((exp (in 0))) '(f32) #t)))
    (test "outside a plan the backend declines" 'segop/flat-unary
      (parameterize ((segop-backends (cg-backends)) (crunch-codegen #t))
        (car (select-segop-executor (tcase-sg c)))))
    (test "with crunch-codegen false the backend declines" 'segop/flat-unary
      (let ((ctx (make-segop-plan-context)))
        (parameterize ((segop-plan-context ctx) (segop-backends (cg-backends)) (crunch-codegen #f))
          (car (select-segop-executor (tcase-sg c))))))
    (test "inside a plan the backend takes the segop" 'segop/codegen-map
      (let ((ctx (make-segop-plan-context)))
        (parameterize ((segop-plan-context ctx) (segop-backends (cg-backends)) (crunch-codegen #t))
          (car (select-segop-executor (tcase-sg c)))))))
  (let ((c (map-case #(4) (list (cons 'a (flat-array 'f32 #(4) value-at)))
                     `((opaque ,(lambda (x) x) (in 0))) '(f32) #t)))
    (test "an opaque body is left to the host" 'segop/flat-unary
      (let ((ctx (make-segop-plan-context)))
        (parameterize ((segop-plan-context ctx) (segop-backends (cg-backends)) (crunch-codegen #t))
          (car (select-segop-executor (tcase-sg c)))))))
  (test "registering places the backend before the scheme backend"
    '(crunch-codegen scheme)
    (map segop-backend-name (cg-backends)))
  (test-assert "registering places the backend after the registry"
    (parameterize ((segop-backends (segop-backends)))
      (register-crunch-codegen-backend!)
      (let ((names (map segop-backend-name (segop-backends))))
        (< (list-index (lambda (n) (eq? n 'registry)) names)
           (list-index (lambda (n) (eq? n 'crunch-codegen)) names))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - f32 and f64 seg-maps
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; The table ops whose Scheme procedures would return a complex number for
;; a negative argument are tested on positive values only.
(define positive-only '(log sqrt pow))

(define (op-cases dt)
  (map (lambda (op)
         (let* ((n (scalar-op-arity op))
                (f (if (memq op positive-only) positive-at value-at))
                (g (if (memq op positive-only) positive-at value2-at))
                (env (if (= n 1)
                         (list (cons 'a (flat-array dt #(41) f)))
                         (list (cons 'a (flat-array dt #(41) f)) (cons 'b (flat-array dt #(41) g))))))
           (map-case #(41) env (list (cons op (map (lambda (k) `(in ,k)) (iota n)))) (list dt))))
       (scalar-op-names)))

(test-group "table ops, flat"
  (for-each
   (lambda (dt)
     (let-values (((results notes) (check-cases (op-cases dt))))
       (for-each (lambda (op r)
                   (test (sprintf "~a ~a: generated, equal to the interpreter" dt op)
                         '(segop/codegen-map . #t) (values r)))
                 (scalar-op-names) results)
       (test (sprintf "~a: every kernel is made" dt) (length results) (codegen-note notes 'kernels))))
   '(f32 f64)))

(define fused-bodies
  '((mul (in 0) (cast f32 (relu-deriv (in 1))))
    (mul (in 0) (cast f32 (sigmoid-deriv (in 1))))
    (add (in 0) (cast f32 (mul (in 1) (in 2))))
    (negate (cast f32 (mul (in 0) (in 1))))
    (exp (cast f32 (negate (in 0))))
    (mul (in 0) (cast f32 (mul (in 1) (cast f32 (sigmoid (in 1))))))
    (sub (cast f32 (tanh (cast f32 (add (in 0) (in 1))))) (cast f32 (div (in 2) (in 0))))
    (gt (cast f32 (mul (in 0) (in 1))) (in 2))
    (add (add (in 0) (in 1)) (add (in 0) (in 1)))
    (+ (* (in 0) (in 1)) (- (in 2)))
    (fpabs (/ (in 0) (expt (in 1) 2.0)))))

(test-group "fused bodies"
  (let ((env (list (cons 'a (flat-array 'f32 #(3 13) value-at))
                   (cons 'b (flat-array 'f32 #(3 13) value2-at))
                   (cons 'c (flat-array 'f32 #(3 13) (lambda (i) (* 0.3 (- i 20))))))))
    (let-values (((results notes) (check-cases (map (lambda (b) (map-case #(3 13) env (list b) '(f32)))
                                                    fused-bodies))))
      (for-each (lambda (b r) (test (sprintf "~s" b) '(segop/codegen-map . #t) (values r)))
                fused-bodies results)
      (test "every fused kernel is made" (length results) (codegen-note notes 'kernels)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - Layouts and ranks
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (layout-cases)
  (let* ((body '(add (mul (in 0) (in 1)) (in 2)))
         ;; a transposed view: a 7 x 5 buffer read as its 5 x 7 transpose
         (tr (lambda (dt) (array dt #(5 7) #(1 5) 0 35 value-at)))
         ;; an offset view with a row stride of 9
         (ov (lambda (dt) (array dt #(5 7) #(9 1) 4 50 value2-at))))
    (append
     (map (lambda (dt)
            (map-case #(5 7)
                      (list (cons 'a (flat-array dt #(7) value-at))
                            (cons 'b (flat-array dt #(5 1) value2-at))
                            (cons 'c (flat-array dt #(1 1) (lambda (i) 0.75))))
                      (list body) (list dt)))
          '(f32 f64))
     (map (lambda (dt)
            (map-case #(5 7) (list (cons 'a (tr dt)) (cons 'b (ov dt)) (cons 'c (flat-array dt #(5 7) value-at)))
                      (list body) (list dt)))
          '(f32 f64))
     (list
      (map-case #() (list (cons 'a (flat-array 'f64 #() (lambda (i) 2.5)))
                          (cons 'b (flat-array 'f64 #() (lambda (i) -0.5))))
                '((sub (in 0) (in 1))) '(f64))
      (map-case #(11) (list (cons 'a (flat-array 'f32 #(11) value-at)) (cons 'b (flat-array 'f32 #(1) (lambda (i) 3.0))))
                '((div (in 0) (in 1))) '(f32))
      (map-case #(3 4 5) (list (cons 'a (flat-array 'f32 #(3 4 5) value-at))
                               (cons 'b (flat-array 'f32 #(4 1) value2-at)))
                '((mul (in 0) (cast f32 (sigmoid (in 1))))) '(f32))
      (map-case #(2 3 1 5) (list (cons 'a (flat-array 'f64 #(2 3 1 5) value-at))
                                 (cons 'b (flat-array 'f64 #(5) value2-at)))
                '((relu (add (in 0) (in 1)))) '(f64))
      (map-case #(3 0 4) (list (cons 'a (flat-array 'f32 #(3 0 4) value-at)))
                '((exp (in 0))) '(f32))))))

(test-group "layouts and ranks"
  (let-values (((results notes) (check-cases (layout-cases))))
    (for-each (lambda (name r) (test name '(segop/codegen-map . #t) (values r)))
              '("f32 broadcast row, column and scalar" "f64 broadcast row, column and scalar"
                "f32 transposed and offset views" "f64 transposed and offset views"
                "rank 0" "rank 1 with a scalar" "rank 3 with a broadcast column"
                "rank 4 with an extent of 1" "an empty space")
              results)
    (test "every layout kernel is made" (length results) (codegen-note notes 'kernels))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - Several outputs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "opaque arithmetic"
  (let* ((neg-zero (lambda (i) (if (even? i) -0.0 (value-at i))))
         (env (list (cons 'a (flat-array 'f64 #(19) neg-zero))
                    (cons 'b (flat-array 'f64 #(19) neg-zero))))
         (cases (list (map-case #(19) env (list `(opaque ,+ (in 0) (in 1))) '(f64))
                      (map-case #(19) env (list `(opaque ,* (in 0) (opaque ,- (in 1)))) '(f64))
                      (map-case #(19) env (list `(opaque ,scalar-add (in 0) (opaque ,/ (in 1) (in 0)))) '(f64)))))
    (let-values (((results notes) (check-cases cases)))
      (for-each (lambda (name r) (test name '(segop/codegen-map . #t) (values r)))
                '("Scheme + of signed zeros" "Scheme * and unary -" "scalar-add and Scheme /")
                results))))

(test-group "several outputs"
  (let* ((shared '(cast f32 (sigmoid (in 0))))
         (env (list (cons 'a (flat-array 'f32 #(6 9) value-at))
                    (cons 'b (flat-array 'f32 #(9) value2-at))))
         (cases (list (map-case #(6 9) env (list shared `(mul (in 1) ,shared)) '(f32 f32))
                      (map-case #(6 9) env (list '(add (in 0) (in 1)) '(sub (in 0) (in 1))
                                                 '(mul (in 0) (in 1)))
                                '(f32 f64 f32))
                      (map-case #(6 9) env (list '(in 0) '(exp (in 1)) `(tanh ,shared) '(negate (in 0)))
                                '(f32 f32 f64 f32)))))
    (let-values (((results notes) (check-cases cases)))
      (for-each (lambda (name r) (test name '(segop/codegen-map . #t) (values r)))
                '("two outputs sharing a value" "three outputs of two dtypes" "four outputs")
                results)
      (test "every kernel with several outputs is made" 3 (codegen-note notes 'kernels)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5 - Mixed and integer dtypes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (int-at i) (- (* 37 (modulo (* i 7919) 101)) 1800))

(define (mixed-cases)
  (list
   (map-case #(17) (list (cons 'a (flat-array 'f32 #(17) value-at)) (cons 'b (flat-array 'f64 #(17) value2-at)))
             '((add (in 0) (in 1))) '(f64) #t)
   (map-case #(17) (list (cons 'a (flat-array 'f64 #(17) value-at)))
             '((sigmoid (in 0))) '(f32) #t)
   (map-case #(17) (list (cons 'a (flat-array 's32 #(17) int-at)) (cons 'b (flat-array 's32 #(17) (lambda (i) (- 5 i)))))
             '((sub (mul (in 0) (in 1)) (in 1))) '(s32) #t)
   (map-case #(17) (list (cons 'a (flat-array 's64 #(17) int-at)) (cons 'b (flat-array 'f32 #(17) value-at)))
             '((mul (in 0) (in 1))) '(f64) #t)
   (map-case #(17) (list (cons 'a (flat-array 's32 #(17) int-at)))
             '((negate (abs (in 0)))) '(s64) #t)
   (map-case #(17) (list (cons 'a (flat-array 's32 #(17) int-at)))
             '((abs (in 0))) '(u32) #t)
   (map-case #(17) (list (cons 'a (flat-array 'f64 #(17) (lambda (i) (* 1.7 (- i 8))))))
             '((cast s32 (in 0))) '(f32) #t)
   (map-case #(17) (list (cons 'a (flat-array 's64 #(17) (lambda (i) (* i 125000000)))))
             '((cast u32 (add (in 0) (in 0)))) '(s64) #t)
   (map-case #(17) (list (cons 'a (flat-array 's32 #(17) int-at)) (cons 'b (flat-array 'f64 #(17) value-at)))
             '((gt (in 0) (in 1))) '(f32) #t)
   (map-case #(17) (list (cons 'a (flat-array 's32 #(17) int-at)) (cons 'b (flat-array 'f32 #(17) value-at)))
             '((relu (add (in 0) (in 1)))) '(f32) #t)))

(test-group "mixed and integer dtypes"
  (let-values (((results notes) (check-cases (mixed-cases))))
    (for-each (lambda (name r) (test name '(segop/codegen-map . #t) (values r)))
              '("f32 + f64 into f64" "f64 sigmoid into f32" "s32 arithmetic into s32"
                "s64 times f32 into f64" "negated s32 abs into s64" "s32 abs into u32"
                "f64 cast to s32 into f32" "an s64 sum cast to u32 into s64" "s32 > f64" "relu of s32 + f32")
              results)
    (test "every mixed kernel is made" (length results) (codegen-note notes 'kernels)))
  ;; Values beyond 64 bits and exact comparisons beyond 2^53.
  (let* ((big (lambda (i) (+ (expt 2 62) (* i 1000003))))
         (cases (list
                 (map-case #(9) (list (cons 'a (flat-array 's64 #(9) big)) (cons 'b (flat-array 's64 #(9) big)))
                           '((mul (in 0) (in 1))) '(f64) #t)
                 (map-case #(9) (list (cons 'a (flat-array 's64 #(9) big)) (cons 'b (flat-array 's64 #(9) big)))
                           '((sub (add (in 0) (in 1)) (in 1))) '(s64) #t)
                 (map-case #(5) (list (cons 'a (flat-array 's64 #(5) (lambda (i) (+ (expt 2 53) (- i 2)))))
                                      (cons 'b (flat-array 'f64 #(5) (lambda (i) (exact->inexact (expt 2 53))))))
                           '((ge (in 0) (in 1))) '(f64) #t))))
    (let-values (((results notes) (check-cases cases)))
      (for-each (lambda (name r) (test name '(segop/codegen-map . #t) (values r)))
                '("an s64 product beyond 64 bits runs on the host"
                  "an s64 sum beyond 64 bits runs on the host"
                  "s64 against f64 near 2^53 compares exactly")
                results)))
  ;; A value that does not fit the output dtype is an error, as on the host.
  (let ((c (map-case #(3) (list (cons 'a (flat-array 's64 #(3) (lambda (i) (* i (expt 2 40))))))
                     '((add (in 0) (in 0))) '(s32) #t))
        (d (map-case #(3) (list (cons 'a (flat-array 'f64 #(3) (lambda (i) 1.0)))
                                (cons 'b (flat-array 's32 #(3) (lambda (i) (- i 1)))))
                     '((div (in 0) (in 1))) '(f64) #t)))
    (let-values (((sels notes) (cg-select-all (list (tcase-sg c) (tcase-sg d)))))
      (test "an s32 output out of range is generated" 'segop/codegen-map (car (car sels)))
      (test-error "an s32 output out of range is an error" (run-executor (cdr (car sels)) c))
      (test-error "a flonum divided by the integer 0 is an error" (run-executor (cdr (cadr sels)) d))))
  (let ((c (map-case #(3) (list (cons 'a (flat-array 's32 #(3) int-at)) (cons 'b (flat-array 's32 #(3) int-at)))
                     '((div (in 0) (in 1))) '(f64) #t)))
    (test-assert "a quotient of two integers is left to the host"
      (not (eq? 'segop/codegen-map
                (let ((ctx (make-segop-plan-context)))
                  (parameterize ((segop-plan-context ctx) (segop-backends (cg-backends)) (crunch-codegen #t))
                    (car (select-segop-executor (tcase-sg c))))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6 - Thread counts
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "thread counts"
  (let* ((env (list (cons 'a (flat-array 'f32 #(13 77) value-at))
                    (cons 'b (array 'f32 #(13 77) #(1 13) 0 1001 value2-at))))
         (c (map-case #(13 77) env '((mul (in 0) (cast f32 (tanh (in 1))))) '(f32))))
    (let-values (((sels notes) (cg-select-all (list (tcase-sg c)))))
      (let ((ref (run-reference c)))
        (for-each
         (lambda (t)
           (test-assert (sprintf "~a threads give the interpreter's bits" t)
             (parameterize ((crunch-codegen-threads t) (crunch-thread-min-chunk 1))
               (every same-bits? (run-executor (cdr (car sels)) c) ref))))
         '(1 2 7))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6b - seg-reds
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (exprs-map . es) (make-index-map (list->vector es) #f 0.0 #f))

;; A seg-red case over space, whose last nred dimensions are reduced.
;; env is a list of (slot array . index-map); results a list of
;; (dtype shape map body combine finalize), each written to a row-major
;; output of that shape through map; map-outs a list of (dtype body),
;; written over the whole space in row-major order.
(define (red-case space nred env results map-outs)
  (let* ((ins (map (lambda (e) (operand (car e) (cadr e) (cddr e))) env))
         (res (map (lambda (r k)
                     (make-red-result
                      (make-segop-operand (string->symbol (sprintf "res~a" k)) (car r) (cadr r)
                                          (compute-strides (cadr r)) 0 (caddr r))
                      (list-ref r 3) (list-ref r 4) (list-ref r 5)))
                   results (iota (length results))))
         (mos (map (lambda (mo k)
                     (cons (out-operand (string->symbol (sprintf "mo~a" k)) (car mo) space) (cadr mo)))
                   map-outs (iota (length map-outs)))))
    (make-tcase (make-seg-red space nred ins res mos)
                (map (lambda (e) (cons (car e) (cadr e))) env)
                (append (map (lambda (r) (shape-size (cadr r))) results)
                        (map (lambda (mo) (shape-size space)) map-outs)))))

(define (add-spec dt) (make-combine-spec 'add 0.0 dt))
(define (max-spec dt) (make-combine-spec 'max -inf.0 dt))
(define (min-spec dt) (make-combine-spec 'min +inf.0 dt))

;; Results compared with the host executor (the strided reduction loop)
;; as well as with the interpreter.
(define (check-red-cases cases)
  (let-values (((sels notes) (cg-select-all (map tcase-sg cases))))
    (let ((hosts (parameterize ((segop-backends
                                 (filter (lambda (b) (eq? (segop-backend-name b) 'scheme)) (segop-backends))))
                   (map (lambda (c) (select-segop-executor (tcase-sg c))) cases))))
      (values (map (lambda (sel host c)
                     (let ((got (run-executor (cdr sel) c)))
                       (list (car sel)
                             (every same-bits? got (run-reference c))
                             (every same-bits? got (run-executor (cdr host) c)))))
                   sels hosts cases)
              notes))))

(define (red-cases)
  (let* ((x  (lambda (dt) (flat-array dt #(7 5) value-at)))   ; rows x cols
         (y  (lambda (dt) (flat-array dt #(7 5) value2-at)))
         (z3 (flat-array 'f64 #(3 4 5) (lambda (i) (* 0.37 (- i 30)))))
         (col (exprs-map '(dim 1) '(dim 0)))                   ; space (cols rows)
         (row (exprs-map '(dim 0) '(dim 1))))                  ; space (rows cols)
    (list
     ;; axis-0 sum of g * relu'(z), f32 accumulator, with the product stored
     (red-case #(5 7) 1 (list (cons* 'a (x 'f32) col) (cons* 'b (y 'f32) col))
               (list (list 'f32 #(5) (exprs-map '(dim 0)) '(mul (in 0) (cast f32 (relu-deriv (in 1))))
                           (add-spec 'f32) #f))
               (list (list 'f32 '(mul (in 0) (cast f32 (relu-deriv (in 1)))))))
     ;; axis-1 sum, f64 accumulator, mean finalize, f32 result
     (red-case #(7 5) 1 (list (cons* 'a (x 'f32) row))
               (list (list 'f32 #(7 1) (exprs-map '(dim 0) 0) '(exp (in 0)) (add-spec 'f64)
                           '(div (reduced) 5.0)))
               '())
     ;; max and min over rows with NaN and infinities, f32
     (red-case #(7 5) 1 (list (cons* 'a (x 'f32) row))
               (list (list 'f32 #(7) (exprs-map '(dim 0)) '(in 0) (max-spec 'f32) #f)
                     (list 'f32 #(7) (exprs-map '(dim 0)) '(in 0) (min-spec 'f32) #f))
               '())
     ;; a product, f64
     (red-case #(5 7) 1 (list (cons* 'a (flat-array 'f64 #(7 5) (lambda (i) (+ 0.9 (* 0.01 i)))) col))
               (list (list 'f64 #(5) (exprs-map '(dim 0)) '(add (in 0) 0.25) (make-combine-spec 'mul 1.0 'f64) #f))
               '())
     ;; two reduced dimensions of a rank-3 array, keepdims, f64
     (red-case #(4 3 5) 2 (list (cons* 'a z3 (exprs-map '(dim 1) '(dim 0) '(dim 2))))
               (list (list 'f64 #(1 4 1) (exprs-map 0 '(dim 0) 0) '(tanh (in 0)) (add-spec 'f64) #f))
               '())
     ;; a full reduction to rank 0
     (red-case #(7 5) 2 (list (cons* 'a (x 'f64) row))
               (list (list 'f64 #() (exprs-map) '(mul (in 0) (in 0)) (add-spec 'f64) #f))
               '())
     ;; the blocked order with a block of 3
     (red-case #(5 7) 1 (list (cons* 'a (x 'f32) col))
               (list (list 'f32 #(5) (exprs-map '(dim 0)) '(in 0)
                           (make-combine-spec 'add 0.0 'f32 'blocked 3) #f))
               '())
     ;; the unspecified order, taken as sequential
     (red-case #(5 7) 1 (list (cons* 'a (x 'f32) col))
               (list (list 'f32 #(5) (exprs-map '(dim 0)) '(in 0)
                           (make-combine-spec 'add 0.0 'f64 'unspecified) #f))
               '())
     ;; a finalize reading an input at the kept point
     (red-case #(7 5) 1 (list (cons* 'a (x 'f32) row) (cons* 'b (flat-array 'f32 #(7) value2-at) (exprs-map '(dim 0))))
               (list (list 'f32 #(7) (exprs-map '(dim 0)) '(sigmoid (in 0)) (add-spec 'f32)
                           '(add (cast f32 (reduced)) (in 1))))
               '())
     ;; an s32 sum into s32, an s64 accumulator
     (red-case #(5 7) 1 (list (cons* 'a (flat-array 's32 #(7 5) int-at) col))
               (list (list 's32 #(5) (exprs-map '(dim 0)) '(mul (in 0) 3) (make-combine-spec 'add 0 's64) #f))
               '())
     ;; max of s32 values against f32 values, f64 accumulator
     (red-case #(7 5) 1 (list (cons* 'a (flat-array 's32 #(7 5) int-at) row)
                              (cons* 'b (y 'f32) row))
               (list (list 'f64 #(7) (exprs-map '(dim 0)) '(add (in 0) (in 1)) (max-spec 'f64) #f))
               '())
     ;; an empty reduction gives the neutral element
     (red-case #(4 0) 1 (list (cons* 'a (flat-array 'f32 #(4 0) value-at) row))
               (list (list 'f32 #(4) (exprs-map '(dim 0)) '(in 0) (max-spec 'f32) #f))
               '())
     ;; the Scheme procedures max and min, over NaN, infinities and signed
     ;; zeros, f32 and f64
     (red-case #(7 5) 1 (list (cons* 'a (x 'f32) row))
               (list (list 'f32 #(7) (exprs-map '(dim 0)) '(negate (in 0))
                           (make-combine-spec (list 'opaque max) -inf.0 'f32) #f)
                     (list 'f64 #(7) (exprs-map '(dim 0)) '(negate (in 0))
                           (make-combine-spec (list 'opaque min) +inf.0 'f64) #f))
               '())
     ;; a mean followed by a Scheme + of an input, as SSA programs write it
     (red-case #(7 5) 1 (list (cons* 'a (x 'f64) row) (cons* 'b (flat-array 'f64 #(7) value2-at) (exprs-map '(dim 0))))
               (list (list 'f64 #(7) (exprs-map '(dim 0)) '(sigmoid (in 0)) (add-spec 'f64)
                           `(opaque ,+ (/ (reduced) 5) (in 1))))
               (list (list 'f64 '(sigmoid (in 0)))))
     ;; max of s32 values with an s32 accumulator
     (red-case #(7 5) 1 (list (cons* 'a (flat-array 's32 #(7 5) int-at) row))
               (list (list 's32 #(7) (exprs-map '(dim 0)) '(mul (in 0) -2)
                           (make-combine-spec (list 'opaque max) (- (expt 2 31)) 's32) #f))
               '()))))

(test-group "seg-reds"
  (let-values (((results notes) (check-red-cases (red-cases))))
    (for-each (lambda (name r) (test name '(segop/codegen-red #t #t) (values r)))
              '("axis-0 sum with a stored product" "axis-1 sum with a mean"
                "max and min with NaN" "a product" "two reduced dimensions, keepdims"
                "a full reduction" "the blocked order" "the unspecified order"
                "a finalize reading an input" "an s32 sum" "max of s32 + f32"
                "an empty reduction" "Scheme max and min" "a mean then a Scheme +"
                "Scheme max of s32")
              results)
    (test "every reduction kernel is made" (length results) (codegen-note notes 'kernels)))
  ;; An accumulator that leaves its dtype is computed by the host, whose
  ;; rounding then decides the value.
  (let ((c (red-case #(1 4) 1 (list (cons* 'a (flat-array 's64 #(4) (lambda (i) (expt 2 30)))
                                           (exprs-map '(dim 1))))
                     (list (list 's32 #(1) (exprs-map '(dim 0)) '(in 0) (make-combine-spec 'add 0 's32) #f))
                     '())))
    (let-values (((results notes) (check-red-cases (list c))))
      (test "an s32 accumulator out of range gives the host's value" '((segop/codegen-red #t #t))
        (values results))))
  ;; Thread counts split the kept points only.
  (let* ((c (red-case #(97 13) 1 (list (cons* 'a (flat-array 'f32 #(97 13) value-at) (exprs-map '(dim 0) '(dim 1))))
                      (list (list 'f32 #(97) (exprs-map '(dim 0)) '(sigmoid (in 0)) (add-spec 'f32) #f))
                      '())))
    (let-values (((sels notes) (cg-select-all (list (tcase-sg c)))))
      (let ((ref (run-reference c)))
        (for-each
         (lambda (t)
           (test-assert (sprintf "a reduction on ~a threads gives the interpreter's bits" t)
             (parameterize ((crunch-codegen-threads t) (crunch-thread-min-chunk 1))
               (every same-bits? (run-executor (cdr (car sels)) c) ref))))
         '(1 2 7)))))
  ;; Standard reductions keep their own executors.
  (let ((c (red-case #(5 7) 1 (list (cons* 'a (flat-array 'f64 #(7 5) value-at) (exprs-map '(dim 1) '(dim 0))))
                     (list (list 'f64 #(5) (exprs-map '(dim 0)) '(in 0) (make-combine-spec 'add 0.0 'f64) #f))
                     '())))
    (test-assert "a seg-red the host runs with another executor is left to it"
      (let ((ctx (make-segop-plan-context)))
        (parameterize ((segop-plan-context ctx) (segop-backends (cg-backends)) (crunch-codegen #t))
          (let ((sel (select-segop-executor (tcase-sg c))))
            (or (eq? (car sel) 'segop/codegen-red)
                (not (memq (car sel) '(segop/strided-red segop/interpreted))))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7 - The cache and compile failures
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "cache and failures"
  (let* ((dir (temp-dir))
         (mk (lambda () (map-case #(10) (list (cons 'a (flat-array 'f64 #(10) value-at)))
                                  '((add (cast f32 (exp (in 0))) 0.5)) '(f64)))))
    (parameterize ((crunch-codegen-cache-directory dir))
      (crunch-codegen-forget-kernels!)
      (let-values (((results notes) (check-cases (list (mk)))))
        (test "a new kernel is compiled" 1 (codegen-note notes 'compiled))
        (test "the compiled kernel is equal to the interpreter" '((segop/codegen-map . #t)) (values results)))
      (let-values (((results notes) (check-cases (list (mk)))))
        (test "a kernel already loaded is reused" '(0 0 1)
          (list (codegen-note notes 'compiled) (codegen-note notes 'cached) (codegen-note notes 'kernels))))
      (crunch-codegen-forget-kernels!)
      (let-values (((results notes) (check-cases (list (mk)))))
        (test "a kernel compiled before is loaded from the cache" '(0 1)
          (list (codegen-note notes 'compiled) (codegen-note notes 'cached)))
        (test "the cached kernel is equal to the interpreter" '((segop/codegen-map . #t)) (values results)))
      (for-each
       (lambda (csc what)
         (crunch-codegen-forget-kernels!)
         (parameterize ((crunch-codegen-csc csc)
                        (crunch-codegen-cache-directory (temp-dir))
                        (current-error-port (open-output-string)))
           (let-values (((results notes) (check-cases (list (mk)))))
             (test (string-append what ": the kernel is reported as failed") 1 (codegen-note notes 'failed))
             (test (string-append what ": the host computes the segop") '((segop/codegen-map . #t)) (values results)))))
       '("/bin/false" "/nonexistent/csc")
       '("a failing compiler" "a missing compiler"))
      (crunch-codegen-forget-kernels!))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 8 - Native queries and SSA programs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "native queries"
  (let* ((tin  (lambda (dt) (make-template-input dt (identity-index-map 2) #t)))
         (tpl  (lambda (dts body out-dt)
                 (make-segop-template #(4 5) (map tin dts) (list (make-template-output out-dt body))))))
    (parameterize ((crunch-codegen #t))
      (test-assert "a translated body is native"
        (crunch-codegen-native? (tpl '(f32 f32) '(mul (in 0) (cast f32 (exp (in 1)))) 'f32)))
      (test-assert "mixed dtypes are native"
        (crunch-codegen-native? (tpl '(s32 f64) '(add (in 0) (in 1)) 'f32)))
      (test "an integer quotient is not native" #f
        (crunch-codegen-native? (tpl '(s32 s32) '(div (in 0) (in 1)) 'f64)))
      (test "a u64 operand is not native" #f
        (crunch-codegen-native? (tpl '(u64) '(in 0) 'u64)))
      (test-assert "a reduction template is native"
        (crunch-codegen-native?
         (make-segop-template #(4 5) (list (tin 'f32))
                              (list (make-template-output 'f32 '(in 0) (make-combine-spec 'add 0.0 'f32) #f
                                                          (identity-index-map 1)))
                              1))))
    (parameterize ((crunch-codegen #f))
      (test "with crunch-codegen false nothing is native" #f
        (crunch-codegen-native? (tpl '(f32) '(exp (in 0)) 'f32))))))

;; A small network whose element-wise steps all fuse under the policy all:
;; the replays with generated kernels equal those with fusion off.
(define (ssa-net dtype)
  (let* ((v (lambda (n f) (map f (iota n))))
         (x (am:make-var (morph-from-list (v 24 (lambda (i) (- (* 0.31 i) 3.0))) #(4 6) dtype) #f))
         (W (am:make-var (morph-from-list (v 30 (lambda (i) (* 0.2 (sin (+ i 1.0))))) #(6 5) dtype) #t))
         (b (am:make-var (morph-from-list (v 5 (lambda (i) (- (* 0.1 i) 0.2))) #(5) dtype) #t))
         (h (am:var-tanh (am:var+ (am:var-matmul x W) b)))
         (y (am:var-sigmoid (am:var* h (am:var-exp (am:var-negate h))))))
    (values (am:var-sum (am:var* y y)) (list W b))))

(define (ssa-replays dtype policy codegen?)
  (let-values (((loss params) (ssa-net dtype)))
    (parameterize ((segop-fusion policy) (crunch-codegen codegen?)
                   (segop-backends (parameterize ((segop-backends (segop-backends)))
                                     (register-crunch-codegen-backend!)
                                     (segop-backends))))
      (let* ((fwd (morphism-to-ssa loss))
             (pv  (map (lambda (p) (ssa-constant-id fwd (am:var-value p))) params))
             (joint (ssa-vjp fwd pv (ssa-loss-binding-val fwd)))
             (ctx (make-morphism-context))
             ;; the elements of each result array, which may sit at the
             ;; start of a larger pool buffer
             (vals (lambda (rs)
                     (map (lambda (r)
                            (let loop ((xs (vec->list (operand-data r)))
                                       (n (shape-size (get-morphism-shape r))))
                              (if (= n 0) '() (cons (car xs) (loop (cdr xs) (- n 1))))))
                          rs))))
        (ssa-realize/ctx ctx joint)
        (finalize-context! ctx)
        (reset-context! ctx)
        (let ((r1 (vals (ssa-realize/ctx ctx joint))))
          (reset-context! ctx)
          (let ((r2 (vals (ssa-realize/ctx ctx joint))))
            (list r1 r2 (cdr (assq 'counts (replay-plan-stats joint))))))))))

(test-group "SSA programs"
  (for-each
   (lambda (dt)
     (let ((off (ssa-replays dt #f #f))
           (all (ssa-replays dt 'all #t)))
       (test-assert (sprintf "~a: generated kernels run under the policy all" dt)
         (assq 'segop/codegen-map (caddr all)))
       (test-assert (sprintf "~a: the replays equal those with fusion off" dt)
         (every (lambda (x y) (every same-element? x y))
                (append (car off) (cadr off)) (append (car all) (cadr all))))))
   '(f32 f64)))

(for-each (lambda (d) (delete-directory d #t)) temp-dirs)

(test-exit)
