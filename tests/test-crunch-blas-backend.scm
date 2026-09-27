;;; tests/test-crunch-blas-backend.scm
;;; Tests for the crunch-compiled GEMM, GEMV, DOT and AXPY kernels
;;; (crunch-blas-backend.scm) and the blas-backend record assembled by
;;; make-crunch-blas-backend (crunch-conv-backend.scm).
;;;
;;; Mirrors the structure of array-morphisms/tests/test-microblas.scm: the
;;; kernels are driven directly against an independent naive-Scheme
;;; reference, and additionally compared with the microBLAS backend.
;;;
;;; Organisation:
;;;   Group 1 - Backend construction
;;;   Group 2 - GEMM correctness (plain, contiguous)
;;;   Group 3 - GEMM-strided correctness (transposed / non-default lda)
;;;   Group 4 - GEMV correctness
;;;   Group 5 - DOT / AXPY correctness
;;;   Group 6 - BLAS edge cases: alpha/beta handling, precision, arguments
;;;   Group 7 - Agreement with the microBLAS backend

(import scheme (chicken base)
         test
         (only srfi-1 iota every)
         srfi-4
         array-morphisms-blas-exec
         array-morphisms-realization
         array-morphisms-micro-blas-backend
         array-morphisms-crunch-blas-backend
         array-morphisms-crunch-conv-backend
         (only array-morphisms-crunch-threads crunch-thread-count)
         (only (chicken format) sprintf))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (approx= a b #!optional (tol 1e-6))
  (< (abs (- a b)) tol))

(define (rel-close? a b tol)
  (<= (abs (- a b)) (* tol (max 1.0 (abs a) (abs b)))))

(define (gen-a i) (sin (* (+ i 1) 0.123)))
(define (gen-b i) (cos (* (+ i 1) 0.071)))
(define (gen-c i) (- (* 0.5 (sin (* (+ i 2) 0.37))) 0.1))

(define (make-f32 n f)
  (let ((v (make-f32vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f32vector-set! v i (exact->inexact (f i))))))

(define (make-f64 n f)
  (let ((v (make-f64vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f64vector-set! v i (exact->inexact (f i))))))

;; Elements of a SRFI-4 vector as a Scheme vector of flonums.
(define (->vec v)
  (list->vector (if (f32vector? v) (f32vector->list v) (f64vector->list v))))

;; Naive reference gemm on plain Scheme vectors: C := alpha*A*B + beta*C
(define (naive-gemm M N K alpha A B beta C)
  (let ((out (make-vector (* M N) 0.0)))
    (do ((i 0 (+ i 1))) ((= i M) out)
      (do ((j 0 (+ j 1))) ((= j N))
        (let ((sum 0.0))
          (do ((k 0 (+ k 1))) ((= k K))
            (set! sum (+ sum (* (vector-ref A (+ (* i K) k))
                                (vector-ref B (+ (* k N) j))))))
          (vector-set! out (+ (* i N) j)
                       (+ (* alpha sum) (* beta (vector-ref C (+ (* i N) j))))))))))

;; The logical dim0 x dim1 matrix of a physically stored operand: when trans
;; is 'trans the storage is dim1 x dim0 with row stride ld, otherwise it is
;; dim0 x dim1 with row stride ld.
(define (logical-ref phys dim0 dim1 ld trans)
  (let ((out (make-vector (* dim0 dim1) 0.0)))
    (do ((i 0 (+ i 1))) ((= i dim0) out)
      (do ((j 0 (+ j 1))) ((= j dim1))
        (vector-set! out (+ (* i dim1) j)
                     (if (eq? trans 'trans)
                         (vector-ref phys (+ (* j ld) i))
                         (vector-ref phys (+ (* i ld) j))))))))

(define (vec-close? a b tol)
  (and (= (vector-length a) (vector-length b))
       (let loop ((i 0))
         (or (= i (vector-length a))
             (and (rel-close? (vector-ref a i) (vector-ref b i) tol)
                  (loop (+ i 1)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - Backend construction
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch BLAS - Backend construction"

  (test-assert "make-crunch-blas-backend returns a blas-backend record"
    (blas-backend? (make-crunch-blas-backend)))

  (test-assert "make-crunch-blas-backend names the backend 'crunch"
    (eq? 'crunch (blas-backend-name (make-crunch-blas-backend))))

  (test-assert "all 16 kernel slots are populated with procedures"
    (let ((b (make-crunch-blas-backend)))
      (every procedure?
             (list (blas-backend-gemm-f64 b) (blas-backend-gemm-f32 b)
                   (blas-backend-gemm-strided-f64 b) (blas-backend-gemm-strided-f32 b)
                   (blas-backend-gemv-f64 b) (blas-backend-gemv-f32 b)
                   (blas-backend-dot-f64 b) (blas-backend-dot-f32 b)
                   (blas-backend-axpy-f64 b) (blas-backend-axpy-f32 b)
                   (blas-backend-conv-fwd-im2col-f32 b)
                   (blas-backend-conv-bwd-data-im2col-f32 b)
                   (blas-backend-conv-bwd-weights-im2col-f32 b)
                   (blas-backend-conv-fwd-nhwc-im2col-f32 b)
                   (blas-backend-conv-bwd-data-nhwc-im2col-f32 b)
                   (blas-backend-conv-bwd-weights-nhwc-im2col-f32 b)))))

  (test-assert "slots hold the kernels in blas-backend field order"
    (let ((b (make-crunch-blas-backend)))
      (and (eq? (blas-backend-gemm-f64 b) crunch-dgemm)
           (eq? (blas-backend-gemm-f32 b) crunch-sgemm)
           (eq? (blas-backend-gemm-strided-f64 b) crunch-dgemm-strided)
           (eq? (blas-backend-gemm-strided-f32 b) crunch-sgemm-strided)
           (eq? (blas-backend-gemv-f64 b) crunch-dgemv)
           (eq? (blas-backend-gemv-f32 b) crunch-sgemv)
           (eq? (blas-backend-dot-f64 b) crunch-ddot)
           (eq? (blas-backend-dot-f32 b) crunch-sdot)
           (eq? (blas-backend-axpy-f64 b) crunch-daxpy)
           (eq? (blas-backend-axpy-f32 b) crunch-saxpy)
           (eq? (blas-backend-conv-fwd-im2col-f32 b) crunch-conv-fwd-im2col-f32)
           (eq? (blas-backend-conv-bwd-data-im2col-f32 b) crunch-conv-bwd-data-im2col-f32)
           (eq? (blas-backend-conv-bwd-weights-im2col-f32 b) crunch-conv-bwd-weights-im2col-f32)
           (eq? (blas-backend-conv-fwd-nhwc-im2col-f32 b) crunch-conv-fwd-nhwc-im2col-f32)
           (eq? (blas-backend-conv-bwd-data-nhwc-im2col-f32 b)
                crunch-conv-bwd-data-nhwc-im2col-f32)
           (eq? (blas-backend-conv-bwd-weights-nhwc-im2col-f32 b)
                crunch-conv-bwd-weights-nhwc-im2col-f32))))

  (test-assert "register-blas-backend! + blas-available? round-trip"
    (let ((saved *active-backend*))
      (register-blas-backend! (make-crunch-blas-backend))
      (let ((r (and (blas-available?)
                    (eq? 'crunch (blas-backend-name (active-blas-backend))))))
        (set! *active-backend* saved)
        r))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - GEMM correctness (plain, contiguous)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch BLAS - GEMM correctness"

  (for-each
   (lambda (dims)
     (let* ((M (car dims)) (N (cadr dims)) (K (caddr dims))
            (label (string-append "M=" (number->string M) " N=" (number->string N)
                                  " K=" (number->string K)))
            (Av (->vec (make-f64 (* M K) gen-a)))
            (Bv (->vec (make-f64 (* K N) gen-b)))
            (Cv (->vec (make-f64 (* M N) gen-c))))
       (test-assert (string-append "gemm-f64 alpha=1 beta=0 " label)
         (let ((C (make-f64 (* M N) gen-c)))
           (crunch-dgemm M N K 1.0 (make-f64 (* M K) gen-a) (make-f64 (* K N) gen-b) 0.0 C)
           (vec-close? (->vec C) (naive-gemm M N K 1.0 Av Bv 0.0 Cv) 1e-12)))
       (test-assert (string-append "gemm-f64 alpha=0.7 beta=-1.3 " label)
         (let ((C (make-f64 (* M N) gen-c)))
           (crunch-dgemm M N K 0.7 (make-f64 (* M K) gen-a) (make-f64 (* K N) gen-b) -1.3 C)
           (vec-close? (->vec C) (naive-gemm M N K 0.7 Av Bv -1.3 Cv) 1e-12)))
       (test-assert (string-append "gemm-f32 alpha=0.7 beta=0.5 " label)
         (let* ((A (make-f32 (* M K) gen-a)) (B (make-f32 (* K N) gen-b))
                (C (make-f32 (* M N) gen-c))
                (ref (naive-gemm M N K 0.7 (->vec A) (->vec B) 0.5 (->vec C))))
           (crunch-sgemm M N K 0.7 A B 0.5 C)
           (vec-close? (->vec C) ref 1e-5)))))
   '((1 1 1) (2 2 2) (5 13 7) (13 5 7) (63 64 65) (128 128 128) (1 40 3) (40 1 3)))

  (test-assert "gemm-f32 zero M is a no-op"
    (let ((C (make-f32vector 0 0.0)))
      (crunch-sgemm 0 3 3 1.0 (make-f32vector 0 0.0) (make-f32vector 9 1.0) 0.0 C)
      #t))

  (test-assert "gemm-f64 zero N is a no-op"
    (let ((C (make-f64vector 0 0.0)))
      (crunch-dgemm 3 0 3 1.0 (make-f64vector 9 1.0) (make-f64vector 0 0.0) 0.0 C)
      #t))

  (test "gemm-f32 zero K with beta=0 zeroes the output"
    '(0.0 0.0 0.0 0.0)
    (let ((C (make-f32vector 4 9.0)))
      (crunch-sgemm 2 2 0 1.0 (make-f32vector 0 0.0) (make-f32vector 0 0.0) 0.0 C)
      (f32vector->list C)))

  (test "gemm-f64 zero K with beta=2 scales the output"
    '(2.0 4.0 6.0 8.0)
    (let ((C (f64vector 1.0 2.0 3.0 4.0)))
      (crunch-dgemm 2 2 0 1.0 (make-f64vector 0 0.0) (make-f64vector 0 0.0) 2.0 C)
      (f64vector->list C))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - GEMM-strided correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Run one strided case: logical A is M x K, logical B is K x N, each stored
;; per its trans flag with row stride = natural width + pad.
(define (strided-case gemm make-vec tol M N K transa transb pad alpha beta)
  (let* ((lda  (+ pad (if (eq? transa 'trans) M K)))
         (ldb  (+ pad (if (eq? transb 'trans) K N)))
         (rowsA (if (eq? transa 'trans) K M))
         (rowsB (if (eq? transb 'trans) N K))
         (A (make-vec (* rowsA lda) gen-a))
         (B (make-vec (* rowsB ldb) gen-b))
         (C (make-vec (* M N) gen-c))
         (ref (naive-gemm M N K alpha
                          (logical-ref (->vec A) M K lda transa)
                          (logical-ref (->vec B) K N ldb transb)
                          beta (->vec C))))
    (gemm M N K alpha A lda transa B ldb transb beta C)
    (vec-close? (->vec C) ref tol)))

(test-group "crunch BLAS - GEMM-strided correctness"
  (for-each
   (lambda (ta)
     (for-each
      (lambda (tb)
        (for-each
         (lambda (pad)
           (let ((label (string-append (symbol->string ta) "/" (symbol->string tb)
                                       " pad=" (number->string pad))))
             (test-assert (string-append "gemm-strided-f64 " label)
               (strided-case crunch-dgemm-strided make-f64 1e-12 7 5 9 ta tb pad 0.9 0.3))
             (test-assert (string-append "gemm-strided-f32 " label)
               (strided-case crunch-sgemm-strided make-f32 1e-5 7 5 9 ta tb pad 0.9 0.3))))
         '(0 3)))
      '(no-trans trans)))
   '(no-trans trans))

  (test-assert "gemm-strided-f64 large transposed case (M=64 N=33 K=70, T/T, pad 2)"
    (strided-case crunch-dgemm-strided make-f64 1e-12 64 33 70 'trans 'trans 2 1.0 0.0))

  (test-assert "gemm-strided with default lda equals plain gemm"
    (let ((A (make-f32 (* 6 4) gen-a)) (B (make-f32 (* 4 5) gen-b))
          (C1 (make-f32 30 gen-c)) (C2 (make-f32 30 gen-c)))
      (crunch-sgemm 6 5 4 1.0 A B 0.0 C1)
      (crunch-sgemm-strided 6 5 4 1.0 A 4 'no-trans B 5 'no-trans 0.0 C2)
      (equal? (f32vector->list C1) (f32vector->list C2)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - GEMV correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (naive-gemv M N alpha A x beta y)
  (let ((out (make-vector M 0.0)))
    (do ((i 0 (+ i 1))) ((= i M) out)
      (let ((sum 0.0))
        (do ((j 0 (+ j 1))) ((= j N))
          (set! sum (+ sum (* (vector-ref A (+ (* i N) j)) (vector-ref x j)))))
        (vector-set! out i (+ (* alpha sum) (* beta (vector-ref y i))))))))

(test-group "crunch BLAS - GEMV correctness"
  (for-each
   (lambda (dims)
     (let* ((M (car dims)) (N (cadr dims))
            (label (string-append "M=" (number->string M) " N=" (number->string N))))
       (test-assert (string-append "gemv-f64 " label)
         (let* ((A (make-f64 (* M N) gen-a)) (x (make-f64 N gen-b)) (y (make-f64 M gen-c))
                (ref (naive-gemv M N 1.5 (->vec A) (->vec x) -0.5 (->vec y))))
           (crunch-dgemv M N 1.5 A x -0.5 y)
           (vec-close? (->vec y) ref 1e-12)))
       (test-assert (string-append "gemv-f32 " label)
         (let* ((A (make-f32 (* M N) gen-a)) (x (make-f32 N gen-b)) (y (make-f32 M gen-c))
                (ref (naive-gemv M N 1.5 (->vec A) (->vec x) -0.5 (->vec y))))
           (crunch-sgemv M N 1.5 A x -0.5 y)
           (vec-close? (->vec y) ref 1e-5)))))
   '((1 1) (3 7) (7 3) (64 65)))

  (test "gemv-f64 N=0 with beta=0 zeroes y"
    '(0.0 0.0)
    (let ((y (f64vector 5.0 6.0)))
      (crunch-dgemv 2 0 1.0 (make-f64vector 0 0.0) (make-f64vector 0 0.0) 0.0 y)
      (f64vector->list y))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5 - DOT / AXPY correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch BLAS - DOT / AXPY correctness"

  (test "dot-f32 basic" 32.0
    (crunch-sdot 3 (f32vector 1.0 2.0 3.0) (f32vector 4.0 5.0 6.0)))

  (test "dot-f64 basic" 32.0
    (crunch-ddot 3 (f64vector 1.0 2.0 3.0) (f64vector 4.0 5.0 6.0)))

  (test "dot of length 0 is 0.0" '(0.0 0.0)
    (list (crunch-sdot 0 (make-f32vector 0 0.0) (make-f32vector 0 0.0))
          (crunch-ddot 0 (make-f64vector 0 0.0) (make-f64vector 0 0.0))))

  (test-assert "dot uses only the first N elements"
    (= 1.0 (crunch-ddot 1 (f64vector 1.0 100.0) (f64vector 1.0 100.0))))

  (test-assert "dot-f64 large (N=1000) matches the naive sum"
    (let* ((x (make-f64 1000 gen-a)) (y (make-f64 1000 gen-b))
           (ref (let loop ((i 0) (s 0.0))
                  (if (= i 1000) s
                      (loop (+ i 1) (+ s (* (f64vector-ref x i) (f64vector-ref y i))))))))
      (= ref (crunch-ddot 1000 x y))))

  (test "axpy-f32 basic: y := 2*x + y" '(12.0 14.0 16.0)
    (let ((x (f32vector 1.0 2.0 3.0)) (y (f32vector 10.0 10.0 10.0)))
      (crunch-saxpy 3 2.0 x y)
      (f32vector->list y)))

  (test "axpy-f64 basic: y := -0.5*x + y" '(9.5 9.0 8.5)
    (let ((x (f64vector 1.0 2.0 3.0)) (y (f64vector 10.0 10.0 10.0)))
      (crunch-daxpy 3 -0.5 x y)
      (f64vector->list y)))

  (test "axpy alpha=0 leaves y unchanged" '(10.0 10.0 10.0)
    (let ((x (f32vector 1.0 2.0 3.0)) (y (f32vector 10.0 10.0 10.0)))
      (crunch-saxpy 3 0.0 x y)
      (f32vector->list y)))

  (test "axpy of length 0 is a no-op" '(10.0 10.0)
    (let ((y (f64vector 10.0 10.0)))
      (crunch-daxpy 0 5.0 (make-f64vector 0 0.0) y)
      (f64vector->list y)))

  (test "axpy with exact integer alpha" '(3.0 5.0)
    (let ((x (f64vector 1.0 2.0)) (y (f64vector 1.0 1.0)))
      (crunch-daxpy 2 2 x y)
      (f64vector->list y))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6 - BLAS edge cases
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch BLAS - alpha/beta handling, precision and arguments"

  (test "beta=0 ignores NaN already in C"
    '(19.0 22.0 43.0 50.0)
    (let ((C (f64vector +nan.0 +nan.0 +nan.0 +nan.0)))
      (crunch-dgemm 2 2 2 1.0 (f64vector 1.0 2.0 3.0 4.0) (f64vector 5.0 6.0 7.0 8.0) 0.0 C)
      (f64vector->list C)))

  (test "beta=0 ignores NaN in C for gemv"
    '(5.0)
    (let ((y (f32vector +nan.0)))
      (crunch-sgemv 1 2 1.0 (f32vector 1.0 2.0) (f32vector 1.0 2.0) 0.0 y)
      (f32vector->list y)))

  (test "alpha=0 ignores NaN in A and B"
    '(2.0 4.0)
    (let ((C (f64vector 1.0 2.0)))
      (crunch-dgemm 1 2 1 0.0 (f64vector +nan.0) (f64vector +nan.0 +nan.0) 2.0 C)
      (f64vector->list C)))

  (test-assert "f64 alpha and beta keep full double precision"
    ;; 0.1 is not representable in single precision; a single-float alpha
    ;; would change the result in the ninth significant digit.
    (let ((C (f64vector 1.0)))
      (crunch-dgemm 1 1 1 0.1 (f64vector 3.0) (f64vector 1.0) 0.1 C)
      (= (f64vector-ref C 0) (+ (* 0.1 3.0) (* 0.1 1.0)))))

  (test-assert "f64 dot result keeps full double precision"
    (let ((x (f64vector 0.1 0.2)) (y (f64vector 1.0 1.0)))
      (= (crunch-ddot 2 x y) (+ 0.1 0.2))))

  (test-assert "f32 gemm accumulates in double precision"
    ;; 1 + 2^-24 * 2^24 terms: a single-float accumulator would lose every
    ;; small term, a double one keeps them.
    (let* ((K 4096)
           (A (make-f32vector K 1.0))
           (B (make-f32vector K (exact->inexact (expt 2 -24))))
           (C (f32vector 0.0)))
      (f32vector-set! B 0 1.0)
      (crunch-sgemm 1 1 K 1.0 A B 0.0 C)
      (> (f32vector-ref C 0) 1.0)))

  (test-error "gemm: A too short is an error"
    (crunch-dgemm 2 2 2 1.0 (make-f64vector 3 1.0) (make-f64vector 4 1.0) 0.0 (make-f64vector 4 0.0)))
  (test-error "gemm: C too short is an error"
    (crunch-sgemm 2 2 2 1.0 (make-f32vector 4 1.0) (make-f32vector 4 1.0) 0.0 (make-f32vector 3 0.0)))
  (test-error "gemm-strided: lda too large for A is an error"
    (crunch-dgemm-strided 2 2 2 1.0 (make-f64vector 4 1.0) 3 'no-trans
                          (make-f64vector 4 1.0) 2 'no-trans 0.0 (make-f64vector 4 0.0)))
  (test-error "gemm: negative dimension is an error"
    (crunch-dgemm -1 2 2 1.0 (make-f64vector 4 1.0) (make-f64vector 4 1.0) 0.0 (make-f64vector 4 0.0)))
  (test-error "gemv: x too short is an error"
    (crunch-dgemv 2 3 1.0 (make-f64vector 6 1.0) (make-f64vector 2 1.0) 0.0 (make-f64vector 2 0.0)))
  (test-error "dot: y too short is an error"
    (crunch-sdot 3 (make-f32vector 3 1.0) (make-f32vector 2 1.0)))
  (test-error "axpy: y too short is an error"
    (crunch-daxpy 3 1.0 (make-f64vector 3 1.0) (make-f64vector 2 1.0))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7 - Agreement with the microBLAS backend
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch BLAS - agreement with microBLAS"
  (let ((mb (make-micro-blas-backend)))
    (for-each
     (lambda (ta)
       (for-each
        (lambda (tb)
          (test-assert (string-append "gemm-strided-f32 " (symbol->string ta) "/"
                                      (symbol->string tb) " agrees with microBLAS")
            (let* ((M 17) (N 11) (K 23)
                   (lda (+ 1 (if (eq? ta 'trans) M K)))
                   (ldb (+ 2 (if (eq? tb 'trans) K N)))
                   (A (make-f32 (* (if (eq? ta 'trans) K M) lda) gen-a))
                   (B (make-f32 (* (if (eq? tb 'trans) N K) ldb) gen-b))
                   (C1 (make-f32 (* M N) gen-c))
                   (C2 (make-f32 (* M N) gen-c)))
              (crunch-sgemm-strided M N K 1.0 A lda ta B ldb tb 0.0 C1)
              ((blas-backend-gemm-strided-f32 mb) M N K 1.0 A lda ta B ldb tb 0.0 C2)
              (vec-close? (->vec C1) (->vec C2) 1e-5))))
        '(no-trans trans)))
     '(no-trans trans))

    (test-assert "gemv-f64 agrees with microBLAS"
      (let* ((A (make-f64 (* 9 8) gen-a)) (x (make-f64 8 gen-b))
             (y1 (make-f64 9 gen-c)) (y2 (make-f64 9 gen-c)))
        (crunch-dgemv 9 8 0.5 A x 2.0 y1)
        ((blas-backend-gemv-f64 mb) 9 8 0.5 A x 2.0 y2)
        (vec-close? (->vec y1) (->vec y2) 1e-12)))

    (test-assert "dot-f32 agrees with microBLAS"
      (let ((x (make-f32 300 gen-a)) (y (make-f32 300 gen-b)))
        (rel-close? (crunch-sdot 300 x y) ((blas-backend-dot-f32 mb) 300 x y) 1e-5)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Exact results and thread-count independence
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Every element of C is a double-precision sum over k in increasing
;; order, exactly as in naive-gemm, so in a portable build the f64 kernel
;; must match the naive reference bit for bit, and the f32 kernel must
;; match it after rounding to single precision.  A build for the host CPU
;; may use fused multiply-add, which rounds differently, so there the
;; results are compared with a tight tolerance instead.  beta is non-zero
;; so that both formulas take the same form.
(define (exact-case gemm make-vec ->srfi4 M N K transa transb)
  (let* ((lda (if (eq? transa 'trans) M K))
         (ldb (if (eq? transb 'trans) K N))
         (A (make-vec (* M K) gen-a))
         (B (make-vec (* K N) gen-b))
         (C (make-vec (* M N) gen-c))
         (ref (->srfi4 (naive-gemm M N K 0.7
                                   (logical-ref (->vec A) M K lda transa)
                                   (logical-ref (->vec B) K N ldb transb)
                                   0.3 (->vec C)))))
    (gemm M N K 0.7 A lda transa B ldb transb 0.3 C)
    (if crunch-native-build?
        (vec-close? (->vec C) (->vec ref) (if (f32vector? C) 1e-6 1e-13))
        (equal? (->vec C) (->vec ref)))))

(define (vector->f32 v) (list->f32vector (vector->list v)))
(define (vector->f64 v) (list->f64vector (vector->list v)))

;; Result of one strided product computed with the given thread count.
(define (product-with-threads threads gemm make-vec M N K transa transb)
  (parameterize ((crunch-thread-count threads))
    (let* ((lda (if (eq? transa 'trans) M K))
           (ldb (if (eq? transb 'trans) K N))
           (C (make-vec (* M N) gen-c)))
      (gemm M N K 1.3 (make-vec (* M K) gen-a) lda transa
            (make-vec (* K N) gen-b) ldb transb 0.5 C)
      (->vec C))))

(test-group "crunch BLAS - exact results and thread-count independence"
  (for-each
   (lambda (dims)
     (let ((M (car dims)) (N (cadr dims)) (K (caddr dims)))
       (for-each
        (lambda (ta tb)
          (let ((label (sprintf "M=~A N=~A K=~A ~A/~A" M N K ta tb)))
            (test-assert (string-append "gemm-f64 matches naive reference " label)
              (exact-case crunch-dgemm-strided make-f64 vector->f64 M N K ta tb))
            (test-assert (string-append "gemm-f32 matches rounded naive reference " label)
              (exact-case crunch-sgemm-strided make-f32 vector->f32 M N K ta tb))
            (test-assert (string-append "gemm-f32 identical for 1, 2, 3 and 8 threads " label)
              (let ((one (product-with-threads 1 crunch-sgemm-strided make-f32 M N K ta tb)))
                (every (lambda (t)
                         (equal? one (product-with-threads t crunch-sgemm-strided make-f32 M N K ta tb)))
                       '(2 3 8))))
            (test-assert (string-append "gemm-f64 identical for 1, 2, 3 and 8 threads " label)
              (let ((one (product-with-threads 1 crunch-dgemm-strided make-f64 M N K ta tb)))
                (every (lambda (t)
                         (equal? one (product-with-threads t crunch-dgemm-strided make-f64 M N K ta tb)))
                       '(2 3 8))))))
        '(no-trans trans no-trans trans)
        '(no-trans no-trans trans trans))))
   ;; Shapes with fewer than eight columns, column counts that are not a
   ;; multiple of eight, single rows and columns, and products large
   ;; enough to be split among threads.
   '((1 1 1) (1 40 3) (40 1 3) (7 5 9) (33 17 12) (300 64 70) (1000 24 40) (37 129 256))))

(test-exit)
