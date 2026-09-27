;;; crunch-blas-backend.scm
;;; GEMM, GEMV, DOT and AXPY kernels compiled to C by crunch, with wrappers
;;; that follow the normalized kernel signatures of array-morphisms-blas-exec.
;;;
;;; Every kernel exists in an f32 and an f64 variant generated from one
;;; template by define-crunch-for-dtypes.  crunch computes in double
;;; precision, so the f32 variants read and write single floats but add up
;;; their products in double.
;;;
;;; Two limits of crunch's embedded foreign wrappers shape the interfaces:
;;;
;;;   * A `float' argument or result crosses the wrapper as a single float.
;;;     Scalars that must keep double precision (alpha, beta and dot
;;;     products) therefore travel in one-element f64vectors.
;;;
;;;   * crunch checks vector indices with C assertions, which stop the
;;;     whole process; the egg is compiled with NDEBUG, which removes them.
;;;     The Scheme wrappers check vector lengths before every call and raise
;;;     an ordinary error instead.
;;;
;;; GEMM operands are described as in the microBLAS shim
;;; (kernels/microblas_wrapper.c in array-morphisms): with lda the physical
;;; row stride of the stored matrix, a non-transposed A (stored M x K) has
;;; element strides (lda, 1) and a transposed A (stored K x M) has strides
;;; (1, lda); B is handled the same way.  An operand that is not already
;;; plain row-major is packed into a row-major copy, and the product is
;;; computed by a kernel for contiguous operands whose rows are shared
;;; among threads.  Each element of C is a sum over k in increasing order,
;;; so the result does not depend on the number of threads.
;;;
;;; As in reference BLAS, C is not read when beta is 0, and the values of A
;;; and B do not affect the result when alpha is 0, so NaN or uninitialised
;;; values there cannot leak into it.
;;;
;;; The complete blas-backend record, including the convolution hooks, is
;;; built by make-crunch-blas-backend in crunch-conv-backend.scm.

(module array-morphisms-crunch-blas-backend

  (;; Normalized kernels (see array-morphisms-blas-exec)
   crunch-dgemm          crunch-sgemm
   crunch-dgemm-strided  crunch-sgemm-strided
   crunch-dgemv          crunch-sgemv
   crunch-ddot           crunch-sdot
   crunch-daxpy          crunch-saxpy

   ;; #t when the module was compiled for the build machine's CPU
   crunch-native-build?)

  (import scheme (chicken base) (chicken foreign) (chicken number-vector) crunch)
  (import (only array-morphisms-crunch-threads
                crunch-dispatch4-f32 crunch-dispatch4-f64 crunch-thread-count))
  (import-for-syntax scheme (chicken base))

  (include "crunch-numvector-fix.scm")

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Kernel templates
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; (define-crunch-for-dtypes (name (arg ...) (argtype ...) rettype body ...) ...)
  ;;
  ;; Defines name-f32 and name-f64 as crunch procedures.  In argtypes and
  ;; body, @vec, @ref and @set! stand for f32vector, f32vector-ref and
  ;; f32vector-set! in the f32 variant, and for their f64 counterparts in
  ;; the f64 variant.
  (define-syntax define-crunch-for-dtypes
    (er-macro-transformer
     (lambda (form r c)
       (define (subst dtype x)
         (cond ((eq? x '@vec)  (if (eq? dtype 'f32) 'f32vector 'f64vector))
               ((eq? x '@ref)  (if (eq? dtype 'f32) 'f32vector-ref 'f64vector-ref))
               ((eq? x '@set!) (if (eq? dtype 'f32) 'f32vector-set! 'f64vector-set!))
               ((pair? x) (cons (subst dtype (car x)) (subst dtype (cdr x))))
               (else x)))
       (define (variant spec dtype)
         (let ((name (string->symbol
                      (string-append (symbol->string (car spec)) "-"
                                     (symbol->string dtype))))
               (args (cadr spec))
               (argtypes (caddr spec))
               (rettype (cadddr spec))
               (body (cddddr spec)))
           `(,(r 'crunch)
             (: (,name ,@(subst dtype argtypes)) ,rettype)
             (define (,name ,@args) ,@(subst dtype body)))))
       `(,(r 'begin)
         ,@(apply append
                  (map (lambda (spec) (list (variant spec 'f32) (variant spec 'f64)))
                       (cdr form)))))))

  (define-crunch-for-dtypes

    ;; y[i] := alpha * sum_j A[i*N + j] x[j] + beta * y[i]
    (%k-gemv (M N scal A x y)
             (integer integer f64vector @vec @vec @vec)
             void
      (let ((alpha (f64vector-ref scal 0))
            (beta  (f64vector-ref scal 1)))
        (do ((i 0 (+ i 1))) ((= i M))
          (let ((sum 0.0)
                (row (* i N)))
            (if (not (= alpha 0.0))
                (do ((j 0 (+ j 1))) ((= j N))
                  (set! sum (+ sum (* (@ref A (+ row j)) (@ref x j))))))
            (@set! y i (if (= beta 0.0)
                           (* alpha sum)
                           (+ (* alpha sum) (* beta (@ref y i)))))))))

    ;; result[0] := sum_i x[i] y[i]
    (%k-dot (N x y result)
            (integer @vec @vec f64vector)
            void
      (let ((sum 0.0))
        (do ((i 0 (+ i 1))) ((= i N))
          (set! sum (+ sum (* (@ref x i) (@ref y i)))))
        (f64vector-set! result 0 sum)))

    ;; y[i] := alpha x[i] + y[i]
    (%k-axpy (N scal x y)
             (integer f64vector @vec @vec)
             void
      (let ((alpha (f64vector-ref scal 0)))
        (do ((i 0 (+ i 1))) ((= i N))
          (@set! y i (+ (* alpha (@ref x i)) (@ref y i)))))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Contiguous GEMM kernels
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; am_crunch_gemm_nn_<t> computes rows [start, end) of
  ;;   C := alpha * A B + beta * C
  ;; for row-major A (M x K), B (K x N) and C (M x N).  Columns are taken
  ;; eight at a time, with the eight partial sums kept in local double
  ;; variables; N8 is N rounded down to a multiple of eight, and the
  ;; remaining columns are done one by one.  Each element's products are
  ;; added in increasing k, in double precision.  scal holds alpha and
  ;; beta.  The kernels follow the calling
  ;; convention of crunch-dispatch4-f32/-f64, whose fourth vector they do
  ;; not use, and their names are valid C identifiers so that the
  ;; dispatcher can call them by address.
  (crunch
    (: (am_crunch_gemm_nn_f32 integer integer integer integer integer
                              f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_gemm_nn_f32 start end N K N8 scal A B C unused)
      (let ((alpha (f64vector-ref scal 0))
            (beta  (f64vector-ref scal 1)))
        (do ((i start (+ i 1))) ((= i end))
          (let ((arow (* i K))
                (crow (* i N)))
            (do ((j0 0 (+ j0 8))) ((= j0 N8))
              (let ((c0 0.0) (c1 0.0) (c2 0.0) (c3 0.0) (c4 0.0) (c5 0.0) (c6 0.0) (c7 0.0))
                (if (not (= alpha 0.0))
                    (do ((k 0 (+ k 1))) ((= k K))
                      (let ((a (f32vector-ref A (+ arow k)))
                            (b (+ (* k N) j0)))
                        (set! c0 (+ c0 (* a (f32vector-ref B (+ b 0)))))
                        (set! c1 (+ c1 (* a (f32vector-ref B (+ b 1)))))
                        (set! c2 (+ c2 (* a (f32vector-ref B (+ b 2)))))
                        (set! c3 (+ c3 (* a (f32vector-ref B (+ b 3)))))
                        (set! c4 (+ c4 (* a (f32vector-ref B (+ b 4)))))
                        (set! c5 (+ c5 (* a (f32vector-ref B (+ b 5)))))
                        (set! c6 (+ c6 (* a (f32vector-ref B (+ b 6)))))
                        (set! c7 (+ c7 (* a (f32vector-ref B (+ b 7))))))))
                  (f32vector-set! C (+ crow (+ j0 0)) (if (= beta 0.0) (* alpha c0)
                                       (+ (* alpha c0) (* beta (f32vector-ref C (+ crow (+ j0 0)))))))
                  (f32vector-set! C (+ crow (+ j0 1)) (if (= beta 0.0) (* alpha c1)
                                       (+ (* alpha c1) (* beta (f32vector-ref C (+ crow (+ j0 1)))))))
                  (f32vector-set! C (+ crow (+ j0 2)) (if (= beta 0.0) (* alpha c2)
                                       (+ (* alpha c2) (* beta (f32vector-ref C (+ crow (+ j0 2)))))))
                  (f32vector-set! C (+ crow (+ j0 3)) (if (= beta 0.0) (* alpha c3)
                                       (+ (* alpha c3) (* beta (f32vector-ref C (+ crow (+ j0 3)))))))
                  (f32vector-set! C (+ crow (+ j0 4)) (if (= beta 0.0) (* alpha c4)
                                       (+ (* alpha c4) (* beta (f32vector-ref C (+ crow (+ j0 4)))))))
                  (f32vector-set! C (+ crow (+ j0 5)) (if (= beta 0.0) (* alpha c5)
                                       (+ (* alpha c5) (* beta (f32vector-ref C (+ crow (+ j0 5)))))))
                  (f32vector-set! C (+ crow (+ j0 6)) (if (= beta 0.0) (* alpha c6)
                                       (+ (* alpha c6) (* beta (f32vector-ref C (+ crow (+ j0 6)))))))
                  (f32vector-set! C (+ crow (+ j0 7)) (if (= beta 0.0) (* alpha c7)
                                       (+ (* alpha c7) (* beta (f32vector-ref C (+ crow (+ j0 7)))))))))
            (do ((j N8 (+ j 1))) ((= j N))
              (let ((acc 0.0))
                (if (not (= alpha 0.0))
                    (do ((k 0 (+ k 1))) ((= k K))
                      (set! acc (+ acc (* (f32vector-ref A (+ arow k))
                                          (f32vector-ref B (+ (* k N) j)))))))
                (f32vector-set! C (+ crow j) (if (= beta 0.0) (* alpha acc)
                                       (+ (* alpha acc) (* beta (f32vector-ref C (+ crow j)))))))))))))

  (crunch
    (: (am_crunch_gemm_nn_f64 integer integer integer integer integer
                              f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_gemm_nn_f64 start end N K N8 scal A B C unused)
      (let ((alpha (f64vector-ref scal 0))
            (beta  (f64vector-ref scal 1)))
        (do ((i start (+ i 1))) ((= i end))
          (let ((arow (* i K))
                (crow (* i N)))
            (do ((j0 0 (+ j0 8))) ((= j0 N8))
              (let ((c0 0.0) (c1 0.0) (c2 0.0) (c3 0.0) (c4 0.0) (c5 0.0) (c6 0.0) (c7 0.0))
                (if (not (= alpha 0.0))
                    (do ((k 0 (+ k 1))) ((= k K))
                      (let ((a (f64vector-ref A (+ arow k)))
                            (b (+ (* k N) j0)))
                        (set! c0 (+ c0 (* a (f64vector-ref B (+ b 0)))))
                        (set! c1 (+ c1 (* a (f64vector-ref B (+ b 1)))))
                        (set! c2 (+ c2 (* a (f64vector-ref B (+ b 2)))))
                        (set! c3 (+ c3 (* a (f64vector-ref B (+ b 3)))))
                        (set! c4 (+ c4 (* a (f64vector-ref B (+ b 4)))))
                        (set! c5 (+ c5 (* a (f64vector-ref B (+ b 5)))))
                        (set! c6 (+ c6 (* a (f64vector-ref B (+ b 6)))))
                        (set! c7 (+ c7 (* a (f64vector-ref B (+ b 7))))))))
                  (f64vector-set! C (+ crow (+ j0 0)) (if (= beta 0.0) (* alpha c0)
                                       (+ (* alpha c0) (* beta (f64vector-ref C (+ crow (+ j0 0)))))))
                  (f64vector-set! C (+ crow (+ j0 1)) (if (= beta 0.0) (* alpha c1)
                                       (+ (* alpha c1) (* beta (f64vector-ref C (+ crow (+ j0 1)))))))
                  (f64vector-set! C (+ crow (+ j0 2)) (if (= beta 0.0) (* alpha c2)
                                       (+ (* alpha c2) (* beta (f64vector-ref C (+ crow (+ j0 2)))))))
                  (f64vector-set! C (+ crow (+ j0 3)) (if (= beta 0.0) (* alpha c3)
                                       (+ (* alpha c3) (* beta (f64vector-ref C (+ crow (+ j0 3)))))))
                  (f64vector-set! C (+ crow (+ j0 4)) (if (= beta 0.0) (* alpha c4)
                                       (+ (* alpha c4) (* beta (f64vector-ref C (+ crow (+ j0 4)))))))
                  (f64vector-set! C (+ crow (+ j0 5)) (if (= beta 0.0) (* alpha c5)
                                       (+ (* alpha c5) (* beta (f64vector-ref C (+ crow (+ j0 5)))))))
                  (f64vector-set! C (+ crow (+ j0 6)) (if (= beta 0.0) (* alpha c6)
                                       (+ (* alpha c6) (* beta (f64vector-ref C (+ crow (+ j0 6)))))))
                  (f64vector-set! C (+ crow (+ j0 7)) (if (= beta 0.0) (* alpha c7)
                                       (+ (* alpha c7) (* beta (f64vector-ref C (+ crow (+ j0 7)))))))))
            (do ((j N8 (+ j 1))) ((= j N))
              (let ((acc 0.0))
                (if (not (= alpha 0.0))
                    (do ((k 0 (+ k 1))) ((= k K))
                      (set! acc (+ acc (* (f64vector-ref A (+ arow k))
                                          (f64vector-ref B (+ (* k N) j)))))))
                (f64vector-set! C (+ crow j) (if (= beta 0.0) (* alpha acc)
                                       (+ (* alpha acc) (* beta (f64vector-ref C (+ crow j)))))))))))))

  ;; %k-pack-<t> copies a rows x cols operand stored with element strides
  ;; (rs, cs) into dst in row-major order, so that a transposed operand
  ;; can be passed to the contiguous kernels.
  (crunch
    (: (%k-pack-f32 integer integer f32vector integer integer f32vector) void)
    (define (%k-pack-f32 rows cols src rs cs dst)
      (do ((r 0 (+ r 1))) ((= r rows))
        (let ((s (* r rs))
              (d (* r cols)))
          (do ((c 0 (+ c 1))) ((= c cols))
            (f32vector-set! dst (+ d c) (f32vector-ref src (+ s (* c cs)))))))))

  (crunch
    (: (%k-pack-f64 integer integer f64vector integer integer f64vector) void)
    (define (%k-pack-f64 rows cols src rs cs dst)
      (do ((r 0 (+ r 1))) ((= r rows))
        (let ((s (* r rs))
              (d (* r cols)))
          (do ((c 0 (+ c 1))) ((= c cols))
            (f64vector-set! dst (+ d c) (f64vector-ref src (+ s (* c cs)))))))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Argument checking
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (check-dims! who . dims)
    (for-each (lambda (d)
                (unless (and (fixnum? d) (>= d 0))
                  (error who "dimension must be a non-negative fixnum" d)))
              dims))

  ;; Error unless vec holds at least need elements.
  (define (check-length! who vec vec-length need)
    (unless (<= need (vec-length vec))
      (error who "vector too short" need (vec-length vec))))

  ;; Elements needed by a rows x cols operand with element strides
  ;; (row-stride, col-stride): one past its highest index, or 0 if empty.
  (define (extent rows cols row-stride col-stride)
    (if (or (= rows 0) (= cols 0))
        0
        (+ 1 (* (- rows 1) row-stride) (* (- cols 1) col-stride))))

  (define (trans? t) (eq? t 'trans))

  (define (scalars alpha beta)
    (f64vector (exact->inexact alpha) (exact->inexact beta)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMM
  ;;; Normalized signatures:
  ;;;   (M N K alpha data-A data-B beta data-C) -> void
  ;;;   (M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C) -> void
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define crunch-native-build? (am-crunch-native-build?))

  (define gemm-nn-f32 (foreign-value "((void *)&am_crunch_gemm_nn_f32)" c-pointer))
  (define gemm-nn-f64 (foreign-value "((void *)&am_crunch_gemm_nn_f64)" c-pointer))

  ;; Smallest number of multiply-adds worth giving to one thread.
  (define gemm-min-work-per-thread 65536)

  ;; Rows of C each thread must receive so that it has at least
  ;; gemm-min-work-per-thread multiply-adds.
  (define (gemm-min-rows N K)
    (let ((row-work (* N K)))
      (if (= row-work 0)
          1
          (max 1 (quotient (+ gemm-min-work-per-thread row-work -1) row-work)))))

  ;; Computes C := alpha op(A) op(B) + beta C.  Operands that are not
  ;; stored in plain row-major order are first packed into row-major
  ;; copies; the product is then computed by the contiguous kernel, with
  ;; the rows of C shared among up to (crunch-thread-count) threads.
  (define (gemm-strided who kernel pack dispatch vec-length make-vec
                        M N K alpha A lda transa B ldb transb beta C)
    (check-dims! who M N K lda ldb)
    (let-values (((sai sak) (if (trans? transa) (values 1 lda) (values lda 1)))
                 ((sbk sbj) (if (trans? transb) (values 1 ldb) (values ldb 1))))
      (check-length! who A vec-length (extent M K sai sak))
      (check-length! who B vec-length (extent K N sbk sbj))
      (check-length! who C vec-length (* M N))
      (let ((A* (if (and (= sai K) (= sak 1))
                    A
                    (let ((packed (make-vec (* M K))))
                      (pack M K A sai sak packed)
                      packed)))
            (B* (if (and (= sbk N) (= sbj 1))
                    B
                    (let ((packed (make-vec (* K N))))
                      (pack K N B sbk sbj packed)
                      packed))))
        (dispatch M N K (* 8 (quotient N 8)) (scalars alpha beta) A* B* C C kernel
                  (crunch-thread-count) (gemm-min-rows N K)))))

  (define (crunch-sgemm-strided M N K alpha A lda transa B ldb transb beta C)
    (gemm-strided 'crunch-sgemm-strided gemm-nn-f32 %k-pack-f32 crunch-dispatch4-f32
                  f32vector-length make-f32vector
                  M N K alpha A lda transa B ldb transb beta C))

  (define (crunch-dgemm-strided M N K alpha A lda transa B ldb transb beta C)
    (gemm-strided 'crunch-dgemm-strided gemm-nn-f64 %k-pack-f64 crunch-dispatch4-f64
                  f64vector-length make-f64vector
                  M N K alpha A lda transa B ldb transb beta C))

  ;; Row-major, no transposes: lda = K, ldb = N.
  (define (crunch-sgemm M N K alpha A B beta C)
    (crunch-sgemm-strided M N K alpha A K 'no-trans B N 'no-trans beta C))

  (define (crunch-dgemm M N K alpha A B beta C)
    (crunch-dgemm-strided M N K alpha A K 'no-trans B N 'no-trans beta C))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMV
  ;;; Normalized signature: (M N alpha data-A data-x beta data-y) -> void
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (gemv who kernel vec-length M N alpha A x beta y)
    (check-dims! who M N)
    (check-length! who A vec-length (* M N))
    (check-length! who x vec-length N)
    (check-length! who y vec-length M)
    (kernel M N (scalars alpha beta) A x y))

  (define (crunch-sgemv M N alpha A x beta y)
    (gemv 'crunch-sgemv %k-gemv-f32 f32vector-length M N alpha A x beta y))

  (define (crunch-dgemv M N alpha A x beta y)
    (gemv 'crunch-dgemv %k-gemv-f64 f64vector-length M N alpha A x beta y))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; DOT
  ;;; Normalized signature: (N data-x data-y) -> number
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (dot who kernel vec-length N x y)
    (check-dims! who N)
    (check-length! who x vec-length N)
    (check-length! who y vec-length N)
    (let ((result (make-f64vector 1 0.0)))
      (kernel N x y result)
      (f64vector-ref result 0)))

  (define (crunch-sdot N x y) (dot 'crunch-sdot %k-dot-f32 f32vector-length N x y))
  (define (crunch-ddot N x y) (dot 'crunch-ddot %k-dot-f64 f64vector-length N x y))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; AXPY
  ;;; Normalized signature: (N alpha data-x data-y) -> void
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (axpy who kernel vec-length N alpha x y)
    (check-dims! who N)
    (check-length! who x vec-length N)
    (check-length! who y vec-length N)
    (kernel N (scalars alpha 0.0) x y))

  (define (crunch-saxpy N alpha x y)
    (axpy 'crunch-saxpy %k-axpy-f32 f32vector-length N alpha x y))

  (define (crunch-daxpy N alpha x y)
    (axpy 'crunch-daxpy %k-axpy-f64 f64vector-length N alpha x y))

) ;; end module array-morphisms-crunch-blas-backend
