;;; tests/test-crunch-conv-backend.scm
;;; Tests for the Crunch-compiled convolution kernels 
;;; (crunch-conv-backend.scm).
;;;
;;; im2col only copies values and col2im adds them in the same order as the
;;; C kernels of array-morphisms (kernels/im2col.c), so both are compared
;;; with naive Scheme references.  The six blas-backend hooks are
;;; exercised through realization.scm's execute-conv-*-blas dispatch with
;;; the crunch backend registered, and compared with the pure-Scheme
;;; convolution references and with the microBLAS backend.
;;;
;;; Organisation:
;;;   Group 1 - im2col (NCHW and NHWC) against a naive reference
;;;   Group 2 - col2im (NCHW and NHWC) against a naive reference
;;;   Group 3 - bias-add
;;;   Group 4 - Convolution hooks through execute-conv-*-blas
;;;   Group 5 - Argument checking

(import scheme (chicken base))
(import test)
(import (only srfi-1 iota every))
(import srfi-4)
(import array-morphisms-blas-exec)
(import array-morphisms-realization)
(import array-morphisms-micro-blas-backend)
(import array-morphisms-crunch-blas-backend)
(import array-morphisms-crunch-conv-backend)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (rel-close? a b tol)
  (<= (abs (- a b)) (* tol (max 1.0 (abs a) (abs b)))))

(define (f32-close? a b tol)
  (and (= (f32vector-length a) (f32vector-length b))
       (every (lambda (x y) (rel-close? x y tol))
              (f32vector->list a) (f32vector->list b))))

(define (gen-a i) (sin (* (+ i 1) 0.123)))
(define (gen-b i) (cos (* (+ i 1) 0.071)))
(define (gen-g i) (sin (* (+ i 3) 0.211)))

(define (make-f32 n f)
  (let ((v (make-f32vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f32vector-set! v i (exact->inexact (f i))))))

;; A convolution geometry: (N C H W KH KW SH SW PH PW out-ch).
;; Output sizes follow OH = (H + 2PH - KH)/SH + 1.
(define geometries
  '((2 2 5 5 3 3 1 1 1 1 3)    ; same padding
    (1 3 7 6 3 2 2 1 1 0 4)    ; non-square kernel, mixed strides
    (2 1 4 4 1 1 1 1 0 0 2)    ; 1x1 kernel
    (1 2 3 3 3 3 1 1 2 2 2)    ; padding wider than the kernel overlap
    (3 2 8 5 2 3 2 2 1 2 5)    ; strides 2, asymmetric padding
    (1 1 1 1 1 1 1 1 0 0 1)))  ; single pixel

(define (geometry-label g)
  (apply string-append
         (map (lambda (x) (string-append (number->string x) " ")) g)))

;; Call (f N C H W KH KW SH SW PH PW OH OW out-ch) for geometry g.
(define (with-geometry g f)
  (apply (lambda (N C H W KH KW SH SW PH PW out-ch)
           (let ((OH (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
                 (OW (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW))))
             (f N C H W KH KW SH SW PH PW OH OW out-ch)))
         g))

;; Image index of element (n, c, ih, iw) in NCHW or NHWC storage.
(define (image-index layout N C H W n c ih iw)
  (if (eq? layout 'nchw)
      (+ (* (+ (* n C) c) H W) (* ih W) iw)
      (+ (* (+ (* n H) ih) W C) (* iw C) c)))

;; Visit every (col-index, image-index-or-#f) pair in the loop order of the
;; C kernels; image-index is #f for positions inside the zero padding.
(define (for-each-col-entry layout N C H W KH KW SH SW PH PW OH OW proc)
  (let ((fan-in (* C KH KW)))
    (do ((n 0 (+ n 1))) ((= n N))
      (do ((oh 0 (+ oh 1))) ((= oh OH))
        (do ((ow 0 (+ ow 1))) ((= ow OW))
          (do ((c 0 (+ c 1))) ((= c C))
            (do ((kh 0 (+ kh 1))) ((= kh KH))
              (do ((kw 0 (+ kw 1))) ((= kw KW))
                (let ((ih (+ (- (* oh SH) PH) kh))
                      (iw (+ (- (* ow SW) PW) kw))
                      (j  (+ (* (+ (* n OH OW) (* oh OW) ow) fan-in)
                             (* c KH KW) (* kh KW) kw)))
                  (proc j (and (>= ih 0) (< ih H) (>= iw 0) (< iw W)
                               (image-index layout N C H W n c ih iw))))))))))))

(define (naive-im2col layout src N C H W KH KW SH SW PH PW OH OW)
  (let ((col (make-f32vector (* N OH OW C KH KW) -7.0)))
    (for-each-col-entry layout N C H W KH KW SH SW PH PW OH OW
      (lambda (j x) (f32vector-set! col j (if x (f32vector-ref src x) 0.0))))
    col))

(define (naive-col2im layout col N C H W KH KW SH SW PH PW OH OW)
  (let ((dx (make-f32vector (* N C H W) 0.0)))
    (for-each-col-entry layout N C H W KH KW SH SW PH PW OH OW
      (lambda (j x)
        (when x
          (f32vector-set! dx x (+ (f32vector-ref dx x) (f32vector-ref col j))))))
    dx))

;; NHWC copy of an NCHW image batch.
(define (nchw->nhwc src N C H W)
  (let ((out (make-f32vector (* N C H W) 0.0)))
    (do ((n 0 (+ n 1))) ((= n N) out)
      (do ((c 0 (+ c 1))) ((= c C))
        (do ((ih 0 (+ ih 1))) ((= ih H))
          (do ((iw 0 (+ iw 1))) ((= iw W))
            (f32vector-set! out (image-index 'nhwc N C H W n c ih iw)
                            (f32vector-ref src (image-index 'nchw N C H W n c ih iw)))))))))

(define (dot-f32 a b)
  (let loop ((i 0) (s 0.0))
    (if (= i (f32vector-length a)) s
        (loop (+ i 1) (+ s (* (f32vector-ref a i) (f32vector-ref b i)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - im2col
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch conv - im2col"
  (for-each
   (lambda (g)
     (with-geometry g
       (lambda (N C H W KH KW SH SW PH PW OH OW out-ch)
         (let ((src (make-f32 (* N C H W) gen-a))
               (col-size (* N OH OW C KH KW)))
           (test-assert (string-append "im2col NCHW exact: " (geometry-label g))
             (let ((col (make-f32vector col-size -7.0)))
               (crunch-im2col-nchw-f32 col src N C H W KH KW SH SW PH PW OH OW)
               (equal? (f32vector->list col)
                       (f32vector->list
                        (naive-im2col 'nchw src N C H W KH KW SH SW PH PW OH OW)))))
           (test-assert (string-append "im2col NHWC exact: " (geometry-label g))
             (let ((col (make-f32vector col-size -7.0)))
               (crunch-im2col-nhwc-f32 col src N C H W KH KW SH SW PH PW OH OW)
               (equal? (f32vector->list col)
                       (f32vector->list
                        (naive-im2col 'nhwc src N C H W KH KW SH SW PH PW OH OW)))))
           (test-assert (string-append "NHWC and NCHW images give the same columns: "
                                       (geometry-label g))
             (let ((col1 (make-f32vector col-size 0.0))
                   (col2 (make-f32vector col-size 0.0)))
               (crunch-im2col-nchw-f32 col1 src N C H W KH KW SH SW PH PW OH OW)
               (crunch-im2col-nhwc-f32 col2 (nchw->nhwc src N C H W)
                                       N C H W KH KW SH SW PH PW OH OW)
               (equal? (f32vector->list col1) (f32vector->list col2))))))))
   geometries))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - col2im
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch conv - col2im"
  (for-each
   (lambda (g)
     (with-geometry g
       (lambda (N C H W KH KW SH SW PH PW OH OW out-ch)
         (let ((col (make-f32 (* N OH OW C KH KW) gen-b))
               (img-size (* N C H W)))
           (test-assert (string-append "col2im NCHW exact: " (geometry-label g))
             (let ((dx (make-f32vector img-size 99.0)))   ; must be overwritten
               (crunch-col2im-nchw-f32 dx col N C H W KH KW SH SW PH PW OH OW)
               (equal? (f32vector->list dx)
                       (f32vector->list
                        (naive-col2im 'nchw col N C H W KH KW SH SW PH PW OH OW)))))
           (test-assert (string-append "col2im NHWC exact: " (geometry-label g))
             (let ((dx (make-f32vector img-size 99.0)))
               (crunch-col2im-nhwc-f32 dx col N C H W KH KW SH SW PH PW OH OW)
               (equal? (f32vector->list dx)
                       (f32vector->list
                        (naive-col2im 'nhwc col N C H W KH KW SH SW PH PW OH OW)))))
           (test-assert (string-append "col2im is the adjoint of im2col: " (geometry-label g))
             ;; <im2col(x), y> = <x, col2im(y)>
             (let ((x   (make-f32 img-size gen-a))
                   (cx  (make-f32vector (* N OH OW C KH KW) 0.0))
                   (dy  (make-f32vector img-size 0.0)))
               (crunch-im2col-nchw-f32 cx x N C H W KH KW SH SW PH PW OH OW)
               (crunch-col2im-nchw-f32 dy col N C H W KH KW SH SW PH PW OH OW)
               (rel-close? (dot-f32 cx col) (dot-f32 x dy) 1e-5)))))))
   geometries))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - bias-add
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch conv - bias-add"
  (test "bias is added to every row"
    '(1.5 2.25 3.5 4.25 5.5 6.25)
    (let ((out (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)))
      (crunch-bias-add-f32 out (f32vector 0.5 0.25) 3 2)
      (f32vector->list out)))
  (test "M=0 leaves the output unchanged"
    '(1.0)
    (let ((out (f32vector 1.0)))
      (crunch-bias-add-f32 out (f32vector 0.5) 0 1)
      (f32vector->list out))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - Convolution hooks through execute-conv-*-blas
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Run thunk with backend registered, restoring the previous backend.
(define (with-backend backend thunk)
  (let ((saved *active-backend*))
    (register-blas-backend! backend)
    (let ((r (thunk)))
      (set! *active-backend* saved)
      r)))

;; For one geometry and layout, the outputs of fwd, bwd-data and bwd-weights
;; computed through execute-conv-*-blas with backend registered.
(define (conv-outputs backend layout N C H W KH KW SH SW PH PW OH OW out-ch)
  (let* ((fan-in (* C KH KW)) (M (* N OH OW))
         (x-shape (if (eq? layout 'nchw) (vector N C H W) (vector N H W C)))
         (src (make-f32 (* N C H W) gen-a))
         (wt  (make-f32 (* fan-in out-ch) gen-b))
         (b   (make-f32 out-ch (lambda (i) (* 0.1 (+ i 1)))))
         (g   (make-f32 (* M out-ch) gen-g))
         (out (make-f32vector (* M out-ch) 0.0))
         (dx  (make-f32vector (* N C H W) 0.0))
         (dwt (make-f32vector (* fan-in out-ch) 0.0))
         (wt-shape (vector fan-in out-ch)))
    (with-backend backend
      (lambda ()
        (if (eq? layout 'nchw)
            (begin
              (execute-conv-fwd-blas out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
              (execute-conv-bwd-data-blas dx x-shape g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
              (execute-conv-bwd-weights-blas dwt wt-shape g #f src N C H W KH KW SH SW PH PW OH OW out-ch 'f32))
            (begin
              (execute-conv-fwd-nhwc-blas out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
              (execute-conv-bwd-data-nhwc-blas dx x-shape g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
              (execute-conv-bwd-weights-nhwc-blas dwt wt-shape g #f src N C H W KH KW SH SW PH PW OH OW out-ch 'f32)))
        (list out dx dwt)))))

;; The same three outputs from the pure-Scheme convolution references.
(define (reference-outputs layout N C H W KH KW SH SW PH PW OH OW out-ch)
  (let* ((fan-in (* C KH KW)) (M (* N OH OW))
         (x-shape (if (eq? layout 'nchw) (vector N C H W) (vector N H W C)))
         (src (make-f32 (* N C H W) gen-a))
         (wt  (make-f32 (* fan-in out-ch) gen-b))
         (b   (make-f32 out-ch (lambda (i) (* 0.1 (+ i 1)))))
         (g   (make-f32 (* M out-ch) gen-g))
         (out (make-f32vector (* M out-ch) 0.0))
         (dx  (make-f32vector (* N C H W) 0.0))
         (dwt (make-f32vector (* fan-in out-ch) 0.0))
         (wt-shape (vector fan-in out-ch)))
    (if (eq? layout 'nchw)
        (begin
          (execute-conv-fwd-nchw out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-data-nchw dx x-shape g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-weights-nchw dwt wt-shape g #f src N C H W KH KW SH SW PH PW OH OW out-ch 'f32))
        (begin
          (execute-conv-fwd-nhwc out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-data-nhwc dx x-shape g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-weights-nhwc dwt wt-shape g #f src N C H W KH KW SH SW PH PW OH OW out-ch 'f32)))
    (list out dx dwt)))

(test-group "crunch conv - hooks against the Scheme reference and microBLAS"
  (let ((crunch-be (make-crunch-blas-backend))
        (micro-be  (make-micro-blas-backend)))
    (for-each
     (lambda (layout)
       (for-each
        (lambda (g)
          (with-geometry g
            (lambda (N C H W KH KW SH SW PH PW OH OW out-ch)
              (let ((cr  (conv-outputs crunch-be layout N C H W KH KW SH SW PH PW OH OW out-ch))
                    (mb  (conv-outputs micro-be layout N C H W KH KW SH SW PH PW OH OW out-ch))
                    (ref (reference-outputs layout N C H W KH KW SH SW PH PW OH OW out-ch)))
                (for-each
                 (lambda (name k)
                   (test-assert (string-append (symbol->string layout) " " name
                                               " matches reference: " (geometry-label g))
                     (f32-close? (list-ref cr k) (list-ref ref k) 1e-4))
                   (test-assert (string-append (symbol->string layout) " " name
                                               " matches microBLAS: " (geometry-label g))
                     (f32-close? (list-ref cr k) (list-ref mb k) 1e-4)))
                 '("conv-fwd" "conv-bwd-data" "conv-bwd-weights")
                 '(0 1 2))))))
        geometries))
     '(nchw nhwc)))

  (test "execute-conv-*-blas calls the crunch hooks"
    '(1 1 1 1 1 1)
    ;; A backend whose six conv slots count their calls before delegating
    ;; to the crunch hooks.
    (let* ((counts (make-vector 6 0))
           (counting (lambda (k hook)
                       (lambda args
                         (vector-set! counts k (+ 1 (vector-ref counts k)))
                         (apply hook args))))
           (be (make-blas-backend
                'counting-crunch
                crunch-dgemm crunch-sgemm
                crunch-dgemm-strided crunch-sgemm-strided
                crunch-dgemv crunch-sgemv
                crunch-ddot crunch-sdot
                crunch-daxpy crunch-saxpy
                (counting 0 crunch-conv-fwd-im2col-f32)
                (counting 1 crunch-conv-bwd-data-im2col-f32)
                (counting 2 crunch-conv-bwd-weights-im2col-f32)
                (counting 3 crunch-conv-fwd-nhwc-im2col-f32)
                (counting 4 crunch-conv-bwd-data-nhwc-im2col-f32)
                (counting 5 crunch-conv-bwd-weights-nhwc-im2col-f32))))
      (with-geometry (car geometries)
        (lambda (N C H W KH KW SH SW PH PW OH OW out-ch)
          (conv-outputs be 'nchw N C H W KH KW SH SW PH PW OH OW out-ch)
          (conv-outputs be 'nhwc N C H W KH KW SH SW PH PW OH OW out-ch)))
      (vector->list counts))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5 - Argument checking
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch conv - argument checking"
  (test-error "im2col: column buffer too short"
    (crunch-im2col-nchw-f32 (make-f32vector 8 0.0) (make-f32vector 9 0.0)
                            1 1 3 3 3 3 1 1 0 0 1 1))
  (test-error "im2col: image too short"
    (crunch-im2col-nhwc-f32 (make-f32vector 9 0.0) (make-f32vector 8 0.0)
                            1 1 3 3 3 3 1 1 0 0 1 1))
  (test-error "col2im: image buffer too short"
    (crunch-col2im-nchw-f32 (make-f32vector 8 0.0) (make-f32vector 9 0.0)
                            1 1 3 3 3 3 1 1 0 0 1 1))
  (test-error "im2col: zero stride"
    (crunch-im2col-nchw-f32 (make-f32vector 9 0.0) (make-f32vector 9 0.0)
                            1 1 3 3 3 3 0 1 0 0 1 1))
  (test-error "im2col: f64vector instead of f32vector"
    (crunch-im2col-nchw-f32 (make-f64vector 9 0.0) (make-f32vector 9 0.0)
                            1 1 3 3 3 3 1 1 0 0 1 1))
  (test-error "bias-add: bias too short"
    (crunch-bias-add-f32 (make-f32vector 6 0.0) (make-f32vector 1 0.0) 3 2)))

(test-exit)
