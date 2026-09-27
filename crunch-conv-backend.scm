;;; crunch-conv-backend.scm
;;; im2col / col2im / bias-add kernels for 2-D convolution compiled to C by
;;; crunch, the six convolution hooks of the blas-backend record built from
;;; them, and make-crunch-blas-backend, which assembles the complete record.
;;;
;;; The kernels are ports of kernels/im2col.c in array-morphisms and keep
;;; its layout contract.  For a batch of N images with C channels, the
;;; column buffer is row-major with shape [N*OH*OW, C*KH*KW]: row
;;; m = n*OH*OW + oh*OW + ow holds the receptive field of output position
;;; (n, oh, ow), and its columns run over (c, kh, kw) in that order, which
;;; matches the fan-in dimension of the weight tensor.  This layout is the
;;; same whether the images are stored NCHW ([N,C,H,W]) or NHWC
;;; ([N,H,W,C]); the two kernel variants differ only in how they address the
;;; image.  Input positions that fall in the zero padding contribute 0.
;;;
;;; Each hook is composed from these kernels and the crunch GEMM exactly as
;;; the corresponding hook of micro-blas-backend.scm is composed from its C
;;; kernels:
;;;
;;;   conv-fwd:         col = im2col(src); out = col wt; out += bias
;;;   conv-bwd-data:    col = g wt^T;      dx  = col2im(col)
;;;   conv-bwd-weights: col = im2col(src); dwt = col^T g
;;;
;;; All convolution kernels are f32 only, as in the other backends.
;;;
;;;   (import array-morphisms-crunch-conv-backend)
;;;   (register-blas-backend! (make-crunch-blas-backend))

(module array-morphisms-crunch-conv-backend

  (make-crunch-blas-backend

   ;; Column-buffer kernels, with argument checking
   crunch-im2col-nchw-f32  crunch-im2col-nhwc-f32
   crunch-col2im-nchw-f32  crunch-col2im-nhwc-f32
   crunch-bias-add-f32

   ;; Convolution hooks (blas-backend signatures)
   crunch-conv-fwd-im2col-f32
   crunch-conv-bwd-data-im2col-f32
   crunch-conv-bwd-weights-im2col-f32
   crunch-conv-fwd-nhwc-im2col-f32
   crunch-conv-bwd-data-nhwc-im2col-f32
   crunch-conv-bwd-weights-nhwc-im2col-f32)

  (import scheme (chicken base) (chicken foreign) (chicken number-vector) crunch)
  (import array-morphisms-blas-exec)
  (import array-morphisms-crunch-blas-backend)

  (include "crunch-numvector-fix.scm")

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Kernels
  ;;; Image geometry arguments, in this order:
  ;;;   N C H W   batch size, channels, image height and width
  ;;;   KH KW     kernel height and width
  ;;;   SH SW     strides
  ;;;   PH PW     zero padding on each side
  ;;;   OH OW     output height and width
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; col[m, (c,kh,kw)] := src[n, c, ih, iw], or 0 inside the padding.
  (crunch
    (: (%k-im2col-nchw f32vector f32vector integer integer integer integer
                       integer integer integer integer integer integer
                       integer integer) void)
    (define (%k-im2col-nchw col src N C H W KH KW SH SW PH PW OH OW)
      (let ((fan-in (* (* C KH) KW))
            (HW     (* H W)))
        (do ((n 0 (+ n 1))) ((= n N))
          (do ((oh 0 (+ oh 1))) ((= oh OH))
            (do ((ow 0 (+ ow 1))) ((= ow OW))
              (let ((row (* (+ (+ (* (* n OH) OW) (* oh OW)) ow) fan-in))
                    (ih0 (- (* oh SH) PH))
                    (iw0 (- (* ow SW) PW)))
                (do ((c 0 (+ c 1))) ((= c C))
                  (let ((src-c (* (+ (* n C) c) HW)))
                    (do ((kh 0 (+ kh 1))) ((= kh KH))
                      (let ((ih   (+ ih0 kh))
                            (base (+ row (* (+ (* c KH) kh) KW))))
                        (if (or (< ih 0) (>= ih H))
                            (do ((kw 0 (+ kw 1))) ((= kw KW))
                              (f32vector-set! col (+ base kw) 0.0))
                            (let ((src-h (+ src-c (* ih W))))
                              (do ((kw 0 (+ kw 1))) ((= kw KW))
                                (let ((iw (+ iw0 kw)))
                                  (if (and (>= iw 0) (< iw W))
                                      (f32vector-set! col (+ base kw)
                                                      (f32vector-ref src (+ src-h iw)))
                                      (f32vector-set! col (+ base kw) 0.0)))))))))))))))))

  ;; col[m, (c,kh,kw)] := src[n, ih, iw, c], or 0 inside the padding.
  (crunch
    (: (%k-im2col-nhwc f32vector f32vector integer integer integer integer
                       integer integer integer integer integer integer
                       integer integer) void)
    (define (%k-im2col-nhwc col src N C H W KH KW SH SW PH PW OH OW)
      (let ((fan-in (* (* C KH) KW))
            (WC     (* W C)))
        (do ((n 0 (+ n 1))) ((= n N))
          (do ((oh 0 (+ oh 1))) ((= oh OH))
            (do ((ow 0 (+ ow 1))) ((= ow OW))
              (let ((row   (* (+ (+ (* (* n OH) OW) (* oh OW)) ow) fan-in))
                    (src-n (* (* n H) WC))
                    (ih0   (- (* oh SH) PH))
                    (iw0   (- (* ow SW) PW)))
                (do ((c 0 (+ c 1))) ((= c C))
                  (do ((kh 0 (+ kh 1))) ((= kh KH))
                    (let ((ih   (+ ih0 kh))
                          (base (+ row (* (+ (* c KH) kh) KW))))
                      (if (or (< ih 0) (>= ih H))
                          (do ((kw 0 (+ kw 1))) ((= kw KW))
                            (f32vector-set! col (+ base kw) 0.0))
                          (let ((src-h (+ src-n (* ih WC))))
                            (do ((kw 0 (+ kw 1))) ((= kw KW))
                              (let ((iw (+ iw0 kw)))
                                (if (and (>= iw 0) (< iw W))
                                    (f32vector-set! col (+ base kw)
                                                    (f32vector-ref src (+ (+ src-h (* iw C)) c)))
                                    (f32vector-set! col (+ base kw) 0.0))))))))))))))))

  ;; dx := 0, then dx[n, c, ih, iw] += col[m, (c,kh,kw)] for every position
  ;; outside the padding.
  (crunch
    (: (%k-col2im-nchw f32vector f32vector integer integer integer integer
                       integer integer integer integer integer integer
                       integer integer) void)
    (define (%k-col2im-nchw dx col N C H W KH KW SH SW PH PW OH OW)
      (let ((fan-in (* (* C KH) KW))
            (HW     (* H W)))
        (do ((i 0 (+ i 1))) ((= i (* (* N C) HW)))
          (f32vector-set! dx i 0.0))
        (do ((n 0 (+ n 1))) ((= n N))
          (do ((oh 0 (+ oh 1))) ((= oh OH))
            (do ((ow 0 (+ ow 1))) ((= ow OW))
              (let ((row (* (+ (+ (* (* n OH) OW) (* oh OW)) ow) fan-in))
                    (ih0 (- (* oh SH) PH))
                    (iw0 (- (* ow SW) PW)))
                (do ((c 0 (+ c 1))) ((= c C))
                  (let ((dx-c (* (+ (* n C) c) HW)))
                    (do ((kh 0 (+ kh 1))) ((= kh KH))
                      (let ((ih (+ ih0 kh)))
                        (if (and (>= ih 0) (< ih H))
                            (let ((dx-h (+ dx-c (* ih W)))
                                  (base (+ row (* (+ (* c KH) kh) KW))))
                              (do ((kw 0 (+ kw 1))) ((= kw KW))
                                (let ((iw (+ iw0 kw)))
                                  (if (and (>= iw 0) (< iw W))
                                      (f32vector-set! dx (+ dx-h iw)
                                                      (+ (f32vector-ref dx (+ dx-h iw))
                                                         (f32vector-ref col (+ base kw))))))))))))))))))))

  ;; NHWC counterpart of %k-col2im-nchw: dx has shape [N, H, W, C].
  (crunch
    (: (%k-col2im-nhwc f32vector f32vector integer integer integer integer
                       integer integer integer integer integer integer
                       integer integer) void)
    (define (%k-col2im-nhwc dx col N C H W KH KW SH SW PH PW OH OW)
      (let ((fan-in (* (* C KH) KW))
            (WC     (* W C)))
        (do ((i 0 (+ i 1))) ((= i (* (* N H) WC)))
          (f32vector-set! dx i 0.0))
        (do ((n 0 (+ n 1))) ((= n N))
          (do ((oh 0 (+ oh 1))) ((= oh OH))
            (do ((ow 0 (+ ow 1))) ((= ow OW))
              (let ((row  (* (+ (+ (* (* n OH) OW) (* oh OW)) ow) fan-in))
                    (dx-n (* (* n H) WC))
                    (ih0  (- (* oh SH) PH))
                    (iw0  (- (* ow SW) PW)))
                (do ((c 0 (+ c 1))) ((= c C))
                  (do ((kh 0 (+ kh 1))) ((= kh KH))
                    (let ((ih (+ ih0 kh)))
                      (if (and (>= ih 0) (< ih H))
                          (let ((dx-h (+ dx-n (* ih WC)))
                                (base (+ row (* (+ (* c KH) kh) KW))))
                            (do ((kw 0 (+ kw 1))) ((= kw KW))
                              (let ((iw (+ iw0 kw)))
                                (if (and (>= iw 0) (< iw W))
                                    (let ((j (+ (+ dx-h (* iw C)) c)))
                                      (f32vector-set! dx j
                                                      (+ (f32vector-ref dx j)
                                                         (f32vector-ref col (+ base kw))))))))))))))))))))

  ;; out[m, co] += b[co] for an out of shape [M, out-ch].
  (crunch
    (: (%k-bias-add f32vector f32vector integer integer) void)
    (define (%k-bias-add out b M out-ch)
      (do ((m 0 (+ m 1))) ((= m M))
        (let ((row (* m out-ch)))
          (do ((co 0 (+ co 1))) ((= co out-ch))
            (f32vector-set! out (+ row co)
                            (+ (f32vector-ref out (+ row co)) (f32vector-ref b co))))))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Checked kernel wrappers
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (check-geometry! who N C H W KH KW SH SW PH PW OH OW)
    (for-each (lambda (d)
                (unless (and (fixnum? d) (>= d 0))
                  (error who "geometry argument must be a non-negative fixnum" d)))
              (list N C H W KH KW PH PW OH OW))
    (unless (and (fixnum? SH) (> SH 0) (fixnum? SW) (> SW 0))
      (error who "strides must be positive fixnums" SH SW)))

  (define (check-f32-length! who vec need)
    (unless (f32vector? vec)
      (error who "expected an f32vector" vec))
    (unless (<= need (f32vector-length vec))
      (error who "vector too short" need (f32vector-length vec))))

  (define (col-size N C KH KW OH OW) (* N OH OW C KH KW))

  (define (crunch-im2col-nchw-f32 col src N C H W KH KW SH SW PH PW OH OW)
    (check-geometry! 'crunch-im2col-nchw-f32 N C H W KH KW SH SW PH PW OH OW)
    (check-f32-length! 'crunch-im2col-nchw-f32 col (col-size N C KH KW OH OW))
    (check-f32-length! 'crunch-im2col-nchw-f32 src (* N C H W))
    (%k-im2col-nchw col src N C H W KH KW SH SW PH PW OH OW))

  (define (crunch-im2col-nhwc-f32 col src N C H W KH KW SH SW PH PW OH OW)
    (check-geometry! 'crunch-im2col-nhwc-f32 N C H W KH KW SH SW PH PW OH OW)
    (check-f32-length! 'crunch-im2col-nhwc-f32 col (col-size N C KH KW OH OW))
    (check-f32-length! 'crunch-im2col-nhwc-f32 src (* N C H W))
    (%k-im2col-nhwc col src N C H W KH KW SH SW PH PW OH OW))

  (define (crunch-col2im-nchw-f32 dx col N C H W KH KW SH SW PH PW OH OW)
    (check-geometry! 'crunch-col2im-nchw-f32 N C H W KH KW SH SW PH PW OH OW)
    (check-f32-length! 'crunch-col2im-nchw-f32 dx (* N C H W))
    (check-f32-length! 'crunch-col2im-nchw-f32 col (col-size N C KH KW OH OW))
    (%k-col2im-nchw dx col N C H W KH KW SH SW PH PW OH OW))

  (define (crunch-col2im-nhwc-f32 dx col N C H W KH KW SH SW PH PW OH OW)
    (check-geometry! 'crunch-col2im-nhwc-f32 N C H W KH KW SH SW PH PW OH OW)
    (check-f32-length! 'crunch-col2im-nhwc-f32 dx (* N C H W))
    (check-f32-length! 'crunch-col2im-nhwc-f32 col (col-size N C KH KW OH OW))
    (%k-col2im-nhwc dx col N C H W KH KW SH SW PH PW OH OW))

  (define (crunch-bias-add-f32 out b M out-ch)
    (for-each (lambda (d)
                (unless (and (fixnum? d) (>= d 0))
                  (error 'crunch-bias-add-f32 "dimension must be a non-negative fixnum" d)))
              (list M out-ch))
    (check-f32-length! 'crunch-bias-add-f32 out (* M out-ch))
    (check-f32-length! 'crunch-bias-add-f32 b out-ch)
    (%k-bias-add out b M out-ch))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Convolution hooks
  ;;;
  ;;; conv-fwd:  out bias col src wt M N K out-ch Nbatch C H W KH KW SH SW PH PW OH OW
  ;;;   M = Nbatch*OH*OW, N = out-ch, K = fan-in; out is [M, out-ch].
  ;;; conv-bwd-data: dx col g wt M K N out-ch Nbatch C H W KH KW SH SW PH PW OH OW
  ;;;   g is [M, N] = [M, out-ch], wt is [K, N] = [fan-in, out-ch].
  ;;; conv-bwd-weights: dwt col src g fan-in out-ch M Nbatch C H W KH KW SH SW PH PW OH OW
  ;;;   dwt is [fan-in, out-ch].
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (crunch-conv-fwd-im2col-f32 out bias col src wt M N K out-ch Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-im2col-nchw-f32 col src Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-sgemm M N K 1.0 col wt 0.0 out)
    (crunch-bias-add-f32 out bias M out-ch))

  (define (crunch-conv-bwd-data-im2col-f32 dx col g wt M K N out-ch Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-sgemm-strided M K N 1.0 g N 'no-trans wt N 'trans 0.0 col)
    (crunch-col2im-nchw-f32 dx col Nbatch C H W KH KW SH SW PH PW OH OW))

  (define (crunch-conv-bwd-weights-im2col-f32 dwt col src g fan-in out-ch M Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-im2col-nchw-f32 col src Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-sgemm-strided fan-in out-ch M 1.0 col fan-in 'trans g out-ch 'no-trans 0.0 dwt))

  (define (crunch-conv-fwd-nhwc-im2col-f32 out bias col src wt M N K out-ch Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-im2col-nhwc-f32 col src Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-sgemm M N K 1.0 col wt 0.0 out)
    (crunch-bias-add-f32 out bias M out-ch))

  (define (crunch-conv-bwd-data-nhwc-im2col-f32 dx col g wt M K N out-ch Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-sgemm-strided M K N 1.0 g N 'no-trans wt N 'trans 0.0 col)
    (crunch-col2im-nhwc-f32 dx col Nbatch C H W KH KW SH SW PH PW OH OW))

  (define (crunch-conv-bwd-weights-nhwc-im2col-f32 dwt col src g fan-in out-ch M Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-im2col-nhwc-f32 col src Nbatch C H W KH KW SH SW PH PW OH OW)
    (crunch-sgemm-strided fan-in out-ch M 1.0 col fan-in 'trans g out-ch 'no-trans 0.0 dwt))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Public Constructor
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (make-crunch-blas-backend)
    "Construct a blas-backend record whose GEMM, GEMV, DOT, AXPY and
    convolution kernels are all compiled by crunch.

    Usage:
      (import array-morphisms-crunch-conv-backend)
      (register-blas-backend! (make-crunch-blas-backend))"
    (make-blas-backend
     'crunch
     crunch-dgemm          crunch-sgemm           ; gemm-f64          gemm-f32
     crunch-dgemm-strided  crunch-sgemm-strided   ; gemm-strided-f64  gemm-strided-f32
     crunch-dgemv          crunch-sgemv           ; gemv-f64          gemv-f32
     crunch-ddot           crunch-sdot            ; dot-f64           dot-f32
     crunch-daxpy          crunch-saxpy           ; axpy-f64          axpy-f32
     crunch-conv-fwd-im2col-f32
     crunch-conv-bwd-data-im2col-f32
     crunch-conv-bwd-weights-im2col-f32
     crunch-conv-fwd-nhwc-im2col-f32
     crunch-conv-bwd-data-nhwc-im2col-f32
     crunch-conv-bwd-weights-nhwc-im2col-f32))

) ;; end module array-morphisms-crunch-conv-backend
