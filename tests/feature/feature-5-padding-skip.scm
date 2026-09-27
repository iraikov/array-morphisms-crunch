;;; smoke-5-padding-skip.scm -- the C "col_idx += KW; continue;" padding
;;; skip of kernels/im2col.c, restated as if/else inside nested do loops.
;;; A 2-D single-channel im2col is compared with a plain-Scheme reference
;;; for several padding/stride combinations, so border handling is checked
;;; value by value rather than only for compilation.
(import scheme (chicken base) (chicken number-vector) crunch)
(include-relative "../../crunch-numvector-fix.scm")

(crunch
  (: (im2col-2d f32vector f32vector integer integer integer integer
                integer integer integer integer integer integer) void)
  (define (im2col-2d col src H W KH KW SH SW PH PW OH OW)
    (let ((fan-in (* KH KW)))
      (do ((oh 0 (+ oh 1))) ((= oh OH))
        (do ((ow 0 (+ ow 1))) ((= ow OW))
          (let ((row (* (+ (* oh OW) ow) fan-in))
                (ih0 (- (* oh SH) PH))
                (iw0 (- (* ow SW) PW)))
            (do ((kh 0 (+ kh 1))) ((= kh KH))
              (let ((ih (+ ih0 kh))
                    (base (+ row (* kh KW))))
                (if (or (< ih 0) (>= ih H))
                    (do ((kw 0 (+ kw 1))) ((= kw KW))
                      (f32vector-set! col (+ base kw) 0.0))
                    (do ((kw 0 (+ kw 1))) ((= kw KW))
                      (let ((iw (+ iw0 kw)))
                        (if (and (>= iw 0) (< iw W))
                            (f32vector-set! col (+ base kw)
                                            (f32vector-ref src (+ (* ih W) iw)))
                            (f32vector-set! col (+ base kw) 0.0)))))))))))))

(define (reference H W KH KW SH SW PH PW OH OW src)
  (let ((col (make-f32vector (* OH OW KH KW) -1.0)))
    (do ((oh 0 (+ oh 1))) ((= oh OH) col)
      (do ((ow 0 (+ ow 1))) ((= ow OW))
        (do ((kh 0 (+ kh 1))) ((= kh KH))
          (do ((kw 0 (+ kw 1))) ((= kw KW))
            (let ((ih (+ (- (* oh SH) PH) kh))
                  (iw (+ (- (* ow SW) PW) kw))
                  (j  (+ (* (+ (* oh OW) ow) KH KW) (* kh KW) kw)))
              (f32vector-set! col j
                (if (and (>= ih 0) (< ih H) (>= iw 0) (< iw W))
                    (f32vector-ref src (+ (* ih W) iw))
                    0.0)))))))))

(for-each
 (lambda (cfg)
   (apply
    (lambda (H W KH KW SH SW PH PW)
      (let* ((OH  (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
             (OW  (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW)))
             (src (make-f32vector (* H W) 0.0))
             (col (make-f32vector (* OH OW KH KW) -1.0)))
        (do ((i 0 (+ i 1))) ((= i (* H W)))
          (f32vector-set! src i (+ 1.0 i)))
        (im2col-2d col src H W KH KW SH SW PH PW OH OW)
        (assert (equal? (f32vector->list col)
                        (f32vector->list
                         (reference H W KH KW SH SW PH PW OH OW src))))))
    cfg))
 '((4 4 3 3 1 1 0 0)
   (4 4 3 3 1 1 1 1)
   (5 4 3 2 2 1 1 0)
   (3 3 3 3 1 1 2 2)
   (6 5 2 3 2 2 1 2)))
(display "smoke-5-padding-skip: ok\n")
