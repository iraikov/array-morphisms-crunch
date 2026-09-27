;;; smoke-7-scalar-precision.scm -- double-precision scalars across the
;;; crunch boundary.  crunch's embedded wrappers declare `float'-typed
;;; arguments and results with CHICKEN's single-precision `float' foreign
;;; type, so scalars that must keep full double precision are passed in, and
;;; returned through, one-element f64vectors instead.
(import scheme (chicken base) (chicken number-vector) crunch)
(include-relative "../../crunch-numvector-fix.scm")

(crunch
  (: (scale-sum f64vector integer f64vector f64vector) void)
  (define (scale-sum scalars n x result)
    (let ((alpha (f64vector-ref scalars 0)))
      (do ((i 0 (+ i 1))
           (acc 0.0 (+ acc (* alpha (f64vector-ref x i)))))
          ((= i n) (f64vector-set! result 0 acc))))))

(let ((x      (f64vector 1.0 2.0 3.0))
      (result (make-f64vector 1 0.0))
      (alpha  0.1))
  (scale-sum (f64vector alpha) 3 x result)
  (assert (= (f64vector-ref result 0)
             (+ (+ (+ 0.0 (* alpha 1.0)) (* alpha 2.0)) (* alpha 3.0)))))
(display "smoke-7-scalar-precision: ok\n")
