;;; smoke-2-f64vector.scm -- numeric-vector arguments cross the crunch FFI
;;; boundary; an axpy loop writes through to the caller's f64vector.
(import scheme (chicken base) (chicken number-vector) crunch)
(include-relative "../../crunch-numvector-fix.scm")

(crunch
  (: (axpy-f64 integer float f64vector f64vector) void)
  (define (axpy-f64 n alpha x y)
    (do ((i 0 (+ i 1))) ((= i n))
      (f64vector-set! y i (+ (f64vector-ref y i)
                             (* alpha (f64vector-ref x i)))))))

(let ((x (f64vector 1.0 2.0 3.0))
      (y (f64vector 10.0 20.0 30.0)))
  (axpy-f64 3 2.0 x y)
  (assert (equal? (f64vector->list y) '(12.0 24.0 36.0))))
(display "smoke-2-f64vector: ok\n")
