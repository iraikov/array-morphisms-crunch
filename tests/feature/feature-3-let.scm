;;; smoke-3-let.scm -- plain (non-loop) let / let* bindings inside crunch.
(import scheme (chicken base) crunch)

(crunch
  (: (poly float) float)
  (define (poly x)
    (let* ((x2 (* x x))
           (x3 (* x2 x)))
      (let ((a 2.0) (b 3.0))
        (+ (+ (* a x3) (* b x2)) x)))))

(assert (= (poly 2.0) 30.0))
(display "smoke-3-let: ok\n")
