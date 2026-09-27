;;; smoke-1-integer.scm -- the crunch macro compiles a trivial integer kernel.
(import scheme (chicken base) crunch)

(crunch
  (: (add-one integer) integer)
  (define (add-one x) (+ x 1)))

(assert (= (add-one 41) 42))
(display "smoke-1-integer: ok\n")
