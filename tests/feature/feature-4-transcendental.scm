;;; smoke-4-transcendental.scm -- exp/tanh and fp* primitives on values read
;;; with f32vector-ref; results narrowed back through f32vector-set!.
(import scheme (chicken base) (chicken flonum) (chicken number-vector) crunch)
(include-relative "../../crunch-numvector-fix.scm")

(crunch
  (: (k-sigmoid-f32 integer f32vector f32vector) void)
  (define (k-sigmoid-f32 n in out)
    (do ((i 0 (+ i 1))) ((= i n))
      (let ((x (f32vector-ref in i)))
        (f32vector-set! out i (/ 1.0 (+ 1.0 (exp (- x)))))))))

(crunch
  (: (k-tanh-f32 integer f32vector f32vector) void)
  (define (k-tanh-f32 n in out)
    (do ((i 0 (+ i 1))) ((= i n))
      (f32vector-set! out i (fptanh (f32vector-ref in i))))))

(let* ((in  (f32vector -1.0 0.0 0.5 3.0))
       (out (make-f32vector 4 0.0))
       (ref (make-f32vector 4 0.0)))
  (k-sigmoid-f32 4 in out)
  (do ((i 0 (+ i 1))) ((= i 4))
    (f32vector-set! ref i (/ 1.0 (+ 1.0 (exp (- (f32vector-ref in i)))))))
  (assert (equal? (f32vector->list out) (f32vector->list ref)))
  (k-tanh-f32 4 in out)
  (do ((i 0 (+ i 1))) ((= i 4))
    (f32vector-set! ref i (fptanh (f32vector-ref in i))))
  (assert (equal? (f32vector->list out) (f32vector->list ref))))
(display "smoke-4-transcendental: ok\n")
