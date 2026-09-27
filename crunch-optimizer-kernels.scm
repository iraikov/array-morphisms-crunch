;;; crunch-optimizer-kernels.scm
;;; Optimizer update kernels compiled to C by crunch.
;;;
;;; crunch-adam-f32 and crunch-adam-f64 perform one Adam update of a
;;; parameter vector in a single pass.  Their calling convention is that
;;; of the adam optimizer kernels of nanograd's optimizer module:
;;;
;;;   (crunch-adam-f32 n hyper param grad m v)
;;;
;;; updates the first n elements of param, grad, m and v in place.  hyper
;;; is an f64vector holding the learning rate, beta1, beta2, epsilon, the
;;; bias corrections 1 - beta1^t and 1 - beta2^t, and the weight decay.
;;; For each element the kernel
;;;
;;;   1. adds the weight-decay term to the gradient, g := g + wd * param,
;;;      when the weight decay is positive;
;;;   2. updates the first moment in two steps, m := beta1 * m, then
;;;      m := m + (1 - beta1) * g;
;;;   3. updates the second moment in two steps, v := beta2 * v, then
;;;      v := v + (1 - beta2) * g * g;
;;;   4. subtracts lr * m' / (sqrt(v') + eps) from the parameter, where m'
;;;      and v' are m and v divided by their bias corrections.
;;;
;;; Each step is computed in double precision and stored in the vector it
;;; updates, which is the order of operations of the Scheme optimizer.
;;; The elements are shared among up to (crunch-thread-count) threads;
;;; every element is updated on its own, so the result does not depend on
;;; the number of threads.
;;;
;;;   (import array-morphisms-crunch-optimizers nanograd-optimizer)
;;;   (register-optimizer-kernel! 'adam 'f32 crunch-adam-f32)
;;;   (register-optimizer-kernel! 'adam 'f64 crunch-adam-f64)

(module array-morphisms-crunch-optimizers

  (crunch-adam-f32
   crunch-adam-f64)

  (import scheme (chicken base) (chicken foreign) (chicken number-vector) crunch)
  (import (only array-morphisms-crunch-threads
                crunch-dispatch4-f32 crunch-dispatch4-f64
                crunch-thread-count crunch-thread-min-chunk))
  (import-for-syntax scheme (chicken base))

  (include "crunch-numvector-fix.scm")

  ;; am_crunch_adam_<t> updates elements [start, end) as described above.
  ;; It follows the calling convention of crunch-dispatch4-f32/-f64 (i0-i2
  ;; are not used), with hyper as the scalar vector and param, grad, m and
  ;; v as the four vectors.
  (define-syntax define-crunch-adam
    (er-macro-transformer
     (lambda (form r c)
       (let* ((dtype (cadr form))
              (name  (string->symbol (string-append "am_crunch_adam_" (symbol->string dtype))))
              (vec   (if (eq? dtype 'f32) 'f32vector 'f64vector))
              (ref   (if (eq? dtype 'f32) 'f32vector-ref 'f64vector-ref))
              (set   (if (eq? dtype 'f32) 'f32vector-set! 'f64vector-set!)))
         `(,(r 'crunch)
           (: (,name integer integer integer integer integer
                     f64vector ,vec ,vec ,vec ,vec) void)
           (define (,name start end i0 i1 i2 hyper param grad m v)
             (let ((lr  (f64vector-ref hyper 0))
                   (b1  (f64vector-ref hyper 1))
                   (b2  (f64vector-ref hyper 2))
                   (eps (f64vector-ref hyper 3))
                   (bc1 (f64vector-ref hyper 4))
                   (bc2 (f64vector-ref hyper 5))
                   (wd  (f64vector-ref hyper 6)))
               (do ((i start (+ i 1))) ((= i end))
                 (if (> wd 0.0)
                     (,set grad i (+ (,ref grad i) (* wd (,ref param i)))))
                 (let ((g (,ref grad i)))
                   (,set m i (* b1 (,ref m i)))
                   (,set m i (+ (,ref m i) (* (- 1.0 b1) g)))
                   (,set v i (* b2 (,ref v i)))
                   (,set v i (+ (,ref v i) (* (* (- 1.0 b2) g) g)))
                   (let ((m-hat (/ (,ref m i) bc1))
                         (v-hat (/ (,ref v i) bc2)))
                     (,set param i (- (,ref param i)
                                      (/ (* lr m-hat) (+ (sqrt v-hat) eps))))))))))))))

  (define-crunch-adam f32)
  (define-crunch-adam f64)

  (define adam-f32 (foreign-value "((void *)&am_crunch_adam_f32)" c-pointer))
  (define adam-f64 (foreign-value "((void *)&am_crunch_adam_f64)" c-pointer))

  (define (adam who kernel dispatch vec-length n hyper param grad m v)
    (unless (and (f64vector? hyper) (>= (f64vector-length hyper) 7))
      (error who "hyperparameters must be an f64vector of 7 elements" hyper))
    (unless (and (fixnum? n) (>= n 0)
                 (<= n (vec-length param)) (<= n (vec-length grad))
                 (<= n (vec-length m)) (<= n (vec-length v)))
      (error who "size exceeds vector length" n))
    (dispatch n 0 0 0 hyper param grad m v kernel
              (crunch-thread-count) (crunch-thread-min-chunk)))

  (define (crunch-adam-f32 n hyper param grad m v)
    "Adam update of the first n elements of the f32vectors param, grad, m
    and v, with the hyperparameters in the f64vector hyper."
    (adam 'crunch-adam-f32 adam-f32 crunch-dispatch4-f32 f32vector-length
          n hyper param grad m v))

  (define (crunch-adam-f64 n hyper param grad m v)
    "f64 counterpart of crunch-adam-f32."
    (adam 'crunch-adam-f64 adam-f64 crunch-dispatch4-f64 f64vector-length
          n hyper param grad m v))

) ;; end module array-morphisms-crunch-optimizers
