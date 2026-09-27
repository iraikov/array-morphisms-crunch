;;; smoke-6-table-macro.scm -- a table-driven macro that expands into several
;;; independent (crunch (define ...)) forms, one per table entry.  This is
;;; the templating pattern the activation-kernel table relies on.
(import scheme (chicken base) (chicken number-vector) crunch)
(import-for-syntax scheme (chicken base))
(include-relative "../../crunch-numvector-fix.scm")

(define-syntax define-crunch-unary-table
  (er-macro-transformer
   (lambda (form r c)
     (let ((entries (cdr form)))
       `(,(r 'begin)
         ,@(map (lambda (e)
                  (let ((name (string->symbol
                               (string-append "k-" (symbol->string (car e))))))
                    `(,(r 'crunch)
                      (: (,name integer f64vector f64vector) void)
                      (define (,name n in out)
                        (do ((i 0 (+ i 1))) ((= i n))
                          (let ((x (f64vector-ref in i)))
                            (f64vector-set! out i ,(cadr e))))))))
                entries))))))

(define-crunch-unary-table
  (double (* x 2.0))
  (square (* x x)))

(let ((in  (f64vector 1.0 -2.0 3.5))
      (out (make-f64vector 3 0.0)))
  (k-double 3 in out)
  (assert (equal? (f64vector->list out) '(2.0 -4.0 7.0)))
  (k-square 3 in out)
  (assert (equal? (f64vector->list out) '(1.0 4.0 12.25))))
(display "smoke-6-table-macro: ok\n")
