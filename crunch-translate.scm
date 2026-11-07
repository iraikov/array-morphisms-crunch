;;; crunch-translate.scm
;;;
;;; Translation of segop bodies (the scalar expressions of array-morphisms,
;;; see segop.scm there) into crunch Scheme.  The translator is used at
;;; syntax time by the kernel-defining macros of the activation module, and
;;; at run time by the code generator, so that every kernel the egg compiles
;;; computes a body in the same way.
;;;
;;; Table ops are expanded from their entries in the scalar-op table of
;;; array-morphisms, so a kernel computes exactly the formula of the op.

(module array-morphisms-crunch-translate

  (branch-free-indicators
   strip-casts
   body-max-input
   translate-fused-body)

  (import scheme (chicken base) array-morphisms-scalar-ops-table)

  (define (branch-free-indicators e)
    "Expression e with every (if test 1.0 0.0) and (if test 0.0 1.0)
    rewritten as (exact->inexact (if test 1 0)) and likewise.  Both give the
    same flonum for every input, NaN, infinities and -0.0 included.  crunch
    compiles an if to a C if statement, and the C compiler keeps a branch on
    a test whose two arms store flonum constants; for inputs whose sign
    varies at random that branch is mispredicted about half the time.  With
    integer arms the compiler sets the integer from the comparison flag and
    converts it, so the loop has no branch on the data."
    (cond ((not (pair? e)) e)
          ((and (eq? (car e) 'if) (= (length e) 4)
                (memv (caddr e) '(0.0 1.0)) (memv (cadddr e) '(0.0 1.0))
                (not (eqv? (caddr e) (cadddr e))))
           `(exact->inexact (if ,(branch-free-indicators (cadr e))
                                ,(inexact->exact (caddr e))
                                ,(inexact->exact (cadddr e)))))
          (else (map branch-free-indicators e))))

  (define (strip-casts e)
    "Body e with every (cast dtype x) replaced by x."
    (cond ((not (pair? e)) e)
          ((eq? (car e) 'cast) (strip-casts (caddr e)))
          ((eq? (car e) 'in) e)
          (else (cons (car e) (map strip-casts (cdr e))))))

  (define (body-max-input e)
    "The largest k for which body e contains (in k), or -1 when it reads no
    input."
    (cond ((not (pair? e)) -1)
          ((eq? (car e) 'in) (cadr e))
          (else (let loop ((xs (cdr e)) (m -1))
                  (if (null? xs) m (loop (cdr xs) (max m (body-max-input (car xs)))))))))

  (define (translate-fused-body e dtype input who)
    "The crunch expression computing body e in a kernel whose operands all
    have dtype f32 or f64.  The body may use numbers, (in k), table ops and,
    in an f32 body, (cast f32 x), which rounds x to single precision with
    am-crunch-round-f32.

    e:     the body
    dtype: f32 or f64
    input: a procedure giving the crunch expression that reads input k
    who:   the name reported in errors"
    (let tr ((e e))
      (cond ((number? e) (exact->inexact e))
            ((not (pair? e)) (error who "unexpected body form" e))
            ((eq? (car e) 'in) (input (cadr e)))
            ((eq? (car e) 'cast)
             (unless (and (eq? (cadr e) 'f32) (eq? dtype 'f32))
               (error who "only casts to f32, in f32 bodies, are supported" e))
             `(am-crunch-round-f32 ,(tr (caddr e))))
            ((assq (car e) scalar-ops-table)
             => (lambda (entry)
                  (let ((params (cadr entry)))
                    (unless (= (length params) (length (cdr e)))
                      (error who "wrong number of arguments" e))
                    `(let ,(map (lambda (p a) (list p (tr a))) params (cdr e))
                       ,(branch-free-indicators (caddr entry))))))
            (else (error who "not a table op" e)))))

) ;; end module array-morphisms-crunch-translate
