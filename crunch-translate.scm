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
   translate-fused-body
   codegen-dtypes
   codegen-float-dtype?
   codegen-dtype-type
   translate-typed-bodies
   typed-store-form
   typed-helper-definitions)

  (import scheme (chicken base) array-morphisms-scalar-ops-table
          (only (scheme base) exact-integer?)
          (only srfi-1 every any))

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

  ;;;; ============================================================
  ;;;; Typed translation, for generated kernels
  ;;;; ============================================================
  ;;;
  ;;; The code generator translates bodies whose operands may have
  ;;; different dtypes, integer dtypes included.  Each value of a body is
  ;;; then either a flonum (type f, a C double) or an exact integer (type i,
  ;;; a C 64-bit integer), as it would be in the Scheme procedures that
  ;;; compute the body on the host:
  ;;;
  ;;;   - an input of dtype f32 or f64 is a flonum, one of dtype s32, s64 or
  ;;;     u32 an integer; an exact literal is an integer;
  ;;;   - the generic ops (add, sub, mul, div, negate, abs, floor, ceiling)
  ;;;     compute exactly on two integers and convert an integer to a flonum
  ;;;     when the other argument is a flonum, as Scheme's generic arithmetic
  ;;;     does; the transcendental ops and pow take flonums only;
  ;;;   - the other table ops convert their arguments to flonums and give a
  ;;;     flonum, and the comparisons compare their arguments exactly and
  ;;;     give 1.0 or 0.0.
  ;;;
  ;;; Scheme integers have no bound, so an integer result that does not fit
  ;;; in 64 bits, a store of a value that does not fit its output dtype, and
  ;;; a division of a flonum by the integer 0 cannot be computed by the
  ;;; kernel.  The helpers that could meet such a case set the module's
  ;;; failure flag and return 0; the caller then runs the whole segop again
  ;;; on the host, which computes it exactly or signals the error.  Bodies
  ;;; whose integer results would not be integers, such as the quotient of
  ;;; two integers, are not translated.
  ;;;
  ;;; Floating-point ops are computed in double precision, as on the host,
  ;;; and an f32 value is rounded where a cast or a store says so.  The ops
  ;;; that the activation kernels also compute (negate, abs, exp, log, sqrt
  ;;; and the activations) use the same table formulas, so a kernel gives
  ;;; the values of the unfused kernels.  Where a Scheme procedure would
  ;;; return a complex number (the logarithm or square root of a negative
  ;;; number, or a negative number raised to a fractional power), the C
  ;;; functions give NaN.

  ;; The dtypes a generated kernel can read and write.
  (define codegen-dtypes '(f32 f64 s32 s64 u32))

  (define (codegen-float-dtype? dt) (memq dt '(f32 f64)))

  (define (codegen-dtype-type dt)
    "The type, f or i, of a value read from an array of dtype dt."
    (if (codegen-float-dtype? dt) 'f 'i))

  ;; The table ops that the generic arithmetic of Scheme computes.
  (define generic-int-ops '(add sub mul negate abs floor ceiling))
  (define float-only-ops '(div pow exp log sqrt sin cos tan))
  (define comparison-ops '((gt . >) (lt . <) (eq . =) (ge . >=) (le . <=)))

  ;; Largest magnitude of an integer literal that a kernel embeds.
  (define max-literal (expt 2 62))

  (define (translate-typed-bodies bodies input helper #!key (reduced #f))
    "Translate bodies, a list of segop bodies over the same inputs, into
    crunch code, or return #f when some body is outside the translated
    grammar.  The result is a list (bindings atoms may-fail?): bindings,
    a list of (variable form) pairs for a let*, computes every value the
    bodies need, each subexpression that occurs more than once (as equal
    expressions) once; atoms gives, for each body, a pair (form . type)
    of its value, a variable or a literal; may-fail? is true when a helper
    that can set the failure flag is used.

    input:   a procedure giving, for input k, a pair (form . type) that
             reads it at the current point
    helper:  a procedure giving the crunch name of the helper of a given
             name (see typed-helper-definitions)
    reduced: for a finalize body, the pair (form . type) of the reduced
             value"
    (call-with-current-continuation
     (lambda (give-up)
       (let ((bindings '()) (memo '()) (counter 0) (may-fail? #f))
         (define (fail) (give-up #f))
         (define (h name) (helper name))
         (define (checked! name) (set! may-fail? #t) (h name))
         (define (bind! form type)
           (set! counter (+ counter 1))
           (let ((v (string->symbol (string-append "%t" (number->string counter)))))
             (set! bindings (cons (list v form) bindings))
             (cons v type)))
         (define (to-float a)
           (if (eq? (cdr a) 'f) (car a) `(,(h 'i2f) ,(car a))))
         (define (tr e)
           (cond ((and (exact-integer? e) (< (abs e) max-literal)) (cons e 'i))
                 ((and (number? e) (inexact? e) (real? e) (finite? e)
                       (not (and (zero? e) (eqv? e -0.0))))
                  (cons e 'f))
                 ((not (and (pair? e) (list? e))) (fail))
                 ((eq? (car e) 'in) (input (cadr e)))
                 ((eq? (car e) 'reduced) (or reduced (fail)))
                 (else
                  (let ((m (assoc e memo)))
                    (if m
                        (cdr m)
                        (let ((r (tr-node e)))
                          (set! memo (cons (cons e r) memo))
                          r))))))
         (define (tr-node e)
           (case (car e)
             ((cast) (tr-cast (cadr e) (tr (caddr e))))
             ((in reduced param idx opaque if let*) (fail))
             (else
              (let ((entry (assq (car e) scalar-ops-table)))
                (unless (and entry (= (length (cadr entry)) (length (cdr e)))) (fail))
                (tr-op entry (map tr (cdr e)))))))
         (define (tr-cast dt a)
           (case dt
             ((f64) (if (eq? (cdr a) 'f) a (bind! (to-float a) 'f)))
             ((f32) (bind! `(,(h 'r32) ,(to-float a)) 'f))
             ((s32 s64 u32) (bind! (int-store-form dt a checked!) 'i))
             (else (fail))))
         (define (all-int? args) (every (lambda (a) (eq? (cdr a) 'i)) args))
         (define (table-expansion entry args)
           `(let ,(map (lambda (p a) (list p (to-float a))) (cadr entry) args)
              ,(branch-free-indicators (caddr entry))))
         (define (tr-op entry args)
           (let ((op (car entry)))
             (cond
              ((assq op comparison-ops)
               => (lambda (c) (bind! (comparison-form (cdr c) args) 'f)))
              ((memq op generic-int-ops)
               (if (all-int? args)
                   (case op
                     ((add) (bind! `(,(checked! 'addi) ,(car (car args)) ,(car (cadr args))) 'i))
                     ((sub) (bind! `(,(checked! 'subi) ,(car (car args)) ,(car (cadr args))) 'i))
                     ((mul) (bind! `(,(checked! 'muli) ,(car (car args)) ,(car (cadr args))) 'i))
                     ((negate) (bind! `(,(checked! 'negi) ,(car (car args))) 'i))
                     ((abs) (bind! `(,(checked! 'absi) ,(car (car args))) 'i))
                     (else (car args)))
                   (case op
                     ((add) (bind! `(+ ,(to-float (car args)) ,(to-float (cadr args))) 'f))
                     ((sub) (bind! `(- ,(to-float (car args)) ,(to-float (cadr args))) 'f))
                     ((mul) (bind! `(* ,(to-float (car args)) ,(to-float (cadr args))) 'f))
                     (else (bind! (table-expansion entry args) 'f)))))
              ((memq op float-only-ops)
               (cond ((eq? op 'div)
                      (let ((a (car args)) (b (cadr args)))
                        (cond ((all-int? args) (fail))
                              ((eq? (cdr b) 'i)
                               (bind! `(,(checked! 'divfi) ,(car a) ,(car b)) 'f))
                              (else (bind! `(/ ,(to-float a) ,(car b)) 'f)))))
                     ((any (lambda (a) (eq? (cdr a) 'i)) args) (fail))
                     ((eq? op 'pow) (bind! `(,(h 'pow) ,(car (car args)) ,(car (cadr args))) 'f))
                     (else (bind! (table-expansion entry args) 'f))))
              (else (bind! (table-expansion entry args) 'f)))))
         (define (comparison-form rel args)
           (let ((a (car args)) (b (cadr args)))
             (define (indicator test) `(exact->inexact (if ,test 1 0)))
             (cond
              ((eq? (cdr a) (cdr b)) (indicator `(,rel ,(car a) ,(car b))))
              (else
               ;; c is the sign of a - b, from an exact comparison of the
               ;; integer with the flonum, or 2 when the flonum is NaN.
               (let* ((swap? (eq? (cdr a) 'f))
                      (c (if swap?
                             `(,(h 'cmpif) ,(car b) ,(car a))
                             `(,(h 'cmpif) ,(car a) ,(car b))))
                      (rel (if swap?
                               (case rel ((>) '<) ((<) '>) ((>=) '<=) ((<=) '>=) (else rel))
                               rel)))
                 (indicator
                  `(let ((%c ,c))
                     ,(case rel
                        ((>) '(= %c 1))
                        ((<) '(= %c -1))
                        ((=) '(= %c 0))
                        ((>=) '(or (= %c 0) (= %c 1)))
                        (else '(or (= %c 0) (= %c -1)))))))))))
         (let ((atoms (map tr bodies)))
           (list (reverse bindings) atoms may-fail?))))))

  ;; The form converting value a, of type f or i, as storing it in an array
  ;; of integer dtype dt does: truncated towards zero, made non-negative for
  ;; u32, and checked to fit.
  (define (int-store-form dt a checked!)
    (if (eq? (cdr a) 'f)
        `(,(checked! (case dt ((s32) 'f2s32) ((s64) 'f2s64) (else 'f2u32))) ,(car a))
        (case dt
          ((s32) `(,(checked! 'cs32) ,(car a)))
          ((u32) `(,(checked! 'cu32) ,(car a)))
          (else (car a)))))

  (define (typed-store-form dt a helper)
    "The form giving the value to store in an array of dtype dt for the
    value a, a pair (form . type), and whether it can set the failure flag,
    as two values.  A flonum stored in an f32 array is rounded by the store
    itself."
    (let ((fails? #f))
      (let ((form (cond ((codegen-float-dtype? dt)
                         (if (eq? (cdr a) 'f) (car a) `(,(helper 'i2f) ,(car a))))
                        (else (int-store-form dt a (lambda (name) (set! fails? #t) (helper name)))))))
        (values form fails?))))

  (define (typed-helper-definitions helper flag)
    "The crunch definitions of the helpers that translated bodies call, as
    a list of forms for a crunch block.  helper gives the crunch name of
    each helper, and flag is the name of the C variable, an int, that the
    helpers set on failure."
    (define (def name args ret body)
      `(define ,(helper name) (c-lambda ,args ,ret ,body)))
    (define (fail-return v) (string-append flag " = 1; return " v ";"))
    (list
     (def 'r32 '((float x)) 'float "return (double)(float)x;")
     (def 'i2f '((integer a)) 'float "return (double)a;")
     (def 'pow '((float a) (float b)) 'float "return __builtin_pow(a, b);")
     (def 'addi '((integer a) (integer b)) 'integer
       (string-append "long r; if (__builtin_add_overflow(a, b, &r)) { " (fail-return "0") " } return r;"))
     (def 'subi '((integer a) (integer b)) 'integer
       (string-append "long r; if (__builtin_sub_overflow(a, b, &r)) { " (fail-return "0") " } return r;"))
     (def 'muli '((integer a) (integer b)) 'integer
       (string-append "long r; if (__builtin_mul_overflow(a, b, &r)) { " (fail-return "0") " } return r;"))
     (def 'negi '((integer a)) 'integer
       (string-append "if (a == (-9223372036854775807L - 1)) { " (fail-return "0") " } return -a;"))
     (def 'absi '((integer a)) 'integer
       (string-append "if (a == (-9223372036854775807L - 1)) { " (fail-return "0") " } return a < 0 ? -a : a;"))
     (def 'divfi '((float a) (integer b)) 'float
       (string-append "if (b == 0) { " (fail-return "0.0") " } return a / (double)b;"))
     (def 'cs32 '((integer a)) 'integer
       (string-append "if (a < -2147483648L || a > 2147483647L) { " (fail-return "0") " } return a;"))
     (def 'cu32 '((integer a)) 'integer
       (string-append "if (a < -4294967295L || a > 4294967295L) { " (fail-return "0") " } return a < 0 ? -a : a;"))
     (def 'f2s32 '((float x)) 'integer
       (string-append "if (!(x > -2147483649.0 && x < 2147483648.0)) { " (fail-return "0") " } return (long)x;"))
     (def 'f2s64 '((float x)) 'integer
       (string-append "if (!(x >= -0x1p63 && x < 0x1p63)) { " (fail-return "0") " } return (long)x;"))
     (def 'f2u32 '((float x)) 'integer
       (string-append "if (!(x > -4294967296.0 && x < 4294967296.0)) { " (fail-return "0") " }"
                      " long t = (long)x; return t < 0 ? -t : t;"))
     ;; The sign of a - b for an integer a and a flonum b, computed exactly,
     ;; or 2 when b is NaN.
     (def 'cmpif '((integer a) (float b)) 'integer
       (string-append
        "if (b != b) return 2;"
        " if (b >= 0x1p63) return -1;"
        " if (b < -0x1p63) return 1;"
        " long t = (long)b;"
        " if (a < t) return -1;"
        " if (a > t) return 1;"
        " double f = b - (double)t;"
        " return f > 0.0 ? -1 : (f < 0.0 ? 1 : 0);"))))

) ;; end module array-morphisms-crunch-translate
