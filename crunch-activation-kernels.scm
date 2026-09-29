;;; crunch-activation-kernels.scm
;;; Whole-array activation kernels compiled to C by crunch, packaged as an
;;; activation backend for array-morphisms (see activation-exec.scm in the
;;; array-morphisms egg).
;;;
;;; Each activation op is one table entry (op (v) expr): expr computes the
;;; op's value from one input element v.  The define-crunch-activations
;;; macro turns every entry into an f32 and an f64 kernel, each a crunch
;;; procedure (kernel n in out) that stores expr for the first n elements of
;;; in into out.  crunch computes in double precision throughout, so the f32
;;; kernels read single floats, compute in double and round on store, which
;;; is exactly what the Scheme fallback does.
;;;
;;; The formulas below are those of basic-ops.scm (relu, sigmoid, tanh) and
;;; of the derivative maps emitted by ssa-vjp (relu-deriv, sigmoid-deriv,
;;; tanh-deriv), written the same way, so that each kernel gives the same
;;; floating-point result as the combiner it replaces.  relu is written as a
;;; comparison rather than max: (if (> v 0.0) v 0.0) gives 0.0 for NaN and
;;; -0.0, as CHICKEN's (max 0.0 v) does.  tanh is computed from
;;; e = exp(-2|x|), which cannot overflow, so large |x| gives +-1.0.
;;; Entries must use fpabs rather than abs: crunch 0.992 compiles abs on a
;;; float to C's integer abs(), which truncates its argument.
;;;
;;; The derivative entries are separate ops because ssa-vjp emits each
;;; derivative as its own unary binding and chooses its input: relu-deriv
;;; reads the forward input x, sigmoid-deriv and tanh-deriv read the forward
;;; output.  A derivative that needs two arrays, such as that of swish,
;;; cannot be written as a table entry.
;;;
;;; Each entry also yields two chunk kernels, am_crunch_<op>_chunk_f32 and
;;; am_crunch_<op>_chunk_f64 (with the hyphens of op replaced by
;;; underscores), crunch procedures (kernel start end in out) that compute
;;; the same expr for elements [start, end).  Their names consist of
;;; letters, digits and underscores only, so crunch emits them as C
;;; functions of exactly those names, whose addresses the threaded backend
;;; passes to the dispatcher of array-morphisms-crunch-threads.
;;;
;;; An entry's expr may use only v, numeric literals and crunch primitives.
;;; crunch procedures cannot close over Scheme values, so a constant such as
;;; a fixed slope must appear as a literal, and each distinct value needs an
;;; entry of its own.
;;;
;;; Adding an entry supplies only a fast kernel.  An op still needs a
;;; morph-<op> constructor, an ssa-vjp rule and a rebuild-morphism case in
;;; array-morphisms before SSA training graphs can contain it, and its name
;;; must be registered with register-activation-op! (done here for every
;;; entry) for the replay compiler to route it to the kernel.
;;;
;;;   (import array-morphisms-crunch-activations)
;;;   (register-activation-backend! (make-crunch-activation-backend))
;;;
;;; or, to split large arrays across threads (see crunch-thread-dispatch.scm):
;;;
;;;   (register-activation-backend! (make-crunch-threaded-activation-backend))

(module array-morphisms-crunch-activations

  (make-crunch-activation-backend
   make-crunch-threaded-activation-backend
   crunch-activation-ops
   crunch-activation-chunk-kernel

   ;; Raw kernels: (kernel n in out) -> void, no argument checking.
   crunch-relu-f32          crunch-relu-f64
   crunch-sigmoid-f32       crunch-sigmoid-f64
   crunch-tanh-f32          crunch-tanh-f64
   crunch-relu-deriv-f32    crunch-relu-deriv-f64
   crunch-sigmoid-deriv-f32 crunch-sigmoid-deriv-f64
   crunch-tanh-deriv-f32    crunch-tanh-deriv-f64)

  (import scheme (chicken base) (chicken foreign) (chicken flonum)
          (chicken number-vector) crunch)
  (import array-morphisms-activation-exec)
  (import array-morphisms-crunch-threads)
  (import-for-syntax scheme (chicken base))

  (include "crunch-numvector-fix.scm")

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Kernel generation
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; (define-crunch-activations (op (v) expr) ...)
  ;;
  ;; For each entry defines crunch-<op>-f32 and crunch-<op>-f64 and the chunk
  ;; kernels am_crunch_<op>_chunk_f32 and am_crunch_<op>_chunk_f64, and binds
  ;; %crunch-activation-table to a list of
  ;;   (op f32-kernel f64-kernel f32-chunk-address f64-chunk-address).
  (define-syntax define-crunch-activations
    (er-macro-transformer
     (lambda (form r c)
       (define (kernel-name op dtype)
         (string->symbol
          (string-append "crunch-" (symbol->string op) "-" (symbol->string dtype))))
       (define (chunk-name op dtype)
         (string->symbol
          (string-append "am_crunch_"
                         (list->string
                          (map (lambda (ch) (if (char=? ch #\-) #\_ ch))
                               (string->list (symbol->string op))))
                         "_chunk_" (symbol->string dtype))))
       (define (chunk-def op v expr dtype)
         (let ((name (chunk-name op dtype))
               (vec  (if (eq? dtype 'f32) 'f32vector 'f64vector))
               (ref  (if (eq? dtype 'f32) 'f32vector-ref 'f64vector-ref))
               (set  (if (eq? dtype 'f32) 'f32vector-set! 'f64vector-set!)))
           `(,(r 'crunch)
             (: (,name integer integer ,vec ,vec) void)
             (define (,name start end in out)
               (do ((i start (+ i 1))) ((= i end))
                 (let ((,v (,ref in i)))
                   (,set out i ,expr)))))))
       (define (chunk-address op dtype)
         `(,(r 'foreign-value)
           ,(string-append "((void *)&" (symbol->string (chunk-name op dtype)) ")")
           ,(r 'c-pointer)))
       (define (kernel-def op v expr dtype)
         (let ((name (kernel-name op dtype))
               (vec  (if (eq? dtype 'f32) 'f32vector 'f64vector))
               (ref  (if (eq? dtype 'f32) 'f32vector-ref 'f64vector-ref))
               (set  (if (eq? dtype 'f32) 'f32vector-set! 'f64vector-set!)))
           `(,(r 'crunch)
             (: (,name integer ,vec ,vec) void)
             (define (,name n in out)
               (do ((i 0 (+ i 1))) ((= i n))
                 (let ((,v (,ref in i)))
                   (,set out i ,expr)))))))
       (let ((entries (cdr form)))
         (for-each
          (lambda (e)
            (unless (and (list? e) (= (length e) 3) (symbol? (car e))
                         (list? (cadr e)) (= (length (cadr e)) 1)
                         (symbol? (car (cadr e))))
              (error 'define-crunch-activations
                     "entry must have the form (op (v) expr)" e)))
          entries)
         `(,(r 'begin)
           ,@(apply append
                    (map (lambda (e)
                           (let ((op (car e)) (v (car (cadr e))) (expr (caddr e)))
                             (list (kernel-def op v expr 'f32)
                                   (kernel-def op v expr 'f64)
                                   (chunk-def op v expr 'f32)
                                   (chunk-def op v expr 'f64))))
                         entries))
           (,(r 'define) %crunch-activation-table
            (,(r 'list)
             ,@(map (lambda (e)
                      `(,(r 'list) (,(r 'quote) ,(car e))
                                   ,(kernel-name (car e) 'f32)
                                   ,(kernel-name (car e) 'f64)
                                   ,(chunk-address (car e) 'f32)
                                   ,(chunk-address (car e) 'f64)))
                    entries))))))))

  (define-crunch-activations
    (relu          (x) (if (> x 0.0) x 0.0))
    (sigmoid       (x) (/ 1.0 (+ 1.0 (exp (- x)))))
    (tanh          (x) (let* ((e (exp (* -2.0 (fpabs x))))
                                  (t (/ (- 1.0 e) (+ 1.0 e))))
                             (if (< x 0.0) (- t) t)))
    (relu-deriv    (x) (if (> x 0.0) 1.0 0.0))
    (sigmoid-deriv (s) (* s (- 1.0 s)))
    (tanh-deriv    (t) (- 1.0 (* t t))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Binary element-wise kernels
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; am_crunch_<op>_binary_<t> stores a[i] op b[i] into out[i] for i in
  ;; [start, end).  They follow the calling convention of
  ;; crunch-dispatch4-f32/-f64 (i0-i2, scal and the fourth vector are not
  ;; used).  The operation is done in double precision and rounded on
  ;; store, as by the Scheme combiners add, sub, mul and div.

  (crunch
    (: (am_crunch_add_binary_f32 integer integer integer integer integer
                                  f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_add_binary_f32 start end i0 i1 i2 scal a b out unused)
      (do ((i start (+ i 1))) ((= i end))
        (f32vector-set! out i (+ (f32vector-ref a i) (f32vector-ref b i))))))

  (crunch
    (: (am_crunch_add_binary_f64 integer integer integer integer integer
                                  f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_add_binary_f64 start end i0 i1 i2 scal a b out unused)
      (do ((i start (+ i 1))) ((= i end))
        (f64vector-set! out i (+ (f64vector-ref a i) (f64vector-ref b i))))))

  (crunch
    (: (am_crunch_sub_binary_f32 integer integer integer integer integer
                                  f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_sub_binary_f32 start end i0 i1 i2 scal a b out unused)
      (do ((i start (+ i 1))) ((= i end))
        (f32vector-set! out i (- (f32vector-ref a i) (f32vector-ref b i))))))

  (crunch
    (: (am_crunch_sub_binary_f64 integer integer integer integer integer
                                  f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_sub_binary_f64 start end i0 i1 i2 scal a b out unused)
      (do ((i start (+ i 1))) ((= i end))
        (f64vector-set! out i (- (f64vector-ref a i) (f64vector-ref b i))))))

  (crunch
    (: (am_crunch_mul_binary_f32 integer integer integer integer integer
                                  f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_mul_binary_f32 start end i0 i1 i2 scal a b out unused)
      (do ((i start (+ i 1))) ((= i end))
        (f32vector-set! out i (* (f32vector-ref a i) (f32vector-ref b i))))))

  (crunch
    (: (am_crunch_mul_binary_f64 integer integer integer integer integer
                                  f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_mul_binary_f64 start end i0 i1 i2 scal a b out unused)
      (do ((i start (+ i 1))) ((= i end))
        (f64vector-set! out i (* (f64vector-ref a i) (f64vector-ref b i))))))

  (crunch
    (: (am_crunch_div_binary_f32 integer integer integer integer integer
                                  f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_div_binary_f32 start end i0 i1 i2 scal a b out unused)
      (do ((i start (+ i 1))) ((= i end))
        (f32vector-set! out i (/ (f32vector-ref a i) (f32vector-ref b i))))))

  (crunch
    (: (am_crunch_div_binary_f64 integer integer integer integer integer
                                  f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_div_binary_f64 start end i0 i1 i2 scal a b out unused)
      (do ((i start (+ i 1))) ((= i end))
        (f64vector-set! out i (/ (f64vector-ref a i) (f64vector-ref b i))))))

  (define %crunch-binary-table
    ;; (op f32-chunk-address f64-chunk-address)
    (list
     (list 'add (foreign-value "((void *)&am_crunch_add_binary_f32)" c-pointer)
               (foreign-value "((void *)&am_crunch_add_binary_f64)" c-pointer))
     (list 'sub (foreign-value "((void *)&am_crunch_sub_binary_f32)" c-pointer)
               (foreign-value "((void *)&am_crunch_sub_binary_f64)" c-pointer))
     (list 'mul (foreign-value "((void *)&am_crunch_mul_binary_f32)" c-pointer)
               (foreign-value "((void *)&am_crunch_mul_binary_f64)" c-pointer))
     (list 'div (foreign-value "((void *)&am_crunch_div_binary_f32)" c-pointer)
               (foreign-value "((void *)&am_crunch_div_binary_f64)" c-pointer))))

  ;; am_crunch_<op>_bcast_<t> computes rows [start, end) of the broadcast
  ;; binary op on a row-major output with n columns:
  ;;   out[i*n + j] = a[i*rsa + j*csa] op b[i*rsb + j*csb]
  ;; where the steps follow from the operand modes ma and mb (0 full,
  ;; 1 per row, 2 per column, 3 scalar; see lookup-broadcast-kernel).
  ;; They follow the calling convention of crunch-dispatch4-f32/-f64 with
  ;; i0 = n, i1 = ma and i2 = mb.  

  (crunch
    (: (am_crunch_add_bcast_f32 integer integer integer integer integer
                                 f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_add_bcast_f32 start end n ma mb scal a b out unused)
      (let ((rsa (if (= ma 0) n (if (= ma 1) 1 0)))
            (csa (if (= ma 0) 1 (if (= ma 2) 1 0)))
            (rsb (if (= mb 0) n (if (= mb 1) 1 0)))
            (csb (if (= mb 0) 1 (if (= mb 2) 1 0))))
        (do ((i start (+ i 1))) ((= i end))
          (let ((o (* i n)) (ai (* i rsa)) (bi (* i rsb)))
            (do ((j 0 (+ j 1))) ((= j n))
              (f32vector-set! out (+ o j)
                               (+ (f32vector-ref a (+ ai (* j csa)))
                                  (f32vector-ref b (+ bi (* j csb)))))))))))

  (crunch
    (: (am_crunch_add_bcast_f64 integer integer integer integer integer
                                 f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_add_bcast_f64 start end n ma mb scal a b out unused)
      (let ((rsa (if (= ma 0) n (if (= ma 1) 1 0)))
            (csa (if (= ma 0) 1 (if (= ma 2) 1 0)))
            (rsb (if (= mb 0) n (if (= mb 1) 1 0)))
            (csb (if (= mb 0) 1 (if (= mb 2) 1 0))))
        (do ((i start (+ i 1))) ((= i end))
          (let ((o (* i n)) (ai (* i rsa)) (bi (* i rsb)))
            (do ((j 0 (+ j 1))) ((= j n))
              (f64vector-set! out (+ o j)
                               (+ (f64vector-ref a (+ ai (* j csa)))
                                  (f64vector-ref b (+ bi (* j csb)))))))))))

  (crunch
    (: (am_crunch_sub_bcast_f32 integer integer integer integer integer
                                 f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_sub_bcast_f32 start end n ma mb scal a b out unused)
      (let ((rsa (if (= ma 0) n (if (= ma 1) 1 0)))
            (csa (if (= ma 0) 1 (if (= ma 2) 1 0)))
            (rsb (if (= mb 0) n (if (= mb 1) 1 0)))
            (csb (if (= mb 0) 1 (if (= mb 2) 1 0))))
        (do ((i start (+ i 1))) ((= i end))
          (let ((o (* i n)) (ai (* i rsa)) (bi (* i rsb)))
            (do ((j 0 (+ j 1))) ((= j n))
              (f32vector-set! out (+ o j)
                               (- (f32vector-ref a (+ ai (* j csa)))
                                  (f32vector-ref b (+ bi (* j csb)))))))))))

  (crunch
    (: (am_crunch_sub_bcast_f64 integer integer integer integer integer
                                 f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_sub_bcast_f64 start end n ma mb scal a b out unused)
      (let ((rsa (if (= ma 0) n (if (= ma 1) 1 0)))
            (csa (if (= ma 0) 1 (if (= ma 2) 1 0)))
            (rsb (if (= mb 0) n (if (= mb 1) 1 0)))
            (csb (if (= mb 0) 1 (if (= mb 2) 1 0))))
        (do ((i start (+ i 1))) ((= i end))
          (let ((o (* i n)) (ai (* i rsa)) (bi (* i rsb)))
            (do ((j 0 (+ j 1))) ((= j n))
              (f64vector-set! out (+ o j)
                               (- (f64vector-ref a (+ ai (* j csa)))
                                  (f64vector-ref b (+ bi (* j csb)))))))))))

  (crunch
    (: (am_crunch_mul_bcast_f32 integer integer integer integer integer
                                 f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_mul_bcast_f32 start end n ma mb scal a b out unused)
      (let ((rsa (if (= ma 0) n (if (= ma 1) 1 0)))
            (csa (if (= ma 0) 1 (if (= ma 2) 1 0)))
            (rsb (if (= mb 0) n (if (= mb 1) 1 0)))
            (csb (if (= mb 0) 1 (if (= mb 2) 1 0))))
        (do ((i start (+ i 1))) ((= i end))
          (let ((o (* i n)) (ai (* i rsa)) (bi (* i rsb)))
            (do ((j 0 (+ j 1))) ((= j n))
              (f32vector-set! out (+ o j)
                               (* (f32vector-ref a (+ ai (* j csa)))
                                  (f32vector-ref b (+ bi (* j csb)))))))))))

  (crunch
    (: (am_crunch_mul_bcast_f64 integer integer integer integer integer
                                 f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_mul_bcast_f64 start end n ma mb scal a b out unused)
      (let ((rsa (if (= ma 0) n (if (= ma 1) 1 0)))
            (csa (if (= ma 0) 1 (if (= ma 2) 1 0)))
            (rsb (if (= mb 0) n (if (= mb 1) 1 0)))
            (csb (if (= mb 0) 1 (if (= mb 2) 1 0))))
        (do ((i start (+ i 1))) ((= i end))
          (let ((o (* i n)) (ai (* i rsa)) (bi (* i rsb)))
            (do ((j 0 (+ j 1))) ((= j n))
              (f64vector-set! out (+ o j)
                               (* (f64vector-ref a (+ ai (* j csa)))
                                  (f64vector-ref b (+ bi (* j csb)))))))))))

  (crunch
    (: (am_crunch_div_bcast_f32 integer integer integer integer integer
                                 f64vector f32vector f32vector f32vector f32vector) void)
    (define (am_crunch_div_bcast_f32 start end n ma mb scal a b out unused)
      (let ((rsa (if (= ma 0) n (if (= ma 1) 1 0)))
            (csa (if (= ma 0) 1 (if (= ma 2) 1 0)))
            (rsb (if (= mb 0) n (if (= mb 1) 1 0)))
            (csb (if (= mb 0) 1 (if (= mb 2) 1 0))))
        (do ((i start (+ i 1))) ((= i end))
          (let ((o (* i n)) (ai (* i rsa)) (bi (* i rsb)))
            (do ((j 0 (+ j 1))) ((= j n))
              (f32vector-set! out (+ o j)
                               (/ (f32vector-ref a (+ ai (* j csa)))
                                  (f32vector-ref b (+ bi (* j csb)))))))))))

  (crunch
    (: (am_crunch_div_bcast_f64 integer integer integer integer integer
                                 f64vector f64vector f64vector f64vector f64vector) void)
    (define (am_crunch_div_bcast_f64 start end n ma mb scal a b out unused)
      (let ((rsa (if (= ma 0) n (if (= ma 1) 1 0)))
            (csa (if (= ma 0) 1 (if (= ma 2) 1 0)))
            (rsb (if (= mb 0) n (if (= mb 1) 1 0)))
            (csb (if (= mb 0) 1 (if (= mb 2) 1 0))))
        (do ((i start (+ i 1))) ((= i end))
          (let ((o (* i n)) (ai (* i rsa)) (bi (* i rsb)))
            (do ((j 0 (+ j 1))) ((= j n))
              (f64vector-set! out (+ o j)
                               (/ (f64vector-ref a (+ ai (* j csa)))
                                  (f64vector-ref b (+ bi (* j csb)))))))))))

  (define %crunch-broadcast-table
    ;; (op f32-chunk-address f64-chunk-address)
    (list
     (list 'add (foreign-value "((void *)&am_crunch_add_bcast_f32)" c-pointer)
               (foreign-value "((void *)&am_crunch_add_bcast_f64)" c-pointer))
     (list 'sub (foreign-value "((void *)&am_crunch_sub_bcast_f32)" c-pointer)
               (foreign-value "((void *)&am_crunch_sub_bcast_f64)" c-pointer))
     (list 'mul (foreign-value "((void *)&am_crunch_mul_bcast_f32)" c-pointer)
               (foreign-value "((void *)&am_crunch_mul_bcast_f64)" c-pointer))
     (list 'div (foreign-value "((void *)&am_crunch_div_bcast_f32)" c-pointer)
               (foreign-value "((void *)&am_crunch_div_bcast_f64)" c-pointer))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Reduction kernels
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; %k-<rop>-axis<a>-<t> reduces the row-major rows x cols array src over
  ;; axis a (0 or 1) into out, rounding exactly as the Scheme fast path of
  ;; execute-reduction-morphism in array-morphisms does:
  ;;   * over axis 0, each running result is stored in out after every
  ;;     step, so f32 sums round at every addition;
  ;;   * over axis 1, the running result is kept in double precision and
  ;;     stored once per row;
  ;;   * mean divides the sum by the number of reduced elements at the end;
  ;;   * max and min start from -inf.0 and +inf.0 and use strict
  ;;     comparisons, so a NaN never replaces the running result.

  (crunch
    (: (%k-sum-axis0-f32 integer integer f32vector f32vector) void)
    (define (%k-sum-axis0-f32 rows cols src out)
      (do ((n 0 (+ n 1))) ((= n cols))
        (f32vector-set! out n 0.0))
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols)))
          (do ((n 0 (+ n 1))) ((= n cols))
            (f32vector-set! out n (+ (f32vector-ref out n) (f32vector-ref src (+ base n)))))))))

  (crunch
    (: (%k-sum-axis0-f64 integer integer f64vector f64vector) void)
    (define (%k-sum-axis0-f64 rows cols src out)
      (do ((n 0 (+ n 1))) ((= n cols))
        (f64vector-set! out n 0.0))
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols)))
          (do ((n 0 (+ n 1))) ((= n cols))
            (f64vector-set! out n (+ (f64vector-ref out n) (f64vector-ref src (+ base n)))))))))

  (crunch
    (: (%k-sum-axis1-f32 integer integer f32vector f32vector) void)
    (define (%k-sum-axis1-f32 rows cols src out)
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols))
              (acc 0.0))
          (do ((n 0 (+ n 1))) ((= n cols))
            (set! acc (+ acc (f32vector-ref src (+ base n)))))
          (f32vector-set! out m acc)))))

  (crunch
    (: (%k-sum-axis1-f64 integer integer f64vector f64vector) void)
    (define (%k-sum-axis1-f64 rows cols src out)
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols))
              (acc 0.0))
          (do ((n 0 (+ n 1))) ((= n cols))
            (set! acc (+ acc (f64vector-ref src (+ base n)))))
          (f64vector-set! out m acc)))))

  (crunch
    (: (%k-mean-axis0-f32 integer integer f32vector f32vector) void)
    (define (%k-mean-axis0-f32 rows cols src out)
      (do ((n 0 (+ n 1))) ((= n cols))
        (f32vector-set! out n 0.0))
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols)))
          (do ((n 0 (+ n 1))) ((= n cols))
            (f32vector-set! out n (+ (f32vector-ref out n) (f32vector-ref src (+ base n)))))))
      (do ((n 0 (+ n 1))) ((= n cols))
        (f32vector-set! out n (/ (f32vector-ref out n) (exact->inexact rows))))))

  (crunch
    (: (%k-mean-axis0-f64 integer integer f64vector f64vector) void)
    (define (%k-mean-axis0-f64 rows cols src out)
      (do ((n 0 (+ n 1))) ((= n cols))
        (f64vector-set! out n 0.0))
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols)))
          (do ((n 0 (+ n 1))) ((= n cols))
            (f64vector-set! out n (+ (f64vector-ref out n) (f64vector-ref src (+ base n)))))))
      (do ((n 0 (+ n 1))) ((= n cols))
        (f64vector-set! out n (/ (f64vector-ref out n) (exact->inexact rows))))))

  (crunch
    (: (%k-mean-axis1-f32 integer integer f32vector f32vector) void)
    (define (%k-mean-axis1-f32 rows cols src out)
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols))
              (acc 0.0))
          (do ((n 0 (+ n 1))) ((= n cols))
            (set! acc (+ acc (f32vector-ref src (+ base n)))))
          (f32vector-set! out m (/ acc (exact->inexact cols)))))))

  (crunch
    (: (%k-mean-axis1-f64 integer integer f64vector f64vector) void)
    (define (%k-mean-axis1-f64 rows cols src out)
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols))
              (acc 0.0))
          (do ((n 0 (+ n 1))) ((= n cols))
            (set! acc (+ acc (f64vector-ref src (+ base n)))))
          (f64vector-set! out m (/ acc (exact->inexact cols)))))))

  (crunch
    (: (%k-max-axis0-f32 integer integer f32vector f32vector) void)
    (define (%k-max-axis0-f32 rows cols src out)
      (do ((n 0 (+ n 1))) ((= n cols))
        (f32vector-set! out n -inf.0))
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols)))
          (do ((n 0 (+ n 1))) ((= n cols))
            (let ((v (f32vector-ref src (+ base n))))
                (if (> v (f32vector-ref out n)) (f32vector-set! out n v))))))))

  (crunch
    (: (%k-max-axis0-f64 integer integer f64vector f64vector) void)
    (define (%k-max-axis0-f64 rows cols src out)
      (do ((n 0 (+ n 1))) ((= n cols))
        (f64vector-set! out n -inf.0))
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols)))
          (do ((n 0 (+ n 1))) ((= n cols))
            (let ((v (f64vector-ref src (+ base n))))
                (if (> v (f64vector-ref out n)) (f64vector-set! out n v))))))))

  (crunch
    (: (%k-max-axis1-f32 integer integer f32vector f32vector) void)
    (define (%k-max-axis1-f32 rows cols src out)
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols))
              (acc -inf.0))
          (do ((n 0 (+ n 1))) ((= n cols))
            (let ((v (f32vector-ref src (+ base n))))
                (if (> v acc) (set! acc v))))
          (f32vector-set! out m acc)))))

  (crunch
    (: (%k-max-axis1-f64 integer integer f64vector f64vector) void)
    (define (%k-max-axis1-f64 rows cols src out)
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols))
              (acc -inf.0))
          (do ((n 0 (+ n 1))) ((= n cols))
            (let ((v (f64vector-ref src (+ base n))))
                (if (> v acc) (set! acc v))))
          (f64vector-set! out m acc)))))

  (crunch
    (: (%k-min-axis0-f32 integer integer f32vector f32vector) void)
    (define (%k-min-axis0-f32 rows cols src out)
      (do ((n 0 (+ n 1))) ((= n cols))
        (f32vector-set! out n +inf.0))
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols)))
          (do ((n 0 (+ n 1))) ((= n cols))
            (let ((v (f32vector-ref src (+ base n))))
                (if (< v (f32vector-ref out n)) (f32vector-set! out n v))))))))

  (crunch
    (: (%k-min-axis0-f64 integer integer f64vector f64vector) void)
    (define (%k-min-axis0-f64 rows cols src out)
      (do ((n 0 (+ n 1))) ((= n cols))
        (f64vector-set! out n +inf.0))
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols)))
          (do ((n 0 (+ n 1))) ((= n cols))
            (let ((v (f64vector-ref src (+ base n))))
                (if (< v (f64vector-ref out n)) (f64vector-set! out n v))))))))

  (crunch
    (: (%k-min-axis1-f32 integer integer f32vector f32vector) void)
    (define (%k-min-axis1-f32 rows cols src out)
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols))
              (acc +inf.0))
          (do ((n 0 (+ n 1))) ((= n cols))
            (let ((v (f32vector-ref src (+ base n))))
                (if (< v acc) (set! acc v))))
          (f32vector-set! out m acc)))))

  (crunch
    (: (%k-min-axis1-f64 integer integer f64vector f64vector) void)
    (define (%k-min-axis1-f64 rows cols src out)
      (do ((m 0 (+ m 1))) ((= m rows))
        (let ((base (* m cols))
              (acc +inf.0))
          (do ((n 0 (+ n 1))) ((= n cols))
            (let ((v (f64vector-ref src (+ base n))))
                (if (< v acc) (set! acc v))))
          (f64vector-set! out m acc)))))

  (define %crunch-reduction-table
    ;; (rop axis f32-kernel f64-kernel)
    (list
     (list 'sum 0 %k-sum-axis0-f32 %k-sum-axis0-f64)
     (list 'sum 1 %k-sum-axis1-f32 %k-sum-axis1-f64)
     (list 'mean 0 %k-mean-axis0-f32 %k-mean-axis0-f64)
     (list 'mean 1 %k-mean-axis1-f32 %k-mean-axis1-f64)
     (list 'max 0 %k-max-axis0-f32 %k-max-axis0-f64)
     (list 'max 1 %k-max-axis1-f32 %k-max-axis1-f64)
     (list 'min 0 %k-min-axis0-f32 %k-min-axis0-f64)
     (list 'min 1 %k-min-axis1-f32 %k-min-axis1-f64)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Strided copy kernels
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; %k-copy4-<t> copies the n0 x n1 x n2 x n3 array whose element (i,j,k,l)
  ;; is src[off + i*s0 + j*s1 + k*s2 + l*s3] into dst in row-major order.

  (crunch
    (: (%k-copy4-f32 f32vector integer integer integer integer integer
                     integer integer integer integer f32vector) void)
    (define (%k-copy4-f32 src off n0 n1 n2 n3 s0 s1 s2 s3 dst)
      (let ((d 0))
        (do ((i 0 (+ i 1))) ((= i n0))
          (do ((j 0 (+ j 1))) ((= j n1))
            (do ((k 0 (+ k 1))) ((= k n2))
              (let ((base (+ off (+ (* i s0) (+ (* j s1) (* k s2))))))
                (do ((l 0 (+ l 1))) ((= l n3))
                  (f32vector-set! dst d (f32vector-ref src (+ base (* l s3))))
                  (set! d (+ d 1))))))))))

  (crunch
    (: (%k-copy4-f64 f64vector integer integer integer integer integer
                     integer integer integer integer f64vector) void)
    (define (%k-copy4-f64 src off n0 n1 n2 n3 s0 s1 s2 s3 dst)
      (let ((d 0))
        (do ((i 0 (+ i 1))) ((= i n0))
          (do ((j 0 (+ j 1))) ((= j n1))
            (do ((k 0 (+ k 1))) ((= k n2))
              (let ((base (+ off (+ (* i s0) (+ (* j s1) (* k s2))))))
                (do ((l 0 (+ l 1))) ((= l n3))
                  (f64vector-set! dst d (f64vector-ref src (+ base (* l s3))))
                  (set! d (+ d 1))))))))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Backend construction
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; Wrap a raw kernel so that a size larger than either vector raises a
  ;; Scheme error; the raw kernels do not check their indices.
  (define (checked-kernel who kernel vec-length)
    (lambda (n in out)
      (unless (and (fixnum? n) (>= n 0)
                   (<= n (vec-length in))
                   (<= n (vec-length out)))
        (error who "size exceeds vector length" n))
      (kernel n in out)))

  ;; Scalar-parameter vector for the dispatched kernels that take none.
  (define no-scalars (make-f64vector 1 0.0))

  ;; Adds the binary, reduction and strided-copy kernels to be and
  ;; registers the binary op names.  Binary kernels run on up to
  ;; (threads) threads, with at least (min-chunk) elements each.
  (define (add-extended-kernels! be threads min-chunk)
    (for-each
     (lambda (entry)
       (let ((op (car entry)))
         (register-binary-op! op)
         (for-each
          (lambda (dtype address vec-length dispatch)
            (activation-backend-add-binary-kernel!
             be op dtype
             (lambda (n a b out)
               (unless (and (fixnum? n) (>= n 0) (<= n (vec-length a))
                            (<= n (vec-length b)) (<= n (vec-length out)))
                 (error op "size exceeds vector length" n))
               (dispatch n 0 0 0 no-scalars a b out out address (threads) (min-chunk)))))
          '(f32 f64)
          (list (cadr entry) (caddr entry))
          (list f32vector-length f64vector-length)
          (list crunch-dispatch4-f32 crunch-dispatch4-f64))))
     %crunch-binary-table)
    ;; Broadcast kernels split the rows across threads; min-chunk counts
    ;; elements, so each thread gets at least min-chunk / cols rows.
    (for-each
     (lambda (entry)
       (let ((op (car entry)))
         (for-each
          (lambda (dtype address vec-length dispatch)
            (activation-backend-add-broadcast-kernel!
             be op dtype
             (lambda (rows cols a ma b mb out)
               (let ((need (lambda (mode)
                             (case mode
                               ((0) (* rows cols)) ((1) rows) ((2) cols) ((3) 1)
                               (else (error op "invalid broadcast mode" mode))))))
                 (unless (and (fixnum? rows) (fixnum? cols) (> rows 0) (> cols 0)
                              (<= (* rows cols) (vec-length out))
                              (<= (need ma) (vec-length a))
                              (<= (need mb) (vec-length b)))
                   (error op "broadcast operands do not fit their vectors" rows cols ma mb)))
               (dispatch rows cols ma mb no-scalars a b out out address (threads)
                         (max 1 (quotient (min-chunk) cols))))))
          '(f32 f64)
          (list (cadr entry) (caddr entry))
          (list f32vector-length f64vector-length)
          (list crunch-dispatch4-f32 crunch-dispatch4-f64))))
     %crunch-broadcast-table)
    (for-each
     (lambda (entry)
       (let ((rop (car entry)) (axis (cadr entry)))
         (for-each
          (lambda (dtype kernel vec-length)
            (activation-backend-add-reduction-kernel!
             be rop axis dtype
             (lambda (rows cols src out)
               (unless (and (fixnum? rows) (fixnum? cols) (>= rows 0) (>= cols 0)
                            (<= (* rows cols) (vec-length src))
                            (<= (if (= axis 0) cols rows) (vec-length out)))
                 (error rop "array does not fit its vectors" rows cols))
               (kernel rows cols src out))))
          '(f32 f64)
          (list (caddr entry) (cadddr entry))
          (list f32vector-length f64vector-length))))
     %crunch-reduction-table)
    (for-each
     (lambda (dtype kernel vec-length)
       (activation-backend-add-copy-kernel! be dtype (checked-copy kernel vec-length)))
     '(f32 f64)
     (list %k-copy4-f32 %k-copy4-f64)
     (list f32vector-length f64vector-length)))

  ;; Wrap a rank-4 copy kernel as a copy of rank at most 4: the shape is
  ;; padded at the front with extents of 1, and every index the copy will
  ;; read or write is checked first.
  (define (checked-copy kernel vec-length)
    (lambda (src off shape strides dst)
      (let* ((rank (vector-length shape))
             (pad  (- 4 rank))
             (dim  (lambda (v k fill) (if (< k pad) fill (vector-ref v (- k pad)))))
             (n    (let loop ((k 0) (p 1)) (if (= k rank) p (loop (+ k 1) (* p (vector-ref shape k))))))
             (hi   (let loop ((k 0) (h off))
                     (if (= k rank)
                         h
                         (loop (+ k 1) (+ h (* (max 0 (- (vector-ref shape k) 1))
                                                (vector-ref strides k))))))))
        (unless (and (<= rank 4) (>= off 0) (<= n (vec-length dst))
                     (let loop ((k 0)) (or (= k rank) (and (>= (vector-ref strides k) 0) (loop (+ k 1)))))
                     (or (= n 0) (< hi (vec-length src))))
          (error 'crunch-copy "array does not fit its vectors" shape strides off))
        (kernel src off
                (dim shape 0 1) (dim shape 1 1) (dim shape 2 1) (dim shape 3 1)
                (dim strides 0 0) (dim strides 1 0) (dim strides 2 0) (dim strides 3 0)
                dst))))

  (define (crunch-activation-ops)
    "List of the activation ops this backend has kernels for."
    (map car %crunch-activation-table))

  (define (make-crunch-activation-backend)
    "Construct an activation backend holding the crunch-compiled f32 and f64
    kernels of every op in the activation table, and register each op name
    with register-activation-op!.  The backend also holds the binary
    kernels (add, sub, mul, div, whose names it registers with
    register-binary-op!), the axis-0 and axis-1 reduction kernels (sum,
    mean, max, min) and a strided copy kernel.

    Usage:
      (import array-morphisms-crunch-activations)
      (register-activation-backend! (make-crunch-activation-backend))"
    (let ((be (make-activation-backend 'crunch)))
      (for-each
       (lambda (entry)
         (let ((op (car entry)))
           (register-activation-op! op)
           (activation-backend-add-kernel!
            be op 'f32 (checked-kernel op (cadr entry) f32vector-length))
           (activation-backend-add-kernel!
            be op 'f64 (checked-kernel op (caddr entry) f64vector-length))))
       %crunch-activation-table)
      (add-extended-kernels! be (lambda () 1) crunch-thread-min-chunk)
      be))

  (define (crunch-activation-chunk-kernel op dtype)
    "C address of the chunk kernel of op for dtype ('f32 or 'f64), for use
    with crunch-dispatch-f32 and crunch-dispatch-f64, or #f if the table
    has no entry for op."
    (let ((entry (assq op %crunch-activation-table)))
      (and entry
           (case dtype
             ((f32) (list-ref entry 3))
             ((f64) (list-ref entry 4))
             (else (error 'crunch-activation-chunk-kernel "unsupported dtype" dtype))))))

  (define (make-crunch-threaded-activation-backend)
    "Construct an activation backend, named crunch-threaded, whose kernels
    split each array across threads with crunch-dispatch-f32 and
    crunch-dispatch-f64, and register each op name with
    register-activation-op!.  Its binary kernels are split across threads in
    the same way; its reduction and copy kernels are those of
    make-crunch-activation-backend.  Every call uses the values of
    crunch-thread-count and crunch-thread-min-chunk current at that call.
    The results are identical to those of make-crunch-activation-backend.

    Usage:
      (import array-morphisms-crunch-activations)
      (register-activation-backend! (make-crunch-threaded-activation-backend))"
    (let ((be (make-activation-backend 'crunch-threaded)))
      (for-each
       (lambda (entry)
         (let ((op        (car entry))
               (chunk-f32 (list-ref entry 3))
               (chunk-f64 (list-ref entry 4)))
           (register-activation-op! op)
           (activation-backend-add-kernel!
            be op 'f32 (lambda (n in out) (crunch-dispatch-f32 n in out chunk-f32)))
           (activation-backend-add-kernel!
            be op 'f64 (lambda (n in out) (crunch-dispatch-f64 n in out chunk-f64)))))
       %crunch-activation-table)
      (add-extended-kernels! be crunch-thread-count crunch-thread-min-chunk)
      be))

) ;; end module array-morphisms-crunch-activations
