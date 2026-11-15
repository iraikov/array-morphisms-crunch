;;; crunch-codegen.scm
;;;
;;; A segop backend of array-morphisms that generates a crunch kernel for
;;; each segop it takes, compiles the kernels of a replay plan together
;;; when the plan is compiled, and runs them on the thread pool of
;;; array-morphisms-crunch-threads.
;;;
;;; The backend is a codegen backend (see segop.scm in array-morphisms).
;;; It takes the seg-maps whose operands are read and written through
;;; affine maps (identities, broadcasts, permutations and views) and whose
;;; bodies the typed translator of crunch-translate.scm accepts, with
;;; operands of dtypes f32, f64, s32, s64 and u32, mixed as the bodies
;;; need.  Each kernel is fixed in the rank of its space, its bodies and
;;; the dtypes of its operands; the extents, offsets and strides are passed
;;; when it runs, so one kernel serves every batch size and layout.
;;;
;;; While a plan is compiled the backend gives each segop it takes an
;;; executor cell holding the executor that the backends after it would
;;; use.  A plan finalizer then writes one module of crunch Scheme holding
;;; the kernels the process does not have yet, compiles it with csc, loads
;;; it and puts the kernels' executors in the cells.  When csc is missing
;;; or fails, the cells keep their host executors and a warning is
;;; written.  Compiled modules are kept in a cache directory, under a name
;;; derived from their source, so a later process loads them instead of
;;; compiling them again.
;;;
;;; The backend is used only while the parameter crunch-codegen is true;
;;; its default is true unless the environment variable AM_CRUNCH_CODEGEN
;;; is set to "" or "0".  It reports to the fusion pass the
;;; fused segops it can generate, so that the default fusion policy makes
;;; them, for the fusion classes listed in crunch-codegen-fusions.
;;;
;;;   (import array-morphisms-crunch-codegen)
;;;   (register-crunch-codegen-backend!)

(module array-morphisms-crunch-codegen

  (crunch-codegen
   crunch-codegen-cache-directory
   crunch-codegen-csc
   crunch-codegen-threads
   crunch-codegen-fusions
   register-crunch-codegen-backend!
   unregister-crunch-codegen-backend!
   crunch-codegen-native?
   crunch-codegen-stats
   crunch-codegen-forget-kernels!
   crunch-codegen-install-module!)

  (import scheme (chicken base) (chicken foreign) (chicken format) (chicken string)
          (chicken file) (chicken pathname) (chicken process) (chicken process-context)
          (chicken time) (chicken condition) (chicken number-vector) (chicken port)
          (chicken bitwise) (chicken pretty-print) (chicken fixnum)
          (only (chicken process-context posix) current-process-id)
          (only (scheme base) make-parameter exact-integer? vector-append)
          (only (chicken io) read-string)
          (only (chicken platform) chicken-version)
          (only (scheme time) current-jiffy jiffies-per-second)
          (only (chicken memory) pointer?)
          (only (chicken sort) sort)
          (only srfi-1 every any iota filter list-index delete-duplicates)
          (only srfi-69 make-hash-table hash-table-ref/default hash-table-set!)
          datatype
          array-morphisms-core
          array-morphisms-activation-exec
          array-morphisms-segop
          (only array-morphisms-scalar-ops scalar-add)
          (only array-morphisms-blas-exec execute-gemm-plan/into!)
          (only array-morphisms-ssa standard-gemm)
          array-morphisms-crunch-threads
          array-morphisms-crunch-translate)
  (import-for-syntax scheme (chicken base) (chicken io))

  ;;;; ============================================================
  ;;;; Settings
  ;;;; ============================================================

  ;; True unless the environment variable name is set to "" or "0".
  (define (env-flag? name)
    (let ((v (get-environment-variable name)))
      (not (and v (member v '("" "0")) #t))))

  ;; True while the backend takes segops and answers native queries.
  (define crunch-codegen (make-parameter (env-flag? "AM_CRUNCH_CODEGEN")
                                         (lambda (x) (and x #t))))

  (define (default-cache-directory)
    (or (get-environment-variable "AM_CRUNCH_CODEGEN_CACHE")
        (let ((xdg (get-environment-variable "XDG_CACHE_HOME")))
          (and xdg (not (string=? xdg ""))
               (make-pathname (list xdg "array-morphisms") "kernels")))
        (let ((home (get-environment-variable "HOME")))
          (and home (make-pathname (list home ".cache" "array-morphisms") "kernels")))))

  ;; The directory holding compiled modules, or #f for none (modules are
  ;; then compiled in a temporary directory and not kept).
  (define crunch-codegen-cache-directory (make-parameter (default-cache-directory)))

  ;; The csc program that compiles generated modules.
  (define crunch-codegen-csc
    (make-parameter (make-pathname (foreign-value "C_INSTALL_BIN_HOME" c-string) "csc")))

  ;; The fusion classes (see segop templates in array-morphisms) whose
  ;; fused segops the backend reports as native, so that the default fusion
  ;; policy makes them: a list of map, kept, horizontal, redomap, epilogue
  ;; and gemm-epilogue (default: redomap and epilogue).  The segops of
  ;; the other classes are not fused under the default policy; under the
  ;; policy all they are fused and still run as generated kernels.
  (define all-fusion-classes '(map kept horizontal redomap epilogue gemm-epilogue))

  ;; The default: the classes named in the environment variable
  ;; AM_CRUNCH_CODEGEN_FUSIONS, separated by commas or spaces ("" for
  ;; none), or else the fusions of maps into reductions and of maps after
  ;; reductions.  Those fuse passes over the same data into one kernel.
  ;; A map fused into a map can take the producer of a GEMM's epilogue,
  ;; and the GEMM then loses that epilogue, so map fusions are left to
  ;; the catalogue unless asked for.
  (define default-classes '(redomap epilogue))

  (define (default-fusion-classes)
    (let ((v (get-environment-variable "AM_CRUNCH_CODEGEN_FUSIONS")))
      (if v
          (filter (lambda (c) (memq c all-fusion-classes))
                  (map string->symbol (string-split v ", ")))
          default-classes)))

  (define crunch-codegen-fusions
    (make-parameter (default-fusion-classes)
                    (lambda (x)
                      (unless (and (list? x) (every (lambda (c) (memq c all-fusion-classes)) x))
                        (error 'crunch-codegen-fusions "not a list of fusion classes" x))
                      x)))

  ;; The number of threads a kernel runs on, or #f to follow the active
  ;; activation backend: (crunch-thread-count) threads with the threaded
  ;; crunch backend, one thread otherwise.
  (define crunch-codegen-threads
    (make-parameter #f (lambda (x)
                         (unless (or (not x) (and (fixnum? x) (> x 0)))
                           (error 'crunch-codegen-threads "not #f or a positive fixnum" x))
                         x)))

  (define (kernel-threads)
    (or (crunch-codegen-threads)
        (let ((be (active-activation-backend)))
          (if (and be (eq? (activation-backend-name be) 'crunch-threaded))
              (crunch-thread-count)
              1))))

  ;; The options of csc for generated modules.  -ffp-contract=off keeps
  ;; the C compiler from fusing a multiplication and an addition into one
  ;; instruction that rounds once.
  (define csc-options
    '("-s" "-O2" "-d1" "-C" "-O3 -DNDEBUG -fno-semantic-interposition -ffp-contract=off"))

  ;; Changed whenever the generated code changes in a way the source text
  ;; alone does not show.
  (define format-version 1)

  ;;;; ============================================================
  ;;;; Files written beside generated modules
  ;;;; ============================================================

  ;; The text of a file of the egg's source directory, read when this
  ;; module is compiled.
  (define-syntax file-text
    (er-macro-transformer
     (lambda (form r c)
       (call-with-input-file (cadr form) (lambda (p) (read-string #f p))))))

  (define support-files
    (list (cons "crunch-numvector-fix.scm" (file-text "crunch-numvector-fix.scm"))
          (cons "crunch-dispatch-n.h" (file-text "crunch-dispatch-n.h"))))

  ;;;; ============================================================
  ;;;; Kernel specifications
  ;;;; ============================================================

  ;; A kernel to generate.
  ;;   kind:       map
  ;;   rank:       rank of the iteration space
  ;;   in-dtypes:  dtypes of the inputs, in the order of the canonical bodies
  ;;   out-dtypes: dtypes of the outputs
  ;;   bodies:     the canonical bodies, one per output
  ;;   key:        a string naming all of the above
  (define-record kspec kind rank in-dtypes out-dtypes bodies key)

  (define (kspec-key-string kind rank in-dtypes out-dtypes bodies)
    (with-output-to-string
      (lambda () (write (list kind rank in-dtypes out-dtypes bodies)))))

  (define (vector-type dt)
    (case dt
      ((f32) 'f32vector) ((f64) 'f64vector) ((s32) 's32vector)
      ((s64) 's64vector) ((u32) 'u32vector)
      (else (error 'crunch-codegen "unsupported dtype" dt))))

  (define (vector-ref-name dt) (symbol-append (vector-type dt) '-ref))
  (define (vector-set-name dt) (symbol-append (vector-type dt) '-set!))
  (define (c-vector-type dt) (string-append "crunch_" (symbol->string (vector-type dt))))

  (define (element-size dt)
    (case dt ((f32 s32 u32) 4) (else 8)))

  ;; The helper names of a module are prefixed with the module's id, so
  ;; that the C symbols of different modules never clash.  While the
  ;; source is built the id is the placeholder below, which is replaced
  ;; by the hash of the source.
  (define id-placeholder "IDPLACEHOLDER")

  (define (helper-name name)
    (string->symbol (string-append "am_g" id-placeholder "_" (symbol->string name))))

  (define (flag-name) (string-append "am_g" id-placeholder "_flag"))

  (define (sym . parts)
    (string->symbol (apply string-append (map (lambda (p) (if (number? p) (number->string p)
                                                             (if (symbol? p) (symbol->string p) p)))
                                              parts))))

  ;; The translation of the bodies of a seg-map kernel whose inputs have
  ;; in-dtypes and are read by (read k), or #f.
  (define (translate-map-bodies bodies in-dtypes read)
    (translate-typed-bodies
     bodies
     (lambda (k)
       (if (< k (length in-dtypes))
           (cons (read k) (codegen-dtype-type (list-ref in-dtypes k)))
           (cons 0 'i)))
     helper-name))

  ;; Body e, or #f, with each (opaque proc x ...) whose proc is one of the
  ;; Scheme procedures of the arithmetic ops written symbolically: scalar-add
  ;; as add, - as sub or negate, * as mul, / as div, and the generic + as
  ;; %gadd (see crunch-translate.scm).  Other opaque parts are kept, and the
  ;; translator rejects them.
  (define (symbolic-body e)
    (cond ((not (pair? e)) e)
          ((memq (car e) '(in reduced param idx)) e)
          ((and (eq? (car e) 'opaque) (pair? (cdr e)))
           (let ((p (cadr e)) (args (map symbolic-body (cddr e))))
             (cond ((and (eq? p scalar-add) (= (length args) 2)) (cons 'add args))
                   ((and (eq? p +) (= (length args) 2)) (cons '%gadd args))
                   ((and (eq? p -) (= (length args) 2)) (cons 'sub args))
                   ((and (eq? p -) (= (length args) 1)) (cons 'negate args))
                   ((and (eq? p *) (= (length args) 2)) (cons 'mul args))
                   ((and (eq? p /) (= (length args) 2)) (cons 'div args))
                   (else (cons 'opaque (cons p args))))))
          ((eq? (car e) 'cast) (list 'cast (cadr e) (symbolic-body (caddr e))))
          (else (cons (car e) (map symbolic-body (cdr e))))))

  ;; yes when a seg-map kernel with these bodies and dtypes can set the
  ;; failure flag, no when it cannot, and #f when the bodies are not
  ;; translated.
  (define (map-may-fail? bodies in-dts out-dts)
    (let ((tr (translate-map-bodies bodies in-dts (lambda (k) 0))))
      (and tr
           (if (or (caddr tr)
                   (any (lambda (dt atom)
                          (let-values (((form fails?) (typed-store-form dt atom helper-name)))
                            fails?))
                        out-dts (cadr tr)))
               'yes
               'no))))

  ;;;; ============================================================
  ;;;; Kernel source
  ;;;; ============================================================
  ;;;
  ;;; A seg-map kernel visits the points [start, end) of its space in
  ;;; row-major order.  ip holds the extents e_d of the space, then for each
  ;;; operand j (the inputs, then the outputs) its base offset b_j and its
  ;;; coefficients c_jd: the element of operand j at point (i_0 ...) is at
  ;;; b_j + sum of c_jd i_d.  The kernel decodes start into indices once,
  ;;; and then runs along the last dimension, stepping each operand's
  ;;; offset by its last coefficient; when every last coefficient is 1 the
  ;;; inner loop indexes the operands directly.

  ;; The sum of forms, as nested two-argument additions (crunch's + takes
  ;; two arguments).
  (define (sum-form forms)
    (if (null? (cdr forms))
        (car forms)
        (sum-form (cons `(+ ,(car forms) ,(cadr forms)) (cddr forms)))))

  (define (map-kernel-forms k spec)
    (let* ((r      (kspec-rank spec))
           (ins    (kspec-in-dtypes spec))
           (outs   (kspec-out-dtypes spec))
           (n      (length ins))
           (m      (length outs))
           (nops   (+ n m))
           (name   (sym "am_g" id-placeholder "_k" k))
           (v      (lambda (j) (sym "%v" j)))
           (e      (lambda (d) (sym "%e" d)))
           (b      (lambda (j) (sym "%b" j)))
           (c      (lambda (j d) (sym "%c" j "_" d)))
           (i      (lambda (d) (sym "%i" d)))
           (o      (lambda (j) (sym "%o" j)))
           (ops    (iota nops))
           (dims   (iota r))
           (dt-of  (lambda (j) (if (< j n) (list-ref ins j) (list-ref outs (- j n)))))
           (params (append (map (lambda (d) (list (e d) `(s64vector-ref ip ,d))) dims)
                           (apply append
                                  (map (lambda (j)
                                         (let ((at (+ r (* j (+ r 1)))))
                                           (cons (list (b j) `(s64vector-ref ip ,at))
                                                 (map (lambda (d) (list (c j d) `(s64vector-ref ip ,(+ at 1 d))))
                                                      dims))))
                                       ops))))
           ;; the statements computing one point, given the offset form of
           ;; each operand
           (point  (lambda (off)
                     (let ((tr (translate-map-bodies (kspec-bodies spec) ins
                                                     (lambda (j) `(,(vector-ref-name (dt-of j)) ,(v j) ,(off j))))))
                       `(let* ,(car tr)
                          ,@(map (lambda (j atom)
                                   (let-values (((form fails?) (typed-store-form (dt-of (+ n j)) atom helper-name)))
                                     `(,(vector-set-name (dt-of (+ n j))) ,(v (+ n j)) ,(off (+ n j)) ,form)))
                                 (iota m) (cadr tr))))))
           (sig    `(: (,name integer integer s64vector f64vector ,@(map (lambda (j) (vector-type (dt-of j))) ops))
                       void)))
      (if (= r 0)
          (list sig
                `(define (,name start end ip sc ,@(map v ops))
                   (let* ,(map (lambda (j) (list (b j) `(s64vector-ref ip ,j))) ops)
                     (when (< start end)
                       ,(point b)))))
          (let* ((last (- r 1))
                 (carry
                  ;; advance the indices past the end of dimension d
                  (let loop ((d last))
                    `(when (= ,(i d) ,(e d))
                       (set! ,(i d) 0)
                       ,@(map (lambda (j) `(set! ,(o j) (- ,(o j) (* ,(e d) ,(c j d))))) ops)
                       ,@(if (= d 0)
                             '()
                             `((set! ,(i (- d 1)) (+ ,(i (- d 1)) 1))
                               ,@(map (lambda (j) `(set! ,(o j) (+ ,(o j) ,(c j (- d 1))))) ops)
                               ,(loop (- d 1))))))))
            (list sig
                  `(define (,name start end ip sc ,@(map v ops))
                     (let* ,params
                       (let ((%p start) (%rem start)
                             ,@(map (lambda (d) (list (i d) 0)) dims)
                             ,@(map (lambda (j) (list (o j) 0)) ops))
                         ,@(apply append
                                  (map (lambda (d)
                                         `((set! ,(i d) (remainder %rem ,(e d)))
                                           (set! %rem (quotient %rem ,(e d)))))
                                       (reverse dims)))
                         ,@(map (lambda (j)
                                  `(set! ,(o j) ,(sum-form (cons (b j) (map (lambda (d) `(* ,(c j d) ,(i d))) dims)))))
                                ops)
                         (let ((%unit (and ,@(map (lambda (j) `(= ,(c j last) 1)) ops))))
                           (do () ((>= %p end))
                             (let ((%cnt (min (- ,(e last) ,(i last)) (- end %p))))
                               (if %unit
                                   (do ((%t 0 (+ %t 1))) ((= %t %cnt))
                                     ,(point (lambda (j) `(+ ,(o j) %t))))
                                   (do ((%t 0 (+ %t 1))) ((= %t %cnt))
                                     ,(point (lambda (j) `(+ ,(o j) (* %t ,(c j last)))))))
                               (set! %p (+ %p %cnt))
                               (set! ,(i last) (+ ,(i last) %cnt))
                               ,@(map (lambda (j) `(set! ,(o j) (+ ,(o j) (* %cnt ,(c j last))))) ops)
                               ,carry)))))))))))

  ;;; A seg-reduce kernel visits the kept points [start, end), the points of
  ;;; the first npar dimensions, in row-major order, one at a time or in
  ;;; tiles along the last kept dimension (see red-tile-size).  For each it sets the
  ;;; accumulators to the neutral elements, walks the reduced points in
  ;;; row-major order, computing at each the bodies of the results and of
  ;;; the map outputs once, storing the map outputs and combining each
  ;;; result's value into its accumulator, and then applies the finalizers
  ;;; and stores the results.  ip holds the extents and the operand forms as
  ;;; for a seg-map, then for each result its block size (for the blocked
  ;;; order) and its neutral element when the accumulator has an integer
  ;;; dtype; sc holds the neutral elements of the float accumulators.  The
  ;;; kept points are split across threads, and each kept point is computed
  ;;; by one thread in the order of the reference interpreter, so the
  ;;; results are the same for every thread count.

  ;; The parts of a seg-reduce kernel specification: its bodies field is
  ;; (nred results map-bodies), each result a list
  ;; (op acc-dtype order body finalize) with finalize #f for none.
  (define (red-nred spec) (car (kspec-bodies spec)))
  (define (red-results spec) (cadr (kspec-bodies spec)))
  (define (red-map-bodies spec) (caddr (kspec-bodies spec)))

  ;; The statements of one reduced point and of one kept point, and
  ;; whether they can set the failure flag, from:
  ;;   results, map-bodies: as in a seg-reduce kernel specification;
  ;;   in-dts, out-dts:     the dtypes of the inputs and of the outputs
  ;;                        (results, then map outputs);
  ;;   read, kept-read:     the forms reading input k at the reduced point
  ;;                        and at the kept point;
  ;;   out-off, kept-off:   the offset forms of output j at the reduced
  ;;                        point (map outputs) and at the kept point;
  ;;   acc, blk, cnt, bsz:  the names of result r's accumulator, block
  ;;                        accumulator, block count and block size;
  ;;   init:                the form of result r's neutral element.
  ;; Returns (point-form flush-forms final-forms fails?) or #f when some
  ;; body is not translated.
  (define (red-statements results map-bodies in-dts out-dts read kept-read out-off kept-off
                          acc blk cnt bsz init)
    (let* ((nres   (length results))
           (in-type (lambda (k) (codegen-dtype-type (list-ref in-dts k))))
           (input  (lambda (rd) (lambda (k) (if (< k (length in-dts)) (cons (rd k) (in-type k)) (cons 0 'i)))))
           (tr     (translate-typed-bodies (append (map (lambda (r) (list-ref r 3)) results) map-bodies)
                                           (input read) helper-name))
           (fails? #f))
      (define (note! f?) (when f? (set! fails? #t)))
      (define (acc-type r) (codegen-dtype-type (cadr (list-ref results r))))
      (define (combine r a v)
        (let-values (((form f?) (typed-combine-form (car (list-ref results r)) a v
                                                    (cadr (list-ref results r)) helper-name)))
          (note! f?)
          form))
      (define (store dt atom)
        (let-values (((form f?) (typed-store-form dt atom helper-name)))
          (note! f?)
          form))
      (and tr
           (let* ((atoms   (cadr tr))
                  (r-atoms (list-head* atoms nres))
                  (m-atoms (list-tail atoms nres))
                  (point
                   `(let* ,(car tr)
                      ,@(map (lambda (j atom)
                               (let ((dt (list-ref out-dts (+ nres j))))
                                 `(,(vector-set-name dt) ,(v-name (+ (length in-dts) nres j))
                                   ,(out-off (+ nres j)) ,(store dt atom))))
                             (iota (length m-atoms)) m-atoms)
                      ,@(map (lambda (r atom)
                               (let ((a (cons (acc r) (acc-type r))) (b (cons (blk r) (acc-type r))))
                                 (if (eq? (caddr (list-ref results r)) 'blocked)
                                     `(begin
                                        (set! ,(blk r) ,(combine r b atom))
                                        (set! ,(cnt r) (+ ,(cnt r) 1))
                                        (when (= ,(cnt r) ,(bsz r))
                                          (set! ,(acc r) ,(combine r a b))
                                          (set! ,(blk r) ,(init r))
                                          (set! ,(cnt r) 0)))
                                     `(set! ,(acc r) ,(combine r a atom)))))
                             (iota nres) r-atoms)))
                  (flush
                   (filter values
                           (map (lambda (r)
                                  (and (eq? (caddr (list-ref results r)) 'blocked)
                                       `(when (> ,(cnt r) 0)
                                          (set! ,(acc r) ,(combine r (cons (acc r) (acc-type r))
                                                                   (cons (blk r) (acc-type r)))))))
                                (iota nres))))
                  (finals
                   (map (lambda (r)
                          (let* ((res (list-ref results r))
                                 (a   (cons (acc r) (acc-type r)))
                                 (dt  (list-ref out-dts r))
                                 (ftr (and (list-ref res 4)
                                           (translate-typed-bodies (list (list-ref res 4)) (input kept-read)
                                                                   helper-name reduced: a))))
                            (cond ((not (list-ref res 4))
                                   `(,(vector-set-name dt) ,(v-name (+ (length in-dts) r)) ,(kept-off r)
                                     ,(store dt a)))
                                  (ftr
                                   (note! (caddr ftr))
                                   `(let* ,(car ftr)
                                      (,(vector-set-name dt) ,(v-name (+ (length in-dts) r)) ,(kept-off r)
                                       ,(store dt (car (cadr ftr))))))
                                  (else #f))))
                        (iota nres))))
             (note! (caddr tr))
             (and (every values finals)
                  (list point flush finals fails?))))))

  (define (list-head* lst n) (if (= n 0) '() (cons (car lst) (list-head* (cdr lst) (- n 1)))))

  (define (v-name j) (sym "%v" j))

  (define (red-kernel-forms k spec)
    (let* ((rank    (kspec-rank spec))
           (nred    (red-nred spec))
           (npar    (- rank nred))
           (results (red-results spec))
           (nres    (length results))
           (ins     (kspec-in-dtypes spec))
           (outs    (kspec-out-dtypes spec))
           (n       (length ins))
           (nops    (+ n (length outs)))
           (name    (sym "am_g" id-placeholder "_k" k))
           (e       (lambda (d) (sym "%e" d)))
           (b       (lambda (j) (sym "%b" j)))
           (c       (lambda (j d) (sym "%c" j "_" d)))
           (i       (lambda (d) (sym "%i" d)))
           (po      (lambda (j) (sym "%po" j)))
           (ops     (iota nops))
           (dims    (iota rank))
           (pdims   (iota npar))
           (rdims   (map (lambda (d) (+ npar d)) (iota nred)))
           (dt-of   (lambda (j) (if (< j n) (list-ref ins j) (list-ref outs (- j n)))))
           (rbase   (+ rank (* nops (+ rank 1))))
           (acc     (lambda (r) (sym "%acc" r)))
           (blk     (lambda (r) (sym "%blk" r)))
           (cnt     (lambda (r) (sym "%cnt" r)))
           (bsz     (lambda (r) (sym "%bs" r)))
           (init    (lambda (r)
                      (if (codegen-float-dtype? (cadr (list-ref results r)))
                          `(f64vector-ref sc ,r)
                          `(s64vector-ref ip ,(+ rbase (* 2 r) 1)))))
           ;; offset of operand j at the current reduced point
           (off     (lambda (j) (sum-form (cons (po j) (map (lambda (d) `(* ,(c j d) ,(i d))) rdims)))))
           (st      (red-statements results (red-map-bodies spec) ins outs
                                    (lambda (j) `(,(vector-ref-name (dt-of j)) ,(v-name j) ,(off j)))
                                    (lambda (j) `(,(vector-ref-name (dt-of j)) ,(v-name j) ,(po j)))
                                    (lambda (j) (off (+ n j)))
                                    (lambda (j) (po (+ n j)))
                                    acc blk cnt bsz init))
           (params  (append (map (lambda (d) (list (e d) `(s64vector-ref ip ,d))) dims)
                            (apply append
                                   (map (lambda (j)
                                          (let ((at (+ rank (* j (+ rank 1)))))
                                            (cons (list (b j) `(s64vector-ref ip ,at))
                                                  (map (lambda (d) (list (c j d) `(s64vector-ref ip ,(+ at 1 d))))
                                                       dims))))
                                        ops))
                            (map (lambda (r) (list (bsz r) `(s64vector-ref ip ,(+ rbase (* 2 r))))) (iota nres))))
           ;; the loops over the reduced dimensions around body
           (loops   (lambda (body)
                      (let loop ((ds rdims))
                        (if (null? ds)
                            body
                            `(do ((,(i (car ds)) 0 (+ ,(i (car ds)) 1))) ((= ,(i (car ds)) ,(e (car ds))))
                               ,(loop (cdr ds)))))))
           ;; the accumulator bindings of one kept point
           (acc-bindings (lambda (acc blk cnt)
                           (apply append
                                  (map (lambda (r) (list (list (acc r) (init r)) (list (blk r) (init r))
                                                         (list (cnt r) 0)))
                                       (iota nres)))))
           (single  `(let* ,(acc-bindings acc blk cnt)
                       ,(loops (car st))
                       ,@(cadr st)
                       ,@(caddr st)))
           (tile    (and (> npar 0) (red-tile-forms spec n nops npar rdims ins outs dt-of c i po
                                                    init))))
      (list
       `(: (,name integer integer s64vector f64vector ,@(map (lambda (j) (vector-type (dt-of j))) ops)) void)
       `(define (,name start end ip sc ,@(map v-name ops))
          (let* ,params
            (let ((%step 1))
              (do ((%p start (+ %p %step))) ((>= %p end))
                ;; The kept indices are computed without assignment: crunch
                ;; loses the initial value of a variable that is bound to
                ;; another variable and assigned later.
                (let* (,@(apply append
                                (map (lambda (d)
                                       (let ((q (if (= d (- npar 1)) '%p (sym "%q" (+ d 1)))))
                                         `((,(i d) (remainder ,q ,(e d)))
                                           (,(sym "%q" d) (quotient ,q ,(e d))))))
                                     (reverse pdims))))
                  (let* ,(map (lambda (j)
                                (list (po j) (sum-form (cons (b j) (map (lambda (d) `(* ,(c j d) ,(i d))) pdims)))))
                              ops)
                    ,(if tile
                         `(if (and (<= (+ %p ,red-tile-size) end)
                                   (<= (+ ,(i (- npar 1)) ,red-tile-size) ,(e (- npar 1))))
                              (let* ,(caddr tile)
                                ,(loops (car tile))
                                ,@(cadr tile)
                                (set! %step ,red-tile-size))
                              (begin ,single
                                     (set! %step 1)))
                         single))))))))))

  ;;; A seg-reduce kernel computes red-tile-size consecutive kept points
  ;;; together when they lie along the last kept dimension: it walks the
  ;;; reduced points once, and at each one computes every kept point of
  ;;; the tile, each with its own accumulators.  Each accumulator combines
  ;;; the same values in the same order as when the kept points are
  ;;; computed one at a time, so the results are the same.  When the
  ;;; inputs are contiguous along the kept dimension, as in a sum over the
  ;;; rows of a matrix, the tile reads neighbouring elements at each
  ;;; reduced point.
  (define red-tile-size 8)

  ;; The statements of a tile of kept points, in the scope of the kept
  ;; offsets (po j) of its first point: a list
  ;; (point-form final-forms bindings), where point-form computes every
  ;; member of the tile at one reduced point, final-forms flush and store
  ;; the results of every member, and bindings bind the offsets and the
  ;; accumulators of the members; or #f when some body is not translated.
  (define (red-tile-forms spec n nops npar rdims ins outs dt-of c i po init)
    (let* ((results (red-results spec))
           (nres    (length results))
           (last    (- npar 1))
           (ops     (iota nops))
           (members (iota red-tile-size))
           (pt      (lambda (j t) (sym "%po" j "t" t)))
           (ro      (lambda (j) (sym "%ro" j)))
           (names   (lambda (prefix t) (lambda (r) (sym prefix r "t" t))))
           (sts     (map (lambda (t)
                           (red-statements
                            results (red-map-bodies spec) ins outs
                            (lambda (j) `(,(vector-ref-name (dt-of j)) ,(v-name j) (+ ,(pt j t) ,(ro j))))
                            (lambda (j) `(,(vector-ref-name (dt-of j)) ,(v-name j) ,(pt j t)))
                            (lambda (j) `(+ ,(pt (+ n j) t) ,(ro (+ n j))))
                            (lambda (j) (pt (+ n j) t))
                            (names "%acc" t) (names "%blk" t) (names "%cnt" t)
                            (lambda (r) (sym "%bs" r))
                            init))
                         members)))
      (and (every values sts)
           (list
            ;; at one reduced point: the reduced offsets, then each member
            `(let* ,(map (lambda (j) (list (ro j) (sum-form (cons 0 (map (lambda (d) `(* ,(c j d) ,(i d))) rdims)))))
                         ops)
               ,@(map car sts))
            ;; after the reduced points: flushes and stores of each member
            (apply append (map (lambda (st) (append (cadr st) (caddr st))) sts))
            ;; the bindings of the member offsets and accumulators
            (append
             (apply append
                    (map (lambda (t)
                           (map (lambda (j) (list (pt j t) `(+ ,(po j) (* ,t ,(c j last))))) ops))
                         members))
             (apply append
                    (map (lambda (t)
                           (apply append
                                  (map (lambda (r)
                                         (list (list ((names "%acc" t) r) (init r))
                                               (list ((names "%blk" t) r) (init r))
                                               (list ((names "%cnt" t) r) 0)))
                                       (iota nres))))
                         members)))))))

  ;;; A seg-scan kernel scans the lines of its space, one parallel point
  ;;; per job, as execute-scan-morphism in array-morphisms does.  It keeps
  ;;; no running value: at each position it reads the value it stored at
  ;;; the position before, which is the value so far rounded to the
  ;;; output's dtype, and combines it with the new element.  An inclusive
  ;;; scan stores at position 0 the element (or the seed combined with it),
  ;;; and an exclusive scan stores there the seed (or the neutral element)
  ;;; and at position r + 1 what the inclusive scan stores at r.
  ;;;
  ;;; The bodies of a scan kernel specification are a list
  ;;; (value seed step neutral exclusive? aux):
  ;;;   value:      the scan's body, over the inputs;
  ;;;   seed:       #f, or the index of the input read as the seed;
  ;;;   step:       (std op swapped?) for a combine op add, mul, max or
  ;;;               min, or (expr e) for a combine expression, e a body
  ;;;               whose (in 0) is the value so far and (in 1) the element;
  ;;;   neutral:    the neutral element, or #f;
  ;;;   exclusive?: #t for an exclusive scan;
  ;;;   aux:        the bodies of the extra values a combine expression
  ;;;               receives, read at the current position; its (in 2),
  ;;;               (in 3) ... in the step.

  (define (scan-step-desc spec swapped?)
    (let ((op (combine-spec-op spec)))
      (cond ((memq op '(add mul max min)) (list 'std op swapped?))
            ((combine-expression? op)
             (let* ((vars (cadr op))
                    (body (substitute-vars (caddr op)
                                           (append
                                            (if swapped?
                                                (list (cons (car vars) '(in 1)) (cons (cadr vars) '(in 0)))
                                                (list (cons (car vars) '(in 0)) (cons (cadr vars) '(in 1))))
                                            (map (lambda (v k) (cons v `(in ,(+ k 2))))
                                                 (cddr vars) (iota (length (cddr vars))))))))
               (and body (list 'expr body))))
            (else #f))))

  ;; e with each variable of the alist subst replaced by its body, or #f
  ;; when e binds variables itself.
  (define (substitute-vars e subst)
    (call-with-current-continuation
     (lambda (k)
       (let walk ((e e))
         (cond ((symbol? e) (let ((p (assq e subst))) (if p (cdr p) e)))
               ((not (pair? e)) e)
               ((eq? (car e) 'let*) (k #f))
               ((eq? (car e) 'cast) (list 'cast (cadr e) (walk (caddr e))))
               (else (cons (car e) (map walk (cdr e)))))))))

  ;; The statements of one parallel point of a scan kernel, as
  ;; (pre body end may-fail?), or #f when a body is not translated: pre
  ;; is a list of statements run first, body the statement run at each
  ;; position %r from 0 below end.
  ;;   in-dts, odt: the dtypes of the inputs and of the output;
  ;;   read:        (read k r) is the form reading input k at position r;
  ;;   out-off:     (out-off r) is the output's offset at position r;
  ;;   n:           the form of the scan length.
  (define (scan-statements bodies in-dts odt read out-off n)
    (let* ((value (list-ref bodies 0)) (seed (list-ref bodies 1)) (step (list-ref bodies 2))
           (neutral (list-ref bodies 3)) (excl? (list-ref bodies 4)) (aux (list-ref bodies 5))
           (aux-vars (map (lambda (j) (sym "%x" j)) (iota (length aux))))
           (otype (codegen-dtype-type odt))
           (oref (vector-ref-name odt)) (oset (vector-set-name odt))
           (ov (v-name (length in-dts)))
           (fails? #f))
      (define (note! f?) (when f? (set! fails? #t)))
      (define (store a)
        (let-values (((form f?) (typed-store-form odt a helper-name))) (note! f?) form))
      ;; (bindings . atom) of the step combining the pairs acc and v, with
      ;; the aux values as inputs 2, 3, ...
      (define (combine acc v)
        (if (eq? (car step) 'std)
            (let ((op (cadr step)) (sw? (caddr step)))
              (if (memq op '(max min))
                  (let-values (((form f?) (typed-combine-form op (if sw? v acc) (if sw? acc v) odt helper-name)))
                    (note! f?)
                    (cons '() (cons form otype)))
                  (let ((tr (translate-typed-bodies (list (if sw? `(,op (in 1) (in 0)) `(,op (in 0) (in 1))))
                                                    (lambda (k) (if (= k 0) acc v)) helper-name)))
                    (and tr (begin (note! (caddr tr)) (cons (car tr) (car (cadr tr))))))))
            (let ((tr (translate-typed-bodies (list (cadr step))
                                              (lambda (k) (cond ((= k 0) acc) ((= k 1) v)
                                                                (else (cons (list-ref aux-vars (- k 2))
                                                                            (list-ref atypes (- k 2))))))
                                              helper-name)))
              (and tr (begin (note! (caddr tr)) (cons (car tr) (car (cadr tr))))))))
      ;; the statement storing at position at the step of acc-form and %v
      (define (step-store acc-form at)
        (let ((c (combine (cons '%a otype) (cons '%v vtype))))
          (and c `(let* ((%a ,acc-form) ,@(car c)) (,oset ,ov ,(out-off at) ,(store (cdr c)))))))
      (define vtype #f)
      (define atypes '())
      (let ((vtr (translate-typed-bodies (cons value aux)
                                         (lambda (k) (cons (read k '%r) (codegen-dtype-type (list-ref in-dts k))))
                                         helper-name)))
        (and vtr
             (begin
               (note! (caddr vtr))
               (set! vtype (cdr (car (cadr vtr))))
               (set! atypes (map cdr (cdr (cadr vtr))))
               (let* ((seed-read (and seed (read seed 0)))
                      (at        (if excl? '(+ %r 1) '%r))
                      (first     (if seed
                                     (step-store seed-read at)
                                     `(,oset ,ov ,(out-off at) ,(store (cons '%v vtype)))))
                      (later     (step-store `(,oref ,ov ,(out-off (if excl? '%r '(- %r 1)))) at)))
                 (and first later
                      (list
                       (if excl?
                           `((when (> ,n 0)
                               (,oset ,ov ,(out-off 0)
                                      ,(if seed
                                           (store (cons seed-read otype))
                                           (store (cons neutral (if (exact-integer? neutral) 'i 'f)))))))
                           '())
                       `(let* (,@(car vtr) (%v ,(car (car (cadr vtr))))
                               ,@(map (lambda (x a) (list x (car a))) aux-vars (cdr (cadr vtr))))
                          (if (= %r 0) ,first ,later))
                       (if excl? `(- ,n 1) n)
                       fails?))))))))

  ;;; A seg-scan kernel scans scan-tile-size consecutive lines together
  ;;; when they lie along the last parallel dimension: at each position it
  ;;; computes the next value of every line of the tile.  Each line
  ;;; combines the same values in the same order as on its own, so the
  ;;; results are the same.  When the lines are the columns of a row-major
  ;;; array (a scan along its first axis), the tile reads and writes
  ;;; neighbouring elements at each position.
  (define scan-tile-size 8)

  ;; The statement scanning the lines of scan statements st (see
  ;; scan-statements) from position 0 on.
  (define (scan-loop sts)
    `(begin ,@(apply append (map car sts))
            (do ((%r 0 (+ %r 1))) ((>= %r ,(caddr (car sts))))
              ,@(map cadr sts))))

  (define (scan-kernel-forms k spec)
    (let* ((rank    (kspec-rank spec))
           (npar    (- rank 1))
           (ins     (kspec-in-dtypes spec))
           (odt     (car (kspec-out-dtypes spec)))
           (n       (length ins))
           (nops    (+ n 1))
           (name    (sym "am_g" id-placeholder "_k" k))
           (e       (lambda (d) (sym "%e" d)))
           (b       (lambda (j) (sym "%b" j)))
           (c       (lambda (j d) (sym "%c" j "_" d)))
           (i       (lambda (d) (sym "%i" d)))
           (po      (lambda (j) (sym "%po" j)))
           (ops     (iota nops))
           (dims    (iota rank))
           (pdims   (iota npar))
           (dt-of   (lambda (j) (if (< j n) (list-ref ins j) odt)))
           ;; the statements of the line whose operand offsets are (base j)
           (line    (lambda (base)
                      (scan-statements (kspec-bodies spec) ins odt
                                       (lambda (k r) `(,(vector-ref-name (list-ref ins k)) ,(v-name k)
                                                       (+ ,(base k) (* ,r ,(c k npar)))))
                                       (lambda (r) `(+ ,(base n) (* ,r ,(c n npar))))
                                       (e npar))))
           (single  (scan-loop (list (line po))))
           (last    (- npar 1))
           (pt      (lambda (j t) (sym "%po" j "t" t)))
           (tile    (and (> npar 0)
                         `(let* ,(apply append
                                        (map (lambda (t)
                                               (map (lambda (j) (list (pt j t) `(+ ,(po j) (* ,t ,(c j last)))))
                                                    ops))
                                             (iota scan-tile-size)))
                            ,(scan-loop (map (lambda (t) (line (lambda (j) (pt j t))))
                                             (iota scan-tile-size))))))
           (params  (append (map (lambda (d) (list (e d) `(s64vector-ref ip ,d))) dims)
                            (apply append
                                   (map (lambda (j)
                                          (let ((at (+ rank (* j (+ rank 1)))))
                                            (cons (list (b j) `(s64vector-ref ip ,at))
                                                  (map (lambda (d) (list (c j d) `(s64vector-ref ip ,(+ at 1 d))))
                                                       dims))))
                                        ops)))))
      (list
       `(: (,name integer integer s64vector f64vector ,@(map (lambda (j) (vector-type (dt-of j))) ops)) void)
       `(define (,name start end ip sc ,@(map v-name ops))
          (let* ,params
            (let ((%step 1))
              (do ((%p start (+ %p %step))) ((>= %p end))
                (let* (,@(apply append
                                (map (lambda (d)
                                       (let ((q (if (= d (- npar 1)) '%p (sym "%q" (+ d 1)))))
                                         `((,(i d) (remainder ,q ,(e d)))
                                           (,(sym "%q" d) (quotient ,q ,(e d))))))
                                     (reverse pdims)))
                       ,@(map (lambda (j)
                                (list (po j) (sum-form (cons (b j) (map (lambda (d) `(* ,(c j d) ,(i d))) pdims)))))
                              ops))
                  ,(if tile
                       `(if (and (<= (+ %p ,scan-tile-size) end)
                                 (<= (+ ,(i last) ,scan-tile-size) ,(e last)))
                            (begin ,tile (set! %step ,scan-tile-size))
                            (begin ,single (set! %step 1)))
                       single)))))))))

  (define (kernel-forms k spec)
    (case (kspec-kind spec)
      ((map) (map-kernel-forms k spec))
      ((red) (red-kernel-forms k spec))
      ((scan) (scan-kernel-forms k spec))
      (else (error 'crunch-codegen "unknown kernel kind" (kspec-kind spec)))))

  ;; The C run procedure of kernel k (see crunch-dispatch-n.h).
  (define (kernel-run-c k spec)
    (let* ((dts  (append (kspec-in-dtypes spec) (kspec-out-dtypes spec)))
           (name (string-append "am_g" id-placeholder "_k" (number->string k))))
      (string-append
       "static void " name "_run(void *p, long c) {\n"
       "  am_jobn *j = (am_jobn *)p + c;\n"
       "  " name "(j->start, j->end, j->ip, j->sc"
       (apply string-append
              (map (lambda (dt j) (string-append ", (" (c-vector-type dt) ")j->v[" (number->string j) "]"))
                   dts (iota (length dts))))
       ");\n}\n")))

  (define (module-source specs)
    "The source of a module holding the kernels of specs, with the id
    placeholder in place of the module's id."
    (let ((ks (iota (length specs))))
      (with-output-to-string
        (lambda ()
          (printf ";;; Generated by array-morphisms-crunch-codegen.\n")
          (printf ";;; format ~a, CHICKEN ~a, AM_CRUNCH_NATIVE=~a\n"
                  format-version (chicken-version)
                  (or (get-environment-variable "AM_CRUNCH_NATIVE") ""))
          (for-each (lambda (s) (printf ";;; kernel ~a\n" (kspec-key s))) specs)
          (pretty-print
           `(module ,(string->symbol (string-append "am-gen-" id-placeholder)) ()
              (import scheme (chicken base) (chicken foreign) (chicken number-vector) (chicken flonum)
                      crunch (crunch c)
                      (only array-morphisms-crunch-codegen crunch-codegen-install-module!))
              (include "crunch-numvector-fix.scm")
              (foreign-declare ,(string-append "#include \"crunch-dispatch-n.h\"\n"
                                               "static volatile int " (flag-name) " = 0;\n"))
              (crunch
               ,@(typed-helper-definitions helper-name (flag-name))
               ,@(apply append (map kernel-forms ks specs)))
              (foreign-declare ,(apply string-append (map kernel-run-c ks specs)))
              (crunch-codegen-install-module!
               ,id-placeholder
               (vector (foreign-value ,(string-append "((void *)&" (flag-name) ")") c-pointer)
                       ,@(map (lambda (k)
                                `(foreign-value
                                  ,(string-append "((void *)&am_g" id-placeholder "_k"
                                                  (number->string k) "_run)")
                                  c-pointer))
                              ks)))))))))

  ;; Two 32-bit FNV-1a hashes of s, with different offsets, as 16 hex digits.
  (define (source-hash s)
    (define (fnv offset)
      (let ((n (string-length s)))
        (do ((k 0 (+ k 1))
             (h offset (bitwise-and (* (bitwise-xor h (char->integer (string-ref s k))) 16777619)
                                    #xffffffff)))
            ((= k n) h))))
    (define (hex8 x)
      (let ((t (number->string x 16)))
        (string-append (make-string (- 8 (string-length t)) #\0) t)))
    (string-append (hex8 (fnv 2166136261)) (hex8 (fnv 84696351))))

  (define (string-replace-all s from to)
    (let ((n (string-length from)))
      (let loop ((start 0) (acc '()))
        (let ((k (substring-index from s start)))
          (if k
              (loop (+ k n) (cons to (cons (substring s start k) acc)))
              (apply string-append (reverse (cons (substring s start) acc))))))))

  ;;;; ============================================================
  ;;;; Compiling and loading
  ;;;; ============================================================

  ;; id -> vector #(flag-pointer run-pointer ...), filled when a generated
  ;; module is loaded.
  (define installed-modules (make-hash-table equal?))

  (define (crunch-codegen-install-module! id table)
    "Record the table of a generated module that has just been loaded.
    Called by the generated modules themselves."
    (hash-table-set! installed-modules id table))

  ;; key string -> kernel, #(run-pointer flag-pointer), for the kernels
  ;; loaded in this process.
  (define loaded-kernels (make-hash-table equal?))

  (define (crunch-codegen-forget-kernels!)
    "Forget the kernels loaded in this process, so that the next plan loads
    or compiles its kernels again."
    (set! loaded-kernels (make-hash-table equal?)))

  ;; Totals over the life of the process.
  (define totals (vector 0 0 0 0 0.0))

  (define (crunch-codegen-stats)
    "An alist of totals over the life of the process: kernels (selected
    kernels that ran as compiled code), compiled (kernels compiled),
    cached (kernels loaded from the cache), failed (kernels that could not
    be compiled), compile-seconds (time spent compiling)."
    `((kernels . ,(vector-ref totals 0)) (compiled . ,(vector-ref totals 1))
      (cached . ,(vector-ref totals 2)) (failed . ,(vector-ref totals 3))
      (compile-seconds . ,(vector-ref totals 4))))

  (define (total-add! k x) (vector-set! totals k (+ x (vector-ref totals k))))

  (define (shell-quote s)
    (string-append "'" (string-translate* s '(("'" . "'\\''"))) "'"))

  (define (write-file path text)
    (call-with-output-file path (lambda (p) (display text p))))

  (define (read-file path)
    (and (file-exists? path) (call-with-input-file path (lambda (p) (read-string #f p)))))

  (define (ensure-directory dir)
    (handle-exceptions e #f
      (unless (directory-exists? dir) (create-directory dir #t))
      (and (directory-exists? dir) dir)))

  ;; The directory to compile in: the cache, or a fresh temporary one.
  (define (work-directory)
    (or (let ((d (crunch-codegen-cache-directory))) (and d (ensure-directory d)))
        (create-temporary-directory)))

  (define (warn fmt . args)
    (let ((p (current-error-port)))
      (display "array-morphisms-crunch-codegen: " p)
      (display (apply format #f fmt args) p)
      (newline p)))

  ;; Loads the module of the given id from dir, compiling it first unless
  ;; dir holds a module compiled from the same source.  Returns the
  ;; module's table and whether it was compiled, as two values, or #f and
  ;; a message.
  (define (load-module dir id key-text source)
    (let* ((base     (make-pathname dir (string-append "am-gen-" id)))
           (so       (string-append base ".so"))
           (keyfile  (string-append base ".key"))
           (cached?  (and (file-exists? so) (equal? (read-file keyfile) key-text))))
      (define (do-load)
        (handle-exceptions e
            (values #f (string-append "loading " so " failed: "
                                      (or ((condition-property-accessor 'exn 'message #f) e) "")))
          (load so)
          (let ((t (hash-table-ref/default installed-modules id #f)))
            (if t (values t (not cached?)) (values #f (string-append so " did not install its kernels"))))))
      (if cached?
          (do-load)
          (let* ((tag  (string-append "." (number->string (current-process-id))))
                 (src  (string-append base tag ".scm"))
                 (tmp  (string-append base tag ".so"))
                 (log  (string-append base ".log")))
            (for-each (lambda (f)
                        (let ((path (make-pathname dir (car f))))
                          (unless (equal? (read-file path) (cdr f))
                            (write-file path (cdr f)))))
                      support-files)
            (write-file src source)
            (let* ((cmd (string-append
                         (shell-quote (crunch-codegen-csc)) " "
                         (string-intersperse (map shell-quote csc-options) " ")
                         " -I " (shell-quote dir) " " (shell-quote src)
                         " -o " (shell-quote tmp) " > " (shell-quote log) " 2>&1"))
                   (status (handle-exceptions e 1 (system cmd))))
              (cond ((and (eqv? status 0) (file-exists? tmp))
                     (rename-file tmp so #t)
                     (rename-file src (string-append base ".scm") #t)
                     (write-file keyfile key-text)
                     (do-load))
                    (else
                     (when (file-exists? tmp) (delete-file tmp))
                     (values #f (string-append "compiling " src " failed; see " log)))))))))

  ;; Makes kernels for specs, a list of kernel specifications with distinct
  ;; keys none of which is loaded, and records them in loaded-kernels.
  ;; Returns an alist of statistics.
  (define (build-kernels! specs)
    (if (null? specs)
        '((compiled . 0) (cached . 0) (failed . 0) (compile-seconds . 0.0))
        (let* ((specs   (sort specs (lambda (a b) (string<? (kspec-key a) (kspec-key b)))))
               (source  (module-source specs))
               (id      (source-hash source))
               (t0      (current-jiffy))
               (dir     (work-directory)))
          (let-values (((table compiled?)
                        (if dir
                            (load-module dir id source (string-replace-all source id-placeholder id))
                            (values #f "no directory to compile in"))))
            (let ((secs (exact->inexact (/ (- (current-jiffy) t0) (jiffies-per-second))))
                  (n    (length specs)))
              (cond
               (table
                (for-each (lambda (s k)
                            (hash-table-set! loaded-kernels (kspec-key s)
                                             (vector (vector-ref table (+ k 1)) (vector-ref table 0))))
                          specs (iota n))
                (total-add! (if compiled? 1 2) n)
                (when compiled? (total-add! 4 secs))
                `((compiled . ,(if compiled? n 0)) (cached . ,(if compiled? 0 n)) (failed . 0)
                  (compile-seconds . ,(if compiled? secs 0.0))))
               (else
                (warn "~a; the segops run on the host" compiled?)
                (total-add! 3 n)
                `((compiled . 0) (cached . 0) (failed . ,n) (compile-seconds . ,secs)))))))))

  ;;;; ============================================================
  ;;;; Executors
  ;;;; ============================================================

  (define no-scalars (make-f64vector 1 0.0))

  (define flag-ref (foreign-lambda* int ((c-pointer p)) "C_return(*(volatile int *)p);"))
  (define flag-clear! (foreign-lambda* void ((c-pointer p)) "*(volatile int *)p = 0;"))

  (define (operand-data arr)
    (cases array-morphism arr
      (concrete-array (data _ _ _ _ _ _) data)
      (else (error 'crunch-codegen "operand not concrete" arr))))

  ;; The executor running kernel kern over npoints points.  refs are the
  ;; slots of the kernel's inputs, ip its integer parameters and esz the
  ;; element sizes of its operands; host is the executor that computes the
  ;; segop when the kernel sets its failure flag.
  (define (kernel-executor kern may-fail? npoints refs n-outs ip sc esz host
                           #!optional (work-per-point 1))
    (let* ((run  (vector-ref kern 0))
           (flag (vector-ref kern 1))
           (n    (length refs))
           (refv (list->vector refs))
           (vecs (make-vector (+ n n-outs) #f)))
      (lambda (deref buf #!optional (extra-bufs '()))
        (unless (= npoints 0)
          (do ((k 0 (fx+ k 1))) ((fx= k n))
            (let ((r (vector-ref refv k)))
              (vector-set! vecs k (if (eq? r product-slot) buf (operand-data (deref r))))))
          (vector-set! vecs n buf)
          (let loop ((k (+ n 1)) (bs extra-bufs))
            (when (pair? bs)
              (vector-set! vecs k (car bs))
              (loop (+ k 1) (cdr bs))))
          (when may-fail? (flag-clear! flag))
          (crunch-dispatch-n npoints ip sc vecs esz run (kernel-threads)
                             (max 1 (quotient (crunch-thread-min-chunk) work-per-point)))
          (when (and may-fail? (not (= 0 (flag-ref flag))))
            (flag-clear! flag)
            (if (null? extra-bufs) (host deref buf) (host deref buf extra-bufs)))))))

  ;;;; ============================================================
  ;;;; The plan batch
  ;;;; ============================================================

  ;; The kernels requested while a plan is compiled: a list of entries
  ;; (spec . install), install a procedure taking the kernel, or #f when it
  ;; could not be made.
  (define (plan-batch)
    (or (segop-plan-property 'crunch-codegen-batch)
        (let ((batch (list '())))
          (segop-plan-property-set! 'crunch-codegen-batch batch)
          (segop-plan-add-finalizer! (lambda () (finalize-batch! (car batch))))
          batch)))

  (define (finalize-batch! entries)
    (let* ((new (let loop ((es entries) (seen '()) (acc '()))
                  (cond ((null? es) (reverse acc))
                        ((or (hash-table-ref/default loaded-kernels (kspec-key (caar es)) #f)
                             (member (kspec-key (caar es)) seen))
                         (loop (cdr es) seen acc))
                        (else (loop (cdr es) (cons (kspec-key (caar es)) seen)
                                    (cons (caar es) acc))))))
           (stats (build-kernels! new))
           (installed 0))
      (for-each (lambda (e)
                  (let ((kern (hash-table-ref/default loaded-kernels (kspec-key (car e)) #f)))
                    (when kern
                      (set! installed (+ installed 1))
                      ((cdr e) kern))))
                entries)
      (total-add! 0 installed)
      `((crunch-codegen . ((segops . ,(length entries)) (kernels . ,installed)
                           (distinct . ,(length (delete-duplicates (map (lambda (e) (kspec-key (car e))) entries))))
                           ,@stats)))))

  ;;;; ============================================================
  ;;;; Selection
  ;;;; ============================================================

  (define max-rank 6)

  ;; The linear forms of ops over a space of the given rank, each with one
  ;; coefficient per dimension, or #f.  The form of a reduced result's
  ;; output may have been computed over the kept dimensions only (it is
  ;; cached in the operand), so its coefficient vector is padded with zeros
  ;; for the reduced dimensions; a GEMM epilogue's input, whose form may
  ;; have been computed over the GEMM's space, keeps its first rank
  ;; coefficients.
  (define (linear-forms ops rank)
    (let ((ls (map (lambda (o) (segop-operand-linear o rank)) ops)))
      (and (every values ls)
           (map (lambda (l)
                  (let ((k (vector-length (cdr l))))
                    (cond ((< k rank) (cons (car l) (vector-append (cdr l) (make-vector (- rank k) 0))))
                          ((> k rank) (cons (car l) (subvector (cdr l) 0 rank)))
                          (else l))))
                ls))))

  ;; The integer parameters of a kernel over space whose operands have the
  ;; linear forms lins.
  (define (kernel-ip space lins)
    (list->s64vector
     (append (vector->list space)
             (apply append (map (lambda (l) (cons (car l) (vector->list (cdr l)))) lins)))))

  (define (kernel-esz dtypes) (list->s32vector (map element-size dtypes)))

  (define (map-select sg space ins outs bodies)
    (let ((host (select-segop-executor sg 'crunch-codegen)))
      (and host (map-request space ins outs bodies host 'segop/codegen-map (lambda (exec) exec)))))

  ;; Requests a seg-map kernel over space with operands ins and outs and one
  ;; body per output, for a segop whose executor in the backends after this
  ;; one is host, a pair (tag . executor).  Returns (tag . executor), an
  ;; executor cell's, or #f when the kernel cannot be generated.  When the
  ;; kernel is made, the cell gets (wrap kernel-executor).
  (define (map-request space ins outs bodies host tag wrap)
    (let* ((rank (vector-length space))
           (m    (length outs)))
      (and (<= rank max-rank)
           (every (lambda (o) (memq (segop-operand-dtype o) codegen-dtypes)) (append ins outs))
           (let-values (((canon order) (canonical-scalar-body (cons '%outs (map symbolic-body bodies)))))
             (let* ((cins   (map (lambda (k) (list-ref ins k)) order))
                    (in-dts (map segop-operand-dtype cins))
                    (out-dts (map segop-operand-dtype outs))
                    (cbodies (cdr canon))
                    (lins   (linear-forms (append cins outs) rank)))
               (and lins
                    (<= (+ (length cins) m) crunch-dispatch-n-max-vectors)
                    (map-may-fail? cbodies in-dts out-dts)
                    (let* ((spec (make-kspec 'map rank in-dts out-dts cbodies
                                             (kspec-key-string 'map rank in-dts out-dts cbodies)))
                           (cell (make-segop-executor-cell (cdr host)))
                           (fails? (eq? 'yes (map-may-fail? cbodies in-dts out-dts)))
                           (batch (plan-batch)))
                      (set-car! batch
                                (cons (cons spec
                                            (lambda (kern)
                                              (segop-executor-cell-set!
                                               cell
                                               (wrap
                                                (kernel-executor kern fails? (shape-size space)
                                                                 (map segop-operand-slot cins) m
                                                                 (kernel-ip space lins) no-scalars
                                                                 (kernel-esz (append in-dts out-dts))
                                                                 (cdr host))))))
                                      (car batch)))
                      (cons tag (segop-executor-cell-executor cell)))))))))

  ;;; A GEMM with a map epilogue runs as the GEMM of its prepared plan into
  ;;; the output, followed by a seg-map kernel that applies the epilogue in
  ;;; place over the product: its input 0 is the output itself, read at each
  ;;; point before the point is written.  Should the kernel fail, the host
  ;;; executor computes the GEMM and the epilogue again.

  ;; The slot that stands for the GEMM's product in an epilogue kernel.
  (define product-slot '%gemm-product)

  ;; The epilogue body of a GEMM finalize fin of dtype dt: the product as
  ;; stored, (cast dt (reduced)) or (reduced), is input 0, and the
  ;; finalize's input j, j >= 2, is input j - 1.
  (define (gemm-epilogue-body fin dt)
    (let walk ((e fin))
      (cond ((not (pair? e)) e)
            ((equal? e '(reduced)) '(in 0))
            ((and (eq? (car e) 'cast) (eq? (cadr e) dt) (equal? (caddr e) '(reduced))) '(in 0))
            ((eq? (car e) 'in) `(in ,(- (cadr e) 1)))
            (else (map walk e)))))

  (define (epilogue-select sg host)
    (let ((g (standard-gemm sg)))
      (and g
           (let ((epilogue (list-ref g 5)))
             (and epilogue
                  (eq? (car epilogue) 'map)
                  (let* ((A-ref  (segop-operand-slot (list-ref g 0)))
                         (B-ref  (segop-operand-slot (list-ref g 1)))
                         (out    (list-ref g 2))
                         (plan   (list-ref g 4))
                         (dt     (segop-operand-dtype out))
                         (shape  (segop-operand-shape out))
                         (fin    (cases segop sg
                                   (seg-reduce (space nred ins results map-outs fallback)
                                     (red-result-finalize (car results)))
                                   (else #f)))
                         (prod   (make-segop-operand product-slot dt shape (segop-operand-strides out)
                                                     (segop-operand-offset out) (identity-index-map 2))))
                    (and fin
                         (map-request shape (cons prod (cadr epilogue)) (list out)
                                      (list (gemm-epilogue-body fin dt))
                                      host 'segop/codegen-epilogue
                                      (lambda (exec)
                                        (lambda (deref buf)
                                          (execute-gemm-plan/into! plan (deref A-ref) (deref B-ref) buf)
                                          (exec deref buf)))))))))))

  (define max-red-rank 8)

  ;; The combine specs a generated kernel follows: the unspecified order is
  ;; taken as sequential, as the host executors of non-standard seg-reduces
  ;; take it.
  (define (red-spec-ok? spec)
    (and (spec-op spec)
         (memq (combine-spec-acc-dtype spec) codegen-dtypes)))

  ;; The combine op a kernel computes for spec, or #f.  The Scheme
  ;; procedures max and min, written (opaque max) and (opaque min), skip a
  ;; NaN argument where the max and min ops keep the accumulator; the two
  ;; agree because the accumulator starts at a neutral element that is not
  ;; NaN and a NaN value never replaces it, so it is never NaN.
  (define (spec-op spec)
    (let ((op (combine-spec-op spec)))
      (cond ((memq op '(add mul max min)) op)
            ((and (pair? op) (eq? (cadr op) max)) 'max)
            ((and (pair? op) (eq? (cadr op) min)) 'min)
            (else #f))))

  (define (spec-order spec)
    (if (eq? (combine-spec-order spec) 'blocked) 'blocked 'sequential))

  ;; The canonical form of a seg-reduce's bodies: the list (results
  ;; map-bodies) of a kernel specification, and the input order, as two
  ;; values.
  (define (canonical-red results map-bodies)
    (let-values (((canon order)
                  (canonical-scalar-body
                   `(%red (%results ,@(map (lambda (r) `(%result ,(car r) ,(or (cadr r) '%none))) results))
                          (%maps ,@map-bodies)))))
      (values (list (map (lambda (x) (list (cadr x) (if (eq? (caddr x) '%none) #f (caddr x))))
                         (cdadr canon))
                    (cdaddr canon))
              order)))

  ;; yes, no or #f, as map-may-fail? does, for a seg-reduce kernel.
  (define (red-may-fail? results map-bodies in-dts out-dts)
    (let* ((z  (lambda (j) 0))
           (st (red-statements results map-bodies in-dts out-dts z z z z
                               (lambda (r) 'a) (lambda (r) 'b) (lambda (r) 'c) (lambda (r) 'd)
                               (lambda (r) 0))))
      (and st (if (cadddr st) 'yes 'no))))

  (define (red-select sg space nred ins results map-outs)
    (let* ((rank (vector-length space))
           (outs (append (map red-result-out results) (map car map-outs)))
           (host (select-segop-executor sg 'crunch-codegen)))
      (if (and host (eq? (car host) 'segop/gemm-epilogue-host))
          (epilogue-select sg host)
      (and host
           ;; standard reductions and contractions keep their own executors
           (memq (car host) '(segop/strided-red segop/interpreted))
           (<= rank max-red-rank)
           (every (lambda (o) (memq (segop-operand-dtype o) codegen-dtypes)) (append ins outs))
           (every (lambda (r) (red-spec-ok? (red-result-combine r))) results)
           (let-values (((canon order)
                         (canonical-red (map (lambda (r) (list (symbolic-body (red-result-body r))
                                                                (symbolic-body (red-result-finalize r))))
                                             results)
                                        (map (lambda (mo) (symbolic-body (cdr mo))) map-outs))))
             (let* ((cins    (map (lambda (k) (list-ref ins k)) order))
                    (in-dts  (map segop-operand-dtype cins))
                    (out-dts (map segop-operand-dtype outs))
                    (kres    (map (lambda (r cr)
                                    (let ((spec (red-result-combine r)))
                                      (list (spec-op spec) (combine-spec-acc-dtype spec)
                                            (spec-order spec) (car cr) (cadr cr))))
                                  results (car canon)))
                    (kbodies (list nred kres (cadr canon)))
                    (lins    (linear-forms (append cins outs) rank))
                    (fails   (and lins (<= (+ (length cins) (length outs)) crunch-dispatch-n-max-vectors)
                                  (red-may-fail? kres (cadr canon) in-dts out-dts))))
               (and fails
                    (let* ((spec (make-kspec 'red rank in-dts out-dts kbodies
                                             (kspec-key-string 'red rank in-dts out-dts kbodies)))
                           (cell (make-segop-executor-cell (cdr host)))
                           (npar (- rank nred))
                           (ip   (list->s64vector
                                  (append (s64vector->list (kernel-ip space lins))
                                          (apply append
                                                 (map (lambda (r)
                                                        (let ((spec (red-result-combine r)))
                                                          (list (or (combine-spec-block-size spec) 1)
                                                                (if (codegen-float-dtype? (combine-spec-acc-dtype spec))
                                                                    0
                                                                    (round-to-dtype (combine-spec-neutral spec)
                                                                                    (combine-spec-acc-dtype spec))))))
                                                      results)))))
                           (sc   (list->f64vector
                                  (map (lambda (r)
                                         (let ((spec (red-result-combine r)))
                                           (if (codegen-float-dtype? (combine-spec-acc-dtype spec))
                                               (exact->inexact (round-to-dtype (combine-spec-neutral spec)
                                                                               (combine-spec-acc-dtype spec)))
                                               0.0)))
                                       results)))
                           (batch (plan-batch)))
                      (set-car! batch
                                (cons (cons spec
                                            (lambda (kern)
                                              (segop-executor-cell-set!
                                               cell
                                               (kernel-executor kern (eq? fails 'yes)
                                                                (shape-size (subvector space 0 npar))
                                                                (map segop-operand-slot cins) (length outs)
                                                                ip (if (= 0 (f64vector-length sc)) no-scalars sc)
                                                                (kernel-esz (append in-dts out-dts))
                                                                (cdr host)
                                                                (max 1 (shape-size (subvector space npar rank)))))))
                                      (car batch)))
                      (cons 'segop/codegen-red (segop-executor-cell-executor cell))))))))))

  ;; A seg-scan the backends after this one run with a Scheme loop
  ;; (segop/scan, segop/strided-scan or the interpreter): one result whose
  ;; accumulator has the output's dtype, the sequential order, linear
  ;; operands, a seed read as an input of the output's dtype, and a step
  ;; and body the translator accepts.  Its lines are the jobs of the
  ;; thread pool.
  (define (scan-select sg space ins results)
    (let* ((rank (vector-length space))
           (host (select-segop-executor sg 'crunch-codegen)))
      (and host
           (memq (car host) '(segop/scan segop/strided-scan segop/interpreted))
           (= (length results) 1)
           (<= 1 rank max-red-rank)
           (let* ((res  (car results))
                  (out  (scan-result-out res))
                  (odt  (segop-operand-dtype out))
                  (spec (scan-result-combine res))
                  (in-dts (map segop-operand-dtype ins))
                  (seed (scan-result-seed res))
                  (seed-k (and seed (pair? seed) (eq? (car seed) 'in) (cadr seed)))
                  (step (scan-step-desc spec (scan-result-swapped? res)))
                  (excl? (and (scan-result-exclusive? res) #t))
                  (neutral (let ((v (combine-spec-neutral spec)))
                             (and v (round-to-dtype v odt))))
                  (bodies (list (symbolic-body (scan-result-body res)) seed-k step neutral excl?
                                (map symbolic-body (scan-result-aux res)))))
             (and (memq odt codegen-dtypes)
                  (every (lambda (dt) (memq dt codegen-dtypes)) in-dts)
                  (eq? (combine-spec-acc-dtype spec) odt)
                  (memq (combine-spec-order spec) '(sequential unspecified))
                  step
                  (or (not seed) (and seed-k (eq? (list-ref in-dts seed-k) odt)))
                  (or (not excl?) seed neutral)
                  (<= (+ (length ins) 1) crunch-dispatch-n-max-vectors)
                  (let ((lins (linear-forms (append ins (list out)) rank))
                        (st   (scan-statements bodies in-dts odt (lambda (k r) 0) (lambda (r) 0) 1)))
                    (and lins st
                         (let* ((kspec (make-kspec 'scan rank in-dts (list odt) bodies
                                                   (kspec-key-string 'scan rank in-dts (list odt) bodies)))
                                (cell  (make-segop-executor-cell (cdr host)))
                                (npar  (- rank 1))
                                (ip    (kernel-ip space lins))
                                (batch (plan-batch)))
                           (set-car! batch
                                     (cons (cons kspec
                                                 (lambda (kern)
                                                   (segop-executor-cell-set!
                                                    cell
                                                    (kernel-executor kern (cadddr st)
                                                                     (shape-size (subvector space 0 npar))
                                                                     (map segop-operand-slot ins) 1
                                                                     ip no-scalars
                                                                     (kernel-esz (append in-dts (list odt)))
                                                                     (cdr host)
                                                                     (max 1 (vector-ref space npar))))))
                                           (car batch)))
                           (cons 'segop/codegen-scan (segop-executor-cell-executor cell))))))))))

  (define (codegen-select sg)
    (and (crunch-codegen)
         (segop-plan-context)
         (cases segop sg
           (seg-map (space ins outs bodies fallback) (map-select sg space ins outs bodies))
           (seg-reduce (space nred ins results map-outs fallback)
             (and (> nred 0) (red-select sg space nred ins results map-outs)))
           (seg-scan (space ins results fallback) (scan-select sg space ins results))
           (else #f))))

  ;;;; ============================================================
  ;;;; Native queries
  ;;;; ============================================================

  ;; True when index map m reads an operand at an affine function of the
  ;; point with no guard.
  (define (affine-map? m rank)
    (and (index-map-exprs m)
         (index-map->linear m (make-vector (vector-length (index-map-exprs m)) 1) 0 rank)
         #t))

  (define (crunch-codegen-native? tpl)
    "The native-capability query of the backend: true, while crunch-codegen
    is true, when segop template tpl is a seg-map with affine maps over a
    space of rank at most 6, a seg-reduce with affine maps over a space of
    rank at most 8 whose results combine with add, mul, max or min, or a
    GEMM with an epilogue (see gemm-template-native?); its operands have
    the dtypes f32, f64, s32, s64 and u32, and the typed translator
    accepts its bodies.  A template made for a fusion is native only when
    its class is in (crunch-codegen-fusions)."
    (and (crunch-codegen)
         (let ((class (segop-template-class tpl)))
           (or (not class) (memq class (crunch-codegen-fusions))))
         (cond ((= (segop-template-nred tpl) 0) (map-template-native? tpl))
               ((gemm-template? tpl) (gemm-template-native? tpl))
               (else (red-template-native? tpl)))))

  (define (gemm-template? tpl)
    (let ((outs (segop-template-outs tpl)))
      (and (pair? outs) (eq? (template-output-combine (car outs)) 'gemm))))

  ;; A GEMM template is native when it has an epilogue that the typed
  ;; translator accepts, over the product (of the output's dtype, f32 or
  ;; f64) and inputs read through affine maps of the product's indices.
  ;; The GEMM itself runs on the BLAS backend.
  (define (gemm-template-native? tpl)
    (let* ((ins  (segop-template-ins tpl))
           (outs (segop-template-outs tpl))
           (res  (car outs))
           (dt   (template-output-dtype res))
           (fin  (template-output-finalize res))
           (extras (if (and (pair? ins) (pair? (cdr ins))) (cddr ins) '())))
      (and (null? (cdr outs))
           (codegen-float-dtype? dt)
           fin
           (<= (+ 2 (length extras)) crunch-dispatch-n-max-vectors)
           (every (lambda (i) (memq (template-input-dtype i) codegen-dtypes)) extras)
           (every (lambda (i) (affine-map? (template-input-map i) 2)) extras)
           (map-may-fail? (list (symbolic-body (gemm-epilogue-body fin dt)))
                          (cons dt (map template-input-dtype extras))
                          (list dt))
           #t)))

  (define (red-template-native? tpl)
    (let* ((space (segop-template-space tpl))
           (rank  (vector-length space))
           (nred  (segop-template-nred tpl))
           (ins   (segop-template-ins tpl))
           (outs  (segop-template-outs tpl))
           (res   (filter template-output-combine outs))
           (mos   (filter (lambda (o) (not (template-output-combine o))) outs))
           (in-dts (map template-input-dtype ins)))
      (and (<= rank max-red-rank)
           (pair? res)
           (every (lambda (o) (combine-spec? (template-output-combine o))) res)
           (every (lambda (o) (red-spec-ok? (template-output-combine o))) res)
           (<= (+ (length ins) (length outs)) crunch-dispatch-n-max-vectors)
           (every (lambda (dt) (memq dt codegen-dtypes))
                  (append in-dts (map template-output-dtype outs)))
           (every (lambda (i) (affine-map? (template-input-map i) rank)) ins)
           (every (lambda (o) (or (not (template-output-map o))
                                  (affine-map? (template-output-map o) rank)))
                  outs)
           (red-may-fail? (map (lambda (o)
                                 (let ((spec (template-output-combine o)))
                                   (list (spec-op spec) (combine-spec-acc-dtype spec)
                                         (spec-order spec) (symbolic-body (template-output-body o))
                                         (symbolic-body (template-output-finalize o)))))
                               res)
                          (map (lambda (o) (symbolic-body (template-output-body o))) mos)
                          in-dts
                          (map template-output-dtype (append res mos)))
           #t)))

  (define (map-template-native? tpl)
         (let* ((space (segop-template-space tpl))
                (rank  (vector-length space))
                (ins   (segop-template-ins tpl))
                (outs  (segop-template-outs tpl))
                (in-dts (map template-input-dtype ins)))
           (and (<= rank max-rank)
                (<= (+ (length ins) (length outs)) crunch-dispatch-n-max-vectors)
                (every (lambda (dt) (memq dt codegen-dtypes))
                       (append in-dts (map template-output-dtype outs)))
                (every (lambda (i) (affine-map? (template-input-map i) rank)) ins)
                (every (lambda (o) (or (not (template-output-map o))
                                       (affine-map? (template-output-map o) rank)))
                       outs)
                (map-may-fail? (map (lambda (o) (symbolic-body (template-output-body o))) outs) in-dts
                               (map template-output-dtype outs))
                #t)))

  ;;;; ============================================================
  ;;;; Registration
  ;;;; ============================================================

  (define (register-crunch-codegen-backend!)
    "Add the code generator to the segop backends, after the registry
    backend when there is one (so that catalogue and BLAS kernels keep
    priority) and otherwise before the scheme backend."
    (let ((names (map segop-backend-name (segop-backends)))
          (be (make-segop-backend 'crunch-codegen codegen-select
                                  kind: 'codegen native?: crunch-codegen-native?)))
      (register-segop-backend!
       be
       (cond ((memq 'registry names) '(after registry))
             ((memq 'scheme names) '(before scheme))
             (else 'last)))))

  (define (unregister-crunch-codegen-backend!)
    "Remove the code generator from the segop backends."
    (unregister-segop-backend! 'crunch-codegen))

) ;; end module array-morphisms-crunch-codegen
