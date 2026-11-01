;;; tests/test-crunch-threads.scm
;;; Tests for the thread dispatcher (crunch-thread-dispatch.scm) and
;;; the threaded activation backend (make-crunch-threaded-activation-backend).
;;;
;;; Splitting an element-wise kernel across threads must not change any
;;; result, so every comparison here is exact: eqv? element by element,
;;; with NaN accepted only against NaN.  
;;;
;;; Organisation:
;;;   Group 1 - Thread-count decisions
;;;   Group 2 - Exact results for every op, dtype, size and thread count
;;;   Group 3 - Buffers: in place, prefixes
;;;   Group 4 - Repeated calls
;;;   Group 5 - Settings: parameters, defaults
;;;   Group 6 - Argument checking
;;;   Group 7 - SSA replay through the threaded backend
;;;   Group 8 - Thread pool: growth, shutdown, wait paths, fork, signals, GC

(import scheme (chicken base)
        test
        (only srfi-1 iota every filter-map take)
        srfi-4
        (only (chicken number-vector)
         f32vector->bytevector/shared f64vector->bytevector/shared
         subf32vector subf64vector)
        (only (chicken gc) gc)
        (only (chicken process) process-fork process-wait process-id process-signal)
        (only (chicken process signal) set-signal-handler! signal-handler signal/usr1)
        (only (chicken process-context posix) current-process-id)
        datatype
        array-morphisms-core
        array-morphisms-basic-ops
        array-morphisms-realization
        array-morphisms-context
        array-morphisms-activation-exec
        (prefix array-morphisms-grad am:)
        array-morphisms-ssa
        array-morphisms-crunch-activations
        array-morphisms-crunch-optimizers
        array-morphisms-crunch-threads)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (th-same? a b) (or (and (nan? a) (nan? b)) (eqv? a b)))

(define (th-same-lists? a b)
  (and (= (length a) (length b)) (every th-same? a b)))

(define th-ops '(relu sigmoid tanh relu-deriv sigmoid-deriv tanh-deriv
                 exp negate abs))

;; Special values first, then a deterministic spread covering both signs,
;; small and large magnitudes, and exp overflow.
(define th-specials (list 0.0 -0.0 +inf.0 -inf.0 +nan.0 710.0 -710.0 400.0 -400.0))

(define (th-input n)
  (let ((v (make-f64vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f64vector-set! v i
                      (if (< i (length th-specials))
                          (list-ref th-specials i)
                          (* 12.0 (sin (* 0.37 i))))))))

(define (th-vector dtype n)
  (let ((src (th-input n)))
    (if (eq? dtype 'f32)
        (list->f32vector (f64vector->list src))
        src)))

(define (th->list v) (if (f32vector? v) (f32vector->list v) (f64vector->list v)))

(define (th-make dtype n fill)
  (if (eq? dtype 'f32) (make-f32vector n fill) (make-f64vector n fill)))

(define th-plain    (make-crunch-activation-backend))
(define th-threaded (make-crunch-threaded-activation-backend))

(define (plain-kernel op dtype)    (lookup-activation-kernel th-plain op dtype))
(define (threaded-kernel op dtype) (lookup-activation-kernel th-threaded op dtype))

(define (dispatch dtype n in out op threads min-chunk)
  ((if (eq? dtype 'f32) crunch-dispatch-f32 crunch-dispatch-f64)
   n in out (crunch-activation-chunk-kernel op dtype) threads min-chunk))

;; The Scheme combiners the kernels replace.
(define th-combiners
  `((relu . ,(lambda (x) (max 0.0 x)))
    (sigmoid . ,(lambda (x) (/ 1.0 (+ 1.0 (exp (- x))))))
    (tanh . ,(lambda (x)
               (let* ((e (exp (* -2.0 (abs x))))
                      (t (/ (- 1.0 e) (+ 1.0 e))))
                 (if (< x 0.0) (- t) t))))
    (relu-deriv . ,(lambda (x) (if (> x 0.0) 1.0 0.0)))
    (sigmoid-deriv . ,(lambda (s) (* s (- 1.0 s))))
    (tanh-deriv . ,(lambda (t) (- 1.0 (* t t))))
    (exp . ,exp)
    (negate . ,(lambda (x) (- x)))
    (abs . ,abs)))

(define (combiner-result op dtype in n)
  (let ((out (th-make dtype n 0.0)))
    (execute-flat-unary-compute (cdr (assq op th-combiners)) in dtype out n dtype)
    (th->list out)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - Thread-count decisions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - thread-count decisions"
  (define (used n threads min-chunk)
    (dispatch 'f32 n (th-vector 'f32 (max n 1)) (th-make 'f32 (max n 1) 0.0)
              'relu threads min-chunk))
  (test "fewer elements than min-chunk: one thread" 1 (used 1000 8 4096))
  (test "exactly two chunks' worth: two threads" 2 (used 8192 8 4096))
  (test "one thread requested: one thread" 1 (used 100000 1 1))
  (test "min-chunk 1, n 1000, 4 threads requested: four" 4 (used 1000 4 1))
  (test "n 3 with 16 threads requested: three" 3 (used 3 16 1))
  (test "more threads than the maximum are clamped" crunch-max-threads
        (used 10000 100 1))
  (test "n 0: one thread" 1 (used 0 8 1))
  (test "the maximum is 64" 64 crunch-max-threads))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - Exact results
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - results identical for every thread count"
  (for-each
   (lambda (dtype)
     (for-each
      (lambda (op)
        (test-assert (string-append (symbol->string op) " " (symbol->string dtype)
                                    ": all sizes and thread counts")
          (every
           (lambda (n)
             (let* ((in  (th-vector dtype (max n 1)))
                    (ref (let ((o (th-make dtype (max n 1) 0.0)))
                           ((plain-kernel op dtype) n in o)
                           (th->list o)))
                    (comb (combiner-result op dtype in n)))
               (and (th-same-lists? (take ref n) comb)
                    (every
                     (lambda (threads)
                       (let ((out (th-make dtype (max n 1) 0.0)))
                         (dispatch dtype n in out op threads 1)
                         (th-same-lists? (th->list out) ref)))
                     '(1 2 3 4 7 16)))))
           '(0 1 2 6 7 8 1000 100003))))
      th-ops))
   '(f64 f32)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - Buffers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - buffers"
  (test-assert "in place with 8 threads equals out of place"
    (let* ((n 50001)
           (buf (th-vector 'f32 n))
           (ref (let ((o (th-make 'f32 n 0.0)))
                  ((plain-kernel 'tanh 'f32) n buf o)
                  (th->list o))))
      (and (= 8 (dispatch 'f32 n buf buf 'tanh 8 1))
           (th-same-lists? (th->list buf) ref))))

  (test "a prefix leaves the rest of the output untouched"
    '(0.0 0.0 1.0 9.0 9.0)
    (let ((out (f64vector 9.0 9.0 9.0 9.0 9.0)))
      (dispatch 'f64 3 (f64vector -1.0 0.0 2.0 -3.0 4.0) out 'relu-deriv 3 1)
      (f64vector->list out)))

  (test "output longer than input: only n elements are written"
    '(0.5 7.0)
    (let ((out (f32vector 7.0 7.0)))
      (dispatch 'f32 1 (f32vector 0.0) out 'sigmoid 4 1)
      (f32vector->list out))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - Repeated calls
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - repeated calls"
  (test-assert "2000 threaded calls give the same result every time"
    ;; Each call uses 2 to 8 threads, so the chunk boundaries move from
    ;; call to call; one element is spoiled before each call and must be
    ;; rewritten by it.  Outputs are compared byte for byte through shared
    ;; bytevector views of the two f32vectors, which checks that they are
    ;; bit-identical (a NaN matches only a NaN with the same bits).
    (let* ((n 50000)
           (in  (th-vector 'f32 n))
           (ref (let ((o (th-make 'f32 n 0.0)))
                  ((plain-kernel 'sigmoid 'f32) n in o)
                  o))
           (out (th-make 'f32 n 0.0))
           (ref-bytes (f32vector->bytevector/shared ref))
           (out-bytes (f32vector->bytevector/shared out)))
      (let loop ((k 0))
        (or (= k 2000)
            (begin
              (f32vector-set! out (modulo (* k 7919) n) -1.0)
              (dispatch 'f32 n in out 'sigmoid (+ 2 (modulo k 7)) 1)
              (and (equal? out-bytes ref-bytes) (loop (+ k 1)))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5 - Settings
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - settings"
  (test-assert "the default thread count is between 1 and the maximum"
    (<= 1 (crunch-thread-count) crunch-max-threads))
  (test-assert "the default min-chunk is positive"
    (> (crunch-thread-min-chunk) 0))
  (test-assert "at least one processor is reported"
    (>= (crunch-available-processors) 1))

  (test "backend kernels follow parameterized settings at call time"
    '(1 4 2)
    (let* ((n 4096) (in (th-vector 'f64 n)) (out (th-make 'f64 n 0.0))
           (k (threaded-kernel 'sigmoid 'f64)))
      (list (parameterize ((crunch-thread-count 1)) (k n in out))
            (parameterize ((crunch-thread-count 4) (crunch-thread-min-chunk 1)) (k n in out))
            (parameterize ((crunch-thread-count 4) (crunch-thread-min-chunk 2048)) (k n in out)))))

  (test "with the default min-chunk a small array stays on one thread"
    1
    ((threaded-kernel 'relu 'f32) 100 (th-vector 'f32 100) (th-make 'f32 100 0.0)))

  (test-error "crunch-thread-count rejects 0" (crunch-thread-count 0))
  (test-error "crunch-thread-min-chunk rejects -5" (crunch-thread-min-chunk -5))

  (test "the threaded backend is named crunch-threaded"
    'crunch-threaded (activation-backend-name th-threaded))
  (test-assert "the threaded backend has f32 and f64 kernels for every op"
    (every (lambda (op) (and (threaded-kernel op 'f32) (threaded-kernel op 'f64))) th-ops))
  (test "an unknown op has no chunk kernel"
    #f (crunch-activation-chunk-kernel 'no-such-op 'f32)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6 - Argument checking
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "crunch threads - argument checking"
  (define k32 (crunch-activation-chunk-kernel 'relu 'f32))
  (test-error "size larger than the input"
    (crunch-dispatch-f32 5 (make-f32vector 4 0.0) (make-f32vector 5 0.0) k32 2 1))
  (test-error "size larger than the output"
    (crunch-dispatch-f32 5 (make-f32vector 5 0.0) (make-f32vector 4 0.0) k32 2 1))
  (test-error "negative size"
    (crunch-dispatch-f32 -1 (make-f32vector 4 0.0) (make-f32vector 4 0.0) k32 2 1))
  (test-error "thread count 0"
    (crunch-dispatch-f32 4 (make-f32vector 4 0.0) (make-f32vector 4 0.0) k32 0 1))
  (test-error "min-chunk 0"
    (crunch-dispatch-f32 4 (make-f32vector 4 0.0) (make-f32vector 4 0.0) k32 2 0))
  (test-error "an f64vector given to the f32 dispatcher"
    (crunch-dispatch-f32 4 (make-f64vector 4 0.0) (make-f32vector 4 0.0) k32 2 1))
  (test-error "a kernel that is not a pointer"
    (crunch-dispatch-f64 4 (make-f64vector 4 0.0) (make-f64vector 4 0.0) 'relu 2 1))
  (test-error "the threaded backend kernel checks sizes too"
    ((threaded-kernel 'tanh 'f64) 10 (make-f64vector 4 0.0) (make-f64vector 10 0.0))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7 - SSA replay through the threaded backend
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (th-array-values m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (map (lambda (i)
             (exact->inexact
              (typed-vector-ref data dtype
                                (multi-to-linear-index (linear-to-multi-index i shape)
                                                       strides offset))))
           (iota (shape-size shape))))
    (else (error "th-array-values: not concrete"))))

;; Trace, finalize and replay twice; the loss and gradients of the second
;; replay as value lists.
(define (th-replay make-loss)
  (let-values (((loss params) (make-loss)))
    (let* ((ctx    (make-morphism-context))
           (fwd    (morphism-to-ssa loss))
           (p-vals (filter-map (lambda (p) (ssa-constant-id fwd (am:var-value p))) params))
           (joint  (ssa-vjp fwd p-vals (ssa-loss-binding-val fwd))))
      (ssa-realize/ctx ctx joint)
      (finalize-context! ctx)
      (reset-context! ctx)
      (ssa-realize/ctx ctx joint)
      (reset-context! ctx)
      (map th-array-values (ssa-realize/ctx ctx joint)))))

;; h = relu(x W1 + b1); y = sigmoid(tanh(h W2)); loss = mean(y)
(define (th-mlp dtype)
  (lambda ()
    (let* ((mk  (lambda (n shape f g?)
                  (am:make-var (morph-from-list (map f (iota n)) shape dtype) g?)))
           (xv  (mk 24 #(8 3) (lambda (i) (- (* 0.29 i) 3.0)) #f))
           (W1  (mk 15 #(3 5) (lambda (i) (* 0.21 (sin (+ i 1.0)))) #t))
           (b1  (mk 5  #(5)   (lambda (i) (- (* 0.1 i) 0.2)) #t))
           (W2  (mk 10 #(5 2) (lambda (i) (* 0.3 (cos (+ i 0.5)))) #t))
           (h   (am:var-relu (am:var+ (am:var-matmul xv W1) b1)))
           (y   (am:var-sigmoid (am:var-tanh (am:var-matmul h W2)))))
      (values (am:var-mean y) (list W1 b1 W2)))))

;; A copy of the threaded backend whose kernels record the largest thread
;; count any call used.
(define (recording-backend record)
  (let ((be (make-activation-backend 'recording-threaded)))
    (for-each
     (lambda (key)
       (let ((k (lookup-activation-kernel th-threaded (car key) (cdr key))))
         (activation-backend-add-kernel!
          be (car key) (cdr key)
          (lambda (n in out)
            (let ((t (k n in out)))
              (set-car! record (max (car record) t))
              t)))))
     (activation-backend-kernels th-threaded))
    be))

(test-group "crunch threads - SSA replay equivalence"
  (for-each
   (lambda (dtype)
     (let ((record (list 0)))
       (register-activation-backend! #f)
       (let ((without (th-replay (th-mlp dtype))))
         (register-activation-backend! (recording-backend record))
         (let ((with (parameterize ((crunch-thread-count 4) (crunch-thread-min-chunk 1))
                       (th-replay (th-mlp dtype)))))
           (register-activation-backend! #f)
           (test-assert (string-append "MLP " (symbol->string dtype)
                                       ": replay used several threads")
             (> (car record) 1))
           (test-assert (string-append "MLP " (symbol->string dtype)
                                       ": loss and gradients identical to the fallback")
             (and (= (length without) (length with) 4)
                  (every th-same-lists? without with)))))))
   '(f64 f32)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 8 - Thread pool
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Every result is compared byte for byte with a single-threaded
;; reference through shared bytevector views.
(define (th-bytes v)
  (if (f32vector? v) (f32vector->bytevector/shared v) (f64vector->bytevector/shared v)))

(define (th-copy v)
  (if (f32vector? v) (subf32vector v 0 (f32vector-length v)) (subf64vector v 0 (f64vector-length v))))

(define (th-reference op dtype in n)
  (let ((o (th-make dtype (max n 1) 0.0)))
    ((plain-kernel op dtype) n in o)
    o))

(define (th-expected-threads n threads min-chunk)
  (max 1 (min threads (quotient n min-chunk) crunch-max-threads)))

(define (th-stat name) (cdr (assq name (crunch-thread-pool-stats))))

;; Adam through crunch-dispatch4-f32: param, grad, m and v of length n,
;; updated once with the given thread settings; returns the four vectors.
(define th-adam-hyper (f64vector 0.001 0.9 0.999 1e-8 0.1 0.001 0.01))

(define (th-adam-initial n)
  (map (lambda (f)
         (let ((v (make-f32vector n 0.0)))
           (do ((i 0 (+ i 1))) ((= i n) v) (f32vector-set! v i (f i)))))
       (list (lambda (i) (sin (* 0.1 i)))
             (lambda (i) (cos (* 0.3 i)))
             (lambda (i) (* 0.01 (sin (* 0.7 i))))
             (lambda (i) (* 0.001 (+ 1.0 (cos (* 0.2 i))))))))

(define (th-adam-run initial n threads min-chunk)
  (let ((state (map th-copy initial)))
    (parameterize ((crunch-thread-count threads) (crunch-thread-min-chunk min-chunk))
      (apply crunch-adam-f32 n th-adam-hyper state))
    state))

(define (th-adam-same? a b)
  (every (lambda (x y) (equal? (th-bytes x) (th-bytes y))) a b))

;; Run thunk in a forked child; #t if it returned a true value there.
;; The child leaves through emergency-exit whatever happens, so that it
;; never goes on to run the rest of the test file.
(define (th-in-child thunk)
  (flush-output)
  (let ((pid (process-fork
              (lambda ()
                (emergency-exit (if (handle-exceptions e #f (thunk)) 0 1))))))
    (let-values (((p normal? status) (process-wait pid)))
      (and normal? (= status 0)))))

(define th-initial-spin (crunch-thread-pool-spin-us))

(test-group "crunch threads - pool: results around the chunk thresholds"
  (for-each
   (lambda (dtype)
     (test-assert (string-append "sigmoid " (symbol->string dtype)
                                 ": T = 1..16, min-chunk 4096, n near multiples of it")
       (every
        (lambda (n)
          (let* ((in  (th-vector dtype n))
                 (ref (th-bytes (th-reference 'sigmoid dtype in n))))
            (every
             (lambda (threads)
               (let ((out (th-make dtype n -1.0)))
                 (and (= (dispatch dtype n in out 'sigmoid threads 4096)
                         (th-expected-threads n threads 4096))
                      (equal? (th-bytes out) ref))))
             (iota 16 1))))
        '(4095 4096 8191 8192 8193 12287 12288 12289 65537))))
   '(f32 f64)))

(test-group "crunch threads - pool: growth and shutdown"
  (define in  (th-vector 'f32 1000))
  (define ref (th-bytes (th-reference 'tanh 'f32 in 1000)))
  (define (run threads)
    (let ((out (th-make 'f32 1000 0.0)))
      (dispatch 'f32 1000 in out 'tanh threads 1)
      (equal? (th-bytes out) ref)))

  (crunch-thread-pool-shutdown!)
  (test "empty after shutdown" 0 (crunch-thread-pool-size))
  (test "shutdown of an empty pool stops nothing" 0 (crunch-thread-pool-shutdown!))
  (test-assert "a one-thread dispatch creates no worker"
    (and (run 1) (= 0 (crunch-thread-pool-size))))
  (test-assert "a one-thread dispatch does not count as a pool dispatch"
    (let ((d (th-stat 'dispatches)))
      (run 1)
      (= d (th-stat 'dispatches))))
  (test "three threads: two workers" 2 (begin (run 3) (crunch-thread-pool-size)))
  (test "six threads: five workers" 5 (begin (run 6) (crunch-thread-pool-size)))
  (test "two threads afterwards: the pool keeps five workers" 5
        (begin (run 2) (crunch-thread-pool-size)))
  (test "each threaded dispatch counts once" 3
        (let ((d (th-stat 'dispatches)))
          (run 2) (run 1) (run 5) (run 6)
          (- (th-stat 'dispatches) d)))
  (test "shutdown stops five workers" 5 (crunch-thread-pool-shutdown!))
  (test "shutdown again stops none" 0 (crunch-thread-pool-shutdown!))
  (test-assert "a dispatch after shutdown recreates the pool and is correct"
    (and (run 4) (= 3 (crunch-thread-pool-size))))
  (test-assert "the largest thread count gives 63 workers, and results stay correct"
    (and (run 100) (= 63 (crunch-thread-pool-size)) (run 8) (run 3)))
  (crunch-thread-pool-shutdown!))

(test-group "crunch threads - pool: spin setting"
  (test-assert "the spin time is between 0 and 1000000 microseconds"
    (<= 0 th-initial-spin 1000000))
  (test "the spin time can be set and read back" '(0 1000)
        (list (begin (crunch-thread-pool-spin-us-set! 0) (crunch-thread-pool-spin-us))
              (begin (crunch-thread-pool-spin-us-set! 1000) (crunch-thread-pool-spin-us))))
  (test-error "a negative spin time is rejected" (crunch-thread-pool-spin-us-set! -1))
  (test-error "a spin time above 1000000 is rejected" (crunch-thread-pool-spin-us-set! 1000001))
  (test-error "a non-integer spin time is rejected" (crunch-thread-pool-spin-us-set! 1.5))
  (crunch-thread-pool-spin-us-set! th-initial-spin))

;; Back-to-back dispatches of three kinds, cycling through sizes and
;; thread counts: f32 sigmoid, f64 tanh and an f32 Adam step through
;; crunch-dispatch4-f32.  Returns #t when every result matched its
;; reference, and the number of calls that used more than one thread.
(define th-stress-sizes '(1000 4096 5000 20011 65536))
(define th-stress-min-chunk 1024)

(define th-stress-cases
  (map (lambda (n)
         (let ((in32 (th-vector 'f32 n)) (in64 (th-vector 'f64 n))
               (adam (th-adam-initial n)))
           (list n
                 in32 (th-bytes (th-reference 'sigmoid 'f32 in32 n))
                 in64 (th-bytes (th-reference 'tanh 'f64 in64 n))
                 adam (th-adam-run adam n 1 1))))
       th-stress-sizes))

(define (th-stress iterations)
  (let loop ((k 0) (threaded 0))
    (if (= k iterations)
        (values #t threaded)
        (let* ((c (list-ref th-stress-cases (modulo (quotient k 3) (length th-stress-cases))))
               (n (car c))
               (threads (+ 1 (modulo k 8)))
               (T (th-expected-threads n threads th-stress-min-chunk))
               (ok (case (modulo k 3)
                     ((0) (let ((out (th-make 'f32 n 0.0)))
                            (and (= T (dispatch 'f32 n (list-ref c 1) out 'sigmoid
                                                threads th-stress-min-chunk))
                                 (equal? (th-bytes out) (list-ref c 2)))))
                     ((1) (let ((out (th-make 'f64 n 0.0)))
                            (and (= T (dispatch 'f64 n (list-ref c 3) out 'tanh
                                                threads th-stress-min-chunk))
                                 (equal? (th-bytes out) (list-ref c 4)))))
                     (else (th-adam-same?
                            (th-adam-run (list-ref c 5) n threads th-stress-min-chunk)
                            (list-ref c 6))))))
          (if ok
              (loop (+ k 1) (if (> T 1) (+ threaded 1) threaded))
              (values #f k))))))

(test-group "crunch threads - pool: back-to-back dispatches"
  (for-each
   (lambda (spin)
     (crunch-thread-pool-spin-us-set! spin)
     (let ((d (th-stat 'dispatches)) (w (th-stat 'wakeups)) (h (th-stat 'spin-hits)))
       (let-values (((ok threaded) (th-stress 3000)))
         (let ((label (string-append "spin " (number->string spin) " us: ")))
           (test-assert (string-append label "3000 mixed dispatches all correct") ok)
           (test (string-append label "the pool counts each threaded dispatch once")
                 threaded (- (th-stat 'dispatches) d))
           (case spin
             ((0) (test-assert (string-append label "sleeping workers were woken")
                    (> (th-stat 'wakeups) w)))
             ((1000) (when (>= (crunch-available-processors) 2)
                       (test-assert (string-append label "spinning workers found work")
                         (> (th-stat 'spin-hits) h)))))))))
   (list 0 th-initial-spin 1000))
  (crunch-thread-pool-spin-us-set! th-initial-spin)
  (test-assert "no more workers than the largest request of 8 threads needs"
    (<= (crunch-thread-pool-size) 7)))

(test-group "crunch threads - pool: fork"
  (define n 50000)
  (define in (th-vector 'f32 n))
  (define ref (th-bytes (th-reference 'sigmoid 'f32 in n)))
  (define (run threads)
    (let ((out (th-make 'f32 n 0.0)))
      (and (= threads (dispatch 'f32 n in out 'sigmoid threads 1))
           (equal? (th-bytes out) ref))))

  (test-assert "the parent has a pool before the fork"
    (and (run 8) (= 7 (crunch-thread-pool-size))))
  (test-assert "the child starts with no workers, then dispatches correctly on 4 threads"
    (th-in-child
     (lambda ()
       (and (= 0 (crunch-thread-pool-size))
            (run 4)
            (= 3 (crunch-thread-pool-size))
            (run 8)
            (= 7 (crunch-thread-pool-size))))))
  (test-assert "the child can shut its pool down"
    (th-in-child (lambda () (and (run 6) (= 5 (crunch-thread-pool-shutdown!))))))
  (test-assert "the parent's pool is intact after the children exit"
    (and (= 7 (crunch-thread-pool-size)) (run 8) (run 2))))

;; A forked child sends SIGUSR1 to this process count times, with a
;; pause between signals, while this process runs threaded dispatches.
(test-group "crunch threads - pool: signals during dispatch"
  (define n 200000)
  (define in (th-vector 'f32 n))
  (define ref (th-bytes (th-reference 'tanh 'f32 in n)))
  (define old-handler (signal-handler signal/usr1))

  (for-each
   (lambda (spin)
     (crunch-thread-pool-spin-us-set! spin)
     (let* ((received 0)
            (parent (current-process-id))
            (_ (set-signal-handler! signal/usr1
                                    (lambda (s) (set! received (+ received 1)))))
            (_ (flush-output))
            (pid (process-fork
                  (lambda ()
                    (emergency-exit
                     (handle-exceptions e 1
                       (do ((i 0 (+ i 1))) ((= i 50) 0)
                         (process-signal parent signal/usr1)
                         (do ((j 0 (+ j 1))) ((= j 20000))))))))))
       (let loop ((calls 0) (ok #t))
         (let-values (((p normal? status) (process-wait pid #t)))
           (if (and (= p 0) (< calls 100000))
               (let ((out (th-make 'f32 n 0.0)))
                 (dispatch 'f32 n in out 'tanh 8 1)
                 (loop (+ calls 1) (and ok (equal? (th-bytes out) ref))))
               (let ((label (string-append "spin " (number->string spin) " us: ")))
                 (test-assert (string-append label "the signalling child exited normally")
                   (and (= p (process-id pid)) normal? (= status 0)))
                 (test-assert (string-append label "every dispatch was correct") ok)
                 (test-assert (string-append label "dispatches ran while signals arrived")
                   (> calls 0))
                 (test-assert (string-append label "the handler ran on the CHICKEN thread")
                   (> received 0))))))))
   (list 0 th-initial-spin))
  (set-signal-handler! signal/usr1 old-handler)
  (crunch-thread-pool-spin-us-set! th-initial-spin))

(test-group "crunch threads - pool: garbage collection between dispatches"
  (test-assert "500 dispatches on fresh vectors amid garbage and collections"
    (let* ((n 20011)
           (in  (th-vector 'f32 n))
           (ref (th-bytes (th-reference 'sigmoid 'f32 in n))))
      (let loop ((k 0) (junk '()))
        (or (= k 500)
            (let* ((in2 (th-copy in))
                   (out (th-make 'f32 n -1.0))
                   (junk (if (= 0 (modulo k 50))
                             (begin (gc (= 0 (modulo k 100))) '())
                             (cons (make-vector 100 k) junk))))
              (dispatch 'f32 n in2 out 'sigmoid (+ 2 (modulo k 7)) 1)
              (and (equal? (th-bytes out) ref)
                   (loop (+ k 1) junk))))))))

(crunch-thread-pool-shutdown!)

(register-activation-backend! #f)
(test-exit)
