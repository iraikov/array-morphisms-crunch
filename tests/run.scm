;;; tests/run.scm
;;; Run all test-*.scm suites of array-morphisms-crunch in a single process
;;; and report a summary.
;;; Usage: csi -s tests/run.scm   (from the egg root)
;;;        csi -s run.scm         (from within tests/)
;;;
;;; The crunch smoke tests in tests/smoke/ must be compiled, so they are run
;;; separately with tests/smoke/run.sh.

(import scheme (scheme base) (chicken base) test)

;;; Override test-exit so included files do not terminate the process early.
;;; test-failure-count is a global parameter in the test egg that accumulates
;;; across all included files; we read it once at the end.
(define test-exit (lambda () (values)))

(include-relative "test-crunch-activations.scm")
(include-relative "test-crunch-blas-backend.scm")
(include-relative "test-crunch-conv-backend.scm")
(include-relative "test-crunch-threads.scm")
(include-relative "test-crunch-optimizers.scm")

(exit (min 255 (test-failure-count)))
