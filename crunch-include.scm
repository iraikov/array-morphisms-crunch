;;; crunch-include.scm
;;;
;;; Prints the C compiler options that let csc find crunch.h, for the
;;; custom-config entries of array-morphisms-crunch.egg.
;;;
;;; The crunch egg installs crunch.h with c-include.
;;  As of CHICKEN 6.1.0, chicken-install puts
;;; such files in <prefix>/include/chicken by default, but in
;;; $CHICKEN_INSTALL_PREFIX/include when that variable is set, and csc
;;; searches only <prefix>/include/chicken and the directories of
;;; CHICKEN_C_INCLUDE_PATH.  This script reads the location of crunch.h
;;; from crunch's egg-info file in the repository, and prints one
;;; ("-C" "-I<dir>") option pair for it.  It prints nothing when crunch.h
;;; cannot be found, which leaves csc's own search path in effect.

(import (scheme base) (scheme write) (scheme file)
        (chicken platform) (chicken pathname)
        (chicken process-context))

;; The directories that may hold crunch.h: the one recorded in crunch's
;; egg-info files, then the include directories of an installation
;; prefix given in the environment.
(define (candidate-directories)
  (define (egg-info-directories repo)
    (let ((info (make-pathname repo "crunch" "egg-info")))
      (if (file-exists? info)
          (let* ((data (call-with-input-file info read))
                 (files (cond ((and (pair? data) (assq 'installed-files data)) => cdr)
                              (else '()))))
            (let loop ((fs files) (dirs '()))
              (cond ((null? fs) (reverse dirs))
                    ((and (string? (car fs))
                          (equal? (pathname-strip-directory (car fs)) "crunch.h"))
                     (loop (cdr fs) (cons (pathname-directory (car fs)) dirs)))
                    (else (loop (cdr fs) dirs)))))
          '())))
  (append
   (apply append (map egg-info-directories (repository-path)))
   (let ((prefix (get-environment-variable "CHICKEN_INSTALL_PREFIX")))
     (if prefix
         (list (make-pathname prefix "include")
               (make-pathname (list prefix "include") "chicken"))
         '()))))

(let loop ((dirs (candidate-directories)))
  (cond ((null? dirs) #t)
        ((file-exists? (make-pathname (car dirs) "crunch.h"))
         (write (list "-C" (string-append "-I" (car dirs))))
         (newline))
        (else (loop (cdr dirs)))))
