;;; crunch-numvector-fix.scm
;;;
;;; Scheme-to-crunch numeric-vector converters for crunch's
;;; embedded mode.  Include this file before the first (crunch ...) form of
;;; any module that passes SRFI-4 vectors into crunch-compiled procedures.
;;;
;;; crunch's embedded foreign wrappers convert each numeric-vector argument
;;; with crunch_scheme_<t>vector() from crunch.h.  Under CHICKEN 6 a SRFI-4
;;; vector is a two-slot structure (type tag, bytevector), so the converter
;;; has to follow slot 1 to reach the element storage; it must also allocate
;;; a whole block descriptor and give it a reference count of one, so that
;;; the wrapper's closing crunch_unref() releases the descriptor (and never
;;; the element storage, which remains owned by the Scheme heap).

(foreign-declare #<<EOC
#define CRUNCH_EMBEDDED
#include "crunch.h"

#define am_crunch_define_numvector(t, et)                                  \
static crunch_ ## t ## vector am_crunch_scheme_ ## t ## vector(C_word vec) { \
  C_word bv = C_block_item(vec, 1);                                          \
  crunch_ ## t ## vector v =                                                 \
    (crunch_ ## t ## vector)malloc(sizeof(crunch_ ## t ## vector_block));    \
  v->rc    = 1;                                                              \
  v->flags = CRUNCH_BLOCK;                                                   \
  v->size  = C_header_size(bv);                                              \
  v->len   = v->size / sizeof(et);                                           \
  v->data  = (et *)C_data_pointer(bv);                                       \
  return v;                                                                  \
}

am_crunch_define_numvector(s32, int32_t)
am_crunch_define_numvector(s64, int64_t)
am_crunch_define_numvector(f32, float)
am_crunch_define_numvector(f64, double)

#undef crunch_scheme_intvector
#define crunch_scheme_s32vector am_crunch_scheme_s32vector
#define crunch_scheme_s64vector am_crunch_scheme_s64vector
#define crunch_scheme_f32vector am_crunch_scheme_f32vector
#define crunch_scheme_f64vector am_crunch_scheme_f64vector
#ifdef C_SIXTY_FOUR
# define crunch_scheme_intvector am_crunch_scheme_s64vector
#else
# define crunch_scheme_intvector am_crunch_scheme_s32vector
#endif
EOC
)

;;; Optional host-specific code generation.
;;;
;;; When the environment variable AM_CRUNCH_NATIVE is set to a value other
;;; than "" or "0" while a module is compiled, the C code that follows is
;;; compiled for the CPU of the build machine, as with -march=native.  The
;;; CPU name is obtained from the C compiler (the CC environment variable,
;;; or gcc) and passed to a GCC target pragma.  The resulting library runs
;;; only on CPUs that support the same instructions.  On CPUs with fused
;;; multiply-add the compiler may combine a multiplication and an addition
;;; into one instruction that rounds once, so results can differ in the
;;; last bits from those of a portable build.  am-crunch-native-build?
;;; expands to #t in such a build and to #f otherwise.

(import-for-syntax (only (chicken process-context) get-environment-variable)
                   (only (chicken process) call-with-input-pipe)
                   (only (chicken io) read-lines)
                   (only (chicken string) string-split))

(define-syntax am-crunch-native-build?
  (er-macro-transformer
   (lambda (form r c)
     (let ((flag (get-environment-variable "AM_CRUNCH_NATIVE")))
       (and flag (not (member flag '("" "0"))) #t)))))

(define-syntax am-crunch-native-target
  (er-macro-transformer
   (lambda (form r c)
     ;; The value of -march that the C compiler selects for this machine,
     ;; or #f if it cannot be determined.
     (define (host-arch)
       (let* ((cc (or (get-environment-variable "CC") "gcc"))
              (lines (call-with-input-pipe
                      (string-append cc " -march=native -Q --help=target 2>/dev/null")
                      read-lines)))
         (let loop ((lines lines))
           (and (pair? lines)
                (let ((words (string-split (car lines))))
                  (if (and (= (length words) 2) (string=? (car words) "-march="))
                      (cadr words)
                      (loop (cdr lines))))))))
     (let ((flag (get-environment-variable "AM_CRUNCH_NATIVE")))
       (if (and flag (not (member flag '("" "0"))))
           (let ((arch (host-arch)))
             (unless arch
               (error "AM_CRUNCH_NATIVE: cannot determine the host CPU from the C compiler"))
             `(,(r 'foreign-declare)
               ,(string-append "#pragma GCC target(\"arch=" arch "\")\n")))
           `(,(r 'begin)))))))

(am-crunch-native-target)
