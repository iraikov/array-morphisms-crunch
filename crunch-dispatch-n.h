/* crunch-dispatch-n.h
 * The job record of crunch-dispatch-n, shared by the dispatcher and the
 * kernels that array-morphisms-crunch-codegen generates.
 *
 * A generated kernel has the form
 *
 *   kernel(start, end, ip, sc, v0, ..., v(n-1))
 *
 * where ip is an s64vector of integer parameters (extents, offsets and
 * strides), sc an f64vector of scalar parameters and v0 ... v(n-1) are
 * numeric vectors whose element types are fixed in the kernel.  Each
 * kernel comes with a run procedure of type am_pool_run, which takes job
 * k of an array of am_jobn records and calls the kernel with that job's
 * arguments, converting each vector to its crunch type.
 *
 * Every crunch numeric vector descriptor has the layout of
 * crunch_vector_block, so the dispatcher describes all vectors with that
 * type, and the run procedure casts each one to the type the kernel
 * expects.
 */

#ifndef CRUNCH_DISPATCH_N_H
#define CRUNCH_DISPATCH_N_H

#define AM_DN_MAX_VECS 16

typedef struct {
  crunch_integer start, end;
  crunch_s64vector ip;
  crunch_f64vector sc;
  crunch_vector_block *v[AM_DN_MAX_VECS];
} am_jobn;

#endif
