# array-morphisms-crunch

Activation, element-wise, reduction, BLAS, convolution and optimizer kernels for
[array-morphisms](../array-morphisms), compiled to C by CHICKEN 6's
[crunch](http://wiki.call-cc.org/eggref/6/crunch) embedded compiler.

Crunch compiles a statically typed subset of Scheme to C while the
surrounding `.scm` file is being compiled. The kernels here are therefore
written in Scheme, but they run without type dispatch, boxing or allocation
in their inner loops. 

## Modules

| Module | Contents |
|---|---|
| `array-morphisms-crunch-activations` | f32/f64 kernels for `relu`, `sigmoid`, `tanh` and their SSA derivative ops; binary kernels for `add`, `sub`, `mul`, `div`; axis-0 and axis-1 reductions (`sum`, `mean`, `max`, `min`); a strided copy; `make-crunch-activation-backend` and `make-crunch-threaded-activation-backend` |
| `array-morphisms-crunch-threads` | the POSIX-thread dispatchers for chunk kernels (`crunch-dispatch-f32`, `crunch-dispatch-f64`, `crunch-dispatch4-f32`, `crunch-dispatch4-f64`) and their settings |
| `array-morphisms-crunch-blas-backend` | `crunch-{s,d}gemm`, `-gemm-strided`, `-gemv`, `-dot`, `-axpy` with the normalized signatures of `array-morphisms-blas-exec`; `crunch-native-build?` |
| `array-morphisms-crunch-conv-backend` | NCHW/NHWC `im2col`, `col2im`, `bias-add`, the six convolution hooks, and `make-crunch-blas-backend` (the complete `blas-backend` record) |
| `array-morphisms-crunch-optimizers` | `crunch-adam-f32` and `crunch-adam-f64`, whole-vector Adam updates for nanograd's optimizer-kernel registry |

## Usage

```scheme
(import array-morphisms-activation-exec
        array-morphisms-crunch-activations)
(register-activation-backend! (make-crunch-activation-backend))

(import array-morphisms-realization
        array-morphisms-crunch-conv-backend)
(register-blas-backend! (make-crunch-blas-backend))
```

To split large arrays across threads, register the threaded backend
instead of the unthreaded one; its results are identical:

```scheme
(register-activation-backend! (make-crunch-threaded-activation-backend))
```

The two registrations are independent. The activation backend can be
combined with any BLAS backend, including microBLAS and the system-BLAS
backend of `array-morphisms-blas`.

nanograd's Adam optimizer uses the Adam kernel once it is registered:

```scheme
(import nanograd-optimizer array-morphisms-crunch-optimizers)
(register-optimizer-kernel! 'adam 'f32 crunch-adam-f32)
(register-optimizer-kernel! 'adam 'f64 crunch-adam-f64)
```

## Building

The C code is compiled with `-O3 -DNDEBUG`. `NDEBUG` removes crunch's
per-element index assertions; the Scheme wrappers check vector lengths
before every call instead.

Setting `AM_CRUNCH_NATIVE=1` while building compiles the kernels for the
CPU of the build machine, as with `-march=native`:

```bash
rm -f *.so *.o *.link *.import.scm
AM_CRUNCH_NATIVE=1 chicken-install -test
```

The build files must be removed first, because `chicken-install` does not
recompile a module whose outputs are newer than its sources. Such a build
runs only on CPUs with the same instructions. It may use fused
multiply-add, which rounds once where the portable build rounds twice, so
its results can differ from those of a portable build in the last bits;
they still do not depend on the number of threads.
`crunch-native-build?` tells which kind of build is installed.

## How activation kernels are used

When the SSA replay plan is compiled, a unary element-wise binding whose op
is a registered activation op becomes a `ri-activation-unary` instruction.
At execution time that instruction asks the active activation backend for a
kernel for its op and dtype, and processes the whole array in one call. If
no backend is registered, if the dtypes differ, or if the dtype is not f32 or
f64, the instruction applies the op's Scheme combiner per element instead,
as `ri-flat-unary` does.

The kernels use the same formulas as the combiners they replace. crunch
computes in double precision, like the Scheme fallback, so results are
bit-identical, including for signed zeros, infinities and NaN. The tests
check this with `eqv?`.

Bindings produced by the element-wise fusion pass keep the consumer's op name
but carry a composed combiner. They do not use activation kernels.

The backends hold three more kinds of kernels, which array-morphisms uses in
the same way:

- **Binary ops.** The backend constructors register `add`, `sub`, `mul` and
  `div` with `register-binary-op!`, so that element-wise bindings with these
  ops and two row-major operands of the output's shape compile to
  `ri-activation-binary` instead of `ri-flat-binary`. The threaded backend
  splits these kernels across threads like the activations.
- **Reductions.** `execute-reduction-morphism` uses a reduction kernel for a
  row-major 2-D array reduced over axis 0 or 1. Each kernel rounds exactly
  as the Scheme fast path it replaces: over axis 0 an f32 sum is rounded
  after every addition, over axis 1 it is kept in double precision.
- **Strided copies.** Copies of transposed or offset views of rank at most 4
  into row-major order, such as the copies of transposed gradient outputs,
  use the copy kernel.

All of these give results bit-identical to the Scheme code they replace.

## Adding an activation

Kernels come from the table in `crunch-activation-kernels.scm`. Each entry
has the form `(op (v) expr)`:

```scheme
(define-crunch-activations
  (relu       (x) (if (> x 0.0) x 0.0))
  (relu-deriv (x) (if (> x 0.0) 1.0 0.0))
  ...)
```

Each entry defines `crunch-<op>-f32` and `crunch-<op>-f64`. The backend
constructor registers `op` with `register-activation-op!`. `expr` may use
only `v`, numeric literals and crunch primitives. crunch procedures cannot
refer to Scheme variables, so a constant such as a fixed slope must be
written as a literal, and each distinct value needs its own entry.

### Limitations

A table entry supplies only a fast execution path for an op the SSA engine
already knows. Before an op can appear in a differentiable SSA training
graph, array-morphisms also needs the following three components:

- a `morph-<op>` constructor (via `create-unary-morphism` in `basic-ops.scm`);
- a VJP rule in `ssa-vjp` (`ssa.scm`), which emits the derivative binding;
- a matching case in `rebuild-morphism` (`ssa.scm`).

An entry without these is correct and testable on its own, but no training
graph ever routes to it.

Each derivative is its own unary op, because `ssa-vjp` chooses its input:
`relu-deriv` reads the forward input, while `sigmoid-deriv` and `tanh-deriv`
read the forward output. A derivative that needs two arrays, such as that of
swish (the input and an intermediate), cannot be expressed as a table entry.

## Threaded activation kernels

Each activation table entry also yields two *chunk kernels*,
`am_crunch_<op>_chunk_f32` and `am_crunch_<op>_chunk_f64`. Hyphens in
`op` become underscores, in the Crunch-emitted C functions. A chunk
kernel computes the entry for elements `[start, end)`.

The threaded backend passes the chunk kernel's C address to
`crunch-dispatch-f32` or `crunch-dispatch-f64`. The dispatcher splits
`[0, n)` into `T` nearly equal contiguous chunks and runs the first chunk on
the calling thread. It runs each other chunk on a new POSIX thread, then
joins them all. `T` is `max(1, min(threads, n div min-chunk, 64))`, so an
array shorter than two chunks never creates a thread. The chunks are
disjoint and every element is computed on its own, so the result is the
same for every thread count.

Why the threads are safe to use with CHICKEN:
- **No CHICKEN heap access.** The workers only run the crunch-generated C
  function on vector descriptors built on the dispatcher's stack. They do not
  touch the CHICKEN heap or call into the CHICKEN runtime.
- **No reference counting.** The descriptors have a reference count of -1,
  which Crunch never changes.
- **No GC during the call.** The calling thread stays inside the one foreign
  call until every worker has been joined, so the vectors cannot move.
- **Signals stay on the CHICKEN thread.** All signals are blocked in the
  workers.
- **Thread-creation failure is harmless.** If a thread cannot be created, its
  chunk runs on the calling thread instead.

This backend has two configurable parameters, read at every call:

- `(crunch-thread-count)` is the number of threads requested. It defaults to
  the environment variable `AM_CRUNCH_THREADS` if that is set, and otherwise
  to the number of online processors, at most 8.
- `(crunch-thread-min-chunk)` is the smallest number of elements given to one
  thread. It defaults to 16384.

Wall-clock times on a 16-core machine, f32, milliseconds per call
(`tests/bench-threaded-activations.scm`):

| op, n | Scheme combiner | crunch, 1 thread | crunch, 4 threads | crunch, 8 threads |
|---|---|---|---|---|
| sigmoid, 400K | 75.9 | 3.28 | 1.42 | 1.36 |
| tanh, 400K | - | 4.65 | 1.58 | 1.53 |
| relu, 400K | - | 1.67 | 0.72 | 0.82 |
| sigmoid, 4M | 669 | 31.1 | 10.4 | 11.2 |

Threads are created and joined on every call; there is no thread pool.
This costs some tens of microseconds per call, which is why a second
thread only pays off at about 16K elements per thread. relu is limited by
memory bandwidth and gains the least.

## BLAS and convolution kernels

- GEMM operands follow the microBLAS shim's storage convention: `lda` is
  the row stride of the stored matrix, and a transposed operand is stored
  transposed. An operand that is not plain row-major is first packed into a
  row-major copy.
- The product is computed by a kernel for contiguous operands that takes the
  columns of C eight at a time and keeps the eight partial sums in local
  double-precision variables. The rows of C are shared among up to
  `(crunch-thread-count)` threads, with at least 65536 multiply-adds per
  thread.
- Each element of C is a sum over k in increasing order, in double
  precision for f32 as well as f64, so the result does not depend on the
  number of threads and, in a portable build, equals that of a naive loop.
- As in reference BLAS, `C` is not read when `beta` is 0, and the values of
  `A` and `B` do not affect the result when `alpha` is 0.
- The convolution kernels follow the column layout of `kernels/im2col.c` in
  array-morphisms: `[N*OH*OW, C*KH*KW]`, with columns ordered `(c, kh, kw)`
  for both NCHW and NHWC images, and call the GEMM above.

Single-precision GEMM throughput on an 8-core Ryzen 7 7730U, in GFLOP/s,
for the shapes of a small CNN:

| shape (M x K x N) | 1 thread | 8 threads | 8 threads, `AM_CRUNCH_NATIVE=1` |
|---|---|---|---|
| 6272 x 144 x 32 | 7.3 | 32.0 | 43.8 |
| 1568 x 288 x 64 | 7.3 | 29.7 | 41.8 |
| 3136 x 32 x 128 | 7.1 | 25.7 | 37.0 |
| 32 x 3136 x 128 | 5.4 | 17.9 | 23.9 |

## Generic thread dispatcher

`crunch-dispatch4-f32` and `crunch-dispatch4-f64` run chunk kernels of the
form `(start end i0 i1 i2 scal v0 v1 v2 v3)`: three integers passed through
unchanged, an f64vector of scalar parameters, and up to four vectors of the
element type (a kernel that needs fewer ignores the rest). They split
`[0, n)` like `crunch-dispatch-f32` and follow the same safety rules. The
GEMM, binary and Adam kernels all use this calling convention.

## Notes on crunch 0.992

Four problems in crunch's embedded mode affect this egg. The egg works
around them without modifying crunch:

- **Numeric-vector arguments.** In CHICKEN 6 an SRFI-4 vector is a two-slot
  structure, but crunch's `crunch_scheme_<t>vector` converters read the
  structure itself as the element block. They also allocate too little memory
  for the block descriptor (`sizeof` of the pointer type) and leave its
  reference count uninitialised. The result is heap corruption at the first
  call. `crunch-numvector-fix.scm` defines corrected converters and redirects
  the generated wrappers to them. Every module includes it before its first
  `(crunch ...)` form.
- **`float` arguments and results** cross the generated wrapper as C `float`,
  which truncates them to single precision. Double-precision scalars (`alpha`,
  `beta`, dot products) are therefore passed in and returned through
  one-element f64vectors.
- **Index checks** in crunch are C `assert`s, which abort the process. The
  egg is compiled with `NDEBUG`, which removes them; the Scheme wrappers
  check vector lengths first and raise an ordinary error.
- **`abs` on a float** is compiled to C's integer `abs()`, which truncates
  its argument (`(abs -3.5)` gives 3). Kernels must use `fpabs` from
  `(chicken flonum)`, which compiles to `fabs`.

Also, crunch requires a type declaration `(: (name argtype ...) rettype)`
for each exported procedure, and its `+` and `*` take exactly two arguments.

## Tests

`tests/feature/` holds standalone checks of the crunch features the egg relies
on:
- vector arguments, local bindings and transcendental functions;
- the padding-skip loop structure;
- generating several `(crunch ...)` forms from one macro;
- scalar precision;
- calling a crunch chunk kernel from several POSIX threads.

They are kept as regression checks against future crunch or CHICKEN releases.
`tests/run.scm` runs the activation (including the binary, reduction and copy
kernels), BLAS, convolution, threading and optimizer suites.
`tests/bench-threaded-activations.scm` is an informational benchmark;
compile it with `csc -O3`.
