# array-morphisms-crunch

Activation, element-wise, reduction, BLAS, convolution and optimizer kernels for
[array-morphisms](../array-morphisms), compiled to C by CHICKEN's
[crunch](http://wiki.call-cc.org/eggref/6/crunch) embedded compiler.

Crunch compiles a statically typed subset of Scheme to C. The kernels
here are therefore written in Scheme, but they run without type
dispatch, boxing or allocation in their inner loops.

## Modules

| Module | Contents |
|---|---|
| `array-morphisms-crunch-activations` | f32/f64 kernels for `relu`, `sigmoid`, `tanh` and their SSA derivative ops, and for `exp`, `log`, `sqrt`, `negate`, `abs`; row softmax (`crunch-softmax-rows-f32`/`-f64`) and interleaved rotary embedding (`crunch-rope-f32`/`-f64`); binary kernels for `add`, `sub`, `mul`, `div`; axis-0 and axis-1 reductions (`sum`, `mean`, `max`, `min`); a strided copy; `make-crunch-activation-backend` and `make-crunch-threaded-activation-backend` |
| `array-morphisms-crunch-threads` | the POSIX-thread dispatchers for chunk kernels (`crunch-dispatch-f32`, `crunch-dispatch-f64`, `crunch-dispatch4-f32`, `crunch-dispatch4-f64`, and `crunch-dispatch-n` for generated kernels), their settings, and the persistent worker pool they share |
| `array-morphisms-crunch-blas-backend` | `crunch-{s,d}gemm`, `-gemm-strided`, `-gemv`, `-dot`, `-axpy` with the normalized signatures of `array-morphisms-blas-exec`; `crunch-native-build?` |
| `array-morphisms-crunch-conv-backend` | NCHW/NHWC `im2col`, `col2im`, `bias-add`, the six convolution hooks, and `make-crunch-blas-backend` (the complete `blas-backend` record) |
| `array-morphisms-crunch-optimizers` | `crunch-adam-f32` and `crunch-adam-f64`, whole-vector Adam updates for nanograd's optimizer-kernel registry |
| `array-morphisms-crunch-codegen` | a segop backend that generates, compiles and runs a kernel for each segop of a replay plan (see Code generation) |
| `array-morphisms-crunch-translate` | the translation of segop bodies into crunch Scheme, shared by the generated kernels |

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

When the array-morphisms SSA replay plan is compiled, each
element-wise binding is lowered to a segop (an `ri-segop` instruction
of array-morphisms). A unary binding whose body is a registered
activation op applied to a row-major input of the output's shape gets
the `segop/activation-unary` executor. At execution time that executor
asks the active activation backend for a kernel for its op and dtype,
and processes the whole array in one call. If no backend is
registered, if the dtypes differ, or if the dtype is not f32 or f64,
the executor applies the op's Scheme combiner per element instead, as
the `segop/flat-unary` executor does.

The kernels are generated from the scalar-op table of array-morphisms
(`scalar-ops-table.scm`), which also supplies the Scheme combiners, so each
kernel uses the same formula as the combiner it replaces. crunch computes in
double precision, like the Scheme fallback, so results are bit-identical,
including for signed zeros, infinities and NaN. The tests check this with
`eqv?`.

Bindings produced by the element-wise fusion pass have a composed body
rather than a single op. They do not use activation kernels.

The backends hold three more kinds of kernels, which array-morphisms uses in
the same way:

- **Binary ops.** The backend constructors register `add`, `sub`, `mul` and
  `div` with `register-binary-op!`, so that element-wise bindings with these
  ops and two row-major operands of the output's shape get the
  `segop/activation-binary` executor instead of `segop/flat-binary`. The threaded backend
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

## Unary math kernels

The table also has entries for `exp`, `log`, `sqrt`, `negate` and `abs`.
These names are the op names that `morph-exp`, `morph-log`, `morph-sqrt`,
`morph-negate` and `morph-abs` give their nodes, so once the backend is
registered the replay compiler runs these ops in compiled SSA programs with
the crunch kernels. Their results are bit-identical to the Scheme
combiners, except that `log` and `sqrt` of a negative number give NaN where
Scheme gives a complex number, which an f32 or f64 array cannot hold.

## Fused kernels

Both backends also hold fused kernels. Each fused kernel computes a
whole segop body of several table ops in one pass over its inputs,
instead of one kernel per op with an intermediate array between
them. 


| Body | Where the fusion pass forms it |
|---|---|
| `(mul (in 0) (cast f32 (relu-deriv (in 1))))` | gradient times relu derivative |
| `(mul (in 0) (cast f32 (sigmoid-deriv (in 1))))` | gradient times sigmoid derivative |
| `(mul (in 0) (cast f32 (tanh-deriv (in 1))))` | gradient times tanh derivative |
| `(add (in 0) (cast f32 (mul (in 1) (in 2))))` | `x*y + z` |
| `(negate (cast f32 (mul (in 0) (in 1))))` | negated product |
| `(exp (cast f32 (negate (in 0))))` | `exp(-x)` |
| `(mul (in 0) (cast f32 (sigmoid (in 0))))` | `silu(a)`, the first step towards the next body |
| `(mul (in 0) (cast f32 (mul (in 1) (cast f32 (sigmoid (in 1))))))` | `b * silu(a)`, Llama's feed-forward block |

The fusion pass of array-morphisms fuses bindings into one of these bodies
only when the active backend holds its kernel. It then runs it as
`segop/fused-kernel`. Without a crunch backend registered, the pass makes
no such fusion, and plans are unchanged.

- Exact rounding: Each body is written for f32, with `(cast f32 e)`
  where the unfused program would have stored an intermediate f32 value.
  The f32 kernel rounds there with a C `(double)(float)` conversion,
  reached through `c-lambda`. The extension is compiled with
  `-fno-semantic-interposition`, so that the C compiler inlines that
  conversion into the kernel loops. The f64 kernel is the same body without
  casts. 
- Canonical bodies: Bodies are in the canonical form of
  `canonical-scalar-body` (array-morphisms `segop.scm`), under which the
  pass and the replay compiler look kernels up. `crunch-fused-bodies`
  returns them.
- Adding a body: Add it to the `define-crunch-fused` form in
  `crunch-activation-kernels.scm`, in canonical form, with one to three
  inputs. The bodies the pass forms for a program are listed by
  `ssa-fusion-pass-stats` (`candidates`) under the fusion policy `'all`.

### Fused column sums

`define-crunch-fused` can also generate *column-sum kernels*, from bodies
listed in a final `(colsum ...)` clause. Each one computes a body over
row-major rows x cols inputs, stores the values, and sums each column,
adding the rows in order as the unfused sum does. The fusion pass uses
such a kernel for a map and the axis-0 sum that reads it.

The catalogue currently has no column-sum bodies. For the bias gradient of
a relu layer, the one candidate, the fused kernel was no faster than the
map kernel followed by the separate sum, because the sum reads the map's
values while they are still in cache. A column-sum kernel must also run
on one thread, since splitting the rows would change the order of the
additions.

## Code generation

The fused kernels above are a fixed catalogue. The module
`array-morphisms-crunch-codegen` instead writes a kernel for each segop
of a replay plan, from the segop's body and index maps, and compiles it
with crunch. It is a segop backend of kind `codegen`. Register it after
the other backends:

```scheme
(import array-morphisms-crunch-codegen)
(register-crunch-codegen-backend!)
```

It is placed after the registry backend, so the BLAS and catalogue
kernels keep the segops they already run, and before the Scheme backend,
so it takes many of the segops that would otherwise run as Scheme loops.

What it runs:

- element-wise maps, with one or several outputs, over spaces of up to
  rank 6, whose operands are read through identities, broadcasts,
  transposes and other strided views;
- reductions over spaces of up to rank 8 that combine with add, mul, max
  or min, with the map that the fusion pass put inside them and the
  final step it put after them;
- matrix products with an epilogue: the product is computed by the BLAS
  backend, and the epilogue runs as a generated kernel over the stored
  product.

The operands may be f32, f64, s32, s64 or u32, mixed as the body needs.
A kernel is fixed in the rank of its space, its bodies and its dtypes;
extents, offsets and strides are passed when it runs, so one kernel
serves every batch size and layout.

**Results.** A generated kernel computes the same values as the Scheme
code it replaces, bit for bit. It rounds intermediate values where the
body says so, adds in the same order, and follows Scheme's arithmetic
on mixed integers and floats. A reduction keeps each output point's
additions in order; it computes up to 8 neighbouring output points
together, each with its own accumulator, so that sums over the rows of
a matrix read memory in order. When an integer result would leave the
range of its dtype, the kernel stops and the segop runs on the host for
that call. Kernels are split across threads like the threaded
activation kernels, and their results do not depend on the number of
threads.

**When kernels are compiled.** The kernels of a plan are compiled
together, as one module, when the plan is compiled, which is at the
second step of a program run through `ssa-realize/ctx`. This takes
under a second for a small network. Compiled modules are kept in a
cache directory, under a name derived from their source, so later
processes load them instead of compiling them again.

**Requirements.** Compiling needs `csc` and the C compiler it uses at
run time. If a module cannot be compiled, its segops run on the
backends after the code generator, and a warning is written.

**Settings.**

| Setting | Meaning |
|---|---|
| `crunch-codegen` | `#t` (the default) to use the backend; the environment variable `AM_CRUNCH_CODEGEN=0` makes the default `#f` |
| `crunch-codegen-fusions` | the fusion classes whose fused segops the backend reports as native to the fusion pass; default `(redomap epilogue)`, or the classes named in `AM_CRUNCH_CODEGEN_FUSIONS` |
| `crunch-codegen-threads` | the number of threads a kernel runs on; `#f` (the default) follows the activation backend: the crunch thread count with the threaded backend, otherwise one |
| `crunch-codegen-cache-directory` | the cache directory; default `AM_CRUNCH_CODEGEN_CACHE`, else `$XDG_CACHE_HOME/array-morphisms/kernels`, else `~/.cache/array-morphisms/kernels`; `#f` keeps no cache |
| `crunch-codegen-csc` | the `csc` program used to compile |

The fusion classes come from array-morphisms: `map` (a map into a map),
`kept`, `horizontal`, `redomap` (a map into a reduction), `epilogue` (a
map after a reduction) and `gemm-epilogue`. Under the default fusion
policy, the pass fuses only what some backend reports as native. The
default classes fuse passes over the same data into one kernel. A map
fused into a map can take the producer of a matrix product's epilogue,
so that class is left to the catalogue unless asked for. Under the
policy `'all`, every legal fusion is made, and the code generator runs
the fused segops of every class.

**Reports.** The timing profile and `replay-plan-stats` name the
generated kernels `segop/codegen-map`, `segop/codegen-red` and
`segop/codegen-epilogue`. `replay-plan-stats` also reports, under
`crunch-codegen`, how many of the plan's segops became kernels, and how
many of those were compiled, loaded from the cache, or could not be
compiled. `crunch-codegen-stats` gives the same totals for the whole
process, and `crunch-codegen-forget-kernels!` makes the next plan load
or compile its kernels again.

## Row softmax and rotary embedding

These kernels have no morphism op. They are called directly, on
preallocated vectors, by code that works outside compiled SSA programs,
such as attention over a key/value cache whose length grows with each
token.

- `(crunch-softmax-rows-f32 rows cols src out)` replaces each row of the
  row-major `rows` x `cols` array `src` by its softmax and stores it in
  `out`, which may be `src`. The row maximum is subtracted before `exp`, so
  large inputs do not overflow; the sum is accumulated in double precision.
- `(crunch-rope-f32 n head-size base cos sin x)` rotates the first `n`
  elements of `x` in place as consecutive pairs `(x[i], x[i+1])`. Pair `p`
  of each head of `head-size` elements is rotated by the angle with cosine
  `cos[base + p]` and sine `sin[base + p]`. This is the interleaved rotary
  position embedding of Llama, with `base` selecting the table row of the
  token position.

The `-f64` variants take f64vectors. Both kernels check their arguments
against the vector lengths and raise an error when they do not fit.

## Threaded activation kernels

Each activation table entry also yields two *chunk kernels*,
`am_crunch_<op>_chunk_f32` and `am_crunch_<op>_chunk_f64`. Hyphens in
`op` become underscores, in the Crunch-emitted C functions. A chunk
kernel computes the entry for elements `[start, end)`.

The threaded backend passes the chunk kernel's C address to
`crunch-dispatch-f32` or `crunch-dispatch-f64`. The dispatcher splits
`[0, n)` into `T` nearly equal contiguous chunks and runs the first chunk on
the calling thread. It hands each other chunk to a worker of a persistent
POSIX thread pool and waits until all are done. `T` is
`max(1, min(threads, n div min-chunk, 64))`, so an array shorter than two
chunks runs on the calling thread alone. The chunks are disjoint and every
element is computed on its own, so the result is the same for every thread
count.

The pool (`crunch-thread-pool.h`) is created at the first dispatch with
`T > 1`. It grows to the largest `T` requested, up to 63 workers, and keeps
its workers until `(crunch-thread-pool-shutdown!)` or the end of the
process. A worker that has finished a chunk spins for a short time, so the
next kernel of a replay finds it awake, and then sleeps on a condition
variable. The calling thread waits for the workers in the same way.
Spinning is turned off for a dispatch that requests more threads than there
are processors online. A child process created by `fork` starts with an
empty pool and creates it again at its first threaded dispatch.

Why the threads are safe to use with CHICKEN:
- **No CHICKEN heap access.** The workers only run the crunch-generated C
  function on vector descriptors built on the dispatcher's stack. They do not
  touch the CHICKEN heap or call into the CHICKEN runtime.
- **No reference counting.** The descriptors have a reference count of -1,
  which Crunch never changes.
- **No GC during the call.** The calling thread stays inside the one foreign
  call until every chunk has finished, so the vectors cannot move.
- **Signals stay on the CHICKEN thread.** All signals are blocked in the
  workers.
- **Thread-creation failure is harmless.** If a worker cannot be created, its
  chunk runs on the calling thread instead.

This backend has two configurable parameters, read at every call:

- `(crunch-thread-count)` is the number of threads requested. It defaults to
  the environment variable `AM_CRUNCH_THREADS` if that is set, and otherwise
  to the number of online processors, at most 8.
- `(crunch-thread-min-chunk)` is the smallest number of elements given to one
  thread. It defaults to 4096.

The pool has its own setting and introspection procedures:

- `(crunch-thread-pool-spin-us)` is the number of microseconds a waiting
  thread spins before it sleeps. It defaults to the environment variable
  `AM_CRUNCH_SPIN_US` if that is set, and otherwise to 20; 0 disables
  spinning. `(crunch-thread-pool-spin-us-set! us)` changes it.
- `(crunch-thread-pool-size)` is the number of workers.
- `(crunch-thread-pool-stats)` returns counters for tuning: pool dispatches,
  chunks found by a spinning worker, chunks found after sleeping, and
  dispatches whose caller had to sleep.

The system BLAS backend's OpenBLAS has a thread pool of its own. The two
are never busy at the same time, since a replay runs one kernel at a time;
`AM_CRUNCH_THREADS` and `OPENBLAS_NUM_THREADS` set their sizes
separately.

Wall-clock times on a 16-core machine, f32, milliseconds per call
(`tests/bench-threaded-activations.scm`):

| op, n | Scheme combiner | crunch, 1 thread | crunch, 4 threads | crunch, 8 threads |
|---|---|---|---|---|
| sigmoid, 400K | 75.9 | 3.28 | 1.42 | 1.36 |
| tanh, 400K | - | 4.65 | 1.58 | 1.53 |
| relu, 400K | - | 1.67 | 0.72 | 0.82 |
| sigmoid, 4M | 669 | 31.1 | 10.4 | 11.2 |

These times were measured before the thread pool, when every call
created and joined its threads at a cost of some tens of microseconds.
Because of that cost, a second thread only paid off at about 16K
elements per thread, and `crunch-thread-min-chunk` defaulted to 16384. relu is
limited by memory bandwidth and gains the least.

With the pool, a dispatch costs about 2 microseconds. Microseconds per
pair of f32 kernels (sigmoid then relu), best of 5, on the same machine:

| n | 1 thread | 2 threads | 4 threads | 8 threads | 8 threads, no pool |
|---|---|---|---|---|---|
| 4,096 | 29 | 20 | 14 | 12 | 450–520 |
| 100,352 | 700 | 370–410 | 220–260 | 200–230 | 670–700 |
| 802,816 | 5,500–6,300 | 3,100–3,400 | 1,800–2,200 | 1,600–1,700 | 2,000–2,200 |
| 4,194,304 | 30,000–33,000 | 17,500–21,600 | 11,800–13,300 | 10,000–10,400 | 10,400–11,200 |

A second thread now pays off at about 750 elements per thread for
sigmoid and about 4K for relu, so `crunch-thread-min-chunk` defaults to
4096.

## BLAS and convolution kernels

- GEMM operands follow the microBLAS shim's storage convention: `lda` is
  the row stride of the stored matrix, and a transposed operand is stored
  transposed. An operand that is not plain row-major is first packed into a
  row-major copy. Packing buffers are kept and reused by later calls, so
  once they have grown to the sizes in use a product allocates nothing.
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
kernels), BLAS, convolution, threading, optimizer and code generation
suites. The code generation tests compare every generated kernel with the
reference interpreter of array-morphisms and with the Scheme executor it
replaces, bit for bit.
`tests/bench-threaded-activations.scm` is an informational benchmark;
compile it with `csc -O3`.
