# Performance Optimization Lecture

## Requirements

- GCC
- AVX2-capable CPU for `gather_dxx`, `transpose_dxx`, and
  `transpose_dxx2`

## Build

```sh
make                    # build all solver versions
make all
make naive
make prealloc
make fused
make stream
make blocked
make gather_dxx
make transpose_dxx
make transpose_dxx2
make clean
```

Binaries are written to `build/`.

## Command-line options

```text
NUM_POINTS_PER_DIR=260  grid points per direction
NUM_TIMESTEPS=20        number of timesteps
OUTPUT_VTK_FREQ=0       write u-XXXX.vti every N steps (0 = off)
TIMER_LOG_FREQ=1        timer report interval (0 = off)
DEFINES="..."           extra compiler definitions, e.g. -DNU=0.01
CC=gcc                  compiler
OPT=-O3                 optimization flags
CFLAGS="..."            additional compiler flags
AVX2_FLAGS="-mavx2 -mfma"  flags for the vectorized versions
```

## Examples

```sh
make all
make blocked NUM_POINTS_PER_DIR=128 NUM_TIMESTEPS=50
make gather_dxx OUTPUT_VTK_FREQ=10
make naive TIMER_LOG_FREQ=5
make naive DEFINES="-DNU=0.01"
```

Run a solver, for example:

```sh
./build/naive
```

VTK files are written to the current directory.

## STREAM benchmark

The benchmark has its own Makefile:

```sh
cd stream_bench
make
./stream_bench
```

Use `DEFINES` for benchmark knobs, for example:

```sh
make DEFINES="-DSTREAM_ARRAY_SIZE=8388608 -DNTIMES=5"
```

## Notes

- The default grid is 260^3. Use a smaller grid for quick tests.
- `gather_dxx`, `transpose_dxx`, and `transpose_dxx2` require AVX2.
- `transpose_dxx` and `transpose_dxx2` require
  `NUM_POINTS_PER_DIR` to be a multiple of 4.
- `transpose_dxx2` processes two `y`-tiles concurrently in the `Dxx`
  sweep.
- `make` does not rebuild when only a make variable changes. Use
  `make -B`, or `make clean`, after changing such a variable.
