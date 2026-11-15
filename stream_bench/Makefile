CC = gcc

OPT ?= -O2
CFLAGS ?= -march=native -fopenmp -fno-tree-loop-distribute-patterns
LDLIBS ?= -lm
DEFINES ?=

stream_bench: stream_bench.c
	$(CC) $(OPT) $(CFLAGS) $(DEFINES) -o $@ $< $(LDLIBS)

clean:
	rm -f stream_bench

.PHONY: clean
