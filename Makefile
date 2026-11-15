CC = gcc

NUM_POINTS_PER_DIR ?= 260
NUM_TIMESTEPS ?= 20

OPT ?= -O3
CFLAGS ?=
LDLIBS ?= -lm

OUTPUT_VTK_FREQ ?= 0

DEFINES ?=

TIMER_LOG_FREQ ?= 0
ifneq ($(filter-out 0,$(TIMER_LOG_FREQ)),)
TIMER_DEFINES = -DTIMER -DTIMER_LOG_FREQ=$(TIMER_LOG_FREQ)
else
TIMER_DEFINES =
endif

COMMON_DEFINES = \
	-DNUM_POINTS_PER_DIR=$(NUM_POINTS_PER_DIR) \
	-DNUM_TIMESTEPS=$(NUM_TIMESTEPS) \
	-DOUTPUT_VTK_FREQ=$(OUTPUT_VTK_FREQ) \
	$(TIMER_DEFINES) \
	$(DEFINES)

ALL_CFLAGS = $(OPT) $(CPPFLAGS) $(CFLAGS)

AVX2_FLAGS ?= -mavx2 -mfma

HEADERS = timer.h write_vtk.h
TARGETS = naive prealloc fused stream blocked gather_dxx transpose_dxx transpose_dxx2

all: $(TARGETS)

naive: naive.c $(HEADERS)
	$(CC) $(ALL_CFLAGS) $(COMMON_DEFINES) -o $@ $< $(LDFLAGS) $(LDLIBS)

prealloc: prealloc.c $(HEADERS)
	$(CC) $(ALL_CFLAGS) $(COMMON_DEFINES) -o $@ $< $(LDFLAGS) $(LDLIBS)

fused: fused.c $(HEADERS)
	$(CC) $(ALL_CFLAGS) $(COMMON_DEFINES) -o $@ $< $(LDFLAGS) $(LDLIBS)

stream: stream.c $(HEADERS)
	$(CC) $(ALL_CFLAGS) $(COMMON_DEFINES) -o $@ $< $(LDFLAGS) $(LDLIBS)

blocked: blocked.c $(HEADERS)
	$(CC) $(ALL_CFLAGS) $(COMMON_DEFINES) -o $@ $< $(LDFLAGS) $(LDLIBS)

gather_dxx: gather_dxx.c $(HEADERS)
	$(CC) $(ALL_CFLAGS) $(AVX2_FLAGS) $(COMMON_DEFINES) -o $@ $< $(LDFLAGS) $(LDLIBS)

transpose_dxx: transpose_dxx.c $(HEADERS)
	$(CC) $(ALL_CFLAGS) $(AVX2_FLAGS) $(COMMON_DEFINES) -o $@ $< $(LDFLAGS) $(LDLIBS)

transpose_dxx2: transpose_dxx2.c $(HEADERS)
	$(CC) $(ALL_CFLAGS) $(AVX2_FLAGS) $(COMMON_DEFINES) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGETS)

.DELETE_ON_ERROR:
.PHONY: all clean
