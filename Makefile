CC = gcc

SRC_DIR = tridiag
INC_DIR = tridiag/utils
BUILD_DIR = build

NUM_POINTS_PER_DIR ?= 260
NUM_TIMESTEPS ?= 20

OPT ?= -O3
CFLAGS ?=
LDLIBS ?= -lm

OUTPUT_VTK_FREQ ?= 0

DEFINES ?=

TIMER_LOG_FREQ ?= 1
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

HEADERS = $(INC_DIR)/timer.h $(INC_DIR)/write_vtk.h
TARGETS = naive prealloc fused stream blocked gather_dxx transpose_dxx transpose_dxx2

all: $(TARGETS)

$(TARGETS): %: $(BUILD_DIR)/%

$(BUILD_DIR)/%: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(ALL_CFLAGS) $(TARGET_CFLAGS) $(COMMON_DEFINES) -I$(INC_DIR) -o $@ $< $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/gather_dxx $(BUILD_DIR)/transpose_dxx $(BUILD_DIR)/transpose_dxx2: TARGET_CFLAGS = $(AVX2_FLAGS)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

.DELETE_ON_ERROR:
.PHONY: all clean $(TARGETS)
