NVIDIA := $(HOME)/.local/lib/python3.13/site-packages/nvidia
GPU_DIRS := $(NVIDIA)/cuda_runtime/lib $(NVIDIA)/cufft/lib $(NVIDIA)/cublas/lib
empty :=
space := $(empty) $(empty)

CC := gcc
CFLAGS := -std=c11 -O3 -march=native -fopenmp-simd -D_FILE_OFFSET_BITS=64 -Wall -Wextra
LIBS := -l:libfftw3.so.3 $(addprefix -L,$(GPU_DIRS)) \
	-l:libcudart.so.12 -l:libcufft.so.11 -l:libcublas.so.12 \
	-Wl,-rpath,$(subst $(space),:,$(GPU_DIRS)) -pthread -lm

WAVS := /mnt/sda/SiteH/SOCAL_H_76/SOCAL_H_76_disk01_df20
# The recordings the Python detector was run on, in its order.
STAMPS := 230425_075615 230426_054845 230427_034115 230428_013345 230428_232615 230429_211845

all: song_detector

# The GPUs work in single precision.
song_detector: song_detector.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# The same detector with double precision on the GPUs: slower, and its placements equal the Python detector's.
song_detector_double: song_detector.c
	$(CC) $(CFLAGS) -DGPU_DOUBLE -o $@ $< $(LIBS)

build/song_model.bin: export_model.py ../results/theme_model.npz
	mkdir -p build
	python3 export_model.py $@

SIX := $(addprefix $(WAVS)/SOCAL_H_76_,$(addsuffix _df20.x.wav,$(STAMPS)))

# Run the same six recordings and compare with the Python detector's results.
test: song_detector build/song_model.bin
	mkdir -p build/out
	./song_detector build/song_model.bin build/out $(SIX)
	python3 compare.py ../results build/out

test-double: song_detector_double build/song_model.bin
	mkdir -p build/out_double
	./song_detector_double build/song_model.bin build/out_double $(SIX)
	python3 compare.py ../results build/out_double

clean:
	rm -rf song_detector song_detector_double build

.PHONY: all test test-double clean
