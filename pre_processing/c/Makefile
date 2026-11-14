CC = gcc
CFLAGS = -std=c11 -O3 -D_FILE_OFFSET_BITS=64 -Wall -Wextra
NV = $(HOME)/.local/lib/python3.13/site-packages/nvidia
LDFLAGS = -l:libfftw3f_threads.so.3 -l:libfftw3f.so.3 -pthread -lm \
	-L$(NV)/cuda_runtime/lib -L$(NV)/cufft/lib \
	-l:libcudart.so.12 -l:libcufft.so.11 \
	-Wl,-rpath,$(NV)/cuda_runtime/lib:$(NV)/cufft/lib

.PHONY: all check

all: energy_filter

energy_filter: energy_filter.c
	$(CC) $(CFLAGS) -o $@ energy_filter.c $(LDFLAGS)

check: energy_filter
	./energy_filter --self-test
