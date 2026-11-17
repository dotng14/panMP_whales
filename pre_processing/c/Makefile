CC = gcc
CFLAGS = -std=c11 -O3 -D_FILE_OFFSET_BITS=64 -Wall -Wextra
NV = $(HOME)/.local/lib/python3.13/site-packages/nvidia
LDFLAGS = -l:libfftw3f_threads.so.3 -l:libfftw3f.so.3 -pthread -lm \
	-L$(NV)/cuda_runtime/lib -L$(NV)/cufft/lib \
	-l:libcudart.so.12 -l:libcufft.so.11 \
	-Wl,-rpath,$(NV)/cuda_runtime/lib:$(NV)/cufft/lib

.PHONY: all check check-minute

all: energy_filter minute_energy

energy_filter: energy_filter.c
	$(CC) $(CFLAGS) -o $@ energy_filter.c $(LDFLAGS)

minute_energy: minute_energy.c
	$(CC) $(CFLAGS) -fopenmp -o $@ minute_energy.c $(LDFLAGS) -fopenmp

check: energy_filter
	./energy_filter --self-test

check-minute: minute_energy
	./minute_energy --self-test
