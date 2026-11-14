CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Iinclude
LDFLAGS = -lm

SRC = src/dict_hmm.c src/hmm.c
BUILD = build

.PHONY: all test clean

all: $(BUILD)/test_core $(BUILD)/hmm_run

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_core: tests/test_core.c $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/test_core.c $(SRC) $(LDFLAGS)

$(BUILD)/hmm_run: src/hmm_main.c $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/hmm_main.c $(SRC) $(LDFLAGS)

test: all
	PYTHONPATH=/home/yfreund/projects/panMP_whales python3 tests/oracle.py $(BUILD)/oracle
	$(BUILD)/test_core $(BUILD)/oracle
	PYTHONPATH=/home/yfreund/projects/panMP_whales python3 tests/export_placements.py $(BUILD)/placements.txt
	$(BUILD)/hmm_run $(BUILD)/placements.txt $(BUILD)/wav32.csv $(BUILD)/motif_segments.csv
	python3 tests/check_songs.py $(BUILD)/wav32.csv $(BUILD)/motif_segments.csv

clean:
	rm -rf $(BUILD)
