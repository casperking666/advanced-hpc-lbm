# Makefile

EXE=d2q9-bgk-optimized

CC=icc
CFLAGS= -std=c99 -Wall -Ofast -mtune=native -fma -xHOST -fopenmp -restrict -align
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/256x256.final_state.dat
REF_AV_VELS_FILE=check/256x256.av_vels.dat

all: $(EXE)

debug:
	$(CC) $(CFLAGS) -g -debug $(EXE).c $(LIBS) -o $@

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

prof:
	$(CC) $(CFLAGS) -pg -g -o $(EXE) $(LIBS) $(EXE).c

.PHONY: all check clean

clean:
	rm -f $(EXE)
