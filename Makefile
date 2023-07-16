# Makefile

EXE=d2q9-bgk

CC=mpiicc
# CC=icc
# CC=/user/home/yf20630/compiler/nvc_install/Linux_x86_64/23.3/compilers/bin/nvc
# CC=/user/work/yf20630/compiler/Linux_x86_64/23.3/compilers/bin/nvc

CFLAGS= -std=c99 -Wall -Ofast -mtune=native -fma -xHOST -qopenmp -align 
# CFLAGS= -std=c99 -Wall -Ofast -mtune=native -fma -xHOST -fopenmp -foffload=nvptx-none -foffload=-lm  -fno-fast-math -fno-associative-math
# CFLAGS= -std=c99 -Wall -Ofast -mtune=native -fma -xHOST -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75  -nocudalib --no-cuda-version-check
# CFLAGS= -std=c++11 -O3 -Xcompiler="-fopenmp" -arch=sm_XX -x c
# CFLAGS= -mp -target=gpu -gpu=cc60,fastmath -fast
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/1024x1024.final_state.dat
REF_AV_VELS_FILE=check/1024x1024.av_vels.dat

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
