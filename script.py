#!/usr/bin/env python3

import os
import time
import subprocess


"""
Use this commond line to run the file will keep the result in the terminal
python -u run.py > log.log 2>&1 &
log.log will keep the log in the terminal
"""

#cflags = ['CFLAGS="-std=c99 -Wall -O0"', 'CFLAGS="-std=c99 -Wall -O1"', 'CFLAGS="-std=c99 -Wall -O2"', 'CFLAGS="-std=c99 -Wall -O3"', 'CFLAGS="-std=c99 -Wall -Ofast"', 'CFLAGS="-std=c99 -Wall -Ofast -mtune=native"']
cflags = ['CFLAGS="-std=c99 -Wall -Ofast -mtune=native -xHOST -fma"']
# cflags = ['CFLAGS="-std=c99 -Wall -Ofast -mtune=native"']

def readFile():
    with open("compiler.txt", "r") as f:
        lines = f.read().splitlines()
    return lines

def job_submission(lines):
    for line in lines:
        os.system('module load ' + line)
        os.system('echo ' + 'module load ' + line)
        for flag in cflags:
            if os.path.exists('d2q9-bgk'):
                os.remove('d2q9-bgk')

            print(flag + '\n')
            
            if line.split('/')[0] == "icc":
                flag = 'CC="icc" ' + flag
            os.system('make '+ flag)
            #os.system('./d2q9-bgk input_128x128.params obstacles_128x128.dat')
            if os.path.exists('d2q9-bgk.out'):
                os.remove('d2q9-bgk.out')
            os.system('sbatch -W job_submit_d2q9-bgk')
            while True:
                if os.path.exists('d2q9-bgk.out'):
                    os.system('cat d2q9-bgk.out')
                    break
        # os.system('module unload ' + line) this might have been in a separate process, in hpc, 
        # every process has a fresh environment


if __name__ == "__main__":
    compiler_lines = readFile()
    job_submission(compiler_lines)
    print("======ALL DONE======")

