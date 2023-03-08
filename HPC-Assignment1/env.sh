# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.

module load icc/2017.1.132-GCC-5.4.0-2.26
module load languages/anaconda2/5.0.1

export OMP_NUM_THREADS=28 OMP_PROC_BIND=close OMP_PLACES=cores