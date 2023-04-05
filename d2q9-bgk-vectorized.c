/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER 0

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct
{
  float *speed0;
  float *speed1;
  float *speed2;
  float *speed3;
  float *speed4;
  float *speed5;
  float *speed6;
  float *speed7;
  float *speed8;
} t_speed_new;
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_new** cells_ptr, t_speed_new** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int size, int* local_nrows, int* local_ncols, int rank);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed_new** cells, t_speed_new** tmp_cells, int* obstacles, int tot_cells, int rank, int size, int local_nrows);
float rebound_collision(const t_param params, t_speed_new* cells, t_speed_new* tmp_cells, int* obstacles,int local_nrows);
int accelerate_flow(const t_param params, t_speed_new* cells, int* obstacles, int local_nrows);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed_new* cells, int* obstacles, float* av_vels, int local_nrows);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_new** cells_ptr, t_speed_new** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_new* cells,int local_nrows);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_new* cells, int* obstacles, int tot_cells,int local_nrows);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_new* cells, int* obstacles, float av_vel);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  // t_speed* cells     = NULL;    /* grid containing fluid densities */
  // t_speed* tmp_cells = NULL;    /* scratch space */
  t_speed_new* cells = NULL;
  t_speed_new* tmp_cells = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  int ii, jj;
  int rank;              /* the rank of this process */
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  float *sendbuf;       /* buffer to hold values to send */
  float *recvbuf;       /* buffer to hold received values */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */

  /* MPI_Init returns once it has started up processes */
  /* get size and rank */ 
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  /* 
  ** determine process ranks to the left and right of rank
  ** respecting periodic boundary conditions
  */
  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  int tot_cells = initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, size, &local_nrows, &local_ncols, rank);

  // printf("tot_cells: %d\n", tot_cells);
  // printf("test length, rows:%d, cols:%d\n", local_nrows, local_ncols);

  sendbuf = (float*)malloc(sizeof(float) * 9 * local_ncols);
  // for(jj=0;jj<local_nrows;jj++) {
  //   sendbuf[jj] = (float*)malloc(sizeof(float) * local_ncols);
  // }
  recvbuf = (float*)malloc(sizeof(float) * 9 * local_ncols);
  // for(jj=0;jj<local_nrows;jj++) {
  //   recvbuf[jj] = (float*)malloc(sizeof(float) * local_ncols);
  // }

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //feel like its gonna take ages, will first need to do flow, its like init for cells, then sendrecv each
    if (rank == size - 1) {
      accelerate_flow(params, cells, obstacles, local_nrows);
    }

    // mpi send_recv
    //for(jj = 0; jj < 9; jj++) {
    memcpy(sendbuf, cells->speed0 + params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols, cells->speed1 + params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*2, cells->speed2 + params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*3, cells->speed3 + params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*4, cells->speed4 + params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*5, cells->speed5 + params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*6, cells->speed6 + params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*7, cells->speed7 + params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*8, cells->speed8 + params.nx, local_ncols*sizeof(float));


    // for(ii=0;ii<local_ncols;ii++) {
    //   // sendbuf[0][ii] = cells->speed0[ii + (1 * params.nx)];
    // printf("check buffer 1 %f\n", cells->speed0[ii + (1 * params.nx)]);
    // }
    

    MPI_Sendrecv(sendbuf, local_ncols * 9, MPI_FLOAT, left, tag,
		 recvbuf, local_ncols * 9, MPI_FLOAT, right, tag,
		 MPI_COMM_WORLD, &status);

    // for(ii=0;ii<local_ncols;ii++) {
    //   // cells->speed0[(local_nrows+1) * params.nx + ii] = recvbuf[0][ii];
    // printf("check buffer 2 %f\n", recvbuf[ii]);

    memcpy(cells->speed0 + (local_nrows+1)*params.nx, recvbuf, local_ncols*sizeof(float));
    memcpy(cells->speed1 + (local_nrows+1)*params.nx, recvbuf+local_ncols, local_ncols*sizeof(float));
    memcpy(cells->speed2 + (local_nrows+1)*params.nx, recvbuf+local_ncols*2, local_ncols*sizeof(float));
    memcpy(cells->speed3 + (local_nrows+1)*params.nx, recvbuf+local_ncols*3, local_ncols*sizeof(float));
    memcpy(cells->speed4 + (local_nrows+1)*params.nx, recvbuf+local_ncols*4, local_ncols*sizeof(float));
    memcpy(cells->speed5 + (local_nrows+1)*params.nx, recvbuf+local_ncols*5, local_ncols*sizeof(float));
    memcpy(cells->speed6 + (local_nrows+1)*params.nx, recvbuf+local_ncols*6, local_ncols*sizeof(float));
    memcpy(cells->speed7 + (local_nrows+1)*params.nx, recvbuf+local_ncols*7, local_ncols*sizeof(float));
    memcpy(cells->speed8 + (local_nrows+1)*params.nx, recvbuf+local_ncols*8, local_ncols*sizeof(float));

    // printf("we got to here 1\n");

   
    /* send to the right, receive from left */
    // for(ii=0;ii<local_ncols;ii++)
    //   sendbuf[0][ii] = cells->speed0[ii + (local_nrows * params.nx)];

    memcpy(sendbuf, cells->speed0 + local_nrows * params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols, cells->speed1 + local_nrows * params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*2, cells->speed2 + local_nrows * params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*3, cells->speed3 + local_nrows * params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*4, cells->speed4 + local_nrows * params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*5, cells->speed5 + local_nrows * params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*6, cells->speed6 + local_nrows * params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*7, cells->speed7 + local_nrows * params.nx, local_ncols*sizeof(float));
    memcpy(sendbuf+local_ncols*8, cells->speed8 + local_nrows * params.nx, local_ncols*sizeof(float));

    MPI_Sendrecv(sendbuf, local_ncols * 9, MPI_FLOAT, right, tag,
		 recvbuf, local_ncols * 9, MPI_FLOAT, left, tag,
		 MPI_COMM_WORLD, &status);

    memcpy(cells->speed0, recvbuf, local_ncols*sizeof(float));
    memcpy(cells->speed1, recvbuf+local_ncols, local_ncols*sizeof(float));
    memcpy(cells->speed2, recvbuf+local_ncols*2, local_ncols*sizeof(float));
    memcpy(cells->speed3, recvbuf+local_ncols*3, local_ncols*sizeof(float));
    memcpy(cells->speed4, recvbuf+local_ncols*4, local_ncols*sizeof(float));
    memcpy(cells->speed5, recvbuf+local_ncols*5, local_ncols*sizeof(float));
    memcpy(cells->speed6, recvbuf+local_ncols*6, local_ncols*sizeof(float));
    memcpy(cells->speed7, recvbuf+local_ncols*7, local_ncols*sizeof(float));
    memcpy(cells->speed8, recvbuf+local_ncols*8, local_ncols*sizeof(float));
    // for(ii=0;ii<local_ncols;ii++)
    //   cells->speed0[ii] = recvbuf[0][ii];
    
    // printf("test value %f\n", cells->speed0[0]);
    av_vels[tt] = timestep(params, &cells, &tmp_cells, obstacles, tot_cells, rank, size, local_nrows); 
#ifdef DEBUG
    if (rank == MASTER) {
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      //printf("tot density: %.12E\n", total_density(params, cells));
    }
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here
  t_speed_new* cells_final;
  int* obstacles_final = malloc((params.ny) * params.nx * sizeof(int));
  cells_final = (t_speed_new*)malloc(sizeof(t_speed_new));
  if (rank == MASTER) {
    (cells_final)->speed0 = malloc(sizeof(float)*((params.ny) * params.nx));
    (cells_final)->speed1 = malloc(sizeof(float)*((params.ny) * params.nx));
    (cells_final)->speed2 = malloc(sizeof(float)*((params.ny) * params.nx));
    (cells_final)->speed3 = malloc(sizeof(float)*((params.ny) * params.nx));
    (cells_final)->speed4 = malloc(sizeof(float)*((params.ny) * params.nx));
    (cells_final)->speed5 = malloc(sizeof(float)*((params.ny) * params.nx));
    (cells_final)->speed6 = malloc(sizeof(float)*((params.ny) * params.nx));
    (cells_final)->speed7 = malloc(sizeof(float)*((params.ny) * params.nx));
    (cells_final)->speed8 = malloc(sizeof(float)*((params.ny) * params.nx));
  }
  MPI_Gather(cells->speed0 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed0, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gather(cells->speed1 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed1, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gather(cells->speed2 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed2, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gather(cells->speed3 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed3, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gather(cells->speed4 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed4, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gather(cells->speed5 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed5, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gather(cells->speed6 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed6, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gather(cells->speed7 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed7, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gather(cells->speed8 + local_ncols, local_ncols*local_nrows, MPI_FLOAT, cells_final->speed8, local_ncols*local_nrows, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  MPI_Gather(obstacles, local_ncols*local_nrows, MPI_INT, obstacles_final, local_ncols*local_nrows, MPI_INT, MASTER, MPI_COMM_WORLD);

  // test for obstacles
  // if (rank==MASTER) {
  //   for (int j = 0; j < 128; j++) {
  //     for (int i = 0; i < 128; i++) {
  //       if (obstacles_final[j*128 + i] != 0) {
  //         printf("%d %d %d\n", i, j, obstacles_final[j*128+i]);
  //       }
  //     }
  //   }
  // }

  
  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  

  /* write final values and free memory */
  if (rank == MASTER) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, av_vels[params.maxIters-1]));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  }
  if (rank == MASTER) 
    write_values(params, cells_final, obstacles_final, av_vels, local_nrows);

  // printf("we get here 1\n");
  MPI_Finalize();
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  // printf("we get here 2\n");

  // for(ii=0;ii<9;ii++) {
  //   free(sendbuf[ii]);
  //   free(recvbuf[ii]);
  // } 
  free(sendbuf);
  free(recvbuf);
  free(cells_final);
  free(obstacles_final);
  // printf("we get here 3\n");

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed_new** cells, t_speed_new** tmp_cells, int* obstacles, int tot_cells, int rank, int size, int local_nrows)
{
  // propagate(params, cells, tmp_cells);
  // rebound(params, cells, tmp_cells, obstacles);
  // t_speed_new *cellsPtr = cells;
  // t_speed_new *tmp_cellsPtr = tmp_cells;
  float rank_temp = 0;
  float result = rebound_collision(params, *cells, *tmp_cells, obstacles, local_nrows);
  t_speed_new *temp = *tmp_cells;
  *tmp_cells = *cells;
  *cells = temp;
  //printf("we get here 1\n");
  if (rank == MASTER) {
    for (int i = 1; i < size; i++) {
      MPI_Recv(&rank_temp, 1, MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      result += rank_temp;
      // printf("we get here %f\n", result);
    }

  } else {
    MPI_Send(&result, 1, MPI_FLOAT,MASTER,0,MPI_COMM_WORLD);
  }
  // MPI_Gather(&result, 1, MPI_FLOAT, rank_temp, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  // printf("%d\n", tot_cells);
  return result / tot_cells;
}

int accelerate_flow(const t_param params, t_speed_new* cells, int* obstacles, int local_nrows)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = local_nrows - 1; // should be fine
  int ii = 0;
  for (ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + (jj - 1)*params.nx] // this should be fixed now
        && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[ii + jj*params.nx] += w1;
      cells->speed5[ii + jj*params.nx] += w2;
      cells->speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[ii + jj*params.nx] -= w1;
      cells->speed6[ii + jj*params.nx] -= w2;
      cells->speed7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float rebound_collision(const t_param params, t_speed_new* restrict cells, t_speed_new* restrict tmp_cells, int* restrict obstacles, int local_nrows)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  // float speed[NSPEEDS];
  float speed0 = 0.f;
  float speed1 = 0.f;
  float speed2 = 0.f;
  float speed3 = 0.f;
  float speed4 = 0.f;
  float speed5 = 0.f;
  float speed6 = 0.f;
  float speed7 = 0.f;
  float speed8 = 0.f;
  //float* restrict speeds = speed;

  // t_speed_new* cells = *cellsPtr;
  // t_speed_new* tmp_cells = *tmp_cellsPtr;
  // t_speed_new* temp;
  __assume((params.nx)%2==0);
  __assume((params.ny)%2==0);
  __assume_aligned(cells, 64);
  __assume_aligned(tmp_cells, 64);
  __assume_aligned((*cells).speed0, 64);
  __assume_aligned((*cells).speed1, 64);
  __assume_aligned((*cells).speed2, 64);
  __assume_aligned((*cells).speed3, 64);
  __assume_aligned((*cells).speed4, 64);
  __assume_aligned((*cells).speed5, 64);
  __assume_aligned((*cells).speed6, 64);
  __assume_aligned((*cells).speed7, 64);
  __assume_aligned((*cells).speed8, 64);
  __assume_aligned((*tmp_cells).speed0, 64);
  __assume_aligned((*tmp_cells).speed1, 64);
  __assume_aligned((*tmp_cells).speed2, 64);
  __assume_aligned((*tmp_cells).speed3, 64);
  __assume_aligned((*tmp_cells).speed4, 64);
  __assume_aligned((*tmp_cells).speed5, 64);
  __assume_aligned((*tmp_cells).speed6, 64);
  __assume_aligned((*tmp_cells).speed7, 64);
  __assume_aligned((*tmp_cells).speed8, 64);
  // int index = 0;
  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  //#pragma omp parallel for collapse(2)
  //#pragma omp simd
  for (int jj = 1; jj < local_nrows+1; ++jj)
  { 
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ++ii)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = jj + 1;
      const int y_s = (jj - 1);
      const int x_e = (ii + 1 < params.nx) ? ii + 1 : 0;
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      /* if the cell contains an obstacle */
      if (obstacles[ii+(jj - 1)*params.nx])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speed1[ii+jj*params.nx] = cells->speed3[x_e + jj*params.nx];
        tmp_cells->speed2[ii+jj*params.nx] = cells->speed4[ii + y_n*params.nx];
        tmp_cells->speed3[ii+jj*params.nx] = cells->speed1[x_w + jj*params.nx];
        tmp_cells->speed4[ii+jj*params.nx] = cells->speed2[ii + y_s*params.nx];
        tmp_cells->speed5[ii+jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
        tmp_cells->speed6[ii+jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
        tmp_cells->speed7[ii+jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
        tmp_cells->speed8[ii+jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
      } /* don't consider occupied cells */
      else
      {
        /* propagate densities from neighbouring cells, following
        ** appropriate directions of travel and writing into
        ** scratch space grid */
        speed0 = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
        speed1 = cells->speed1[x_w + jj*params.nx]; /* east */
        speed2 = cells->speed2[ii + y_s*params.nx]; /* north */
        speed3 = cells->speed3[x_e + jj*params.nx]; /* west */
        speed4 = cells->speed4[ii + y_n*params.nx]; /* south */
        speed5 = cells->speed5[x_w + y_s*params.nx]; /* north-east */
        speed6 = cells->speed6[x_e + y_s*params.nx]; /* north-west */
        speed7 = cells->speed7[x_e + y_n*params.nx]; /* south-west */
        speed8 = cells->speed8[x_w + y_n*params.nx]; /* south-east */
    
        /* compute local density total */
        float local_density = 0.f;
        // for (int kk = 0; kk < NSPEEDS; kk++)
        // {
        //   local_density += speeds[kk];
        // }
        local_density += speed0; 
        local_density += speed1; 
        local_density += speed2; 
        local_density += speed3; 
        local_density += speed4; 
        local_density += speed5; 
        local_density += speed6; 
        local_density += speed7; 
        local_density += speed8; 
        /* compute x velocity component */
        float u_x = (speed1 // 2.34s
                      + speed5
                      + speed8
                      - (speed3
                         + speed6
                         + speed7))
                     / local_density;
        /* compute y velocity component */
        float u_y = (speed2
                      + speed5
                      + speed6
                      - (speed4
                         + speed7
                         + speed8))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        //float tmp2 = u_sq / (2.f * c_sq);

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* local density total */
        local_density = 0.f;
       
        /* relaxation step */
        // for (int kk = 0; kk < NSPEEDS; kk++)
        // {
        //   tmp_cells[index].speeds[kk] = speeds[kk]
        //                                           + params.omega
        //                                           * (d_equ[kk] - speeds[kk]);
        //   local_density += tmp_cells[index].speeds[kk]; 
        tmp_cells->speed0[ii+jj*params.nx] = speed0 + params.omega * (d_equ[0] - speed0);
        tmp_cells->speed1[ii+jj*params.nx] = speed1 + params.omega * (d_equ[1] - speed1);
        tmp_cells->speed2[ii+jj*params.nx] = speed2 + params.omega * (d_equ[2] - speed2);
        tmp_cells->speed3[ii+jj*params.nx] = speed3 + params.omega * (d_equ[3] - speed3);
        tmp_cells->speed4[ii+jj*params.nx] = speed4 + params.omega * (d_equ[4] - speed4);
        tmp_cells->speed5[ii+jj*params.nx] = speed5 + params.omega * (d_equ[5] - speed5);
        tmp_cells->speed6[ii+jj*params.nx] = speed6 + params.omega * (d_equ[6] - speed6);
        tmp_cells->speed7[ii+jj*params.nx] = speed7 + params.omega * (d_equ[7] - speed7);
        tmp_cells->speed8[ii+jj*params.nx] = speed8 + params.omega * (d_equ[8] - speed8);

        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y)); // 2.63s
        /* increase counter of inspected cells */
      }
      //index++;
    }
  } 
  // temp = cells;
  // *cellsPtr = tmp_cells;
  // *tmp_cellsPtr = temp;
  // printf("tot_u %f\n", tot_u);
  
  return tot_u;
}


float av_velocity(const t_param params, t_speed_new* cells, int* obstacles, int tot_cells, int local_nrows)
{
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 1; jj < local_nrows+1; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx - 1])
      {
        /* local density total */
        float local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[ii + jj*params.nx];
        local_density += cells->speed2[ii + jj*params.nx];
        local_density += cells->speed3[ii + jj*params.nx];
        local_density += cells->speed4[ii + jj*params.nx];
        local_density += cells->speed5[ii + jj*params.nx];
        local_density += cells->speed6[ii + jj*params.nx];
        local_density += cells->speed7[ii + jj*params.nx];
        local_density += cells->speed8[ii + jj*params.nx];

        /* x-component of velocity */
        float u_x = (cells->speed1[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed8[ii + jj*params.nx]
                      - (cells->speed3[ii + jj*params.nx]
                         + cells->speed6[ii + jj*params.nx]
                         + cells->speed7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed6[ii + jj*params.nx]
                      - (cells->speed4[ii + jj*params.nx]
                         + cells->speed7[ii + jj*params.nx]
                         + cells->speed8[ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y)); // 2.63s
        /* increase counter of inspected cells */
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_new** cells_ptr, t_speed_new** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int size, int* local_nrows, int* local_ncols, int rank)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed_new*)_mm_malloc(sizeof(t_speed_new), 64);
  (*cells_ptr)->speed0 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*cells_ptr)->speed1 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*cells_ptr)->speed2 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*cells_ptr)->speed3 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*cells_ptr)->speed4 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*cells_ptr)->speed5 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*cells_ptr)->speed6 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*cells_ptr)->speed7 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*cells_ptr)->speed8 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  if (cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed_new*)_mm_malloc(sizeof(t_speed_new) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed0 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*tmp_cells_ptr)->speed1 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*tmp_cells_ptr)->speed2 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*tmp_cells_ptr)->speed3 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*tmp_cells_ptr)->speed4 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*tmp_cells_ptr)->speed5 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*tmp_cells_ptr)->speed6 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*tmp_cells_ptr)->speed7 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  (*tmp_cells_ptr)->speed8 = _mm_malloc(sizeof(float)*((params->ny / size + 2) * params->nx),64);
  if (tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* 
  ** determine local grid size
  ** each rank gets all the rows, but a subset of the number of columns
  */
  *local_nrows = params->ny / size;
  *local_ncols = params->nx;

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * ((params->ny / size) * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < *local_nrows + 2; jj++)
  {
    for (int ii = 0; ii < *local_ncols; ii++)
    {
      if (jj == 0 || jj == *local_nrows + 1) {
        (*cells_ptr)->speed0[ii + jj*params->nx] = 0;
        (*cells_ptr)->speed1[ii + jj*params->nx] = 0;
        (*cells_ptr)->speed2[ii + jj*params->nx] = 0;
        (*cells_ptr)->speed3[ii + jj*params->nx] = 0;
        (*cells_ptr)->speed4[ii + jj*params->nx] = 0;
        (*cells_ptr)->speed5[ii + jj*params->nx] = 0;
        (*cells_ptr)->speed6[ii + jj*params->nx] = 0;
        (*cells_ptr)->speed7[ii + jj*params->nx] = 0;
        (*cells_ptr)->speed8[ii + jj*params->nx] = 0;
      } else {
        (*cells_ptr)->speed0[ii + jj*params->nx] = w0;
        (*cells_ptr)->speed1[ii + jj*params->nx] = w1;
        (*cells_ptr)->speed2[ii + jj*params->nx] = w1;
        (*cells_ptr)->speed3[ii + jj*params->nx] = w1;
        (*cells_ptr)->speed4[ii + jj*params->nx] = w1;
        (*cells_ptr)->speed5[ii + jj*params->nx] = w2;
        (*cells_ptr)->speed6[ii + jj*params->nx] = w2;
        (*cells_ptr)->speed7[ii + jj*params->nx] = w2;
        (*cells_ptr)->speed8[ii + jj*params->nx] = w2;
      }
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < *local_nrows; jj++)
  {
    for (int ii = 0; ii < *local_ncols; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  int count = 0;
  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    if (yy >= rank * (*local_nrows) && yy < (rank + 1) * (*local_nrows)) {
      /* assign to array */
      (*obstacles_ptr)[xx + (yy - rank * (*local_nrows))*params->nx] = blocked;
      // printf("%d %d %d\n", xx, yy, blocked);
    }
    count++;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
  // this was the last bug I had for halo exchange, basically we count all of the cells, not just the partitioned ones
  return params->nx * params->ny - count - 4; // this has sth to do with how the data is arranged in the dat file
}

int finalise(const t_param* params, t_speed_new** cells_ptr, t_speed_new** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
//  printf("we got here again 1\n");
//  _mm_free((*cells_ptr)->speed0);
//  _mm_free((*cells_ptr)->speed1);
//  _mm_free((*cells_ptr)->speed2);
//  _mm_free((*cells_ptr)->speed3);
//  _mm_free((*cells_ptr)->speed4);
//  _mm_free((*cells_ptr)->speed5);
//  _mm_free((*cells_ptr)->speed6);
//  _mm_free((*cells_ptr)->speed7);
//  _mm_free((*cells_ptr)->speed8);

  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

//  _mm_free((*tmp_cells_ptr)->speed0);
//  _mm_free((*tmp_cells_ptr)->speed1);
//  _mm_free((*tmp_cells_ptr)->speed2);
//  _mm_free((*tmp_cells_ptr)->speed3);
//  _mm_free((*tmp_cells_ptr)->speed4);
//  _mm_free((*tmp_cells_ptr)->speed5);
//  _mm_free((*tmp_cells_ptr)->speed6);
//  _mm_free((*tmp_cells_ptr)->speed7);
//  _mm_free((*tmp_cells_ptr)->speed8);
  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;
  //  printf("we got here again 2\n");


  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_new* cells, int* obstacles, float av_vel)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_vel * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed_new* cells, int local_nrows)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < local_nrows; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += cells->speed0[ii + jj*params.nx]; 
      total += cells->speed1[ii + jj*params.nx]; 
      total += cells->speed2[ii + jj*params.nx]; 
      total += cells->speed3[ii + jj*params.nx]; 
      total += cells->speed4[ii + jj*params.nx]; 
      total += cells->speed5[ii + jj*params.nx]; 
      total += cells->speed6[ii + jj*params.nx]; 
      total += cells->speed7[ii + jj*params.nx];
      total += cells->speed8[ii + jj*params.nx];
    }
  }
  return total;
}

int write_values(const t_param params, t_speed_new* cells, int* obstacles, float* av_vels, int local_nrows)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        local_density += cells->speed0[ii + jj*params.nx]; 
        local_density += cells->speed1[ii + jj*params.nx]; 
        local_density += cells->speed2[ii + jj*params.nx]; 
        local_density += cells->speed3[ii + jj*params.nx]; 
        local_density += cells->speed4[ii + jj*params.nx]; 
        local_density += cells->speed5[ii + jj*params.nx]; 
        local_density += cells->speed6[ii + jj*params.nx]; 
        local_density += cells->speed7[ii + jj*params.nx]; 
        local_density += cells->speed8[ii + jj*params.nx];

        /* compute x velocity component */
        u_x = (cells->speed1[ii + jj*params.nx]
               + cells->speed5[ii + jj*params.nx]
               + cells->speed8[ii + jj*params.nx]
               - (cells->speed3[ii + jj*params.nx]
                  + cells->speed6[ii + jj*params.nx]
                  + cells->speed7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speed2[ii + jj*params.nx]
               + cells->speed5[ii + jj*params.nx]
               + cells->speed6[ii + jj*params.nx]
               - (cells->speed4[ii + jj*params.nx]
                  + cells->speed7[ii + jj*params.nx]
                  + cells->speed8[ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

// there are some problems with obstacles I would say, check initialize and write