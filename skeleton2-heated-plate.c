/*
** A simple example of halo-exchange for a 2d grid.
** The example is heat diffusion on a heated plate
**
** Boundary conditions for the full grid are:
**
**                      W = 0
**             +--------------------+
**             |                    |
**    W = 100  |                    | W = 100
**             |                    |
**             +--------------------+
**                     W = 100
**
** i.e. 3 sides are held at 100 degress, while the fourth
** is held at 0 degrees.
**
** The grid will be partitioned into 4 subgrids, used by
** each of four ranks:
**
**                       W = 0
**                   |     |     |
**             +-----|-----|-----|-----+
**             |     |     |     |     |
**    W = 100  |     |     |     |     | W = 100
**             |     |     |     |     |
**             +-----|-----|-----|-----+
**                   |     |     |
**                      W = 100
**
** A pattern of communication using only column-based
** halos will be employed, e.g. for 4 ranks:
**
**   +-----+     +-----+     +-----+     +-----+
**   ||   ||     ||   ||     ||   ||     ||   ||
** <-|| 0 || <-> || 1 || <-> || 2 || <-> || 3 || -> 
**   ||   ||     ||   ||     ||   ||     ||   ||
**   +-----+     +-----+     +-----+     +-----+
**
** 
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

#define NROWS 16
#define NCOLS 16
#define EPSILON 0.01
#define ITERS 18
#define MASTER 0

/* function prototypes */
int calc_ncols_from_rank(int rank, int size);

int main(int argc, char* argv[])
{
  int ii,jj;             /* row and column indices for the grid */
  int kk;                /* index for looping over ranks */
  int start_row,end_row; /* rank dependent looping indices */
  int iter;              /* index for timestep iterations */ 
  int rank;              /* the rank of this process */
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */
  int remote_ncols;      /* number of columns apportioned to a remote rank */
  float boundary_mean;  /* mean of boundary values used to initialise inner cells */
  float **u;            /* local temperature grid at time t - 1 */
  float **w;            /* local temperature grid at time t     */
  float *sendbuf;       /* buffer to hold values to send */
  float *recvbuf;       /* buffer to hold received values */
  float *printbuf;      /* buffer to hold values for printing */

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

  /* 
  ** determine local grid size
  ** each rank gets all the rows, but a subset of the number of columns
  */
  local_nrows = 4;
  local_ncols = 16;

  /*
  ** allocate space for:
  ** - the local grid (2 extra columns added for the halos)
  ** - we'll use local grids for current and previous timesteps
  ** - buffers for message passing
  */
  u = (float**)malloc(sizeof(float*) * (local_nrows + 2));
  for(ii=0;ii<local_nrows;ii++) {
    u[ii] = (float*)malloc(sizeof(float) * local_ncols);
  }
  w = (float**)malloc(sizeof(float*) * (local_nrows + 2));
  for(ii=0;ii<local_nrows;ii++) {
    w[ii] = (float*)malloc(sizeof(float) * local_ncols);
  }
  sendbuf = (float*)malloc(sizeof(float) * local_ncols);
  recvbuf = (float*)malloc(sizeof(float) * local_ncols);
  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */ 
  // remote_ncols = calc_ncols_from_rank(size-1, size); this is the old code
  printbuf = (float*)malloc(sizeof(float) * (local_ncols + 2)); // honestly dont know if its okay
  
  /*
  ** initialize the local grid for the present time (w):
  ** - set boundary conditions for any boundaries that occur in the local grid
  ** - initialize inner cells to the average of all boundary cells
  ** note the looping bounds for index jj is modified 
  ** to accomodate the extra halo columns
  ** no need to initialise the halo cells at this point
  */
  boundary_mean = ((NROWS - 2) * 100.0 * 2 + (NCOLS - 2) * 100.0) / (float) ((2 * NROWS) + (2 * NCOLS) - 4);

  for(ii=0;ii<local_nrows + 2;ii++) {
    for(jj=0;jj<local_ncols;jj++) {
      if(rank == 0 && ii == 1)
	      w[ii][jj] = 0.1;
      else if(rank == size - 1 && ii == local_nrows)
	      w[ii][jj] = 100.0;
      else if(jj == 0)                  /* rank 0 gets leftmost subrid */
	      w[ii][jj] = 100.0;
      else if(jj == local_ncols - 1) /* rank (size - 1) gets rightmost subrid */
	      w[ii][jj] = 100.0;
      else if (rank == 0)
	      w[ii][jj] = boundary_mean;
    }
  }
  // printf("%f\n", boundary_mean);
  if (rank == 0) {
    for (ii=0;ii<local_nrows+2;ii++) {
      for (jj=0; jj < local_ncols; jj++) {
        printf("%6.2f ", w[ii][jj]);
      }
      printf("\n");
    }
    for (kk=1; kk < size; kk++) {
      for (ii=0; ii < local_nrows+2; ii++) {  
        MPI_Recv(printbuf,local_ncols,MPI_FLOAT,kk,tag,MPI_COMM_WORLD,&status);
        for(jj=0;jj<local_ncols;jj++) {
	          printf("%6.2f ",printbuf[jj]);
	      }
        printf("\n");
      }
    }
    
  } else {
      for (ii=0; ii < local_nrows+2; ii++) {  
        MPI_Send(w[ii],local_ncols,MPI_FLOAT,MASTER,tag,MPI_COMM_WORLD);
      }
  }
  

  if(rank == MASTER)
    printf("\n");

  /*
  ** time loop
  */
 // idea is first send up then down
  for(iter=0;iter<ITERS;iter++) {
    /*
    ** halo exchange for the local grids w:
    ** - first send to the left and receive from the right,
    ** - then send to the right and receive from the left.
    ** for each direction:
    ** - first, pack the send buffer using values from the grid
    ** - exchange using MPI_Sendrecv()
    ** - unpack values from the recieve buffer into the grid
    */

    /* send to the left, receive from right */
    
    for(ii=0;ii<local_ncols;ii++)
      sendbuf[ii] = w[1][ii];

    MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, left, tag,
		 recvbuf, local_ncols, MPI_FLOAT, right, tag,
		 MPI_COMM_WORLD, &status);
    // printf("we got to here 1\n");
    for(ii=0;ii<local_ncols;ii++)
      w[local_nrows + 1][ii] = recvbuf[ii];

    /* send to the right, receive from left */
    for(ii=0;ii<local_ncols;ii++)
      sendbuf[ii] = w[local_nrows][ii];
    MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, right, tag,
		 recvbuf, local_ncols, MPI_FLOAT, left, tag,
		 MPI_COMM_WORLD, &status);
    for(ii=0;ii<local_ncols;ii++)
      w[0][ii] = recvbuf[ii];

    /*
    ** copy the old solution into the u grid
    */ 
    for(ii=0;ii<local_nrows+2;ii++) {
      for(jj=0;jj<local_ncols;jj++) {
	u[ii][jj] = w[ii][jj];
      }
    }

  if (rank == 0) {
    for (ii=0;ii<local_nrows+2;ii++) {
      for (jj=0; jj < local_ncols; jj++) {
        printf("%6.2f ", w[ii][jj]);
      }
      printf("\n");
    }
    printf("\n");

    for (kk=1; kk < size; kk++) {
      for (ii=0; ii < local_nrows+2; ii++) {  
        MPI_Recv(printbuf,local_ncols,MPI_FLOAT,kk,tag,MPI_COMM_WORLD,&status);
        for(jj=0;jj<local_ncols;jj++) {
	          printf("%6.2f ",printbuf[jj]);
	      }
        printf("\n");
      }
      printf("\n");
    }
    
  } else {
    // for (kk=1; kk < size; kk++) {
      for (ii=0; ii < local_nrows+2; ii++) {  
        MPI_Send(w[ii],local_ncols,MPI_FLOAT,MASTER,tag,MPI_COMM_WORLD);
      }
    // } 
  }

    /*
    ** compute new values of w using u
    ** looping extents depend on rank, as we don't
    ** want to overwrite any boundary conditions
    */
    // for(ii=1;ii<local_ncols-1;ii++) {
    //   if(rank == 0) {
	  //     start_row = 2;
	  //     end_row = local_nrows;
    //   }
    //   else if(rank == size -1) {
	  //     start_row = 1;
	  //     end_row = local_nrows - 1;
    //   }
    //   else {
	  //     start_row = 1;
	  //     end_row = local_nrows;
    //   }
    //   for(jj=start_row;jj<end_row + 1;jj++) {
	  //     w[jj][ii] = (u[jj][ii - 1] + u[jj][ii + 1] + u[jj - 1][ii] + u[jj + 1][ii]) / 4.0;
    //   }
    // }
  }
  
  /*
  ** at the end, write out the solution.
  ** for each row of the grid:
  ** - rank 0 first prints out its cell values
  ** - then it receives row values sent from the other
  **   ranks in order, and prints them.
  */
  if(rank == MASTER) {
    printf("NROWS: %d\nNCOLS: %d\n",NROWS,NCOLS);
    printf("Final temperature distribution over heated plate:\n");
  }
  
  if (rank == 0) {
    for (ii=1;ii<local_nrows+1;ii++) {
      for (jj=0; jj < local_ncols; jj++) {
        printf("%6.2f ", w[ii][jj]);
      }
      printf("\n");
    }
    for (kk=1; kk < size; kk++) {
      for (ii=1; ii < local_nrows+1; ii++) {  
        MPI_Recv(printbuf,local_ncols,MPI_FLOAT,kk,tag,MPI_COMM_WORLD,&status);
        for(jj=0;jj<local_ncols;jj++) {
	          printf("%6.2f ",printbuf[jj]);
	      }
        printf("\n");
      }
    }
    
  } else {
    for (kk=1; kk < size; kk++) {
      for (ii=1; ii < local_nrows+1; ii++) {  
        MPI_Send(w[ii],local_ncols,MPI_FLOAT,MASTER,tag,MPI_COMM_WORLD);
      }
    } 
  }
  

  if(rank == MASTER)
    printf("\n");

  /* don't forget to tidy up when we're done */
  MPI_Finalize();

  /* free up allocated memory */
  for(ii=0;ii<local_nrows;ii++) {
    free(u[ii]);
    free(w[ii]);
  }
  free(u);
  free(w);
  free(sendbuf);
  free(recvbuf);
  free(printbuf);

  /* and exit the program */
  return EXIT_SUCCESS;
}

int calc_ncols_from_rank(int rank, int size)
{
  int ncols;

  ncols = NCOLS / size;       /* integer division */
  if ((NCOLS % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += NCOLS % size;  /* add remainder to last rank */
  }
  
  return ncols;
}


// its a mess, using 2018 would leave only 2 right ans rest being wrong
// 2020 leave about 12 right, rest being wrong for the fifth (index 4) row
// if change it to float, it works fine but wont terminate for some reason
// 18 float wouldn't work lol, honestly fuck that