/****************************************************
 * cvc_hpe5_sigma.c
 *
 * Tue Nov 24 14:30:20 CET 2009
 *
 * PURPOSE:
 * - read output Pi_\mu\nu from different source pairs
 *   and calculate variance due to stochastic estimation
 * TODO: 
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include "ifftw.h"
#include <getopt.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"

void usage() {
  fprintf(stdout, "Code to perform light neutral contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, nu;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int dxm[4], dxn[4], ixpm, ixpn;
  int sid;
  double *disc  = (double*)NULL;
  double *work  = (double*)NULL;
  double *work2 = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  complex w, w1, w2, *cp1, *cp2, *cp3;
  FILE *ofs;

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vgf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'g':
      do_gt = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");

  /* read the input file */
  read_input(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);
#ifdef MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  disc  = (double*)calloc(32*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  work  = (double*)calloc(32*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) work[ix] = 0.;

  work2  = (double*)calloc(32*VOLUME, sizeof(double));
  if( work2 == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work2\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) work2[ix] = 0.;

  /***********************************************
   * start loop on source id.s 
   ***********************************************/
  count=0;
  for(sid=g_sourceid; sid<=g_sourceid2; sid++) {
    count++;
    /********************************
     * read the contractions
     ********************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    sprintf(filename, "cvc_hpe5_ft.%.4d.%.2d", Nconf, sid);
    read_contraction(disc, NULL, filename, 16);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    fprintf(stdout, "# time to read contractions: %e seconds\n", retime-ratime);

    /************************************************
     * add to work(2) 
     ************************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(mu=0; mu<16; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        work[_GWI(mu,ix,VOLUME)  ] += disc[_GWI(mu,ix,VOLUME)  ];
        work[_GWI(mu,ix,VOLUME)+1] += disc[_GWI(mu,ix,VOLUME)+1];

        work2[_GWI(mu,ix,VOLUME)  ] += disc[_GWI(mu,ix,VOLUME)  ]*disc[_GWI(mu,ix,VOLUME)  ];
        work2[_GWI(mu,ix,VOLUME)+1] += disc[_GWI(mu,ix,VOLUME)+1]*disc[_GWI(mu,ix,VOLUME)+1];
      }
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "time to add up: %e seconds\n", retime-ratime);

    /************************************************
     * save results 
     ************************************************/
    if(count%Nsave==0) {
      fprintf(stdout, "# save result for count = %d\n", count);
      for(mu=0; mu<16; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          disc[_GWI(0,ix,VOLUME)  ] = work2[_GWI(mu,ix,VOLUME)  ]/(double)count - 
            work[_GWI(mu,ix,VOLUME)  ]*work[_GWI(mu,ix,VOLUME)  ]/(double)(count*count);
          disc[_GWI(0,ix,VOLUME)+1] = work2[_GWI(mu,ix,VOLUME)+1]/(double)count - 
            work[_GWI(mu,ix,VOLUME)+1]*work[_GWI(mu,ix,VOLUME)+1]/(double)(count*count);
        }
        sprintf(filename, "cvc_hpe5_sigma_%.2d.%.4d.%.2d", mu, Nconf, count);
        write_contraction(disc+_GWI(0,0,VOLUME), NULL, filename, 1, 2, 0);
      }
    }

  }  /* loop on sid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(disc);
  free(work);
  free(work2);

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
