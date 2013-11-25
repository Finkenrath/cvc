/*********************************************************************************
 * WI_ud_xspace.c
 *
 * Wed Nov 10 14:12:32 CET 2010
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 *********************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
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
#include "contractions_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"

void usage() {
  fprintf(stdout, "Code to perform quark-disconnected conserved vector current contractions\n");
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
 
  int c, i; 
  int count        = 0;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int ixm0, ixm1, ixm2, ixm3;
  int sid1, status, gid, ix;
  double *data=NULL, *work=NULL, *ptr0, *ptr1, *ptr2, *ptr3;
  char filename[100];
  double ratime, retime;
  complex w, w2;
  FILE *ofs=NULL;

  /****************************************
   * initialize the distance vectors
   ****************************************/

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
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
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* jc_ud_p\n");
  fprintf(stdout, "**************************************************\n\n");

  /* initialize */
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "#       T            = %3d\n"\
		  "#       Tstart       = %3d\n"\
		  "#       l_LX_at      = %3d\n"\
		  "#       l_LXstart_at = %3d\n"\
		  "#       FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, T, Tstart, l_LX_at, l_LXstart_at, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  data = (double*)calloc(8*FFTW_LOC_VOLUME, sizeof(double));
  if( data==NULL ) { 
    fprintf(stderr, "could not allocate memory for data\n");
    exit(3);
  }

  work = (double*)calloc(2*FFTW_LOC_VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
    exit(7);
  }

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(sid1=g_sourceid; sid1<=g_sourceid2; sid1+=g_sourceid_step) {
      sprintf(filename, "jc_ud_x.%.4d.%.4d", gid, sid1);
      if(read_lime_contraction(data, filename, 4, 0) != 0) {
        fprintf(stderr, "Error, could not read field no. %d\n", sid1);
        exit(15);
      }

      for(ix=0; ix<2*FFTW_LOC_VOLUME; ix++) work[ix] = 0.;
      ptr0 = data;
      ptr1 = data + 2*VOLUME;
      ptr2 = data + 4*VOLUME;
      ptr3 = data + 6*VOLUME;

      for(ix=0; ix<VOLUME; ix++) {
        ixm0 = g_idn[ix][0];
        ixm1 = g_idn[ix][1];
        ixm2 = g_idn[ix][2];
        ixm3 = g_idn[ix][3];

        work[2*ix  ] = ptr0[2*ix  ] - ptr0[2*ixm0  ] 
                     + ptr1[2*ix  ] - ptr1[2*ixm1  ]
                     + ptr2[2*ix  ] - ptr2[2*ixm2  ]
                     + ptr3[2*ix  ] - ptr3[2*ixm3  ];

        work[2*ix+1] = ptr0[2*ix+1] - ptr0[2*ixm0+1] 
                     + ptr1[2*ix+1] - ptr1[2*ixm1+1]
                     + ptr2[2*ix+1] - ptr2[2*ixm2+1]
                     + ptr3[2*ix+1] - ptr3[2*ixm3+1];

      }

     
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    if(g_cart_id == 0) fprintf(stdout, "# time for calculating WI: %e seconds\n", retime-ratime);

  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(work);
  free(data);
  return(0);

}
