/*********************************************************************************
 * jc_corr.c
 *
 * Tue Mar  8 10:44:50 CET 2011
 *
 * PURPOSE:
 * - calculate disconnected contribution to t-dependent conrrelator
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
#include <omp.h>
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
 
  int Thm1; 
  int c, i, mu, nthreads;
  int count        = 0;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, y0;
  int ix, iy, idx1, idx2;
  int VOL3;
  int sid1, sid2, status, gid;
  size_t nprop=0;
  double *data=NULL, *data2=NULL, *data3=NULL;
  double fnorm;
  double *mom2=NULL, *mom4=NULL;
  char filename[100];
  double ratime, retime;
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
  fprintf(stdout, "* jc_corr\n");
  fprintf(stdout, "**************************************************\n\n");

  T            = T_global;
  Thm1         = T / 2 - 1;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  VOL3         = LX*LY*LZ;
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
  nprop = (size_t)(g_sourceid2 - g_sourceid) / (size_t)g_sourceid_step + 1;
  fprintf(stdout, "\n# [jc_corr] number of stoch. propagators = %lu\n", nprop);

  data = (double*)calloc(8*FFTW_LOC_VOLUME, sizeof(double));
  if( data==NULL ) { 
    fprintf(stderr, "could not allocate memory for data\n");
    exit(3);
  }

  /* nprop * T * 3(i=1,2,3) * 2(real and imaginary part) */
  data2 = (double*)calloc(nprop*T*6, sizeof(double));
  if( data2==NULL ) { 
    fprintf(stderr, "could not allocate memory for data2\n");
    exit(3);
  }

  data3 = (double*)calloc(2*T, sizeof(double));
  if( data3==NULL ) { 
    fprintf(stderr, "could not allocate memory for data3\n");
    exit(3);
  }


  fnorm = 1. / ( (double)nprop * (double)(nprop-1) * (double)(LX*LY*LZ) );
  fprintf(stdout, "\n# [jc_corr] fnorm = %25.16e\n", fnorm);

  for(ix=0; ix<nprop*T; ix++) data2[ix] = 0.;

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    /* calculate the t-dependent current at zero spatial momentum */
    for(sid1=0; sid1<nprop; sid1++) {
      sprintf(filename, "jc_ud_x.%.4d.%.4d", gid, g_sourceid + sid1*g_sourceid_step);
      if(read_lime_contraction(data, filename, 4, 0) != 0) {
        fprintf(stderr, "\n[jc_corr] Error, could not read field no. %d\n", sid1);
        exit(15);
      }

      for(mu=0;mu<3;mu++) {
        for(x0=0;x0<T;x0++) {
          ix = g_ipt[x0][0][0][0];
          ix = _GWI(5*(mu+1), ix, VOLUME);
          for(iy=0;iy<VOL3;iy++) {
            data2[2*(sid1*3*T + mu*T + x0)  ] += data[ix + 2*iy  ];
            data2[2*(sid1*3*T + mu*T + x0)+1] += data[ix + 2*iy+1];
          }
        }
      }

    }

    /***********************************************
     * calculate the correlator
     *  - remember: x1 is the time difference of the correlator,
     *    x0 and y0 are the time coordinates of the currents
     ***********************************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(i=0;i<2*T; i++) data3[i] = 0.;
    for(sid1=0; sid1<nprop-1; sid1++) {
    for(sid2=sid1+1; sid2<nprop; sid2++) {
      for(y0=0;y0<T;y0++) {
        for(x1=0;x1<T;x1++) {
          x0 = (y0 + x1) % T;
          // first component
          idx1 = 2 * ( sid1*3*T + 0*T + y0 );
          idx2 = 2 * ( sid2*3*T + 0*T + x0 );
          // real part of the product
          data3[2*x1  ] += data2[idx1  ] * data2[idx2  ] - data2[idx1+1]*data2[idx2+1];
          // imaginary part of the product
          data3[2*x1+1] += data2[idx1+1] * data2[idx2  ] + data2[idx1  ]*data2[idx2+1];

          // second component
          idx1 = 2 * ( sid1*3*T + 1*T + y0 );
          idx2 = 2 * ( sid2*3*T + 1*T + x0 );
          // real part of the product
          data3[2*x1  ] += data2[idx1  ] * data2[idx2  ] - data2[idx1+1]*data2[idx2+1];
          // imaginary part of the product
          data3[2*x1+1] += data2[idx1+1] * data2[idx2  ] + data2[idx1  ]*data2[idx2+1];

          // third component
          idx1 = 2 * ( sid1*3*T + 2*T + y0 );
          idx2 = 2 * ( sid2*3*T + 2*T + x0 );
          // real part of the product
          data3[2*x1  ] += data2[idx1  ] * data2[idx2  ] - data2[idx1+1]*data2[idx2+1];
          // imaginary part of the product
          data3[2*x1+1] += data2[idx1+1] * data2[idx2  ] + data2[idx1  ]*data2[idx2+1];
        }
      }
    }}  // of sid2 and sid1


    // normalization
    for(x0=0;x0<2*T;x0++) { data3[x0] *= fnorm; }

    for(x0=0;x0<T/2-1;x0++) {
      mom2[x0] = 0.;
      mom4[x0] = 0.;
    }

    for(x0=1;x0<T/2;x0++) {
      if(x0==1) {
        mom2[0] = ( data3[2*x0] + data3[2*(T-x0)] ) * (double)(x0*x0);
        mom4[0] = ( data3[2*x0] + data3[2*(T-x0)] ) * (double)(x0*x0*x0*x0);
      } else {
        mom2[x0-1] = mom2[x0-2] + ( data3[2*x0] + data3[2*(T-x0)] ) * (double)(x0*x0);
        mom4[x0-1] = mom4[x0-2] + ( data3[2*x0] + data3[2*(T-x0)] ) * (double)(x0*x0*x0*x0);
      }
    }
    for(i=0;i<Thm1;i++) mom2[i] /= 6.;
    for(i=0;i<Thm1;i++) mom4[i] /= 72.;


    /************************************************
     * save results in position space
     ************************************************/
    sprintf(filename, "pi_ud_tp0.%4d.%.4d", gid, nprop);
    ofs = fopen(filename, "w");
    if (ofs==NULL) {
     fprintf(stderr, "\n[jc_corr] Error, could not open file %s for writing\n", filename);
     exit(9);
    }
    fprintf(ofs, "0 1  0%25.16e%25.16e%d\n", data3[0], 0., gid);
    for(x0=1;x0<=Thm1;x0++)
      fprintf(ofs, "0 1 %2d%25.16e%25.16e%d\n", x0, data3[x0], data3[T-x0], gid);
    fprintf(ofs, "0 1 %2d%25.16e%25.16e%d\n", x0, data3[x0], 0., gid);
    fclose(ofs);

    sprintf(filename, "pi_ud_mom.%4d.%.4d", gid, nprop);
    ofs = fopen(filename, "w");
    if (ofs==NULL) {
     fprintf(stderr, "\n[jc_corr] Error, could not open file %s for writing\n", filename);
     exit(9);
    }
    for(i=0;i<Thm1;i++)
      fprintf(ofs, "%2d%25.16e%25.16e\n", i, mom2[i], mom4[i]);
    fclose(ofs);
    
    retime = (double)clock() / CLOCKS_PER_SEC;
    if(g_cart_id == 0) fprintf(stdout, "# time for building correl.: %e seconds\n", retime-ratime);

  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(data);
  free(data2);
  free(data3);
  free(mom2);
  free(mom4);
  return(0);

}
