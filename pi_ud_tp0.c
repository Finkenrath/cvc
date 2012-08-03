/*********************************************************************************
 * pi_ud_tp0.c 
 *
 * Fr 3. Aug 18:24:54 CEST 2012
 *
 * PURPOSE:
 * - use the out put fields of jc_ud_x to construct the disconn. correlator
 *   D(t,pvec=0)
 * - sum over samples and subtract diagonal part
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
#include "contractions_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"

#ifndef CLOCK
#  ifdef MPI
#    define CLOCK MPI_Wtime()
#  else
#    define CLOCK ((double)clock() / CLOCKS_PER_SEC)
#  endif
#endif


void usage() {
  fprintf(stdout, "Code to perform quark-disconnected conserved vector current contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, nu;
  int count        = 0;
  int filename_set = 0;
  //int use_real_part = 1;
  int ix, iix;
  int sid, status, gid, it, ir, it2;
  double *disc = (double*)NULL;
  double *work = (double*)NULL;
  double *bias = (double*)NULL;
  //double fnorm;
  int verbose = 0;
  unsigned int VOL3;
  char filename[100];
  double ratime, retime;
  double *tmp = NULL;
  complex w;
  FILE *ofs = NULL;

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
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

  g_the_time = time(NULL);
  fprintf(stdout, "# [pi_ud_tp0] using global time stamp %s", ctime(&g_the_time));


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  fprintf(stdout, "# [pi_ud_tp0] **************************************************\n");
  fprintf(stdout, "# [pi_ud_tp0] pi_ud_p\n");
  fprintf(stdout, "# [pi_ud_tp0] **************************************************\n\n");

  /*********************************
   * initialize MPI parameters 
   *********************************/
  mpi_init(argc, argv);

#ifdef MPI
  if(T==0) {
    fprintf(stderr, "[%2d] local T is zero; exit\n", g_cart_id);
    EXIT(2);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  disc  = (double*)calloc(16*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
    EXIT(3);
  }

  work  = (double*)calloc(2*T_global, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "[pi_ud_tp0] could not allocate memory for work\n");
    EXIT(5);
  }
  bias  = (double*)calloc(2*T_global, sizeof(double));
  if( bias == (double*)NULL ) { 
    fprintf(stderr, "[pi_ud_tp0] could not allocate memory for bias\n");
    EXIT(6);
  }
  tmp = (double*)calloc(2*T_global, sizeof(double));
  if( tmp == (double*)NULL ) { 
    fprintf(stderr, "[pi_ud_tp0] could not allocate memory for tmp\n");
    EXIT(8);
  }

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    memset(work, 0, 2*T_global*sizeof(double));
    memset(bias, 0, 2*T_global*sizeof(double));
 
    count = 0;
    /***********************************************
     * start loop on source id.s 
     ***********************************************/
    for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
      memset(disc, 0, 16*VOLUME*sizeof(double));

      ratime = CLOCK;
      sprintf(filename, "jc_ud_x.%.4d.%.4d", gid, sid);
      status = read_lime_contraction(disc, filename, 4, 0);

      if(status!=0) {
        fprintf(stderr, "Error, could not read contraction data from file %s\n", filename);
        EXIT(7);
      }

      retime = CLOCK;
      if(g_cart_id==0) fprintf(stdout, "# time to read contractions: %e seconds\n", retime-ratime);

      count++;

      ratime = CLOCK;

      // add current to sum
      for(it=0; it<T; it++) {
        tmp[2*it  ] = 0.;
        tmp[2*it+1] = 0.;
        for(iix=0; iix<VOL3; iix++) {
          ix = it * VOL3 + iix;
          tmp[2*it  ] += disc[_GWI(1,ix,VOLUME)  ] + disc[_GWI(2,ix,VOLUME)  ] + disc[_GWI(3,ix,VOLUME)  ];
          tmp[2*it+1] += disc[_GWI(1,ix,VOLUME)+1] + disc[_GWI(2,ix,VOLUME)+1] + disc[_GWI(3,ix,VOLUME)+1];
        }
      }

      for(it=0; it<2*T_global; it++) { work[it] += tmp[it]; }

      // add to bias
      for(it=0; it<T_global; it++) {
      for(ir=0; ir<T_global; ir++) {
        it2 = (it + ir ) % T_global;

        _co_eq_co_ti_co( &w, (complex*)&(tmp[2*it2]), (complex*)&(tmp[2*it]) );
        bias[2*it  ] += w.re;
        bias[2*it+1] += w.im;
      }}
      retime = CLOCK;
      if(g_cart_id==0) fprintf(stdout, "# [pi_ud_tp0] time to calculate contractions: %e seconds\n", retime-ratime);

      if(count==Nsave) {
        memset(disc, 0, 2*T_global*sizeof(double));
 
        for(it=0; it<T_global; it++) {
        for(ir=0; ir<T_global; ir++) {
          it2 = (it + ir ) % T_global;

          _co_eq_co_ti_co( &w, (complex*)&(work[2*it2]), (complex*)&(work[2*it]) );
          disc[2*it  ] += w.re;
          disc[2*it+1] += w.im;
        }}

        for(it=0; it<2*T_global; it++) {
          disc[it] -= bias[it];
        }
        

        sprintf(filename, "pi_ud_t.%.4d.%.4d", gid, count);
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[pi_ud_tp0] Error, could not open file %s for writing\n", filename);
          EXIT(8);
        }
        fprintf(ofs, "# [pi_ud_tp0] results for disc. t-dependent correlator at zero spatial momentum\n# %s", ctime(&g_the_time));
        fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 11, 1, 0, disc[0], 0., Nconf);
        for(it=1; it<T_global/2; it++) {
          fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 11, 1, it, disc[it], disc[2*(T_global-it)], Nconf);
        }
        fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 11, 1, T_global/2, disc[T_global/2], 0., Nconf);
        fclose(ofs);
        retime = CLOCK;
        if(g_cart_id==0) fprintf(stdout, "# [pi_ud_tp0] time to save cvc results: %e seconds\n", retime-ratime);
      }  // of count % Nsave == 0
    }    // of loop on sid
  }      // of loop on gid

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  if(disc != NULL) free(disc);
  if(work != NULL) free(work);
  if(bias != NULL) free(bias);
  if(tmp != NULL) free(tmp);

  if(g_cart_id == 0) {
    fprintf(stdout, "# [pi_ud_tp0] %s# [pi_ud_tp0] end of run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [pi_ud_tp0] %s# [pi_ud_tp0] end of run\n", ctime(&g_the_time));
    fflush(stderr);
  }
#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}
