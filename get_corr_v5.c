/****************************************************
 * get_corr_v5.c
 *
 * Fr 12. Okt 13:18:41 CEST 2012
 *
 * PURPOSE
 * - originally copied from get_corr_v2.c
 * - build Pi_jj(K, \xvec=0)
 *   file pattern:
 *     vacpol.NCONF
 * - is there any additive renormalization in this case?
 * DONE:
 * TODO:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
// #include "ifftw.h"
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
#include "get_index.h"
#include "read_input_parser.h"

#ifndef CLOCK
#define CLOCK ((double)clock() / CLOCKS_PER_SEC)
#endif

void usage() {
  fprintf(stdout, "# [get_corr_v5] Code to build Pi_00(K, x=y=z=0) and Pi_jj(K, x=y=z=0)\n");
  fprintf(stdout, "# [get_corr_v5] Usage:    [options]\n");
  fprintf(stdout, "# [get_corr_v5] Options: -v verbose [default minimal verbosity]\n");
  fprintf(stdout, "# [get_corr_v5]          -f input filename [default cvc.input]\n");
  fprintf(stdout, "# [get_corr_v5]          -W check Ward Identity [default do not check]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  int c, mu, nu, status, gid;
  int filename_set = 0;
  int source_location, have_source_flag = 0;
  int x0, x1, x2, x3, ix, iix;
  int sx0, sx1, sx2, sx3;
  int tsize = 0;
  double *conn  = NULL;
  double *conn2 = NULL;
  double *conn3 = NULL;
  int verbose = 0;
  char filename[200];
  double ratime, retime;
  FILE *ofs;
  double q[4], wre, wim, dtmp;
  int check_WI = 0, write_ascii=0;
  unsigned int VOL3=0;

  while ((c = getopt(argc, argv, "AWh?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'W':
      check_WI = 1;
      fprintf(stdout, "# [get_corr_v5] check Ward Identity\n");
      break;
    case 'A':
      write_ascii = 1;
      fprintf(stdout, "# [get_corr_v5] write Pi_mn in ASCII format\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  // set the default values
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [get_corr_v5] reading input parameters from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    fprintf(stdout, "# [get_corr_v5] T=%d, LX=%d, LY=%d, LZ=%d\n", T_global, LX, LY, LZ);
    if(g_proc_id==0) fprintf(stderr, "[get_corr_v5] Error, T and L's must be set\n");
    usage();
  }

  // initialize MPI parameters
  mpi_init(argc, argv);

  T            = T_global;
  Tstart       = 0;
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n",
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "[get_corr_v5] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  VOL3 = LX*LY*LZ;
  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(32 * VOLUME, sizeof(double));
  if( (conn==NULL) ) {
    fprintf(stderr, "[get_corr_v5] Error, could not allocate memory for contr. fields\n");
    EXIT(2);
  }

  conn2= (double*)calloc(2 * T, sizeof(double));
  if( (conn2==NULL) ) {
    fprintf(stderr, "[get_corr_v5] Error, could not allocate memory for corr.\n");
    EXIT(3);
  }

  conn3= (double*)calloc(2 * T, sizeof(double));
  if( (conn3==NULL) ) {
    fprintf(stderr, "[get_corr_v5] Error, could not allocate memory for corr.\n");
    EXIT(3);
  }

  /********************************
   * determine source coordinates *
   ********************************/
/*
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "# [get_corr_v5] process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  if(have_source_flag==1) { 
    fprintf(stdout, "# [get_corr_v5] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
  }
  have_source_flag = 0;
*/

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    memset(conn, 0, 32*VOLUME*sizeof(double));
    /***********************
     * read contractions   *
     ***********************/
    ratime = CLOCK;
    sprintf(filename, "%s.%.4d", filename_prefix, gid);
    if(format==2 || format==3) {
      status = read_contraction(conn, NULL, filename, 16);
    } else if( format==0) {
      status = read_lime_contraction(conn, filename, 16, 0);
    }
    if(status != 0) {
      // fprintf(stderr, "[get_corr_v5] Error from read_contractions, status was %d\n", status);
      // EXIT(5);
      fprintf(stderr, "[get_corr_v5] Warning, could not read contractions for gid %d, status was %d\n", gid, status);
      continue;
    }
    retime = CLOCK;
    fprintf(stdout, "# [get_corr_v5] time to read contractions %e seconds\n", retime-ratime);
  
    // TEST Pi_mm
    if(write_ascii) {
      sprintf(filename, "pimm_test.%.4d", gid);
      ofs = fopen(filename, "w");
      if(ofs == NULL) exit(33);
      fprintf(ofs, "# Pi_mm\n# %s", ctime(&g_the_time));
      for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        fprintf(ofs, "# t=%3d x=%3d y=%3d z=%3d\n", x0, x1, x2, x3);
        ix = g_ipt[x0][x1][x2][x3];
        for(nu=0;nu<4;nu++) {
          wre = conn[_GWI(5*nu,ix,VOLUME)];
          wim = conn[_GWI(5*nu,ix,VOLUME)+1];
          fprintf(ofs, "%3d%16.7e%16.7e\n", nu, wre, wim);
        }
      }}}}
      fclose(ofs);
    }  // of if write_ascii

    // TEST Ward Identity
    if(check_WI) {
      fprintf(stdout, "# [get_corr_v5] Ward identity\n");
      sprintf(filename, "WI.%.4d", gid);
      ofs = fopen(filename, "w");
      if(ofs == NULL) exit(32);
      for(x0=0; x0<T; x0++) {
        q[0] = 2. * sin(M_PI * (double)x0 / (double)T);
      for(x1=0; x1<LX; x1++) {
        q[1] = 2. * sin(M_PI * (double)x1 / (double)LX);
      for(x2=0; x2<LY; x2++) {
        q[2] = 2. * sin(M_PI * (double)x2 / (double)LY);
      for(x3=0; x3<LZ; x3++) {
        q[3] = 2. * sin(M_PI * (double)x3 / (double)LZ);
        ix = g_ipt[x0][x1][x2][x3];
        for(nu=0;nu<4;nu++) {
          wre =   q[0] * conn[_GWI(4*0+nu,ix,VOLUME)] + q[1] * conn[_GWI(4*1+nu,ix,VOLUME)] \
                + q[2] * conn[_GWI(4*2+nu,ix,VOLUME)] + q[3] * conn[_GWI(4*3+nu,ix,VOLUME)];
          wim =   q[0] * conn[_GWI(4*0+nu,ix,VOLUME)+1] + q[1] * conn[_GWI(4*1+nu,ix,VOLUME)+1] \
                + q[2] * conn[_GWI(4*2+nu,ix,VOLUME)+1] + q[3] * conn[_GWI(4*3+nu,ix,VOLUME)+1];
          fprintf(ofs, "\t%3d%3d%3d%3d%3d%16.7e%16.7e\n", nu, x0, x1, x2, x3, wre, wim);
        }
      }}}}
      fclose(ofs);
    }
  
    /***********************
     * fill the correlator *
     ***********************/
    ratime = CLOCK;
    memset(conn2, 0, 2*T*sizeof(double));
    // (1) V0V0
    for(x0=0; x0<T; x0++) {
      for(ix=0; ix<VOL3; ix++) {
        iix = _GWI(0,x0*VOL3+ix,VOLUME);
        conn2[2*x0  ] += conn[iix  ];
        conn2[2*x0+1] += conn[iix+1];
      }
    }
    // (2) VKVK
    memset(conn3, 0, 2*T*sizeof(double));
    for(x0=0; x0<T; x0++) {
      for(ix=0; ix<VOL3; ix++) {
        iix = x0 * VOL3 + ix;
        conn3[2*x0  ] += conn[_GWI(5,iix,VOLUME)  ] + conn[_GWI(10,iix,VOLUME)  ] + conn[_GWI(15,iix,VOLUME)  ];
        conn3[2*x0+1] += conn[_GWI(5,iix,VOLUME)+1] + conn[_GWI(10,iix,VOLUME)+1] + conn[_GWI(15,iix,VOLUME)+1];
      }
    }

    // normalization
    dtmp = 1. / (double)VOL3;
    for(x0=0; x0<2*T; x0++) { conn2[x0] *= dtmp; }
    for(x0=0; x0<2*T; x0++) { conn3[x0] *= dtmp; }
    
    retime = CLOCK;
    fprintf(stdout, "# [get_corr_v5] time to fill correlator %e seconds\n", retime-ratime);
   
    // TEST
/*
    fprintf(stdout, "# [get_corr_v5] V0V0 correlator\n");
    for(x0=0; x0<T; x0++) {
      fprintf(stdout, "\t%3d%25.16e%25.16e\n",x0, conn2[2*x0], conn2[2*x0+1]);
    }
    fprintf(stdout, "# [get_corr_v5] VKVK correlator\n");
    for(x0=0; x0<T; x0++) {
      fprintf(stdout, "\t%3d%25.16e%25.16e\n",x0, conn3[2*x0], conn3[2*x0+1]);
    }
*/  
    /*****************************************
     * write to file
     *****************************************/
    ratime = CLOCK;
    sprintf(filename, "p00_corr.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "[get_corr_v5] Error, could not open file %s for writing\n", filename);
      EXIT(6);
    }
    x0 = 0;
    fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], 0., gid);
    for(x0=1; x0<T/2; x0++) {
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], conn2[2*(T-x0)], gid);
    }
    x0 = T / 2;
    fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], 0., gid);
    fclose(ofs);
  
    sprintf(filename, "pkk_corr.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "[get_corr_v5] Error, could not open file %s for writing\n", filename);
      EXIT(7);
    }
    x0 = 0;
    fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn3[2*x0], 0., gid);
    for(x0=1; x0<T/2; x0++) {
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn3[2*x0], conn3[2*(T-x0)], gid);
    }
    x0 = T / 2;
    fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn3[2*x0], 0., gid);
    fclose(ofs);

    retime = CLOCK;
    fprintf(stdout, "# [get_corr_v5] time to write correlator %e seconds\n", retime-ratime);
  }  // of loop on gid

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free_geometry();
  if(conn  != NULL) free(conn);
  if(conn2 != NULL) free(conn2);
  if(conn3 != NULL) free(conn3);

  fprintf(stdout, "# [get_corr_v5] %s# [get_corr_v5] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "[get_corr_v5] %s[get_corr_v5] end of run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);

}
