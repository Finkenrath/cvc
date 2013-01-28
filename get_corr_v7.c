/****************************************************
 * get_corr_v7.c
 *
 * Mon Jan 28 13:23:45 EET 2013
 *
 * PURPOSE
 * - originally copied from get_corr_v5.c
 * - build Pi_0k(K_0, K_k)
 *   file pattern:
 *     vacpol.NCONF
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
#include "contractions_io.h"

#ifndef CLOCK
#define CLOCK ((double)clock() / CLOCKS_PER_SEC)
#endif

void usage() {
  fprintf(stdout, "# [get_corr_v7] Code to build Pi_00(K, x=y=z=0) and Pi_jj(K, x=y=z=0)\n");
  fprintf(stdout, "# [get_corr_v7] Usage:    [options]\n");
  fprintf(stdout, "# [get_corr_v7] Options: -v verbose [default minimal verbosity]\n");
  fprintf(stdout, "# [get_corr_v7]          -f input filename [default cvc.input]\n");
  fprintf(stdout, "# [get_corr_v7]          -W check Ward Identity [default do not check]\n");
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
  double *connx = NULL;
  double *conny = NULL;
  double *connz = NULL;
  int verbose = 0;
  char filename[200];
  double ratime, retime;
  FILE *ofs;
  double q[4], wre, wim, dtmp;
  int check_WI = 0, write_ascii=0;
  unsigned int VOL3=0;
  int dims[2];
  double phase;
  complex w, w2;

  fftw_complex *in=NULL;
  fftwnd_plan plan_m;

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
      fprintf(stdout, "# [get_corr_v7] check Ward Identity\n");
      break;
    case 'A':
      write_ascii = 1;
      fprintf(stdout, "# [get_corr_v7] write Pi_mn in ASCII format\n");
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
  fprintf(stdout, "# [get_corr_v7] reading input parameters from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    fprintf(stdout, "# [get_corr_v7] T=%d, LX=%d, LY=%d, LZ=%d\n", T_global, LX, LY, LZ);
    if(g_proc_id==0) fprintf(stderr, "[get_corr_v7] Error, T and L's must be set\n");
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
    fprintf(stderr, "[get_corr_v7] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  dims[0]=T_global; dims[1]=LX;
  plan_m = fftwnd_create_plan(2, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);


  VOL3 = LX*LY*LZ;
  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(32 * VOLUME, sizeof(double));
  if( (conn==NULL) ) {
    fprintf(stderr, "[get_corr_v7] Error, could not allocate memory for contr. fields\n");
    EXIT(2);
  }

  connx= (double*)calloc(2 * T * LX, sizeof(double));
  if( (connx==NULL) ) {
    fprintf(stderr, "[get_corr_v7] Error, could not allocate memory for corr.\n");
    EXIT(3);
  }

  conny= (double*)calloc(2 * T * LY, sizeof(double));
  if( (conny==NULL) ) {
    fprintf(stderr, "[get_corr_v7] Error, could not allocate memory for corr.\n");
    EXIT(3);
  }

  connz= (double*)calloc(2 * T * LZ, sizeof(double));
  if( (connz==NULL) ) {
    fprintf(stderr, "[get_corr_v7] Error, could not allocate memory for corr.\n");
    EXIT(3);
  }

  /********************************
   * determine source coordinates *
   ********************************/

  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "# [get_corr_v7] process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  if(have_source_flag==1) { 
    fprintf(stdout, "# [get_corr_v7] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
  }
  have_source_flag = 0;


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
      // fprintf(stderr, "[get_corr_v7] Error from read_contractions, status was %d\n", status);
      // EXIT(5);
      fprintf(stderr, "[get_corr_v7] Warning, could not read contractions for gid %d, status was %d\n", gid, status);
      continue;
    }
    retime = CLOCK;
    fprintf(stdout, "# [get_corr_v7] time to read contractions %e seconds\n", retime-ratime);
  
    // TEST Ward Identity
    if(check_WI) {
      fprintf(stdout, "# [get_corr_v7] Ward identity\n");
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
    memset(connx, 0, 2*T*LX*sizeof(double));
    memset(conny, 0, 2*T*LY*sizeof(double));
    memset(connz, 0, 2*T*LZ*sizeof(double));
    // (1) 
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {

      phase = 2. * M_PI * ( (double)x0 / (double)T * ((double)sx0 - 0.5) + (double)x1 / (double)LX * ((double)sx1 + 0.5) );
      w.re = cos(phase);
      w.im = sin(phase);

      ix = g_ipt[x0][x1][0][0];  // q_y = q_z = 0
      iix = _GWI(1, ix, VOLUME);
      w2.re = conn[iix  ];
      w2.im = conn[iix+1];
      _co_eq_co_ti_co((complex*)(connx+2*(x0*LX+x1)), &w2, &w);
    }}

    for(x0=0; x0<T; x0++) {
    for(x2=0; x2<LY; x2++) {
      phase = 2. * M_PI * ( (double)x0 / (double)T * ((double)sx0 - 0.5) + (double)x2 / (double)LY * ((double)sx2 + 0.5) );
      w.re = cos(phase);
      w.im = sin(phase);

      ix = g_ipt[x0][0][x2][0];  // q_x = q_z = 0
      iix = _GWI(2, ix, VOLUME);

      w2.re = conn[iix  ];
      w2.im = conn[iix+1];
      _co_eq_co_ti_co((complex*)(conny+2*(x0*LY+x2)), &w2, &w);
    }}

    for(x0=0; x0<T; x0++) {
    for(x3=0; x3<LZ; x3++) {
      phase = 2. * M_PI * ( (double)x0 / (double)T * ((double)sx0 - 0.5) + (double)x3 / (double)LZ * ((double)sx3 + 0.5) );
      w.re = cos(phase);
      w.im = sin(phase);

      ix = g_ipt[x0][0][0][x3];  // q_x = q_y = 0
      iix = _GWI(3, ix, VOLUME);

      w2.re = conn[iix  ];
      w2.im = conn[iix+1];
      _co_eq_co_ti_co((complex*)(connz+2*(x0*LZ+x3)), &w2, &w);
    }}
    retime = CLOCK;
    fprintf(stdout, "# [get_corr_v7] time to fill correlator %e seconds\n", retime-ratime);
   
    // Fourier transform
    // 0 x
    memcpy(in, connx, 2*T*LX*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy(connx, in, 2*T*LX*sizeof(double));
    
    // 0 y
    memcpy(in, conny, 2*T*LY*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy(conny, in, 2*T*LY*sizeof(double));

    // 0 z
    memcpy(in, connz, 2*T*LZ*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy(connz, in, 2*T*LZ*sizeof(double));

    // write to file
    ratime = CLOCK;
    sprintf(filename, "pi0x_corr.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "[get_corr_v7] Error, could not open file %s for writing\n", filename);
      EXIT(6);
    }
    ix = 0;
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, x0, x1, connx[ix], connx[ix+1], gid);
      ix += 2;
    }}
    fclose(ofs);
  
    sprintf(filename, "pi0y_corr.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "[get_corr_v7] Error, could not open file %s for writing\n", filename);
      EXIT(6);
    }
    ix = 0;
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LY; x1++) {
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, x0, x1, conny[ix], conny[ix+1], gid);
      ix += 2;
    }}
    fclose(ofs);
  
    sprintf(filename, "pi0z_corr.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "[get_corr_v7] Error, could not open file %s for writing\n", filename);
      EXIT(6);
    }
    ix = 0;
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LZ; x1++) {
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, x0, x1, connz[ix], connz[ix+1], gid);
      ix += 2;
    }}
    fclose(ofs);

    retime = CLOCK;
    fprintf(stdout, "# [get_corr_v7] time to write correlator %e seconds\n", retime-ratime);
  }  // of loop on gid

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free_geometry();
  if(conn  != NULL) free(conn);
  if(connx != NULL) free(connx);
  if(conny != NULL) free(conny);
  if(connz != NULL) free(connz);
  fftwnd_destroy_plan(plan_m);

  fprintf(stdout, "# [get_corr_v7] %s# [get_corr_v7] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "[get_corr_v7] %s[get_corr_v7] end of run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);

}
