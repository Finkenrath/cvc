/****************************************************
 * get_rho_corr.c
 *
 * Wed Sep 23 17:58:30 CEST 2009
 *
 * PURPOSE
 * - recover the time dep. rho-rho correlator from
 *   Dru's/Xu's vacuum pol. tensor files
 *   file pattern:
 *     vacpol_con_cc_q_3320_x19y04z00t22.dat
 * DONE:
 * TODO:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
#include "get_index.h"
#include "read_input_parser.h"

void usage() {
  fprintf(stdout, "Code to recover rho-rho correl.\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, mu;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int source_location, have_source_flag = 0;
  int x0, ix;
  int sx0, sx1, sx2, sx3;
  int check_WI=0;
  double *conn  = (double*)NULL;
  double *conn2 = (double*)NULL;
  int verbose = 0;
  char filename[800];
  double ratime, retime;
  FILE *ofs;
/**************************
 * variables for WI check */
  int x1, x2, x3, nu;
  double wre, wim, q[4];
/**************************/

  fftw_complex *in=(fftw_complex*)NULL, *out=(fftw_complex*)NULL;

  fftw_plan plan_m;

  while ((c = getopt(argc, argv, "wh?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_WI = 1;
      fprintf(stdout, "# [get_rho_corr] check WI in momentum space\n");
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

  // set the default values
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [get_rho_corr] reading input parameters from file %s\n", filename);
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

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize fftw, create plan with FFTW_FORWARD ---  in contrast to
   * FFTW_BACKWARD in e.g. avc_exact */
  plan_m = fftw_create_plan(T_global, FFTW_FORWARD, FFTW_MEASURE);
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
    exit(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  if( (conn==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] = 0.;

  conn2= (double*)calloc(2 * T, sizeof(double));
  if( (conn2==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for corr.\n");
    exit(2);
  }
  for(ix=0; ix<2*T; ix++) conn2[ix] = 0.;

  /*****************************************
   * prepare Fourier transformation arrays * 
   *****************************************/
  in   = (fftw_complex*)malloc(T*sizeof(fftw_complex));
  out  = (fftw_complex*)malloc(T*sizeof(fftw_complex));
  if( (in==(fftw_complex*)NULL) || (out==(fftw_complex*)NULL) ) exit(4);

  /********************************
   * determine source coordinates *
   ********************************/
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  if(have_source_flag==1) { 
    fprintf(stdout, "local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
  }
  have_source_flag = 0;

  /***********************
   * read contractions   *
   ***********************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  // read_contraction(conn, (int*)NULL, filename_prefix, 16);
  read_lime_contraction(conn, filename_prefix, 16, 0);

  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "time to read contractions %e seconds\n", retime-ratime);

  // TEST Ward Identity
  if(check_WI) {
    fprintf(stdout, "# [get_corr_v5] Ward identity\n");
    sprintf(filename, "WI.%.4d", Nconf);
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
  ratime = (double)clock() / CLOCKS_PER_SEC;
  for(x0=0; x0<T; x0++) {
    for(mu=1; mu<4; mu++) {
      ix = get_indexf(x0,0,0,0,mu,mu);
      fprintf(stdout, "x0=%3d, mu=%3d\tix=%8d\n", x0, mu, ix);
      conn2[2*x0  ] += conn[ix  ];
      conn2[2*x0+1] += conn[ix+1];
    }
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "time to fill correlator %e seconds\n", retime-ratime);
 
  /********************************
   * test: print correl to stdout *
   ********************************/
  for(x0=0; x0<T; x0++) {
    fprintf(stdout, "%3d%25.16e%25.16e\n", x0, conn2[2*x0], conn[2*x0+1]);
  }

  /*****************************************
   * do the reverse Fourier transformation *
   *****************************************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  memcpy((void*)in, (void*)conn2, 2*T*sizeof(double));
  fftw_one(plan_m, in, out);
  for(ix=0; ix<T; ix++) {
    conn2[2*ix  ] = out[ix].re / (double)T;
    conn2[2*ix+1] = out[ix].im / (double)T;
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "time for Fourier transform %e seconds\n", retime-ratime);

  
  ratime = (double)clock() / CLOCKS_PER_SEC;
  sprintf(filename, "rho_corr.%.4d", Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "could not open file %s for writing\n", filename);
    exit(5);
  }
  //for(x0=0; x0<T; x0++) {
  //  fprintf(ofs, "%3d%25.16e%25.16e\n", x0, conn2[2*x0], conn2[2*x0+1]);
  //}

  x0 = 0;
  fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], 0., Nconf);
  for(x0=1; x0<T/2; x0++) {
    fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], conn2[2*(T-x0)], Nconf);
  }
  x0 = T/2;
  fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], 0., Nconf);

  fclose(ofs);
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "time to write correlator %e seconds\n", retime-ratime);
  

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free_geometry();
  fftw_free(in);
  fftw_free(out);
  free(conn);
  free(conn2);
  fftw_destroy_plan(plan_m);

  return(0);

}
