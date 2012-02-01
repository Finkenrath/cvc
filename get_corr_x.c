/****************************************************
  
 * get_corr_x.c
 *
 * Wed Aug 11 22:06:14 CEST 2010
 *
 * PURPOSE
 * DONE:
 * TODO:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "get_index.h"
#include "make_H3orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to recover rho-rho correl.\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, mu, status, nu;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, iiy, gid;
  int tsrc, xsrc, ysrc, zsrc;
  int Thp1;
  double *conn = (double*)NULL;
  double tmp[2];
  int verbose = 0;
  char filename[800];
  double ratime, retime;
  FILE *ofs;
  double *corrt=NULL;

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

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  T            = T_global;
  Thp1         = T/2 + 1;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] parameters:\n"\
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
  conn = (double*)calloc(32*VOLUME, sizeof(double));
  if( (conn==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    exit(3);
  }

  corrt = (double*)calloc(2*T, sizeof(double));

  tsrc = g_source_location/(LX*LY*LZ);
  xsrc = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  ysrc = (g_source_location%(LY*LZ)) / LZ;
  zsrc = (g_source_location%LZ);
  fprintf(stdout, "# source location: (%d, %d, %d, %d)\n", tsrc, xsrc, ysrc, zsrc);

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    /***********************
     * read contractions   *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    sprintf(filename, "%s", filename_prefix);
    fprintf(stdout, "# Reading data from file %s\n", filename);
    status = read_lime_contraction(conn, filename, 16, 0);
    if(status == 106) {
      fprintf(stderr, "Error: could not read from file %s; status was %d\n", filename, status);
      continue;
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time to read contractions %e seconds\n", retime-ratime);

    /**********************************************
     * check the Ward identity
     **********************************************/
/*
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[x0][x1][x2][x3];
      fprintf(stdout, "# t=%d, x=%d, y=%d, z=%d\n", x0, x1, x2, x3);
      for(nu=0; nu<4; nu++) {
        tmp[0] = 0.;
        tmp[1] = 0.;
        for(mu=0; mu<4; mu++) {
          iix = g_idn[g_ipt[x0][x1][x2][x3]][mu];
          tmp[0] += conn[_GWI(4*mu+nu,ix,VOLUME)  ] - conn[_GWI(4*mu+nu,iix,VOLUME)  ];
          tmp[1] += conn[_GWI(4*mu+nu,ix,VOLUME)+1] - conn[_GWI(4*mu+nu,iix,VOLUME)+1];
        }
        fprintf(stdout, "%3d%25.16e%25.16e\n", nu, tmp[0], tmp[1]);
      }
    }}}}
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time to check Ward identity %e seconds\n", retime-ratime);
*/

    /***********************
     * fill the correlator *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(x0=0; x0<2*T; x0++) corrt[x0] = 0.;
    for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix = g_ipt[x0][x1][x2][x3];
        for(mu=1; mu<4; mu++) {
          iix = _GWI(5*mu,ix,VOLUME);
          corrt[2*x0  ] += conn[iix  ];
          corrt[2*x0+1] += conn[iix+1];
        }
      }}}
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time to fill correlator %e seconds\n", retime-ratime);
 
    free(conn);

    /********************************
     * print correl to file *
     ********************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    sprintf(filename, "rho_x.%.4d", gid);
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "Error, could not open file %s for writing\n", filename);
      exit(1);
    }
    fprintf(ofs, "# %6d\t%3d%3d%3d%3d%f\t%f\n", gid, T, LX, LY, LZ, g_kappa, g_mu);
    ix = ( 0 + tsrc ) % T;
    fprintf(ofs, "%3d%25.16e%25.16e\n", 0, corrt[2*ix], 0.);
    for(x0=1; x0<T/2; x0++) {
      ix = ( x0 + tsrc ) % T;
      iix = ( T - x0 + tsrc ) %T;
      fprintf(ofs, "%3d%25.16e%25.16e\n", x0, corrt[2*ix], corrt[2*iix]);
    }
    ix = ( T/2 + tsrc ) % T;
    fprintf(ofs, "%3d%25.16e%25.16e\n", T/2, corrt[2*ix], 0.);
    fclose(ofs);
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time to write correlator %e seconds\n", retime-ratime);
  }

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free(corrt);
  free_geometry();
  return(0);
}
