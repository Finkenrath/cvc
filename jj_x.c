/****************************************************
 * jj_x.c
 *
 * Mon May 24 21:26:20 CEST 2010
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
#include "read_input_parser.h"
#include "get_index.h"
#include "make_H3orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to calculate <j_mu j_mu>(x) from momentum space data.\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, mu, nu, status, dims[4];
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int tsrc, xsrc, ysrc, zsrc;
  int Thp1, VOL3;
  int *h4_count=NULL, *h4_id=NULL, h4_nc;
  double *conn    = NULL;
/*  double *conn2 = NULL;  */
  double **h4_val = NULL;
  double *jjx0=NULL, *jjxi=NULL, *jjt0=NULL, *jjti=NULL;
  char filename[800];
  double ratime, retime;
  double q[4], phase;
  FILE *ofs=NULL;
  complex w, w1;

  fftw_complex *in=NULL;

  fftwnd_plan plan_m;

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

  /* initialize fftw, create plan with FFTW_FORWARD ---  in contrast to
   * FFTW_BACKWARD in e.g. avc_exact */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Thp1         = T/2 + 1;
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
  VOL3 = LX*LY*LZ;

  geometry();

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  /* make the H4 orbits */
  status = make_H4_r_orbits(&h4_id, &h4_count, &h4_val, &h4_nc);
  if(status != 0) {
    fprintf(stderr, "Error while creating h4-lists\n");
    exit(4);
  }

  in = (fftw_complex*)malloc(VOLUME*sizeof(fftw_complex));

  conn = (double*)calloc(32*VOLUME, sizeof(double));
  if( (conn==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    exit(3);
  }
/*  conn2 = (double*)calloc(32*VOLUME, sizeof(double)); */

  jjx0 = (double*)calloc(2 * h4_nc, sizeof(double));
  jjxi = (double*)calloc(2 * h4_nc, sizeof(double));
  if( jjx0==NULL || jjxi==NULL ) {
    fprintf(stderr, "could not allocate memory for jjx0 or jjxi\n");
    exit(2);
  }
  jjt0 = (double*)calloc(2 * T, sizeof(double));
  jjti = (double*)calloc(2 * T, sizeof(double));
  if( jjt0==NULL || jjti==NULL) {
    fprintf(stderr, "could not allocate memory for jjt0 or jjti\n");
    exit(2);
  }
  
  tsrc = g_source_location/(LX*LY*LZ);
  xsrc = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  ysrc = (g_source_location%(LY*LZ)) / LZ;
  zsrc = (g_source_location%LZ);
  fprintf(stdout, "# source location: (%d, %d, %d, %d)\n", tsrc, xsrc, ysrc, zsrc);

  /***********************
   * read contractions   *
   ***********************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# Reading data from file %s\n", filename_prefix);
  status = read_contraction(conn, NULL, filename_prefix, 16);
  if(status == 106) {
    fprintf(stderr, "Error: could not read from file %s; status was %d\n", filename_prefix, status);
    exit(6);
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to read contractions: %e seconds\n", retime-ratime);

/*  status = read_contraction(conn2, NULL, gaugefilename_prefix, 16); */
  /******************************
   * WI check in momentum space
   ******************************/
/*
  ratime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# check WI in position space\n");
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * sin( M_PI * (double)(x0+Tstart) / (double)T_global );
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin( M_PI * (double)(x1) / (double)LX );
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * sin( M_PI * (double)(x2) / (double)LY );
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * sin( M_PI * (double)(x3) / (double)LZ );
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(stdout, "# WICheck t=%2d, x=%2d, y=%2d, z=%2d\n", x0, x1, x2, x3);
    for(nu=0; nu<4; nu++) {
      w.re = q[0] * conn[_GWI(4*0+nu, ix, VOLUME)  ] +
             q[1] * conn[_GWI(4*1+nu, ix, VOLUME)  ] +
             q[2] * conn[_GWI(4*2+nu, ix, VOLUME)  ] +
             q[3] * conn[_GWI(4*3+nu, ix, VOLUME)  ];

      w.im = q[0] * conn[_GWI(4*0+nu, ix, VOLUME)+1] +
             q[1] * conn[_GWI(4*1+nu, ix, VOLUME)+1] +
             q[2] * conn[_GWI(4*2+nu, ix, VOLUME)+1] +
             q[3] * conn[_GWI(4*3+nu, ix, VOLUME)+1];

      fprintf(stdout, "# WICheck\t %3d%25.16e%25.16e\n", nu, w.re, w.im);
    }
  }}}}
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to check WI in momentum space: %e seconds\n", retime-ratime);
*/
  /******************************
   * backward Fourier transform
   ******************************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  ix = _GWI(4*mu+nu, 0, VOLUME);  /* start index for \Pi_{\mu\nu} */
  for(x0=0; x0<T; x0++) {
    q[0] = M_PI * (double)(x0+Tstart) / (double)T_global;
  for(x1=0; x1<LX; x1++) {
    q[1] = M_PI * (double)(x1) / (double)LX;
  for(x2=0; x2<LY; x2++) {
    q[2] = M_PI * (double)(x2) / (double)LY;
  for(x3=0; x3<LZ; x3++) {
    q[3] = M_PI * (double)(x3) / (double)LZ;
/*    phase = 2. * ( q[0]*tsrc + q[1]*xsrc + q[2]*ysrc + q[3]*zsrc ); */
    phase = 0.;
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      w.re =  cos( q[mu] - q[nu] - phase );
      w.im = -sin( q[mu] - q[nu] - phase );
      _co_eq_co_ti_co(&w1, (complex*)(conn+_GWI(4*mu+nu, ix, VOLUME)), &w);
      conn[_GWI(4*mu+nu, ix, VOLUME)  ] = w1.re;
      conn[_GWI(4*mu+nu, ix, VOLUME)+1] = w1.im;
    }}
  }}}}

  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)(conn+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy((void*)(conn+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time for backward Fourier transform: %e seconds\n", retime-ratime);

  /****************************************
   * check Ward identity in position space
   ****************************************/
/*
  ratime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# check WI in position space\n");
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(stdout, "# WICheck t=%2d, x=%2d, y=%2d, z=%2d\n", x0, x1, x2, x3);
    for(nu=0; nu<4; nu++) {
      w.re =  conn[_GWI(0*4+nu, ix, VOLUME)  ]           
            + conn[_GWI(1*4+nu, ix, VOLUME)  ] 
            + conn[_GWI(2*4+nu, ix, VOLUME)  ] 
            + conn[_GWI(3*4+nu, ix, VOLUME)  ] 
            - conn[_GWI(0*4+nu, g_idn[ix][0], VOLUME)  ] 
            - conn[_GWI(1*4+nu, g_idn[ix][1], VOLUME)  ] 
            - conn[_GWI(2*4+nu, g_idn[ix][2], VOLUME)  ] 
            - conn[_GWI(3*4+nu, g_idn[ix][3], VOLUME)  ];

      w.im =  conn[_GWI(0*4+nu, ix, VOLUME)+1]           
            + conn[_GWI(1*4+nu, ix, VOLUME)+1] 
            + conn[_GWI(2*4+nu, ix, VOLUME)+1]
            + conn[_GWI(3*4+nu, ix, VOLUME)+1] 
            - conn[_GWI(0*4+nu, g_idn[ix][0], VOLUME)+1] 
            - conn[_GWI(1*4+nu, g_idn[ix][1], VOLUME)+1]
            - conn[_GWI(2*4+nu, g_idn[ix][2], VOLUME)+1] 
            - conn[_GWI(3*4+nu, g_idn[ix][3], VOLUME)+1];

      fprintf(stdout, "# WICheck\t %3d%25.16e%25.16e\n", nu, w.re, w.im);
    }
  }}}}
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time for checking Ward identity: %e seconds\n", retime-ratime);
*/
/*
  fprintf(stdout, "# write position space correlator\n");
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(stdout, "# t=%2d, x=%2d, y=%2d, z=%2d\n", x0, x1, x2, x3);
    for(mu=0; mu<16; mu++) {
      fprintf(stdout, "%3d%25.16e%25.16e\n", mu, conn[_GWI(mu, ix, VOLUME)]/(double)VOLUME, conn[_GWI(mu, ix, VOLUME)+1]/(double)VOLUME);
    }
  }}}}
*/

  /***********************
   * fill the correlator
   ***********************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  for(ix=0; ix<2*h4_nc; ix++) jjx0[ix] = 0.;
  for(ix=0; ix<2*h4_nc; ix++) jjxi[ix] = 0.;
  for(ix=0; ix<VOLUME; ix++) {
    iix = h4_id[ix];
    jjx0[2*iix  ] += conn[_GWI( 0, ix, VOLUME)  ];
    jjx0[2*iix+1] += conn[_GWI( 0, ix, VOLUME)+1];
    jjxi[2*iix  ] += conn[_GWI( 5, ix, VOLUME)  ] + conn[_GWI(10, ix, VOLUME)  ] + conn[_GWI(15, ix, VOLUME)  ];
    jjxi[2*iix+1] += conn[_GWI( 5, ix, VOLUME)+1] + conn[_GWI(10, ix, VOLUME)+1] + conn[_GWI(15, ix, VOLUME)+1];
  }
  for(ix=0; ix<h4_nc; ix++) {
    jjx0[2*ix  ] /= (double)h4_count[ix] * (double)VOLUME;
    jjx0[2*ix+1] /= (double)h4_count[ix] * (double)VOLUME;
    jjxi[2*ix  ] /= (double)h4_count[ix] * (double)VOLUME;
    jjxi[2*ix+1] /= (double)h4_count[ix] * (double)VOLUME;
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to fill correlator: %e seconds\n", retime-ratime);

  ratime = (double)clock() / CLOCKS_PER_SEC;
  for(x0=0; x0<T; x0++) {
    jjt0[2*x0  ] = 0.;
    jjt0[2*x0+1] = 0.;
    jjti[2*x0  ] = 0.;
    jjti[2*x0+1] = 0.;
    for(ix=0; ix<VOL3; ix++) {
      iix = x0*VOL3 + ix;
      jjt0[2*x0  ] += conn[_GWI( 0, iix, VOLUME)  ];
      jjt0[2*x0+1] += conn[_GWI( 0, iix, VOLUME)+1];
      jjti[2*x0  ] += conn[_GWI( 5, iix, VOLUME)  ] + conn[_GWI(10, iix, VOLUME)  ] + conn[_GWI(15, iix, VOLUME)  ];
      jjti[2*x0+1] += conn[_GWI( 5, iix, VOLUME)+1] + conn[_GWI(10, iix, VOLUME)+1] + conn[_GWI(15, iix, VOLUME)+1];
    }
  }
  sprintf(filename, "%s/jj_t.%.4d", filename_prefix2, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "Error: could not open file %s for writing\n", filename);
    exit(5);
  }
  fprintf(stdout, "# writing jjt-data to file %s\n", filename);
  fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", Nconf, T_global, LX, LY, LZ, g_kappa, g_mu);
  for(ix=0; ix<T; ix++) {
    fprintf(ofs, "%5d%25.16e%25.16e%25.16e%25.16e\n", ix, jjt0[2*ix], jjt0[2*ix+1], jjti[2*ix], jjti[2*ix+1]);
  }
  fclose(ofs);
  retime = (double)clock() / CLOCKS_PER_SEC;
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to do jj_t: %e seconds\n", retime-ratime);

  /*****************************************
   * write to file
   *****************************************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  sprintf(filename, "%s/jj_x.%.4d", filename_prefix2, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "Error: could not open file %s for writing\n", filename);
    exit(5);
  }
  fprintf(stdout, "# writing jjx-data to file %s\n", filename);
  fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", Nconf, T_global, LX, LY, LZ, g_kappa, g_mu);
  for(ix=0; ix<h4_nc; ix++) {
    fprintf(ofs, "%5d%16.7e%16.7e%16.7e%16.7e%25.16e%25.16e%25.16e%25.16e%5d\n", 
      ix, h4_val[0][ix], h4_val[1][ix], h4_val[2][ix],
      h4_val[3][ix], jjx0[2*ix], jjx0[2*ix+1], jjxi[2*ix], jjxi[2*ix+1], h4_count[ix]);
  }
  fclose(ofs);

  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to write correlator %e seconds\n", retime-ratime);


  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  finalize_H4_r_orbits(&h4_id, &h4_count, &h4_val);
  free_geometry();
  free(conn);
/*  free(conn2); */
  free(jjx0);
  free(jjxi);
  free(jjt0);
  free(jjti);
  fftwnd_destroy_plan(plan_m);
  free(in);
  return(0);

}
