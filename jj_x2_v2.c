/****************************************************
 * jj_x2_v2.c
 *
 * Sa 21. Jul 12:31:29 CEST 2012
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
#include "make_x_orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to calculate <j_mu j_mu>(x) from momentum space data.\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  EXIT(0);
}

#ifdef MPI
#define CLOCK MPI_Wtime()
#else
#define CLOCK ((double)clock() / CLOCKS_PER_SEC)
#endif

int main(int argc, char **argv) {
  
  int c, mu, nu, status, dims[4], i, gid;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  double *conn = NULL;
  double *jjx0=NULL;
  char filename[800], contype[400];
  double ratime, retime;
  double q[4];
  FILE *ofs=NULL;
  int check_WI = 0;
  complex w;

  fftw_complex *in=NULL;

  fftwnd_plan plan_m;

  /*******************************************/
  /*                                         */
  int *h4_count=NULL, *h4_id=NULL, h4_nc, **h4_rep=NULL;
  double **h4_val = NULL;
  /*                                         */
  /*******************************************/

  while ((c = getopt(argc, argv, "wh?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_WI = 1;
      fprintf(stdout, "# [jj_x2_v2] will check WI in momentum space\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // global time
  g_the_time = time(NULL);

  // set the default values
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  // initialize fftw, create plan with FFTW_FORWARD ---  in contrast to
  //   FFTW_BACKWARD in e.g. avc_exact
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
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
    EXIT(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  // make the H4 orbits
  status = make_x_orbits_4d(&h4_id, &h4_count, &h4_val, &h4_nc, &h4_rep);
  if(status != 0) {
    fprintf(stderr, "[jj_x2_v2] Error while creating orbit-lists, status was %d\n", status);
    EXIT(2);
  }

  in = (fftw_complex*)malloc(VOLUME*sizeof(fftw_complex));

  conn = (double*)calloc(32*VOLUME, sizeof(double));
  if( conn==NULL ) {
    fprintf(stderr, "[jj_x2_v2] Error, could not allocate memory for contr. fields\n");
    EXIT(3);
  }

  jjx0 = (double*)calloc(2 * h4_nc, sizeof(double));
  if( jjx0==NULL ) {
    fprintf(stderr, "[jj_x2_v2] Error, could not allocate memory for jjx0\n");
    EXIT(4);
  }

  for(gid = g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

    /***********************
     * read contractions   *
     ***********************/
    ratime = CLOCK;
    sprintf(filename, "%s.%.4d", filename_prefix, gid);
    fprintf(stdout, "# [jj_x2_v2] Reading data from file %s\n", filename);
    status = read_contraction(conn, NULL, filename, 16);
    if(status == 106) {
      fprintf(stderr, "[jj_x2_v2] Error: could not read from file %s; status was %d\n", filename, status);
      EXIT(5);
    }
    retime = CLOCK;
    fprintf(stdout, "# [jj_x2_v2] time to read contractions: %e seconds\n", retime-ratime);
  
    if(check_WI) {
      /******************************
       * WI check in momentum space
       ******************************/
      ratime = CLOCK;
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
      retime = CLOCK;
      fprintf(stdout, "# [jj_x2_v2] time to check WI in momentum space: %e seconds\n", retime-ratime);
    }

    /******************************
     * backward Fourier transform
     ******************************/
    ratime = CLOCK;
    // construct Pi_mu_mu
    for(ix=0;ix<VOLUME;ix++) {
      conn[_GWI(0, ix,VOLUME)  ] += conn[_GWI(5,ix,VOLUME)  ] + conn[_GWI(10,ix,VOLUME)  ] + conn[_GWI(15,ix,VOLUME)  ];
      conn[_GWI(0, ix,VOLUME)+1] += conn[_GWI(5,ix,VOLUME)+1] + conn[_GWI(10,ix,VOLUME)+1] + conn[_GWI(15,ix,VOLUME)+1];
    }
    memcpy((void*)in, (void*)conn, 2*VOLUME*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy((void*)conn, (void*)in, 2*VOLUME*sizeof(double));
    retime = CLOCK;
    fprintf(stdout, "# [jj_x2_v2] time for backward Fourier transform: %e seconds\n", retime-ratime);
  
    ratime = CLOCK;
    sprintf(filename, "vacpol_x.%.4d", gid);
    sprintf(contype, "Pi_mu_mu with full x-dependence");
    status =  write_lime_contraction(conn, filename, 64, 1, contype, Nconf, 0);
    if(status != 0) {
      fprintf(stderr, "Error from write_lime_contraction, status was %d\n", status);
      EXIT(8);
    }
    retime = CLOCK;
    fprintf(stdout, "# [jj_x2_v2] time write x-dep. correlator: %e seconds\n", retime-ratime);


    /********************************************
     * construct the x^2-dependent correlator
     ********************************************/
    ratime = CLOCK;
    memset(jjx0, 0, 2*h4_nc*sizeof(double));
  
    for(ix=0; ix<VOLUME; ix++) {
      jjx0[2*h4_id[ix]  ] += conn[2*ix  ];
      jjx0[2*h4_id[ix]+1] += conn[2*ix+1];
    }
    for(i=0; i<h4_nc; i++) {
      jjx0[2*i  ] /= (double)h4_count[i] * (double)VOLUME;
      jjx0[2*i+1] /= (double)h4_count[i] * (double)VOLUME;
    }
    retime = CLOCK;
    fprintf(stdout, "# [jj_x2_v2] time to fill correlator: %e seconds\n", retime-ratime);
  
    /*****************************************
     * write to file
     *****************************************/
    ratime = CLOCK;
    sprintf(filename, "vacpol_x2.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "[jj_x2_v2] Error: could not open file %s for writing\n", filename);
      EXIT(6);
    }
    fprintf(stdout, "# [jj_x2_v2] writing jjx-data to file %s\n", filename);
  
    fprintf(ofs, "# %s", ctime(&g_the_time));
    fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
    for(ix=0; ix<h4_nc; ix++) {
      fprintf(ofs, "%4d%4d%4d%4d%16.7e%16.7e%16.7e%16.7e%25.16e%25.16e%5d\n",
        h4_rep[ix][0], h4_rep[ix][1], h4_rep[ix][2], h4_rep[ix][3],
        h4_val[ix][0], h4_val[ix][1], h4_val[ix][2], h4_val[ix][3], 
        jjx0[2*ix], jjx0[2*ix+1], h4_count[ix]);
    }
    fclose(ofs);
    retime = CLOCK;
    fprintf(stdout, "# [jj_x2_v2] time to write correlator %e seconds\n", retime-ratime);
#if 0
#endif  // of if 0
  }  // of loop on gauge id

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  finalize_x_orbits(&h4_id, &h4_count, &h4_val, &h4_rep);
  free_geometry();
  if(conn != NULL) free(conn);
  if(jjx0 != NULL) free(jjx0);
  fftwnd_destroy_plan(plan_m);
  if(in!=NULL) free(in);

  fprintf(stdout, "\n# [jj_x2_v2] %s# [jj_x2_v2] end of run\n", ctime(&g_the_time));
  fprintf(stderr, "\n[jj_x2_v2] %s[jj_x2_v2] end of run\n", ctime(&g_the_time));

  return(0);

}
