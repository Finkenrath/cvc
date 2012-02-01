/*********************************************************************************
 * vp_disc_ft.c
 *
 * Mon Jan 11 09:27:34 CET 2010
 *
 * PURPOSE:
 * - _ATTENTION_: the model function data is already normalized
 *   in pidisc_modelN
 * TODO:
 * - maybe also set t-range
 * - same analysis for r larger than some minimal radius
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
#include "pidisc_model.h"

void usage() {
  fprintf(stdout, "Code to perform quark-disconnected conserved vector current contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, nu;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int xx0, xx1, xx2, xx3;
  int y0min, y0max, y1min, y1max, y2min, y2max, y3min, y3max;
  int y0, y1, y2, y3, iy;
  int z0, z1, z2, z3, iz;
  int gid, status;
  int model_type = -1;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  char filename[100], contype[200];
  double ratime, retime;
  double rmin2, rmax2, rsqr;
  complex w, w1;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
#else
  fftwnd_plan plan_p, plan_m;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:t:")) != -1) {
    switch (c) {
    case 't':
      model_type = atoi(optarg);
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

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* vp_disc_ft\n");
  fprintf(stdout, "**************************************************\n\n");
#ifdef MPI
  if(g_cart_id==0) fprintf(stdout, "# Warning: MPI-version not yet available; exit\n");
  exit(200);
#endif


  /*********************************
   * initialize MPI parameters 
   *********************************/
  mpi_init(argc, argv);

  /* initialize fftw */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  plan_m = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_FORWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD,  FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
#endif
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

#ifdef MPI
  if(T==0) {
    fprintf(stderr, "[%2d] local T is zero; exit\n", g_cart_id);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(2);
  }
#endif

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
  disc  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  disc2 = (double*)calloc( 32*VOLUME, sizeof(double));
  if( disc2 == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc2\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) disc2[ix] = 0.;


  work  = (double*)calloc(32*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  /****************************************
   * prepare Fourier transformation arrays
   ****************************************/
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  /***************************************
   * set model type function
   ***************************************/
  switch (model_type) {
    case 0:
      model_type_function = pidisc_model;
      fprintf(stdout, "# function pointer set to type pidisc_model\n");
    case 1:
      model_type_function = pidisc_model1;
      fprintf(stdout, "# function pointer set to type pidisc_model1\n");
      break;
    case 2:
      model_type_function = pidisc_model2;
      fprintf(stdout, "# function pointer set to type pidisc_model2\n");
      break;
    case 3:
      model_type_function = pidisc_model3;
      fprintf(stdout, "# function pointer set to type pidisc_model3\n");
      break;
    default:
      model_type_function = NULL;
      fprintf(stdout, "# no model function selected; will add zero\n");
      break;
  }

  /****************************************
   * prepare the model for pidisc
   * - same for all gauge configurations
   ****************************************/
  rmin2 = g_rmin * g_rmin;
  rmax2 = g_rmax * g_rmax;
  if(model_type > -1) {
    for(mu=0; mu<16; mu++) {
      model_type_function(model_mrho, model_dcoeff_re, model_dcoeff_im, work, plan_m, mu);
      for(x0=-(T-1);  x0<T;  x0++) {
        y0 = (x0 + T_global) % T_global;
      for(x1=-(LX-1); x1<LX; x1++) {
        y1 = (x1 + LX) % LX;
      for(x2=-(LY-1); x2<LY; x2++) {
        y2 = (x2 + LY) % LY;
      for(x3=-(LZ-1); x3<LZ; x3++) {
        y3 = (x3 + LZ) % LZ;
        iy = g_ipt[y0][y1][y2][y3];
        rsqr = (double)(x1*x1) + (double)(x2*x2) + (double)(x3*x3);
        if(rmin2-rsqr<=_Q2EPS && rsqr-rmax2<=_Q2EPS) continue; /* radius in range for data usage, so continue */
        disc2[_GWI(mu,iy,VOLUME)  ] += work[2*iy  ];
        disc2[_GWI(mu,iy,VOLUME)+1] += work[2*iy+1];
      }}}}
      memcpy((void*)in, (void*)(disc2+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
      fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
      fftwnd_one(plan_p, in, NULL);
#endif
      memcpy((void*)(disc2+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
    }
  } else {
    for(ix=0; ix<32*VOLUME; ix++) disc2[ix] = 0.; 
  }
  
  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

    if(g_cart_id==0) fprintf(stdout, "# Start working on gauge id %d\n", gid);

    /* read the new contractions */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    sprintf(filename, "%s.%.4d.%.4d", filename_prefix, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Reading contraction data from file %s\n", filename);
    if(read_lime_contraction(disc, filename, 4, 0) == 106) {
      if(g_cart_id==0) fprintf(stderr, "Error, could not read from file %s, continue\n", filename);
      continue;
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to read contraction: %e seconds\n", retime-ratime);

    /************************************************
     * prepare \Pi_\mu\nu (x,y)
     ************************************************/
#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    for(x0=-T+1; x0<T; x0++) {
      y0min = x0<0 ? -x0 : 0;
      y0max = x0<0 ? T   : T-x0;
    for(x1=-LX+1; x1<LX; x1++) {
      y1min = x1<0 ? -x1 : 0;
      y1max = x1<0 ? LX  : LX-x1;
    for(x2=-LY+1; x2<LY; x2++) {
      y2min = x2<0 ? -x2 : 0;
      y2max = x2<0 ? LY  : LY-x2;
    for(x3=-LZ+1; x3<LZ; x3++) {
      y3min = x3<0 ? -x3 : 0;
      y3max = x3<0 ? LZ  : LZ-x3;
      xx0 = (x0+T ) % T;
      xx1 = (x1+LX) % LX;
      xx2 = (x2+LX) % LY;
      xx3 = (x3+LX) % LZ;
      ix = g_ipt[xx0][xx1][xx2][xx3];

      rsqr = (double)(x1*x1) + (double)(x2*x2) + (double)(x3*x3);
      if(rmin2-rsqr>_Q2EPS || rsqr-rmax2>_Q2EPS) continue;
      
      for(y0=y0min; y0<y0max; y0++) {
        z0 = y0 + x0;
      for(y1=y1min; y1<y1max; y1++) {
        z1 = y1 + x1;
      for(y2=y2min; y2<y2max; y2++) {
        z2 = y2 + x2;
      for(y3=y3min; y3<y3max; y3++) {
        z3 = y3 + x3;
        iy = g_ipt[y0][y1][y2][y3];
        iz = g_ipt[z0][z1][z2][z3];

        i=0;
        for(mu=0; mu<4; mu++) {
        for(nu=0; nu<4; nu++) {
          iix = _GWI(i,ix,VOLUME);
          _co_eq_co_ti_co(&w, (complex*)(disc+_GWI(mu,iz,VOLUME)), (complex*)(disc+_GWI(nu,iy,VOLUME)));
          work[iix  ] += w.re;
          work[iix+1] += w.im;
          i++;
        }}
      }}}}
    }}}}
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to calculate \\Pi_\\mu\\nu in position space: %e seconds\n", retime-ratime);

    /***********************************************
     * Fourier transform
     ***********************************************/
    for(mu=0; mu<16; mu++) {
      memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
      fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
      fftwnd_one(plan_p, in, NULL);
#endif      
      memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
    }


    fnorm = 1. / ((double)T_global * (double)(LX*LY*LZ));
    if(g_cart_id==0) fprintf(stdout, "# P-fnorm = %16.5e\n", fnorm);
    for(x0=0; x0<T; x0++) {
      q[0] = (double)(x0+Tstart) / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      q[1] = (double)x1 / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = (double)x2 / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = (double)x3 / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      i=0;
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        iix = _GWI(i,ix,VOLUME);
        w.re = cos(M_PI * (q[mu] - q[nu]));
        w.im = sin(M_PI * (q[mu] - q[nu]));
        work[iix  ] = work[iix  ] * fnorm + disc2[iix  ];
        work[iix+1] = work[iix+1] * fnorm + disc2[iix+1];
        _co_eq_co_ti_co(&w1, (complex*)(work+iix), &w);
        work[iix  ] = w1.re;
        work[iix+1] = w1.im;
        i++;
      }}
    }}}}

    /***********************************************
     * save results
     ***********************************************/
    sprintf(filename, "%s.%.4d.%.4d", filename_prefix2, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Saving results to file %s\n", filename);
    sprintf(contype, "cvc-disc-P");
    write_lime_contraction(work, filename, 64, 16, contype, gid, Nsave);

/*
    sprintf(filename, "%sascii.%.4d.%.4d", filename_prefix2, gid, Nsave);
    write_contraction(work, NULL, filename, 16, 2, 0);
*/

    if(g_cart_id==0) fprintf(stdout, "# Finished working on gauge id %d\n", gid);
  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  fftw_free(in);
  free(disc);
  free(disc2);
  free(work);

#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  fftwnd_mpi_destroy_plan(plan_m);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);
#endif

  return(0);

}
