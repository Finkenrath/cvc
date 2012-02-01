/*********************************************************************************
 * vp_disc_ft_2.c
 *
 * Mon Jan 11 09:27:34 CET 2010
 *
 * PURPOSE:
 * - perform Fourier transformation for limited radius R of data,
 *   use model function for |x-y| > R
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
  int x0, x1, x2, x3;
  unsigned long int ix, iix, iiy;
  int gid, status;
  int model_type = -1;
  fftw_complex *disc  = NULL;
  fftw_complex *disc2 = NULL;
  fftw_complex *work  = NULL;
  double q[4], fnorm;
  char filename[100], contype[200];
  double ratime, retime;
  double rmin2, rmax2, rsqr;
  complex w, w1;
  FILE *ofs=NULL;

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
  fprintf(stdout, "* vp_disc_ft_2\n");
  fprintf(stdout, "**************************************************\n\n");


  /*********************************
   * initialize MPI parameters 
   *********************************/
  mpi_init(argc, argv);

  /* initialize fftw */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  plan_m = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_FORWARD,  FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
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

  disc  = (fftw_complex*)fftw_malloc(16*VOLUME*sizeof(fftw_complex));
  if( disc == (fftw_complex*)NULL ) {
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  disc2 = (fftw_complex*)fftw_malloc(16*VOLUME*sizeof(fftw_complex));
  if( disc2 == (fftw_complex*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc2\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  work = (fftw_complex*)fftw_malloc(16*VOLUME*sizeof(fftw_complex));
  if( work == (fftw_complex*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }


#ifndef MPI
  plan_p = fftwnd_create_plan_specific(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE, work, 16, work, 16);
  plan_m = fftwnd_create_plan_specific(4, dims, FFTW_FORWARD,  FFTW_MEASURE | FFTW_IN_PLACE, work, 16, work, 16);
#endif

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

    if(g_cart_id==0) fprintf(stdout, "# Start working on gauge id %d\n", gid);

#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    sprintf(filename, "%s_P.%.4d.%.4d", filename_prefix, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Reading contraction data from file %s\n", filename);
    if(read_lime_contraction((double*)disc, filename, 16, 0) == 106) {
      if(g_cart_id==0) fprintf(stderr, "Error, could not read from file %s, continue\n", filename);
      continue;
    }
    sprintf(filename, "%s_P.%.4d.%.4d", filename_prefix2, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Reading contraction data from file %s\n", filename);
    if(read_lime_contraction((double*)disc2, filename, 16, 0) == 106) {
      if(g_cart_id==0) fprintf(stderr, "Error, could not read from file %s, continue\n", filename);
      continue;
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to read contraction: %e seconds\n", retime-ratime);

    /****************************************************
     * prepare \Pi_\mu\nu (q) for Fourier transformation
     * - subtract bias from biased operator
     * - multiply with inverse phase factor
     ****************************************************/

#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    for(mu=0; mu<16; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        iix = mu * VOLUME + ix;
        iiy = 16 * ix + mu;
        work[iiy].re = (disc[iix].re - disc2[iix].re) * (double)Nsave / (double)(Nsave-1);
        work[iiy].im = (disc[iix].im - disc2[iix].im) * (double)Nsave / (double)(Nsave-1);
      }
    }
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
        iix = 16 * ix + i;
        w.re =  cos(M_PI * (q[mu] - q[nu]));
        w.im = -sin(M_PI * (q[mu] - q[nu]));
        _co_eq_co_ti_co(&w1, work+iix, &w);
        work[iix].re = w1.re;
        work[iix].im = w1.im;
        i++;
      }}
    }}}}

    /************************************************
     * inverse Fourier transformation
     ************************************************/

#ifdef MPI
    fftwnd_mpi(plan_m, 16, work, disc, FFTW_NORMAL_ORDER);
#else
    fftwnd(plan_m, 16, work, 16, 1, work, 16, 1);
#endif
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for inverse Fourier transform: %e seconds\n", retime-ratime);


    /************************************************
     * prepare \Pi_\mu\nu (z)
     * - separate according to the radii g_rmin and
     *   g_rmax
     ************************************************/

#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    memcpy((void*)disc,  (void*)work, 16*VOLUME*sizeof(fftw_complex));
    memcpy((void*)disc2, (void*)work, 16*VOLUME*sizeof(fftw_complex));
    rmin2 = g_rmin * g_rmin;
    rmax2 = g_rmax * g_rmax;
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = 16 * g_ipt[x0][x1][x2][x3];

      rsqr = (double)(x1*x1) + (double)(x2*x2) + (double)(x3*x3);
      if(rsqr>rmin2-_Q2EPS && rsqr<rmax2+_Q2EPS) {
        for(mu=0; mu<16; mu++) {
          disc2[ix+mu].re = 0.;
          disc2[ix+mu].im = 0.;
        }
      } else {
        for(mu=0; mu<16; mu++) {
          disc[ix+mu].re = 0.;
          disc[ix+mu].im = 0.;
        }
      }
    }}}}
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to separate radii: %e seconds\n", retime-ratime);


    /***********************************************
     * Fourier transform
     ***********************************************/

#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    memcpy((void*)work, (void*)disc2, 16*VOLUME*sizeof(fftw_complex));
#ifdef MPI
    fftwnd_mpi(plan_p, 16, work, disc2, FFTW_NORMAL_ORDER);
#else
    fftwnd(plan_p, 16, work, 16, 1, work, 16, 1);
#endif      

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
        iix = 16 * ix + i;
        iiy = i*VOLUME + ix;
        w.re = cos(M_PI * (q[mu] - q[nu]));
        w.im = sin(M_PI * (q[mu] - q[nu]));
        _co_eq_co_ti_co(&w1, work+iix, &w);
        disc2[iiy].re = fnorm * w1.re;
        disc2[iiy].im = fnorm * w1.im;
        i++;
      }}
    }}}}
    sprintf(filename, "%s_P_R.%.4d.%.4d", gaugefilename_prefix, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Saving results to file %s\n", filename);
    sprintf(contype, "cvc-disc-P; spacial radii below rmin=%e or above rmax=%e (exclusive)", g_rmin, g_rmax);
    write_lime_contraction((double*)disc2, filename, 64, 16, contype, gid, Nsave);
/*
    sprintf(filename, "%s_P_R_ascii.%.4d.%.4d", gaugefilename_prefix, gid, Nsave);
    write_contraction((double*)disc2, NULL, filename, 16, 2, 0);
*/

    memcpy((void*)work, (void*)disc, 16*VOLUME*sizeof(fftw_complex)); 
#ifdef MPI
    fftwnd_mpi(plan_p, 16, work, disc, FFTW_NORMAL_ORDER);
#else
    fftwnd(plan_p, 16, work, 16, 1, work, 16, 1);
#endif      
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
        iix = 16 * ix + i;
        iiy = i*VOLUME + ix;
        w.re = cos(M_PI * (q[mu] - q[nu]));
        w.im = sin(M_PI * (q[mu] - q[nu]));
        _co_eq_co_ti_co(&w1, work+iix, &w);
        disc[iiy].re = fnorm * w1.re;
        disc[iiy].im = fnorm * w1.im;
        i++;
      }}
    }}}}

    sprintf(filename, "%s_P_r.%.4d.%.4d", gaugefilename_prefix, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Saving results to file %s\n", filename);
    sprintf(contype, "cvc-disc-P; spacial radii between rmin=%e and rmax=%e (inclusive)", g_rmin, g_rmax);
    write_lime_contraction((double*)disc, filename, 64, 16, contype, gid, Nsave);

/*
    sprintf(filename, "%s_P_r_ascii.%.4d.%.4d", gaugefilename_prefix, gid, Nsave);
    write_contraction((double*)disc, NULL, filename, 16, 2, 0);
*/

/*
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for Fourier transformation: %e seconds\n", retime-ratime);
*/
    if(g_cart_id==0) fprintf(stdout, "# Finished working on gauge id %d\n", gid);
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();

  fftw_free(disc);
  fftw_free(disc2);
  fftw_free(work);

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
