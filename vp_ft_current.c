/*********************************************************************************
 * vp_ft_current.c
 *
 * Sat Jan 30 12:07:05 CET 2010
 *
 * PURPOSE:
 * - calculate the Fourier transform of a contracted current 
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
#include "pidisc_model.h"

void usage() {
  fprintf(stdout, "Code to Fourier transform a current contraction\n");
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
  int x0, x1, x2, x3, ix;
  int gid, status;
  double *disc  = (double*)NULL;
  double q[4], fnorm;
  char filename[100], contype[200];
  double ratime, retime;
  complex w, w1, w2, *cp1, *cp2;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
#else
  fftwnd_plan plan_p, plan_m;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

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

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* vp_ft_current\n");
  fprintf(stdout, "**************************************************\n\n");

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
    exit(103);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(104);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  disc  = (double*)calloc(16*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(105);
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
    exit(108);
  }

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    for(ix=0; ix<16*VOLUME; ix++) disc[ix] = 0.;

    if(g_cart_id==0) fprintf(stdout, "# Start working on gauge id %d\n", gid);

    /* read the new contractions */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(Nsave>=0) {
      sprintf(filename, "%s.%.4d.%.4d", filename_prefix, gid, Nsave);
    } else {
      sprintf(filename, "%s.%.4d", filename_prefix, gid);
    }
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

    /***********************************************
     * Fourier transform
     ***********************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(mu=0; mu<4; mu++) {
      memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
      fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
      fftwnd_one(plan_m, in, NULL);
#endif      
      memcpy((void*)(disc+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
      fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
      fftwnd_one(plan_p, in, NULL);
#endif      
      memcpy((void*)(disc+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
    }

    for(mu=0; mu<4; mu++) {
      cp1 = (complex*)( disc + _GWI(mu,  0,VOLUME) );
      cp2 = (complex*)( disc + _GWI(4+mu,0,VOLUME) );
      for(x0=0; x0<T; x0++) {
        q[0] = (double)(x0+Tstart) / (double)T_global;
      for(x1=0; x1<LX; x1++) {
        q[1] = (double)x1 / (double)LX;
      for(x2=0; x2<LY; x2++) {
        q[2] = (double)x2 / (double)LY;
      for(x3=0; x3<LZ; x3++) {
        q[3] = (double)x3 / (double)LZ;
        w.re  =  cos(M_PI * q[mu]);
        w.im  =  sin(M_PI * q[mu]);
        w1.re =  cos(M_PI * q[mu]);
        w1.im = -sin(M_PI * q[mu]);
        _co_eq_co_ti_co(&w2, cp1, &w);
        _co_eq_co(cp1, &w2);
        _co_eq_co_ti_co(&w2, cp2, &w1);
        _co_eq_co(cp2, &w2);
        cp1++; cp2++;
      }}}}
    }

    /***********************************************
     * save results
     ***********************************************/
    if(Nsave>=0) {
      sprintf(filename, "%s_ftp.%.4d.%.4d", filename_prefix, gid, Nsave);
    } else {
      sprintf(filename, "%s_ftp.%.4d", filename_prefix, gid);
    }
    if(g_cart_id==0) fprintf(stdout, "# Saving +q-results to file %s\n", filename);
    sprintf(contype, "Fourier-transform-of-%s", filename_prefix);
/*    write_lime_contraction(disc, filename, 64, 4, contype, gid, Nsave); */
    write_contraction(disc, NULL, filename, 4, 2, 0);
    if(Nsave>=0) {
      sprintf(filename, "%s_ftn.%.4d.%.4d", filename_prefix, gid, Nsave);
    } else {
      sprintf(filename, "%s_ftn.%.4d", filename_prefix, gid);
    }
    if(g_cart_id==0) fprintf(stdout, "# Saving -q-results to file %s\n", filename);
    sprintf(contype, "Fourier-transform-of-%s", filename_prefix);
/*    write_lime_contraction(disc+_GWI(4,0,VOLUME), filename, 64, 4, contype, gid, Nsave); */
    write_contraction(disc+_GWI(4,0,VOLUME), NULL, filename, 4, 2, 0);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for Fourier transform and writing to file: %e seconds\n", 
      retime-ratime);

    if(g_cart_id==0) fprintf(stdout, "# Finished working on gauge id %d\n", gid);
  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  fftw_free(in);
  free(disc);
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
