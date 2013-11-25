/****************************************************
 * test_avc_ft.ci
 *
 * Mon Oct 12 13:03:41 CEST 2009
 *
 * TODO: 
 * CHANGES:
 ****************************************************/

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
#include "Q_phi.h"

void usage() {
  fprintf(stdout, "Code to perform light neutral contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, nu;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int it, iy, iz;
  int sid;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
#ifdef AVC_WI
  double *pseu=(double*)NULL, *scal=(double*)NULL, *xavc=(double*)NULL; 
#endif
  double *work = (double*)NULL;
  double *disc_diag = (double*)NULL;
  double *disc_diag2 = (double*)NULL;
  double q[4], fnorm;
  double unit_trace[2], shift_trace[2], D_trace[2];
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1, *cp1, *cp2, *cp3;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
  int *status;
#else
  fftwnd_plan plan_p, plan_m;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vgf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'g':
      do_gt = 1;
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
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");

  /* read the input file */
  read_input(filename);

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
#ifdef MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

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

  /* allocate memory for the contractions */
#ifdef CVC
  disc  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
  for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
#endif

  work  = (double*)calloc(48*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  /* read the contractions */

  sprintf(filename, "%s", filename_prefix);
  if( (ofs = fopen(filename, "r")) ==(FILE*)NULL) exit(110);
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      fscanf(ofs, "%lf%lf", disc+_GWI(mu,ix,VOLUME), disc+_GWI(mu,ix,VOLUME)+1);
    }
  }
  fclose(ofs);

  for(mu=0; mu<4; mu++) {
    memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

    memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    fftwnd_one(plan_p, in, NULL);
    memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
  }

  for(it=0; it<T; it++) {
   q[0] = M_PI * (double)it / (double)T;
  for(ix=0; ix<LX; ix++) {
   q[1] = M_PI * (double)ix / (double)LX;
  for(iy=0; iy<LY; iy++) {
   q[2] = M_PI * (double)iy / (double)LY;
  for(iz=0; iz<LZ; iz++) {
   q[3] = M_PI * (double)iz / (double)LZ;
   iix = g_ipt[it][ix][iy][iz];
   fprintf(stdout, "t=%3d, x=%3d, y=%3d,z=%3d\n", it, ix, iy, iz);
   for(mu=0; mu<4; mu++) {
   for(nu=0; nu<4; nu++) {
     w.re = cos( q[mu] - q[nu] ) / VOLUME / 4.;
     w.im = sin( q[mu] - q[nu] ) / VOLUME / 4.;
     _co_eq_co_ti_co(&w1, (complex*)(work+_GWI(mu,iix,VOLUME)), (complex*)(work+_GWI(4+nu,iix,VOLUME)));
     _co_eq_co_ti_co((complex*)(work+_GWI(8+4*mu+nu,iix,VOLUME)), &w1, &w);
     fprintf(stdout, "%3d%25.16e%25.16e\n", 4*mu+nu, work[_GWI(8+4*mu+nu,iix,VOLUME)], work[_GWI(8+4*mu+nu,iix,VOLUME)+1]);
   }
   }
  }
  }
  }
  }


/*
  sprintf(filename, "%s", filename_prefix2);
  ofs = fopen(filename, "r");
  fread(work, sizeof(double), 32*VOLUME, ofs);
  fclose(ofs);
  x0=-2;
  for(it=0; it<T; it++) {
  for(ix=0; ix<LX; ix++) {
  for(iy=0; iy<LY; iy++) {
  for(iz=0; iz<LZ; iz++) {
   iix = g_ipt[it][ix][iy][iz];
   fprintf(stdout, "t=%3d, x=%3d, y=%3d,z=%3d\n", it, ix, iy, iz);
   for(mu=0; mu<4; mu++) {
   for(nu=0; nu<4; nu++) {
     x0+=2;
     fprintf(stdout, "%3d%25.16e%25.16e\n", 4*mu+nu, work[x0], work[x0+1]);
   }
   }
  }
  }
  }
  }
*/
  /* free the allocated memory, finalize */
  free_geometry();
  fftw_free(in);
  free(disc);

#ifdef AVC
  free(disc2);
#endif

  free(work);

  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);

  return(0);

}
