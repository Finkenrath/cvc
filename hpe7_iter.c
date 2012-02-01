/****************************************************
 * hpe7_iter.c
 *
 * Wed Nov 18 11:02:41 CET 2009
 *
 * TODO: 
 * DONE:
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
  fprintf(stdout, "Code to perform quark-disconnected conserved vector current contractions\n");
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
  int dxm[4], dxn[4], ixpm, ixpn;
  int sid, steps[4], nloop;
  int *tcf, *tcb, **loop_tab;
  double *disc  = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq, _2kappamu, hpe3_coeff, onepmutilde2, mutilde2;
  double spinor1[24], spinor2[24], spinor3[24], U_[18], U1_[18], U2_[18];
  double *gauge_trafo=(double*)NULL;
  complex w, w1, w2, *cp1, *cp2, *cp3;
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

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
#ifdef MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  if(do_gt==1) {
    /***********************************
     * initialize gauge transformation
     ***********************************/
    init_gauge_trafo(&gauge_trafo, 1.);
    apply_gt_gauge(gauge_trafo);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "measured plaquette value after gauge trafo: %25.16e\n", plaq);
  }

/*
  test_hpem();
  test_cm_eq_cm_dag_ti_cm();
*/


  if(g_cart_id==0) fprintf(stdout, "calling init_trace_coeff\n");
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  init_trace_coeff(&tcf, &tcb, &loop_tab, 5, &nloop);
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "time to initialize trace coeff.s: %e seconds\n", retime-ratime);


  /****************************************
   * allocate memory for the contractions
   ****************************************/
  disc  = (double*)calloc( 16*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
  for(ix=0; ix<16*VOLUME; ix++) disc[ix] = 0.;

  /*******************************************
   * test Hopping_iter
   *******************************************/

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      Hopping_iter(disc+_GWI(mu,ix,VOLUME), disc+_GWI(4+mu,ix,VOLUME), tcf, tcb, ix, mu, 
        5, nloop, loop_tab);
    }
  }
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "time to run Hopping_iter: %e seconds\n", retime-ratime);

  sprintf(filename, "hpe7_5th.%.4d", Nconf);
  ofs = fopen(filename, "w");
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      fprintf(ofs, "%6d%3d%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e\n",
        ix, mu, disc[_GWI(mu,ix,VOLUME)], disc[_GWI(mu,ix,VOLUME)+1],
                disc[_GWI(4+mu,ix,VOLUME)], disc[_GWI(4+mu,ix,VOLUME)+1],
        disc[_GWI(mu,ix,VOLUME)]+disc[_GWI(4+mu,ix,VOLUME)],
        disc[_GWI(mu,ix,VOLUME)+1]+disc[_GWI(4+mu,ix,VOLUME)+1]);
    }
  } 
  fclose(ofs);

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  free_geometry();
  free(disc);

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
