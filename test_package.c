/****************************************************
 * test_package.c
 *
 * Wed Dec  9 21:06:00 CET 2009
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
#include "read_input_parser.h"
#include "invert_Qtm.h"
#include "gauge_io.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to test teh cvc package\n");
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
  
  int c, i, mu, status;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int sl0, sl1, sl2, sl3, have_source_flag=0;
  double *disc, *work;
  double fnorm, q[4];
  int do_gt   = 0;
  char filename[200];
  double ratime, retime;
  double plaq, norm, norm2;
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1, *cp1, *cp2, *cp3;
  FILE *ofs;

  fftw_complex *in = (fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
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

  /**************************************
   * set the default values, read input
   **************************************/
  if(filename_set==0) strcpy(filename, "cvc.input.test");
  if(g_proc_id==0) fprintf(stdout, "# Reading test input from file %s\n", filename);
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

  dims[0] = T_global;
  dims[1] = LX;
  dims[2] = LY;
  dims[3] = LZ;
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
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] l_LX_at      = %3d\n"\
                  "# [%2d] l_LXstart_at = %3d\n"\
                  "# [%2d] FFTW_LOC_VOLUME = %3d\n",\
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at, g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

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

  /*****************************************
   * initialize the gauge field 
   *****************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  if(g_cart_id==0) fprintf(stdout, "# Using seed = %u\n", g_seed);
  srand(g_seed);
  random_gauge_field2(g_gauge_field);
  xchange_gauge();

/*
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      for(i=0; i<9; i++) fprintf(stdout, "[%d] gauge[%8d,%3d]\t(%3d,%3d)%25.16e +i %25.16e\n", g_cart_id, ix, 
        mu, i/3, i%3, g_gauge_field[_GGI(ix,mu)+2*i], g_gauge_field[_GGI(ix,mu)+2*i+1]);
    }
  }
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      _co_eq_det_cm(&w, g_gauge_field+_GGI(ix, mu));
      fprintf(stdout, "[%d] %8d%3d%25.16e +i %25.16e\n", g_cart_id, ix, mu, w.re, w.im);
      _cm_eq_cm_ti_cm_dag(U_, g_gauge_field+_GGI(ix,mu), g_gauge_field+_GGI(ix,mu));
      U_[ 0] -= 1.;
      U_[ 8] -= 1.;
      U_[16] -= 1.;
      norm=-1.;
      for(i=0; i<18; i++) { norm = _MAX(norm, fabs(U_[i])); }
      fprintf(stdout, "[%d] (%d, %d) max diff from unity %25.16e\n", g_cart_id, ix, mu, norm);
    }
  }
*/



  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# Measured plaquette value: %25.16e\n", plaq);


  /*************************************************
   * (1) check write/read of gauge field
   *************************************************/
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# Writing gauge field to file %s\n", filename);
  write_lime_gauge_field(filename, plaq, Nconf, 64);

  exit(0);

  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# Reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();

/*
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      for(i=0; i<9; i++) fprintf(stdout, "[%d] gauge[%8d,%3d]\t(%3d,%3d)%25.16e +i %25.16e\n", g_cart_id, ix, mu, i/3, i%3, g_gauge_field[_GGI(ix,mu)+2*i], g_gauge_field[_GGI(ix,mu)+2*i+1]); 
      _cm_eq_cm(U_, g_gauge_field+_GGI(ix,mu));
      _cm_eq_cm(g_gauge_field+_GGI(ix,mu), U_);
      _cm_eq_id(g_gauge_field+_GGI(ix, mu));

    }
  }

*/
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# Measured plaquette value after rereading: %25.16e\n", plaq);


  /* allocate memory for the spinor fields */
  no_fields = 10;
  if(no_fields>0) {
    g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
    for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);
  }
 
  /* the source locaton */
  sl0 = g_source_location/(LX*LY*LZ);
  sl1 = ( g_source_location%(LX*LY*LZ) ) / (LY*LZ);
  sl2 = ( g_source_location%(LY*LZ) ) / (LZ);
  sl3 = g_source_location%LZ;
  if(g_cart_id==0) fprintf(stdout, "# global sl = (%d, %d, %d, %d)\n", sl0, sl1, sl2, sl3);
  have_source_flag = sl0-Tstart>=0 && sl0-Tstart<T;
  sl0 -= Tstart;
  fprintf(stdout, "# [%d] have source: %d\n", g_cart_id, have_source_flag);
  if(have_source_flag==1) fprintf(stdout, "# local sl = (%d, %d, %d, %d)\n", sl0, sl1, sl2, sl3);
  
  /******************************************************
   * (2) check the Dirac operator / invert point sources 
   ******************************************************/

  for(i=0; i<12; i++) {

    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_zero(g_spinor_field[0]+_GSI(ix));
    }
    if(have_source_flag==1) {
      g_spinor_field[0][_GSI(g_ipt[sl0][sl1][sl2][sl3])+2*i] = 2.*sqrt(15.);
    }
    xchange_field(g_spinor_field[0]);

    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_zero(g_spinor_field[1]+_GSI(ix));
    }
    xchange_field(g_spinor_field[1]);

/*    if( (status = invert_Qtm(g_spinor_field[1], g_spinor_field[0], 2)) < 0 ) { */
    if( (status = invert_Qtm_her(g_spinor_field[1], g_spinor_field[0], 2)) < 0 ) {
      fprintf(stderr, "Error from imvert_Qtm, exit\n");
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(201);
    }

    sprintf(filename, "source.%.4d.%.2d.inverted", Nconf, 4*12+i);
    if(g_cart_id==0) fprintf(stdout, "# Writing prop. number %d from file %s\n", 4*12+i, filename);
    write_propagator(g_spinor_field[1], filename, 0, 64);

    if(i==0) {
      if(g_cart_id==0) fprintf(stdout, "# Rereading prop. number %d from file %s\n", 4*12+i, filename);
      read_lime_spinor(g_spinor_field[0], filename, 0);
    }

  }


  for(mu=0; mu<4; mu++) {
    for(i=0; i<12; i++) {
      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_zero(g_spinor_field[0]+_GSI(ix));
      }
      if(have_source_flag==1) {
        g_spinor_field[0][_GSI(g_iup[g_ipt[sl0][sl1][sl2][sl3]][mu])+2*i] = 2.*sqrt(15.);
      }
      xchange_field(g_spinor_field[0]);

      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_zero(g_spinor_field[1]+_GSI(ix));
      }
      xchange_field(g_spinor_field[1]);

/*      if( (status = invert_Qtm(g_spinor_field[1], g_spinor_field[0], 2)) < 0 ) { */
      if( (status = invert_Qtm_her(g_spinor_field[1], g_spinor_field[0], 2)) < 0 ) {
        fprintf(stdout, "Error from imvert_Qtm, exit\n");
#ifdef MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
#endif
        exit(201);
      }

      sprintf(filename, "source.%.4d.%.2d.inverted", Nconf, mu*12+i);
      if(g_cart_id==0) fprintf(stdout, "# Writing prop. number %d from file %s\n", mu*12+i, filename);
      write_propagator(g_spinor_field[1], filename, 0, 64);

    }
  }


  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
 
  if( (disc = (double*)malloc(32*VOLUME*sizeof(double))) == (double*)NULL) {
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(102);
  }
   
  /***********************************************
   * (3) check i/o of contractions 
   ***********************************************/

  for(mu=0; mu<16; mu++) {
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
        disc[_GWI(mu,ix,VOLUME)  ] =  2. * ( mu * (double)(T_global*LX*LY*LZ) + (double)( (((x0+Tstart)*LX + x1)*LY + x2)*LZ + x3 ) );
        disc[_GWI(mu,ix,VOLUME)+1] =  2. * ( mu * (double)(T_global*LX*LY*LZ) + (double)( (((x0+Tstart)*LX + x1)*LY + x2)*LZ + x3 ) ) + 1;
    }}}}
  }

  sprintf(filename, "contraction_test.%.4d.%.2d", Nconf, 0);
  if(g_cart_id==0) fprintf(stdout, "# writing disc to file %s\n", filename);
  write_lime_contraction(disc, filename, 64, 16, "cvc_test_X", Nconf, 0);
 
  if(g_cart_id==0) fprintf(stdout, "# reading disc from file %s\n", filename);
  read_lime_contraction(disc, filename, 16, 0);
 
  norm = -1.;
  for(mu=0; mu<16; mu++) {
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      norm2 = fabs( 2.*( (double)mu*(double)(T_global*LX*LY*LZ) + (double)((((x0+Tstart)*LX + x1)*LY + x2)*LZ + x3)) - disc[_GWI(mu,ix,VOLUME)] );
      if(norm2>norm) norm = norm2;
       
      norm2 = fabs( 2.*( (double)mu*(double)(T_global*LX*LY*LZ) + (double)((((x0+Tstart)*LX + x1)*LY + x2)*LZ + x3))+1. - disc[_GWI(mu,ix,VOLUME)+1] );
      if(norm2>norm) norm = norm2;
    }}}}
  }
  fprintf(stdout, "# [%d] max diff = %25.16e\n", g_cart_id, norm);


  if( (work = (double*)malloc(4*VOLUME*sizeof(double))) == (double*)NULL) {
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(112);
  }

  
  /***********************************************
   * (4) check Fourier transformation
   ***********************************************/

  if( (in = (fftw_complex*)fftw_malloc(2*VOLUME*sizeof(fftw_complex))) == (fftw_complex*)NULL ) {
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(103);
  }

  q[0] = (double)(sl0+Tstart) / (double)T_global;
  q[1] = (double)sl1 / (double)LX;
  q[2] = (double)sl2 / (double)LY;
  q[3] = (double)sl3 / (double)LZ;
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    disc[_GWI(0,ix,VOLUME)  ] =  cos( 2. * M_PI * (q[0]*(double)(x0+Tstart) +  q[1]*(double)x1 + q[2]*(double)x2 + q[3]*(double)x3 ) );
    disc[_GWI(0,ix,VOLUME)+1] = -sin( 2. * M_PI * (q[0]*(double)(x0+Tstart) +  q[1]*(double)x1 + q[2]*(double)x2 + q[3]*(double)x3 ) );
 
    disc[_GWI(1,ix,VOLUME)  ] =  cos( 2. * M_PI * (q[0]*(double)(x0+Tstart) +  q[1]*(double)x1 + q[2]*(double)x2 + q[3]*(double)x3 ) );
    disc[_GWI(1,ix,VOLUME)+1] =  sin( 2. * M_PI * (q[0]*(double)(x0+Tstart) +  q[1]*(double)x1 + q[2]*(double)x2 + q[3]*(double)x3 ) );
  }}}}
 
  memcpy((void*)in, (void*)disc, 2*VOLUME*sizeof(double));
#ifdef MPI
  fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
  fftwnd_one(plan_p, in, NULL);
#endif
  memcpy((void*)work, (void*)in, 2*VOLUME*sizeof(double));

  memcpy((void*)in, (void*)(disc+_GWI(1,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
  fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
  fftwnd_one(plan_m, in, NULL);
#endif
  memcpy((void*)(work+_GWI(1,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

  if(have_source_flag==1) {
    work[_GWI(0, g_ipt[sl0][sl1][sl2][sl3], VOLUME)] /= (double)(T_global*LX*LY*LZ);
    work[_GWI(0, g_ipt[sl0][sl1][sl2][sl3], VOLUME)] -= 1.;

    work[_GWI(1, g_ipt[sl0][sl1][sl2][sl3], VOLUME)] /= (double)(T_global*LX*LY*LZ);
    work[_GWI(1, g_ipt[sl0][sl1][sl2][sl3], VOLUME)] -= 1.;
  }

  norm=-1.;
  for(ix=0; ix<2*VOLUME; ix++) {
    norm2 = fabs(work[ix]);
    if(norm2>norm) norm = norm2;
  }
  fprintf(stdout, "# [%d] max diff for first part = %25.16e\n", g_cart_id, norm);

  norm=-1.;
  for(ix=2*VOLUME; ix<4*VOLUME; ix++) {
    norm2 = fabs(work[ix]);
    if(norm2>norm) norm = norm2;
  }
  fprintf(stdout, "# [%d] max diff for second part = %25.16e\n", g_cart_id, norm);

  free(in);

  free(disc); free(work);

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
