#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
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
#include "gauge_io.h"

int main (int argc, char **argv) {
  int i,j, k, l;
  int c, x0, x1, x2, x3, ix, count;
  int status;
  double dtmp, dtmp2;
  char filename[400];
  int filename_set = 0;
  double plaq;
  double *smeared_gauge_field=NULL;
  // fermion_propagator_type fp1=NULL, fp2=NULL, fp3 = NULL;
  double ratime, retime;
  unsigned int VOL3;
  time_t ttime1, ttime2;

  // FILE *ofs=NULL;

  while ((c = getopt(argc, argv, "h?f:l:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      break;
    }
  }

#ifdef OPENMP
//#pragma omp parallel shared(g_num_threads)
//  { if(omp_get_thread_num() == 0) { g_num_threads = omp_get_num_threads(); } }
#endif 

  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);


  fprintf(stdout, "# [] global thread number = %d\n", g_num_threads);
  
  
  /* initialize MPI parameters */
  mpi_init(argc, argv);

  VOL3 = LX*LY*LZ;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
                  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
                  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
                  "# [%2d] LYstart      = %3d\n"\
                  "# [%2d] LZ_global    = %3d\n"\
                  "# [%2d] LZ           = %3d\n"\
                  "# [%2d] LZstart      = %3d\n",
                  g_cart_id, g_cart_id, T_global,  g_cart_id, T,  g_cart_id, Tstart, 
                  g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
                  g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart,
                  g_cart_id, LZ_global, g_cart_id, LZ, g_cart_id, LZstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  // read the gauge field
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "conf.%.5d", Nconf);
  fprintf(stdout, "# Reading gauge field from file %s\n", filename);
  status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
//  sprintf(filename, "conf.%.4d", Nconf);
//  fprintf(stdout, "# Reading gauge field from file %s\n", filename);
//  status = read_lime_gauge_field_doubleprec(filename);

  if(status != 0) {
    fprintf(stderr, "[] Error, could not read gauge field");
  }
  fprintf(stdout, "# [] read plaquette value %25.16e\n", plaq);

  plaquette(&plaq);
  fprintf(stdout, "# [] calculated plaquette value: %25.16e\n", plaq);


  // smeared gauge field
  alloc_gauge_field(&smeared_gauge_field, VOLUMEPLUSRAND);
  memcpy(smeared_gauge_field, g_gauge_field, VOLUME*72*sizeof(double));
//  smeared_gauge_field = g_gauge_field;  
  plaquette2(&plaq, smeared_gauge_field);
  fprintf(stdout, "# [] calculated plaquette value of smeared_gauge_field = %25.16e\n", plaq);

  /*********************************************************************
   * test the APE smearing; compare the previously used serial version
   *   with the threaded one
   *********************************************************************/

#ifdef OPENMP
  ttime1 = time(NULL);
  //APE_Smearing_Step_threads(g_gauge_field, N_ape, alpha_ape);
  for(x0=0;x0<T;x0++) {
    APE_Smearing_Step_Timeslice_threads(g_gauge_field+_GGI(x0*VOL3,0), N_ape, alpha_ape);
  }
  ttime2 = time(NULL);
  fprintf(stdout, "# [main] time for threaded APE smearing = %e\n", difftime(ttime2, ttime1));
  plaquette2(&plaq, g_gauge_field);
  fprintf(stdout, "# [] calculated plaquette value of g_gauge_field after: %25.16e\n", plaq);
#endif
  ttime1 = time(NULL);
  APE_Smearing_Step(smeared_gauge_field, N_ape, alpha_ape);
  ttime2 = time(NULL);
  fprintf(stdout, "# [main] time for serial APE smearing = %e\n", difftime(ttime2, ttime1));
  plaquette2(&plaq, smeared_gauge_field);
  fprintf(stdout, "# [] calculated plaquette value of smeared_gauge_field after: %25.16e\n", plaq);

  dtmp = fabs( g_gauge_field[0] - smeared_gauge_field[0] );
  x0 = 0;
  for(ix=1;ix<72*VOLUME;ix++) {
    double dtmp2 = fabs( g_gauge_field[ix] - smeared_gauge_field[ix] );
    if(dtmp2>dtmp) {
      dtmp=dtmp2;
      x0 = ix;
    }
  }
  fprintf(stdout, "# [] maximal absolut difference = %e at site %d; values: (%25.16e, %25.16e)\n", dtmp, x0, g_gauge_field[x0], smeared_gauge_field[x0]);

#ifdef _UNDEF
  // alloc spinor fields; read
  no_fields = 3;
  if(no_fields>0) {
    g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
    for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  }

  status = read_lime_spinor(g_spinor_field[0], filename_prefix, 0);
  memcpy(g_spinor_field[1], g_spinor_field[0], 24*VOLUME*sizeof(double));

  /*********************************************************************
   * test the Jacobi smearing; compare the previously used serial version
   *   with the threaded one
   *********************************************************************/
#ifdef OPENMP
//  ratime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# [] threaded Jacobi smearing with paramters N_Jacobi=%d, kappa_Jacobi=%f\n", N_Jacobi, kappa_Jacobi);
  ttime1 = time(NULL);
//  ratime = (double)clock() / CLOCKS_PER_SEC;
  for(x0=0;x0<T;x0++) {
    Jacobi_Smearing_Step_one_Timeslice_threads(smeared_gauge_field+_GGI(x0*VOL3,0), g_spinor_field[0]+_GSI(x0*VOL3), g_spinor_field[2], N_Jacobi, kappa_Jacobi);
  }
  ttime2 = time(NULL);
  fprintf(stdout, "# [main] time for threaded smearing = %e\n", difftime(ttime2, ttime1));
//  retime = (double)clock() / CLOCKS_PER_SEC;
//  fprintf(stdout, "# [main] time for threaded smearing = %e \n", retime-ratime);
#endif
  fprintf(stdout, "# [] serial Jacobi smearing with paramters N_Jacobi=%d, kappa_Jacobi=%f\n", N_Jacobi, kappa_Jacobi);
  ttime1 = time(NULL);
//  ratime = (double)clock() / CLOCKS_PER_SEC;
  Jacobi_Smearing_Step_one(smeared_gauge_field, g_spinor_field[1], g_spinor_field[2], N_Jacobi, kappa_Jacobi);
  ttime2 = time(NULL);
  fprintf(stdout, "# [main] time for serial smearing = %e\n", difftime(ttime2, ttime1));
//  retime = (double)clock() / CLOCKS_PER_SEC;
//  fprintf(stdout, "# [main] time for serial smearing = %e \n", retime-ratime);

  // find maximum deviation
  dtmp = fabs( g_spinor_field[0][0] - g_spinor_field[1][0] );
  x0 = 0;
  for(ix=1;ix<24*VOLUME;ix++) {
    dtmp2 = fabs( g_spinor_field[0][ix] - g_spinor_field[1][ix] );
    if(dtmp2>dtmp) {
      dtmp=dtmp2;
      x0 = ix;
    }
  }
  fprintf(stdout, "# [] maximal absolut difference = %e at site %d; values: (%25.16e, %25.16e)\n", dtmp, x0, g_spinor_field[0][x0], g_spinor_field[1][x0]);
#endif
/*
  create_fp(&fp1);
  create_fp(&fp2);
  create_fp(&fp3);
  
  for(i=0;i<g_sv_dim;i++) {
  for(j=0;j<g_sv_dim;j++) {
    for(k=0;k<g_cv_dim;k++) {
      fp1[3*i+k][2*(3*j+k)] = 1.;
    }
  }}

  printf_fp(fp1, "(fp)", stdout);

  _fp_eq_cm_ti_fp(fp2, g_gauge_field+_GGI(0,0), fp1);

  printf_fp(fp2, "(cm fp)", stdout);

  _fp_eq_fp_ti_cm_dagger(fp3, g_gauge_field+_GGI(0,0), fp2);

  printf_fp(fp3, "(cm fp cm^dagger)", stdout);

  dtmp = fabs( fp1[0][0] - fp3[0][0] );
  for(i=0;i<g_fv_dim;i++) {
  for(j=0;j<2*g_fv_dim;j++) {
    dtmp2 = fabs( fp1[i][j] - fp3[i][j] );
    if(dtmp2>dtmp) dtmp = dtmp2;
  }}
  fprintf(stdout, "# [] maximal differnce = %e\n", dtmp);

  free_fp(&fp1);
  free_fp(&fp2);
  free_fp(&fp3);
*/
  if(g_gauge_field!=NULL) free(g_gauge_field);
  if(no_fields>0) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field);
  }

  return(0);

}

