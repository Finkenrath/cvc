/****************************************************
 * apply_Ddw_v2.c
 *
 * Fri Feb 10 08:56:31 EET 2012
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#ifdef MPI
#  include <mpi.h>
#endif
#ifdef OPENMP
#include <omp.h>
#endif

#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "fuzz.h"
#include "fuzz2.h"
#include "smearing_techniques.h"

#define _SQR(_a) ((_a)*(_a))

void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu, nu, status;
  int i, j, ncon=-1, ir, is, ic, id;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3, iy, iiy;
  int start_valuet=0, start_valuex=0, start_valuey=0;
  int num_threads=1, threadid, nthreads;
  int seed, seed_set=0;
  double diff1, diff2;
/*  double *chi=NULL, *psi=NULL; */
  double plaq=0., pl_ts, pl_xs, pl_global;
  double *gauge_field_smeared = NULL;
  double s[18], t[18], u[18], pl_loc;
  double spinor1[24], spinor2[24], *work=NULL;
  double *pl_gather=NULL;
  double dtmp, *gauge_qdp=NULL;
  complex prod, w, w2;
  int verbose = 0;
  char filename[200];
  char file1[200];
  char file2[200];
  FILE *ofs=NULL;
  double norm, norm2;
  fermion_propagator_type *prop=NULL, prop2=NULL, seq_prop=NULL, seq_prop2=NULL, prop_aux=NULL, prop_aux2=NULL;
  int idx, eoflag, shift;
  float *buffer = NULL;
  unsigned int VOL3;
  size_t items, bytes;
  int mu_trans[4] = {3, 0, 1, 2};
#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:N:c:C:t:s:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      ncon = atoi(optarg);
      break;
    case 'c':
      strcpy(file1, optarg);
      break;
    case 'C':
      strcpy(file2, optarg);
      break;
    case 't':
      num_threads = atoi(optarg);
      break;
    case 's':
      seed = atoi(optarg);
      fprintf(stdout, "# [] use seed value %d\n", seed);
      seed_set = 1;
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
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);


  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    //usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize T etc. */
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",\
		  g_cart_id, g_cart_id, T_global, g_cart_id, T, g_cart_id, Tstart,
		             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
		             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  check_error( init_geometry(),    "init_geometry",    NULL, 101);
  geometry();
  check_error( init_geometry_5d(), "init_geometry_5d", NULL, 103);
  geometry_5d();


  VOL3 = LX*LY*LZ;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);

  if(strcmp(gaugefilename_prefix, "identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    // if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    // status = read_lime_gauge_field_doubleprec(filename);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
    status = 0;
  }
  if(status != 0) {
    fprintf(stderr, "[apply_Dtm] Error, could not read gauge field\n");
    exit(11);
  }
//  xchange_gauge();
//  if(g_cart_id==0) fprintf(stdout, "# read plaquette value 1st field: %25.16e\n", plaq);


  g_kappa5d = 0.5 / (5. + g_m5);
  fprintf(stdout, "# [] g_kappa5d = %e\n", g_kappa5d);

  no_fields=4;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], L5*VOLUMEPLUSRAND);
  work = g_spinor_field[no_fields-1];


  /****************************************
   * read read the gauge field
   ****************************************/
  items = VOLUME*18;
  bytes = sizeof(double);
  if( (gauge_qdp = (double*)malloc(items*bytes)) == NULL ) EXIT(109);

  sprintf(filename, "gauge.%.2d", g_cart_id);
  if( (ofs = fopen(filename, "r")) == NULL ) EXIT(107);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);

  // read direction i
  for(i=0;i<4;i++) {
    if( fread(gauge_qdp, bytes, items, ofs) != items ) EXIT(108);
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_cm(g_gauge_field+_GGI(ix, (i+1)%4), gauge_qdp+18*g_lexic2eot[ix]);
    }
  }
  fclose(ofs);
  free(gauge_qdp);

  if(g_proc_coords[0]==g_nproc_t-1) {
    fprintf(stdout, "# [] process%.2d multiplies gauge field\n", g_cart_id);
    for(ix=0;ix<LX*LY*LZ;ix++) { _cm_ti_eq_re(g_gauge_field+_GGI((T-1)*LX*LY*LZ+ix, 0), -1.); }
  }

#ifdef MPI
  xchange_gauge();
#endif

  // measure the plaquette
  plaquette(&plaq);
  if(g_cart_id==0) {
    fprintf(stdout, "# measured plaquette value 1st field: %25.16e\n", plaq);
    fflush(stdout);
  }
#ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  /****************************************
   * read read the spinor fields
   ****************************************/

  items = VOLUME*L5*24;
  bytes = sizeof(double);
  sprintf(filename, "source.%.2d", g_cart_id);
  if( (ofs = fopen(filename, "r")) == NULL ) EXIT(101);
  if( fread(work, bytes, items, ofs) != items ) EXIT(102);
  fclose(ofs);

  // transform to my basis
  for(is=0;is<L5;is++) {
    for(ix=0;ix<VOLUME;ix++) {
      iy  = lexic2eot_5d(is, ix);
      iix = is*VOLUME + ix;
      _fv_eq_fv(spinor1, work+_GSI(iy));
      _fv_eq_gamma_ti_fv(g_spinor_field[0]+_GSI(iix), 2, spinor1);
  }}
  xchange_field_5d(g_spinor_field[0]);
  Q_DW_Wilson_phi(g_spinor_field[1], g_spinor_field[0]);

  // normalize
  for(ix=0;ix<VOLUME*L5;ix++) { _fv_ti_eq_re(g_spinor_field[1], 2.*g_kappa5d); }
  xchange_field_5d(g_spinor_field[1]);


  sprintf(filename, "Dsource.%.2d", g_cart_id);
  if( (ofs = fopen(filename, "w")) == NULL ) exit(106);
  printf_spinor_field_5d(g_spinor_field[1], ofs);

  items = VOLUME*L5*24;
  bytes = sizeof(double);
  sprintf(filename, "prop.%.2d", g_cart_id);
  if( (ofs = fopen(filename, "r")) == NULL ) exit(103);
  if( fread(work, bytes, items, ofs) != items ) exit(104);
  fclose(ofs);
 
  // transform to my basis
  for(is=0;is<L5;is++) {
    for(ix=0;ix<VOLUME;ix++) {
      iy  = lexic2eot_5d(is, ix);
      iix = is*VOLUME + ix;
      _fv_eq_fv(spinor1, work+_GSI(iy));
      _fv_eq_gamma_ti_fv(g_spinor_field[2]+_GSI(iix), 2, spinor1);
  }}
  xchange_field_5d(g_spinor_field[2]);
  sprintf(filename, "prop_rot.%.2d", g_cart_id);
  if( (ofs = fopen(filename, "w")) == NULL ) exit(105);
  printf_spinor_field_5d(g_spinor_field[2], ofs);

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  free_geometry();
  if(gauge_field_smeared != NULL) free(gauge_field_smeared);
  if(g_spinor_field != NULL) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field);
  }
  free(buffer);

  free_fp_field(&prop);
  free_fp(&prop2);
  free_fp(&prop_aux);
  free_fp(&prop_aux2);
  free_fp(&seq_prop);
  free_fp(&seq_prop2);

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [] %s# [] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [] %s# [] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}

