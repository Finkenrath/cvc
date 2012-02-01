/****************************************************
 * apply_Ddw.c
 *
 * Mon Jan 30 11:15:18 EET 2012
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
  double spinor1[24], spinor2[24];
  double *pl_gather=NULL;
  double dtmp;
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
    usage();
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

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(101);
  }

  geometry();
  init_geometry_5d();
  geometry_5d();

  VOL3 = LX*LY*LZ;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix, "identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    status = read_lime_gauge_field_doubleprec(filename);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
    // status = 0;
  }
  if(status != 0) {
    fprintf(stderr, "[apply_Dtm] Error, could not read gauge field\n");
    exit(11);
  }
  xchange_gauge();

  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value 1st field: %25.16e\n", plaq);

  g_kappa5d = 0.5 / (5. + g_m0);
  fprintf(stdout, "# [] g_kappa5d = %e\n", g_kappa5d);

  no_fields=4;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], L5*VOLUMEPLUSRAND);

/*
  items = VOL3 * 288;
  bytes = items * sizeof(float);
  if( (buffer = (float*)malloc( bytes ) ) == NULL ) {
    fprintf(stderr, "[] Error, could not allocate buffer\n");
    exit(20);
  }
*/
  /****************************************
   * read read the spinor fields
   ****************************************/


/*
  prop = create_fp_field(VOL3);
  create_fp(&prop2);
  create_fp(&prop_aux);
  create_fp(&prop_aux2);
  create_fp(&seq_prop);
  create_fp(&seq_prop2);
*/
#ifdef MPI
  if(!seed_set) { seed = g_seed; }
  srand(seed+g_cart_id);
  for(ix=0;ix<VOLUME*L5;ix++) {
    for(i=0;i<24;i++) {
      spinor1[i] = 2* (double)rand() / (double)RAND_MAX - 1.;
    }
    _fv_eq_fv(g_spinor_field[0]+_GSI(ix), spinor1 );
  }
  for(i=0;i<g_nproc;i++) {
    if(g_cart_id==i) {
      if(i==0) ofs = fopen("source", "w");
      else ofs = fopen("source", "a");
      for(is=0;is<L5;is++) { 
        for(x0=0;x0<T; x0++) {
        for(x1=0;x1<LX; x1++) {
        for(x2=0;x2<LX; x2++) {
        for(x3=0;x3<LX; x3++) {
          iix = is*VOLUME*g_nproc + (((x0+g_proc_coords[0]*T)*LX*g_nproc_x+ x1+g_proc_coords[1]*LX )*LY*g_nproc_y + x2+g_proc_coords[2]*LY )*LZ*g_nproc_z + x3+g_proc_coords[3]*LZ;
          ix = g_ipt_5d[is][x0][x1][x2][x3];
          for(c=0;c<24;c++) {
            fprintf(ofs, "%8d%8d%3d%25.16e\n", iix, ix, c, g_spinor_field[0][_GSI(ix)+c]);
          }
        }}}}
      }
      fclose(ofs);
    }
#ifdef MPI
    MPI_Barrier(g_cart_grid);
#endif
  }
#else
  ofs = fopen("source", "r");
  for(ix=0;ix<24*VOLUME*L5;ix++) {
    fscanf(ofs, "%d%d%d%lf", &x1,&x2,&x3, &dtmp);
    g_spinor_field[0][_GSI(x1)+x3] = dtmp;
  }
  fclose(ofs);

#endif
  xchange_field_5d(g_spinor_field[0]);
  Q_DW_Wilson_dag_phi(g_spinor_field[1], g_spinor_field[0]);
  xchange_field_5d(g_spinor_field[1]);
  Q_DW_Wilson_phi(g_spinor_field[2], g_spinor_field[1]);
  sprintf(filename, "prop_%.2d.%.2d", g_nproc, g_cart_id);
  ofs = fopen(filename, "w");
  printf_spinor_field_5d(g_spinor_field[2], ofs);
  fclose(ofs);

//  for(ix=0;ix<VOLUME*L5;ix++) {
//    for(i=0;i<24;i++) {
//      spinor1[i] = 2* (double)rand() / (double)RAND_MAX - 1.;
//    }
//    _fv_eq_fv(g_spinor_field[1]+_GSI(ix), spinor1 );
//  }
/*
  xchange_field_5d(g_spinor_field[0]);
  sprintf(filename, "spinor.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  printf_spinor_field_5d(g_spinor_field[0], ofs);
  fclose(ofs);
*/
/*
  // 2 = D 0
  Q_DW_Wilson_phi(g_spinor_field[2], g_spinor_field[0]);
  // 3 = D^dagger 1
  Q_DW_Wilson_dag_phi(g_spinor_field[3], g_spinor_field[1]);

  // <1, 2> = <1, D 0 >
  spinor_scalar_product_co(&w, g_spinor_field[1], g_spinor_field[2], VOLUME*L5);
  // <3, 0> = < D^dagger 1, 0 >
  spinor_scalar_product_co(&w2, g_spinor_field[3], g_spinor_field[0], VOLUME*L5);
  fprintf(stdout, "# [] w  = %e + %e*1.i\n", w.re, w.im);
  fprintf(stdout, "# [] w2 = %e + %e*1.i\n", w2.re, w2.im);
  fprintf(stdout, "# [] abs difference = %e \n", sqrt(_SQR(w2.re-w.re)+_SQR(w2.im-w.im)) );
*/

/*
  for(i=0;i<12;i++) {
    fprintf(stdout, "s1[%2d] <- %25.16e + %25.16e*1.i\n", i+1, spinor1[2*i], spinor1[2*i+1]);
  }
  for(i=0;i<24;i++) {
    spinor2[i] = 2* (double)rand() / (double)RAND_MAX - 1.;
  }
  for(i=0;i<12;i++) {
    fprintf(stdout, "s2[%2d] <- %25.16e + %25.16e*1.i\n", i+1, spinor2[2*i], spinor2[2*i+1]);
  }

  _fv_mi_eq_PRe_fv(spinor2, spinor1);
  for(i=0;i<12;i++) {
    fprintf(stdout, "s3[%2d] <- %25.16e + %25.16e*1.i\n", i+1, spinor2[2*i], spinor2[2*i+1]);
  }
*/
/*
  ofs = fopen("dw_spinor", "w");
  Q_DW_Wilson_phi(g_spinor_field[1], g_spinor_field[0]);
  printf_spinor_field(g_spinor_field[1], ofs);
  fclose(ofs);

  g_kappa = g_kappa5d;
  ofs = fopen("wilson_spinor", "w");
  Q_Wilson_phi(g_spinor_field[2], g_spinor_field[0]);
  printf_spinor_field(g_spinor_field[2], ofs);
  fclose(ofs);
*/
#ifdef _UNDEF
  /*******************************************************************
   * propagators
   *******************************************************************/
//  for(i=0; i<12;i++)
  for(i=0; i<1;i++)
  {

    //sprintf(file1, "source.%.4d.t00x00y00z00.%.2d.inverted", Nconf, i);

    sprintf(file1, "/home/mpetschlies/quda-0.3.2/tests/prop");
    if(g_cart_id==0) fprintf(stdout, "# Reading prop. from file %s\n", file1);
    fflush(stdout);
    //if( read_lime_spinor(g_spinor_field[0], file1, 0) != 0 ) {
    ofs = fopen(file1, "rb");
    if( fread(g_spinor_field[0], sizeof(double), 24*L5*VOLUME, ofs) !=  24*L5*VOLUME) {
      fprintf(stderr, "Error, could not read proper amount of data from file %s\n", file1);
      exit(100);
    }
    fclose(ofs);

    for(ix=0;ix<VOLUME*L5;ix++) {
      _fv_ti_eq_re(g_spinor_field[0]+_GSI(ix), 2.*g_kappa5d);
    }

/*
    if( (ofs = fopen("prop_full", "w")) == NULL ) exit(22);
    for(ix=0;ix<L5;ix++) {
      fprintf(ofs, "# [] s = %d\n", ix);
      printf_spinor_field(g_spinor_field[0]+_GSI(ix*VOLUME), ofs);
    }
    fclose(ofs);
*/

    // reorder, multiply with g2
    for(is=0,iix=0; is<L5; is++) {
    for(ix=0; ix<VOLUME; ix++) {
      iiy = lexic2eot_5d (is, ix);
      _fv_eq_fv(spinor1, g_spinor_field[0]+_GSI(iiy));
      _fv_eq_gamma_ti_fv(g_spinor_field[1]+_GSI(iix), 2, spinor1 );
      iix++;
    }}

    Q_DW_Wilson_phi(g_spinor_field[2], g_spinor_field[1]);
//    Q_DW_Wilson_dag_phi(g_spinor_field[2], g_spinor_field[1]);
    fprintf(stdout, "# [] finished  application of Dirac operator\n");
    fflush(stdout);


    // reorder, multiply with g2
    for(is=0, iix=0;is<L5;is++) {
    for(ix=0; ix<VOLUME; ix++) {
      iiy = lexic2eot_5d(is, ix);
      _fv_eq_fv(spinor1, g_spinor_field[2]+_GSI(iix));
      _fv_eq_gamma_ti_fv(g_spinor_field[1]+_GSI(iiy), 2, spinor1 );
      iix++;
    }}

    if( (ofs = fopen("my_out", "w")) == NULL ) exit(23);
    for(ix=0;ix<L5;ix++) {
      fprintf(ofs, "# [] s = %d\n", ix);
      printf_spinor_field(g_spinor_field[1]+_GSI(ix*VOLUME), ofs);
    }
    fclose(ofs);


    sprintf(file1, "/home/mpetschlies/quda-0.3.2/tests/source");
    if(g_cart_id==0) fprintf(stdout, "# Reading prop. from file %s\n", file1);
    fflush(stdout);
    //if( read_lime_spinor(g_spinor_field[0], file1, 0) != 0 ) {
    
    ofs = fopen(file1, "rb");
    if( fread(g_spinor_field[2], sizeof(double), 24*L5*VOLUME, ofs) !=  24*L5*VOLUME) {
      fprintf(stderr, "Error, could not read proper amount of data from file %s\n", file1);
      exit(100);
    }
    fclose(ofs);

    
/*
    if( (ofs = fopen("v_out", "w")) == NULL ) exit(23);
    for(ix=0;ix<L5;ix++) {
      fprintf(ofs, "# [] s = %d\n", ix);
      printf_spinor_field(g_spinor_field[2]+_GSI(ix*VOLUME), ofs);
    }
    fclose(ofs);
*/
    spinor_scalar_product_re(&norm2, g_spinor_field[2], g_spinor_field[2], VOLUME*L5);
    for(ix=0;ix<VOLUME*L5;ix++) {
      _fv_mi_eq_fv(g_spinor_field[1]+_GSI(ix), g_spinor_field[2]+_GSI(ix));
    }
    spinor_scalar_product_re(&norm, g_spinor_field[1], g_spinor_field[1], VOLUME*L5);
    fprintf(stdout, "\n# [] absolut residuum squared: %e; relative residuum %e\n", norm, sqrt(norm/norm2) );

  }  // of loop on spin color indices
#endif
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

  g_the_time = time(NULL);
  fprintf(stdout, "# [] %s# [] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [] %s# [] end fo run\n", ctime(&g_the_time));
  fflush(stderr);


#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}

