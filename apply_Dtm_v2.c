/****************************************************
 * apply_Dtm_v2.c
 *
 * Wed Jan 18 16:48:55 EET 2012
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

void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu, nu, status;
  int i, j, ncon=-1, ir, is, ic, id;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  int start_valuet=0, start_valuex=0, start_valuey=0;
  int num_threads=1, threadid, nthreads;
  double diff1, diff2;
/*  double *chi=NULL, *psi=NULL; */
  double plaq=0., pl_ts, pl_xs, pl_global;
  double *gauge_field_smeared = NULL;
  double s[18], t[18], u[18], pl_loc;
  double spinor1[24], spinor2[24];
  double *pl_gather=NULL;
  complex prod, w;
  int verbose = 0;
  char filename[200];
  char file1[200];
  char file2[200];
  FILE *ofs=NULL;
  double norm, norm2;
  fermion_propagator_type *prop=NULL, prop2=NULL, seq_prop=NULL, seq_prop2=NULL, prop_aux=NULL, prop_aux2=NULL;
  int idx;
  float *buffer = NULL;
  unsigned int VOL3;
  size_t items, bytes;

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:N:c:C:t:")) != -1) {
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

  VOL3 = LX*LY*LZ;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);

  //status = read_lime_gauge_field_doubleprec(filename);
  status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
  //status = 0;
  if(status != 0) {
    fprintf(stderr, "[invert_quda] Error, could not read gauge field");
    exit(11);
  }
  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value 1st field: %25.16e\n", plaq);


  no_fields=3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  items = VOL3 * 288;
  bytes = items * sizeof(float);
  if( (buffer = (float*)malloc( bytes ) ) == NULL ) {
    fprintf(stderr, "[] Error, could not allocate buffer\n");
    exit(20);
  }

  /****************************************
   * read read the spinor fields
   ****************************************/


  prop = create_fp_field(VOL3);
  create_fp(&prop2);
  create_fp(&prop_aux);
  create_fp(&prop_aux2);
  create_fp(&seq_prop);
  create_fp(&seq_prop2);

  /*******************************************************************
   * propagators
   *******************************************************************/
//  for(i=0; i<12;i++)
  for(i=11; i<12;i++)
  {
/*
    sprintf(file1, "source.%.4d.t00x00y00z00.%.2d.inverted", Nconf, i);
    if(g_cart_id==0) fprintf(stdout, "# Reading prop. from file %s\n", file1);
    fflush(stdout);
    if( read_lime_spinor(g_spinor_field[0], file1, 0) != 0 ) {
      fprintf(stderr, "Error, could not read file %s\n", file1);
      exit(9);
    }
*/
//    for(x0=0;x0<T;x0++)
    for(x0=1;x0<2;x0++)
    {
      //sprintf(file1, "p_k0.1562_1_1_1_1_tr10000.%.2d", x0+1);
      sprintf(file1, "p_k0.1562_1_1_1_1_tr10000.%.2d", x0+1);
      if( (ofs = fopen(file1, "r") ) == NULL ) {
        fprintf(stderr, "[] Error, could not open file %s for reading\n", file1);
        exit(19);
      }
      fprintf(stdout, "# [] reading data from file %s\n", file1);
      fflush(stdout);
      items = VOL3*288;
      if( fread(buffer, sizeof(float), items, ofs) != items ) {
        fprintf(stderr, "[] Error, could not read %u items from file %s\n", items, file1);
        exit(18);
      }
      fclose( ofs );
      byte_swap(buffer, items);

      //sprintf(filename, "prop_t%.2d", x0);
      //if( (ofs = fopen(filename, "w")) == NULL ) exit(21);
      //for(ix=0;ix<items;ix+=2) fprintf(ofs, "%10d%16.7e%16.7e\n", ix/2, buffer[ix], buffer[ix+1]);
      //fclose(ofs);
      //continue;

      for(is=0;is<4;is++) {
      for(id=0;id<3;id++) {
        for(ir=0;ir<4;ir++) {
        for(ic=0;ic<3;ic++) {
          for(x3=0;x3<LZ;x3++) {
          for(x2=0;x2<LY;x2++) {
          for(x1=0;x1<LX;x1++) {
            // fill the complete propagator point
            idx = ((3*is+id)*12 + 3*ir+ic)*VOL3 + (x3*LY + x2)*LX + x1;
            prop[g_ipt[0][x1][x2][x3]][3*is+id][2*(3*ir+ic)  ] = (double)(buffer[2*idx  ]);
            prop[g_ipt[0][x1][x2][x3]][3*is+id][2*(3*ir+ic)+1] = (double)(buffer[2*idx+1]);

            // pick out column no. i from the propagator
//            idx = ( i*12 + 3*ir+ic)*VOL3 + (x3*LY + x2)*LX + x1;
//            ix = g_ipt[x0][x1][x2][x3];
//            g_spinor_field[1][_GSI(ix)+2*(3*ir+ic)  ] = (double)(buffer[2*idx  ]);
//            g_spinor_field[1][_GSI(ix)+2*(3*ir+ic)+1] = (double)(buffer[2*idx+1]);
          }}}
        }}  // of ic, ir
      }}    // of id, is

      fprintf(stdout, "# [] printing propagator data for timeslice %d\n", x0);
      for(x1=0;x1<LX;x1++) {
      for(x2=0;x2<LY;x2++) {
      for(x3=0;x3<LZ;x3++) {
        sprintf(filename, "prop_x%.2dy%.2dz%.2d", x1, x2, x3);
        printf_fp(prop[g_ipt[0][x1][x2][x3]], filename, stdout);
      }}}


      // rotate full propagator to gamma-basis
      for(ix=0;ix<VOL3;ix++) {
        _fp_eq_zero( prop2 );
        _fp_eq_zero( prop_aux );
        _fp_eq_zero( prop_aux2);
        _fp_eq_gamma_rot2_ti_fp( prop_aux2, prop[ix], +1, prop_aux);
        _fp_eq_zero( prop_aux );
        _fp_eq_fp_ti_gamma_rot2( prop2, prop_aux2, +1, prop_aux);
        // normalize
        _fp_ti_eq_re(prop2, 2.*g_kappa );

        // sprintf(filename, "prop_rot_ix%.6d", ix);
        sprintf(filename, "prop_rot_x%.2dy%.2dz%.2d", ix/(LY*LZ), (ix%(LY*LZ))/LZ, ix%LZ);
        printf_fp(prop2, filename, stdout);
        //for(j=0;j<12;j++) {
        //  g_spinor_field[0][_GSI(x0*VOL3+ix)+2*j  ] = prop2[i][2*j  ];
        //  g_spinor_field[0][_GSI(x0*VOL3+ix)+2*j+1] = prop2[i][2*j+1];
        //}
      }

    }  // of loop on timeslices
/*
    if( (ofs = fopen("prop_full", "w")) == NULL ) exit(22);
    printf_spinor_field(g_spinor_field[1], ofs);
    fclose(ofs);

    
    // rotate the propagator column
    for(ix=0;ix<VOLUME;ix++) {
      _fv_eq_gamma_ti_fv(g_spinor_field[0]+_GSI(ix), 0, g_spinor_field[1]+_GSI(ix) );
      _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[1]+_GSI(ix) );
      _fv_pl_eq_fv(g_spinor_field[0]+_GSI(ix), spinor1);
      _fv_ti_eq_re(g_spinor_field[0]+_GSI(ix), _ONE_OVER_SQRT2);
    }
    fprintf(stdout, "# [] finished rotating propagator\n");
    fflush(stdout);

    Q_Wilson_phi(g_spinor_field[2], g_spinor_field[0]);
    fprintf(stdout, "# [] finished  application of Dirac operator\n");
    fflush(stdout);

    // rotate the source column
    for(ix=0;ix<VOLUME;ix++) {
      _fv_eq_gamma_ti_fv(g_spinor_field[1]+_GSI(ix), 0, g_spinor_field[2]+_GSI(ix) );
      _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[2]+_GSI(ix) );
      _fv_pl_eq_fv(g_spinor_field[1]+_GSI(ix), spinor1);
      _fv_ti_eq_re(g_spinor_field[1]+_GSI(ix), _ONE_OVER_SQRT2);
    }
    fprintf(stdout, "# [] finished rotating source\n");
    fflush(stdout);

    if( (ofs = fopen("source_full", "w")) == NULL ) exit(23);
    printf_spinor_field(g_spinor_field[1], ofs);
    fclose(ofs);

    spinor_scalar_product_re(&norm2, g_spinor_field[1], g_spinor_field[1], VOLUME);
    g_spinor_field[1][_GSI(g_source_location)+2*i  ] -= 1.;
    spinor_scalar_product_re(&norm, g_spinor_field[1], g_spinor_field[1], VOLUME);
    fprintf(stdout, "\n# [] absolut residuum squared: %e; relative residuum %e\n", norm, sqrt(norm/norm2) );
*/
/*    
    for(j=0;j<12;j++) {
      idx = g_ipt[1][2][3][4];
      prop[i][2*j  ] = g_spinor_field[0][_GSI(idx)+2*j];
      prop[i][2*j+1] = g_spinor_field[0][_GSI(idx)+2*j+1];
    }
*/
  }  // of loop on spin color indices
/*
  // initialize
  _fp_eq_zero( prop2 );
  _fp_eq_zero( prop_aux );
  _fp_eq_zero( prop_aux2);

  // rotate from the left
  _fp_eq_gamma_rot2_ti_fp( prop_aux2, prop, +1, prop_aux);
  // rotate from the right
  _fp_eq_zero( prop_aux );
  _fp_eq_fp_ti_gamma_rot2( prop2, prop_aux2, +1, prop_aux);

  printf_fp(prop, "prop_DR", stdout);
  printf_fp(prop2, "prop_UKQCD", stdout);
*/
/*
  fprintf(stdout, "# [] propagator for index = %d:\n", idx);
  for(i=0; i<12;i++) {
  for(j=0; j<12;j++) {
    fprintf(stdout, "\t%3d%3d%25.16e%25.16e%25.16e%25.16e\n", i, j, prop[i][2*j], prop[i][2*j+1], prop2[i][2*j], prop2[i][2*j+1]);
  }}
*/

  /*******************************************************************
   * sequential propagators
   *******************************************************************/
/*
//  for(i=0; i<12;i++)
  for(i=0; i<0;i++)
  {
    // construct the source
    sprintf(file1, "source.%.4d.t00x00y00z00.%.2d.inverted", Nconf, i);
    if(g_cart_id==0) fprintf(stdout, "# Reading prop. from file %s\n", file1);
    fflush(stdout);
    if( read_lime_spinor(g_spinor_field[2], file1, 0) != 0 ) {
      fprintf(stderr, "Error, could not read file %s\n", file1);
      exit(9);
    }
    for(ix=0;ix<VOLUME;ix++) {
      _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[2]+_GSI(ix));
      _fv_eq_fv(g_spinor_field[2]+_GSI(ix), spinor1);
    }

    sprintf(file1, "seq_source.%.4d.t00x00y00z00.%.2d.qx00qy00qz01.inverted", Nconf, i);
    if(g_cart_id==0) fprintf(stdout, "# Reading seq. prop. from file %s\n", file1);
    if( read_lime_spinor(g_spinor_field[0], file1, 0) != 0 ) {
      fprintf(stderr, "Error, could not read file %s\n", file1);
      exit(10);
    }

    for(j=0;j<12;j++) {
      seq_prop[i][2*j  ] = g_spinor_field[0][_GSI(idx)+2*j];
      seq_prop[i][2*j+1] = g_spinor_field[0][_GSI(idx)+2*j+1];
    }
  }

  // initialize
  _fp_eq_zero( seq_prop2 );
  _fp_eq_zero( prop_aux );
  _fp_eq_zero( prop_aux2);

  // rotate from the left
  _fp_eq_gamma_rot2_ti_fp( prop_aux2, seq_prop, +1, prop_aux);
  // rotate from the right
  _fp_eq_zero( prop_aux );
  _fp_eq_fp_ti_gamma_rot2( seq_prop2, prop_aux2, -1, prop_aux);

  printf_fp(prop, "seq_prop_DR", stdout);
  printf_fp(prop2, "seq_prop_UKQCD", stdout);
*/

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

