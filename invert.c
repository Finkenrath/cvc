/****************************************************
 * invert.c
 *
 * Wed Dec  9 21:06:00 CET 2009
 *
 * TODO:
 * - test in MPI mode 
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

void usage() {
  fprintf(stdout, "Code to invert D_tm\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  printf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu;
  int count        = 0;
  int filename_set = 0;
  int status;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int sl0, sl1, sl2, sl3, have_source_flag=0;
  double fnorm;
  int do_gt   = 0;
  char filename[200];
  double ratime, retime;
  double plaq=0., norm, norm2;
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1, *cp1, *cp2, *cp3;
  FILE *ofs;

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
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_proc_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
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

  T            = T_global / g_nproc;
  Tstart       = g_cart_id*T;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n",\
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart);

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

  // read the gauge field
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  switch(g_gauge_file_format) {
    case 0:
      sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
      status = read_lime_gauge_field_doubleprec(filename);
      break;
    case 1:
      sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "\n# [] reading gauge field from file %s\n", filename);
      status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
      break;
    }
    if(status != 0) {
      fprintf(stderr, "[] Error, could not read gauge field\n");
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 21);
      MPI_Finalize();
#endif
      exit(21);
    }
#ifdef MPI
  xchange_gauge();
#endif

  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# Read plaquette value: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# Measured plaquette value: %25.16e\n", plaq);

  // allocate memory for the spinor fields
  no_fields = 9;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);


  // the source locaton
  sl0 = g_source_location/(LX*LY*LZ);
  sl1 = ( g_source_location%(LX*LY*LZ) ) / (LY*LZ);
  sl2 = ( g_source_location%(LY*LZ) ) / (LZ);
  sl3 = g_source_location%LZ;
  if(g_cart_id==0) fprintf(stdout, "# global sl = (%d, %d, %d, %d)\n", sl0, sl1, sl2, sl3);
  have_source_flag = sl0-Tstart>=0 && sl0-Tstart<T;
  sl0 -= Tstart;
  fprintf(stdout, "# [%d] have source: %d\n", g_cart_id, have_source_flag);
  if(have_source_flag==1) fprintf(stdout, "# local sl = (%d, %d, %d, %d)\n", sl0, sl1, sl2, sl3);
  
  //sprintf(filename, "%s.%.4d.out", gaugefilename_prefix, Nconf);
  //if(g_cart_id==0) fprintf(stdout, "\n\n# Writing gauge field %d to file %s\n", Nconf, filename);
  //write_lime_gauge_field(filename, plaq, Nconf, 64);

  //if(g_cart_id==0) fprintf(stdout, "\n\n# Reading gauge field %d from file %s\n", Nconf, filename);
  //read_lime_gauge_field_doubleprec(filename);
  //plaquette(&plaq);
  //if(g_cart_id==0) fprintf(stdout, "# Measured plaquette value after write/reread: %25.16e\n", plaq);

  /***********************************************
   * (1) check the Dirac operator against HMC
   ***********************************************/
  for(i=0; i<1; i++) {
/*
    sprintf(filename, "source.%.4d.%.2d.inverted", Nconf, i);
    if(g_cart_id==0) fprintf(stdout, "\n\n# Source number %d from file %s\n", i, filename);
    if( read_lime_spinor(g_spinor_field[0], filename, 0) != 0) {  
      fprintf(stderr, "ERROR, could not read file %s\n", filename);
      break;
    }
    xchange_field(g_spinor_field[0]);

    sprintf(filename, "source.%.4d.%.2d.out", Nconf, i);
    if(g_cart_id==0) fprintf(stdout, "# [%d] Writing to file %s\n", g_cart_id, filename);
    write_propagator(g_spinor_field[0], filename, 1, 64);

    if(g_cart_id==0) fprintf(stdout, "# [%d] Reading from file %s\n", g_cart_id, filename);
    read_lime_spinor(g_spinor_field[1], filename, 0);


    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_mi_fv(g_spinor_field[2]+_GSI(ix), g_spinor_field[1]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
    }
    spinor_scalar_product_re(&norm, g_spinor_field[2], g_spinor_field[2], VOLUME);
    if(g_cart_id==0) fprintf(stdout, "# Norm of difference  of orig. reading and rereading: %e\n", norm);
*/
#ifdef _UNDEF
    // point source
    for(ix=0; ix<VOLUME; ix++) { _fv_eq_zero(g_spinor_field[0]+_GSI(ix)); }
    if(have_source_flag) {
      ix = g_ipt[sl0][sl1][sl2][sl3];
      g_spinor_field[0][_GSI( ix )+2*i  ] = 1.;
      // multiply the soiurce with g2
      if(g_cart_id==0) fprintf(stdout, "# [] rotate from DeGrand-Rossi to local basis\n");
      _fv_eq_fv(spinor1, g_spinor_field[0]+_GSI(ix));
      _fv_eq_gamma_ti_fv(g_spinor_field[0]+_GSI(ix), 2, spinor1);
    }
    xchange_field(g_spinor_field[0]);
 

    for(ix=0; ix<VOLUME; ix++) { _fv_eq_zero(g_spinor_field[1]+_GSI(ix)); }
    g_spinor_field[1][_GSI(g_ipt[sl0][sl1][sl2][sl3])+2*i  ] = 1.;
    xchange_field(g_spinor_field[1]);


    invert_Q_Wilson(g_spinor_field[1], g_spinor_field[0], 2);
    xchange_field(g_spinor_field[1]);

//    sprintf(filename, "source.%.4d.%.2d.inverted", Nconf, i);
//    if(g_cart_id==0) fprintf(stdout, "# Prop. number %d from file %s\n", i, filename);
//    if( read_lime_spinor(g_spinor_field[2], filename, 0) != 0) {  
//      fprintf(stderr, "ERROR, could not read file %s\n", filename);
//      break;
//    }

    Q_Wilson_phi(g_spinor_field[2], g_spinor_field[1]);
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_mi_fv(g_spinor_field[3]+_GSI(ix), g_spinor_field[2]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
    }
    spinor_scalar_product_re(&norm2, g_spinor_field[0], g_spinor_field[0], VOLUME);
    spinor_scalar_product_re(&norm, g_spinor_field[3], g_spinor_field[3], VOLUME);
    if(g_cart_id==0) fprintf(stdout, "absolut residuum squared = %e, relative residuum = %e\n", norm, sqrt(norm / norm2));
    //norm=fabs(g_spinor_field[3][0]);
    //for(ix=1; ix<24*VOLUME; ix++) {
    //  norm2 = fabs(g_spinor_field[3][ix]);
    //  if(norm2>norm) norm=norm2;
    //}
    //fprintf(stdout, "# [%d] max of abs difference HMC-prop. = %e\n", g_cart_id, norm);


    if(g_cart_id==0) fprintf(stdout, "# [] rotate from local to DeGrand-Rossi basis\n");
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv(spinor1, g_spinor_field[1]+_GSI(ix));
      _fv_eq_gamma_ti_fv(g_spinor_field[1]+_GSI(ix), 2, spinor1);
    }
#endif
    sprintf(filename, "source_DR.%.4d.%.2d.inverted", Nconf, i);
    //status = write_propagator(g_spinor_field[1], filename, 0, g_propagator_precision);
    status = read_lime_spinor(g_spinor_field[1], filename, 0);
    if(status != 0) {
      fprintf(stderr, "Error from write_propagator, status was %d\n", status);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 22);
      MPI_Finalize();
#endif
      exit(22);
    }
#ifndef MPI
/*
    sprintf(filename, "source_DR_tzyx.%.4d.%.2d.inverted.ascii", Nconf, i);
    if( (ofs = fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
    } else {
      printf_spinor_field_tzyx(g_spinor_field[1], ofs);
      fclose(ofs);
    }
*/
/*
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      iix = ((x0 * LZ + x3 ) * LY + x2 ) * LX + x1;
      _fv_eq_fv( g_spinor_field[3]+ _GSI(iix), g_spinor_field[1]+_GSI(ix) );
    }}}}
    sprintf(filename, "source_DR_tzyx.%.4d.%.2d.bin", Nconf, i);
    if( (ofs = fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
    } else {
      status = fwrite(g_spinor_field[3], sizeof(double), 24*VOLUME, ofs);
      if(status != 24*VOLUME) {
        fprintf(stderr, "[] Error, could not write proper amount of items\n");
      }
      fclose(ofs);
    }
*/
    for(ix=0;ix<VOLUME;ix++) {
      iix = g_lexic2eot[ix];
      _fv_eq_fv(g_spinor_field[3]+_GSI(iix), g_spinor_field[1]+_GSI(ix));
    }
    sprintf(filename, "source_DR_tzyx.%.4d.%.2d.eo.bin", Nconf, i);
    if( (ofs = fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
    } else {
      status = fwrite(g_spinor_field[3], sizeof(double), 24*VOLUME, ofs);
      if(status != 24*VOLUME) {
        fprintf(stderr, "[] Error, could not write proper amount of items\n");
      }
      fclose(ofs);
    }

#endif
/*
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<12; mu++) {
        fprintf(stdout, "ix=%d; mu=%d; sp= %25.16e +i %25.16e\n", ix, mu, 
          g_spinor_field[3][_GSI(ix)+2*mu], g_spinor_field[3][_GSI(ix)+2*mu+1]);
      }
    }
*/

/* 
    Q_phi_tbc(g_spinor_field[4], g_spinor_field[1]);
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_mi_fv(g_spinor_field[3]+_GSI(ix), g_spinor_field[0]+_GSI(ix), g_spinor_field[4]+_GSI(ix));
    }
    spinor_scalar_product_re(&norm, g_spinor_field[3], g_spinor_field[3], VOLUME);
    if(g_cart_id==0) fprintf(stdout, "norm of difference D sol - source from orig. source = %e\n", sqrt(norm));
    
*/

  }
 

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [invert] %s# [invert_quda] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [invert] %s# [invert_quda] end of run\n", ctime(&g_the_time));
  }

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
