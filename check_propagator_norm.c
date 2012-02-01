/*********************************************************************************
 * check_propagator_norm.c
 *
 * Tue Jan  5 23:17:43 CET 2010
 *
 * PURPOSE:
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

void usage() {
  fprintf(stdout, "Code to \n");
  fprintf(stdout, "Usage: <name>   [options]\n");
  fprintf(stdout, "Options:\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, ix, x0;
  int filename_set = 0;
  char filename[200];
  double sum, ts_sum, sp_sum;
  double m1=0., m2=0., m3=0., m4=0.;
  double plaq=0.;

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?v:f:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

#ifdef MPI
  MPI_Finalize();
#endif

   /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  T = T_global;

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();

  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

  /* alloc mem for spinor fields */
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

/*
  sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, g_sourceid);
  if(read_lime_spinor(g_spinor_field[0], filename, 0) != 0) exit(1);
*/
  sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, g_sourceid);
  if(read_cmi(g_spinor_field[1], filename) != 0) exit(1);


  Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);

  /* calculate the moments */
  sum = 0.;
  for(x0=0; x0<T; x0++) {
    ts_sum = 0;
    for(ix=0; ix<LX*LY*LZ; ix++) {
      sp_sum = 0.;
      for(i=0; i<24; i++) sp_sum += g_spinor_field[0][24*(x0*LX*LY*LZ+ix)+i];
      ts_sum += sp_sum;
    }
    sum += ts_sum;
  }
  m1 = sum / ( 12. * (double)VOLUME );
  fprintf(stdout, "m1 = %25.16e\n", m1);

  sum = 0.;
  for(x0=0; x0<T; x0++) {
    ts_sum = 0;
    for(ix=0; ix<LX*LY*LZ; ix++) {
      sp_sum = 0.;
      for(i=0; i<24; i++) sp_sum += g_spinor_field[0][24*(x0*LX*LY*LZ+ix)+i] * g_spinor_field[0][24*(x0*LX*LY*LZ+ix)+i] ;
      ts_sum += sp_sum;
    }
    sum += ts_sum;
  }
  m2 = sum / ( 12. * (double)VOLUME );
  fprintf(stdout, "m2 = %25.16e\n", m2);

/*
  sum = 0.;
  for(x0=0; x0<T; x0++) {
    ts_sum = 0;
    for(ix=0; ix<LX*LY*LZ; ix++) {
      sp_sum = 0.;
      for(i=0; i<24; i++) sp_sum += g_spinor_field[0][24*(x0*LX*LY*LZ+ix)+i] * g_spinor_field[0][24*(x0*LX*LY*LZ+ix)+i]* g_spinor_field[0][24*(x0*LX*LY*LZ+ix)+i];
      ts_sum += sp_sum;
    }
    sum += ts_sum;
  }
  m3 = sum / ( 12. * (double)VOLUME );
  fprintf(stdout, "m3 = %25.16e\n", m3);
*/

  return(0);

}
