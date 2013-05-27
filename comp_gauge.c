/****************************************************
 * comp_gauge.c
 *
 * Fr 24. Mai 12:57:08 EEST 2013
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
#include "dml.h"
#include "invert_Qtm.h"

void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu;
  int i, j, k, ncon=-1;
  int filename_set = 0;
  int x0, x1, x2, x3, ix;
  double adiffre, adiffim, mdiffre, mdiffim, Mdiffre, Mdiffim, hre, him;
  double *gauge1=NULL, *gauge2=NULL;
  double plaq1, plaq2; 
  char filename[200];
  char file1[200];
  char file2[200];
  char file3[200];
  char file4[200];
  DML_Checksum checksum;
  FILE *ofs = NULL;
  double norm, norm2, norm3;
  int status;

  while ((c = getopt(argc, argv, "h?vf:c:C:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      strcpy(file1, optarg);
      break;
    case 'C':
      strcpy(file2, optarg);
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

  /* initialize */
  T      = T_global;
  Tstart = 0;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n",\
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(101);
  }

  geometry();
  init_geometry_5d();
  geometry_5d();

  alloc_gauge_field(&gauge1, VOLUMEPLUSRAND);
  alloc_gauge_field(&gauge2, VOLUMEPLUSRAND);
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  sprintf(filename, "%s.%.4d", filename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  memcpy(gauge1, g_gauge_field, 72*VOLUME*sizeof(double));

  sprintf(filename, "%s.%.4d", filename_prefix2, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  memcpy(gauge2, g_gauge_field, 72*VOLUME*sizeof(double));

  free(g_gauge_field);

  // measure the plaquette
  plaquette2(&plaq1, gauge1);
  plaquette2(&plaq2, gauge2);
      ;
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value 1: %25.16e\n", plaq1);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value 2: %25.16e\n", plaq2);

  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(stdout, "# x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", x0, x1, x2, x3);
    for(i=0; i<4; i++) {
      ix = _GGI( g_ipt[x0][x1][x2][x3], i );
      fprintf(stdout, "#\t direction i=%3d\n", i);
      for(j=0; j<9; j++) {
        fprintf(stdout, "%3d%25.16e%25.16e%25.16e%25.16e\n", j,
            gauge1[ix+2*j], gauge1[ix+2*j+1],
            gauge2[ix+2*j], gauge2[ix+2*j+1]);
    }}
  }}}}

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(gauge1);
  free(gauge2);

  g_the_time = time(NULL);
  fprintf(stdout, "# [comp_gauge] %s# [comp_gauge] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [comp_gauge] %s# [comp_gauge] end fo run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);

}
