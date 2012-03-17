/****************************************************
 * 
 * gauge_ildg_single2double.c
 *
 * Tue Feb 28 10:09:09 EET 2012
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

  while ((c = getopt(argc, argv, "h?vf:g:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'g':
      strcpy(file1, optarg);
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

  if(init_geometry_5d() != 0) {
    fprintf(stderr, "ERROR from init_geometry_5d\n");
    exit(102);
  }
  geometry_5d();

  VOL3 = LX*LY*LZ;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(g_cart_id==0) fprintf(stdout, "# gauge field file name %s\n", file1);

    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    status = read_lime_gauge_field_doubleprec(file1);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
    // status = 0;
  if(status != 0) {
    fprintf(stderr, "[apply_Dtm] Error, could not read gauge field\n");
    EXIT(11);
  }
#ifdef MPI
  xchange_gauge();
#endif

  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value 1st field: %25.16e\n", plaq);


  sprintf(filename, "%s.dbl", file1);
  if(g_cart_id==0) fprintf(stdout, "# [] writing gauge field in double precision to file %s\n", filename);
  status = write_lime_gauge_field(filename, plaq, Nconf, 64);
  if(status != 0) {
    fprintf(stderr, "[apply_Dtm] Error, could not write gauge field\n");
    EXIT(12);
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  free_geometry();

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

