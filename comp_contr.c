/****************************************************
 * comp_contr.c
 *
 * Mon Feb  1 09:52:25 CET 2010
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

void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c;
  int count, ncon=-1;
  int filename_set = 0;
  int ix;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
  double adiffre, adiffim, mdiffre, mdiffim, Mdiffre, Mdiffim, hre, him;
  int verbose = 0;
  char filename[200];
  char file1[200];
  char file2[200];


  while ((c = getopt(argc, argv, "h?vf:N:c:C:")) != -1) {
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

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  if(ncon<=0) {
    fprintf(stderr, "Error, incompatible contraction type specified; exit\n");
    exit(102);
  } else {
    fprintf(stdout, "# Using contraction type %d\n", ncon);
  }
  disc  = (double*)calloc(2*ncon*VOLUME, sizeof(double));
  if( disc  == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
    exit(103);
  }
  disc2 = (double*)calloc(2*ncon*VOLUME, sizeof(double));
  if( disc2 == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc2\n");
    exit(104);
  }

  /****************************************
   * read contractions
   ****************************************/
  if( read_lime_contraction(disc,  file1, 64, ncon, 0) != 0 ) {
    fprintf(stderr, "Error, could not read from file %s; exit\n", file1);
    exit(105);
  }
  if( read_lime_contraction(disc2, file2, 64, ncon, 0) != 0 ) {
    fprintf(stderr, "Error, could not read from file %s; exit\n", file2);
    exit(106);
  }

  /****************************************
   * calculate difference
   ****************************************/
  mdiffre = fabs(disc[0] - disc2[0]);
  mdiffim = fabs(disc[1] - disc2[1]);
  Mdiffre = 0.;
  Mdiffim = 0.;
  adiffre = 0.;
  adiffim = 0.;
  for(ix=0; ix<ncon*VOLUME; ix++) {
    adiffre += disc[2*ix  ] - disc2[2*ix  ];
    adiffim += disc[2*ix+1] - disc2[2*ix+1];
    hre = fabs(disc[2*ix  ] - disc2[2*ix  ]);
    him = fabs(disc[2*ix+1] - disc2[2*ix+1]);
    if(hre<mdiffre) mdiffre = hre;
    if(hre>Mdiffre) Mdiffre = hre;
    if(him<mdiffim) mdiffim = him;
    if(him>Mdiffim) Mdiffim = him;
  }
  adiffre /= (double)VOLUME * (double)ncon;
  adiffim /= (double)VOLUME * (double)ncon;

  fprintf(stdout, "# Results for files %s and %s:\n", file1, file2);
  fprintf(stdout, "average difference\t%25.16e\t%25.16e\n", adiffre, adiffim);
  fprintf(stdout, "minimal abs. difference\t%25.16e\t%25.16e\n", mdiffre, mdiffim);
  fprintf(stdout, "maximal abs. difference\t%25.16e\t%25.16e\n", Mdiffre, Mdiffim);

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(disc);
  free(disc2);

  return(0);

}
