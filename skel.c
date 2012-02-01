/*********************************************************************************
 * skel.c
 *
 * Tue Jan  5 23:17:43 CET 2010
 *
 * PURPOSE:
 * - program skeleton
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
  
  int c;


#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?v:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
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

  return(0);

}
