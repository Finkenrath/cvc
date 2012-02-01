/****************************************************
 * fft_3pt_block.c
 *
 * Di 22. Nov 06:14:04 PST 2011
 *
 * PURPOSE:
 * TODO:
 * - NOTE ON FFTW EXPONENT SIGN: FFTW_FORWARD  = -1
 * -                             FFTW_BACKWARD = +1
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#ifdef OPENMP
#include <omp.h>
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
#include "gauge_io.h"
#include "Q_phi.h"
#include "fuzz.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "make_H3orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for connected contributions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose [no effect, lots of stdout output it]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, j;
  int filename_set = 0;
  int l_LZ_at, l_LZstart_at;
  unsigned int VOL3;
  int dims[3];
  size_t bytes, items, shift;
  double *block_data_x=NULL, *block_data_q=NULL;
  int verbose = 0, status;
  int sx0, sx1, sx2, sx3;
  int write_ascii=0;
  int num_threads=1;
  int position, it;
  char filename[200], type_string[200];
  double ratime, retime;
  double scs[18];
  double q[3], phase;
  FILE *ofs;

  fftw_complex *ft_in=NULL;
#ifdef MPI
   fftwnd_mpi_plan plan_p, plan_m;
#else
   fftwnd_plan plan_p, plan_m;
#endif 

#ifdef MPI
  MPI_Status status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ah?vf:t:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [] will write in ascii format\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

#ifdef OPENMP
  omp_set_num_threads(num_threads);
#endif

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# reading input from file %s\n", filename);
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

#ifdef OPENMP
  status = fftw_threads_init();
  if(status != 0) {
    fprintf(stderr, "\n[] Error from fftw_init_threads; status was %d\n", status);
    exit(120);
  }
#endif

  dims[0]=LZ; dims[1]=LY; dims[2]=LX;
#ifndef MPI
  plan_p = fftwnd_create_plan(3, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  l_LZ_at      = LX;
  l_LZstart_at = 0;
  FFTW_LOC_VOLUME = LX*LY*LZ;
  VOL3 = LX*LY*LZ;
#else
#endif
  fprintf(stdout, "# [%2d] parameters:\n"\
		  "# [%2d] l_LZ_at      = %3d\n"\
		  "# [%2d] l_LZstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, l_LZ_at,
		  g_cart_id, l_LZstart_at, g_cart_id, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  // determine the source location
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  fprintf(stdout, "# [] source location %d = (%d,%d,%d,%d)\n", g_source_location, sx0, sx1, sx2, sx3);


  // initialize permutation tables
  init_perm_tabs();


  // allocate memory for the spinor fields
  items = VOLUME;
  bytes = sizeof(double);
  block_data_x = (double*)malloc(items*bytes);
  if(block_data_x == NULL) {
    fprintf(stderr, "[] Error, could not allocate block_data_x\n");
    exit(16);
  }
 
  block_data_q = (double*)malloc(items*bytes);
  if(block_data_q == NULL) {
    fprintf(stderr, "[] Error, could not allocate block_data_q\n");
    exit(17);
  }
  
  items = VOL3;
  bytes = sizeof(fftw_complex);
  ft_in  = (fftw_complex*)malloc(items*bytes);
  if(ft_in == NULL) {
    fprintf(stderr, "[] Error, could not allocate ft_in\n");
    exit(18);
  }
  

  /***********************************************
   * read and FT
   ***********************************************/
  status = 0;
  position = 0;
  while(!status) {
    sprintf(filename, "%s.%.4d", filename_prefix, Nconf);
    status = read_lime_contraction(block_data_x, filename, 1, position);
    if(status != 0) {
      fprintf(stderr, "[] Error, could not read lime contraction from file %s; status was %d\n", filename, status);
      continue;
    }

    shift = 0;
    items = 2*VOL3;
    bytes = sizeof(double);
    for(it=0;it<T_global;it++) {
      memcpy(ft_in, block_data_x+shift, items*bytes);
#ifndef MPI
#  ifdef OPENMP
      fftwnd_threads_one(num_threads, plan_p, ft_in, NULL);
#else
      fftwnd_one(plan_p, ft_in, NULL);
#  endif
#else
#endif
      memcpy(block_data_q+shift, ft_in, items*bytes);
      shift += items;
    }

    
    sprintf(filename, "%s.%.4d.%.2d", filename_prefix2, Nconf, position);
    sprintf(type_string, "3-point function, t,qvec dependent, gamma insertion no. %d", position);
    status = write_lime_contraction(block_data_q, filename, 64, 1, type_string, Nconf, position);
    if(status != 0) {
      fprintf(stderr, "[] Error, could not read lime contraction from file %s; status was %d\n", filename, status);
      exit(19);
    }

    if(write_ascii) {
      strcat(filename, ".ascii");
      ofs = fopen(filename, "w");
      if(ofs == NULL) {
        fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
        exit(20);
      }
    }

    position++;
  }

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();
  if(block_data_x!= NULL) free(block_data_x);
  if(block_data_q!= NULL) free(block_data_q);

  free(ft_in);
  fftwnd_destroy_plan(plan_p);

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
