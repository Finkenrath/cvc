/****************************************************
 * bin2ascii.c
 *
 * Wed Nov 11 09:00:53 CET 2009
 *
 * PURPOSE:
 * - read in a contraction file in old ascii format
 *   and rewrite to a file of same name in binary format
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
  
  int c, i, mu, status;
  int count, ncon=-1;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int sid, gid;
  double *work = (double*)NULL;
  double *disc = (double*)NULL;
  int verbose = 0;
  char filename[200];
  char filename2[200];
  char text_buffer[200];
  FILE *ifs;


  while ((c = getopt(argc, argv, "h?vf:N:")) != -1) {
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
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "bin2ascii.input");
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
    exit(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  switch(ncon) {
    case -24:
      disc = (double*)calloc( 24*VOLUME, sizeof(double));
      if( disc == (double*)NULL ) { 
        fprintf(stderr, "could not allocate memory for disc\n");
        exit(3);
      }
      strcpy(filename, filename_prefix);
      check_error(read_lime_spinor(disc, filename, 0), "read_lime_spinor", NULL, 1);
      strcat(filename, ".ascii");
      if( (ifs = fopen(filename, "w")) == NULL ) {
        EXIT_WITH_MSG(2, "Error, could not open file for reading\n");
      }
      for(ix=0;ix<VOLUME;ix++) {
        for(mu=0;mu<24;mu+=2) {
          fprintf(ifs, "%8d%3d%25.16e%25.16e\n", ix, mu/2, disc[_GSI(ix)+mu], disc[_GSI(ix)+mu+1]);
      }}
      fclose(ifs); ifs=NULL;
      break;
    case -72:
      disc = (double*)calloc( 72*VOLUME, sizeof(double));
      if( disc == (double*)NULL ) { 
        fprintf(stderr, "could not allocate memory for disc\n");
        exit(3);
      }
      break;
    case -1:
      fprintf(stdout, "# No contraction type specified; exit\n");
      exit(100);
      break;
    default:
      fprintf(stdout, "# Using contraction type %d\n", ncon);
      disc = (double*)calloc( 2*ncon*VOLUME, sizeof(double));
      if( disc == (double*)NULL ) { 
        fprintf(stderr, "could not allocate memory for disc\n");
        exit(3);
      }
      // loop on gauge id's 
      for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
      for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
        fprintf(stdout, "# Starting gid %d\n", gid);
          sprintf(filename, "%s.%.4d.%.4d", filename_prefix, gid, sid);
    //      sprintf(filename, "%s.%.4d", filename_prefix, gid);
          fprintf(stdout, "# Reading binary from file %s\n", filename);
          if( read_lime_contraction(disc, filename, ncon, 0) == 106) {
            fprintf(stderr, "Error, could not read from file %s; continue\n", filename);
            continue;
          }
    
          sprintf(filename, "%s.%.4d.%.4d.ascii", filename_prefix, gid, sid);
    //      sprintf(filename, "%s.%.4d.ascii", filename_prefix, gid);
          fprintf(stdout, "# Writing ascii data to file %s\n", filename);
          if( (status=write_contraction(disc, NULL, filename, ncon, 2, 0)) != 0 ) {
            fprintf(stderr, "Error, could not write to file %s; exit\n", filename);
            fflush(stderr);
            exit(123);
          }
    
        fprintf(stdout, "# Finished gid %d\n", gid);
      }}
      break;
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(disc);

  return(0);

}
