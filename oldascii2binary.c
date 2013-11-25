/****************************************************
 * oldascii2binary.c
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
  int count, contype=-1;
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


  while ((c = getopt(argc, argv, "h?vf:t:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 't':
      contype = atoi(optarg);
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
    exit(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  work = (double*)calloc( 32*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
    exit(3);
  }

  disc = (double*)calloc( 32*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
    exit(3);
  }

  if(contype==-1) {
    fprintf(stdout, "# No contraction type specified; exit\n");
    exit(100);
  } else {
    fprintf(stdout, "# Using contraction type %d\n", contype);
  }

  /****************************************
   * loop on gauge id's 
   ****************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    fprintf(stdout, "# Starting gid %d\n", gid);
    /****************************************
     * loop on source id's 
     ****************************************/
    for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
      fprintf(stdout, "#\t Starting sid %d\n", sid);

      sprintf(filename, "%s.%.4d.%.2d", filename_prefix, gid, sid);
      if( (ifs = fopen(filename, "r")) == (FILE*)NULL ) {
        fprintf(stderr, "Error, could not open file %s, try next one\n", filename);
        continue;
      }
      fprintf(stdout, "# Reading from file %s\n", filename);

      fgets(text_buffer, 100, ifs);
      count=0;
      for(ix=0; ix<VOLUME; ix++) {
        fgets(text_buffer, 100, ifs);
        for(mu=0; mu<contype; mu++) {
          fscanf(ifs, "%s", text_buffer);
          fscanf(ifs, "%s", text_buffer);
          work[count  ] = atof(text_buffer);
          fgets(text_buffer, 100, ifs);
          work[count+1] = atof(text_buffer);
          count+=2;
        }
      }
      fclose(ifs);

/*
      read_contraction(work, NULL, filename, contype);
*/

      count=0;
      for(ix=0; ix<VOLUME; ix++) {
        for(mu=0; mu<contype; mu++) {
          disc[_GWI(mu,ix,VOLUME)  ] = work[count  ];
          disc[_GWI(mu,ix,VOLUME)+1] = work[count+1];
          count+=2;
        }
      }

/*
      count=0;
      for(ix=0; ix<VOLUME; ix++) {
        for(mu=0; mu<contype; mu++) {
          fprintf(stdout, "%8d%3d%25.16e%25.16e\n", ix, mu, disc[_GWI(mu,ix,VOLUME)], disc[_GWI(mu,ix,VOLUME)+1]);
        }
      }
*/

      Nconf = gid;
      sprintf(filename2, "%s.%.4d.%.2d.tmp", filename_prefix, gid, sid);
      fprintf(stdout, "# Saving in file %s\n", filename2);
      status = write_contraction(disc, NULL, filename2, contype, 0, 0);


      if(status==0) {
        rename(filename2, filename);
      }

      fprintf(stdout, "# \tFinished sid %d\n", sid);
    }
    fprintf(stdout, "# Finished gid %d\n", gid);
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(work);

  return(0);

}
