/************************************************
 * gss_timeslice.c
 *
 * Sat Feb 13 19:29:39 CET 2010
 *
 * PURPOSE:
 * - generate stochastic sources
 * - source types:
 *   [ 0 point sources at source location and neighbours
 *     1 volume source
     ]
 *   2 spin diluted timeslice source
 * - noise type
 *   1 Gaussian noise
 *   2 Z2 noise
 * DONE:
 * TODO:
 * CHANGES:
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#undef MPI
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
#include "Q_phi.h"
#include "read_input_parser.h"
#include "ranlxd.h"
#include "smearing_techniques.h"
#include "fuzz.h"

void usage() {
  fprintf(stdout, "Code to prepare stochastic timeslice sources\n");
  fprintf(stdout, "Usage:   generate_stochastic_sources [options]\n");
  exit(0);
}

int main(int argc, char **argv) {

  int c, gid, sid;
  unsigned long int ix, iix;
  int i, x0;
  int filename_set   = 0;
  int N_ape=0, N_Jacobi=0, timeslice=0, Nlong=-1;
  int precision = 32;
  int *rng_state=NULL;
  int rng_readin=0;
  double alpha_ape=0., kappa_Jacobi = 0.;
  double ran[24];
  double *gauge_field_f = (double*)NULL;
  char filename[800];
  unsigned long int VOL3;
  FILE *ofs;
  DML_Checksum checksum;

  while ((c = getopt(argc, argv, "h?prf:i:a:n:l:K:t:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set = 1;
      break;
    case 'i':
      N_ape = atoi(optarg);
      break;
    case 'a':
      alpha_ape = atof(optarg);
      break;
    case 'n':
      N_Jacobi = atoi(optarg);
      break;
    case 'K':
      kappa_Jacobi = atof(optarg);
      break;
    case 'l':
      Nlong = atoi(optarg);
      break;
    case 't':
      timeslice = atoi(optarg);
      break;
    case 'p':
      precision = 64;
      break;
    case 'r':
      rng_readin = 1;
      break;
    case '?':
    default:
      usage();
      break;
    }
  }


  /***********************
   * read the input file *
   ***********************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * some checks on the input data *
   *********************************/
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  T      = T_global;
  Tstart = 0;
  VOL3   = LX*LY*LZ;

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    return(102);
  }

  geometry();

  /*************************
   * reset T to 1
   *************************/
  T      = 1;
  Tstart = timeslice;

  /*******************************************
   * check for source type, has to be 2
   *******************************************/
  if(g_source_type!=1) { /* timeslice sources */
    fprintf(stderr, "Warning, source type is %d, but will generate volume source\n", 
      g_source_type);
  }

  /* initialize random number generator */
  if(rng_readin==0) {
    fprintf(stdout, "# ranldxd: using seed %u and level 2\n", g_seed);
    rlxd_init(2, g_seed);
  } else {
    sprintf(filename, ".ranlxd_state");
    if( (ofs = fopen(filename, "r")) == (FILE*)NULL) {
      fprintf(stderr, "Error, could not read the random number generator state\n");
      return(105);
    }
    fprintf(stdout, "# reading rng state from file %s\n", filename);
    fscanf(ofs, "%d\n", &c);
    if( (rng_state = (int*)malloc(c*sizeof(int))) == (int*)NULL ) {
      fprintf(stderr, "Error, could not read the random number generator state\n");
      return(106);
    }
    rng_state[0] = c;
    fprintf(stdout, "# rng_state[%3d] = %3d\n", 0, rng_state[0]);
    for(i=1; i<c; i++) {
      fscanf(ofs, "%d", rng_state+i);
      fprintf(stdout, "# rng_state[%3d] = %3d\n", i, rng_state[i]);
    }
    fclose(ofs);
    rlxd_reset(rng_state);
    free(rng_state);
  }

  /* prepare the spinor field */
  no_fields=1;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) {alloc_spinor_field(&g_spinor_field[i], VOL3);}

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

    for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {

      fprintf(stdout, "# Generating volume sources for gid=%d and sid=%d\n", gid, sid);
      for(x0=0; x0<T_global; x0++) {

        for(ix=0; ix<VOL3; ix++) {
          switch(g_noise_type) {
            case 1:
              rangauss(ran, 24);
              break;
            case 2:
              ranz2(ran, 24);
              break;
          }
          _fv_eq_fv(g_spinor_field[0]+_GSI(ix), ran);
        }

        fprintf(stdout, "# finished generating source\n");

/*
        fprintf(stdout, "# source spinor field for timeslice no. %d\n", x0);
        for(ix=0; ix<VOL3; ix++) {
          for(c=0; c<12; c++) {
            fprintf(stdout, "%3d%6d%3d%25.16e%25.16e\n", x0, ix, c, 
              g_spinor_field[0][_GSI(ix)+2*c], g_spinor_field[0][_GSI(ix)+2*c+1]);
          }
        }
*/
      
        /******************************************************************
         * write the source
         ******************************************************************/
        //sprintf(filename, "%s.%.4d.%.2d", filename_prefix, gid, sid);
        sprintf(filename, "%s.%.4d.%.5d", filename_prefix, gid, sid);
        fprintf(stdout, "# writing source to file %s\n", filename);
        write_lime_spinor_timeslice(g_spinor_field[0], filename, precision, x0, &checksum);
      }
      fprintf(stdout, "#\t finished all for sid = %d\n", sid);
    }  /* loop on sid */

    fprintf(stdout, "# finished all for gid = %d\n", gid);

  }    /* loop on gid */
  

  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  c = rlxd_size();
  if( (rng_state = (int*)malloc(c*sizeof(int))) == (int*)NULL ) {
    fprintf(stderr, "Error, could not save the random number generator state\n");
    return(102);
  }
  rlxd_get(rng_state);
  sprintf(filename, ".ranlxd_state");
  if( (ofs = fopen(filename, "w")) == (FILE*)NULL) {
    fprintf(stderr, "Error, could not save the random number generator state\n");
    return(103);
  }
  fprintf(stdout, "# writing rng state to file %s\n", filename);
  for(i=0; i<c; i++) fprintf(ofs, "%d\n", rng_state[i]);
  fclose(ofs);
  free(rng_state);
 
  return(0);
}

