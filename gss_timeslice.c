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

  int c, gid, sid, count;
  unsigned long int ix, iix;
  int i, x0;
  /* int k; */
  int filename_set   = 0;
  int N_ape=0, N_Jacobi=0, timeslice=-1, Nlong=-1;
  int timeslice_tab[96];
  int precision = 32;
  int *rng_state=NULL;
  int rng_readin=0;
  double alpha_ape=0., kappa_Jacobi = 0.;
  double ran[6];
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

  /**************************************
   * reset T to 1, check for timeslice
   **************************************/
  if(timeslice==-1) {
    fprintf(stdout, "# Using %d for srand()\n", g_seed);
    srand(g_seed); 
  }
  T      = 1;

  /*******************************************
   * check for source type, has to be 2
   *******************************************/
  if(g_source_type!=2) { /* timeslice sources */
    fprintf(stderr, "Warning, source type is %d, but will generate spin diluted timeslice source\n", 
      g_source_type);
  }

  /* initialize random number generator */
  if(rng_readin==0) {
    fprintf(stdout, "# ranldxd: using seed %u and level 2\n", g_seed);
    rlxd_init(2, g_seed);  
/**************************************************/
/*      srand(g_seed);                            */
/**************************************************/
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

  /* prepare the spinor fields */
  no_fields=5;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) {alloc_spinor_field(&g_spinor_field[i], VOL3);}

  /* prepare the gauge field, if necessary */
  if(Nlong>0 || N_Jacobi>0) {
    fprintf(stdout, "# preparing gauge field\n");
    alloc_gauge_field(&g_gauge_field, VOL3);
  }

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

    /* for timeslice book keeping */
    for(i=0; i<T_global; i++) {
      timeslice_tab[i] = i;
    }

    count = -1;
    for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {

      count++;
      x0 = IRand( 0, T_global-1-count );
      timeslice = timeslice_tab[x0];
      Tstart = timeslice;
      if(x0==T_global-1-count) {
        timeslice_tab[x0] = -1;
      } else {
        timeslice_tab[x0] = timeslice_tab[T_global-1-count];
        timeslice_tab[T_global-1-count] = -1;
      }
      if(count==T_global-1) {
        for(i=0; i<T_global; i++) timeslice_tab[i] = i;
        count = -1;
      }

      fprintf(stdout, "# Generating spin diluted timeslice sources "\
                      "on timeslice %d for gid=%d and sid=%d\n", timeslice, gid, sid);

      if(Nlong>0 || N_Jacobi>0) {
        sprintf(filename, "%s.%.4d", gaugefilename_prefix, gid);
        fprintf(stdout, "# reading gauge field timeslice from file %s\n", filename);
        read_lime_gauge_field_doubleprec_timeslice(g_gauge_field, filename, timeslice, &checksum);

/*
        for(ix=0; ix<VOL3; ix++) {
          for(x0=0; x0<4; x0++) {
            for(i=0; i<9; i++) {
              fprintf(stdout, "%6d%3d%3d%25.16e%25.16e\n", ix, x0, i, 
                g_gauge_field[_GGI(ix,x0)+2*i], g_gauge_field[_GGI(ix,x0)+2*i+1]);
            }
          }
        }
*/

        for(i=0; i<N_ape; i++) {
          APE_Smearing_Step_Timeslice(g_gauge_field, alpha_ape);
        }
        if(Nlong>0) {
          alloc_gauge_field(&gauge_field_f, VOL3);
          fuzzed_links_Timeslice(gauge_field_f, g_gauge_field, Nlong, 0);
          memcpy((void*)g_gauge_field, (void*)gauge_field_f, 72*VOL3*sizeof(double));
          free(gauge_field_f);
        }
      }


      /* set field 4 to zero, serves as "all other timeslices than timeslice" */
      for(ix=0; ix<VOL3; ix++) {
        _fv_eq_zero(g_spinor_field[4]+_GSI(ix));
      }

      /* initialize spinor fields 0 to 3 with 0 */
      for(i=0; i<4; i++) {
        for(ix=0; ix<VOL3; ix++) {
          _fv_eq_zero(g_spinor_field[i]+_GSI(ix));
        }
      }

      for(ix=0; ix<VOL3; ix++) {

        switch(g_noise_type) {
          case 1:
            rangauss(ran, 6);
            break;
          case 2:
            ranz2(ran, 6);
            break;
        }

/**************************************************/
/*        for(k=0; k<6; k++) {                    */
/*          ran[k] = Random_Z2();                 */
/*        }                                       */
/**************************************************/
        iix = _GSI(ix);
        g_spinor_field[0][iix + 0] = ran[0];
        g_spinor_field[1][iix + 6] = ran[0];
        g_spinor_field[2][iix +12] = ran[0];
        g_spinor_field[3][iix +18] = ran[0];

        g_spinor_field[0][iix + 1] = ran[1];
        g_spinor_field[1][iix + 7] = ran[1];
        g_spinor_field[2][iix +13] = ran[1];
        g_spinor_field[3][iix +19] = ran[1];

        g_spinor_field[0][iix + 2] = ran[2];
        g_spinor_field[1][iix + 8] = ran[2];
        g_spinor_field[2][iix +14] = ran[2];
        g_spinor_field[3][iix +20] = ran[2];

        g_spinor_field[0][iix + 3] = ran[3];
        g_spinor_field[1][iix + 9] = ran[3];
        g_spinor_field[2][iix +15] = ran[3];
        g_spinor_field[3][iix +21] = ran[3];

        g_spinor_field[0][iix + 4] = ran[4];
        g_spinor_field[1][iix +10] = ran[4];
        g_spinor_field[2][iix +16] = ran[4];
        g_spinor_field[3][iix +22] = ran[4];

        g_spinor_field[0][iix + 5] = ran[5];
        g_spinor_field[1][iix +11] = ran[5];
        g_spinor_field[2][iix +17] = ran[5];
        g_spinor_field[3][iix +23] = ran[5];
      }

      fprintf(stdout, "# finished generating source\n");

/*
      for(i=0; i<4; i++) {
        fprintf(stdout, "# source spinor field no. %d\n", i);
        for(ix=0; ix<VOL3; ix++) {
          for(c=0; c<12; c++) {
            fprintf(stdout, "%6d%3d%25.16e%25.16e\n", ix, c, g_spinor_field[i][_GSI(ix)+2*c], g_spinor_field[i][_GSI(ix)+2*c+1]);
          }
        }
      }
*/

      /******************************************************************
       * write the sources
       ******************************************************************/
      for(i=0; i<4; i++) {
        sprintf(filename, "%s.%.4d.%.2d.%.2d", filename_prefix, gid, timeslice, i);
        fprintf(stdout, "# writing source to file %s\n", filename);
        for(x0=0; x0<T_global; x0++) {
          if(x0==timeslice) {
            write_lime_spinor_timeslice(g_spinor_field[i], filename, precision, x0, &checksum);
          } else {
            write_lime_spinor_timeslice(g_spinor_field[4], filename, precision, x0, &checksum);
          }
        }

        fprintf(stdout, "# generating fuzzed [Nlong=%d] / Jacobi-smeared [N_Jacobi=%d,kappa_Jacobi=%f] source for number %d\n", Nlong, N_Jacobi, kappa_Jacobi, i);
        if(Nlong>0) {
          Fuzz_prop(g_gauge_field, g_spinor_field[i], Nlong);
        } else if(N_Jacobi>0) {
          Jacobi_Smearing_Steps(g_gauge_field, g_spinor_field[i], N_Jacobi, kappa_Jacobi, 0);
        }
    
        sprintf(filename, "%s.%.4d.%.2d.%.2d", filename_prefix, gid, timeslice, (4+i));
        fprintf(stdout, "# writing fuzzed/smeared source to file %s\n", filename);
        for(x0=0; x0<T_global; x0++) {
          if(x0==timeslice) {
            write_lime_spinor_timeslice(g_spinor_field[i], filename, precision, x0, &checksum);
          } else {
            write_lime_spinor_timeslice(g_spinor_field[4], filename, precision, x0, &checksum);
          }
        }
      }
      fprintf(stdout, "#\t finished all for sid = %d\n", sid);
    }  /* loop on sid */

    fprintf(stdout, "# finished all for gid = %d\n", gid);

  }    /* loop on gid */
  

  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  if(Nlong>-1) free(g_gauge_field);

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

