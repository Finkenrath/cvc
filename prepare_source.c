/************************************************
 * prepare_timeslice_source.c
 *
 * Tue Nov  8 16:00:41 EET 2011
 *
 * PURPOSE:
 * - generate stochastic timeslice source
 * - based on gss_timeslice.c
 * DONE:
 * TODO:
 * CHANGES:
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

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

int prepare_timeslice_source(double *s, double *gauge_field, int timeslice, unsigned int V, char*rng_file_in, char*rng_file_out) {

  int c;
  unsigned int ix, iix;
  int i;
  int *rng_state=NULL;
  double ran[24];
  unsigned int VOL3 = LX*LY*LZ;
  FILE *ofs=NULL;
  char filename[200];

  if(rng_file_in==NULL) {
    // initialize random number generator
    fprintf(stdout, "# [] ranldxd: using seed %u and level 2\n", g_seed);
    rlxd_init(2, g_seed);
//    rng_file_in = (char*)malloc(14*sizeof(char));
//    strcpy(rng_file_in, "ranlxd_state");
//    fprintf(stdout, "# [] reset rng filename to %s\n", rng_file_in);
  } else {
    // read from file
    strcpy(filename, rng_file_in);
    if( (ofs = fopen(filename, "r")) == (FILE*)NULL) {
      fprintf(stderr, "[] Error, could not read the random number generator state\n");
      return(105);
    }
    fprintf(stdout, "# [] reading rng state from file %s\n", filename);
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

  /* set field 4 to zero, serves as "all other timeslices than timeslice" */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(s+_GSI(ix));
  }

  for(ix=0; ix<VOL3; ix++) {

    switch(g_noise_type) {
      case 1:
        rangauss(ran, 24);
        break;
      case 2:
        ranz2(ran, 24);
        break;
    }

    iix = _GSI(timeslice * VOL3 + ix);
    s[iix + 0] = ran[0];
    s[iix + 1] = ran[1];
    s[iix + 2] = ran[2];
    s[iix + 3] = ran[3];
    s[iix + 4] = ran[4];
    s[iix + 5] = ran[5];
    s[iix + 6] = ran[6];
    s[iix + 7] = ran[7];
    s[iix + 8] = ran[8];
    s[iix + 9] = ran[9];
    s[iix + 10] = ran[10];
    s[iix + 11] = ran[11];
    s[iix + 12] = ran[12];
    s[iix + 13] = ran[13];
    s[iix + 14] = ran[14];
    s[iix + 15] = ran[15];
    s[iix + 16] = ran[16];
    s[iix + 17] = ran[17];
    s[iix + 18] = ran[18];
    s[iix + 19] = ran[19];
    s[iix + 20] = ran[20];
    s[iix + 21] = ran[21];
    s[iix + 22] = ran[22];
    s[iix + 23] = ran[23];

  }  // of ix

  if(rng_file_out != NULL) {
    c = rlxd_size();
    if( (rng_state = (int*)malloc(c*sizeof(int))) == (int*)NULL ) {
      fprintf(stderr, "Error, could not save the random number generator state\n");
      return(102);
    }
    rlxd_get(rng_state);
    strcpy(filename, rng_file_out);
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL) {
      fprintf(stderr, "Error, could not save the random number generator state\n");
      return(103);
    }
    fprintf(stdout, "# writing rng state to file %s\n", filename);
    for(i=0; i<c; i++) fprintf(ofs, "%d\n", rng_state[i]);
    fclose(ofs);
    free(rng_state);
  }
  return(0);
}

int prepare_coherent_timeslice_source(double *s, double *gauge_field, int base, int delta, unsigned int V, char*rng_file_in, char*rng_file_out) {

  int c;
  unsigned int ix, iix;
  int nt, it;
  int i;
  int *rng_state=NULL;
  int timeslice;
  double ran[24];
  unsigned int VOL3 = LX*LY*LZ;
  FILE *ofs=NULL;
  char filename[200];

  if(rng_file_in==NULL) {
    // initialize random number generator
    fprintf(stdout, "# [] ranldxd: using seed %u and level 2\n", g_seed);
    rlxd_init(2, g_seed);
//    rng_file = (char*)malloc(14*sizeof(char));
//    strcpy(rng_file_in, "ranlxd_state");
//    fprintf(stdout, "# [] reset rng filename to %s\n", rng_file_in);
  }

  /* set field 4 to zero, serves as "all other timeslices than timeslice" */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(s+_GSI(ix));
  }

  nt = T / delta;

  for(it=0;it<nt;it++) {
    
    if(rng_file_in!=NULL) {

      // read from file
      sprintf(filename, "%s.%d", rng_file_in, it);
      if( (ofs = fopen(filename, "r")) == (FILE*)NULL) {
        fprintf(stderr, "[] Error, could not read the random number generator state\n");
        return(105);
      }
      fprintf(stdout, "# [] timeslice %d: reading rng state from file %s\n", it, filename);
      fscanf(ofs, "%d\n", &c);
      if( (rng_state = (int*)malloc(c*sizeof(int))) == (int*)NULL ) {
        fprintf(stderr, "Error, could not read the random number generator state\n");
        return(106);
      }
      rng_state[0] = c;
      // fprintf(stdout, "# rng_state[%3d] = %3d\n", 0, rng_state[0]);
      for(i=1; i<c; i++) {
        fscanf(ofs, "%d", rng_state+i);
        // fprintf(stdout, "# rng_state[%3d] = %3d\n", i, rng_state[i]);
      }
      fclose(ofs);
      rlxd_reset(rng_state);
      free(rng_state);
    }  // of if rng_file_in != NULL

    timeslice = (base + it * delta) % T_global;

  for(ix=0; ix<VOL3; ix++) {

    switch(g_noise_type) {
      case 1:
        rangauss(ran, 24);
        break;
      case 2:
        ranz2(ran, 24);
        break;
    }

    iix = _GSI(timeslice * VOL3 + ix);
    s[iix + 0] = ran[0];
    s[iix + 1] = ran[1];
    s[iix + 2] = ran[2];
    s[iix + 3] = ran[3];
    s[iix + 4] = ran[4];
    s[iix + 5] = ran[5];
    s[iix + 6] = ran[6];
    s[iix + 7] = ran[7];
    s[iix + 8] = ran[8];
    s[iix + 9] = ran[9];
    s[iix + 10] = ran[10];
    s[iix + 11] = ran[11];
    s[iix + 12] = ran[12];
    s[iix + 13] = ran[13];
    s[iix + 14] = ran[14];
    s[iix + 15] = ran[15];
    s[iix + 16] = ran[16];
    s[iix + 17] = ran[17];
    s[iix + 18] = ran[18];
    s[iix + 19] = ran[19];
    s[iix + 20] = ran[20];
    s[iix + 21] = ran[21];
    s[iix + 22] = ran[22];
    s[iix + 23] = ran[23];

  }  // of ix
  }  // of it

  if(rng_file_out != NULL) {
    c = rlxd_size();
    if( (rng_state = (int*)malloc(c*sizeof(int))) == (int*)NULL ) {
      fprintf(stderr, "Error, could not save the random number generator state\n");
      return(102);
    }
    rlxd_get(rng_state);
    strcpy(filename, rng_file_out);
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL) {
      fprintf(stderr, "Error, could not save the random number generator state\n");
      return(103);
    }
    fprintf(stdout, "# writing rng state to file %s\n", filename);
    for(i=0; i<c; i++) fprintf(ofs, "%d\n", rng_state[i]);
    fclose(ofs);
    free(rng_state);
  }
  return(0);
}

/* timeslice sources for one-end trick with spin dilution */
int prepare_timeslice_source_one_end(double *s, double *gauge_field, int timeslice, int*momentum, unsigned int isc, int*rng_state, int rng_reset) {

  int c;
  unsigned int ix, iix, x1, x2, x3;
  int i;
  double ran[6];
  unsigned int VOL3 = LX*LY*LZ;

  if(isc<0 || isc>4) {
    fprintf(stderr, "[] Error, component number too large\n");
    return(15);
  }

  if(rng_state == NULL) {
    fprintf(stderr, "[] Error, rng_state is NULL\n");
    return(16);
  } 
  // reset rng_state
  rlxd_reset(rng_state);
  //fprintf(stdout, "# [] rng_state for source %d\n", isc);
  //for(i=0;i<rng_state[0];i++) {
  //  fprintf(stdout, "\t %d %d\n", i, rng_state[i]);
  //}

  /* set field 4 to zero, serves as "all other timeslices than timeslice" */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(s+_GSI(ix));
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

    iix = _GSI(timeslice * VOL3 + ix);
    s[iix + 6*isc+0] = ran[0];
    s[iix + 6*isc+1] = ran[1];
    s[iix + 6*isc+2] = ran[2];
    s[iix + 6*isc+3] = ran[3];
    s[iix + 6*isc+4] = ran[4];
    s[iix + 6*isc+5] = ran[5];

  }  // of ix

  if(g_source_momentum_set) {
    double phase, cphase, sphase, px, py, pz, tmp[6], *ptr;
    px = 2.*M_PI*(double)momentum[0]/(double)LX;
    py = 2.*M_PI*(double)momentum[1]/(double)LY;
    pz = 2.*M_PI*(double)momentum[2]/(double)LZ;
    // fprintf(stdout, "# [] multiply source with momentum (%d, %d, %d) <-> (%e, %e, %e)\n", momentum[0], momentum[1], momentum[2], px, py, pz);
    for(x1=0;x1<LX;x1++) {
    for(x2=0;x2<LY;x2++) {
    for(x3=0;x3<LZ;x3++) {
      phase = x1 * px + x2 * py + x3 * pz;
      cphase = cos( phase );
      sphase = sin( phase );
      iix = _GSI( g_ipt[timeslice][x1][x2][x3] )+ 6*isc;
      // fprintf(stdout, "# [] (%d, %d) phase=%e, c=%e; s=%e\n", iix, isc, phase, cphase, sphase);
      ptr = s+iix;
      memcpy(tmp, ptr, 6*sizeof(double));
      ptr[0] = tmp[0] * cphase - tmp[1] * sphase;
      ptr[1] = tmp[0] * sphase + tmp[1] * cphase;
      ptr[2] = tmp[2] * cphase - tmp[3] * sphase;
      ptr[3] = tmp[2] * sphase + tmp[3] * cphase;
      ptr[4] = tmp[4] * cphase - tmp[5] * sphase;
      ptr[5] = tmp[4] * sphase + tmp[5] * cphase;
    }}}
  }


  if(rng_reset) {
    fprintf(stdout , "# [] setting global rng_state to new state\n");
    rlxd_get(rng_state);
  }
  return(0);
}

/* timeslice sources for one-end trick with spin and color dilution */
int prepare_timeslice_source_one_end_color(double *s, double *gauge_field, int timeslice, int*momentum, unsigned int isc, int*rng_state, int rng_reset) {

  int c;
  unsigned int ix, iix, x1, x2, x3;
  int i;
  double ran[2];
  unsigned int VOL3 = LX*LY*LZ;

  if(isc<0 || isc>=12) {
    fprintf(stderr, "[] Error, component number too large\n");
    return(15);
  }

  if(rng_state == NULL) {
    fprintf(stderr, "[] Error, rng_state is NULL\n");
    return(16);
  } 
  // reset rng_state
  rlxd_reset(rng_state);
  //fprintf(stdout, "# [] rng_state for source %d\n", isc);
  //for(i=0;i<rng_state[0];i++) {
  //  fprintf(stdout, "\t %d %d\n", i, rng_state[i]);
  //}

  /* set field 4 to zero, serves as "all other timeslices than timeslice" */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(s+_GSI(ix));
  }

  for(ix=0; ix<VOL3; ix++) {

    switch(g_noise_type) {
      case 1:
        rangauss(ran, 2);
        break;
      case 2:
        ranz2(ran, 2);
        break;
    }

    iix = _GSI(timeslice * VOL3 + ix);
    s[iix + 2*isc+0] = ran[0];
    s[iix + 2*isc+1] = ran[1];

  }  // of ix

  if(g_source_momentum_set) {
    double phase, cphase, sphase, px, py, pz, tmp[6], *ptr;
    px = 2.*M_PI*(double)momentum[0]/(double)LX;
    py = 2.*M_PI*(double)momentum[1]/(double)LY;
    pz = 2.*M_PI*(double)momentum[2]/(double)LZ;
    // fprintf(stdout, "# [] multiply source with momentum (%d, %d, %d) <-> (%e, %e, %e)\n", momentum[0], momentum[1], momentum[2], px, py, pz);
    for(x1=0;x1<LX;x1++) {
    for(x2=0;x2<LY;x2++) {
    for(x3=0;x3<LZ;x3++) {
      phase = x1 * px + x2 * py + x3 * pz;
      cphase = cos( phase );
      sphase = sin( phase );
      iix = _GSI( g_ipt[timeslice][x1][x2][x3] )+ 2*isc;
      // fprintf(stdout, "# [] (%d, %d) phase=%e, c=%e; s=%e\n", iix, isc, phase, cphase, sphase);
      ptr = s+iix;
      memcpy(tmp, ptr, 2*sizeof(double));
      ptr[0] = tmp[0] * cphase - tmp[1] * sphase;
      ptr[1] = tmp[0] * sphase + tmp[1] * cphase;
    }}}
  }


  if(rng_reset) {
    fprintf(stdout , "# [prepare_timeslice_source_one_end_color] setting global rng_state to new state\n");
    rlxd_get(rng_state);
  }
  return(0);
}
