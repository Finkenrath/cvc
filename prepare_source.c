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
 * - test 1-dim. MPI version
 * - implement 2-, 3-dim. MPI version
 * - parallel version of coherent source generation
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

int prepare_timeslice_source(double *s, double *gauge_field, int timeslice, unsigned int V, int*rng_state, int rng_reset) {
#if !(defined PARALLELTX) && !(defined PARALLELTXY)
  int c;
  unsigned int ix, iix;
  int i, id=0;
  int coords[4], ts = timeslice;
  double ran[24];
  unsigned int VOL3 = LX*LY*LZ;
  FILE *ofs=NULL;
  char filename[200];

  if(rng_state==NULL) {
    if(g_cart_id == 0) fprintf(stderr, "[prepare_timeslice_source] Error, rng_state is NULL\n ");
    return(1);
  }

  // set field 4 to zero, serves as "all other timeslices than timeslice"
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(s+_GSI(ix));
  }

#ifdef MPI
  coords[0] = timeslice / T;
  coords[1] = 0;
  coords[2] = 0;
  coords[3] = 0;
  MPI_Cart_rank(g_cart_grid, coords, &id);
  ts = (g_cart_id==id) ? timeslice % T : -1;
#endif

  if(g_cart_id==id) {
    for(ix=0; ix<VOL3; ix++) {
  
      switch(g_noise_type) {
        case 1:
          rangauss(ran, 24);
          break;
        case 2:
          ranz2(ran, 24);
          break;
      }

      iix = _GSI(ts * VOL3 + ix);
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
  }    // of if g_cart_id == id

  sync_rng_state(id, rng_reset);

  return(0);
#else
  if(g_cart_id==0) fprintf(stderr, "[prepare_timeslice_source] Error, 2-d, 3-d parallel version not implemented\n");
  return(1);
#endif
}

int prepare_coherent_timeslice_source(double *s, double *gauge_field, int base, int delta, unsigned int V, int*rng_state, int rng_reset) {
#if !(defined PARALLELTX) && !(defined PARALLELTXY)
  int c;
  unsigned int ix, iix;
  int nt, it;
  int i;
  int timeslice;
  double ran[24];
  unsigned int VOL3 = LX*LY*LZ;
  int coords[4], id=0;
  FILE *ofs=NULL;
  char filename[200];

  if(rng_state==NULL) {
    fprintf(stderr, "[] Error, rng_state is NULL\n");
    return(1);
  }

  // set field 4 to zero, serves as "all other timeslices than timeslice"
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(s+_GSI(ix));
  }

  nt = T_global / delta;
  if(g_cart_id==0) fprintf(stdout, "# [] number of timeslices = %d\n", nt);

  for(it=0;it<nt;it++) {
    
    timeslice = (base + it * delta) % T_global;
#ifdef MPI
    coords[0] = timeslice / T;
    coords[1] = 0;
    coords[2] = 0;
    coords[3] = 0;
    MPI_Cart_rank(g_cart_grid, coords, &id);
    timeslice = g_cart_id==id ? timeslice % T : -1;
#endif
    if(g_cart_id == id) {

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
    }    // of if g_cart_id == id
#ifdef MPI
    sync_rng_state(id, 0);
#endif
  }  // of it

  sync_rng_state(id, rng_reset);

  return(0);
#else
  if(g_cart_id==0) fprintf(stderr, "[prepare_coherent_timeslice_source] Error, 2-d, 3-d parallel version not implemented\n");
  return(1);
#endif
}

/* timeslice sources for one-end trick with spin dilution */
int prepare_timeslice_source_one_end(double *s, double *gauge_field, int timeslice, int*momentum, unsigned int isc, int*rng_state, int rng_reset) {
#if !(defined PARALLELTX) && !(defined PARALLELTXY)
  int c;
  unsigned int ix, iix, x1, x2, x3;
  int i, id, coords[4]; ;
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
  // fprintf(stdout, "# [] rng_state for source %d\n", isc);
  // for(i=0;i<rng_state[0];i++) {
  //   fprintf(stdout, "\t %d %d\n", i, rng_state[i]);
  // }

  /* set field 4 to zero, serves as "all other timeslices than timeslice" */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(s+_GSI(ix));
  }

  // which process fills the timeslice ?
#ifdef MPI
    coords[0] = timeslice / T;
    coords[1] = 0;
    coords[2] = 0;
    coords[3] = 0;
    MPI_Cart_rank(g_cart_grid, coords, &id);
    timeslice = g_cart_id==id ? timeslice % T : -1;
#endif

  if(timeslice >= 0) {
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
      px = 2.*M_PI*(double)momentum[0]/(double)LX_global;
      py = 2.*M_PI*(double)momentum[1]/(double)LY_global;
      pz = 2.*M_PI*(double)momentum[2]/(double)LZ_global;
      // fprintf(stdout, "# [] multiply source with momentum (%d, %d, %d) <-> (%e, %e, %e)\n", momentum[0], momentum[1], momentum[2], px, py, pz);
      for(x1=0;x1<LX;x1++) {
      for(x2=0;x2<LY;x2++) {
      for(x3=0;x3<LZ;x3++) {
        phase = (x1+g_proc_coords[1]*LX) * px + (x2+g_proc_coords[2]*LY) * py + (x3+g_proc_coords[3]*LZ) * pz;
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
  }  // of if timeslice >= 0

  sync_rng_state(id, rng_reset);

  return(0);
#else
  if(g_cart_id==0) fprintf(stderr, "[prepare_timeslice_source_one_end] Error, 2-d, 3-d parallel version not implemented\n");
  return(1);
#endif
}

/* timeslice sources for one-end trick with spin and color dilution */
int prepare_timeslice_source_one_end_color(double *s, double *gauge_field, int timeslice, int*momentum, unsigned int isc, int*rng_state, int rng_reset) {
#if !(defined PARALLELTX) && !(defined PARALLELTXY)
  int c;
  unsigned int ix, iix, x1, x2, x3;
  int i, id, coords[4];
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

  // which process fills the timeslice ?
#ifdef MPI
    coords[0] = timeslice / T;
    coords[1] = 0;
    coords[2] = 0;
    coords[3] = 0;
    MPI_Cart_rank(g_cart_grid, coords, &id);
    timeslice = g_cart_id==id ? timeslice % T : -1;
#endif

  if(timeslice>=0) {
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
      px = 2.*M_PI*(double)momentum[0]/(double)LX_global;
      py = 2.*M_PI*(double)momentum[1]/(double)LY_global;
      pz = 2.*M_PI*(double)momentum[2]/(double)LZ_global;
      // fprintf(stdout, "# [] multiply source with momentum (%d, %d, %d) <-> (%e, %e, %e)\n", momentum[0], momentum[1], momentum[2], px, py, pz);
      for(x1=0;x1<LX;x1++) {
      for(x2=0;x2<LY;x2++) {
      for(x3=0;x3<LZ;x3++) {
        phase = (x1+g_proc_coords[1]*LX) * px + (x2*g_proc_coords[2]*LY) * py + (x3*g_proc_coords[3]*LZ) * pz;
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
  }  // of if timeslice >= 0

  sync_rng_state(id, rng_reset);

  return(0);
#else
  if(g_cart_id==0) fprintf(stderr, "[prepare_timeslice_source_one_end_color] Error, 2-d, 3-d parallel version not implemented\n");
  return(1);
#endif
}
