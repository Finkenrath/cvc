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

#ifndef _NON_ZERO
#  define _NON_ZERO (5.e-14)
#endif


int display_source_entries(double *s, int*plane);
int count_nonzero_source_entries(double *s);
double minimal_distance(double *s, int y0, int y1, int y2, int y3);

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

  if(isc>4) {
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
#else
    id = 0;
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

  if(isc>=12) {
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
#else
    id = 0;
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

/*************************************************************************
 * prepare a sequential source
 *************************************************************************/
int prepare_sequential_point_source (double*source, int isc, int timeslice, int*momentum, int smear, double*work, double*gauge_field_smeared) {
  int i, ix;
  int gx0, gx1, gx2, gx3;
  int lx0, lx1, lx2, lx3;
  int x1, x2, x3;
  int scoords[4], id=0;
  int have_source=0, lts=-1;
  char filename[200];
  double spinor1[24], phase;
  complex w;

  if(source==NULL) {
    EXIT_WITH_MSG(1, "[prepare_sequential_source] Error, source is NULL\n");
  }

  if(work==NULL && smear) {
    EXIT_WITH_MSG(2, "[prepare_sequential_source] Error, work is NULL, but need to smear\n");
  }

  // determine source location
  gx0 = g_source_location / (LX_global*LY_global*LZ_global);
  gx3 = g_source_location - gx0* LX_global*LY_global*LZ_global;
  gx1 = gx3 / (LY_global*LZ_global);
  gx3 -= gx1 * LY_global*LZ_global;
  gx2 = gx3 / LZ_global;
  gx3 -= gx2 * LZ_global;

  if(g_cart_id==0) fprintf(stdout, "# [prepare_sequential_source] global source location = (%d,%d,%d,%d)\n",
     gx0, gx1, gx2, gx3);
  scoords[0] = gx0 / T;
  scoords[1] = gx1 / LX;
  scoords[2] = gx2 / LY;
  scoords[3] = gx3 / LZ;
#ifdef MPI
  MPI_Cart_rank(g_cart_grid, scoords, &id);
#endif
  lx0 = gx0 % T;
  lx1 = gx1 % LX;
  lx2 = gx2 % LY;
  lx3 = gx3 % LZ;

  if(g_cart_id==id) fprintf(stdout, "# [prepare_sequential_source] process %d has the source location at "\
      "(%d,%d,%d,%d)\n", id, lx0, lx1, lx2, lx3);

  // (0) which processes have source?
#if ( (defined PARALLELTX) || (defined PARALLELTXY) ) && (defined HAVE_QUDA)
  if(g_proc_coords[3] == timeslice / T ) {
#else
  if(g_proc_coords[0] == timeslice / T ) {
#endif
    have_source = 1;
    lts = timeslice % T;
  } else {
    have_source = 0;
    lts = -1;
  }
  if(have_source) fprintf(stdout, "# [] process %d has source\n", g_cart_id);

  // (1) propagator filename
  sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix, Nconf, gx0, gx1, gx2, gx3,\
      isc);
  // (2) read the propagator
  if(g_cart_id==0) fprintf(stdout, "# [prepare_sequential_source] reading propagator from file %s\n", filename);
  check_error(read_lime_spinor(source, filename, 0), "read_lime_spinor", NULL, 3);

  // (3) set source to zero outside source timeslice
  for(i=0;i<T;i++) {
    if(i!=lts) {
      memset((void*)(source+_GSI(g_ipt[i][0][0][0])), 0, LX*LY*LZ*24*sizeof(double) );
    }
  }

  // (4) smear the source
  if(smear && N_Jacobi>0) {
    if(g_cart_id==0) fprintf(stdout, "#  [prepare_sequential_source] smearing source with N_Jacobi=%d, kappa_Jacobi=%e\n",\
        N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
      Jacobi_Smearing_Step_one_threads(gauge_field_smeared, source, work, N_Jacobi, kappa_Jacobi);
#else
      for(i=0; i<N_Jacobi; i++) {
        Jacobi_Smearing_Step_one(gauge_field_smeared, source, work, kappa_Jacobi);
      }
#endif
  }

  // (5) multiply with phase and Gamma structure
  if(have_source) {
    for(x1=0;x1<LX;x1++) {
    for(x2=0;x2<LY;x2++) {
    for(x3=0;x3<LZ;x3++) {
      ix = g_ipt[lts][x1][x2][x3];
      phase = 2. * M_PI * ( ( x1 + g_proc_coords[1]*LX - gx1 ) * momentum[0] / (double)LX_global
                          + ( x2 + g_proc_coords[2]*LY - gx2 ) * momentum[1] / (double)LY_global
                          + ( x3 + g_proc_coords[3]*LZ - gx3 ) * momentum[2] / (double)LZ_global );
      w.re =  cos(phase);
      w.im = -sin(phase);
      _fv_eq_gamma_ti_fv(spinor1, 5, source + _GSI(ix));
      _fv_eq_fv_ti_co(source + _GSI(ix), spinor1, &w);
    }}}
  }  // of if have_source

/*
  // TEST
/*
  {
    FILE*ofs;
    sprintf(filename, "seq_source.ascii.%.2d.%.2d.%.2d", isc, g_nproc, g_cart_id);
    ofs = fopen(filename, "w");
    fprintf(ofs, "# [prepare_sequential_source] the sequential source:\n");
    for(ix=0;ix<VOLUME;ix++) {
      for(i=0;i<12;i++) {
        fprintf(ofs, "%6d%3d%25.16e%25.16e\n", ix, i, source[_GSI(ix)+2*i], source[_GSI(ix)+2*i+1]);
      }
    }
    fclose(ofs);
  }
*/
  return(0);
}  // end of function


int prepare_space_diluted_source(double *s, unsigned int degree, unsigned int number, int is, int ic, int *isLat, int*rng_state, int rng_reset) {

  unsigned int ix, iix, iiy;
  unsigned int gx[4], modval, length;
  int i, pid;
  double *ran = NULL;
  unsigned int VLat = 0, count;
  char filename[200];
  unsigned int degree2 = 1;
  unsigned int num_comp = 0, step = 0, offset = 0;

  if(rng_state == NULL) {
    fprintf(stderr, "[prepare_space_diluted_source] Error, rng_state is NULL\n");
    EXIT(16);
  }
  rlxd_reset(rng_state);

  // initialize
  memset(s, 0, 24*VOLUME*sizeof(double));

  //for(i=0;i<degree;i++) degree2 *= 2;
  degree2 = 1 << ((degree/2)*4 + degree%2 );
  if( number >= degree2 ) {
    EXIT_WITH_MSG(17, "[prepare_space_diluted_source] Error, number is larger/equal 2^degree\n");
  }
 
  if(g_cart_id==0) fprintf(stdout, "# [prepare_space_diluted_source] source parameters: number=%u; is=%d; ic=%d\n", number , is, ic);


  if(is==-1 && ic==-1) {
    num_comp = 24;  // only space dilution
    offset   =  0;
    step     =  1;
  } else if(ic == -1) {
    num_comp =  6;  // space and spin dilution
    offset   =  6*is;
    step     =  1;
  } else if(is == -1) {
    num_comp =  8;  // space and color dilution
    offset   =  2*ic;
    step     =  3;
  } else {
    num_comp =  2;  // space, spin and color dilution
    offset   =  2*( 3 * is + ic );
    step     =  1;
  }

  VLat = VOLUME / degree2;
  if(g_cart_id==0) fprintf(stdout, "# [prepare_space_diluted_source] sub-lattice volume = %u; number of components = %u; step = %u, offset = %u\n",
      VLat, num_comp, step, offset);

  ran = (double*)malloc(num_comp*VLat*sizeof(double));
  if(ran == NULL) {
    EXIT_WITH_MSG(18, "[prepare_space_diluted_source] Error, could not alloc ran\n");
  }

  for(pid = 0; pid<g_nproc; pid++) {
    if(g_cart_id == pid) {
      switch(g_noise_type) {
        case 1:
          rangauss(ran, num_comp*VLat);
          break;
        case 2:
          ranz2(ran, num_comp*VLat);
          break;
      }
    }
#ifdef MPI
    MPI_Barrier(g_cart_grid);
    sync_rng_state(pid, 0);
#endif
  }

  count = 0;
  for(ix=0; ix<VOLUME; ix++) {

    if( isLat[ix] != number ) continue;

    iix = _GSI( ix );
    iiy = num_comp * count;

    for(i=0; i < num_comp; i += 2) {
      s[iix + offset + i*step    ] = ran[iiy + i    ];  // real part
      s[iix + offset + i*step + 1] = ran[iiy + i + 1];  // imag part
    }
    count++;
  }  // of ix

  if(ran != NULL) free(ran);

  sync_rng_state(0, rng_reset);

#ifndef MPI
#if 0
  int plane[4];
  plane[0] = 0; plane[1] = 0; plane[2] = -1; plane[3] = -1;
  display_source_entries(s, plane );

  plane[0] = 0; plane[1] = 1; plane[2] = -1; plane[3] = -1;
  display_source_entries(s, plane );

  plane[0] = 0; plane[1] = 2; plane[2] = -1; plane[3] = -1;
  display_source_entries(s, plane );

  plane[0] = 1; plane[1] = 0; plane[2] = -1; plane[3] = -1;
  display_source_entries(s, plane );
#endif
#endif
  // TEST
  //if(number == 0) {
  //  fprintf(stdout, "\n# [%2d] minimal distance for degree %u = %e\n", g_cart_id, degree, minimal_distance(s, 0, 0, 0, 0));
  //}
  //count_nonzero_source_entries(s);

  return(0);
}

int display_source_entries(double *s, int*plane) {

  int i, j, k;
  int Lmax[2], Lmaxordered[4];
  int coords[4];
  double norm[2];
  int LL[4], LLordered[4];
  FILE *ofs = NULL;
  char filename[200];

  LLordered[0] = LX*LY*LZ;
  LLordered[1] =    LY*LZ;
  LLordered[2] =       LZ;
  LLordered[3] =        1;

  Lmaxordered[0] = T;
  Lmaxordered[1] = LX;
  Lmaxordered[2] = LY;
  Lmaxordered[3] = LZ;
 
  sprintf(filename, "source_field");

  j = 0;
  k = 0;
  for(i=0;i<4;i++) {
    if(plane[i]>=0) {  // fixed coordinate
      coords[2+j] = plane[i];
      LL[2+j] = LLordered[i];
      j++;
    } else {
      Lmax[k] = Lmaxordered[i];
      LL[k] = LLordered[i];
      sprintf(filename, "%s_%d", filename, i);
      k++;
    }
  }

  sprintf(filename, "%s.%.2d%.2d", filename, coords[2], coords[3]);

  if( (k!=2) || (j!=2) ) {
    return(1);
  }

  ofs = fopen(filename, "w");

  for(i=0;i<Lmax[0]; i++) {
    coords[0] = i;
  for(j=0;j<Lmax[1]; j++) {
    coords[1] = j;
    k = coords[0]*LL[0] + coords[1]*LL[1] +coords[2]*LL[2] +coords[3]*LL[3];
    _co_eq_fv_dag_ti_fv((complex*)norm, s+_GSI(k), s+_GSI(k));
    if(norm[0]>0) fprintf(ofs, " 1");
    else fprintf(ofs, " 0");
  }
    fprintf(ofs, "\n");
  }
  fclose(ofs);
  return(0);
}


int count_nonzero_source_entries(double *s) {

  unsigned int i, k;
  double norm[2];

  k = 0;
  for(i=0;i<VOLUME;i++) {
    _co_eq_fv_dag_ti_fv((complex*)norm, s+_GSI(i), s+_GSI(i));
    k = k + (norm[0] > _NON_ZERO);
  }

  fprintf(stdout, "# [%2d] Number of non-zero source entries is %u\n", g_cart_id, k);

  return(0);
}

double minimal_distance(double *s, int y0, int y1, int y2, int y3) {
#define _SQR(_a) ((_a)*(_a))

  unsigned int ix, ix_min=0;
  int x0, x1, x2, x3;
  int z0, z1, z2, z3;
  double dist=0, dist_min = (double)LX * 2.;
  complex w;

  for(x0=0;x0<T; x0++) {
    z0 = x0 + g_proc_coords[0]* T - y0;
  for(x1=0;x1<LX;x1++) {
    z1 = x1 + g_proc_coords[1]*LX - y1;
  for(x2=0;x2<LY;x2++) {
    z2 = x2 + g_proc_coords[2]*LY - y2;
  for(x3=0;x3<LZ;x3++) {
    z3 = x3 + g_proc_coords[3]*LZ - y3;

    if( (z0==0) && (z1==0) && (z2==0) && (z3==0) ) continue;

    ix = g_ipt[x0][x1][x2][x3];

    _co_eq_fv_dag_ti_fv(&w, s+_GSI(ix), s+_GSI(ix));
    if(w.re > _NON_ZERO) {
      dist = sqrt( _SQR( z0 ) + _SQR( z1 ) + _SQR( z2 ) + _SQR( z3 ) );
      if(dist < dist_min) {
        dist_min = dist;
        ix_min   = ix;
      }
    }
  }}}}
  return(dist_min);
#undef _SQR
}
