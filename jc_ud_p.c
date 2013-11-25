/*********************************************************************************
 * jc_ud_p.c
 *
 * Wed Sep  8 22:52:45 CEST 2010
 *
 * PURPOSE:
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
#include "ifftw.h"
#include <omp.h>
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
  fprintf(stdout, "Code to perform quark-disconnected conserved vector current contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
 
  int Tsub = 2; 
  int Lsub = 2; 
  int c, i, mu, dims[4], tid, nthreads;
  int count        = 0;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, iiy, iix, ip;
  int x0b, x0e, x1b, x1e, x2b, x2e, x3b, x3e;
  int it, ix, iy, iz;
  int sid1, sid2, status, gid;
  int include_negative=0, t_start=0, x_start=0, y_start=0, z_start=0;
  size_t nprop=0;
  double *data=NULL, *work[48], *work2=NULL;
  double q2, fnorm, q[4], r2;
  char filename[100];
  double ratime, retime;
  complex w, w2;
  FILE *ofs=NULL;

/*
  fftw_complex *ft_in=NULL;
  fftwnd_plan plan_p;
*/

  /****************************************
   * initialize the distance vectors
   ****************************************/

  while ((c = getopt(argc, argv, "h?f:l:t:m")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'l':
      Lsub = atoi(optarg);
      fprintf(stdout, "# using Lsub = %d\n", Lsub);
      break;
    case 't':
      Tsub = atoi(optarg);
      fprintf(stdout, "# using Tsub = %d\n", Tsub);
      break;
    case 'm':
      include_negative = 1;
      fprintf(stdout, "# will do negative R_i, too\n");
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
  fprintf(stdout, "# Reading input from file %s\n", filename);
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

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* jc_ud_p\n");
  fprintf(stdout, "**************************************************\n\n");

  /* initialize fftw */
/*
  dims[0] = Tsub;
  dims[1] = Lsub;
  dims[2] = Lsub;
  dims[3] = Lsub;
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
*/
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  if(!include_negative) {
    FFTW_LOC_VOLUME = Tsub * Lsub*Lsub*Lsub;
  } else {
    FFTW_LOC_VOLUME = (2*Tsub-1) * (2*Lsub-1) * (2*Lsub-1) * (2*Lsub-1);
  }
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "#       T            = %3d\n"\
		  "#       Tstart       = %3d\n"\
		  "#       l_LX_at      = %3d\n"\
		  "#       l_LXstart_at = %3d\n"\
		  "#       FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, T, Tstart, l_LX_at, l_LXstart_at, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  data = (double*)calloc(8*FFTW_LOC_VOLUME, sizeof(double));
  if( data==NULL ) { 
    fprintf(stderr, "could not allocate memory for data\n");
    exit(3);
  }

  nprop = (size_t)(g_sourceid2 - g_sourceid) / (size_t)g_sourceid_step + 1;
  fprintf(stdout, "# number of stoch. propagators = %lu\n", nprop);
  work[0] = (double*)calloc(nprop*8*VOLUME, sizeof(double));
  if( work[0] == NULL ) { 
    fprintf(stderr, "could not allocate memory for work field\n");
    exit(5);
  }
  for(i=1; i< nprop; i++) {
    work[i] = work[i-1] + 8*VOLUME;
  }
/*
  ft_in = (fftw_complex*)malloc(FFTW_LOC_VOLUME * sizeof(fftw_complex));
  if(ft_in == NULL) {
    fprintf(stderr, "could not allocate memory ft_in\n");
    exit(6);
  }
*/
  work2 = (double*)calloc(8*FFTW_LOC_VOLUME, sizeof(double));
  if( work2 == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work2\n");
    exit(7);
  }

  fnorm = 1. / ( (double)nprop * (double)(nprop-1));
  fprintf(stdout, "# fnorm = %25.16e\n", fnorm);

  /***********************************************
   * choose the start values for the entries
   *   of the R-vector
   ***********************************************/
  if(include_negative) {
    t_start = -Tsub+1;
    x_start = -Lsub+1;
    y_start = -Lsub+1;
    z_start = -Lsub+1;
  } else {
    t_start = 0;
    x_start = 0;
    y_start = 0;
    z_start = 0;
  }
 fprintf(stdout, "#\n# t_start=%d, x_start=%d, y_start=%d, z_start=%d\n", 
    t_start, x_start, y_start, z_start);

/*
#pragma omp parallel private(tid) num_threads(Tsub)
{
  tid = omp_get_thread_num();
  fprintf(stdout, "# Hello World from thread = %d\n", tid);
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    fprintf(stdout, "# Number of threads = %d\n", nthreads);
  }
}
  nthreads = include_negative ? 2 * Tsub - 1 : Tsub;
  fprintf(stdout, "#\n# number of threads = %d\n", nthreads);
*/
  
  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    for(ix=0; ix<8*FFTW_LOC_VOLUME; ix++) data[ix] = 0.;

    for(sid1=0; sid1<nprop; sid1++) {
      sprintf(filename, "jc_ud_x.%.4d.%.4d", gid, g_sourceid + sid1*g_sourceid_step);
      if(read_lime_contraction(work[sid1], filename, 4, 0) != 0) {
        fprintf(stderr, "Error, could not read field no. %d\n", sid1);
        exit(15);
      }
    }
    /***********************************************
     * start loop on source id.s 
     ***********************************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(sid1=0; sid1<nprop-1; sid1++) {
    for(sid2=sid1+1; sid2<nprop; sid2++) {

      for(ix=0; ix<8*FFTW_LOC_VOLUME; ix++) work2[ix] = 0.;

#pragma omp parallel for private(it,ix,iy,iz,ip,iix,iiy,w,w2,x0b,x0e,x1b,x1e,x2b,x2e,x3b,x3e) shared(sid1,sid2,t_start,x_start,y_start,z_start) num_threads((Tsub-t_start))
      for(it=t_start; it<Tsub; it++) {
      for(ix=x_start; ix<Lsub; ix++) {
      for(iy=y_start; iy<Lsub; iy++) {
      for(iz=z_start; iz<Lsub; iz++) {
        ip = (( (it-t_start)*(Lsub-x_start) + (ix-x_start) ) * (Lsub-y_start)  + (iy-y_start)) * (Lsub-z_start)+ (iz-z_start);
        x0b = it>=0 ? it : 0; x0e = it>=0 ? T_global : T_global+it;
        x1b = ix>=0 ? ix : 0; x1e = ix>=0 ? LX : LX + ix;
        x2b = iy>=0 ? iy : 0; x2e = iy>=0 ? LY : LY + iy;
        x3b = iz>=0 ? iz : 0; x3e = iz>=0 ? LZ : LZ + iz;
/*
        fprintf(stdout, "# ip=(%3d,%3d,%3d,%3d), "\
                        "x0be=(%3d,%3d), x1be=(%3d,%3d), x2be=(%3d,%3d), x3be=(%3d,%3d)\n", 
          it, ix, iy, iz, x0b, x0e, x1b, x1e, x2b, x2e, x3b, x3e);
        fprintf(stdout, "# ip=(%3d,%3d,%3d,%3d)=%8d\n", it, ix, iy, iz, ip);
       w2.re = 0.; w2.im = 0.;
*/

        for(x0=x0b; x0<x0e; x0++) {
        for(x1=x1b; x1<x1e; x1++) {
        for(x2=x2b; x2<x2e; x2++) {
        for(x3=x3b; x3<x3e; x3++) {
          iix = g_ipt[x0][x1][x2][x3];
          iiy = g_ipt[x0-it][x1-ix][x2-iy][x3-iz];
          _co_eq_co_ti_co(&w,  (complex*)(work[sid1]+_GWI(0,iix,VOLUME)), (complex*)(work[sid2]+_GWI(0,iiy,VOLUME)));
//          _co_eq_co_ti_co(&w2, (complex*)(work[sid2]+_GWI(0,iix,VOLUME)), (complex*)(work[sid1]+_GWI(0,iiy,VOLUME)));
          work2[_GWI(0,ip,FFTW_LOC_VOLUME)  ] += w.re + w2.re;
          work2[_GWI(0,ip,FFTW_LOC_VOLUME)+1] += w.im + w2.im;
//          fprintf(stdout, "%8d%8d%4d%25.16e%25.16e\n", iix, iiy, 0, w.re, w.im);

          _co_eq_co_ti_co(&w,  (complex*)(work[sid1]+_GWI(1,iix,VOLUME)), (complex*)(work[sid2]+_GWI(1,iiy,VOLUME)));
//          _co_eq_co_ti_co(&w2, (complex*)(work[sid2]+_GWI(1,iix,VOLUME)), (complex*)(work[sid1]+_GWI(1,iiy,VOLUME)));
          work2[_GWI(1,ip,FFTW_LOC_VOLUME)  ] += w.re + w2.re;
          work2[_GWI(1,ip,FFTW_LOC_VOLUME)+1] += w.im + w2.im;
//          fprintf(stdout, "%8d%8d%4d%25.16e%25.16e\n", iix, iiy, 1, w.re, w.im);

          _co_eq_co_ti_co(&w,  (complex*)(work[sid1]+_GWI(2,iix,VOLUME)), (complex*)(work[sid2]+_GWI(2,iiy,VOLUME)));
//          _co_eq_co_ti_co(&w2, (complex*)(work[sid2]+_GWI(2,iix,VOLUME)), (complex*)(work[sid1]+_GWI(2,iiy,VOLUME)));
          work2[_GWI(2,ip,FFTW_LOC_VOLUME)  ] += w.re + w2.re;
          work2[_GWI(2,ip,FFTW_LOC_VOLUME)+1] += w.im + w2.im;
//          fprintf(stdout, "%8d%8d%4d%25.16e%25.16e\n", iix, iiy, 2, w.re, w.im);

          _co_eq_co_ti_co(&w,  (complex*)(work[sid1]+_GWI(3,iix,VOLUME)), (complex*)(work[sid2]+_GWI(3,iiy,VOLUME)));
//          _co_eq_co_ti_co(&w2, (complex*)(work[sid2]+_GWI(3,iix,VOLUME)), (complex*)(work[sid1]+_GWI(3,iiy,VOLUME)));
          work2[_GWI(3,ip,FFTW_LOC_VOLUME)  ] += w.re + w2.re;
          work2[_GWI(3,ip,FFTW_LOC_VOLUME)+1] += w.im + w2.im;
//          fprintf(stdout, "%8d%8d%4d%25.16e%25.16e\n", iix, iiy, 3, w.re, w.im);
        }}}}
      }}}}

      for(ix=0; ix<8*FFTW_LOC_VOLUME; ix++) data[ix] += work2[ix];
     
    }  /* of loop on sid2 */
    }  /* of loop on sid1 */

    for(ix=0; ix<8*FFTW_LOC_VOLUME; ix++) data[ix] *= fnorm;
/*
    for(mu=0; mu<4; mu++) {
      ip = 0;
      for(it=t_start; it<Tsub; it++) {
      for(ix=x_start; ix<Lsub; ix++) {
      for(iy=y_start; iy<Lsub; iy++) {
      for(iz=z_start; iz<Lsub; iz++) {
        //p = ((it*Lsub+ix)*Lsub+iy)*Lsub+iz;
        fprintf(stdout, "%3d%3d%3d%3d%3d%25.16e%25.16e\n", mu, it, ix, iy, iz,
          data[_GWI(mu,ip,FFTW_LOC_VOLUME)], data[_GWI(mu,ip,FFTW_LOC_VOLUME)+1]);
        ip++;
      }}}}
    }
*/

    /************************************************
     * save results in position space
     ************************************************/
    sprintf(filename, "pi_ud_r.%4d", gid);
    ofs = fopen(filename, "w");
    if (ofs==NULL) {
     fprintf(stderr, "Error, could not open file %s for writing\n", filename);
     exit(9);
    }
    for(mu=0; mu<4; mu++) {
      ip = 0;
      for(it=t_start; it<Tsub; it++) {
        for(ix=x_start; ix<Lsub; ix++) {
        for(iy=y_start; iy<Lsub; iy++) {
        for(iz=z_start; iz<Lsub; iz++) {
          r2 = (double)(ix*ix) + (double)(iy*iy) + (double)(iz*iz);
          //ip = ((it*Lsub+ix)*Lsub+iy)*Lsub+iz;
          fprintf(ofs, "%3d%3d%3d%3d%3d%16.7e%25.16e%25.16e\n", mu, it, ix, iy, iz, r2,
            data[_GWI(mu,ip,FFTW_LOC_VOLUME)], data[_GWI(mu,ip,FFTW_LOC_VOLUME)+1]);
          ip++;
        }}}
      }
    }
    fclose(ofs);

/*
    for(mu=0; mu<4; mu++) { 
      memcpy( (void*)ft_in, (void*)(data + 2*mu*FFTW_LOC_VOLUME), 2*FFTW_LOC_VOLUME*sizeof(double));
      fftwnd_one(plan_p, ft_in, NULL);
      memcpy( (void*)(data + 2*mu*FFTW_LOC_VOLUME), (void*)ft_in, 2*FFTW_LOC_VOLUME*sizeof(double));
    }
*/

    /************************************************
     * save results in momentum space
     ************************************************/
/*
    sprintf(filename, "pi_ud_p.%4d", gid);
    ofs = fopen(filename, "w");
    if (ofs==NULL) {
     fprintf(stderr, "Error, could not open file %s for writing\n", filename);
     exit(8);
    }

    for(mu=0; mu<4; mu++) {
      for(it=0; it<Tsub; it++) {
        q[0] = 2. * sin( M_PI * (double)it / (double)T  );
      for(ix=0; ix<Lsub; ix++) {
        q[1] = 2. * sin( M_PI * (double)ix / (double)LX );
      for(iy=0; iy<Lsub; iy++) {
        q[2] = 2. * sin( M_PI * (double)iy / (double)LY );
      for(iz=0; iz<Lsub; iz++) {
        q[3] = 2. * sin( M_PI * (double)iz / (double)LZ );
        q2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
        ip = ((it*Lsub+ix)*Lsub+iy)*Lsub+iz;
        fprintf(ofs, "%3d%3d%3d%3d%3d%16.7e%25.16e%25.16e\n", mu, it, ix, iy, iz, q2, 
          data[_GWI(mu,ip,FFTW_LOC_VOLUME)], data[_GWI(mu,ip,FFTW_LOC_VOLUME)+1]);
      }}}}
    }
    fclose(ofs);
*/
    retime = (double)clock() / CLOCKS_PER_SEC;
    if(g_cart_id == 0) fprintf(stdout, "# time for building correl.: %e seconds\n", retime-ratime);

  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(work[0]);
  free(work2);
  free(data);
/*
  free(ft_in);
  fftwnd_destroy_plan(plan_p);
*/
  return(0);

}
