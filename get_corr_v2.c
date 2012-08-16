/****************************************************
  
 * get_corr_v2.c
 *
 * Do 16. Aug 06:45:56 CEST 2012
 *
 * PURPOSE
 * - originally copied from get_rho_corr.c
 * - extend it to include all sum_mu Pi_mumu - Pi_nunu for all nu
 *   file pattern:
 *     vacpol_con_cc_q_3320_x19y04z00t22.dat
 * DONE:
 * TODO:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ifftw.h"
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
#include "get_index.h"
#include "read_input_parser.h"

void usage() {
  fprintf(stdout, "# [get_corr_v2] Code to recover rho-rho correl.\n");
  fprintf(stdout, "# [get_corr_v2] Usage:    [options]\n");
  fprintf(stdout, "# [get_corr_v2] Options: -v verbose\n");
  fprintf(stdout, "# [get_corr_v2]          -f input filename [default cvc.input]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  int c, mu, nu, status, gid;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int source_location, have_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int sx0, sx1, sx2, sx3;
  int tsize = 0;
  double *conn  = NULL;
  double *conn2 = (double*)NULL;
  int verbose = 0;
  char filename[800];
  double ratime, retime;
  FILE *ofs;
  int ivec[4], idx[4], imu;
  double q[4], wre, wim;

  fftw_complex *inT=NULL, *outT=NULL, *inL=NULL, *outL=NULL;

  fftw_plan plan_m_T, plan_m_L;

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // set the default values
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [get_corr_v2] reading input parameters from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    fprintf(stdout, "# [get_corr_v2] T=%d, LX=%d, LY=%d, LZ=%d\n", T_global, LX, LY, LZ);
    if(g_proc_id==0) fprintf(stderr, "[get_corr_v2] Error, T and L's must be set\n");
    usage();
  }

  // initialize MPI parameters
  mpi_init(argc, argv);

  /* initialize fftw, create plan with FFTW_FORWARD ---  in contrast to
   * FFTW_BACKWARD in e.g. avc_exact */
  plan_m_T = fftw_create_plan(T_global, FFTW_FORWARD, FFTW_MEASURE);
  plan_m_L = fftw_create_plan(LX, FFTW_FORWARD, FFTW_MEASURE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "[get_corr_v2] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(32 * VOLUME, sizeof(double));
  if( (conn==NULL) ) {
    fprintf(stderr, "[get_corr_v2] Error, could not allocate memory for contr. fields\n");
    EXIT(2);
  }

  conn2= (double*)calloc(8 * T, sizeof(double));
  if( (conn2==NULL) ) {
    fprintf(stderr, "[get_corr_v2] Error, could not allocate memory for corr.\n");
    EXIT(3);
  }

  /*****************************************
   * prepare Fourier transformation arrays * 
   *****************************************/
  inT   = (fftw_complex*)malloc(T  * sizeof(fftw_complex));
  inL   = (fftw_complex*)malloc(LX * sizeof(fftw_complex));
  outT  = (fftw_complex*)malloc(T  * sizeof(fftw_complex));
  outL  = (fftw_complex*)malloc(LX * sizeof(fftw_complex));
  if( inT==NULL || inL==NULL || outT==NULL || outL==NULL ) {
    fprintf(stderr, "[get_corr_v2] Error, could not allocate fftw fields\n");
    EXIT(4);
  }

  /********************************
   * determine source coordinates *
   ********************************/
/*
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "# [get_corr_v2] process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  if(have_source_flag==1) { 
    fprintf(stdout, "# [get_corr_v2] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
  }
  have_source_flag = 0;
*/

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    memset(conn, 0, 32*VOLUME*sizeof(double));
    memset(conn2, 0, 2*T*sizeof(double));
    /***********************
     * read contractions   *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    sprintf(filename, "%s.%.4d", filename_prefix, gid);
    status = read_contraction(conn, NULL, filename, 16);
    if(status != 0) {
      fprintf(stderr, "[get_corr_v2] Error from read_contractions, status was %d\n", status);
      EXIT(5);
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v2] time to read contractions %e seconds\n", retime-ratime);
  
    // TEST Pi_mm
/*
    fprintf(stdout, "# [get_corr_v2] Pi_mm\n");
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      for(nu=0;nu<4;nu++) {
        wre = conn[_GWI(5*nu,ix,VOLUME)];
        wim = conn[_GWI(5*nu,ix,VOLUME)+1];
        fprintf(stdout, "\t%3d%3d%3d%3d%3d%16.7e%16.7e\n", nu, x0, x1, x2, x3, wre, wim);
      }
    }}}}
*/
    // TEST Ward Identity
/*
    fprintf(stdout, "# [get_corr_v2] Ward identity\n");
    for(x0=0; x0<T; x0++) {
      q[0] = 2. * sin(M_PI * (double)x0 / (double)T);
    for(x1=0; x1<LX; x1++) {
      q[1] = 2. * sin(M_PI * (double)x1 / (double)LX);
    for(x2=0; x2<LY; x2++) {
      q[2] = 2. * sin(M_PI * (double)x2 / (double)LY);
    for(x3=0; x3<LZ; x3++) {
      q[3] = 2. * sin(M_PI * (double)x3 / (double)LZ);
      ix = g_ipt[x0][x1][x2][x3];
      for(nu=0;nu<4;nu++) {
        wre =   q[0] * conn[_GWI(4*0+nu,ix,VOLUME)] + q[1] * conn[_GWI(4*1+nu,ix,VOLUME)] \
              + q[2] * conn[_GWI(4*2+nu,ix,VOLUME)] + q[3] * conn[_GWI(4*3+nu,ix,VOLUME)];
        wim =   q[0] * conn[_GWI(4*0+nu,ix,VOLUME)+1] + q[1] * conn[_GWI(4*1+nu,ix,VOLUME)+1] \
              + q[2] * conn[_GWI(4*2+nu,ix,VOLUME)+1] + q[3] * conn[_GWI(4*3+nu,ix,VOLUME)+1];
        fprintf(stdout, "\t%3d%3d%3d%3d%3d%16.7e%16.7e\n", nu, x0, x1, x2, x3, wre, wim);
      }
    }}}}
*/
  
    /***********************
     * fill the correlator *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(mu=0; mu<4; mu++) {
      ivec[0] = (0 + mu)%4;
      ivec[1] = (1 + mu)%4;
      ivec[2] = (2 + mu)%4;
      ivec[3] = (3 + mu)%4;
      idx[ivec[1]] = 0;
      idx[ivec[2]] = 0;
      idx[ivec[3]] = 0;
      tsize = (mu==0) ? T : LX;
      for(x0=0; x0<tsize; x0++) {
        idx[ivec[0]] = x0;
        for(nu=1; nu<4; nu++) {
          imu = (mu+nu) % 4;
          // ix = get_indexf(idx[0],idx[1],idx[2],idx[3],imu,imu);
          ix = _GWI(5*imu, g_ipt[idx[0]][idx[1]][idx[2]][idx[3]], VOLUME);
          // TEST
          //fprintf(stdout, "\tPi_%d_%d x0=%3d mu=%3d\tix=%8d\n", mu, mu, x0, imu, ix);
          conn2[2*(mu*T+x0)  ] += conn[ix  ];
          conn2[2*(mu*T+x0)+1] += conn[ix+1];
        }
      }
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v2] time to fill correlator %e seconds\n", retime-ratime);
   
    // TEST
/*
    fprintf(stdout, "# [get_corr_v2] correlators\n");
    for(mu=0;mu<4;mu++) {
    for(x0=0; x0<T; x0++) {
      fprintf(stdout, "\t%3d%3d%25.16e%25.16e\n", mu, x0, conn2[2*(mu*T+x0)], conn2[2*(mu*T+x0)+1]);
    }}
*/  
    /*****************************************
     * reverse Fourier transformation
     *****************************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    memcpy((void*)inT, (void*)conn2, 2*T*sizeof(double));
    fftw_one(plan_m_T, inT, outT);
    for(ix=0; ix<T; ix++) {
      conn2[2*ix  ] = outT[ix].re / (double)T;
      conn2[2*ix+1] = outT[ix].im / (double)T;
    }
    for(mu=1; mu<4; mu++) {
      memcpy((void*)inL, (void*)(conn2+2*mu*T), 2*LX*sizeof(double));
      fftw_one(plan_m_L, inL, outL);
      for(ix=0; ix<LX; ix++) {
        conn2[2*(mu*T+ix)  ] = outL[ix].re / (double)LX;
        conn2[2*(mu*T+ix)+1] = outL[ix].im / (double)LX;
      }
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v2] time for Fourier transform %e seconds\n", retime-ratime);
  
    ratime = (double)clock() / CLOCKS_PER_SEC;
    sprintf(filename, "v0v0_corr.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "[get_corr_v2] Error, could not open file %s for writing\n", filename);
      EXIT(6);
    }
    x0 = 0;
    fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], 0., gid);
    for(x0=1; x0<T/2; x0++) {
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], conn2[2*(T-x0)], gid);
    }
    x0 = T / 2;
    fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*x0], 0., gid);
    fclose(ofs);
  
    for(mu=1; mu<4; mu++) {
      sprintf(filename, "v%dv%d_corr.%.4d", mu, mu, gid);
      if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
        fprintf(stderr, "[get_corr_v2] Error, could not open file %s for writing\n", filename);
        EXIT(7);
      }
      x0 = 0;
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*(mu*T+x0)], 0., gid);
      for(x0=1; x0<LX/2; x0++) {
        fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*(mu*T+x0)], conn2[2*(mu*T+ LX-x0)], gid);
      }
      x0 = LX / 2;
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[2*(mu*T+x0)], 0., gid);
      fclose(ofs);
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v2] time to write correlator %e seconds\n", retime-ratime);
  }  // of loop on gid

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free_geometry();
  fftw_free(inT);
  fftw_free(outT);
  fftw_free(inL);
  fftw_free(outL);
  free(conn);
  free(conn2);
  fftw_destroy_plan(plan_m_T);
  fftw_destroy_plan(plan_m_L);

  fprintf(stdout, "# [get_corr_v2] %s# [get_corr_v2] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "[get_corr_v2] %s[get_corr_v2] end of run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);

}
