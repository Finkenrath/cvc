/****************************************************
 * 
 * get_corr_v3.c
 *
 * Do 13. Sep 16:03:23 CEST 2012
 *
 * PURPOSE
 * - originally copied from get_corr_v2.c
 * - calculate Pi_jj(q_0, q_j)
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
  fprintf(stdout, "# [get_corr_v3] Code to recover rho-rho correl.\n");
  fprintf(stdout, "# [get_corr_v3] Usage:    [options]\n");
  fprintf(stdout, "# [get_corr_v3] Options: -v verbose\n");
  fprintf(stdout, "# [get_corr_v3]          -f input filename [default cvc.input]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  int c, mu, nu, status, gid, i;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int source_location, have_source_flag = 0;
  int x0, x1, x2, x3, ix, ip;
  int sx0, sx1, sx2, sx3;
  int tsize = 0;
  int Lhp1;
  double *conn  = NULL;
  double **conn2 = NULL;
  int verbose = 0;
  char filename[800];
  double ratime, retime;
  FILE *ofs;
  int ivec[4], idx[4], imu;
  double q[4], wre, wim;
  int append = 0;
  int byte_swap=0;
  fftw_complex *inT=NULL, *outT=NULL;

  fftw_plan plan_m_T;

  while ((c = getopt(argc, argv, "bah?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'a':
      append = 1;
      fprintf(stdout, "# [] will append to output file\n");
      break;
    case 'b':
      byte_swap = 1;
      fprintf(stdout, "# [] will carry out byte swap\n");
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
  fprintf(stdout, "# [get_corr_v3] reading input parameters from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    fprintf(stdout, "# [get_corr_v3] T=%d, LX=%d, LY=%d, LZ=%d\n", T_global, LX, LY, LZ);
    if(g_proc_id==0) fprintf(stderr, "[get_corr_v3] Error, T and L's must be set\n");
    usage();
  }

  // initialize MPI parameters
  mpi_init(argc, argv);

  /* initialize fftw, create plan with FFTW_FORWARD ---  in contrast to
   * FFTW_BACKWARD in e.g. avc_exact */
  plan_m_T = fftw_create_plan(T_global, FFTW_FORWARD, FFTW_MEASURE);
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
    fprintf(stderr, "[get_corr_v3] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  Lhp1 = LX / 2 + 1;

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(32 * VOLUME, sizeof(double));
  if( (conn==NULL) ) {
    fprintf(stderr, "[get_corr_v3] Error, could not allocate memory for contr. fields\n");
    EXIT(2);
  }

  conn2= (double**)calloc(Lhp1, sizeof(double*));
  conn2[0]= (double*)calloc(Lhp1*2* T, sizeof(double));
  for(i=1; i<Lhp1;i++) conn2[i] = conn2[i-1] + 2*T;
  //if( (conn2==NULL) ) {
  //  fprintf(stderr, "[get_corr_v3] Error, could not allocate memory for corr.\n");
  //  EXIT(3);
  //}

  /*****************************************
   * prepare Fourier transformation arrays * 
   *****************************************/
  inT   = (fftw_complex*)malloc(T  * sizeof(fftw_complex));
  outT  = (fftw_complex*)malloc(T  * sizeof(fftw_complex));
  if(inT==NULL || outT==NULL) {
    fprintf(stderr, "[get_corr_v3] Error, could not allocate fftw fields\n");
    EXIT(4);
  }

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    memset(conn, 0, 32*VOLUME*sizeof(double));
    memset(conn2[0], 0, 2*Lhp1*T*sizeof(double));
    /***********************
     * read contractions   *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    sprintf(filename, "%s.%.4d", filename_prefix, gid);
    if(format==2 || format==3) {
      status = read_contraction(conn, NULL, filename, 16);
    } else if( format==0) {
      status = read_lime_contraction(conn, filename, 16, 0);
    }
    if(status != 0) {
      // fprintf(stderr, "[get_corr_v3] Error from read_contractions, status was %d\n", status);
      // EXIT(5);
      fprintf(stderr, "[get_corr_v3] Warning, could not read contractions for gid %d, status was %d\n", gid, status);
      continue;
    }

    // byte swap
    if(byte_swap) {
      byte_swap64_v2(conn, 32*VOLUME);
    }

    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v3] time to read contractions %e seconds\n", retime-ratime);
  
    // TEST Pi_mm
/*
    fprintf(stdout, "# [get_corr_v3] Pi_mm\n");
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
    fprintf(stdout, "# [get_corr_v3] Ward identity\n");
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
    ip = 0;
      for(x0=0; x0<T; x0++) {
        ix = _GWI( 5, g_ipt[x0][0][0][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        //
        ix = _GWI(10, g_ipt[x0][0][0][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        //
        ix = _GWI(15, g_ipt[x0][0][0][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
      }
    for(ip=1;ip<Lhp1-1;ip++) {
      for(x0=0; x0<T; x0++) {
        ix = _GWI( 5, g_ipt[x0][ip][0][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        ix = _GWI( 5, g_ipt[x0][LX-ip][0][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        //
        ix = _GWI(10, g_ipt[x0][0][ip][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        ix = _GWI(10, g_ipt[x0][0][LY-ip][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        //
        ix = _GWI(15, g_ipt[x0][0][0][ip], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        ix = _GWI(15, g_ipt[x0][0][0][LZ-ip], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
      }
    }
    ip = L/2;
      for(x0=0; x0<T; x0++) {
        ix = _GWI( 5, g_ipt[x0][ip][0][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        //
        ix = _GWI(10, g_ipt[x0][0][ip][0], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
        //
        ix = _GWI(15, g_ipt[x0][0][0][ip], VOLUME);
        conn2[ip][2*x0  ] += conn[ix  ];
        conn2[ip][2*x0+1] += conn[ix+1];
      }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v3] time to fill correlator %e seconds\n", retime-ratime);
    /*****************************************
     * reverse Fourier transformation
     *****************************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(ip=0;ip<Lhp1;ip++) {
      memcpy((void*)inT, (void*)conn2[ip], 2*T*sizeof(double));
      fftw_one(plan_m_T, inT, outT);
      if(ip==0 || ip==L/2) {
        for(ix=0; ix<T; ix++) {
          conn2[ip][2*ix  ] = outT[ix].re / ( (double)T * 3.);
          conn2[ip][2*ix+1] = outT[ix].im / ( (double)T * 3.);
        }
      } else {
        for(ix=0; ix<T; ix++) {
          conn2[ip][2*ix  ] = outT[ix].re / ( (double)T * 6.);
          conn2[ip][2*ix+1] = outT[ix].im / ( (double)T * 6.);
        }
      }
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v3] time for Fourier transform %e seconds\n", retime-ratime);
  
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(ip=0;ip<Lhp1;ip++) {
      sprintf(filename, "vkvk_corr.p%.2d", ip);
      if(append) {
        ofs = fopen(filename, "a");
      } else {
        ofs = fopen(filename, "w");
      }
      if( ofs == (FILE*)NULL ) {
        fprintf(stderr, "[get_corr_v3] Error, could not open file %s for writing\n", filename);
        EXIT(6);
      }
      x0 = 0;
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[ip][2*x0], 0., gid);
      for(x0=1; x0<T/2; x0++) {
        fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[ip][2*x0], conn2[ip][2*(T-x0)], gid);
      }
      x0 = T / 2;
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 5, 1, x0, conn2[ip][2*x0], 0., gid);
      fclose(ofs);
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v3] time to write correlator %e seconds\n", retime-ratime);
    if(gid==g_gaugeid && !append) append=1;
  }  // of loop on gid

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free_geometry();
  fftw_free(inT);
  fftw_free(outT);
  free(conn);
  free(conn2[0]);
  free(conn2);
  fftw_destroy_plan(plan_m_T);

  fprintf(stdout, "# [get_corr_v3] %s# [get_corr_v3] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "[get_corr_v3] %s[get_corr_v3] end of run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);

}
