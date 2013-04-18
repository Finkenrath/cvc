/****************************************************
 * get_corr_v8.c
 *
 * Thu Apr 18 08:45:32 EEST 2013
 *
 * PURPOSE
 * - originally copied from get_rho_corr_v2.c
 * - 
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
#include "make_q_orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "# [get_corr_v8] Code to recover rho-rho correl.\n");
  fprintf(stdout, "# [get_corr_v8] Usage:    [options]\n");
  fprintf(stdout, "# [get_corr_v8] Options: -v verbose\n");
  fprintf(stdout, "# [get_corr_v8]          -f input filename [default cvc.input]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  int c, mu, nu, status, gid, i;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix;
  int qx, qy, qz;
  double *conn  = NULL;
  double *conn2 = (double*)NULL;
  int verbose = 0;
  char filename[800];
  double ratime, retime;
  FILE *ofs;
  int idx, idy;
  double q[4], wre, wim;
  int test_WI=0;
  unsigned int VOL3;
  size_t offset;

  int qlatt_nclass;
  int *qlatt_id=NULL, *qlatt_count=NULL, **qlatt_rep=NULL, **qlatt_map=NULL;
  double **qlatt_list=NULL;
  int momentum_class[3], momentum_id=-1, *momentum_members=NULL, momentum_num=0;
  char momentum_str[20];

  fftw_complex *inT=NULL, *outT=NULL;

  fftw_plan plan_m_T;

  while ((c = getopt(argc, argv, "Wh?vf:q:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'q':
      sscanf(optarg, "qx%dqy%dqy%d", momentum_class, momentum_class+1, momentum_class+2);
      fprintf(stdout, "# [get_corr_v8] using momentum vector (%d, %d, %d)\n", momentum_class[0], momentum_class[1],
          momentum_class[2]);
      strcpy(momentum_str, optarg);
      break;
    case 'W':
      test_WI = 1;
      fprintf(stdout, "# [get_corr_v8] will test Ward identity in momentum space\n");
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
  fprintf(stdout, "# [get_corr_v8] reading input parameters from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    fprintf(stdout, "# [get_corr_v8] T=%d, LX=%d, LY=%d, LZ=%d\n", T_global, LX, LY, LZ);
    if(g_proc_id==0) fprintf(stderr, "[get_corr_v8] Error, T and L's must be set\n");
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
    fprintf(stderr, "[get_corr_v8] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  VOL3 = LX * LY * LZ;

  status = make_qcont_orbits_3d_parity_avg( &qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
  if(status != 0) {
    fprintf(stderr, "\n[baryon_corr_qdep] Error while creating O_3-lists\n");
    exit(4);
  }
  momentum_id = qlatt_id[g_ipt[0][momentum_class[0]][momentum_class[1]][momentum_class[2]]];
  fprintf(stdout, "# [get_corr_v8] momentum_id set to %d\n", momentum_id);
  momentum_num = qlatt_count[momentum_id];
  momentum_members = (int*)malloc(momentum_num * sizeof(int));
  for(i=0; i<momentum_num; i++) {
    momentum_members[i] = qlatt_map[momentum_id][i];
    fprintf(stdout, "# [get_corr_v8] member no %d : (%d, %d, %d) = %d\n", i,
        qlatt_map[momentum_id][i] / (LY*LZ), (qlatt_map[momentum_id][i]%(LY*LZ))/LZ, qlatt_map[momentum_id][i]%LZ,
        momentum_members[i]);
  }

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(32 * VOLUME, sizeof(double));
  if( (conn==NULL) ) {
    fprintf(stderr, "[get_corr_v8] Error, could not allocate memory for contr. fields\n");
    EXIT(2);
  }

  conn2= (double*)calloc(momentum_num * 6 * T, sizeof(double));
  if( (conn2==NULL) ) {
    fprintf(stderr, "[get_corr_v8] Error, could not allocate memory for corr.\n");
    EXIT(3);
  }

  /*****************************************
   * prepare Fourier transformation arrays * 
   *****************************************/
  inT   = (fftw_complex*)malloc(T * sizeof(fftw_complex));
  outT  = (fftw_complex*)malloc(T * sizeof(fftw_complex));
  if( inT==NULL || outT==NULL ) {
    fprintf(stderr, "[get_corr_v8] Error, could not allocate fftw fields\n");
    EXIT(4);
  }

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    memset(conn, 0, 32*VOLUME*sizeof(double));
    memset(conn2, 0, momentum_num*6*T*sizeof(double));
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
      // fprintf(stderr, "[get_corr_v8] Error from read_contractions, status was %d\n", status);
      // EXIT(5);
      fprintf(stderr, "[get_corr_v8] Warning, could not read contractions for gid %d, status was %d\n", gid, status);
      continue;
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v8] time to read contractions %e seconds\n", retime-ratime);
  
    if(test_WI) {
      fprintf(stdout, "# [get_corr_v8] Ward identity\n");
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
    }
  
    /***********************
     * fill the correlator *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(mu=1; mu<4; mu++) {
      for(x0=0; x0<T; x0++) {
        for(i=0; i<momentum_num; i++) {

          ix = x0 * VOL3 + momentum_members[i];
          idx = _GWI(5*mu,ix,VOLUME);

          idy = (i * 3 + (mu-1) ) * T + x0;

          conn2[2*idy  ] = conn[idx  ];
          conn2[2*idy+1] = conn[idx+1];
        }
      }
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v8] time to fill correlator %e seconds\n", retime-ratime);
   
    // TEST
/*
    fprintf(stdout, "# [get_corr_v8] correlators\n");
    for(i=0; i<momentum_num; i++) {
      for(mu=0;mu<3;mu++) {
        for(x0=0; x0<T; x0++) {
          fprintf(stdout, "\t%6d%3d%3d%25.16e%25.16e\n", i, mu, x0, conn2[2*((i*3+mu)*T+x0)], conn2[2*((i*3+mu)*T+x0)+1]);
    }}}
*/
    /*****************************************
     * reverse Fourier transformation
     *****************************************/
    for(i=0; i<momentum_num; i++) {
    for(mu=0; mu<3; mu++) {
      offset = 2 * ( (i * 3 + mu) * T );
      memcpy((void*)inT, (void*)(conn2 + offset), 2*T*sizeof(double));
      fftw_one(plan_m_T, inT, outT);
      for(ix=0; ix<T; ix++) {
        conn2[offset + 2*ix  ] = outT[ix].re / (double)T;
        conn2[offset + 2*ix+1] = outT[ix].im / (double)T;
      }
    }}
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v8] time for Fourier transform %e seconds\n", retime-ratime);
  
    sprintf(filename, "vkvk_%s_corr.%.4d", momentum_str, gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "[get_corr_v8] Error, could not open file %s for writing\n", filename);
      EXIT(7);
    }
  
    for(i=0; i<momentum_num; i++) {
      qx = qlatt_map[momentum_id][i]/(LX*LY);
      qy = (qlatt_map[momentum_id][i]%(LX*LY))/LZ;
      qz = qlatt_map[momentum_id][i]%LZ;

      for(mu=0; mu<3; mu++) {
        offset = 2 * ( (i * 3 + mu) * T );
        x0 = 0;
        fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d%3d%3d\n", 5, 1, x0, conn2[offset + 2*x0], 0., gid, qx, qy, qz);
            
        for(x0=1; x0<T/2; x0++) {
          fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d%3d%3d\n", 5, 1, x0, conn2[offset + 2*x0], conn2[offset + 2*(T-x0)], gid,
              qx, qy, qz);
        }
        x0 = T / 2;
        fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d%3d%3d\n", 5, 1, x0, conn2[offset + 2*x0], 0., gid, qx, qy, qz);
      }
    }
    fclose(ofs);
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_corr_v8] time to write correlator %e seconds\n", retime-ratime);
  }  // of loop on gid

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free_geometry();
  fftw_free(inT);
  fftw_free(outT);
  free(conn);
  free(conn2);
  fftw_destroy_plan(plan_m_T);
  free(momentum_members);

  finalize_q_orbits(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_rep);
  if(qlatt_map != NULL) {
    free(qlatt_map[0]);
    free(qlatt_map);
  }

  fprintf(stdout, "# [get_corr_v8] %s# [get_corr_v8] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "[get_corr_v8] %s[get_corr_v8] end of run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);

}
