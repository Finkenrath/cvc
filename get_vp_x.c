/****************************************************
 * get_vp_x.c
 *
 * Mo 17. Dez 10:56:07 EET 2012
 *
 * PURPOSE
 * - originally copied from get_corr_v2.c
 * - Fourier transform momentum space vacuum polarization
 *   and write it to file
 * DONE
 * TODO
 * CHANGES
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
  fprintf(stdout, "# [get_vp_x] Code to recover rho-rho correl.\n");
  fprintf(stdout, "# [get_vp_x] Usage:    [options]\n");
  fprintf(stdout, "# [get_vp_x] Options: -v verbose\n");
  fprintf(stdout, "# [get_vp_x]          -f input filename [default cvc.input]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  int c, mu, nu, status, gid, imunu;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int source_location, have_source_flag = 0;
  int x0, x1, x2, x3, ix;
  // int sx0, sx1, sx2, sx3;
  double *conn  = NULL;
  int verbose = 0;
  int dims[4], itmp[2];
  char filename[800];
  double ratime, retime;
  FILE *ofs;
  double q[4], wre, wim;
  complex w, w1;
  double bc_phase[4], phase, fnorm;

  fftw_complex *inT=NULL;

  fftwnd_plan plan_p;

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
  fprintf(stdout, "# [get_vp_x] reading input parameters from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    fprintf(stdout, "# [get_vp_x] T=%d, LX=%d, LY=%d, LZ=%d\n", T_global, LX, LY, LZ);
    if(g_proc_id==0) fprintf(stderr, "[get_vp_x] Error, T and L's must be set\n");
    usage();
  }

  // initialize MPI parameters
  mpi_init(argc, argv);

  // initialize fftw, create plan with FFTW_BACKWARD
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
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
    fprintf(stderr, "[get_vp_x] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(32 * VOLUME, sizeof(double));
  if( (conn==NULL) ) {
    fprintf(stderr, "[get_vp_x] Error, could not allocate memory for contr. fields\n");
    EXIT(2);
  }

  /*****************************************
   * prepare Fourier transformation arrays * 
   *****************************************/
  inT   = (fftw_complex*)malloc(VOLUME  * sizeof(fftw_complex));
  if( inT==NULL) {
    fprintf(stderr, "[get_vp_x] Error, could not allocate fftw fields\n");
    EXIT(4);
  }

//  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    memset(conn, 0, 32*VOLUME*sizeof(double));
    /***********************
     * read contractions   *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    //sprintf(filename, "%s.%.4d", filename_prefix, gid);
    sprintf(filename, "%s", filename_prefix);
    //if(format==2 || format==3) {
    //  status = read_contraction(conn, NULL, filename, 16);
    //} else if( format==0) {
    //  status = read_lime_contraction(conn, filename, 16, 0);
    //}
    //if(status != 0) {
    //  fprintf(stderr, "[get_vp_x] Warning, could not read contractions for gid %d, status was %d\n", gid, status);
    //  continue;
    //}
    // read in ascii format
    ofs = fopen(filename, "r");
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      imunu = 0;
      for(mu=0;mu<4;mu++) {
      for(nu=0;nu<4;nu++) {
        //fscanf(ofs, "%d%d%lf%lf", itmp, itmp+1, conn+_GWI(imunu,ix,VOLUME), conn+_GWI(imunu,ix,VOLUME)+1);
        fscanf(ofs, "%d%lf%lf", itmp, conn+_GWI(imunu,ix,VOLUME), conn+_GWI(imunu,ix,VOLUME)+1);
        imunu++;
      }}
    }}}}
    fclose(ofs);
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_vp_x] time to read contractions %e seconds\n", retime-ratime);
  
    // TEST
    // - write Pi_mn
    fprintf(stdout, "# [get_vp_x] Pi_mn\n");
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      for(nu=0;nu<16;nu++) {
        wre = conn[_GWI(nu,ix,VOLUME)];
        wim = conn[_GWI(nu,ix,VOLUME)+1];
        fprintf(stdout, "\t%3d%3d%3d%3d%3d%3d%16.7e%16.7e\n", x0, x1, x2, x3, nu/4, nu%4, wre, wim);
      }
    }}}}

    // TEST
    // - Ward Identity
    fprintf(stdout, "# [get_vp_x] Ward identity\n");
    for(x0=0; x0<T; x0++) {
      q[0] = 2. * sin(M_PI * (double)x0 / (double)T);
    for(x1=0; x1<LX; x1++) {
      q[1] = 2. * sin(M_PI * ((double)x1+BCangle[1]*0.5) / (double)LX);
    for(x2=0; x2<LY; x2++) {
      q[2] = 2. * sin(M_PI * ((double)x2+BCangle[2]*0.5) / (double)LY);
    for(x3=0; x3<LZ; x3++) {
      q[3] = 2. * sin(M_PI * ((double)x3+BCangle[3]*0.5) / (double)LZ);
      ix = g_ipt[x0][x1][x2][x3];
      for(nu=0;nu<4;nu++) {
        wre =   q[0] * conn[_GWI(4*0+nu,ix,VOLUME)] + q[1] * conn[_GWI(4*1+nu,ix,VOLUME)] \
              + q[2] * conn[_GWI(4*2+nu,ix,VOLUME)] + q[3] * conn[_GWI(4*3+nu,ix,VOLUME)];
        wim =   q[0] * conn[_GWI(4*0+nu,ix,VOLUME)+1] + q[1] * conn[_GWI(4*1+nu,ix,VOLUME)+1] \
              + q[2] * conn[_GWI(4*2+nu,ix,VOLUME)+1] + q[3] * conn[_GWI(4*3+nu,ix,VOLUME)+1];
        fprintf(stdout, "\t%3d%3d%3d%3d%3d%16.7e%16.7e\n", nu, x0, x1, x2, x3, wre, wim);
      }
    }}}}
  
    // add phase factors
    for(x0=0; x0<T; x0++) {
      q[0] = (double)x0 / (double)T;
    for(x1=0; x1<LX; x1++) {
      q[1] = (double)x1 / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = (double)x2 / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = (double)x3 / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        if(mu==nu) continue;
        w.re = cos( M_PI * ( q[mu]-q[nu] ) );
        w.im = sin( M_PI * ( q[mu]-q[nu] ) );
        w1.re = conn[_GWI(4*mu+nu,ix,VOLUME)  ];
        w1.im = conn[_GWI(4*mu+nu,ix,VOLUME)+1];
        _co_eq_co_ti_co((complex*)(conn+_GWI(4*mu+nu,ix,VOLUME)), &w, &w1);
      }}
    }}}}

    // Fourier transformation
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(mu=0; mu<16;mu++) {
      memcpy((void*)inT, (void*)(conn+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
      fftwnd_one(plan_p, inT, NULL);
      memcpy((void*)(conn+_GWI(mu,0,VOLUME)), (void*)inT, 2*VOLUME*sizeof(double));
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_vp_x] time for Fourier transform %e seconds\n", retime-ratime);
  
    bc_phase[0] = 0.;
    bc_phase[1] = M_PI * BCangle[1] / (double)LX;
    bc_phase[2] = M_PI * BCangle[2] / (double)LY;
    bc_phase[3] = M_PI * BCangle[3] / (double)LZ;

    // add boundary condition phase factor
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        phase = x1 * bc_phase[1] + x2 * bc_phase[2] + x3 * bc_phase[3] + 0.5*(bc_phase[mu]-bc_phase[nu]);
        w.re = cos( phase );
        w.im = sin( phase );
        w1.re = conn[_GWI(4*mu+nu,ix,VOLUME)  ];
        w1.im = conn[_GWI(4*mu+nu,ix,VOLUME)+1];
        _co_eq_co_ti_co((complex*)(conn+_GWI(4*mu+nu,ix,VOLUME)), &w, &w1);
      }}
    }}}}
    
    // normalization
    fnorm = 1. / (double)VOLUME;
    for(mu=0; mu<16;mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        conn[_GWI(mu,ix,VOLUME)  ] *= fnorm;
        conn[_GWI(mu,ix,VOLUME)+1] *= fnorm;
      }
    }
    ratime = (double)clock() / CLOCKS_PER_SEC;
    sprintf(filename, "%s.ft", filename_prefix);
    write_contraction(conn, NULL, filename, 16, 2, 0);
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [get_vp_x] time to write correlator %e seconds\n", retime-ratime);
    //  }  // of loop on gid

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free_geometry();
  fftw_free(inT);
  free(conn);
  fftwnd_destroy_plan(plan_p);

  fprintf(stdout, "# [get_vp_x] %s# [get_vp_x] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "[get_vp_x] %s[get_vp_x] end of run\n", ctime(&g_the_time));
  fflush(stderr);
  return(0);
}
