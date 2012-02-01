/****************************************************
 * piq_ft_rmax.c
 *
 * Tue Jun  7 19:05:21 CEST 2011
 *
 * PURPOSE:
 * - analysis programme for \Pi_{\mu\nu}(\hat{q}^2)
 *   * read Pi_mu_nu(q), Fourier transfrom, get Pi_mu_nu(z)
 *   * restrict support of Pi_mu_nu(z) in Fourier transfrom;
 *     use disc of radius rmax
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include <getopt.h>
#include "ifftw.h"

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
#include "make_H3orbits.h"
#include "make_q2orbits.h"
#include "make_cutlist.h"
#include "get_index.h"
#include "read_input_parser.h"
#include "contractions_io.h"


void usage(void) {
  fprintf(stdout, "Program analyse_piq\n");
  fprintf(stdout, "Options:\n");
  fprintf(stdout, "-m INTEGER - mode of usage [default mode = 0]\n");
  fprintf(stdout, "             0 -> all; 1 -> averaging without cuts; 2 -> cylinder/cone cuts\n");
  /* ... */
  exit(0);
}

/***********************************************************************/

int main(int argc, char **argv) {

  int c, i, read_flag=0, iconf, status, dims[4];
  int verbose=0;
  int mode=0, ntag=0;
  int filename_set=0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, mu, nu, ix;
  int y0, y1, y2, y3, sx0, sx1, sx2, sx3;
  int xsrc
  int *q2id=NULL, *qhat2id=NULL, *q4id=NULL, *q6id=NULL, *q8id=NULL;
  int q2count=0, qhat2count=0, *picount=NULL;
  int *workid = NULL;
  int *h4_count=NULL, *h4_id=NULL, h4_nc;
  int *h3_count=NULL, *h3_id=NULL, h3_nc;
  int proj_type = 0;
  int check_WI = 0;
  int check_wi_xspace = 0;
  int force_byte_swap = 0;
  int qhat2id_filename_set=0;
  int rmax_set = 0;
  int *support_site=NULL;
  double *pimn_orig=NULL, *pimn_copy=NULL;
  double *pi=NULL, piq, deltamn, *piavg=NULL;
  double q[4], qhat[4], q2, qhat2, *q2list=NULL, *qhat2list=NULL;
  double **h4_val=NULL, **h3_val=NULL;
  double rmax_value = 0.;
  double dtmp, dtmp2, phase[4];
  char filename[800], qhat2id_filename[400];
  complex w, w1;
  FILE *ofs;

  fftw_complex *in=NULL;
  fftwnd_plan plan_p, plan_m;

  while ((c = getopt(argc, argv, "bh?vawWf:m:n:t:s:r:R:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'a':
      read_flag = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set = 1;
      break;
    case 'm':
      mode = atoi(optarg);
      break;
    case 't':
      proj_type = atoi(optarg);
      break;
    case 'n':
      ntag = atoi(optarg);
      break;
    case 'w':
      check_wi_xspace = 1;
      fprintf(stdout, "\n# [piq_ft_rmax] will check Ward identity in position space and then exit\n");
      break;
    case 'W':
      check_WI = 1;
      fprintf(stdout, "\n# [piq_ft_rmax] will check Ward identity in momentum space\n");
      break;
    case 'b':
      force_byte_swap = 1;
      fprintf(stdout, "\n# [piq_ft_rmax] will enforce byte swap\n");
      break;
    case 's':
      qhat2id_filename_set = 1;
      strcpy(qhat2id_filename, optarg);
      fprintf(stdout, "\n# [piq_ft_rmax] will save qhat2 id in file %s\n", qhat2id_filename);
      break;
    case 'r':
      qhat2id_filename_set = 2;
      sscanf(optarg, "%s %d", qhat2id_filename, &qhat2count );
      fprintf(stdout, "\n# [piq_ft_rmax] will read qhat2 id from file %s, number of orbits is %d\n",
          qhat2id_filename, qhat2count);
      break;
    case 'R':
      rmax_value = atof(optarg);
      rmax_set = 1;
      fprintf(stdout, "\n# [piq_ft_rmax] will use rmax = %e\n", rmax_value);
       break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // global time stamp
  g_the_time = time(NULL);
  if(g_proc_id == 0) fprintf(stdout, "\n# [piq_ft_rmax] using global time_stamp %s", ctime(&g_the_time));
  

  /**************************
   * set the default values *
   **************************/
  if(filename_set==0) strcpy(filename, "analyse.input");
  fprintf(stdout, "# [piq_ft_rmax] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * some checks on the input data *
   *********************************/
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "\n[piq_ft_rmax] Error, T and L's must be set\n");
    usage();
  }

  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  dims[0]=T; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);

  if(init_geometry() != 0) {
    fprintf(stderr, "\n[piq_ft_rmax] ERROR from init_geometry\n");
    return(102);
  }

  geometry();

  /* allocating memory for pimn */
  size_t bytes = 32 * (size_t)VOLUME * sizeof(double);
  pimn_orig = (double*)malloc( bytes );
  if(pimn_orig == NULL) {
    fprintf(stderr, "\n[piq_ft_rmax] Error, could not allocate memory for pimn\n");
    return(101);
  }
  pimn_copy = (double*)malloc( bytes );
  if(pimn_copy == NULL) {
    fprintf(stderr, "\n[piq_ft_rmax] Error, could not allocate memory for pimn\n");
    return(101);
  }

  bytes = 2*(size_t)VOLUME * sizeof(double);
  pi = (double*)malloc( bytes );
  if(pi == NULL) {
    fprintf(stderr, "\n[piq_ft_rmax] Error, could not allocate memory for pi\n");
    return(103);
  }

  /***************************************
   * allocate mem for id lists
   ***************************************/
  q2id = NULL;
  //q2id = (int*)malloc(VOLUME*sizeof(int));
  //if(q2id==(int*)NULL) {
  //  fprintf(stderr, "\n[piq_ft_rmax] could not allocate memory for q2id\n");
  //  return(105);
  //}
/*
  q4id = (int*)malloc(VOLUME*sizeof(int));
  if(q4id==(int*)NULL) {
    fprintf(stderr, "could not allocate memory for q4id\n");
    return(115);
  }

  q6id = (int*)malloc(VOLUME*sizeof(int));
  if(q6id==(int*)NULL) {
    fprintf(stderr, "could not allocate memory for q6id\n");
    return(116);
  }

  q8id = (int*)malloc(VOLUME*sizeof(int));
  if(q8id==(int*)NULL) {
    fprintf(stderr, "could not allocate memory for q8id\n");
    return(117);
  }
*/

  qhat2id = (int*)malloc(VOLUME*sizeof(int));
  if(qhat2id==(int*)NULL) {
    fprintf(stderr, "\n[piq_ft_rmax] Error, could not allocate memory for qhat2id\n");
    return(106);
  }

  workid = (int*)malloc(VOLUME*sizeof(int));
  if(workid==(int*)NULL) {
    fprintf(stderr, "\n[piq_ft_rmax] Error, could not allocate memory for workid\n");
    return(106);
  }

  /***********************************
   * make lists for id, count and val
   ***********************************/
  if( qhat2id_filename_set == 1 ) {
    fprintf(stdout, "\n# [piq_ft_rmax] making qid lists\n");
    if(make_qid_lists(q2id, qhat2id, &q2list, &qhat2list, &q2count, &qhat2count) != 0) {
      fprintf(stderr, "\n[piq_ft_rmax] Error from make_qid_lists\n");
      return(122);
    }

    ofs = fopen(qhat2id_filename, "w");
    if(ofs==NULL) {
      fprintf(stderr, "\n[piq_ft_rmax] Error, could not open file %s for writing\n", qhat2id_filename);
      exit(123);
    }
    if( fwrite(qhat2id, sizeof(int), VOLUME, ofs) != VOLUME || \
        fwrite(qhat2list, sizeof(double), qhat2count, ofs) != qhat2count ) {
      fprintf(stderr, "\n[piq_ft_rmax] Error, could not write prop amount of items to file %s\n", qhat2id_filename);
      exit(127);
    }
    fclose(ofs);
  } else if( qhat2id_filename_set == 2 ) {
    // allocate memory for qhat2list
    if( (qhat2list = (double*)malloc(qhat2count*sizeof(double))) == NULL ) {
      fprintf(stderr, "\nError, could not alloc memory for qhat2list\n");
      exit(125);
    }
    if( (ofs = fopen(qhat2id_filename, "r")) == NULL ) {
      fprintf(stderr, "\n[piq_ft_rmax] Error, could not open file %s for reading\n", qhat2id_filename);
      exit(124);
    }
    if( fread(qhat2id, sizeof(int), VOLUME, ofs) != VOLUME ) {
      fprintf(stderr, "\n[piq_ft_rmax] Error, could not read prop amount of items from file %s\n", qhat2id_filename);
      exit(126);
    }
    if( fread(qhat2list, sizeof(double), qhat2count, ofs) != qhat2count ) {
      fprintf(stderr, "\n[piq_ft_rmax] Error, could not read prop amount of items from file %s\n", qhat2id_filename);
      exit(126);
    }
    fclose(ofs);
  } else {
    fprintf(stdout, "\n# [piq_ft_rmax] making qid lists\n");
    if(make_qid_lists(q2id, qhat2id, &q2list, &qhat2list, &q2count, &qhat2count) != 0) {
      fprintf(stderr, "\n[piq_ft_rmax] Error from make_qid_lists\n");
      return(122);
    }
  }
  fprintf(stdout, "\n# [piq_ft_rmax] number of qhat2 orbits = %d\n", qhat2count);

  if(mode==0 || mode==3) {
    fprintf(stdout, "\n# [piq_ft_rmax] make H3 orbits\n");  
    if(make_H3orbits(&h3_id, &h3_count, &h3_val, &h3_nc) != 0) return(123);
  }

  if(mode==0 || mode==4) {
    fprintf(stdout, "\n# [piq_ft_rmax] make H4 orbits\n");  
    if(make_H4orbits(&h4_id, &h4_count, &h4_val, &h4_nc) != 0) return(124);
  }
  fprintf(stdout, "\n# [piq_ft_rmax] finished making qid lists\n");

  // make the support list
  if( rmax_set ) {
    support_site = (int*)malloc(VOLUME*sizeof(int));
    if(support_site == NULL) {
      fprintf(stdout, "\n[piq_ft_rmax] Error, could not alloc support_site\n");
      exit(102);
    }
    dtmp2 = rmax_value * rmax_value;
    for(x0=0;x0<T;x0++) {
      y0 = x0<=T/2 ? x0 : T-x0;
    for(x1=0;x1<LX;x1++) {
      y1 = x1<=LX/2 ? x1 : LX-x1;
    for(x2=0;x2<LY;x2++) {
      y2 = x2<=LY/2 ? x2 : LY-x2;
    for(x3=0;x3<LZ;x3++) {
      y3 = x3<=LZ/2 ? x3 : LZ-x3;
      // dtmp = x0*x0 + x1*x1 + x2*x2 + x3*x3;
      dtmp = y0*y0 + y1*y1 + y2*y2 + y3*y3;
      ix = g_ipt[x0][x1][x2][x3];
      support_site[ix] = dtmp < dtmp2 ? 1 : 0;
      // check
      // fprintf(stdout, "%3d%3d%3d%3d%6d\n", x0, x1, x2, x3, support_site[ix]);
    }}}}
  }

  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {
    exit(4);
  }

/*******************************************
 * loop on the configurations
 *******************************************/
for(iconf=g_gaugeid; iconf<=g_gaugeid2; iconf+=g_gauge_step) {

  Nconf = iconf;
  fprintf(stdout, "\n# [piq_ft_rmax] iconf = %d\n", iconf);

  /*******************************
   * calculate source coordinates
   * - source_location might have to come from different source
   *   than g_source_location
   *******************************/
  sx0 = g_source_location/(LX*LY*LZ);
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  fprintf(stdout, "\n# [piq_ft_rmax] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);

  /****************
   * reading pimn *
   ****************/
  if(format != 2) {
    //sprintf(filename, "%s.%.4d.%.4d", filename_prefix, iconf, Nsave);
    sprintf(filename, "%s.%.4d", filename_prefix, iconf);
  } else {
    sprintf(filename, "%s", filename_prefix);
  }
  fprintf(stdout, "# [piq_ft_rmax] Reading data from file %s\n", filename);
  status = read_lime_contraction(pimn_orig, filename, 16, 0);
  if(status != 0) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on reading of pimn, status was %d\n", status);
    continue;
    //exit(123);
  }

  if(force_byte_swap) {
    fprintf(stdout, "\n# [piq_ft_rmax] starting byte swap ...");
    fflush(stdout);
    byte_swap64_v2(pimn_orig, 32*(unsigned int)VOLUME);
    fprintf(stdout, " done\n");
    fflush(stdout);
  }

  /**********************************************
   * test the Ward identity in momentum space
   **********************************************/
  if(check_WI==1) {
    ofs = fopen("WI_check", "w");
    for(x0=0; x0<T; x0++) {
      q[0] = 2. * sin( M_PI * (double)x0 / (double)T );
    for(x1=0; x1<LX; x1++) {
      q[1] = 2. * sin( M_PI * (double)x1 / (double)LX );
    for(x2=0; x2<LY; x2++) {
      q[2] = 2. * sin( M_PI * (double)x2 / (double)LY );
    for(x3=0; x3<LZ; x3++) {
      q[3] = 2. * sin( M_PI * (double)x3 / (double)LZ );
      ix = g_ipt[x0][x1][x2][x3];
      fprintf(ofs, "# qt=%.2d, qx=%.2d, qy=%.2d, qz=%.2d\n", x0, x1, x2, x3);
      for(mu=0; mu<4; mu++) {
        w.re = q[0] * pimn_orig[_GWI(0*4+mu,ix,VOLUME)  ] + q[1] * pimn_orig[_GWI(1*4+mu,ix,VOLUME)  ]
             + q[2] * pimn_orig[_GWI(2*4+mu,ix,VOLUME)  ] + q[3] * pimn_orig[_GWI(3*4+mu,ix,VOLUME)  ];
        w.im = q[0] * pimn_orig[_GWI(0*4+mu,ix,VOLUME)+1] + q[1] * pimn_orig[_GWI(1*4+mu,ix,VOLUME)+1]
             + q[2] * pimn_orig[_GWI(2*4+mu,ix,VOLUME)+1] + q[3] * pimn_orig[_GWI(3*4+mu,ix,VOLUME)+1];
        fprintf(ofs, "%3d%25.16e%25.16e\n", mu, w.re, w.im);
      }
    }}}}
    fclose(ofs);
  }

  /**********************************************
   * backward Fourier transformation
   * - get results in position space
   **********************************************/
/*
  for(mu=0; mu<3; mu++) {
  for(nu=mu+1; nu<4; nu++) {
    for(x0=0; x0<T; x0++) {
      phase[0] =  (double)(x0) * M_PI / (double)T;
    for(x1=0; x1<LX; x1++) {
      phase[1] =  (double)(x1) * M_PI / (double)LX;
    for(x2=0; x2<LY; x2++) {
      phase[2] =  (double)(x2) * M_PI / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      phase[3] =  (double)(x3) * M_PI / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re =  cos( phase[mu] - phase[nu] );
      w.im = -sin( phase[mu] - phase[nu] );
      _co_eq_co_ti_co(&w1,(complex*)( pimn_orig + _GWI(4*mu+nu,ix,VOLUME)), &w);
      pimn_orig[_GWI(4*mu+nu,ix,VOLUME)  ] = w1.re;
      pimn_orig[_GWI(4*mu+nu,ix,VOLUME)+1] = w1.im;

      w.re =  cos( phase[nu] - phase[mu] );
      w.im = -sin( phase[nu] - phase[mu] );
      _co_eq_co_ti_co(&w1,(complex*)( pimn_orig + _GWI(4*nu+mu,ix,VOLUME) ), &w);
      pimn_orig[_GWI(4*nu+mu,ix,VOLUME)  ] = w1.re;
      pimn_orig[_GWI(4*nu+mu,ix,VOLUME)+1] = w1.im;
    }}}}
  }}  // of mu and nu
*/
  for(mu=0;mu<4;mu++) {
    memcpy((void*)in, (void*)&pimn_orig[_GWI(5*mu,0,VOLUME)], 2*VOLUME*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy((void*)&pimn_orig[_GWI(5*mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

  /**********************************************
   * test the Ward identity in position space
   **********************************************/
/*
  if(check_wi_xspace == 1) {
    ofs = fopen("WI_check_x", "w");
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      fprintf(ofs, "# t=%.2d, x=%.2d, y=%.2d, z=%.2d\n", x0, x1, x2, x3);
      for(mu=0; mu<4; mu++) {
        w.re = pimn_orig[_GWI(0*4+mu,ix,VOLUME)  ] + pimn_orig[_GWI(1*4+mu,ix,VOLUME)  ]
             + pimn_orig[_GWI(2*4+mu,ix,VOLUME)  ] + pimn_orig[_GWI(3*4+mu,ix,VOLUME)  ]
             - pimn_orig[_GWI(0*4+mu,g_idn[ix][0],VOLUME)  ] - pimn_orig[_GWI(1*4+mu,g_idn[ix][1],VOLUME)  ]
             - pimn_orig[_GWI(2*4+mu,g_idn[ix][2],VOLUME)  ] - pimn_orig[_GWI(3*4+mu,g_idn[ix][3],VOLUME)  ];
            
        w.im = pimn_orig[_GWI(0*4+mu,ix,VOLUME)+1] + pimn_orig[_GWI(1*4+mu,ix,VOLUME)+1]
             + pimn_orig[_GWI(2*4+mu,ix,VOLUME)+1] + pimn_orig[_GWI(3*4+mu,ix,VOLUME)+1]
             - pimn_orig[_GWI(0*4+mu,g_idn[ix][0],VOLUME)+1] - pimn_orig[_GWI(1*4+mu,g_idn[ix][1],VOLUME)+1]
             - pimn_orig[_GWI(2*4+mu,g_idn[ix][2],VOLUME)+1] - pimn_orig[_GWI(3*4+mu,g_idn[ix][3],VOLUME)+1];
        fprintf(ofs, "%3d%25.16e%25.16e\n", mu, w.re, w.im);
      }
    }}}}
    fclose(ofs);
    continue; 
  }
*/

  /**********************************************
   * forward Fourier transformation
   **********************************************/
  if(rmax_set) {
    for(mu=0;mu<4;mu++) {
      for(ix=0;ix<VOLUME;ix++) {
        pimn_copy[_GWI(5*mu,ix,VOLUME)  ] = (double)support_site[ix] * pimn_orig[_GWI(5*mu,ix,VOLUME)  ];
        pimn_copy[_GWI(5*mu,ix,VOLUME)+1] = (double)support_site[ix] * pimn_orig[_GWI(5*mu,ix,VOLUME)+1];
      }
    }
  } else {
    memcpy(pimn_copy, pimn_orig, 32*VOLUME*sizeof(double));
  }
  
  for(mu=0;mu<4;mu++) {
    memcpy((void*)in, (void*)&pimn_copy[_GWI(5*mu,0,VOLUME)], 2*VOLUME*sizeof(double));
    fftwnd_one(plan_p, in, NULL);
    memcpy((void*)&pimn_copy[_GWI(5*mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

/*
  for(mu=0; mu<3; mu++) {
  for(nu=mu+1; nu<4; nu++) {
    for(x0=0; x0<T; x0++) {
      phase[0] =  (double)(x0) * M_PI / (double)T;
    for(x1=0; x1<LX; x1++) {
      phase[1] =  (double)(x1) * M_PI / (double)LX;
    for(x2=0; x2<LY; x2++) {
      phase[2] =  (double)(x2) * M_PI / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      phase[3] =  (double)(x3) * M_PI / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re =  cos( phase[mu] - phase[nu] );
      w.im =  sin( phase[mu] - phase[nu] );
      _co_eq_co_ti_co(&w1,(complex*)( pimn_orig + _GWI(4*mu+nu,ix,VOLUME)), &w);
      pimn_orig[_GWI(4*mu+nu,ix,VOLUME)  ] = w1.re;
      pimn_orig[_GWI(4*mu+nu,ix,VOLUME)+1] = w1.im;

      w.re =  cos( phase[nu] - phase[mu] );
      w.im =  sin( phase[nu] - phase[mu] );
      _co_eq_co_ti_co(&w1,(complex*)( pimn_orig + _GWI(4*nu+mu,ix,VOLUME) ), &w);
      pimn_orig[_GWI(4*nu+mu,ix,VOLUME)  ] = w1.re;
      pimn_orig[_GWI(4*nu+mu,ix,VOLUME)+1] = w1.im;
    }}}}
  }}  // of mu and nu
*/

  /**************************
   * calculate pi from pimn:
   **************************/
  fprintf(stdout, "\n# [piq_ft_rmax] calculate pi from pimn\n");
  for(x0=0; x0<T; x0++) {
    q[0]    = 2. * sin( M_PI / (double)T  * (double)(x0) );
  for(x1=0; x1<LX; x1++) {
    q[1]    = 2. * sin( M_PI / (double)LX * (double)(x1) );
  for(x2=0; x2<LY; x2++) {
    q[2]    = 2. * sin( M_PI / (double)LY * (double)(x2) );
  for(x3=0; x3<LZ; x3++) {
    q[3]    = 2. * sin( M_PI / (double)LZ * (double)(x3) );
    ix = g_ipt[x0][x1][x2][x3];
    q2    = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];

    pi[2*ix  ] = 0.;
    pi[2*ix+1] = 0.;

    for(mu=0; mu<4; mu++) {
      pi[2*ix  ] += pimn_copy[_GWI(5*mu,ix,VOLUME)  ];
      pi[2*ix+1] += pimn_copy[_GWI(5*mu,ix,VOLUME)+1];
    }
    
    q2 = q[0]*q[0] +  q[1]*q[1] + q[2]*q[2] + q[3]*q[3];

    if(q2==0.) {
      pi[2*ix  ] = 0.;
      pi[2*ix+1] = 0.;
    } else {
      pi[2*ix  ] /= -3.*q2 * (double)VOLUME; 
      pi[2*ix+1] /= -3.*q2 * (double)VOLUME;
    }
  }}}}
  fprintf(stdout, "\n# [piq_ft_rmax] finished calculating pi\n");

/************************************************
 * mode 99: only write full q-dep pi
 ************************************************/
if(mode== 99) {
  sprintf(filename, "pi.%.2d.%.4d", 99, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on opening file %s\n", filename);
    return(110);
  }
  for(i=0; i<VOLUME; i++) {
   fprintf(ofs, "%25.16e\t%25.16e\n", pi[2*i], pi[2*i+1]);
  }
  fclose(ofs); 
}

/************************************************
 * mode 1: average qhat2 orbits
 ************************************************/
if(mode==0 || mode==1) {

   fprintf(stdout, "\n# [piq_ft_rmax] averaging over qhat2-orbits\n");

  if( (piavg = (double*)malloc(2*qhat2count*sizeof(double))) == (double*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on using malloc\n");
    return(108);
  }
  if( (picount = (int*)malloc(qhat2count*sizeof(int))) == (int*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on using malloc\n");
    return(109);
  }
  for(i=0; i<2*qhat2count; i++) piavg[i]   = 0.;
  for(i=0; i<  qhat2count; i++) picount[i] = 0;
  for(ix=0; ix<VOLUME; ix++) {
    piavg[2*qhat2id[ix]  ] += pi[2*ix  ];
    piavg[2*qhat2id[ix]+1] += pi[2*ix+1];
    picount[qhat2id[ix]]++;
  }
  for(i=0; i<  qhat2count; i++) {
    piavg[2*i  ] /= picount[i];
    piavg[2*i+1] /= picount[i];
  }

  sprintf(filename, "pi%1d.%.2d.%.4d", proj_type, 1, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on opening file %s\n", filename);
    return(110);
  }
  for(i=0; i<  qhat2count; i++) {
   if(qhat2list[i] > _Q2EPS) {
     fprintf(ofs, "%21.12e\t%25.16e%25.16e%6d\n", qhat2list[i], 
       piavg[2*i], piavg[2*i+1], picount[i]);
     }
  }
  fclose(ofs); 
  free(piavg);
  free(picount);

}  // of if mode = 0/1

/************************************************
 * mode 2: qhat2 orbits with cylinder / cone cut
 ************************************************/
if(mode==0 || mode==2) {
  /*********************
   * apply cuts 
   *********************/
  for(ix=0; ix<VOLUME; ix++) workid[ix]=qhat2id[ix];

  if( make_cutid_list(workid, g_cutdir, g_cutradius, g_cutangle) != 0 ) return(125);

  /* average over qhat2-orbits */
  if( (piavg = (double*)malloc(2*qhat2count*sizeof(double))) == (double*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on using malloc\n");
    return(111);
  }
  if( (picount = (int*)malloc(qhat2count*sizeof(int))) == (int*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on using malloc\n");
    return(112);
  }
  for(i=0; i<2*qhat2count; i++) piavg[i]   = 0.;
  for(i=0; i<  qhat2count; i++) picount[i] = 0;
  for(ix=0; ix<VOLUME; ix++) {
    if(workid[ix]!=-1) {
      piavg[2*workid[ix]  ] += pi[2*ix  ];
      piavg[2*workid[ix]+1] += pi[2*ix+1];
      picount[workid[ix]]++;
    }
  }
  for(i=0; i<  qhat2count; i++) {
    if(picount[i]>0) {
      piavg[2*i  ] /= picount[i];
      piavg[2*i+1] /= picount[i];
    }
  }

  sprintf(filename, "pi%1d.%.2d.%.4d", proj_type, 2, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on opening file %s\n", filename);
    return(113);
  }
  for(i=0; i<  qhat2count; i++) {
   if(picount[i]>0  && qhat2list[i]>_Q2EPS)
     fprintf(ofs, "%21.12e\t%25.16e%25.16e%6d\n", qhat2list[i], 
       piavg[2*i], piavg[2*i+1], picount[i]);
  }
  fclose(ofs);
  free(piavg);
  free(picount);

  sprintf(filename, "pi%1d.%.2d.%.4d.info", proj_type, 2, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on opening file %s\n", filename);
    return(114);
  }
  fprintf(ofs, "Nconf:\t%.4d\navg:\t%s\nradius:\t%12.7e\nangle:\t%12.7e\n"\
               "dir:\t%3d%3d%3d%3d\n", Nconf, "qhat2", g_cutradius, g_cutangle, 
               g_cutdir[0], g_cutdir[1], g_cutdir[2], g_cutdir[3]);
  fclose(ofs);

}  // of mode = 0/2 

/**************************
 * mode 4: H4 orbits
 **************************/
if(mode==0 || mode==4) {

  for(ix=0; ix<VOLUME; ix++) workid[ix]=h4_id[ix];
  if( make_cutid_list(workid, g_cutdir, g_cutradius, g_cutangle) != 0 ) return(125);

  /**************************
   * average over orbits 
   **************************/
  if( (piavg = (double*)malloc(2 * h4_nc * sizeof(double))) == (double*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on using malloc\n");
    return(111);
  }
  if( (picount = (int*)malloc(h4_nc * sizeof(int))) == (int*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on using malloc\n");
    return(112);
  }
  for(i=0; i<2*h4_nc; i++) piavg[i] = 0.;
  for(i=0; i<h4_nc; i++) picount[i] = 0;

  for(ix=0; ix<VOLUME; ix++) {
    if(workid[ix] != -1) {
      piavg[2*workid[ix]  ] += pi[2*ix  ];
      piavg[2*workid[ix]+1] += pi[2*ix+1];
      picount[workid[ix]]++;
    }
  } 
  for(ix=0; ix<h4_nc; ix++) {
    if(picount[ix]>0) {
      piavg[2*ix  ] /= picount[ix];
      piavg[2*ix+1] /= picount[ix];
    }
    else {
      piavg[2*ix  ] = 0.;
      piavg[2*ix+1] = 0.;
    }
  }

  sprintf(filename, "pi%1d.%.2d.%.4d", proj_type, 4, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on opening file %s\n", filename);
    return(128);
  }
  for(i=0; i<h4_nc; i++) {
   if(picount[i]>0 && h4_val[0][i]>_Q2EPS)
     fprintf(ofs, "%21.12e%21.12e%21.12e%21.12e%25.16e%25.16e%6d\n", \
       h4_val[0][i], h4_val[1][i], h4_val[2][i], h4_val[3][i],       \
       piavg[2*i], piavg[2*i+1], picount[i]);
  }
  fclose(ofs);

  free(piavg);
  free(picount);

  sprintf(filename, "pi%1d.%.2d.%.4d.info", proj_type, 4, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on opening file %s\n", filename);
    return(129);
  }
  fprintf(ofs, "Nconf:\t%.4d\navg:\t%s\nradius:\t%12.7e\nangle:\t%12.7e\n"\
               "dir:\t%3d%3d%3d%3d\n", Nconf, "h4", g_cutradius, g_cutangle,
               g_cutdir[0], g_cutdir[1], g_cutdir[2], g_cutdir[3]);
  fclose(ofs);
 
}  // of mode = 0/4
  
/**************************
 * mode 3: H3 orbits
 **************************/
if(mode==0 || mode==3) {

  for(ix=0; ix<VOLUME; ix++) workid[ix]=h3_id[ix];
  if( make_cutid_list(workid, g_cutdir, g_cutradius, g_cutangle) != 0 ) return(125);

  /**************************
   * average over orbits 
   **************************/
  if( (piavg = (double*)malloc(2 * h3_nc * sizeof(double))) == (double*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on using malloc\n");
    return(111);
  }
  if( (picount = (int*)malloc(h3_nc * sizeof(int))) == (int*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on using malloc\n");
    return(112);
  }
  for(i=0; i<2*h3_nc; i++) piavg[i] = 0.;
  for(i=0; i<h3_nc; i++) picount[i] = 0;

  for(ix=0; ix<VOLUME; ix++) {
    if(workid[ix] != -1) {
      piavg[2*workid[ix]  ] += pi[2*ix  ];
      piavg[2*workid[ix]+1] += pi[2*ix+1];
      picount[workid[ix]]++;
    }
  } 
  for(ix=0; ix<h3_nc; ix++) {
    if(picount[ix]>0) {
      piavg[2*ix  ] /= picount[ix];
      piavg[2*ix+1] /= picount[ix];
    }
    else {
      piavg[2*ix  ] = 0.;
      piavg[2*ix+1] = 0.;
    }
  }

  sprintf(filename, "pi%1d.%.2d.%.4d", proj_type, 3, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on opening file %s\n", filename);
    return(128);
  }
  for(i=0; i<h3_nc; i++) {
   if(picount[i]>0 && h3_val[0][i]>_Q2EPS)
     fprintf(ofs, "%21.12e%21.12e%21.12e%21.12e%25.16e%25.16e%6d\n", \
       h3_val[0][i], h3_val[1][i], h3_val[2][i], h3_val[3][i],       \
       piavg[2*i], piavg[2*i+1], picount[i]);
  }
  fclose(ofs);

  free(piavg);
  free(picount);

  sprintf(filename, "pi%1d.%.2d.%.4d.info", proj_type, 3, Nconf);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "\n[piq_ft_rmax] Error on opening file %s\n", filename);
    return(129);
  }
  fprintf(ofs, "Nconf:\t%.4d\navg:\t%s\nradius:\t%12.7e\nangle:\t%12.7e\n"\
               "dir:\t%3d%3d%3d%3d\n", Nconf, "h3", g_cutradius, g_cutangle,
               g_cutdir[0], g_cutdir[1], g_cutdir[2], g_cutdir[3]);
  fclose(ofs);
 
}  // of mode == 0/3

} // of loop on iconf


  /**************************
   * free and finalize
   **************************/

  if(h3_val != NULL) {
    if(*h3_val != NULL) free(*h3_val);
    free(h3_val);
  }
  if(h4_val != NULL) {
    if(*h4_val != NULL) free(*h4_val);
    free(h4_val);
  }
  if(h3_count     != NULL) free(h3_count);
  if(h4_count     != NULL) free(h4_count);
  if(h3_id        != NULL) free(h3_id);
  if(h4_id        != NULL) free(h4_id);
  if(pimn_orig    != NULL) free(pimn_orig);
  if(pimn_copy    != NULL) free(pimn_copy);
  if(pi           != NULL) free(pi);
  if(q2id         != NULL) free(q2id);
  if(qhat2id      != NULL) free(qhat2id);
  if(q2list       != NULL) free(q2list);
  if(qhat2list    != NULL) free(qhat2list);
  if(support_site != NULL) free(support_site);

  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);
  if(in != NULL) free(in);


  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [piq_ft_rmax] %s# [piq_ft_rmax] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [piq_ft_rmax] %s# [piq_ft_rmax] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
