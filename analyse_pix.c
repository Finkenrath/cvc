/****************************************************
 * analyse_pix.c
 *
 * Fri Nov 20 09:50:41 CET 2009
 *
 * PURPOSE:
 * - implementation of different analysis modes:
 *   (1) for an r-window r\in [R_min, R_max]
 *   (2) for on-axis r, averaged over 3 space dir.
 *   (3) average over O_h orbits (perm. of the r_i)
 *   (-1) as (1), but with data from model function
 *   (-2) as (2), but with data from model function
 *   (-3) as (3), but with data from model function
 * - in all cases t treated seperately (i.e. considering
 *   pairs (t,r))
 * TODO:
 * - retest the generation of orbits
 * - test averaging over shifts and over orbits
 * DONE:
 * CHANGES:
 * - the volume in pos. space to average over for a certain
 *   shift x=(x0,x1,x2,x3) is given by avgT and avgL
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
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
#include "make_q2orbits.h"
#include "make_H3orbits.h"
#include "read_input_parser.h"
#include "pidisc_model.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to analyse the contractions in position space\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

int mu_nu_comb[4][2] = {{0,0},{0,1},{1,2},{1,1}};
int mu_nu_tab[16] = {0, 1, 1, 1, 1, 3, 2, 2, 1, 2, 3, 2, 1, 2, 2, 3};


int main(int argc, char **argv) {
  
  int c, i,j, mu, nu, gid, mode, status, imunu;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, idx;
  int xx0, xx1, xx2, xx3;
  int y0, y1, y2, y3, iy;
  int iy1, iy2, iy3, iz1, iz2, iz3;
  int z0, z1, z2, z3, iz, iiz[3];
  int Thp1, nclass;
  int y0min, y0max, y1min, y1max, y2min, y2max, y3min, y3max;
  int Lhalf=0, Lhp1=0;
  int model_type = 0;
  int *oh_count=(int*)NULL, *oh_id=(int*)NULL, oh_nc=0;
  int *rid = (int*)NULL, rcount, *picount=(int*)NULL;
  double *rlist = (double*)NULL;
  double *disc  = (double*)NULL;
  double *pir   = (double*)NULL;
  double **oh_val = (double**)NULL;
  char filename[100];
  double ratime, retime;
  double mrho=0., dcoeffre=0., dcoeffim=0.;
  complex w;
  FILE *ofs;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
  MPI_Status mstatus;
#else
  fftwnd_plan plan_p, plan_m;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?gf:m:D:M:t:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'm':
      mode=atoi(optarg);
      break;
    case 'M':
      mrho = atof(optarg);
      fprintf(stdout, "# rho meson mass set to %s\n", optarg);
      break;
    case 'D':
      dcoeffre = atof(optarg);
      fprintf(stdout, "# real part of d-coeff. set to %s\n", optarg);
      break;
    case 'd':
      dcoeffim = atof(optarg);
      fprintf(stdout, "# imaginary part of d-coeff. set to %s\n", optarg);
      break;
    case 't':
      model_type = atoi(optarg);
      fprintf(stdout, "# model type set to set to %d\n", model_type);
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "analyse.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);
#ifdef MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

  /* initialize fftw */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  plan_m = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_FORWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD,  FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
#endif
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

#ifdef MPI
  if(T==0) {
    fprintf(stderr, "[%2d] local T is zero; exit\n", g_cart_id);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(2);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  Lhalf = LX / 2;
  Lhp1  = Lhalf + 1;

  /* allocate memory for the contractions */
  disc  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  pir  = (double*)calloc(32*VOLUME, sizeof(double));
  if( pir == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for pir\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  /***************************************
   * initialize the r=const list
   ***************************************/
  make_rid_list(&rid, &rlist, &rcount, g_rmin, g_rmax); 

  /***************************************
   * initialize the O-h orbits
   ***************************************/
  make_Oh_orbits_r(&oh_id, &oh_count, &oh_val, &oh_nc, g_rmin, g_rmax);

  /***************************************
   * set model type function
   ***************************************/
  switch (model_type) {
    case 1:
      model_type_function = pidisc_model1;
      fprintf(stdout, "# function pointer set to type pidisc_model1\n");
      break;
    case 2:
      model_type_function = pidisc_model2;
      fprintf(stdout, "# function pointer set to type pidisc_model2\n");
      break;
    case 3:
      model_type_function = pidisc_model3;
      fprintf(stdout, "# function pointer set to type pidisc_model3\n");
      break;
    case 0:
    default:
      model_type_function = pidisc_model;
      fprintf(stdout, "# function pointer set to type pidisc_model\n");
      break;
  }

  /***************************************
   * loop on gauge id
   ***************************************/
if(mode>=0) {
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

    fprintf(stdout, "# processing gid number %d\n", gid);

    for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
    for(ix=0; ix<32*VOLUME; ix++) pir[ix] = 0.;

    sprintf(filename, "%s.%.4d.%.4d", filename_prefix, gid, Nsave);
    fprintf(stdout, "# reading data from file %s\n", filename);
/*    status = read_contraction(disc, (int*)NULL, filename, 4); */
    status = read_lime_contraction(disc, filename, 4, 0);
    if(status == 106 ) {
      fprintf(stderr, "continue with next file...\n");
      continue;
    }

    /* test: write the contractions to file in ascii format */
/*
    sprintf(filename, "%s.%.4d.%.4d.ascii", filename_prefix, gid, Nsave);
    fprintf(stdout, "# writing data to file %s\n", filename);
    write_contraction(disc, NULL, filename, 4, 2, 0);
*/

    /* test: write the contractions to stdout */
/*
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<4; mu++) {
        fprintf(stdout, "%6d%3d%25.16e%25.16e\n", ix, mu, disc[_GWI(mu,ix,VOLUME)], disc[_GWI(mu,ix,VOLUME)+1]);
      }
    }
*/

    /**********************************************
     * build pir from disc in different modes
     **********************************************/

    if(mode==0 || mode==1) {
      /**********************************************
       * r-window
       **********************************************/
      fprintf(stdout, "# starting mode 1 ...\n");

      for(ix=0; ix<32*T*rcount; ix++) pir[ix] = 0;

      if( (picount = (int*)malloc(T*rcount*sizeof(int)))==(int*)NULL) exit(112);

      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {

        for(ix=0; ix<T*rcount; ix++) picount[ix] = 0;

        for(x0=0; x0<T; x0++) {
        for(x1=0; x1<LX; x1++) {
        for(x2=0; x2<LY; x2++) {
        for(x3=0; x3<LZ; x3++) {

          iix = g_ipt[0][x1][x2][x3];
          if(rid[iix] == -1) continue;
      
          for(y0=0; y0<avgT; y0++) {
            z0 = (x0+y0)%T;
          for(y1=0; y1<avgL; y1++) {
            z1 = (x1+y1)%LX;
          for(y2=0; y2<avgL; y2++) {
            z2 = (x2+y2)%LY;
          for(y3=0; y3<avgL; y3++) {
            z3 = (x3+y3)%LZ;
            iy = g_ipt[y0][y1][y2][y3];
            iz = g_ipt[z0][z1][z2][z3];
            _co_eq_co_ti_co(&w,(complex*)(disc+_GWI(mu,iz, VOLUME)), (complex*)(disc+_GWI(nu,iy,VOLUME)));
            pir[_GWI(4*mu+nu,x0*rcount+rid[iix], T*rcount)  ] += w.re;
            pir[_GWI(4*mu+nu,x0*rcount+rid[iix], T*rcount)+1] += w.im;
            picount[x0*rcount+rid[iix]]++;
          }
          }
          }
          }
        }
        }
        }
        }
        for(x0=0; x0<T; x0++) {
          for(x1=0; x1<rcount; x1++) {
            pir[_GWI(4*mu+nu,x0*rcount+x1, T*rcount)  ] /= (double)(picount[x0*rcount+x1]);
            pir[_GWI(4*mu+nu,x0*rcount+x1, T*rcount)+1] /= (double)(picount[x0*rcount+x1]);
          }
        }

      }
      }
      sprintf(filename, "pir.%.2d.%.4d.%.4d", mode, gid, Nsave);
      ofs = fopen(filename, "w");
      if(ofs == (FILE*)NULL) exit(110);
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        for(x0=0; x0<T; x0++) {
          for(x1=0; x1<rcount; x1++) {
            ix = x0*rcount + x1;
            fprintf(ofs, "%3d%3d%3d%16.9e%25.16e%25.16e\n", mu, nu, x0, rlist[x1],
              pir[_GWI(4*mu+nu,ix,T*rcount)  ], pir[_GWI(4*mu+nu,ix,T*rcount)+1]);
          }
        }
      }
      }
      fclose(ofs);
      sprintf(filename, "pir.%.2d.%.4d.%.4d.info", mode, gid, Nsave);
      if( (ofs = fopen(filename, "w")) == (FILE*)NULL) exit(110);
      fprintf(ofs, "Rmin = %25.16e\n", g_rmin);
      fprintf(ofs, "Rmax = %25.16e\n", g_rmax);
      fprintf(ofs, "rcount = %d\n", rcount);
      fprintf(ofs, "avgL = %d\n", avgL);
      fprintf(ofs, "avgT = %d\n", avgT);
      fclose(ofs);
      free(picount);

    } else if(mode==0 || mode==2) {
      /******************************************************
       * use only on-axis r's
       ******************************************************/
      fprintf(stdout, "# starting mode 2 ...\n");

      Lhp1 = LX;
      L = LX;
      nclass = T*Lhp1;
      for(ix=0; ix<8*nclass; ix++) pir[ix] = 0;

      if( (picount = (int*)malloc(4*nclass*sizeof(int))) == (int*)NULL ) exit(125);
      for(ix=0; ix<4*nclass; ix++) picount[ix] = 0;

      for(x0=0; x0<T; x0++) {
      for(x1=0; x1<Lhp1; x1++) {
        iix = x0*Lhp1+x1;
        for(y0=0; y0<T-x0; y0++) {
        for(y1=0; y1<(L-x1); y1++) {
        for(y2=0; y2<L; y2++) {
        for(y3=0; y3<L; y3++) {
          iz1 = g_ipt[y0+x0][y1+x1][y2   ][y3   ];
          iz2 = g_ipt[y0+x0][y3   ][y1+x1][y2   ];
          iz3 = g_ipt[y0+x0][y2   ][y3   ][y1+x1];
          iy1 = g_ipt[y0   ][y1   ][y2   ][y3   ];
          iy2 = g_ipt[y0   ][y3   ][y1   ][y2   ];
          iy3 = g_ipt[y0   ][y2   ][y3   ][y1   ];
          for(mu=0; mu<4; mu++) {
          for(nu=0; nu<4; nu++) {
            imunu = mu_nu_tab[4*mu+nu];
            idx = _GWI(imunu, iix, nclass);
            _co_eq_co_ti_co(&w,(complex*)(disc+_GWI(mu,iz1, VOLUME)), (complex*)(disc+_GWI(nu,iy1,VOLUME)));
            pir[idx  ] += w.re;
            pir[idx+1] += w.im;
            _co_eq_co_ti_co(&w,(complex*)(disc+_GWI(mu,iz2, VOLUME)), (complex*)(disc+_GWI(nu,iy2,VOLUME)));
            pir[idx  ] += w.re;
            pir[idx+1] += w.im;
            _co_eq_co_ti_co(&w,(complex*)(disc+_GWI(mu,iz3, VOLUME)), (complex*)(disc+_GWI(nu,iy3,VOLUME)));
            pir[idx  ] += w.re;
            pir[idx+1] += w.im;
            picount[imunu*nclass+iix]+=3;
          }}
        }}}}
      }}
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<nclass; ix++) {
          fprintf(stdout, "# mu=%d, ix=%4d, picount=%6d\n", mu,ix, picount[mu*nclass+ix]);
          pir[_GWI(mu, ix, nclass)  ] /= (double)picount[mu*nclass+ix];
          pir[_GWI(mu, ix, nclass)+1] /= (double)picount[mu*nclass+ix];
        }
      }
      sprintf(filename, "pir.%.2d.%.4d.%.4d", mode, gid, Nsave);
      ofs = fopen(filename, "w");
      if(ofs == (FILE*)NULL) exit(110);
      for(mu=0; mu<4; mu++) {
        for(x0=0; x0<T; x0++) {
          for(x1=0; x1<Lhp1; x1++) {
            ix = x0*Lhp1 + x1;
            fprintf(ofs, "%3d%3d%6.1f%6.1f%25.16e%25.16e\n", mu_nu_comb[mu][0], mu_nu_comb[mu][1], 
              (double)x0, (double)x1,
              pir[_GWI(mu,ix,nclass)  ], pir[_GWI(mu,ix,nclass)+1]);
          }
        }
      } 
      fclose(ofs);
      sprintf(filename, "pir.%.2d.%.4d.%.4d.info", mode, gid, Nsave);
      if( (ofs = fopen(filename, "w")) == (FILE*)NULL) exit(110);
      fprintf(ofs, "Rmin = %25.16e\n", g_rmin);
      fprintf(ofs, "Rmax = %25.16e\n", g_rmax);
      fprintf(ofs, "rcount = %d\n", rcount);
      fprintf(ofs, "avgL = %d\n", avgL);
      fprintf(ofs, "avgT = %d\n", avgT);
      fclose(ofs);

    } else if(mode==0 || mode==3) {
      /**************************************
       * average over O-h orbits
       **************************************/
      fprintf(stdout, "# starting mode 3 ...\n");

      for(ix=0; ix<8*oh_nc; ix++) pir[ix] = 0.;

      if( (picount = (int*)malloc(4*oh_nc*sizeof(int))) == (int*)NULL ) exit(125);

      for(ix=0; ix<4*oh_nc; ix++) picount[ix] = 0;

      for(x0=-T+1;  x0<T;  x0++) {
        if(x0<0) {
          xx0 = T+x0;
          y0min = -x0;
          y0max = T;
        } else {
          xx0 = x0;
          y0min = 0;
          y0max = T-x0;
        }
      for(x1=-LX+1; x1<LX; x1++) {
        if(x1<0) {
          xx1 = LX+x1;
          y1min = -x1;
          y1max = LX;
        } else {
          xx1 = x1;
          y1min = 0;
          y1max = LX-x1;
        }
      for(x2=-LY+1; x2<LY; x2++) {
        if(x2<0) {
          xx2 = LY+x2;
          y2min = -x2;
          y2max = LY;
        } else {
          xx2 = x2;
          y2min = 0;
          y2max = LY-x2;
        }
      for(x3=-LZ+1; x3<LZ; x3++) {
        if(x3<0) {
          xx3 = LZ+x3;
          y3min = -x3;
          y3max = LZ;
        } else {
          xx3 = x3;
          y3min = 0;
          y3max = LZ-x3;
        }
/*
        fprintf(stdout, "x  = (%2d%2d%2d%2d)\nxx = (%2d%2d%2d%2d)\n", x0, x1, x2, x3, xx0, xx1, xx2, xx3);
        fprintf(stdout, "y0-range: %2d\t%2d\ny1-range: %2d\t%2d\ny2-range: %2d\t%2d\ny3-range: %2d\t%2d\n", 
          y0min, y0max, y1min, y1max, y2min, y2max, y3min, y3max);
        fprintf(stdout, "# xx = (%d,%d,%d,%d)\n", xx0, xx1, xx2, xx3);
*/

        ix = g_ipt[xx0][xx1][xx2][xx3];
        if(oh_id[ix] == -1) continue;
     
/*        if(oh_id[ix]==1) fprintf(stdout, "x=(%2d,%2d,%2d,%2d);\n", x0, x1, x2, x3); */

        for(y0=y0min; y0<y0max; y0++) {
          z0 = x0 + y0;
        for(y1=y1min; y1<y1max; y1++) {
          z1 = x1 + y1;
        for(y2=y2min; y2<y2max; y2++) {
          z2 = x2 + y2;
        for(y3=y3min; y3<y3max; y3++) {
          z3 = x3 + y3;
          iy = g_ipt[y0][y1][y2][y3];
          iz = g_ipt[z0][z1][z2][z3];
          for(mu=0; mu<4; mu++) {
          for(nu=0; nu<4; nu++) {
            _co_eq_co_ti_co(&w,(complex*)(disc+_GWI(mu,iz, VOLUME)), (complex*)(disc+_GWI(nu,iy,VOLUME)));
            imunu = mu_nu_tab[4*mu+nu];
            pir[_GWI(imunu, oh_id[ix], oh_nc)  ] += w.re;
            pir[_GWI(imunu, oh_id[ix], oh_nc)+1] += w.im;
            picount[imunu*oh_nc+oh_id[ix]]++;
/*            if(imunu==0 && oh_id[ix]==1) fprintf(stdout, "x=(%2d,%2d,%2d,%2d); y=(%2d,%2d,%2d,%2d); z=(%2d,%2d,%2d,%2d)\n", x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3);
*/
          }}
        }}}}


      }}}}

      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<oh_nc; ix++) {
          fprintf(stdout, "mu=%1d, ix=%2d, picount = %4d\n", mu, ix, picount[mu*oh_nc+ix]);
          if(picount[ix]>0) {
            pir[_GWI(mu, ix, oh_nc)  ] /= (double)picount[mu*oh_nc+ix];
            pir[_GWI(mu, ix, oh_nc)+1] /= (double)picount[mu*oh_nc+ix];
          }
        }
      }
      Thp1 = T/2 + 1;
      sprintf(filename, "pir.%.2d.%.4d.%.4d", mode, gid, Nsave);
      if( (ofs = fopen(filename, "w")) == (FILE*)NULL) exit(110);
      for(mu=0; mu<4; mu++) {
        for(x0=0; x0<oh_nc; x0++) {
          if(picount[x0]>0) {
            fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%25.16e\n", 
              mu_nu_comb[mu][0], mu_nu_comb[mu][1], (x0*Thp1)/oh_nc, oh_val[0][x0],
              pir[_GWI(mu, x0, oh_nc)  ], pir[_GWI(mu, x0, oh_nc)+1]);
          }
        }
      } 
      fclose(ofs);
      sprintf(filename, "pir.%.2d.%.4d.%.4d.info", mode, gid, Nsave);
      if( (ofs = fopen(filename, "w")) == (FILE*)NULL) exit(110);
      fprintf(ofs, "mode = %d\n", mode);
      fprintf(ofs, "Rmin = %25.16e\n", g_rmin);
      fprintf(ofs, "Rmax = %25.16e\n", g_rmax);
      fprintf(ofs, "rcount = %d\n", oh_nc);
      fprintf(ofs, "avgL = %d\n", avgL);
      fprintf(ofs, "avgT = %d\n", avgT);
      fclose(ofs);
      free(picount);
     
    }

  }

}  /* end of if mode>=0 */

  if(mode==0 || mode==-1) {
    /**************************************
     * average over sets |r|=R
     **************************************/

    if( (picount = (int*)malloc(T*rcount*sizeof(int)))==(int*)NULL) exit(112);

    for(ix=0; ix<32*T*rcount; ix++) pir[ix] = 0;

    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      model_type_function(mrho, dcoeffre, dcoeffim, disc, plan_m, 4*mu+nu);

/*
      fprintf(stdout, "# pimodel in position space:\n");
      for(ix=0; ix<VOLUME; ix++) {
        fprintf(stdout, "pimodel-X[%2d,%5d] = %25.16e +i %25.16e\n", 4*mu+nu, ix, disc[2*ix], disc[2*ix+1]);
      }
*/

      for(ix=0; ix<T*rcount; ix++) picount[ix] = 0;

      for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix = g_ipt[x0][x1][x2][x3];
        iix = g_ipt[0][x1][x2][x3];
        if(rid[iix] == -1) continue;
        pir[_GWI(4*mu+nu,x0*rcount+rid[iix], T*rcount)  ] += disc[2*ix  ];
        pir[_GWI(4*mu+nu,x0*rcount+rid[iix], T*rcount)+1] += disc[2*ix+1];
        picount[x0*rcount+rid[iix]]++;
      }}}}
      for(x0=0; x0<T; x0++) {
        for(x1=0; x1<rcount; x1++) {
          pir[_GWI(4*mu+nu,x0*rcount+x1, T*rcount)  ] /= (double)(picount[x0*rcount+x1]);
          pir[_GWI(4*mu+nu,x0*rcount+x1, T*rcount)+1] /= (double)(picount[x0*rcount+x1]);
        }
      }

    }
    }
    sprintf(filename, "pimodel%1d.%.2d", model_type, abs(mode));
    ofs = fopen(filename, "w");
    if(ofs == (FILE*)NULL) exit(110);
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      for(x0=0; x0<T; x0++) {
        for(x1=0; x1<rcount; x1++) {
          ix = x0*rcount + x1;
          fprintf(ofs, "%3d%3d%3d%16.9e%25.16e%25.16e%6d\n", mu, nu, x0, rlist[x1],
            pir[_GWI(4*mu+nu,ix,T*rcount)  ], pir[_GWI(4*mu+nu,ix,T*rcount)+1], picount[ix]);
        }
      }
    }}
    fclose(ofs);
    sprintf(filename, "pimodel%1d.%.2d.info", model_type, abs(mode));
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL) exit(110);
    fprintf(ofs, "mode = %d\n", mode);
    fprintf(ofs, "Rmin = %25.16e\n", g_rmin);
    fprintf(ofs, "Rmax = %25.16e\n", g_rmax);
    fprintf(ofs, "rcount = %d\n", rcount);
    fclose(ofs);
    free(picount);

  } else if(mode==0 || mode==-2) {
    /**********************************************
     * take on-axis distances only, avg. over dir.
     **********************************************/
    
    nclass = T*LX;
    for(ix=0; ix<8*nclass; ix++) pir[ix] = 0.;

    for(i=0; i<4; i++) {
      imunu = 4*mu_nu_comb[i][0] + mu_nu_comb[i][1];
      model_type_function(mrho, dcoeffre, dcoeffim, disc, plan_m, imunu);

      for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
        iix = x0*LX+x1;
        iiz[0] = g_ipt[x0][x1][0][0];
        iiz[1] = g_ipt[x0][0][x1][0];
        iiz[2] = g_ipt[x0][0][0][x1];
        idx = _GWI(i, iix, nclass);
        pir[idx  ] += ( disc[2*iiz[0]  ] + disc[2*iiz[1]  ] + disc[2*iiz[2]  ] ) / 3.;
        pir[idx+1] += ( disc[2*iiz[0]+1] + disc[2*iiz[1]+1] + disc[2*iiz[2]+1] ) / 3.;
      }}
    }
    sprintf(filename, "pimodel%1d.%.2d", model_type, abs(mode));
    ofs = fopen(filename, "w");
    if(ofs == (FILE*)NULL) exit(110);
    for(mu=0; mu<4; mu++) {
      for(x0=0; x0<T; x0++) {
        for(x1=0; x1<LX; x1++) {
          ix = x0*LX + x1;
          fprintf(ofs, "%3d%3d%6.0f%6.0f%25.16e%25.16e\n", mu_nu_comb[mu][0], mu_nu_comb[mu][1], 
            (double)x0, (double)x1, pir[_GWI(mu,ix,nclass)  ], pir[_GWI(mu,ix,nclass)+1]);
        }
      }
    } 
    fclose(ofs);
    sprintf(filename, "pimodel%1d.%.2d.info", model_type, abs(mode));
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL) exit(110);
    fprintf(ofs, "mode = %d\n", mode);
    fprintf(ofs, "Rmin = %25.16e\n", 0.);
    fprintf(ofs, "Rmax = %25.16e\n", (double)LX);
    fprintf(ofs, "rcount = %d\n", 3);
    fclose(ofs);

  } else if(mode==0 || mode==-3) {
    /**********************************************
     * average over O_h orbits
     **********************************************/

    for(ix=0; ix<8*oh_nc; ix++) pir[ix] = 0.;

    if( (picount = (int*)malloc(oh_nc*sizeof(int))) == (int*)NULL ) exit(125);
 
    for(i=0; i<4; i++) {
      mu = 4*mu_nu_comb[i][0] + mu_nu_comb[i][1];

      for(ix=0; ix<oh_nc; ix++) picount[ix] = 0;
      model_type_function(mrho, dcoeffre, dcoeffim, disc, plan_m, mu);

      for(ix=0; ix<VOLUME; ix++) {
        if(oh_id[ix] == -1) continue;
        pir[_GWI(i, oh_id[ix], oh_nc)  ] += disc[2*ix  ];
        pir[_GWI(i, oh_id[ix], oh_nc)+1] += disc[2*ix+1];
        picount[oh_id[ix]]++;
      }
      for(ix=0; ix<oh_nc; ix++) {
        if(picount[ix]>0) {
          pir[_GWI(i, ix, oh_nc)  ] /= (double)picount[ix];
          pir[_GWI(i, ix, oh_nc)+1] /= (double)picount[ix];
        }
      }
    }
    Thp1 = T/2 + 1;
/*    Thp1 = T; */
    sprintf(filename, "pimodel%1d.%.2d", model_type, abs(mode));
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL) exit(110);
    for(i=0; i<4; i++) {
      for(x0=0; x0<oh_nc; x0++) {
        if(picount[x0]>0) {
          fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%25.16e%6d\n", 
            mu_nu_comb[i][0], mu_nu_comb[i][1], (x0*Thp1)/oh_nc, oh_val[0][x0],
            pir[_GWI(i, x0, oh_nc)  ], pir[_GWI(i, x0, oh_nc)+1], x0);
        }
      }
    } 
    fclose(ofs);
    sprintf(filename, "pimodel%1d.%.2d.info", model_type, abs(mode));
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL) exit(110);
    fprintf(ofs, "mode = %d\n", mode);
    fprintf(ofs, "Rmin = %25.16e\n", 0.);
    fprintf(ofs, "Rmax = %25.16e\n", (double)LX);
    fprintf(ofs, "rcount = %d\n", oh_nc);
    fclose(ofs);
    free(picount);
  }
  
  /**************************************
   * free the allocated memory, finalize
   **************************************/
  free_geometry();
  free(disc);
  free(pir);
  if(rlist     != (double*)NULL) free(rlist);
  if(rid       != (int*)   NULL) free(rid);
  if(oh_id     != (int*)   NULL) free(oh_id);
  if(oh_count  != (int*)   NULL) free(oh_count);
  if(oh_val    != (double**)  NULL) {
    free(*oh_val); free(oh_val);
  }

#ifdef MPI
  free(status);
  MPI_Finalize();
#endif

  return(0);

}
