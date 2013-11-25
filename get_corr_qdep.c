/****************************************************
 * get_corr_qdep.c
 *
 *
 * PURPOSE
 * - recover time and momentum dep. correlators from
 *   Dru's/Xu's vacuum pol. tensor files
 *   file pattern:
 *     vacpol_con_cc_q_3320_x19y04z00t22.dat
 * - correlators: \Pi_00 and 1/3 (\Pi_11+\Pi_22+\Pi_33)
 * DONE:
 * TODO:
 * CHANGES:
 *
 * Fri Sep 23 10:58:13 CEST 2011
 *
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
#include "read_input_parser.h"
#include "get_index.h"
#include "contractions_io.h"
#include "make_q_orbits.h"

void usage() {
  fprintf(stdout, "Code to recover rho-rho correl. from vacuum polarization at non-zero momentum\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, mu, status;
  int filename_set = 0;
  int mode = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, iiy, gid, iclass;
  int Thp1, nclass;
  int *picount;
  double *conn = (double*)NULL;
  double *conn2 = (double*)NULL;
  double q[4], qsqr;
  int verbose = 0;
  char filename[800];
  double ratime, retime;

  int *qid=NULL, *qcount=NULL, **qrep=NULL, **qmap=NULL;
  double **qlist=NULL, qmax=0.; 
  int VOL3;

  FILE *ofs;
  fftw_complex *corrt=NULL;

  fftw_complex *pi00=(fftw_complex*)NULL, *pijj=(fftw_complex*)NULL, *piavg=(fftw_complex*)NULL;

  fftw_plan plan_m;

  while ((c = getopt(argc, argv, "h?vf:m:q:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'm':
      mode = atoi(optarg);
      break;
    case 'q':
      qmax = atof(optarg);
      fprintf(stdout, "\n# [] qmax set to %e\n", qmax);
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

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize fftw, create plan with FFTW_FORWARD ---  in contrast to
   * FFTW_BACKWARD in e.g. avc_exact */
  plan_m = fftw_create_plan(T_global, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  if(plan_m==NULL) {
    fprintf(stderr, "Error, could not create fftw plan\n");
    return(1);
  }

  T            = T_global;
  Thp1         = T/2 + 1;
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
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  VOL3 = LX*LY*LZ;

  status = make_qlatt_orbits_3d_parity_avg(&qid, &qcount, &qlist, &nclass, &qrep, &qmap);
  if(status != 0) {
    fprintf(stderr, "\n[] Error while creating h4-lists\n");
    exit(4);
  }
  fprintf(stdout, "# [] number of classes = %d\n", nclass);
//  exit(255);

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(32*VOLUME, sizeof(double));
  if( (conn==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    exit(3);
  }

/*
  conn2 = (double*)calloc(32*VOLUME, sizeof(double));
  if( (conn2==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    exit(4);
  }

  pi00 = (fftw_complex*)malloc(VOLUME*sizeof(fftw_complex));
  if( (pi00==(fftw_complex*)NULL) ) {
    fprintf(stderr, "could not allocate memory for pi00\n");
    exit(2);
  }

  pijj = (fftw_complex*)fftw_malloc(VOLUME*sizeof(fftw_complex));
  if( (pijj==(fftw_complex*)NULL) ) {
    fprintf(stderr, "could not allocate memory for pijj\n");
    exit(2);
  }
*/
  corrt = fftw_malloc(T*sizeof(fftw_complex));
  if(corrt == NULL) {
    fprintf(stderr, "\nError, could not alloc corrt\n");
    exit(3);
  }

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

//    for(ix=0; ix<VOLUME; ix++) {pi00[ix].re = 0.; pi00[ix].im = 0.;}
//    for(ix=0; ix<VOLUME; ix++) {pijj[ix].re = 0.; pijj[ix].im = 0.;}
    /***********************
     * read contractions   *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;

    sprintf(filename, "%s.%.4d", filename_prefix, gid);
    fprintf(stdout, "# Reading data from file %s\n", filename);
    if(format==2) {
      status = read_contraction(conn, NULL, filename, 16);
    } else {
      status = read_lime_contraction(conn, filename, 16, 0);
    }
    if(status != 0) {
      fprintf(stderr, "Error: could not read from file %s; status was %d\n", filename, status);
      continue;
    }
/*
    sprintf(filename, "%s.%.4d.%.4d", filename_prefix2, gid);
    fprintf(stdout, "# Reading data from file %s\n", filename);
    status = read_lime_contraction(conn2, filename, 16, 0);
    if(status == 106) {
      fprintf(stderr, "Error: could not read from file %s; status was %d\n", filename, status);
      continue;
    }
*/
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time to read contractions %e seconds\n", retime-ratime);

    /***********************
     * fill the correlator *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
/*
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      for(x0=0; x0<T; x0++) {
        iix = g_ipt[0][x1][x2][x3]*T+x0;
        for(mu=1; mu<4; mu++) {
          ix = _GWI(5*mu,g_ipt[x0][x1][x2][x3],VOLUME);
          pijj[iix].re += ( conn[ix  ] - conn2[ix  ] ) * (double)Nsave / (double)(Nsave-1);
          pijj[iix].im += ( conn[ix+1] - conn2[ix+1] ) * (double)Nsave / (double)(Nsave-1);
        }
        ix = 2*g_ipt[x0][x1][x2][x3];
        pi00[iix].re += ( conn[ix  ] - conn2[ix  ] ) * (double)Nsave / (double)(Nsave-1);
        pi00[iix].im += ( conn[ix+1] - conn2[ix+1] ) * (double)Nsave / (double)(Nsave-1);
      }
    }}}
*/

    for(iclass=0;iclass<nclass;iclass++) {
      if(qlist[iclass][0] >= qmax) {
//        fprintf(stdout, "\n# [] will skip class %d, momentum squared = %f is too large\n", iclass, qlist[iclass][0]);
        continue;
//      } else {
//        fprintf(stdout, "\n# [] processing class %d, momentum squared = %f\n", iclass, qlist[iclass][0]);
      }

      for(x0=0; x0<T; x0++) {
        corrt[x0].re = 0.;
        corrt[x0].im = 0.;
      }

/* 
      for(x1=0;x1<VOL3;x1++) {
        if(qid[x1]==iclass) {
          fprintf(stdout, "# using mom %d ---> (%d, %d, %d)\n", x1, qrep[iclass][1], qrep[iclass][2], qrep[iclass][3]);
          for(x0=0; x0<T; x0++) {
            ix = x0*VOL3 + x1;
            corrt[x0].re += conn[_GWI(5,ix,VOLUME)  ] + conn[_GWI(10,ix,VOLUME)  ] + conn[_GWI(15,ix,VOLUME)  ];
            corrt[x0].im += conn[_GWI(5,ix,VOLUME)+1] + conn[_GWI(10,ix,VOLUME)+1] + conn[_GWI(15,ix,VOLUME)+1];
          }
        }
      }
*/
      for(x0=0; x0<T; x0++) {
        for(x1=0;x1<qcount[iclass];x1++) {
          x2 = qmap[iclass][x1];
          // if(x0==0) fprintf(stdout, "# using mom %d ---> (%d, %d, %d)\n", x2, qrep[iclass][1], qrep[iclass][2], qrep[iclass][3]);
            ix = x0*VOL3 + x2;
            corrt[x0].re += conn[_GWI(5,ix,VOLUME)  ] + conn[_GWI(10,ix,VOLUME)  ] + conn[_GWI(15,ix,VOLUME)  ];
            corrt[x0].im += conn[_GWI(5,ix,VOLUME)+1] + conn[_GWI(10,ix,VOLUME)+1] + conn[_GWI(15,ix,VOLUME)+1];
        }
      }
      // fprintf(stdout, "\n\n# ------------------------------\n");

      for(x0=0; x0<T; x0++) {
        corrt[x0].re /= (double)T * qcount[iclass];
        corrt[x0].im /= (double)T * qcount[iclass];
      }
/*      fftw(plan_m, 1, corrt, 1, T, (fftw_complex*)NULL, 0, 0); */
      fftw_one(plan_m, corrt, NULL);
      sprintf(filename, "rho.%.4d.x%.2dy%.2dz%.2d", gid, qrep[iclass][1], qrep[iclass][2], qrep[iclass][3]);
      if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
        fprintf(stderr, "Error: could not open file %s for writing\n", filename);
        exit(5);
      }
      fprintf(stdout, "# writing VKVK data to file %s\n", filename);
      fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f%21.12f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu, qlist[iclass][0]);
    
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 0, 0, 0, corrt[0].re, 0., gid);
      for(x0=1; x0<(T/2); x0++) {
        fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 0, 0, x0, 
          corrt[x0].re, corrt[T-x0].re, gid);
      }
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 0, 0, (T/2), corrt[T/2].re, 0., gid);
      fflush(ofs);
      fclose(ofs);

      retime = (double)clock() / CLOCKS_PER_SEC;
      fprintf(stdout, "# time to fill correlator %e seconds\n", retime-ratime);

    }  // of loop on classes

  }  // end of loop on gauge id

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  if(corrt != NULL) free(corrt);
  free_geometry();

  if(pi00 != NULL) free(pi00);
  if(pijj != NULL) free(pijj);

  fftw_destroy_plan(plan_m);

  finalize_q_orbits(&qid, &qcount, &qlist, &qrep);
  if(qmap != NULL) {
    free(qmap[0]);
    free(qmap);
  }

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [] %s# [] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [] %s# [] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
