/****************************************************
 * analyseH3.c
 *
 * Tue Oct 13 17:09:00 CEST 2009
 *
 * PURPOSE:
 * - c-implementation of analysis programme for \Pi_{\mu\nu}(\hat{q}^2)
 * - implement the H4 method as described in 0705.3523v2 [hep-lat]
 * DONE:
 * TODO:
 * CHANGES:
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
#include "get_index.h"
#include "uwerr.h"


void usage(void) {
  fprintf(stdout, "Program analyseH3\n");
  fprintf(stdout, "Options:\n");
  /* ... */
  exit(0);
}

int read_pimn_timeslice(double *pimn, const int ts, const int confid, const int saveid) {

  char filename[800];
  int iostat, ix, mu, nu, iix, np;
  int x0, x1, x2, x3;
  int VOL3=LX*LY*LZ, L=LX;
  double ratime, retime, buff[32], buff2[2];

  FILE *ofs;

  /* read the data file */
  if(format==0 || format==1) {
    if(saveid>=0) {
      sprintf(filename, "%s.%.4d.%.4d", filename_prefix, confid, saveid);
    }
    else {
      sprintf(filename, "%s.%.4d", filename_prefix, confid);
    }
  }
  else {
    sprintf(filename, "%s", gaugefilename_prefix);
  }

  if((void*)(ofs = fopen(filename, "r"))==NULL) {
    fprintf(stderr, "could not open file %s for reading\n", filename);
    return(-1);
  }

  ratime = clock() / CLOCKS_PER_SEC;
    if(format==1) {
      fprintf(stdout, "reading of binary data from file %s for ts=%d\n", filename, ts);

      fseek(ofs, ts*LX*LY*LZ*32*sizeof(double), SEEK_SET);

      for(ix=0; ix<VOL3; ix++) {
        iostat = fread(buff, sizeof(double), 32, ofs);
        if(iostat != 32) {
          fprintf(stderr, "could not read proper amount of data\n");
          return(-3);
        }
        /* fprintf(stdout, "ix = %d\n", ix); */
        for(mu=0; mu<16; mu++) {
          pimn[_GWI(mu,ix,VOL3)  ] = buff[2*mu  ];
          pimn[_GWI(mu,ix,VOL3)+1] = buff[2*mu+1];
        }
      }
    }
    else if(format==2) {
      fprintf(stdout, "buffered reading of binary data for ts = %d from file %s in format %2d\n", ts, filename, format);
      for(x1=0; x1<L; x1++) {
      for(x2=0; x2<L; x2++) {
      for(x3=0; x3<L; x3++) {
        for(mu=0; mu<4; mu++) {
        for(nu=0; nu<4; nu++) {
          /* find position in Xu's/Dru's file */
          iix = get_indexf(ts, x1, x2, x3, mu, nu);
/*          fprintf(stdout, "x1=%3d, x2=%3d, x3=%3d, mu=%3d, nu=%3d, iix = %d\n", x1, x2, x3, mu, nu, iix); */
          fseek(ofs, iix*sizeof(double), SEEK_SET);
          iostat = fread(buff2, sizeof(double), 2, ofs);
          if(iostat != 2) {
            fprintf(stderr, "could not read proper amount of data\n");
            return(-3);
          }
          ix  = g_ipt[0][x1][x2][x3];
          /* fprintf(stdout, "Dru's index: %8d\t my index: %8d\n", 16*ix+mu, ix); */
          pimn[_GWI(4*mu+nu,ix,VOL3)  ] = buff2[0];
          pimn[_GWI(4*mu+nu,ix,VOL3)+1] = buff2[1];
        }
        }
      }
      }
      }
    }
  retime = clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "time for reading pimn in %e seconds\n", retime-ratime);
  fclose(ofs);

  return(0);
}

/***********************************************************************/

int main(int argc, char **argv) {

  int c, i, n, m, read_flag=0, iconf;
  int verbose=0;
  int mode=0, ntag=0;
  int filename_set=0;
  int *Nconf_list;
  int l_LX_at, l_LXstart_at, VOL3;
  int x0, x1, x2, x3, mu, nu, ix;
  int *q2_id=(int*)NULL, **q2_list=(int**)NULL, *q2_count=(int*)NULL;
  int **q4_id=(int**)NULL, **q4_count=(int**)NULL;
  int *h3_count=(int*)NULL, *h3_id=(int*)NULL, h3_nc, q2_nc, *q4_nc=(int*)NULL;
  double *pimn=(double*)NULL;
  double *pi=(double*)NULL, *pi2=(double*)NULL, deltamn, **piavg=(double**)NULL;
  double q2, q[4], qhat[4], *q2_val=(double*)NULL, **q4_val=(double**)NULL;
  double **h3_val=(double**)NULL;
  char filename[800];
  FILE *ofs, *ifs;

  while ((c = getopt(argc, argv, "h?vaf:m:n:")) != -1) {
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
    case 'n':
      ntag = atoi(optarg);
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /**************************
   * set the default values *
   **************************/
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "analyse.input");

  /***********************
   * read the input file *
   ***********************/
  read_input(filename);

  /*********************************
   * some checks on the input data *
   *********************************/
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
/*
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }
*/

  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  VOL3 = LX*LY*LZ;

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    return(102);
  }

  geometry();

  /* allocating memory for pimn */
  pimn = (double*)malloc(32*VOL3*sizeof(double));
  if(pimn==(double*)NULL) {
    fprintf(stderr, "could not allocate memory for pimn\n");
    return(101);
  }

  /***********************************
   * make lists for id, count and val
   ***********************************/
  fprintf(stdout, "# make H3 orbits\n");  
  if(make_H3orbits_timeslice(&h3_id, &h3_count, &h3_val, &h3_nc) != 0) return(123);
  set_qid_val(h3_val, 0);
/*
  for(ix=0; ix<h3_nc; ix++) {
    fprintf(stdout, "h3_val[%d] = %12.5e, %12.5e, %12.5e, %12.5e\n", ix, h3_val[0][ix], h3_val[1][ix], h3_val[2][ix], h3_val[3][ix]);
  }
*/
  
  fprintf(stdout, "# make q2 orbits\n");
  make_q2orbits(&q2_id, &q2_list, &q2_val, &q2_count, &q2_nc, h3_val, h3_nc);

  fprintf(stdout, "# make q4 orbits\n");
  make_q4orbits(&q4_id, &q4_val, &q4_count, &q4_nc, q2_list, q2_count, q2_nc, h3_val);

  pi = (double*)malloc(2*VOL3*sizeof(double));
  if(pi==(double*)NULL) {
    fprintf(stderr, "could not allocate memory for pi\n");
    return(103);
  }
  pi2= (double*)malloc(2*VOL3*sizeof(double));
  if(pi2==(double*)NULL) {
    fprintf(stderr, "could not allocate memory for pi2\n");
    return(103);
  }
  piavg = (double**)malloc(h3_nc*sizeof(double*));
  if(piavg==(double**)NULL) {
    fprintf(stderr, "could not allocate memory for piavg\n");
    return(103);
  }
  piavg[0] = (double*)malloc(2*Nconf*h3_nc*sizeof(double));
  if(piavg[0]==(double*)NULL) {
    fprintf(stderr, "could not allocate memory for piavg\n");
    return(104);
  }
  for(ix=1; ix<h3_nc; ix++) piavg[ix] = piavg[ix-1] + 2*Nconf;

  if(Nconf>0) {
    if( (Nconf_list=(int*)malloc(Nconf*sizeof(int))) == (int*)NULL ) exit(106);
    for(ix=0; ix<Nconf; ix++) Nconf_list[ix] = -1;
    sprintf(filename, "%s", filename_prefix2);
    fprintf(stdout, "reading conf list from file %s\n", filename);
    if( (ofs = fopen(filename, "r")) == (FILE*)NULL ) exit(105);
    for(ix=0; ix<Nconf; ix++) fscanf(ofs, "%i", Nconf_list+ix);
    fclose(ofs);
    if( (ifs = fopen("Nconf_names", "r"))==(FILE*)NULL) exit(129);
  }

  /*******************************************
   * loop on time slices
   *******************************************/
  for(x0=0; x0<=T/2; x0++) {

    fprintf(stdout, "[main] ts = %d\n", x0);
    set_qid_val(h3_val, x0);
    for(ix=0; ix<q2_nc; ix++) {
      q2_val[ix] = h3_val[0][q2_list[ix][0]];
    }
    /* for(ix=0; ix<q2_nc; ix++) fprintf(stdout, "i=%d, q2_val=%f\n", ix, q2_val[ix]); */

    /*******************************************
     * loop on the gauge configurations
     *******************************************/
    rewind(ifs);
    for(iconf=0; iconf<Nconf; iconf++) {
      
      fprintf(stdout, "[main] iconf = %d\n", iconf);
      fscanf(ifs, "%s", filename);
      sprintf(gaugefilename_prefix, "%s/%s", filename_prefix, filename);

      fprintf(stdout, "# reading pimn for conf %d\n", Nconf_list[iconf]);
      if(read_pimn_timeslice(pimn, x0, Nconf_list[iconf], Nsave) != 0) {
        fprintf(stderr, "Error on reading of pimn\n");
        exit(101);
      }

      /* test: write the contraction data */
/*
      fprintf(stdout, "timeslice: %d\n", x0);
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix = g_ipt[0][x1][x2][x3];
        fprintf(stdout, "# t=%3d, x=%3d, y=%3d, z=%3d\n", x0, x1, x2, x3);
        for(mu=0; mu<16; mu++) {
          fprintf(stdout, "%3d%25.16e%25.16e\n", mu, pimn[_GWI(mu,ix,VOL3)], pimn[_GWI(mu,ix,VOL3)+1]);
        }
      }
      }
      }
*/

      fprintf(stdout, "# calculate pi from pimn\n");
        q[0]    = 2. * sin( M_PI / (double)T  * (double)(x0) );
      for(x1=0; x1<LX; x1++) {
        q[1]    = 2. * sin( M_PI / (double)LX * (double)(x1) );
      for(x2=0; x2<LY; x2++) {
        q[2]    = 2. * sin( M_PI / (double)LY * (double)(x2) );
      for(x3=0; x3<LZ; x3++) {
        q[3]    = 2. * sin( M_PI / (double)LZ * (double)(x3) );
        ix = g_ipt[0][x1][x2][x3];
        q2    = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
        pi[2*ix  ] = 0.;
        pi[2*ix+1] = 0.;
        for(mu=0; mu<4; mu++) {
        for(nu=0; nu<4; nu++) {
          pi[2*ix  ] += ( q[mu]*q[nu] - q2*(double)(mu==nu) ) * pimn[_GWI(4*mu+nu,ix,VOL3)  ];
          pi[2*ix+1] += ( q[mu]*q[nu] - q2*(double)(mu==nu) ) * pimn[_GWI(4*mu+nu,ix,VOL3)+1];
        }
        }
        if(q2>_Q2EPS) {
          pi[2*ix  ] /= 3. * q2*q2; 
          pi[2*ix+1] /= 3. * q2*q2;
        }
      }
      }
      }
 
    if(x0>0 && x0<T/2) {
 
      fprintf(stdout, "# reading pimn from equiv. timeslice %d\n", T-x0);
      if(read_pimn_timeslice(pimn, T-x0, Nconf_list[iconf], Nsave) != 0) {
        fprintf(stderr, "Error on reading of pimn\n");
        exit(121);
      }

      /* test: write the contraction data */
/*
      fprintf(stdout, "timeslice: %d\n", T-x0);
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix = g_ipt[0][x1][x2][x3];
        fprintf(stdout, "# t=%3d, x=%3d, y=%3d, z=%3d\n", T-x0, x1, x2, x3);
        for(mu=0; mu<16; mu++) {
          fprintf(stdout, "%3d%25.16e%25.16e\n", mu, pimn[_GWI(mu,ix,VOL3)], pimn[_GWI(mu,ix,VOL3)+1]);
        }
      }
      }
      }
*/
      fprintf(stdout, "# calculate pi2 from pimn\n");
        q[0]    = 2. * sin( M_PI / (double)T  * (double)(x0) );
      for(x1=0; x1<LX; x1++) {
        q[1]    = 2. * sin( M_PI / (double)LX * (double)(x1) );
      for(x2=0; x2<LY; x2++) {
        q[2]    = 2. * sin( M_PI / (double)LY * (double)(x2) );
      for(x3=0; x3<LZ; x3++) {
        q[3]    = 2. * sin( M_PI / (double)LZ * (double)(x3) );
        ix = g_ipt[0][x1][x2][x3];
        q2    = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
        pi2[2*ix  ] = 0.;
        pi2[2*ix+1] = 0.;
        for(mu=0; mu<4; mu++) {
        for(nu=0; nu<4; nu++) {
          pi2[2*ix  ] += ( q[mu]*q[nu] - q2*(double)(mu==nu) ) * pimn[_GWI(4*mu+nu,ix,VOL3)  ];
          pi2[2*ix+1] += ( q[mu]*q[nu] - q2*(double)(mu==nu) ) * pimn[_GWI(4*mu+nu,ix,VOL3)+1];
        }
        }
        if(q2>_Q2EPS) {
          pi2[2*ix  ] /= 3. * q2*q2; 
          pi2[2*ix+1] /= 3. * q2*q2;
        }
      }
      }
      }
    }
 
      fprintf(stdout, "# average over H3 orbits\n");
      for(i=0; i<h3_nc; i++) {
        piavg[i][2*iconf  ] = 0.;
        piavg[i][2*iconf+1] = 0.;
      }

      for(ix=0; ix<VOL3; ix++) {
        if(h3_id[ix] != -1) {
          if(x0==0 || x0==T/2) {
            piavg[h3_id[ix]][2*iconf  ] += pi[2*ix  ];
            piavg[h3_id[ix]][2*iconf+1] += pi[2*ix+1];
          } else {
            piavg[h3_id[ix]][2*iconf  ] += (pi[2*ix  ] + pi2[2*ix  ]) / 2.;
            piavg[h3_id[ix]][2*iconf+1] += (pi[2*ix+1] + pi2[2*ix+1]) / 2.;
          }
        }
      }
      for(ix=0; ix<h3_nc; ix++) {
        if(h3_count[ix]>0) {
          piavg[ix][2*iconf  ] /= (double)(h3_count[ix]);
          piavg[ix][2*iconf+1] /= (double)(h3_count[ix]);
          /* fprintf(stdout, "h3_count[%d] = %d\n", ix, h3_count[ix]); */
        }
        else {
          piavg[ix][2*iconf  ] = 0.;
          piavg[ix][2*iconf+1] = 0.;
        }
      }

    } /* of loop over gauge configurations */

    sprintf(filename, "%s.t%.2d", "piavg", x0);
    ofs = fopen(filename, "w");
    if( ofs==(FILE*)NULL ) exit(122);
    for(iconf=0; iconf<Nconf; iconf++) {
      for(ix=0; ix<h3_nc; ix++) {
        fprintf(ofs, "%6d%6d%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e\n", Nconf_list[iconf], ix, \
          piavg[ix][2*iconf], piavg[ix][2*iconf+1], \
          h3_val[0][ix], h3_val[1][ix], h3_val[2][ix], h3_val[3][ix]);
      }
    }
    fclose(ofs);

    /* print everything to file */
    sprintf(filename, "orbits_t%.2d", x0);
    ofs = fopen(filename, "w");
    for(n=0; n<q2_nc; n++) {
      for(m=0; m<q4_nc[n]; m++) {
        for(ix=0; ix<q2_count[n]; ix++) {
          if(q4_id[n][ix] == m ) {
            for(iconf=0; iconf<Nconf; iconf++) {
              fprintf(ofs, "%3d%3d%16.9e%16.9e%25.16e%25.16e%6d\n", 
                n, m, q2_val[n], q4_val[n][m], 
                piavg[q2_list[n][ix]][2*iconf], piavg[q2_list[n][ix]][2*iconf+1], Nconf_list[iconf]);
            }
          }
        }
      }
    }
    fclose(ofs);

    /* UWerr analyse the data for the current time slice */
/*
    nalpha = 1;
    nrep = 1;
    n_r = (int*)malloc(sizeof(int));
    for(n=0; n<q2_nc; n++) {
      for(m=0; m<q4_nc[n]; m++) {
      }
    }    
*/

  } /* of loop on time slices */

  fclose(ifs);

  free(*h3_val);
  free(h3_val);
  free(h3_count);
  free(h3_id);
  free(pimn);
  free(pi);
  free(pi2);
  free(*piavg);
  free(piavg);
  free(q2_val);
  free(q2_count);
  free(q2_id);
  free(*q2_list);
  free(q2_list);
  free(*q4_val);
  free(*q4_count);
  free(*q4_id);

  return(0);
}

