/****************************************************
 * ll_conn_x2dep.c
 *
 * Tue Jun 22 18:14:52 CEST 2010
 *
 * PURPOSE
 * DONE:
 * TODO:
 * - Question: is there still an issue about the source location?
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#define MAIN_PROGRAM

#include "lime.h"
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "get_index.h"
#include "make_H3orbits.h"
#include "make_x_orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to calculate <j_mu j_mu>(x) from momentum space data.\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {
  
  const int K      = 32;
  const int idUsed = 6;
  const int NumFlComb=2;
  size_t bytes;
  int c, mu, nu, status, dims[4], count;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, i, j;
  long unsigned int ix, iix;
  int Thp1, VOL3, source_coords[4];
  int *h4_count=NULL, *h4_id=NULL, h4_nc, **h4_rep=NULL;
  double *conn    = NULL;
  double **h4_val = NULL;
  double **jjx=NULL;
  char filename[800];
  double ratime, retime;
  double q[4], phase;
  double y[4], ysqr, tmp[2];
  FILE *ofs=NULL;
  int *xisimag[2];
  complex w, w1;
  time_t the_time;

  int isimag[]  = {0, 0, 0, 0, 
                   0, 1, 1, 1,  
                   1, 0, 1, 1,
                   1, 1, 0, 0,
  
                   0, 0, 0, 0,  
                   0, 1, 1, 1, 
                   1, 0, 1, 1,  
                   1, 1, 0, 0};

  int nisimag[]  = {0, 0, 0, 0,
                    0, 1, 1, 1,
                    1, 0, 1, 1,
                    1, 1, 0, 0,

                    0, 0, 0, 0,
                    0, 1, 1, 1,
                    1, 0, 1, 1,
                    1, 1, 0, 0};

  xisimag[0] = isimag;
  xisimag[1] = nisimag;

  //int idList[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,\
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  //int cidList[] = { 0, 9, 3, 4, 18, 17};
  //int nidList[] = { 4, 0, 9, 3, 17, 25};
  int idList[NumFlComb][idUsed];
  idList[0][0]  =  0;
  idList[0][1]  =  9;
  idList[0][2]  =  3;
  idList[0][3]  =  4;
  idList[0][4]  = 18;
  idList[0][5]  = 17;
  idList[1][0]  =  4;
  idList[1][1]  =  0;
  idList[1][2]  =  9;
  idList[1][3]  =  3;
  idList[1][4]  = 17;
  idList[1][5]  = 25;

  int idOutList[] = {0, 7, 3, 4, 5, 6};
/*  char nameList[idUsed][10] = {"PP", "A0A0", "V0V0", "SS", "VKVK", "AKAK"}; */

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
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

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [ll_conn_x2dep] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[ll_conn_x2dep] T and L's must be set\n");
    usage();
  }

  T            = T_global;
  Thp1         = T/2 + 1;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  fprintf(stdout, "# [ll_conn_x2dep] [%2d] parameters:\n"\
                  "# [ll_conn_x2dep]       T            = %3d\n"\
		  "# [ll_conn_x2dep]       Tstart       = %3d\n"\
		  "# [ll_conn_x2dep]       l_LX_at      = %3d\n"\
		  "# [ll_conn_x2dep]       l_LXstart_at = %3d\n",
		  g_cart_id, T, Tstart, l_LX_at, l_LXstart_at);
  fflush(stdout);

  if(init_geometry() != 0) {
    fprintf(stderr, "[ll_conn_x2dep] ERROR from init_geometry\n");
    exit(1);
  }
  VOL3 = LX*LY*LZ;

  geometry();

  /**************************************
   * determine the source coordinates 
   **************************************/
  source_coords[0] = g_source_location/(LX*LY*LZ);
  source_coords[1] = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  source_coords[2] = (g_source_location%(LY*LZ)) / LZ;
  source_coords[3] = (g_source_location%LZ);
  fprintf(stdout, "# [ll_conn_x2dep] source location: (%d, %d, %d, %d)\n", 
    source_coords[0], source_coords[1], source_coords[2], source_coords[3]);

  /****************************************
   * make the H4 orbits
   ****************************************/
  status = make_x_orbits_3d(&h4_id, &h4_count, &h4_val, &h4_nc, &h4_rep);
  if(status != 0) {
    fprintf(stderr, "[ll_conn_x2dep] Error while creating h4-lists\n");
    exit(4);
  }

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  bytes = sizeof(double) * 2 * NumFlComb * K * (size_t)VOLUME;
  conn = (double*)malloc(bytes);
  if( (conn==(double*)NULL) ) {
    fprintf(stderr, "[ll_conn_x2dep] could not allocate memory for contr. fields\n");
    exit(3);
  }

  jjx = (double**)calloc(NumFlComb*idUsed, sizeof(double*));
  if( jjx==NULL ) {
    fprintf(stderr, "[ll_conn_x2dep] could not allocate memory for jjx\n");
    exit(2);
  }
  jjx[0] = (double*)calloc(2*NumFlComb*idUsed*h4_nc, sizeof(double));
  if( jjx[0]==NULL ) {
    fprintf(stderr, "[ll_conn_x2dep] could not allocate memory for jjx[0]\n");
    exit(3);
  }
  for(i=1; i<NumFlComb*idUsed; i++) {
    jjx[i] = jjx[i-1] + 2*h4_nc;
  }

  /***********************
   * read contractions   *
   ***********************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  sprintf(filename, "correl.%.4d.t%.2dx%.2dy%.2dz%.2d", Nconf, source_coords[0],
      source_coords[1], source_coords[2], source_coords[3]);
  fprintf(stdout, "# [ll_conn_x2dep] Reading data from file %s\n", filename);

  ofs = fopen(filename, "r");
  if(ofs == NULL) {
    fprintf(stderr, "[ll_conn_x2dep] Error, could not open file %s for reading\n", filename);
    exit(4);
  }
  size_t items = 2 * (size_t)NumFlComb * (size_t)K * (size_t)VOLUME;
  fprintf(stdout, "\n# [] trying to read %lu items of size %lu bytes\n", items, sizeof(double));
  if( fread(conn, sizeof(double), items, ofs) != items) {
    fprintf(stderr, "[ll_conn_x2dep] Error, could not read read proper amount of data\n");
    exit(5);
  }
  fclose(ofs);
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# [ll_conn_x2dep] time to read contractions: %e seconds\n", retime-ratime);


  byte_swap64(conn, 2*NumFlComb*K*VOLUME); 
  /***********************
   * fill the correlator
   ***********************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  for(ix=0; ix<2*NumFlComb*idUsed*h4_nc; ix++) jjx[0][ix] = 0.;

  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[x0][x1][x2][x3];
      iix = h4_id[ix];
      //for(i=0; i<2; i++) {
      //for(j=0; j<idUsed; j++) {
      //  jjx[i*idUsed+j][2*iix  ] += conn[2*( ix*4*K + i*16+idList[j] )  ];
      //  jjx[i*idUsed+j][2*iix+1] += conn[2*( ix*4*K + i*16+idList[j] )+1];
      //}}
      for(i=0; i<NumFlComb; i++) {
      for(j=0; j<idUsed; j++) {
        /********************************************
         ********************************************
         **
         ** remember: jjx[0,...,idUsed-1]        non-singlet currents
         **           jjx[idUsed,...,2*idUsed-1] singlet currents
         **
         ********************************************
         ********************************************/
        jjx[i*idUsed+j][2*iix  ] += conn[2*( ix*NumFlComb*K + i*K + idList[i][j] )  ];
        jjx[i*idUsed+j][2*iix+1] += conn[2*( ix*NumFlComb*K + i*K + idList[i][j] )+1];
      }}
    }}}
  }

  for(i=0;i<NumFlComb*idUsed;i++) {
    for(ix=0; ix<h4_nc; ix++) {
      jjx[i][2*ix  ] /= 2. * h4_count[ix];
      jjx[i][2*ix+1] /= 2. * h4_count[ix];
    }
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# [ll_conn_x2dep] time to fill correlator: %e seconds\n", retime-ratime);


  /*****************************************
   * write to file
   *****************************************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  for(i=0; i<NumFlComb; i++) {
    for(j=0;j<idUsed;j++) {
      switch(i) {
        case 0:
          sprintf(filename, "jj_x2_%.2d_s0.%.4d", idOutList[j], Nconf);
          break;
        case 1:
          sprintf(filename, "jj_x2_%.2d_s1.%.4d", idOutList[j], Nconf);
          break;
        default:
          fprintf(stderr, "Warning, what shall be done for flavor comb. %d?\n", i);
          break;
      }
      if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
        fprintf(stderr, "[ll_conn_x2dep] Error, could not open file %s for writing\n", filename);
        exit(6);
      }
      fprintf(stdout, "# [ll_conn_x2dep] writing data for %2d singlet current to file %s\n", j, filename);
      the_time = time(NULL);
      fprintf(ofs, "# %s", ctime(&the_time));
      fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", Nconf, T_global, LX, LY, LZ, g_kappa, g_mu);
      for(ix=0; ix<h4_nc; ix++) {
        fprintf(ofs, "%4d%4d%4d%4d%16.7e%16.7e%16.7e%16.7e%25.16e%25.16e%6d\n",
            h4_rep[ix][0], h4_rep[ix][1], h4_rep[ix][2], h4_rep[ix][3],
            h4_val[ix][0], h4_val[ix][1], h4_val[ix][2], h4_val[ix][3],
            jjx[i*idUsed+j][2*ix], jjx[i*idUsed+j][2*ix+1], h4_count[ix]);
      }
      fclose(ofs);
    }
  }

  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to write correlator %e seconds\n", retime-ratime);
#ifdef _UNDEF
#endif
  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  if(conn!=NULL) free(conn);
  if(jjx!=NULL) {
    if(jjx[0]!=NULL) free(jjx[0]);
    free(jjx);
  }

  free_geometry();
  finalize_x_orbits(&h4_id, &h4_count, &h4_val, &h4_rep);

  fprintf(stdout, "# %s# end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# %s# end of run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);
}
