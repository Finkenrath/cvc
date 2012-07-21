/****************************************************
 * ll_conn_nonsinglet.c
 *
 * Tue Jan 11 14:35:51 CET 2011
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
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
#include "fuzz.h"
#include "fuzz2.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"

void usage() {
  fprintf(stdout, "\n\nCode to perform contractions for LL connected contributions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -h, -? this help and exit\n");
  fprintf(stdout, "         -v verbose [no effect, lots of stdout output anyway]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -p <n> number of colours [default 1]\n\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}
/*
#ifdef _UNDEF
#undef _UNDEF
#endif
*/
#ifndef _UNDEF
#define _UNDEF
#endif


int main(int argc, char **argv) {
  
  int K=32;
  int c, i, j, k, ll, t, id;
  int count, coords[4], position=-1, position_set=0;
  size_t shift;
  long unsigned int VOL3;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, idx;
  int n_c=1, n_s=4;
  int *xgindex1=NULL, *xgindex2=NULL, *xisimag=NULL;
  int write_ascii=0;
  double *xvsign=NULL;
  double *cconn=NULL;
  double *buffer = (double*)NULL;
  int sigmalight=0, sigmaheavy=0;
  double correlator_norm = 1.;
  int source_coords[4];
  size_t prec = 64;

  int verbose = 0;
  char filename[200];
  double ratime, retime;
  double **chi=NULL, **psi=NULL;
  FILE *ofs=NULL, *ofs2=NULL;
  double c_conf_gamma_sign[]  = {1., 1., 1., -1., -1., -1., -1., 1., 1., 1., -1., -1.,  1.,  1., 1., 1.};
  double n_conf_gamma_sign[]  = {1., 1., 1., -1., -1., -1., -1., 1., 1., 1.,  1.,  1., -1., -1., 1., 1.};
  double *conf_gamma_sign=NULL;

  /**************************************************************************************************
   * charged stuff
   *
   * (pseudo-)scalar:
   * g5 - g5,	g5   - g0g5,	g0g5 - g5,	g0g5 - g0g5,
   * g0 - g0,	g5   - g0,	g0   - g5,	g0g5 - g0,
   * g0 - g0g5,	1    - 1,	1    - g5,	g5   - 1,
   * 1  - g0g5,	g0g5 - 1,	1    - g0,	g0   - 1
   *
   * (pseudo-)vector:
   * gig0 - gig0,	gi     - gi,		gig5 - gig5,	gig0   - gi,
   * gi   - gig0,	gig0   - gig5,		gig5 - gig0,	gi     - gig5,
   * gig5 - gi,		gig0g5 - gig0g5,	gig0 - gig0g5,	gig0g5 - gig0,
   * gi   - gig0g5,	gig0g5 - gi,		gig5 - gig0g5,	gig0g5 - gig5
   **************************************************************************************************/
  int gindex1[] = {5, 5, 6, 6, 0, 5, 0, 6, 0, 4, 4, 5, 4, 6, 4, 0,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 10, 11, 12, 1, 2, 3, 10, 11, 12, 7, 8, 9, 1, 2, 3, 7, 8, 9,
                   13, 14, 15, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13, 7, 8, 9, 15, 14, 13};

  int gindex2[] = {5, 6, 5, 6, 0, 0, 5, 0, 6, 4, 5, 4, 6, 4, 0, 4,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 1, 2, 3, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 1, 2, 3,
                   13, 14, 15, 15, 14, 13, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13, 7, 8, 9};

  /* due to twisting we have several correlators that are purely imaginary */
  int isimag[]  = {0, 0, 0, 0, 
                   0, 1, 1, 1, 
                   1, 0, 1, 1, 
                   1, 1, 0, 0,

                   0, 0, 0, 0, 
                   0, 1, 1, 1, 
                   1, 0, 1, 1, 
                   1, 1, 0, 0};

  double vsign[]  = {1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1., -1., 1.,   1., -1., 1., 
                     1., -1., 1.,   1., -1., 1.,   1., -1., 1.,   1., -1., 1.};


  /**************************************************************************************************
   * neutral stuff 
   *
   * (pseudo-)scalar:
   * g5 - g5,	g5   - g0g5,	g0g5 - g5,	g0g5 - g0g5,
   * 1  - 1,	g5   - 1,	1    - g5,	g0g5 - 1,
   * 1  - g0g5,	g0   - g0,	g0   - g5,	g5   - g0,
   * g0 - g0g5,	g0g5 - g0,	g0   - 1,	1    - g0
   *
   * (pseudo-)vector:
   * gig0   - gig0,		gi   - gi,	gig0g5 - gig0g5,	gig0   - gi, 
   * gi     - gig0,		gig0 - gig0g5,	gig0g5 - gig0,		gi     - gig0g5,
   * gig0g5 - gi		gig5 - gig5,	gig5   - gi,		gi     - gig5,
   * gig5   - gig0,		gig0 - gig5,	gig5   - gig0g5,	gig0g5 - gig5
   **************************************************************************************************/
  int ngindex1[] = {5, 5, 6, 6, 4, 5, 4, 6, 4, 0, 0, 5, 0, 6, 0, 4,
                    10, 11, 12, 1, 2, 3, 13, 14, 15, 10, 11, 12,  1,  2,  3, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13,
                     7,  8,  9, 7, 8, 9,  1,  2,  3,  7,  8,  9, 10, 11, 12,  7,  8,  9, 15, 14, 13};

  int ngindex2[] = {5, 6, 5, 6, 4, 4, 5, 4, 6, 0, 5, 0, 6, 0, 4, 0,
                    10, 11, 12, 1, 2, 3, 13, 14, 15,  1,  2,  3, 10, 11, 12, 15, 14, 13, 10, 11, 12, 15, 14, 13, 1, 2, 3,
                     7,  8,  9, 1, 2, 3,  7,  8,  9, 10, 11, 12,  7,  8,  9, 15, 14, 13,  7,  8, 9};

  int nisimag[]  = {0, 0, 0, 0,
                    0, 1, 1, 1,
                    1, 0, 1, 1,
                    1, 1, 0, 0,

                    0, 0, 0, 0,
                    0, 1, 1, 1, 
                    1, 0, 1, 1,
                    1, 1, 0, 0};

  double nvsign[] = {1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1., 
                     1.,  1., 1.,   1., -1., 1.,   1., -1., 1.,   1., -1., 1.,
                     1., -1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1., -1., 1.,   1., -1., 1. };
 

/*
  double isneg_std[]=    {+1., -1., +1., -1., +1., +1., +1., +1., -1., +1., +1., +1., +1., +1., +1., +1.,    
                          -1., +1., -1., -1., +1., +1., +1., -1., +1., -1., +1., +1., +1., +1., +1., +1.}; 
*/
  double isneg_std[]=    {+1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1.,    
                          +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1.};

  double isneg[32];


#ifdef MPI
  MPI_Status status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ah?vf:c:p:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      n_c = atoi(optarg);
      break;
    case 'p':
      position = atoi(optarg);
      position_set = 1;
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [] will write in ascii format\n");
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
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  if( Nlong > 0 ) {
    if(g_proc_id==0) fprintf(stdout, "Fuzzing not available in this version.\n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  VOL3 = LX*LY*LZ;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",
		  g_cart_id, g_cart_id, T_global,  g_cart_id, T,  g_cart_id, Tstart, 
                             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
                             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  if(g_cart_id==0) {
#ifdef MPI
#  ifdef HAVE_LIBLEMON
    fprintf(stdout, "# ll_conn with MPI and with LEMON\n");
#  else
    fprintf(stdout, "# ll_conn with MPI and wout LEMON\n");
#  endif
#else
    fprintf(stdout, "# ll_conn wout MPI\n");
#endif
    fprintf(stdout, "# number of colours = %d\n", n_c);
  }

  if(position_set == 0) {
    position = g_propagator_position;
    if(g_cart_id == 0) fprintf(stdout, "# using input file value for prop pos %d\n", position);
  } else {
    if(g_cart_id == 0) fprintf(stdout, "# using command line arg value for prop pos %d\n", position);
  }

  /*********************************************
   * set the isneg field
   *********************************************/
  for(i = 0; i < K; i++) isneg[i] = isneg_std[i];

  /*********************************************************
   * allocate memory for the spinor fields 
   *********************************************************/
//  no_fields = 2 * n_s * n_c;
  no_fields = n_s * n_c;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  if(g_cart_id==0) fprintf(stdout, "# no. of spinor fields is %d\n", no_fields);

#if !( (defined PARALLELTX) || (defined PARALLELTXY) )
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUME + RAND);
#else
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME + RAND);
#endif

  /*********************************************************
   * allocate memory for the contractions
   *********************************************************/
  cconn = (double*)calloc(2*K*VOLUME, sizeof(double));
  if( cconn==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for cconn\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 4);
    MPI_Finalize();
#endif
    exit(3);
  }

  buffer  = (double*)calloc(2*K*LZ, sizeof(double));
  if( buffer==(double*)NULL) {
    fprintf(stderr, "could not allocate memory for buffers\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 5);
    MPI_Finalize();
#endif
    exit(4);
  }

  /******************************************************************
   * calculate source coordinates
   ******************************************************************/
  source_coords[0] = g_source_location / (LX_global*LY_global*LZ);
  source_coords[1] = ( g_source_location % (LX_global*LY_global*LZ) ) / (LY_global*LZ);
  source_coords[2] = ( g_source_location % (LY_global*LZ) ) / LZ;
  source_coords[3] = g_source_location % LZ;
  if(g_cart_id==0) fprintf(stdout, "# source coords = %3d%3d%3d%3d\n", source_coords[0], source_coords[1],
    source_coords[2], source_coords[3]);

  /******************************************************************
   * final normalization of the correlators
   ******************************************************************/
/*  correlator_norm = 1. / ( 2. * g_kappa * g_kappa * (double)(LX_global*LY_global*LZ) );*/
  correlator_norm = 1.;
  if(g_cart_id==0) fprintf(stdout, "# correlator_norm = %12.5e\n", correlator_norm);

  /******************************************************************
   ******************************************************************
   **                                                              **
   **  local - local                                               **
   **                                                              **
   ******************************************************************
   ******************************************************************/
  if(g_cart_id==0) fprintf(stdout, "# Starting LL\n");

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  for(i=0; i<n_s*n_c; i++) {
    prepare_propagator2(source_coords, i,  1, g_spinor_field[i],         position, 0, prec);
//    prepare_propagator2(source_coords, i, -1, g_spinor_field[i+n_s*n_c], position, format);
  }
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time for preparing light prop.: %e seconds\n", retime-ratime);
  
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  for(idx=0; idx<2*K*VOLUME; idx++) cconn[idx] = 0.;

  count = -1;
//  for(sigmalight=1; sigmalight>=-1; sigmalight-=2) { 
//  for(sigmaheavy=1; sigmaheavy>=-1; sigmaheavy-=2) {
  for(sigmalight=1; sigmalight>=1; sigmalight-=2) { 
  for(sigmaheavy=1; sigmaheavy>=1; sigmaheavy-=2) {
    if(g_cart_id==0) fprintf(stdout, "\n# processing pair (%d, %d)\n", sigmalight, sigmaheavy);
    count++;
    chi = &g_spinor_field[( (1-sigmalight)/2 )*n_s*n_c];
    psi = &g_spinor_field[( (1-sigmaheavy)/2 )*n_s*n_c];
    if(sigmalight == sigmaheavy) {
      xgindex1 = gindex1;  xgindex2 = gindex2;  xisimag=isimag;  xvsign=vsign;  conf_gamma_sign = c_conf_gamma_sign;
    } else {
      xgindex1 = ngindex1; xgindex2 = ngindex2; xisimag=nisimag; xvsign=nvsign; conf_gamma_sign = n_conf_gamma_sign;
    }

    /* (pseudo-)scalar sector */
    for(idx=0; idx<16; idx++) {
      contract_twopoint_xdep(cconn + 2*(count*K + idx), xgindex1[idx], xgindex2[idx], chi, psi, n_c, K, 1.0, prec);
    }
    /* (pseudo-)vector sector */
    for(idx = 16; idx < 64; idx+=3) {
      for(i = 0; i < 3; i++) {
        contract_twopoint_xdep(cconn + 2*(count*K + (16+(idx-16)/3)), 
          xgindex1[idx+i], xgindex2[idx+i], chi, psi, n_c, K, conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i], prec);
      }
    }
  }}

  /***************************************************************
   * write contractions to file
   ***************************************************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) {
    sprintf(filename, "correl_nonsinglet.%.4d.t%.2dx%.2dy%.2dz%.2d", Nconf, source_coords[0], 
      source_coords[1], source_coords[2], source_coords[3]);

    ofs = fopen(filename, "w");
    if(ofs==NULL) {
      fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(7);
    }
    if(write_ascii) {
      sprintf(filename, "correl_nonsinglet.%.4d.t%.2dx%.2dy%.2dz%.2d.ascii", Nconf, source_coords[0], 
        source_coords[1], source_coords[2], source_coords[3]);
      ofs2 = fopen(filename, "w");
    }
  }
  for(x0=0; x0<T_global; x0++) {
    coords[0] = x0 / T;
  for(x1=0; x1<LX_global; x1++) {
    coords[1] = x1 / LX;
  for(x2=0; x2<LY_global; x2++) {
    coords[2] = x2 / LY;
    coords[3] = 0;
#ifdef MPI
    MPI_Cart_rank(g_cart_grid, coords, &id);
#else
    id = 0;
#endif
    shift = ( ( (x0 % T) * (size_t)LX + (x1 % LX) ) * LY + (x2 % LY) ) * LZ;

    if(g_cart_id==0) {
/*      fprintf(stdout, "# shfit(%3d,%3d,%3d) = %8d\n", x0, x1, x2, shift); */
      if(id == g_cart_id) {
        if( fwrite((void*)(cconn+shift*2*K), sizeof(double), 2*K*LZ, ofs) != 2*K*LZ ) {
          fprintf(stderr, "Error, could not write proper amount of data\n");
#ifdef MPI
          MPI_Abort(MPI_COMM_WORLD, 1);
          MPI_Finalize();
#endif
          exit(8);
        }

        if(write_ascii) {
          for(x3=0; x3<LZ; x3++) {
            count = -1;
            for(j=0; j<1; j++) {
            for(i=0; i<K; i++) {
              count++;
              fprintf(ofs2, "%3d%3d%3d%3d%3d%3d%8lu%25.16e%25.16e\n", j, i, x0, x1, x2, x3, shift,
                cconn[(shift+x3)*2*K+2*count], cconn[(shift+x3)*2*K+2*count+1]);
            }}
          }
        }

      }
#ifdef MPI
      else {
        MPI_Recv(buffer, 2*K*LZ, MPI_DOUBLE, id, shift, g_cart_grid, &status);
        if( fwrite((void*)(buffer), sizeof(double), 2*K*LZ, ofs) != 2*K*LZ ) {
          fprintf(stderr, "Error, could not write proper amount of data\n");
          MPI_Abort(MPI_COMM_WORLD, 1);
          MPI_Finalize();
          exit(8);
        }

        if(write_ascii) {
          for(x3=0; x3<LZ; x3++) {
            count = -1;
            for(j=0; j<1; j++) {
            for(i=0; i<K; i++) {
              count++;
              fprintf(ofs2, "%3d%3d%3d%3d%3d%3d%6lu%25.16e%25.16e\n", 
                j, i, x0, x1, x2, x3, shift, 
                buffer[x3*2*K+2*count], buffer[x3*2*K+2*count+1]);
            }}
          }
        }

      }
#endif
    }
#ifdef MPI
    else {
      if(g_cart_id == id) {
        MPI_Send(cconn+shift*2*K, 2*K*LZ, MPI_DOUBLE, 0, shift, g_cart_grid);
      }
    }
#endif
  }}}
  if(g_cart_id==0) {
    if(ofs  != NULL) fclose(ofs);
    if(ofs2 != NULL) fclose(ofs2);
  }

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time to write LL contractions: %e seconds\n", retime-ratime);


  if(g_cart_id==0) fprintf(stdout, "# finished LL contractions\n");

  /**************************************************
   * free the allocated memory, finalize 
   **************************************************/
  if(no_fields>0) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field); 
  }
  free_geometry(); 
  free(cconn);
  free(buffer); 
#ifdef MPI
  MPI_Finalize();
#endif
  return(0);

}
