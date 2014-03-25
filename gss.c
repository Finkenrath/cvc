/************************************************
 * gss.c
 *
 * PURPOSE:
 * - generate stochastic sources
 * - source types:
 *   0 point sources at source location and neighbours
 *   1 volume source
 *   2 spin diluted timeslice source
 * - noise type
 *   1 Gaussian noise
 *   2 Z2 noise
 * DONE:
 * - tested volume source part
 * - tested point source part
 * TODO:
 * - source types 0 and 1
 * - MPI
 * CHANGES:
 ************************************************/

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
#include "read_input_parser.h"
#include "ranlxd.h"

void usage() {
  fprintf(stdout, "Code to prepare stochastic sources\n");
  fprintf(stdout, "Usage:   generate_stochastic_sources [options]\n");
  fprintf(stdout, "Options:\n");
  fprintf(stdout, "         -o file prefix [default \"source\"]\n");
  exit(0);
}

int main(int argc, char **argv) {

  int c;
  unsigned long int ix, i1;
  int mu, i;
  int sx0, sx1, sx2, sx3;
  int filename_set   = 0;
  int index, nsource;
  int N_ape=0, N_Jacobi=0, timeslice=0, Nlong=-1;
  int have_source = 0;
  int shifted_positions=0;
  double alpha_ape=0., kappa_Jacobi = 0.;
  double ran[120];
  char filename[800];

#ifdef MPI
  MPI_Status status;
  MPI_Init(&argc,&argv);
#endif

  while ((c = getopt(argc, argv, "h?sf:i:a:n:l:K:t:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set = 1;
      break;
    case 'i':
      N_ape = atoi(optarg);
      break;
    case 'a':
      alpha_ape = atof(optarg);
      break;
    case 'n':
      N_Jacobi = atoi(optarg);
      break;
    case 'K':
      kappa_Jacobi = atof(optarg);
      break;
    case 'l':
      Nlong = atoi(optarg);
      break;
    case 's':
      shifted_positions = 1;
      break;
    case 't':
      timeslice = atoi(optarg);
      break;
    case '?':
    default:
      usage();
      break;
    }
  }


  /***********************
   * read the input file *
   ***********************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * some checks on the input data *
   *********************************/
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

#ifdef MPI
  T = T_global / g_nproc;
  if(T_global%g_nproc != 0) {
    fprintf(stderr, "Error, T_global not multiple of g_nproc; exit\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
  }
  Tstart = g_cart_id * T;
#else
  T = T_global;
  Tstart = 0;
#endif
  VOL3 = LX*LY*LZ;

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    return(102);
  }

  geometry();

  /*******************************************
   * determine, where the source is
   *******************************************/
  if(g_source_type==2) { /* timeslice sources */
    have_source = ( (Tstart<=timeslice) && (timeslice-Tstart<T) );
  } else if(g_source_type==0) { /* point source */
    sx0 = g_source_location / (LX*LY*LZ);
    sx1 = ( g_source_location % (LX*LY*LZ) ) / (LY*LZ);
    sx2 = ( g_source_location % (LY*LZ) ) / LZ;
    sx3 = g_source_location % LZ;
    if(g_cart_id==0) fprintf(stdout, "# global source coord.: t=%3d, x=%3d, y=%3d, z=%3d\n", sx0, sx1, sx2, sx3);
    if( (Tstart<=sx0) && (sx0-Tstart<T) ) { have_source=1;}
    else if( (Tstart<=(sx0+1)%T_global ) && ((sx0+1)%T_global-Tstart<T) ) {have_source=2;}
    else if( (Tstart<=(sx0-1+T_global)%T_global) && ((sx0-1+T_global)%T_global-Tstart<T) ) {have_source=3;}
  } else if (g_source_type==1) { /* volume source */
    have_source = 1;
  } else {
    if(g_cart_id==0) fprintf(stderr, "Error, source type not recognized\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
  }

  if(have_source==1) fprintf(stdout, "# process no. %d in g_cart_grid has have_source+%d\n", 
    g_cart_id, have_source);

  /* initialize random number generator */
  if(g_source_type==1) {
    /* special case of volume source */
  } else {
    if(have_source=1) {
      fprintf(stdout, "# Using seed %u\n", g_seed);
      rlxd_init(2, g_seed);
    }
  }

  if(g_cart_id==0) fprintf(stdout, "# Initialize stochastic sources with 0.\n");
  no_fields=4;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) {alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);
  for(i=0; i<no_fields; i++) {
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_zero(g_spinor_field[i]+_GSI(ix));
    }
  }
  
  if(g_source_type==0) {

    if(g_cart_id==0) fprintf(stdout, "Generating point source at source "\
      "position %d and shifted positions ...\n", g_source_location);
      switch(g_noise_type) {
        case 1: 
          if(rangauss(ran, 120)!=0) {
            fprintf(stderr, "Error on calling rangauss\n");
            exit(1);
          }
          for(i=0; i<120; i++) { ran[i] /= sqrt(2); }
          break;
        case 2: 
          if(ranz2(ran, 120)!=0) {
            fprintf(stderr, "Error on calling rangauss\n");
            exit(1);
          }
          break;
        default:
          usage();
          break;
      }

      /* first part: at source location  */
      index = _GSI(g_source_location);

      for(i1 = 0; i1 < 24; i1++) {
        g_spinor_field[0][index+i1] = ran[i1];
      }

      /* second part: at source location + mu */
      for(mu=0; mu<4; mu++) {
        index = _GSI( g_iup[g_source_location][mu] );
        for(i1 = 0; i1 < 24; i1++) {
          g_spinor_field[0][index+i1] = ran[(mu+1)*24+i1];
        }
      }

    } else if(g_source_type==1) {
      if(g_cart_id==0) fprintf(stdout, "\nGenerating stochastic volume source ...\n");

      for(ix = 0; ix < VOLUME; ix++) {
        switch(g_noise_type) {
          case 1: /* Gaussian noise */
            if(rangauss(ran, 24)!=0) {
              fprintf(stderr, "Error on calling rangauss\n");
              exit(1);
            }
            for(i=0; i<24; i++) ran[i] /= sqrt(2);
            break;
          case 2: /* Z2 noise */
            if(ranz2(ran, 24)!=0) {
              fprintf(stderr, "Error on calling rangauss\n");
              exit(1);
            }
            break;
          default:
            usage();
            break;
        }
    
        index = _GSI(ix);
        for(i1=0; i1<24; i1++) g_spinor_field[0][index+i1] = ran[i1];
      }
    } else if(g_source_type==2) {
      if(have_source==1) {
        fprintf(stdout, "# Generating spin diluted timeslice source"\
          " on timeslice %d\n", timeslice);
        sx0 = timeslice - Tstart;
        for(iix=0; iix<VOL3; iix++) {
          ix = sx0*VOL3 + iix;
          switch(g_noise_type) {
            case 1:
              rangauss(ran, 24);
              break;
            case 2:
              ranz2(ran, 24);
              break;
          }
          g_spinor_field[0][_GSI(ix)   ] = ran[ 0];
          g_spinor_field[0][_GSI(ix)+ 1] = ran[ 1];
          g_spinor_field[0][_GSI(ix)+ 2] = ran[ 2];
          g_spinor_field[0][_GSI(ix)+ 3] = ran[ 3];
          g_spinor_field[0][_GSI(ix)+ 4] = ran[ 4];
          g_spinor_field[0][_GSI(ix)+ 5] = ran[ 5];

          g_spinor_field[1][_GSI(ix)+ 6] = ran[ 6];
          g_spinor_field[1][_GSI(ix)+ 7] = ran[ 7];
          g_spinor_field[1][_GSI(ix)+ 8] = ran[ 8];
          g_spinor_field[1][_GSI(ix)+ 9] = ran[ 9];
          g_spinor_field[1][_GSI(ix)+10] = ran[10];
          g_spinor_field[1][_GSI(ix)+11] = ran[11];

          g_spinor_field[2][_GSI(ix)+12] = ran[12];
          g_spinor_field[2][_GSI(ix)+13] = ran[13];
          g_spinor_field[2][_GSI(ix)+14] = ran[14];
          g_spinor_field[2][_GSI(ix)+15] = ran[15];
          g_spinor_field[2][_GSI(ix)+16] = ran[16];
          g_spinor_field[2][_GSI(ix)+17] = ran[17];

          g_spinor_field[3][_GSI(ix)+18] = ran[18];
          g_spinor_field[3][_GSI(ix)+19] = ran[19];
          g_spinor_field[3][_GSI(ix)+20] = ran[20];
          g_spinor_field[3][_GSI(ix)+21] = ran[21];
          g_spinor_field[3][_GSI(ix)+22] = ran[22];
          g_spinor_field[3][_GSI(ix)+23] = ran[23];
        }


        fprintf(stdout, "finished generating source\n");
      } 
      for(i=0; i<4; i++) {
        write_lime_spinor();
      }
    }


    /******************************************************************
     * test: write source spinor field to standard out
     ******************************************************************/
/*
    for(it = 0; it < T; it++) {
    for(ix = 0; ix < LX; ix++) {
    for(iy = 0; iy < LY; iy++) {
    for(iz = 0; iz < LZ; iz++) {
      fprintf(stdout, "it =%3d, ix =%3d, iy =%3d, iz =%3d\n", it, ix, iy, iz);
      index = _GSI(g_ipt[it][ix][iy][iz]);
      fprintf(stdout, " (%f, %f)\t (%f, %f)\t (%f, %f)\n (%f, %f)\t (%f, %f)\t (%f, %f)\n"\
                      " (%f, %f)\t (%f, %f)\t (%f, %f)\n (%f, %f)\t (%f, %f)\t (%f, %f)\n",\
        g_spinor_field[0][index+ 0], g_spinor_field[0][index+ 1], 
        g_spinor_field[0][index+ 2], g_spinor_field[0][index+ 3], 
        g_spinor_field[0][index+ 4], g_spinor_field[0][index+ 5],
        g_spinor_field[0][index+ 6], g_spinor_field[0][index+ 7], 
        g_spinor_field[0][index+ 8], g_spinor_field[0][index+ 9], 
        g_spinor_field[0][index+10], g_spinor_field[0][index+11],
        g_spinor_field[0][index+12], g_spinor_field[0][index+13], 
        g_spinor_field[0][index+14], g_spinor_field[0][index+15], 
        g_spinor_field[0][index+16], g_spinor_field[0][index+17],
        g_spinor_field[0][index+18], g_spinor_field[0][index+19], 
        g_spinor_field[0][index+20], g_spinor_field[0][index+21], 
        g_spinor_field[0][index+22], g_spinor_field[0][index+23]);
    }
    }
    }
    }
*/
    sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, nsource);
    write_lime_spinor(g_spinor_field[0], filename, 0, 32);


  } /* loop on sources */

  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  return(EXIT_SUCCESS);
}

