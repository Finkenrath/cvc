/****************************************************
 * ll_conn_x2dep_extract_timeslice.c
 *
 * Mi 16. Nov 21:25:02 EET 2011
 *
 * PURPOSE
 * - like ll_conn_x2dep_extract, but works timeslice-wise
 * DONE:
 * TODO:
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
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to extract position space correlators from main file.\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {
  
  const int K      = 32;
  int idUsed = 6;
  int NumFlComb=2;
  size_t bytes, items, offset;
  int c, status, VOL3;
  int filename_set = 0;
  int x0, x1, x2, x3, i, j;
  unsigned int timeslice;
  int y0, y1, y2, y3, it;
  long unsigned int ix, iy;
  int source_coords[4];
  int force_byte_swap=0;
  int write_ascii = 0;
  double *conn    = NULL;
  double *jjx=NULL, *jjt=NULL;
  char filename[800];
  double ratime, retime;
  FILE *ofs=NULL;
  int *xisimag[2];
  size_t prec=64;
  void *buffer = NULL;

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
  int **idList= NULL;

  int idOutList[] = {0, 7, 3, 4, 5, 6};
/*  char nameList[idUsed][10] = {"PP", "A0A0", "V0V0", "SS", "VKVK", "AKAK"}; */

  while ((c = getopt(argc, argv, "abh?f:n:P:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'n':
      NumFlComb = atoi(optarg);
      fprintf(stdout, "\n# [] number of flavor combinations set to %d\n", NumFlComb);
      break;
    case 'b':
      force_byte_swap = 1;
      fprintf(stdout, "\n# [] will enforce byte swap\n");
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [] will write in ascii format\n");
      break;
    case 'P':
      prec = (size_t)atoi(optarg);
      fprintf(stdout, "\n# [] precision set to %lu\n", prec);
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* the global time stamp */
 g_the_time = time(NULL);
 fprintf(stdout, "\n# [ll_conn_x2dep_extract] using global time stamp %s", ctime(&g_the_time));

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [ll_conn_x2dep_extract] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[ll_conn_x2dep_extract] T and L's must be set\n");
    usage();
  }

  if(prec == 32) {
    fprintf(stderr, "[] Warning, resetting prec to 64\n");
    prec = 64;
  }

  T = T_global;
  VOL3 = LX*LY*LZ;

  if(init_geometry() != 0) {
    fprintf(stderr, "[ll_conn_x2dep_extract] ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /*************************************
   * init the id lists
   *************************************/
  idList = (int**)malloc(NumFlComb * sizeof(int*));
  if(idList==NULL) {
    fprintf(stderr, "\n[ll_conn_x2dep_extract] Error, could not alloc idList\n");
    exit(127);
  }
  idList[0] = (int*)malloc(NumFlComb*idUsed * sizeof(int));
  if(idList[0]==NULL) {
    fprintf(stderr, "\n[ll_conn_x2dep_extract] Error, could not alloc idList[0]\n");
    exit(127);
  }
  for(i=1;i<NumFlComb;i++) idList[i] = idList[i-1] + idUsed;

    // +1 +1
    idList[0][0]  =  0;
    idList[0][1]  =  9;
    idList[0][2]  =  3;
    idList[0][3]  =  4;
    idList[0][4]  = 18;
    idList[0][5]  = 17;
  if(NumFlComb>1) {
    // +1 -1
    idList[1][0]  =  4;
    idList[1][1]  =  0;
    idList[1][2]  =  9;
    idList[1][3]  =  3;
    idList[1][4]  = 17;
    idList[1][5]  = 25;
  }
  if(NumFlComb>2) {  
    // -1 +1
    idList[2][0]  =  4;
    idList[2][1]  =  0;
    idList[2][2]  =  9;
    idList[2][3]  =  3;
    idList[2][4]  = 17;
    idList[2][5]  = 25;
  }
  if(NumFlComb>3) {
    // -1 -1
    idList[3][0]  =  0;
    idList[3][1]  =  9;
    idList[3][2]  =  3;
    idList[3][3]  =  4;
    idList[3][4]  = 18;
    idList[3][5]  = 17;
  }

  /**************************************
   * determine the source coordinates 
   **************************************/
  source_coords[0] = g_source_location/(LX*LY*LZ);
  source_coords[1] = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  source_coords[2] = (g_source_location%(LY*LZ)) / LZ;
  source_coords[3] = (g_source_location%LZ);
  fprintf(stdout, "# [ll_conn_x2dep_extract] source location: (%d, %d, %d, %d)\n", 
    source_coords[0], source_coords[1], source_coords[2], source_coords[3]);

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  bytes = sizeof(double) * 2 * NumFlComb * K * (size_t)VOL3;
  buffer = malloc(bytes);
  if( (buffer==NULL) ) {
    fprintf(stderr, "[ll_conn_x2dep_extract] Error, could not allocate memory for contr. fields\n");
    exit(3);
  }

  jjx = (double*)calloc(2*VOL3, sizeof(double));
  if( jjx==NULL ) {
    fprintf(stderr, "\n[ll_conn_x2dep_extract] Error, could not allocate memory for jjx\n");
    exit(3);
  }

  jjt = (double*)malloc(2*T*NumFlComb*idUsed*sizeof(double));
  if(jjt==NULL) {
    fprintf(stderr, "\n[ll_conn_x2dep_extract] Error, could not alloc jjt\n");
    exit(8);
  }

  for(it=0;it<T_global;it++) {


    timeslice = ( it + source_coords[0] ) % T_global;
    /***********************
     * read contractions   *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    sprintf(filename, "correl.%.4d.t%.2dx%.2dy%.2dz%.2d", Nconf, source_coords[0],
        source_coords[1], source_coords[2], source_coords[3]);
    fprintf(stdout, "# [ll_conn_x2dep_extract] Reading data from file %s\n", filename);

    ofs = fopen(filename, "r");
    if(ofs == NULL) {
      fprintf(stderr, "[ll_conn_x2dep_extract] Error, could not open file %s for reading\n", filename);
      exit(4);
    }
    items = 2 * (size_t)NumFlComb * (size_t)K * (size_t)VOL3;
    bytes = (prec==64) ? sizeof(double) : sizeof(float);
    offset = 2 * (size_t)NumFlComb * (size_t)K * (size_t)VOL3 * (size_t)timeslice * bytes;
    
    fprintf(stdout, "\n# [ll_conn_x2dep_extract] trying to read %lu items of size %lu bytes with offset %lu\n", items, bytes, offset);

    if( fseek(ofs, offset, SEEK_SET) != 0 ) {
      fprintf(stderr, "[ll_conn_x2dep_extract] Error, could not seek file position\n");
      exit(6);
    }
      
    if( fread(buffer, bytes, items, ofs) != items) {
      fprintf(stderr, "[ll_conn_x2dep_extract] Error, could not read read proper amount of data\n");
      exit(5);
    }
    fclose(ofs);

    if(prec == 32) {
      // single to double
      for(i=items-1;i>=0;i--) {
        ((double*)buffer)[i] = ((float*)buffer)[i];
      }
    }
    conn = (double*)buffer;

    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# [ll_conn_x2dep_extract] time to read contractions: %e seconds\n", retime-ratime);

    /******************************************************
     * byte swap; check for necessity; had to be done
     *   on the HU PHYSIK pha machines, probably _NOT_
     *   at JSC since the original data was produced there
     ******************************************************/
    if(force_byte_swap) {
      fprintf(stdout, "# [] enforcing byte swap\n");
      byte_swap64_v2(conn, (unsigned int)2*NumFlComb*K*VOL3);
    }

    /******************************************************
     * fill the correlator
     ******************************************************/
    for(i=0; i<NumFlComb; i++) {
    for(j=0; j<idUsed; j++) {
      ratime = (double)clock() / CLOCKS_PER_SEC;

      // initialize jjx to zero
      for(ix=0; ix<VOL3*2; ix++) jjx[ix] = 0.;

      for(x1=0; x1<LX; x1++) {
        y1 = ( source_coords[1] + x1 ) % LX;
      for(x2=0; x2<LY; x2++) {
        y2 = ( source_coords[2] + x2 ) % LY;
      for(x3=0; x3<LZ; x3++) {
        y3 = ( source_coords[3] + x3 ) % LZ;
        ix  = g_ipt[0][x1][x2][x3];
        iy  = g_ipt[0][y1][y2][y3];
        jjx[2*ix  ] += conn[2*( iy*NumFlComb*K + i*K + idList[i][j] )  ];
        jjx[2*ix+1] += conn[2*( iy*NumFlComb*K + i*K + idList[i][j] )+1];
      }}}

      retime = (double)clock() / CLOCKS_PER_SEC;
      fprintf(stdout, "# [ll_conn_x2dep_extract] time to fill correlator: %e seconds\n", retime-ratime);

      /*****************************************
       * write to file
       *****************************************/
      ratime = (double)clock() / CLOCKS_PER_SEC;
      switch(i) {
        case 0:
          sprintf(filename, "jj_xdep_%.2d_tau31_0_tau32_0.%.4d", idOutList[j], Nconf);
          break;
        case 1:
          sprintf(filename, "jj_xdep_%.2d_tau31_0_tau32_1.%.4d", idOutList[j], Nconf);
          break;
        case 2:
          sprintf(filename, "jj_xdep_%.2d_tau31_1_tau32_0.%.4d", idOutList[j], Nconf);
          break;
        case 3:
          sprintf(filename, "jj_xdep_%.2d_tau31_1_tau32_1.%.4d", idOutList[j], Nconf);
          break;
        default:
          fprintf(stderr, "\n [ll_conn_x2dep_extract] Warning, what shall be done for flavor comb. %d?\n", i);
          break;
      }
      if(it==0) {
        ofs=fopen(filename, "w");
        fprintf(stdout, "\n# [ll_conn_x2dep_extract] writing data for (%d,%d)-%2d current to file %s\n", i/2, i%2, j, filename);
      } else {
        ofs=fopen(filename, "a");
        fprintf(stdout, "\n# [ll_conn_x2dep_extract] appending data for (%d,%d)-%2d current to file %s\n", i/2, i%2, j, filename);
      }
      if( ofs == NULL ) {
        fprintf(stderr, "\n[ll_conn_x2dep_extract] Error, could not open file %s for writing\n", filename);
        exit(6);
      }
      items = 2 * (size_t)VOL3;
      if( fwrite(jjx, sizeof(double), items, ofs) != items) {
        fprintf(stderr, "\n[ll_conn_x2dep_extract] Error, could not write proper amount of data to file %s\n", filename);
        exit(5);
      }
      fclose(ofs);

      if(write_ascii) {
        strcat(filename, ".ascii");
        if(it == 0) {
          ofs=fopen(filename, "w");
        } else {
          ofs=fopen(filename, "a");
        }
        if( ofs == NULL ) {
          fprintf(stderr, "\n[ll_conn_x2dep_extract] Error, could not open file %s for writing\n", filename);
          exit(7);
        }
        for(x1=0;x1<LX;x1++) {
        for(x2=0;x2<LY;x2++) {
        for(x3=0;x3<LZ;x3++) {
          ix = g_ipt[0][x1][x2][x3];
          fprintf(ofs, "%3d%3d%3d%3d%25.16e%25.16e\n", it,x1,x2,x3,jjx[2*ix], jjx[2*ix+1]);
        }}}
        fclose(ofs);
      } // of if write_ascii

      jjt[2*( (idUsed*i+j)*T + it)  ] = 0.;
      jjt[2*( (idUsed*i+j)*T + it)+1] = 0.;

      for(x1=0;x1<LX;x1++) {
      for(x2=0;x2<LY;x2++) {
      for(x3=0;x3<LZ;x3++) {
        ix = g_ipt[0][x1][x2][x3];
        jjt[2*((idUsed*i+j)*T+it)  ] += jjx[2*ix  ];
        jjt[2*((idUsed*i+j)*T+it)+1] += jjx[2*ix+1];
      }}}

    }}  // of j=0,...,idUsed-1 and i=0,...,NumFLComb-1

    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "\n# [ll_conn_x2dep_extract] time to write correlator %e seconds\n", retime-ratime);

  }  // of loop on timeslices

  
  for(i=0; i<NumFlComb; i++) {
  for(j=0; j<idUsed; j++) {
    switch(i) {
      case 0:
        sprintf(filename, "jj_tdep_%.2d_tau31_0_tau32_0.%.4d", idOutList[j], Nconf);
        break;
      case 1:
        sprintf(filename, "jj_tdep_%.2d_tau31_0_tau32_1.%.4d", idOutList[j], Nconf);
        break;
      case 2:
        sprintf(filename, "jj_tdep_%.2d_tau31_1_tau32_0.%.4d", idOutList[j], Nconf);
        break;
      case 3:
        sprintf(filename, "jj_tdep_%.2d_tau31_1_tau32_1.%.4d", idOutList[j], Nconf);
        break;
      default:
        fprintf(stderr, "\n [ll_conn_x2dep_extract] Warning, what shall be done for flavor comb. %d?\n", i);
        break;
    }
    if( (ofs=fopen(filename, "w")) == NULL ) {
      fprintf(stderr, "\n[ll_conn_x2dep_extract] Error, could not open file %s for writing\n", filename);
      exit(9);
    }
    fprintf(stdout, "\n# [ll_conn_x2dep_extract] writing t-data for (%d,%d)-%2d current to file %s\n", i/2, i%2, j, filename);
    ix = 2*(idUsed*i+j)*T;
    for(it=0;it<T;it++) {
      jjt[ix  ] /= (double)VOL3;
      jjt[ix+1] /= (double)VOL3;
      fprintf(ofs, "%2d 1 %2d%25.16e%25.16e%6d\n", idOutList[j], it, jjt[ix], jjt[ix+1], Nconf);
      ix += 2;
    }
    fclose(ofs);
  }}  // of j=0,...,idUsed-1 and i=0,...,NumFLComb-1

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  if(buffer!=NULL) free(buffer);
  if(jjx!=NULL) free(jjx);
  if(jjt!=NULL) free(jjt);

  free_geometry();

  fprintf(stdout, "\n# [ll_conn_x2dep_extract] %s# [ll_conn_x2dep_extract] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "\n# [ll_conn_x2dep_extract] %s# [ll_conn_x2dep_extract] end of run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);
}
