/*********************************************************************************
 * jc_ud_tr.c
 *
 * Mon Aug 30 15:48:55 CEST 2010
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 *********************************************************************************/

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
#include "contractions_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"

void usage() {
  fprintf(stdout, "Code to perform quark-disconnected conserved vector current contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu;
  int count        = 0;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, it, iy;
  int sid1, sid2, status, gid;
  int Thp1, Lhp1, nmom, shift[4], shift2[4], nperm;
  double *disc1=NULL, *disc2=NULL;
  double *work = NULL;
  double r2, fnorm;
  char filename[100];
  double ratime, retime;
  complex w;
  int *mom_tab=NULL, *mom_members=NULL, *mom_perm=NULL;
  FILE *ofs;

  int perm_tab_3[6][3];
  perm_tab_3[0][0] =  0; 
  perm_tab_3[0][1] =  1; 
  perm_tab_3[0][2] =  2;
  perm_tab_3[1][0] =  1; 
  perm_tab_3[1][1] =  2; 
  perm_tab_3[1][2] =  0;
  perm_tab_3[2][0] =  2; 
  perm_tab_3[2][1] =  0; 
  perm_tab_3[2][2] =  1;
  perm_tab_3[3][0] =  0; 
  perm_tab_3[3][1] =  2; 
  perm_tab_3[3][2] =  1;
  perm_tab_3[4][0] =  1; 
  perm_tab_3[4][1] =  0; 
  perm_tab_3[4][2] =  2;
  perm_tab_3[5][0] =  2; 
  perm_tab_3[5][1] =  1; 
  perm_tab_3[5][2] =  0;

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
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* jc_ud_tr\n");
  fprintf(stdout, "**************************************************\n\n");

  /* initialize */
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "#       T            = %3d\n"\
		  "#       Tstart       = %3d\n"\
		  "#       l_LX_at      = %3d\n"\
		  "#       l_LXstart_at = %3d\n"\
		  "#       FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, T, Tstart, l_LX_at, l_LXstart_at, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  Thp1 = T /2 + 1;
  Lhp1 = LX/2 + 1;
  nmom = 3*Lhp1 - 2;

  /****************************************
   * initialize the momenta
   ****************************************/
  mom_tab = (int*)calloc(3*nmom, sizeof(int));
  if( mom_tab==NULL) { 
    fprintf(stderr, "could not allocate memory for mom_tab\n");
    exit(4);
  }
  mom_tab[0] = 0; mom_tab[1] = 0; mom_tab[2] = 0;
  count=3;
  for(x1=1; x1<Lhp1; x1++) {
    mom_tab[count  ] = x1;
    mom_tab[count+1] = 0;
    mom_tab[count+2] = 0;
    mom_tab[count+3] = x1;
    mom_tab[count+4] = 1;
    mom_tab[count+5] = 0;
    mom_tab[count+6] = x1;
    mom_tab[count+7] = 1;
    mom_tab[count+8] = 1;
    count+=9;
  }

  mom_members = (int*)calloc(Thp1*nmom, sizeof(int));
  mom_perm    = (int*)calloc(nmom, sizeof(int));
  mom_perm[0] = 1;
  mom_perm[1] = 3;
  mom_perm[2] = 3;
  mom_perm[3] = 1;
  for (i=2; i<Lhp1; i++) {
    mom_perm[3*i-2] = 3;
    mom_perm[3*i-1] = 6;
    mom_perm[3*i  ] = 3;
  }
  for (i=0; i<nmom; i++)
    fprintf(stdout, "# %d\t(%d, %d, %d)\t%d\n", i, mom_tab[3*i], mom_tab[3*i+1], mom_tab[3*i+2], mom_perm[i]);

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  disc1  = (double*)calloc(8*VOLUME, sizeof(double));
  disc2  = (double*)calloc(8*VOLUME, sizeof(double));
  if( disc1==NULL || disc2==NULL) { 
    fprintf(stderr, "could not allocate memory for disc\n");
    exit(3);
  }

  work  = (double*)calloc(8*Thp1*nmom, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
    exit(3);
  }

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    for(ix=0; ix<8*Thp1*nmom; ix++) work[ix] = 0.;
    for(ix=0; ix<8*VOLUME; ix++) disc2[ix] = 0.;
    for (i=0; i<Thp1*nmom; i++) mom_members[i] = 0;

    /***********************************************
     * start loop on source id.s 
     ***********************************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(sid1=g_sourceid; sid1<=g_sourceid2; sid1+=g_sourceid_step) {

      sprintf(filename, "jc_ud_x.%.4d.%.4d", gid, sid1);
      if(read_lime_contraction(disc1, filename, 4, 0) != 0) break;
      for(ix=0; ix<8*VOLUME; ix++) disc2[ix] += disc1[ix];
      
      count=0;
      for (it=0; it<Thp1; it++) {
        shift[0] = it; shift2[0] = it;
        for (i=0; i<nmom; i++) {
          shift[1] = mom_tab[3*i  ];
          shift[2] = mom_tab[3*i+1];
          shift[3] = mom_tab[3*i+2];
          for (mu=0; mu<mom_perm[i]; mu++) {
            // fprintf(stdout, "# mom=%d,\tperm=%d\n", i, mom_perm[i]);
            shift2[1] = shift[perm_tab_3[mu][0]+1];
            shift2[2] = shift[perm_tab_3[mu][1]+1];
            shift2[3] = shift[perm_tab_3[mu][2]+1];
            for(x0=shift2[0]; x0<T; x0++) {
            for(x1=shift2[1]; x1<LX; x1++) {
            for(x2=shift2[2]; x2<LY; x2++) {
            for(x3=shift2[3]; x3<LZ; x3++) {
              ix = g_ipt[x0][x1][x2][x3];
              iy = g_ipt[x0-shift2[0]][x1-shift2[1]][x2-shift2[2]][x3-shift2[3]];
              // fprintf(stdout, "shift2=(%d,%d,%d,%d); x=(%d,%d,%d,%d); ix=%d, iy=%d\n",
              //   shift2[0], shift2[1],shift2[2],shift2[3], x0, x1, x2, x3, ix, iy);
              _co_eq_co_ti_co(&w, (complex*)(disc1+_GWI(0,ix,VOLUME)), (complex*)(disc1+_GWI(0,iy,VOLUME)));
              work[2*(            count)  ] -= w.re;
              work[2*(            count)+1] -= w.im;
              _co_eq_co_ti_co(&w, (complex*)(disc1+_GWI(1,ix,VOLUME)), (complex*)(disc1+_GWI(1,iy,VOLUME)));
              work[2*(  Thp1*nmom+count)  ] -= w.re;
              work[2*(  Thp1*nmom+count)+1] -= w.im;
              _co_eq_co_ti_co(&w, (complex*)(disc1+_GWI(2,ix,VOLUME)), (complex*)(disc1+_GWI(2,iy,VOLUME)));
              work[2*(2*Thp1*nmom+count)  ] -= w.re;
              work[2*(2*Thp1*nmom+count)+1] -= w.im;
              _co_eq_co_ti_co(&w, (complex*)(disc1+_GWI(3,ix,VOLUME)), (complex*)(disc1+_GWI(3,iy,VOLUME)));
              work[2*(3*Thp1*nmom+count)  ] -= w.re;
              work[2*(3*Thp1*nmom+count)+1] -= w.im;
            }}}}
          }
          count++;
        }
      }  /* of it=0,...,T/2 */
    }  /* of loop on sid1 */
    count=0;
    for (it=0; it<Thp1; it++) {
      shift[0] = it; shift2[0] = it;
      for (i=0; i<nmom; i++) {
        shift[1] = mom_tab[3*i  ];
        shift[2] = mom_tab[3*i+1];
        shift[3] = mom_tab[3*i+2];
        for (mu=0; mu<mom_perm[i]; mu++) {
          // fprintf(stdout, "# mom=%d,\tperm=%d\n", i, mom_perm[i]);
          shift2[1] = shift[perm_tab_3[mu][0]+1];
          shift2[2] = shift[perm_tab_3[mu][1]+1];
          shift2[3] = shift[perm_tab_3[mu][2]+1];
          for(x0=shift2[0]; x0<T; x0++) {
          for(x1=shift2[1]; x1<LX; x1++) {
          for(x2=shift2[2]; x2<LY; x2++) {
          for(x3=shift2[3]; x3<LZ; x3++) {
            ix = g_ipt[x0][x1][x2][x3];
            iy = g_ipt[x0-shift2[0]][x1-shift2[1]][x2-shift2[2]][x3-shift2[3]];
            // fprintf(stdout, "shift2=(%d,%d,%d,%d); x=(%d,%d,%d,%d); ix=%d, iy=%d\n",
            //   shift2[0], shift2[1],shift2[2],shift2[3], x0, x1, x2, x3, ix, iy);
            _co_eq_co_ti_co(&w, (complex*)(disc2+_GWI(0,ix,VOLUME)), (complex*)(disc2+_GWI(0,iy,VOLUME)));
            work[2*(            count)  ] += w.re;
            work[2*(            count)+1] += w.im;
            _co_eq_co_ti_co(&w, (complex*)(disc2+_GWI(1,ix,VOLUME)), (complex*)(disc2+_GWI(1,iy,VOLUME)));
            work[2*(  Thp1*nmom+count)  ] += w.re;
            work[2*(  Thp1*nmom+count)+1] += w.im;
            _co_eq_co_ti_co(&w, (complex*)(disc2+_GWI(2,ix,VOLUME)), (complex*)(disc2+_GWI(2,iy,VOLUME)));
            work[2*(2*Thp1*nmom+count)  ] += w.re;
            work[2*(2*Thp1*nmom+count)+1] += w.im;
            _co_eq_co_ti_co(&w, (complex*)(disc2+_GWI(3,ix,VOLUME)), (complex*)(disc2+_GWI(3,iy,VOLUME)));
            work[2*(3*Thp1*nmom+count)  ] += w.re;
            work[2*(3*Thp1*nmom+count)+1] += w.im;
            mom_members[count]++;
          }}}}
        }
        count++;
      }
    }  /* of it=0,...,T/2 */

    /* normalization */
    count=0;
    for (it=0; it<Thp1; it++) {
      for (i=0; i<nmom; i++) {
        fprintf(stdout, "%d\t%d\t%d\n", it, i, mom_members[count]);
        count++;
      }
    }

    for (mu=0; mu<4; mu++) {
      count=0;
      for (it=0; it<Thp1; it++) {
        for(i=0; i<nmom; i++) {
          fnorm = 1. / ( (double)mom_members[count]
            * (double)(g_sourceid2-g_sourceid+1) * (double)(g_sourceid2-g_sourceid) );
//          fprintf(stdout, "# fnorm(%d,%2d) = %25.16e\n", mu, count, fnorm);
          work[2*(mu*Thp1*nmom+count)  ] *= fnorm;
          work[2*(mu*Thp1*nmom+count)+1] *= fnorm;
          count++;
        }
      }
    }

    retime = (double)clock() / CLOCKS_PER_SEC;
    if(g_cart_id == 0) fprintf(stdout, "# time for building correl.: %e seconds\n", retime-ratime);


    /************************************************
     * save results
     ************************************************/
    sprintf(filename, "jc_ud_tr.%4d", gid);
    ofs = fopen(filename, "w");
    if (ofs==NULL) {
     fprintf(stderr, "Error, could not open file %s for writing\n", filename);
    }
    for(mu=0; mu<4; mu++) {
      count=0;
      for (it=0; it<Thp1; it++) {
        for (i=0; i<nmom; i++) {
          r2 = sqrt( mom_tab[3*i]*mom_tab[3*i] + mom_tab[3*i+1]*mom_tab[3*i+1]
                   + mom_tab[3*i+2]*mom_tab[3*i+2] );
          fprintf(ofs, "%3d%3d%3d%3d%16.7e%25.16e%25.16e\n", it, 
            mom_tab[3*i], mom_tab[3*i+1],mom_tab[3*i+2], r2, 
            work[2*(mu*Thp1*nmom+count)], work[2*(mu*Thp1*nmom+count)+1]);
          count++;
        }
      }
    }
    fclose(ofs);

  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(disc1);
  free(disc2);
  free(work);
  free(mom_tab);
  free(mom_perm);
  free(mom_members);

  return(0);

}
