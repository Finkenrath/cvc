/*************************************************************
 * make_H3orbits.c                                            *
 *************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif

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

#define _IDet3x3(A_) { \
    (A_)[0]*(A_)[4]*(A_)[8]  \
  + (A_)[1]*(A_)[5]*(A_)[6]  \
  + (A_)[2]*(A_)[3]*(A_)[7]  \
  - (A_)[0]*(A_)[5]*(A_)[7]  \
  - (A_)[1]*(A_)[3]*(A_)[8]  \
  - (A_)[2]*(A_)[4]*(A_)[6]  }

void init_perm_tabs(void) {

perm_tab_3[0][0] =  0; 
perm_tab_3[0][1] =  1; 
perm_tab_3[0][2] =  2;
perm_tab_3[1][0] =  0; 
perm_tab_3[1][1] =  2; 
perm_tab_3[1][2] =  1;
perm_tab_3[2][0] =  1; 
perm_tab_3[2][1] =  0; 
perm_tab_3[2][2] =  2;
perm_tab_3[3][0] =  1; 
perm_tab_3[3][1] =  2; 
perm_tab_3[3][2] =  0;
perm_tab_3[4][0] =  2; 
perm_tab_3[4][1] =  0; 
perm_tab_3[4][2] =  1;
perm_tab_3[5][0] =  2; 
perm_tab_3[5][1] =  1; 
perm_tab_3[5][2] =  0;

perm_tab_3_sign[0] =  1.;
perm_tab_3_sign[1] = -1.;
perm_tab_3_sign[2] = -1.;
perm_tab_3_sign[3] = 1.;
perm_tab_3_sign[4] = 1.;
perm_tab_3_sign[5] = -1.;

/********************************/

perm_tab_3e[0][0] =  0; 
perm_tab_3e[0][1] =  1; 
perm_tab_3e[0][2] =  2;

perm_tab_3e[1][0] =  1; 
perm_tab_3e[1][1] =  2; 
perm_tab_3e[1][2] =  0;

perm_tab_3e[2][0] =  2; 
perm_tab_3e[2][1] =  0; 
perm_tab_3e[2][2] =  1;

perm_tab_3o[0][0] =  0; 
perm_tab_3o[0][1] =  2; 
perm_tab_3o[0][2] =  1;

perm_tab_3o[1][0] =  2; 
perm_tab_3o[1][1] =  1; 
perm_tab_3o[1][2] =  0;

perm_tab_3o[2][0] =  1; 
perm_tab_3o[2][1] =  0; 
perm_tab_3o[2][2] =  2;

/********************************/

perm_tab_4[0][0] = 0;
perm_tab_4[0][1] = 1;
perm_tab_4[0][2] = 2;
perm_tab_4[0][3] = 3;

perm_tab_4[1][0] = 1;
perm_tab_4[1][1] = 2;
perm_tab_4[1][2] = 3;
perm_tab_4[1][3] = 0;

perm_tab_4[2][0] = 2;
perm_tab_4[2][1] = 3;
perm_tab_4[2][2] = 0;
perm_tab_4[2][3] = 1;

perm_tab_4[3][0] = 3;
perm_tab_4[3][1] = 0;
perm_tab_4[3][2] = 1;
perm_tab_4[3][3] = 2;

perm_tab_4[4][0] = 0;
perm_tab_4[4][1] = 2;
perm_tab_4[4][2] = 3;
perm_tab_4[4][3] = 1;

perm_tab_4[5][0] = 1;
perm_tab_4[5][1] = 0;
perm_tab_4[5][2] = 2;
perm_tab_4[5][3] = 3;

perm_tab_4[6][0] = 2;
perm_tab_4[6][1] = 3;
perm_tab_4[6][2] = 1;
perm_tab_4[6][3] = 0;

perm_tab_4[7][0] = 3;
perm_tab_4[7][1] = 1;
perm_tab_4[7][2] = 0;
perm_tab_4[7][3] = 2;

perm_tab_4[8][0] = 0;
perm_tab_4[8][1] = 3;
perm_tab_4[8][2] = 2;
perm_tab_4[8][3] = 1;

perm_tab_4[9][0] = 1;
perm_tab_4[9][1] = 0;
perm_tab_4[9][2] = 3;
perm_tab_4[9][3] = 2;

perm_tab_4[10][0] = 2;
perm_tab_4[10][1] = 1;
perm_tab_4[10][2] = 0;
perm_tab_4[10][3] = 3;

perm_tab_4[11][0] = 3;
perm_tab_4[11][1] = 2;
perm_tab_4[11][2] = 1;
perm_tab_4[11][3] = 0;

perm_tab_4[12][0] = 0;
perm_tab_4[12][1] = 3;
perm_tab_4[12][2] = 1;
perm_tab_4[12][3] = 2;

perm_tab_4[13][0] = 1;
perm_tab_4[13][1] = 2;
perm_tab_4[13][2] = 0;
perm_tab_4[13][3] = 3;

perm_tab_4[14][0] = 2;
perm_tab_4[14][1] = 0;
perm_tab_4[14][2] = 3;
perm_tab_4[14][3] = 1;

perm_tab_4[15][0] = 3;
perm_tab_4[15][1] = 1;
perm_tab_4[15][2] = 2;
perm_tab_4[15][3] = 0;

perm_tab_4[16][0] = 0;
perm_tab_4[16][1] = 2;
perm_tab_4[16][2] = 1;
perm_tab_4[16][3] = 3;

perm_tab_4[17][0] = 1;
perm_tab_4[17][1] = 3;
perm_tab_4[17][2] = 0;
perm_tab_4[17][3] = 2;

perm_tab_4[18][0] = 2;
perm_tab_4[18][1] = 1;
perm_tab_4[18][2] = 3;
perm_tab_4[18][3] = 0;

perm_tab_4[19][0] = 3;
perm_tab_4[19][1] = 0;
perm_tab_4[19][2] = 2;
perm_tab_4[19][3] = 1;

perm_tab_4[20][0] = 0;
perm_tab_4[20][1] = 1;
perm_tab_4[20][2] = 3;
perm_tab_4[20][3] = 2;

perm_tab_4[21][0] = 1;
perm_tab_4[21][1] = 3;
perm_tab_4[21][2] = 2;
perm_tab_4[21][3] = 0;

perm_tab_4[22][0] = 2;
perm_tab_4[22][1] = 0;
perm_tab_4[22][2] = 1;
perm_tab_4[22][3] = 3;

perm_tab_4[23][0] = 3;
perm_tab_4[23][1] = 2;
perm_tab_4[23][2] = 0;
perm_tab_4[23][3] = 1;

/********************************/

perm_tab_4e[0][0] = 0;
perm_tab_4e[0][1] = 1;
perm_tab_4e[0][2] = 2;
perm_tab_4e[0][3] = 3;

perm_tab_4e[1][0] = 0;
perm_tab_4e[1][1] = 2;
perm_tab_4e[1][2] = 3;
perm_tab_4e[1][3] = 1;

perm_tab_4e[2][0] = 0;
perm_tab_4e[2][1] = 3;
perm_tab_4e[2][2] = 1;
perm_tab_4e[2][3] = 2;

perm_tab_4e[3][0] = 1;
perm_tab_4e[3][1] = 0;
perm_tab_4e[3][2] = 3;
perm_tab_4e[3][3] = 2;

perm_tab_4e[4][0] = 1;
perm_tab_4e[4][1] = 2;
perm_tab_4e[4][2] = 0;
perm_tab_4e[4][3] = 3;

perm_tab_4e[5][0] = 1;
perm_tab_4e[5][1] = 3;
perm_tab_4e[5][2] = 2;
perm_tab_4e[5][3] = 0;

perm_tab_4e[6][0] = 2;
perm_tab_4e[6][1] = 0;
perm_tab_4e[6][2] = 1;
perm_tab_4e[6][3] = 3;

perm_tab_4e[7][0] = 2;
perm_tab_4e[7][1] = 1;
perm_tab_4e[7][2] = 3;
perm_tab_4e[7][3] = 0;

perm_tab_4e[8][0] = 2;
perm_tab_4e[8][1] = 3;
perm_tab_4e[8][2] = 0;
perm_tab_4e[8][3] = 1;

perm_tab_4e[9][0] = 3;
perm_tab_4e[9][1] = 0;
perm_tab_4e[9][2] = 2;
perm_tab_4e[9][3] = 1;

perm_tab_4e[10][0] = 3;
perm_tab_4e[10][1] = 1;
perm_tab_4e[10][2] = 0;
perm_tab_4e[10][3] = 2;

perm_tab_4e[11][0] = 3;
perm_tab_4e[11][1] = 2;
perm_tab_4e[11][2] = 1;
perm_tab_4e[11][3] = 0;

/*-------------------------*/

perm_tab_4o[0][0] = 0;
perm_tab_4o[0][1] = 2;
perm_tab_4o[0][2] = 1;
perm_tab_4o[0][3] = 3;

perm_tab_4o[1][0] = 0;
perm_tab_4o[1][1] = 3;
perm_tab_4o[1][2] = 2;
perm_tab_4o[1][3] = 1;

perm_tab_4o[2][0] = 0;
perm_tab_4o[2][1] = 1;
perm_tab_4o[2][2] = 3;
perm_tab_4o[2][3] = 2;

perm_tab_4o[3][0] = 1;
perm_tab_4o[3][1] = 3;
perm_tab_4o[3][2] = 0;
perm_tab_4o[3][3] = 2;

perm_tab_4o[4][0] = 1;
perm_tab_4o[4][1] = 0;
perm_tab_4o[4][2] = 2;
perm_tab_4o[4][3] = 3;

perm_tab_4o[5][0] = 1;
perm_tab_4o[5][1] = 2;
perm_tab_4o[5][2] = 3;
perm_tab_4o[5][3] = 0;

perm_tab_4o[6][0] = 2;
perm_tab_4o[6][1] = 1;
perm_tab_4o[6][2] = 0;
perm_tab_4o[6][3] = 3;

perm_tab_4o[7][0] = 2;
perm_tab_4o[7][1] = 3;
perm_tab_4o[7][2] = 1;
perm_tab_4o[7][3] = 0;

perm_tab_4o[8][0] = 2;
perm_tab_4o[8][1] = 0;
perm_tab_4o[8][2] = 3;
perm_tab_4o[8][3] = 1;

perm_tab_4o[9][0] = 3;
perm_tab_4o[9][1] = 2;
perm_tab_4o[9][2] = 0;
perm_tab_4o[9][3] = 1;

perm_tab_4o[10][0] = 3;
perm_tab_4o[10][1] = 0;
perm_tab_4o[10][2] = 1;
perm_tab_4o[10][3] = 2;

perm_tab_4o[11][0] = 3;
perm_tab_4o[11][1] = 1;
perm_tab_4o[11][2] = 2;
perm_tab_4o[11][3] = 0;

/********************************/
}

void set_qid_val(double **qid_val, int ts) {

  int x1, x2, x3, nqhat=-1;
  int L = LX;
  int Lhp1 = L/2+1;
  double q[4];

  q[0] = 2. * sin( M_PI * (double)ts / (double)T );
  for(x1=0; x1<Lhp1; x1++) {
  for(x2=0; x2<=x1;  x2++) {
  for(x3=0; x3<=x2;  x3++) {
    nqhat++;
    q[1] = 2. * sin( M_PI * (double)x1 / (double)L );
    q[2] = 2. * sin( M_PI * (double)x2 / (double)L );
    q[3] = 2. * sin( M_PI * (double)x3 / (double)L );

    (qid_val)[0][nqhat] = _sqr(q[0])+_sqr(q[1])+_sqr(q[2])+_sqr(q[3]);
    (qid_val)[1][nqhat] = _qrt(q[0])+_qrt(q[1])+_qrt(q[2])+_qrt(q[3]);
    (qid_val)[2][nqhat] = _hex(q[0])+_hex(q[1])+_hex(q[2])+_hex(q[3]);
    (qid_val)[3][nqhat] = _oct(q[0])+_oct(q[1])+_oct(q[2])+_oct(q[3]);
  }
  }
  }
}


int make_H3orbits(int **qid, int **qid_count, double ***qid_val, int *nc) {

  int it, ix, iy, iz, iix, n;
  int i0, i1, i2, i3;
  int isigma0, isigma1, isigma2, isigma3;
  int nqhat = -1;
  int Thp1  = T/2+1;
  int L = LX;
  int Lhp1  = L/2+1;
  int iq[4];
  int iq_perm[3], index_s;
  double q[4], qsq, q4;

  /* Nclasses = 1/48 * (L+2)(L+4)(L+6) */
  int Nclasses = (L*L*L+12*L*L+44*L+48)/48; 
  fprintf(stdout, "# Nclasses = %d x %d\n", Thp1, Nclasses);

  /* initialize the permutation tables */
  fprintf(stdout, "# initializing perm tabs\n");
  init_perm_tabs();


  fprintf(stdout, "# allocating memory for qid_count\n");
  *qid_count = (int *)calloc(Thp1 * Nclasses, sizeof(int));
  if(*qid_count==(int*)NULL) return(301);

  fprintf(stdout, "# allocating memory for qid_val\n");
  *qid_val = (double **)calloc(4, sizeof(double*));
  (*qid_val)[0] = (double*)calloc(4 * Thp1 * Nclasses, sizeof(double));
  if((*qid_val)[0]==(double*)NULL) return(302);
  (*qid_val)[1] = (*qid_val)[0] + Thp1 * Nclasses;
  (*qid_val)[2] = (*qid_val)[1] + Thp1 * Nclasses;
  (*qid_val)[3] = (*qid_val)[2] + Thp1 * Nclasses;

  fprintf(stdout, "# setting new fields to zero\n");
  for(it=0; it<Thp1*Nclasses; it++) {
    (*qid_count)[it]  =  0;
    (*qid_val)[0][it] = -1.;
    (*qid_val)[1][it] = -1.;
    (*qid_val)[2][it] = -1.;
    (*qid_val)[3][it] = -1.;
  }

  if((*qid) == (int*)NULL) {
    fprintf(stdout, "# allocating memory for qid & setting to zero\n");
    *qid = (int *)calloc(VOLUME, sizeof(int));
    if(*qid==(int*)NULL) return(303);
    for(ix=0; ix<VOLUME; ix++) (*qid)[ix] = -1;
  }

#ifdef _UNDEF
  for(it=0; it<Thp1; it++) {
    nqhat = -1;
    for(ix=0; ix<Lhp1; ix++) {
      for(iy=0; iy<=ix; iy++) {
      for(iz=0; iz<=iy; iz++) {
        nqhat++;
	q[0] = 2. * sin( M_PI * (double)it / (double)T );
	q[1] = 2. * sin( M_PI * (double)ix / (double)L );
	q[2] = 2. * sin( M_PI * (double)iy / (double)L );
	q[3] = 2. * sin( M_PI * (double)iz / (double)L );

        (*qid_val)[0][it*Nclasses+nqhat] = _sqr(q[0])+_sqr(q[1])+_sqr(q[2])+_sqr(q[3]);
        (*qid_val)[1][it*Nclasses+nqhat] = _qrt(q[0])+_qrt(q[1])+_qrt(q[2])+_qrt(q[3]);
        (*qid_val)[2][it*Nclasses+nqhat] = _hex(q[0])+_hex(q[1])+_hex(q[2])+_hex(q[3]);
        (*qid_val)[3][it*Nclasses+nqhat] = _oct(q[0])+_oct(q[1])+_oct(q[2])+_oct(q[3]);

	iq[0] = it; iq[1] = ix; iq[2] = iy; iq[3] = iz;
/*
        fprintf(stdout, "=====================================\n");
        fprintf(stdout, "it=%3d, ix=%3d, iy=%3d, iz=%3d, nqhat=%3d, id=%3d\n", it, ix, iy, iz, nqhat, it*Nclasses+nqhat);
*/
	for(n=0; n<6; n++) { 
	  iq_perm[0] = iq[perm_tab_3[n][0]+1];
	  iq_perm[1] = iq[perm_tab_3[n][1]+1];
	  iq_perm[2] = iq[perm_tab_3[n][2]+1];
/*          fprintf(stdout, "%3d, %3d, %3d, %3d\n", iq[0], iq_perm[0], iq_perm[1], iq_perm[2]); */
          for(isigma0=0; isigma0<=1; isigma0++) {
            i0 = ( (1-isigma0)*iq[0] + isigma0*(T-iq[0]) ) % T;
          for(isigma1=0; isigma1<=1; isigma1++) {
            i1 = ( (1-isigma1)*iq_perm[0] + isigma1*(L-iq_perm[0]) ) % L;
          for(isigma2=0; isigma2<=1; isigma2++) {
            i2 = ( (1-isigma2)*iq_perm[1] + isigma2*(L-iq_perm[1]) ) % L;
          for(isigma3=0; isigma3<=1; isigma3++) {
            i3 = ( (1-isigma3)*iq_perm[2] + isigma3*(L-iq_perm[2]) ) % L;
	    index_s = g_ipt[i0][i1][i2][i3];

/*            fprintf(stdout, "-------------------------------------\n"); */
/*            fprintf(stdout, "%3d, %3d, %3d, %3d; %5d, %5d\n", i0, i1, i2, i3, index_s, (*qid)[index_s]); */

	    if((*qid)[index_s] == -1) {
              (*qid)[index_s] = it*Nclasses + nqhat;
	      (*qid_count)[it*Nclasses+nqhat] +=1;
	    }

          }}}}
	}  
      }   
      } 
    }  
  }   
#endif
  /************************
   * test: print the lists 
   ************************/
/*
  for(it=0; it<T; it++) {
  for(ix=0; ix<L; ix++) {
  for(iy=0; iy<L; iy++) {
  for(iz=0; iz<L; iz++) {
    index_s = g_ipt[it][ix][iy][iz];
    q[0] = 2. * sin( M_PI * (double)it / (double)T );
    q[1] = 2. * sin( M_PI * (double)ix / (double)L );
    q[2] = 2. * sin( M_PI * (double)iy / (double)L );
    q[3] = 2. * sin( M_PI * (double)iz / (double)L );
    qsq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    q4 = q[0]*q[0]*q[0]*q[0] + q[1]*q[1]*q[1]*q[1] + 
         q[2]*q[2]*q[2]*q[2] + q[3]*q[3]*q[3]*q[3]; 

    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d, qid=%8d\n", it, ix, iy, iz, (*qid)[index_s]);
  }
  }
  }
  }



  for(it=0; it<Thp1; it++) {
    for(n=0; n<Nclasses; n++) {
      fprintf(stdout, "t=%3d, n=%5d, qid_count=%8d, qid_val= %12.5e%12.5e%12.5e%12.5e\n",
        it, n, (*qid_count)[it*Nclasses+n], (*qid_val)[0][it*Nclasses+n], (*qid_val)[1][it*Nclasses+n],
        (*qid_val)[2][it*Nclasses+n],(*qid_val)[3][it*Nclasses+n]);
    }
  }
*/
  *nc = Thp1 * Nclasses;

  return(0);

}

/********************************************************************************/

int make_H4orbits(int **qid, int **qid_count, double ***qid_val, int *nc) {

  int it, ix, iy, iz, iix, iit, n;
  int i0, i1, i2, i3;
  int isigma0, isigma1, isigma2, isigma3;
  int nqhat = -1;
  int Thp1  = T/2+1;
  int L = LX;
  int Lhp1  = L/2+1;
  int iq[4];
  int iq_perm[4], index_s;
  double q[4], qsq, q4;

  /* Nclasses = 1/4!/2^4 * (L+2)(L+4)(L+6)(L+8)  + the ones with odd t */
  int Nclasses = (L+2)*(L+4)*(L+6)*(L+8)/24/16 + T/4 * (L+2)*(L+4)*(L+6)/6/8;
  fprintf(stdout, "# Nclasses = %d\n", Nclasses);

  fprintf(stdout, "# allocating memory for qid_count\n");
  *qid_count = (int *)calloc(Nclasses, sizeof(int));

  fprintf(stdout, "# allocating memory for qid_val\n");
  *qid_val    = (double **)calloc(4, sizeof(double*));
  *(*qid_val) = (double * )calloc(4 * Nclasses, sizeof(double));
  (*qid_val)[1] = (*qid_val)[0] + Nclasses;
  (*qid_val)[2] = (*qid_val)[1] + Nclasses;
  (*qid_val)[3] = (*qid_val)[2] + Nclasses;

  fprintf(stdout, "# setting new fields to zero\n");
  for(it=0; it<Nclasses; it++) {
    (*qid_count)[it]  =  0;
    (*qid_val)[0][it] = -1.;
    (*qid_val)[1][it] = -1.;
    (*qid_val)[2][it] = -1.;
    (*qid_val)[3][it] = -1.;
  }

  if((*qid) == (int*)NULL) {
    fprintf(stdout, "# allocating memory for qid & setting to zero\n");
    *qid = (int *)calloc(VOLUME, sizeof(int));
    for(ix=0; ix<VOLUME; ix++) (*qid)[ix] = -1;
  }

  /* initialize the permutation tables */
  fprintf(stdout, "# initializing perm tabs\n");
  init_perm_tabs();

  fprintf(stdout, "# starting coordinate loops\n");
  nqhat = -1;
  for(it=0; it<Thp1; it++) {
    if(it%2==1) {
      for(ix=0; ix<Lhp1; ix++) {
        for(iy=0; iy<=ix; iy++) {
        for(iz=0; iz<=iy; iz++) {
          nqhat++;
  	  q[0] = 2. * sin( M_PI * (double)it / (double)T );
	  q[1] = 2. * sin( M_PI * (double)ix / (double)L );
	  q[2] = 2. * sin( M_PI * (double)iy / (double)L );
	  q[3] = 2. * sin( M_PI * (double)iz / (double)L );

          (*qid_val)[0][nqhat] = _sqr(q[0])+_sqr(q[1])+_sqr(q[2])+_sqr(q[3]);
          (*qid_val)[1][nqhat] = _qrt(q[0])+_qrt(q[1])+_qrt(q[2])+_qrt(q[3]);
          (*qid_val)[2][nqhat] = _hex(q[0])+_hex(q[1])+_hex(q[2])+_hex(q[3]);
          (*qid_val)[3][nqhat] = _oct(q[0])+_oct(q[1])+_oct(q[2])+_oct(q[3]);

	  iq[0] = it; iq[1] = ix; iq[2] = iy; iq[3] = iz;
/*
          fprintf(stdout, "=====================================\n");
          fprintf(stdout, "odd it=%3d, ix=%3d, iy=%3d, iz=%3d, nqhat=%3d\n", it, ix, iy, iz, nqhat);
*/
          /* go through all permutations */
	  for(n=0; n<6; n++) { 
  	    iq_perm[0] = iq[perm_tab_3[n][0]+1];
	    iq_perm[1] = iq[perm_tab_3[n][1]+1];
	    iq_perm[2] = iq[perm_tab_3[n][2]+1];
/*
            fprintf(stdout, "-------------------------------------\n");
            fprintf(stdout, "%3d, %3d, %3d, %3d\n", iq[0], iq_perm[0], iq_perm[1], iq_perm[2]);
*/
            for(isigma0=0; isigma0<=1; isigma0++) {
              i0 = ( (1-isigma0)*iq[0] + isigma0*(T-iq[0]) ) % T;
            for(isigma1=0; isigma1<=1; isigma1++) {
              i1 = ( (1-isigma1)*iq_perm[0] + isigma1*(L-iq_perm[0]) ) % L;
            for(isigma2=0; isigma2<=1; isigma2++) {
              i2 = ( (1-isigma2)*iq_perm[1] + isigma2*(L-iq_perm[1]) ) % L;
            for(isigma3=0; isigma3<=1; isigma3++) {
              i3 = ( (1-isigma3)*iq_perm[2] + isigma3*(L-iq_perm[2]) ) % L;
  	      index_s = g_ipt[i0][i1][i2][i3];
/*              fprintf(stdout, "%3d, %3d, %3d, %3d; %5d, %5d\n", i0, i1, i2, i3, index_s, (*qid)[index_s]); */
  	      if((*qid)[index_s] == -1) {
                (*qid)[index_s] = nqhat;
	        (*qid_count)[nqhat] +=1;
	      }
            }
            }
            }
            }
	  }  /* end of n = 0, ..., 5 */
        }    /* of iz */
        }    /* of iy */
      }      /* of ix */
    }        /* of if it % 2 == 1 */
    else {   /* it is even */
      iit = it / 2;
      for(ix=0; ix<=iit; ix++) {
      for(iy=0; iy<=ix;  iy++) {
      for(iz=0; iz<=iy;  iz++) {
        nqhat++;
	q[0] = 2. * sin( M_PI * (double)it / (double)T );
	q[1] = 2. * sin( M_PI * (double)ix / (double)L );
	q[2] = 2. * sin( M_PI * (double)iy / (double)L );
	q[3] = 2. * sin( M_PI * (double)iz / (double)L );

        (*qid_val)[0][nqhat] = _sqr(q[0])+_sqr(q[1])+_sqr(q[2])+_sqr(q[3]);
        (*qid_val)[1][nqhat] = _qrt(q[0])+_qrt(q[1])+_qrt(q[2])+_qrt(q[3]);
        (*qid_val)[2][nqhat] = _hex(q[0])+_hex(q[1])+_hex(q[2])+_hex(q[3]);
        (*qid_val)[3][nqhat] = _oct(q[0])+_oct(q[1])+_oct(q[2])+_oct(q[3]);

	iq[0] = iit; iq[1] = ix; iq[2] = iy; iq[3] = iz;
/*
        fprintf(stdout, "=====================================\n");
        fprintf(stdout, "even it=%3d, ix=%3d, iy=%3d, iz=%3d, nqhat=%3d\n", it, ix, iy, iz, nqhat);
*/
        for(n=0; n<24; n++) {
  	  iq_perm[0] = iq[perm_tab_4[n][0]];
	  iq_perm[1] = iq[perm_tab_4[n][1]];
	  iq_perm[2] = iq[perm_tab_4[n][2]];
	  iq_perm[3] = iq[perm_tab_4[n][3]];
/*
          fprintf(stdout, "-------------------------------------\n");
          fprintf(stdout, "%3d, %3d, %3d, %3d\n", iq_perm[0], iq_perm[1], iq_perm[2], iq_perm[3]);
*/
          for(isigma0=0; isigma0<=1; isigma0++) {
            i0 = ( 2*(1-isigma0)*iq_perm[0] + isigma0*(T-2*iq_perm[0]) ) % T;
          for(isigma1=0; isigma1<=1; isigma1++) {
            i1 = ( (1-isigma1)*iq_perm[1] + isigma1*(L-iq_perm[1]) ) % L;
          for(isigma2=0; isigma2<=1; isigma2++) {
            i2 = ( (1-isigma2)*iq_perm[2] + isigma2*(L-iq_perm[2]) ) % L;
          for(isigma3=0; isigma3<=1; isigma3++) {
            i3 = ( (1-isigma3)*iq_perm[3] + isigma3*(L-iq_perm[3]) ) % L;
  	    index_s = g_ipt[i0][i1][i2][i3];
/*            fprintf(stdout, "%3d, %3d, %3d, %3d; %5d, %5d\n", i0, i1, i2, i3, index_s, (*qid)[index_s]); */
  	    if((*qid)[index_s] == -1) {
              (*qid)[index_s] = nqhat;
	      (*qid_count)[nqhat] +=1;
	    }
          }
          }
          }
          }
        }    /* of n = 0, ..., 23  */
      }
      }
      }
    }        /* of else branch in if it % 2 == 1 */
  }          /* of it */
  fprintf(stdout, "# finished coordinate loops\n");

  /************************
   * test: print the lists 
   ************************/
/*
  for(it=0; it<T; it++) {
  for(ix=0; ix<L; ix++) {
  for(iy=0; iy<L; iy++) {
  for(iz=0; iz<L; iz++) {
    index_s = g_ipt[it][ix][iy][iz];
    q[0] = 2. * sin( M_PI * (double)it / (double)T );
    q[1] = 2. * sin( M_PI * (double)ix / (double)L );
    q[2] = 2. * sin( M_PI * (double)iy / (double)L );
    q[3] = 2. * sin( M_PI * (double)iz / (double)L );
    qsq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    q4 = q[0]*q[0]*q[0]*q[0] + q[1]*q[1]*q[1]*q[1] + 
         q[2]*q[2]*q[2]*q[2] + q[3]*q[3]*q[3]*q[3]; 

    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d, qid=%8d\n", it, ix, iy, iz, (*qid)[index_s]);
  }
  }
  }
  }

  fprintf(stdout, "# n\tqid_count\tqid_val\n");
  for(n=0; n<Nclasses; n++) {
    fprintf(stdout, "%5d\t%8d\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      n, (*qid_count)[n], (*qid_val)[0][n], (*qid_val)[1][n],(*qid_val)[2][n],(*qid_val)[3][n]);
  }
*/

  *nc = Nclasses;

  return(0);

}

/********************************************************************************/

int make_H3orbits_timeslice(int **qid, int **qid_count, double ***qid_val, int *nc) {

  int it, ix, iy, iz, n;
  int i1, i2, i3;
  int isigma1, isigma2, isigma3;
  int nqhat = -1;
  int L = LX, VOL3 = LX*LY*LZ;
  int Lhp1  = L/2+1;
  int iq[4];
  int iq_perm[3], index_s;

  int Nclasses = (L*L*L+12*L*L+44*L+48)/48; 
  /* Nclasses = 1/48 * (L+2)(L+4)(L+6) */
  fprintf(stdout, "# Nclasses = %d\n", Nclasses);

  /* initialize the permutation tables */
  fprintf(stdout, "# initializing perm tabs\n");
  init_perm_tabs();

  fprintf(stdout, "# allocating memory for qid_count\n");
  *qid_count = (int *)calloc(Nclasses, sizeof(int));
  if(*qid_count==(int*)NULL) return(301);

  fprintf(stdout, "# allocating memory for qid_val\n");
  *qid_val = (double **)calloc(4, sizeof(double*));
  (*qid_val)[0] = (double*)calloc(4 * Nclasses, sizeof(double));
  if((*qid_val)[0]==(double*)NULL) return(302);
  (*qid_val)[1] = (*qid_val)[0] + Nclasses;
  (*qid_val)[2] = (*qid_val)[1] + Nclasses;
  (*qid_val)[3] = (*qid_val)[2] + Nclasses;

  fprintf(stdout, "# setting new fields to zero\n");
  for(it=0; it<Nclasses; it++) {
    (*qid_count)[it]  =  0;
    (*qid_val)[0][it] = -1.;
    (*qid_val)[1][it] = -1.;
    (*qid_val)[2][it] = -1.;
    (*qid_val)[3][it] = -1.;
  }

  if((*qid) == (int*)NULL) {
    fprintf(stdout, "# allocating memory for qid & setting to zero\n");
    *qid = (int *)calloc(VOL3, sizeof(int));
    if(*qid==(int*)NULL) return(303);
    for(ix=0; ix<VOL3; ix++) (*qid)[ix] = -1;
  }

    nqhat = -1;
    for(ix=0; ix<Lhp1; ix++) {
      for(iy=0; iy<=ix; iy++) {
      for(iz=0; iz<=iy; iz++) {
        nqhat++;

	iq[0] = 0; iq[1] = ix; iq[2] = iy; iq[3] = iz;
/*
        fprintf(stdout, "=====================================\n");
        fprintf(stdout, "it=%3d, ix=%3d, iy=%3d, iz=%3d, nqhat=%3d, id=%3d\n", 0, ix, iy, iz, nqhat, it*Nclasses+nqhat);
*/
	for(n=0; n<6; n++) { 
	  iq_perm[0] = iq[perm_tab_3[n][0]+1];
	  iq_perm[1] = iq[perm_tab_3[n][1]+1];
	  iq_perm[2] = iq[perm_tab_3[n][2]+1];
/*          fprintf(stdout, "%3d, %3d, %3d, %3d\n", iq[0], iq_perm[0], iq_perm[1], iq_perm[2]); */
          for(isigma1=0; isigma1<=1; isigma1++) {
            i1 = ( (1-isigma1)*iq_perm[0] + isigma1*(L-iq_perm[0]) ) % L;
          for(isigma2=0; isigma2<=1; isigma2++) {
            i2 = ( (1-isigma2)*iq_perm[1] + isigma2*(L-iq_perm[1]) ) % L;
          for(isigma3=0; isigma3<=1; isigma3++) {
            i3 = ( (1-isigma3)*iq_perm[2] + isigma3*(L-iq_perm[2]) ) % L;
	    index_s = g_ipt[0][i1][i2][i3];
/*
            fprintf(stdout, "-------------------------------------\n");
            fprintf(stdout, "%3d, %3d, %3d, %3d; %5d, %5d\n", i0, i1, i2, i3, index_s, (*qid)[index_s]);
*/
	    if((*qid)[index_s] == -1) {
              (*qid)[index_s] = nqhat;
	      (*qid_count)[nqhat] +=1;
	    }
          }
          }
          }
        }
      }  
      }   
    } 

  *nc = Nclasses;

  /************************
   * test: print the lists 
   ************************/
/*
  for(ix=0; ix<L; ix++) {
  for(iy=0; iy<L; iy++) {
  for(iz=0; iz<L; iz++) {
    index_s = g_ipt[0][ix][iy][iz];

    q[0] = 0;
    q[1] = 2. * sin( M_PI * (double)ix / (double)L );
    q[2] = 2. * sin( M_PI * (double)iy / (double)L );
    q[3] = 2. * sin( M_PI * (double)iz / (double)L );

    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d, qid=%8d\n", 0, ix, iy, iz, (*qid)[index_s]);
  }
  }
  }
*/

  for(n=0; n<Nclasses; n++) {
    fprintf(stdout, "t=%3d, n=%5d, qid_count=%8d\n", 0, n, (*qid_count)[n]);
  }

  return(0);

}

/********************************************************************************
 * H3 orbits for \vec{r}; should be the same as for \vec{q}
 ********************************************************************************/

int make_Oh_orbits_r(int **rid, int **rid_count, double ***rid_val, int *nc, double Rmin, double Rmax) {

  int it, ix, iy, iz, iix, n;
  int x0, x1, x2, x3;
  int i0, i1, i2, i3;
  int isigma0, isigma1, isigma2, isigma3;
  int nqhat = -1;
  int L = LX;
  int Thp1 = (T/2)+1;
  int Lhp1 = (LX/2)+1;
/*
  int Thp1 = T;
  int Lhp1 = LX;
*/
  int iq[4];
  int iq_perm[3], index_s;
  double q[4], rsqrt;

  int Nclasses = ( Lhp1 * (Lhp1+1) * (Lhp1+2) ) / 6; 
  fprintf(stdout, "# Nclasses = %d x %d\n", Thp1, Nclasses);

  /* initialize the permutation tables */
  fprintf(stdout, "# initializing perm tabs\n");
  init_perm_tabs();

  if(Rmax==-1.) Rmax = sqrt( (double)(LX*LX + LY*LY + LZ*LZ) ) + 0.5;


  *rid_count = (int *)calloc(Thp1 * Nclasses, sizeof(int));
  if(*rid_count==(int*)NULL) return(301);

  *rid_val = (double **)calloc(4, sizeof(double*));
  (*rid_val)[0] = (double*)calloc(4 * Thp1 * Nclasses, sizeof(double));
  if((*rid_val)[0]==(double*)NULL) return(302);
  (*rid_val)[1] = (*rid_val)[0] + Thp1 * Nclasses;
  (*rid_val)[2] = (*rid_val)[1] + Thp1 * Nclasses;
  (*rid_val)[3] = (*rid_val)[2] + Thp1 * Nclasses;

  fprintf(stdout, "# setting new fields to zero\n");
  for(it=0; it<Thp1*Nclasses; it++) {
    (*rid_count)[it]  =  0;
    (*rid_val)[0][it] = -1.;
    (*rid_val)[1][it] = -1.;
    (*rid_val)[2][it] = -1.;
    (*rid_val)[3][it] = -1.;
  }

  if((*rid) == (int*)NULL) {
    fprintf(stdout, "# allocating memory for rid & setting to zero\n");
    *rid = (int *)calloc(VOLUME, sizeof(int));
    if(*rid==(int*)NULL) return(303);
    for(ix=0; ix<VOLUME; ix++) (*rid)[ix] = -1;
  }


  for(it=0; it<Thp1; it++) {
    nqhat = -1;
    iq[0] = it;
    for(ix=0; ix<Lhp1; ix++) {
    for(iy=0; iy<=ix; iy++) {
    for(iz=0; iz<=iy; iz++) {
      nqhat++;
      iq[1] = ix;
      iq[2] = iy;
      iq[3] = iz;

      rsqrt = sqrt((double)(ix*ix + iy*iy + iz*iz));
      if(Rmin-rsqrt >= _Q2EPS || rsqrt-Rmax >= _Q2EPS) continue;

      (*rid_val)[0][it*Nclasses+nqhat] = rsqrt;

      for(n=0; n<6; n++) { 
        iq_perm[0] = iq[perm_tab_3[n][0]+1];
        iq_perm[1] = iq[perm_tab_3[n][1]+1];
        iq_perm[2] = iq[perm_tab_3[n][2]+1];

        for(isigma0=0; isigma0<=1; isigma0++) {
          i0 = ( (1-isigma0)*iq[0] + isigma0*(T-iq[0]) ) % T;
        for(isigma1=0; isigma1<=1; isigma1++) {
          i1 = ( (1-isigma1)*iq_perm[0] + isigma1*(L-iq_perm[0]) ) % L;
        for(isigma2=0; isigma2<=1; isigma2++) {
          i2 = ( (1-isigma2)*iq_perm[1] + isigma2*(L-iq_perm[1]) ) % L;
        for(isigma3=0; isigma3<=1; isigma3++) {
          i3 = ( (1-isigma3)*iq_perm[2] + isigma3*(L-iq_perm[2]) ) % L;
	  index_s = g_ipt[i0][i1][i2][i3];
	  if((*rid)[index_s] == -1) {
            (*rid)[index_s] = it*Nclasses + nqhat;
	    (*rid_count)[it*Nclasses+nqhat] +=1;
	  }
        }}}}
/*
	index_s = g_ipt[iq[0]][iq_perm[0]][iq_perm[1]][iq_perm[2]];
	if((*rid)[index_s] == -1) {
          (*rid)[index_s] = it*Nclasses + nqhat;
	  (*rid_count)[it*Nclasses+nqhat] +=1;
	}
*/
      }
    }}}
  }  
  *nc = Thp1 * Nclasses;

  /************************
   * test: print the lists 
   ************************/
/*
  for(it=0; it<T; it++) {
  for(ix=0; ix<L; ix++) {
  for(iy=0; iy<L; iy++) {
  for(iz=0; iz<L; iz++) {
    index_s = g_ipt[it][ix][iy][iz];
    rsqrt = sqrt((double)(ix*ix + iy*iy + iz*iz));

    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d, rid=%8d, |r|=%16.7e\n", it, ix, iy, iz, 
      (*rid)[index_s], (*rid_val)[0][(*rid)[index_s]]);

  }
  }
  }
  }
*/

  fprintf(stdout, "\n\n");
  for(it=0; it<Thp1; it++) {
    for(n=0; n<Nclasses; n++) {
      fprintf(stdout, "t=%3d, n=%5d, rid_count=%8d, rid_val= %25.16e\n",
        it, n, (*rid_count)[it*Nclasses+n], (*rid_val)[0][it*Nclasses+n]);
    }
  }

  return(0);

}

/********************************************************************************/
#ifdef _UNDEF
int make_H4_r_orbits(int **qid, int **qid_count, double ***qid_val, int *nc, int ***rep) {

  int it, ix, iy, iz, iix, iit, n;
  int i0, i1, i2, i3;
  int isigma0, isigma1, isigma2, isigma3;
  int nqhat = -1;
  int Thp1  = T/2+1;
  int Thm1  = T/2-1;
  int L = LX;
  int Lhp1  = L/2+1;
  int Lhm1  = L/2-1;
  int iq[4];
  int iq_perm[4], index_s;
  double q[4], qsq, q4;

/*  int Nclasses = (L+2)*(L+4)*(L+6)*(L+8)/24/16 + T/4 * (L+2)*(L+4)*(L+6)/6/8; */

  /************************************************
   * determine the number of classes
   ************************************************/
  for(it=0; it<Thp1; it++) {
    for(ix=0; ix<Lhp1; ix++) {
      for(iy=0; iy<=ix; iy++) {
      for(iz=0; iz<=ix; iz++) {
        

      }}
    }
  }




  fprintf(stdout, "# Nclasses = %d\n", Nclasses);
  *nc = Nclasses;

  if( (*qid_count = (int *)calloc(Nclasses, sizeof(int)))==NULL ) {
    fprintf(stderr, "# Error, could not allocate mem for qid_count\n"); 
    return(1);
  }

  if( (*qid_val=(double **)calloc(4, sizeof(double*))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for qid_val\n"); 
    return(2);
  }
  if( (*(*qid_val) = (double * )calloc(4 * Nclasses, sizeof(double))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for qid_val[0]\n"); 
    return(3);
  }
  (*qid_val)[1] = (*qid_val)[0] + Nclasses;
  (*qid_val)[2] = (*qid_val)[1] + Nclasses;
  (*qid_val)[3] = (*qid_val)[2] + Nclasses;

  if( (*rep=(int **)calloc(4, sizeof(double*))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for qid_val\n");
    return(5);
  }
  if( (*(*rep) = (int * )calloc(4 * Nclasses, sizeof(double))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for qid_val[0]\n");
    return(6);
  }
  (*rep)[1] = (*rep)[0] + Nclasses;
  (*rep)[2] = (*rep)[1] + Nclasses;
  (*rep)[3] = (*rep)[2] + Nclasses;

  for(it=0; it<Nclasses; it++) {
    (*qid_count)[it]  =  0;
    (*qid_val)[0][it] = -1.;
    (*qid_val)[1][it] = -1.;
    (*qid_val)[2][it] = -1.;
    (*qid_val)[3][it] = -1.;
  }
  if((*qid) == (int*)NULL) {
    if( (*qid = (int *)calloc(VOLUME, sizeof(int))) == NULL ) {
      fprintf(stderr, "# Error, could not allocate mem for qid\n"); 
      return(4);
    }
    for(ix=0; ix<VOLUME; ix++) (*qid)[ix] = -1;
  }

  /* initialize the permutation tables */
  init_perm_tabs();
  nqhat = -1;
  for(it=0; it<Lhp1; it++) {
    for(ix=0; ix<=it; ix++) {
    for(iy=0; iy<=ix; iy++) {
    for(iz=0; iz<=iy; iz++) {
      nqhat++;
      q[0] = (double)it;
      q[1] = (double)ix;
      q[2] = (double)iy;
      q[3] = (double)iz;

      (*qid_val)[0][nqhat] = pow(q[0], 2.) + pow(q[1], 2.) + pow(q[2], 2.) + pow(q[3], 2.);
      (*qid_val)[1][nqhat] = pow(q[0], 4.) + pow(q[1], 4.) + pow(q[2], 4.) + pow(q[3], 4.);
      (*qid_val)[2][nqhat] = pow(q[0], 6.) + pow(q[1], 6.) + pow(q[2], 6.) + pow(q[3], 6.);
      (*qid_val)[3][nqhat] = pow(q[0], 8.) + pow(q[1], 8.) + pow(q[2], 8.) + pow(q[3], 8.);

      iq[0] = it; 
      iq[1] = ix; 
      iq[2] = iy; 
      iq[3] = iz;
/*
      fprintf(stdout, "=====================================\n");
      fprintf(stdout, "it=%3d, ix=%3d, iy=%3d, iz=%3d, nqhat=%3d\n", it, ix, iy, iz, nqhat);
*/
      /* go through all permutations */
      for(n=0; n<24; n++) { 
        iq_perm[0] = iq[perm_tab_4[n][0]];
        iq_perm[1] = iq[perm_tab_4[n][1]];
        iq_perm[2] = iq[perm_tab_4[n][2]];
        iq_perm[3] = iq[perm_tab_4[n][3]];
/*
        fprintf(stdout, "-------------------------------------\n");
        fprintf(stdout, "%3d, %3d, %3d, %3d\n", iq[0], iq_perm[0], iq_perm[1], iq_perm[2]);
*/
        for(isigma0=0; isigma0<=1; isigma0++) {
          i0 = ( (1-isigma0)*iq_perm[0] + isigma0*(T-iq_perm[0]) ) % T;
        for(isigma1=0; isigma1<=1; isigma1++) {
          i1 = ( (1-isigma1)*iq_perm[1] + isigma1*(L-iq_perm[1]) ) % L;
        for(isigma2=0; isigma2<=1; isigma2++) {
          i2 = ( (1-isigma2)*iq_perm[2] + isigma2*(L-iq_perm[2]) ) % L;
        for(isigma3=0; isigma3<=1; isigma3++) {
          i3 = ( (1-isigma3)*iq_perm[3] + isigma3*(L-iq_perm[3]) ) % L;
          index_s = g_ipt[i0][i1][i2][i3];
/*          fprintf(stdout, "%3d, %3d, %3d, %3d; %5d, %5d\n", i0, i1, i2, i3, index_s, (*qid)[index_s]); */
          if((*qid)[index_s] == -1) {
            (*qid)[index_s] = nqhat;
	    (*qid_count)[nqhat] +=1;
	  }
        }}}}
      }    /* end of n = 0, ..., 24-1 */
    }}}    /* of iz, iy, ix */
  }        /* of if it % 2 == 1 */
  for(it=Lhp1; it<=L; it++) {
    for(ix=0; ix<Lhp1; ix++) {
      for(iy=0; iy<=ix; iy++) {
      for(iz=0; iz<=iy; iz++) {
        nqhat++;
        q[0] = (double)it;
        q[1] = (double)ix;
        q[2] = (double)iy;
        q[3] = (double)iz;

        (*qid_val)[0][nqhat] = pow(q[0], 2.) + pow(q[1], 2.) + pow(q[2], 2.) + pow(q[3], 2.);
        (*qid_val)[1][nqhat] = pow(q[0], 4.) + pow(q[1], 4.) + pow(q[2], 4.) + pow(q[3], 4.);
        (*qid_val)[2][nqhat] = pow(q[0], 6.) + pow(q[1], 6.) + pow(q[2], 6.) + pow(q[3], 6.);
        (*qid_val)[3][nqhat] = pow(q[0], 8.) + pow(q[1], 8.) + pow(q[2], 8.) + pow(q[3], 8.);

        iq[0] = it; 
        iq[1] = ix; 
        iq[2] = iy; 
        iq[3] = iz;
/*
        fprintf(stdout, "=====================================\n");
        fprintf(stdout, "it=%3d, ix=%3d, iy=%3d, iz=%3d, nqhat=%3d\n", it, ix, iy, iz, nqhat);
*/
  	iq_perm[0] = iq[0];
        for(n=0; n<6; n++) {
	  iq_perm[1] = iq[perm_tab_3[n][0]+1];
	  iq_perm[2] = iq[perm_tab_3[n][1]+1];
	  iq_perm[3] = iq[perm_tab_3[n][2]+1];
/*
          fprintf(stdout, "-------------------------------------\n");
          fprintf(stdout, "%3d, %3d, %3d, %3d\n", iq_perm[0], iq_perm[1], iq_perm[2], iq_perm[3]);
*/
          for(isigma0=0; isigma0<=1; isigma0++) {
            i0 = ( (1-isigma0)*iq_perm[0] + isigma0*(T-iq_perm[0]) ) % T;
          for(isigma1=0; isigma1<=1; isigma1++) {
            i1 = ( (1-isigma1)*iq_perm[1] + isigma1*(L-iq_perm[1]) ) % L;
          for(isigma2=0; isigma2<=1; isigma2++) {
            i2 = ( (1-isigma2)*iq_perm[2] + isigma2*(L-iq_perm[2]) ) % L;
          for(isigma3=0; isigma3<=1; isigma3++) {
            i3 = ( (1-isigma3)*iq_perm[3] + isigma3*(L-iq_perm[3]) ) % L;
  	    index_s = g_ipt[i0][i1][i2][i3];
/*            fprintf(stdout, "%3d, %3d, %3d, %3d; %5d, %5d\n", i0, i1, i2, i3, index_s, (*qid)[index_s]); */
  	    if((*qid)[index_s] == -1) {
              (*qid)[index_s] = nqhat;
	      (*qid_count)[nqhat] +=1;
	    }
          }}}}
        }    /* of n = 0, ..., 23  */
      }}     /* of iz, iy */
    }
  }          /* of it */

  /************************
   * test: print the lists 
   ************************/
/*
  for(it=0; it<T; it++) {
  for(ix=0; ix<L; ix++) {
  for(iy=0; iy<L; iy++) {
  for(iz=0; iz<L; iz++) {
    index_s = g_ipt[it][ix][iy][iz];
    q[0] = (double)it;
    q[1] = (double)ix;
    q[2] = (double)iy;
    q[3] = (double)iz;
    qsq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    q4 = q[0]*q[0]*q[0]*q[0] + q[1]*q[1]*q[1]*q[1] + 
         q[2]*q[2]*q[2]*q[2] + q[3]*q[3]*q[3]*q[3]; 

    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d, qid=%4d\n", it, ix, iy, iz, (*qid)[index_s]);
  }}}}

  fprintf(stdout, "# n\tqid_count\tqid_val\n");
  for(n=0; n<Nclasses; n++) {
    fprintf(stdout, "%5d\t%8d\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      n, (*qid_count)[n], (*qid_val)[0][n], (*qid_val)[1][n],(*qid_val)[2][n],(*qid_val)[3][n]);
  }
*/

  return(0);
}
#endif

/***********************************************************************
 * deallocate memory for the fields used in H4 method analysis
 ***********************************************************************/
void finalize_H4_r_orbits(int **qid, int **qid_count, double ***qid_val, int***rep) {

  if( *qid != NULL ) {
    free(*qid);
    *qid = NULL;
  }
  if(*qid_count != NULL) {
    free(*qid_count); *qid_count = NULL;
  }
  if(*qid_val != NULL) {
    if(**qid_val != NULL) {
      free(**qid_val);
      **qid_val = NULL;
    }
    free(*qid_val);
    *qid_val = NULL;
  }
  if(*rep != NULL) {
    if(**rep != NULL) {
      free(**rep);
      **rep = NULL;
    }
    free(*rep);
    rep = NULL;
  }
}
