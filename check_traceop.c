#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include <getopt.h>
/* #include <lapack.h> */

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

void zgetrf_ (int* m, int* n, complex a[], int* lda, int ipiv[], int* info);
void zgetri_ (int *n, complex a[], int* lda, int ipiv[], complex work[], int* lwork, int* info);


int main (void) {

  int i, j, mu;
  unsigned int seed=12345;
  double U[18], C[12][24], B[12][24];
  double spinor1[24], U1[18], U2[18], U3[18];
  complex a[9], work[3];
  complex w;
  int n=3, m=3, lda=3, ipiv[3], lwork=3, info;

  srand(seed);

  /* initialise gamma */
  init_gamma();

  /* 1st test: gamma = id, U = id */
/*  for(i=0; i<18; i++) U[i] = 0.;
  for(i=0; i<3; i++) U[8*i] = 1.; */
  for(i=0; i<18; i++) U[i] = (double)rand() / ((double)RAND_MAX + 1.);

  fprintf(stdout, "\n#This is U:\n");
  for(i=0; i<3; i++) {
      fprintf(stdout, "%15.5e +I%15.5e%15.5e +I%15.5e%15.5e +I%15.5e\n", 
        U[2*(3*i+0)  ], U[2*(3*i+0)+1], U[2*(3*i+1)  ],
	U[2*(3*i+1)+1], U[2*(3*i+2)  ], U[2*(3*i+2)+1]);
  }

  for(i=0; i<3; i++) {
    for(j=0; j<3; j++) {
      a[3*i+j].re = U[2*(3*j+i)  ];
      a[3*i+j].im = U[2*(3*j+i)+1];
    }
  }
  zgetrf_(&m, &n, a, &lda, ipiv, &info); 
  fprintf(stdout, "zgetrf: info = %d\n", info);
  zgetri_(&n, a, &lda, ipiv, work, &lwork, &info);
  fprintf(stdout, "zgetri: info = %d\n", info);
  for(i=0; i<3; i++) {
    for(j=0; j<3; j++) {
      U1[2*(3*j+i)  ] = a[3*i+j].re;
      U1[2*(3*j+i)+1] = a[3*i+j].im;
    }
  }

  for(i=0; i<3; i++) {
  for(j=0; j<3; j++) {
    U2[2*(3*i+j)  ] =  U1[2*(3*j+i)  ];
    U2[2*(3*i+j)+1] = -U1[2*(3*j+i)+1];
  }
  }
/*
  _cm_eq_cm_ti_cm_dag(U3, U2, U);
  fprintf(stdout, "\n#This is U2 * U^+:\n");
  for(i=0; i<3; i++) {
      fprintf(stdout, "%15.5e +I%15.5e%15.5e +I%15.5e%15.5e +I%15.5e\n", 
        U3[2*(3*i+0)], U3[2*(3*i+0)+1], U3[2*(3*i+1)], U3[2*(3*i+1)+1], U3[2*(3*i+2)], U3[2*(3*i+2)+1]);
  }
*/

  _cm_eq_cm_ti_cm(U3, U1, U);
  fprintf(stdout, "\n#This is U1 * U:\n");
  for(i=0; i<3; i++) {
      fprintf(stdout, "%15.5e +I%15.5e%15.5e +I%15.5e%15.5e +I%15.5e\n", 
        U3[2*(3*i+0)], U3[2*(3*i+0)+1], U3[2*(3*i+1)], U3[2*(3*i+1)+1], U3[2*(3*i+2)], U3[2*(3*i+2)+1]);
  }

  for(i=0; i<12; i++) { 
    for(j=0; j<24; j++) C[i][j] = (double)rand() / ((double)RAND_MAX + 1.);
  }

  for(mu=0; mu<16; mu++) {

    /* calculate w1 with the spinor method */
    for(i=0; i<12; i++) {
      _fv_eq_gamma_ti_fv(spinor1, mu, C[i]);
/*      _fv_eq_cm_ti_fv(B[i], U2, spinor1); */
      _fv_eq_cm_ti_fv(B[i], U1, spinor1);
    }

    /* calculate w with the trace method */
    w.re = 0.; w.im = 0.;
/*    _co_eq_tr_gammaUdag_sm(&w, mu, U, B); */
    _co_eq_tr_gammaU_sm(&w, mu, U, B);

    fprintf(stdout, "%3d w= %25.16e +I %25.16e\n", mu, w.re, w.im);
  }

  w.re = 0.; w.im = 0.;
  for(i=0; i<12; i++) {
    w.re += C[i][2*i  ];
    w.im += C[i][2*i+1];
  }
  fprintf(stdout, "\n%3d w= %25.16e +I %25.16e\n", 4, w.re, w.im);


/*
  mu = 1; nu = 1;

  for(i=0; i<12; i++) { 
  for(j=0; j<12; j++) {
    B[i][2*j  ] = gamma_sign[mu][2*i] * C[gamma_permutation[mu][2*i]/2][2*j+gamma_permutation[mu][2*i]%2];
    B[i][2*j+1] = gamma_sign[mu][2*i+1] * C[gamma_permutation[mu][2*i+1]/2][2*j+gamma_permutation[mu][2*i+1]%2];
  }
  }
  for(i=0; i<12; i++) { 
  for(j=0; j<12; j++) {
    D[i][2*j  ] = gamma_sign[nu][2*i] * B[gamma_permutation[nu][2*i]/2][2*j+gamma_permutation[nu][2*i]%2];
    D[i][2*j+1] = gamma_sign[nu][2*i+1] * B[gamma_permutation[nu][2*i+1]/2][2*j+gamma_permutation[nu][2*i+1]%2];
  }
  }
  
  for(i=0; i<12; i++) { 
  for(j=0; j<12; j++) {
    B[i][2*j  ] = gamma_sign[nu][2*i] * C[gamma_permutation[nu][2*i]/2][2*j+gamma_permutation[nu][2*i]%2];
    B[i][2*j+1] = gamma_sign[nu][2*i+1] * C[gamma_permutation[nu][2*i+1]/2][2*j+gamma_permutation[nu][2*i+1]%2];
  }
  }
  for(i=0; i<12; i++) { 
  for(j=0; j<12; j++) {
    A[i][2*j  ] = gamma_sign[mu][2*i] * B[gamma_permutation[mu][2*i]/2][2*j+gamma_permutation[mu][2*i]%2];
    A[i][2*j+1] = gamma_sign[mu][2*i+1] * B[gamma_permutation[mu][2*i+1]/2][2*j+gamma_permutation[mu][2*i+1]%2];
  }
  }
  
  
  for(i=0; i<12; i++) {
  for(j=0; j<12; j++) {
    B[i][2*j  ] = (D[i][2*j  ] + A[i][2*j  ]) / 2.;
    B[i][2*j+1] = (D[i][2*j+1] + A[i][2*j+1]) / 2.;
  }
  }

  fprintf(stdout, "#This is C:\n");
  for(i=0; i<12; i++) {
  for(j=0; j<12; j++) {
    fprintf(stdout, "%3d%3d%15.5e +I%15.5e%15.5e +I%15.5e%15.5e +I%15.5e%15.5e +I%15.5e\n", 
      i, j, C[i][2*j], C[i][2*j+1], B[i][2*j], B[i][2*j+1], D[i][2*j], D[i][2*j+1], A[i][2*j], A[i][2*j+1]);
  }
  }


  j = 0;
  for(i=0; i<12; i++) {
    spinor1[2*i  ] = C[i][2*j  ];
    spinor1[2*i+1] = C[i][2*j+1];
  }
  for(i=0; i<12; i++) {
  for(j=0; j<12; j++) {
    B[i][2*j  ] = gamma_sign[mu][2*i] * C[gamma_permutation[mu][2*i]/2][2*j+gamma_permutation[mu][2*i]%2];
    B[i][2*j+1] = gamma_sign[mu][2*i+1] * C[gamma_permutation[mu][2*i+1]/2][2*j+gamma_permutation[mu][2*i+1]%2];
  }
  }

  _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);

  j=0;
  fprintf(stdout, "#This is B/spinor2:\n");
  for(i=0; i<12; i++) {
    fprintf(stdout, "%3d%15.5e +I%15.5e%15.5e +I%15.5e\n", 
      i, B[i][2*j], B[i][2*j+1], spinor2[2*i], spinor2[2*i+1]);
  }
*/ 

  return(0);
}

