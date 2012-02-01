/****************************************************
 * pidisc_model.c
 *
 * Fri Nov 20 09:50:41 CET 2009
 *
 * PURPOSE:
 * - model functions for the disconnected contribution to hadronic
 *   VP
 * - mrho - rho meson mass
 * - d multiplicative coeff., complex
 * (0) \Pi_\mu\nu(p) = d/(\hat{p}^2 + m^2)^2 (\delta_\mu\nu \hat{p}^2 - \hat{p}_\mu \hat{p}_\nu)
 * (2) \Pi_\mu\nu(p) = d/(\hat{p}^2 + m^2)^2 (\delta_\mu\nu - \hat{p}_\mu \hat{p}_\nu / \hat{p}^2)
 * (1,3) like (0,2) but with \hat{p} --> p
 * TODO:
 * DONE:
 * CHANGES:
 * - took out the p-dependent phase factor exp(-i (p_mu/2 - p_nu/2))
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
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "pidisc_model.h"

int pidisc_model(const double mrho, const double dcoeffre, const double dcoeffim, double *pimn, _FFTW_PLAN plan_m, const int imu) {
  
  int mu, nu;
  int x0, x1, x2, x3, ix;
  double q[4], p[4], fnorm, q2;
  double ratime, retime;
  double mrho2 = mrho*mrho;
  complex w;

  fftw_complex *in = (fftw_complex*)NULL;

  for(ix=0; ix<2*VOLUME; ix++) pimn[ix] = 0.;

  /*****************************************************
   * set the \hat{q}-dep. function for given mu and nu
   *****************************************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  mu = imu / 4;
  nu = imu % 4;
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * sin( M_PI * (double)(x0+Tstart) / (double)T_global );
    p[0] = (double)(x0+Tstart) / (double)T_global;
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin( M_PI * (double)x1 / (double)LX );
    p[1] = (double)x1 / (double)LX;
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * sin( M_PI * (double)x2 / (double)LY );
    p[2] = (double)x2 / (double)LY;
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * sin( M_PI * (double)x3 / (double)LZ );
    p[3] = (double)x3 / (double)LZ;
    q2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    fnorm = ( (double)(mu==nu)*q2 - q[mu]*q[nu] ) / ( (q2 + mrho2) * (q2 + mrho2) ); 
    ix = g_ipt[x0][x1][x2][x3];
    pimn[2*ix  ] = dcoeffre * fnorm;
    pimn[2*ix+1] = dcoeffim * fnorm;
  }}}}
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time to set Pi_mu_nu(p): %e seconds;\n", retime-ratime);
  
/* 
  fprintf(stdout, "# pimodel0 in momentum space:\n");
  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "pimodel0-P[%2d,%5d] = %25.16e +i %25.16e\n", imu, ix, pimn[2*ix], pimn[2*ix+1]);
  }
*/

  /*****************************************************
   * Fourier transform
   *****************************************************/
  if( ( in = (fftw_complex*)fftw_malloc(2*VOLUME*sizeof(double)) ) == (fftw_complex*)NULL ) {
    fprintf(stderr, "Error, could not allocate memory for in; exit\n");
    exit(150);
  }

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  memcpy((void*)in, (void*)pimn, 2*VOLUME*sizeof(double));
#ifdef MPI
  fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
  fftwnd_one(plan_m, in, NULL);
#endif
  memcpy((void*)pimn, (void*)in, 2*VOLUME*sizeof(double));
  fnorm = 1. / (double)T_global / (double)(LX*LY*LZ);
  for(ix=0; ix<2*VOLUME; ix++) pimn[ix] *= fnorm;
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time for FFTW: %e seconds;\n", retime-ratime);

  fftw_free(in);

/*
  fprintf(stdout, "# pimodel in position space:\n");
  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "pimodel-X[%2d,%5d] = %25.16e +i %25.16e\n", imu, ix, pimn[2*ix], pimn[2*ix+1]);
  }
*/
  return(0);
}

/***************************************************************************
 *
 ***************************************************************************/
int pidisc_model1(const double mrho, const double dcoeffre, const double dcoeffim, double *pimn, _FFTW_PLAN plan_m, const int imu) {
  
  int mu, nu;
  int x0, x1, x2, x3, ix;
  double q[4], fnorm, q2;
  double ratime, retime;
  double mrho2 = mrho*mrho;
  complex w;

  fftw_complex *in = (fftw_complex*)NULL;

  for(ix=0; ix<2*VOLUME; ix++) pimn[ix] = 0.;

  /*****************************************************
   * set the \hat{q}-dep. function for given mu and nu
   *****************************************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  mu = imu / 4;
  nu = imu % 4;
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * M_PI * (double)(x0+Tstart) / (double)T_global;
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * M_PI * (double)x1 / (double)LX;
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * M_PI * (double)x2 / (double)LY;
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * M_PI * (double)x3 / (double)LZ;
    q2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    fnorm = ( (double)(mu==nu)*q2 - q[mu]*q[nu] ) / ( (q2 + mrho2) * (q2 + mrho2) ); 
    ix = g_ipt[x0][x1][x2][x3];
    pimn[2*ix  ] = dcoeffre * fnorm;
    pimn[2*ix+1] = dcoeffim * fnorm;
  }}}}
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time to set Pi_mu_nu(p): %e seconds;\n", retime-ratime);
  
/* 
  fprintf(stdout, "# pimodel1 in momentum space:\n");
  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "pimodel1-P[%2d,%5d] = %25.16e +i %25.16e\n", imu, ix, pimn[2*ix], pimn[2*ix+1]);
  }
*/

  /*****************************************************
   * Fourier transform
   *****************************************************/
  if( ( in = (fftw_complex*)fftw_malloc(2*VOLUME*sizeof(double)) ) == (fftw_complex*)NULL ) {
    fprintf(stderr, "Error, could not allocate memory for in; exit\n");
    exit(150);
  }

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  memcpy((void*)in, (void*)pimn, 2*VOLUME*sizeof(double));
#ifdef MPI
  fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
  fftwnd_one(plan_m, in, NULL);
#endif
  memcpy((void*)pimn, (void*)in, 2*VOLUME*sizeof(double));
  fnorm = 1. / (double)T_global / (double)(LX*LY*LZ);
  for(ix=0; ix<2*VOLUME; ix++) pimn[ix] *= fnorm;
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time for FFTW: %e seconds;\n", retime-ratime);

  fftw_free(in);

/*
  fprintf(stdout, "# pimodel in position space:\n");
  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "pimodel-X[%2d,%5d] = %25.16e +i %25.16e\n", imu, ix, pimn[2*ix], pimn[2*ix+1]);
  }
*/
  return(0);

}

/*********************************************************************
 *
 *********************************************************************/
int pidisc_model2(const double mrho, const double dcoeffre, const double dcoeffim, double *pimn, _FFTW_PLAN plan_m, const int imu) {
  
  int mu, nu;
  int x0, x1, x2, x3, ix;
  double q[4], p[4], fnorm, q2;
  double ratime, retime;
  double mrho2 = mrho*mrho;
  complex w;

  fftw_complex *in = (fftw_complex*)NULL;

  for(ix=0; ix<2*VOLUME; ix++) pimn[ix] = 0.;

  /*****************************************************
   * set the \hat{q}-dep. function for given mu and nu
   *****************************************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  mu = imu / 4;
  nu = imu % 4;
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * sin( M_PI * (double)(x0+Tstart) / (double)T_global );
    p[0] = (double)(x0+Tstart) / (double)T_global;
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin( M_PI * (double)x1 / (double)LX );
    p[1] = (double)x1 / (double)LX;
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * sin( M_PI * (double)x2 / (double)LY );
    p[2] = (double)x2 / (double)LY;
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * sin( M_PI * (double)x3 / (double)LZ );
    p[3] = (double)x3 / (double)LZ;
    q2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if(q2<_Q2EPS) {
      fnorm = 0.;
    } else { 
      fnorm = ( (double)(mu==nu) - q[mu]*q[nu]/q2 ) / ( (q2 + mrho2) * (q2 + mrho2) );
    }
    ix = g_ipt[x0][x1][x2][x3];
    pimn[2*ix  ] = dcoeffre * fnorm;
    pimn[2*ix+1] = dcoeffim * fnorm;
  }}}}
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time to set Pi_mu_nu(p): %e seconds;\n", retime-ratime);
  
/* 
  fprintf(stdout, "# pimodel0 in momentum space:\n");
  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "pimodel0-P[%2d,%5d] = %25.16e +i %25.16e\n", imu, ix, pimn[2*ix], pimn[2*ix+1]);
  }
*/

  /*****************************************************
   * Fourier transform
   *****************************************************/
  if( ( in = (fftw_complex*)fftw_malloc(2*VOLUME*sizeof(double)) ) == (fftw_complex*)NULL ) {
    fprintf(stderr, "Error, could not allocate memory for in; exit\n");
    exit(150);
  }

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  memcpy((void*)in, (void*)pimn, 2*VOLUME*sizeof(double));
#ifdef MPI
  fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
  fftwnd_one(plan_m, in, NULL);
#endif
  memcpy((void*)pimn, (void*)in, 2*VOLUME*sizeof(double));
  fnorm = 1. / (double)T_global / (double)(LX*LY*LZ);
  for(ix=0; ix<2*VOLUME; ix++) pimn[ix] *= fnorm;
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time for FFTW: %e seconds;\n", retime-ratime);

  fftw_free(in);

/*
  fprintf(stdout, "# pimodel in position space:\n");
  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "pimodel-X[%2d,%5d] = %25.16e +i %25.16e\n", imu, ix, pimn[2*ix], pimn[2*ix+1]);
  }
*/
  return(0);
}

/***************************************************************************
 *
 ***************************************************************************/
int pidisc_model3(const double mrho, const double dcoeffre, const double dcoeffim, double *pimn, _FFTW_PLAN plan_m, const int imu) {
  
  int mu, nu;
  int x0, x1, x2, x3, ix;
  double q[4], fnorm, q2;
  double ratime, retime;
  double mrho2 = mrho*mrho;
  complex w;

  fftw_complex *in = (fftw_complex*)NULL;

  for(ix=0; ix<2*VOLUME; ix++) pimn[ix] = 0.;

  /*****************************************************
   * set the \hat{q}-dep. function for given mu and nu
   *****************************************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  mu = imu / 4;
  nu = imu % 4;
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * M_PI * (double)(x0+Tstart) / (double)T_global;
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * M_PI * (double)x1 / (double)LX;
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * M_PI * (double)x2 / (double)LY;
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * M_PI * (double)x3 / (double)LZ;
    q2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if(q2<_Q2EPS) {
      fnorm = 0.;
    } else {
      fnorm = ( (double)(mu==nu) - q[mu]*q[nu]/q2 ) / ( (q2 + mrho2) * (q2 + mrho2) ); 
    }
    ix = g_ipt[x0][x1][x2][x3];
    pimn[2*ix  ] = dcoeffre * fnorm;
    pimn[2*ix+1] = dcoeffim * fnorm;
  }}}}
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time to set Pi_mu_nu(p): %e seconds;\n", retime-ratime);
  
/* 
  fprintf(stdout, "# pimodel1 in momentum space:\n");
  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "pimodel1-P[%2d,%5d] = %25.16e +i %25.16e\n", imu, ix, pimn[2*ix], pimn[2*ix+1]);
  }
*/

  /*****************************************************
   * Fourier transform
   *****************************************************/
  if( ( in = (fftw_complex*)fftw_malloc(2*VOLUME*sizeof(double)) ) == (fftw_complex*)NULL ) {
    fprintf(stderr, "Error, could not allocate memory for in; exit\n");
    exit(150);
  }

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  memcpy((void*)in, (void*)pimn, 2*VOLUME*sizeof(double));
#ifdef MPI
  fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
  fftwnd_one(plan_m, in, NULL);
#endif
  memcpy((void*)pimn, (void*)in, 2*VOLUME*sizeof(double));
  fnorm = 1. / (double)T_global / (double)(LX*LY*LZ);
  for(ix=0; ix<2*VOLUME; ix++) pimn[ix] *= fnorm;
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time for FFTW: %e seconds;\n", retime-ratime);

  fftw_free(in);

/*
  fprintf(stdout, "# pimodel in position space:\n");
  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "pimodel-X[%2d,%5d] = %25.16e +i %25.16e\n", imu, ix, pimn[2*ix], pimn[2*ix+1]);
  }
*/
  return(0);
}
