/*********************************************************************************
 * contractions.cu
 *
 * Sat Jul  2 11:19:43 CEST 2011
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
#  include <fftw_mpi.h>
#else
#  include <fftw.h>
#endif
#include <getopt.h>

#define MAIN_PROGRAM
extern "C"
{
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "contractions_io.h"
#include "read_input_parser.h"
#include "contractions.h"
}

__constant__ int devT, devL;
__constant__ float devMu, devMq;
__constant__ float dev_cvc_coeff[2304];

/*************************************************************
 * the kernel for contract cvc
 *************************************************************/
__global__ void cvc_kernel (float*cvc_out, float*ct_out, unsigned int N) {

  int j0, j1, j2, j3, i0, i1, i2, i3;
  unsigned int L1, L2, L3, V4, imu, inu, icount, rest;
  // unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int isigma_mu, isigma_nu, ilambda, ikappa;
  float2 sp[6], sq[6];
  float p[4], phalf[4], sinp[4], cosp[4], sinphalf[4], sinq[4], sinqhalf[4];
  float q[4], qhalf[4], k[4];
  // float khalf[4];
  float2 phase[2][2];
  float ftmp;
  float2 cvc_tmp[16], f2tmp, f2tmp2, counter_term[4];
  float fTinv, fLinv;
  float aMp, aK2p, denomp;
  float aMq, aK2q, denomq;

  // get external momentum k from idx
  L1  = devL;
  L2 = L1 * L1;
  L3 = L2 * L1;
  V4 = L3 * devT;

  if(idx < N) {
    // initialize
    counter_term[0].x = 0.; counter_term[0].y = 0.;    
    counter_term[1].x = 0.; counter_term[1].y = 0.;    
    counter_term[2].x = 0.; counter_term[2].y = 0.;    
    counter_term[3].x = 0.; counter_term[3].y = 0.;    
    
    ftmp = -3.;  
    cvc_tmp[ 0].x = ftmp; cvc_tmp[ 0].y = ftmp;
    cvc_tmp[ 1].x = ftmp; cvc_tmp[ 1].y = ftmp;
    cvc_tmp[ 2].x = ftmp; cvc_tmp[ 2].y = ftmp;
    cvc_tmp[ 3].x = ftmp; cvc_tmp[ 3].y = ftmp;
    cvc_tmp[ 4].x = ftmp; cvc_tmp[ 4].y = ftmp;
    cvc_tmp[ 5].x = ftmp; cvc_tmp[ 5].y = ftmp;
    cvc_tmp[ 6].x = ftmp; cvc_tmp[ 6].y = ftmp;
    cvc_tmp[ 7].x = ftmp; cvc_tmp[ 7].y = ftmp;
    cvc_tmp[ 8].x = ftmp; cvc_tmp[ 8].y = ftmp;
    cvc_tmp[ 9].x = ftmp; cvc_tmp[ 9].y = ftmp;
    cvc_tmp[10].x = ftmp; cvc_tmp[10].y = ftmp;
    cvc_tmp[11].x = ftmp; cvc_tmp[11].y = ftmp;
    cvc_tmp[12].x = ftmp; cvc_tmp[12].y = ftmp;
    cvc_tmp[13].x = ftmp; cvc_tmp[13].y = ftmp;
    cvc_tmp[14].x = ftmp; cvc_tmp[14].y = ftmp;
    cvc_tmp[15].x = ftmp; cvc_tmp[15].y = ftmp;

    ftmp = -(float)idx;
    cvc_out[_GWI( 0,idx,V4)  ] = ftmp; cvc_out[_GWI( 0,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 1,idx,V4)  ] = ftmp; cvc_out[_GWI( 1,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 2,idx,V4)  ] = ftmp; cvc_out[_GWI( 2,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 3,idx,V4)  ] = ftmp; cvc_out[_GWI( 3,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 4,idx,V4)  ] = ftmp; cvc_out[_GWI( 4,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 5,idx,V4)  ] = ftmp; cvc_out[_GWI( 5,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 6,idx,V4)  ] = ftmp; cvc_out[_GWI( 6,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 7,idx,V4)  ] = ftmp; cvc_out[_GWI( 7,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 8,idx,V4)  ] = ftmp; cvc_out[_GWI( 8,idx,V4)+1] = ftmp;
    cvc_out[_GWI( 9,idx,V4)  ] = ftmp; cvc_out[_GWI( 9,idx,V4)+1] = ftmp;
    cvc_out[_GWI(10,idx,V4)  ] = ftmp; cvc_out[_GWI(10,idx,V4)+1] = ftmp;
    cvc_out[_GWI(11,idx,V4)  ] = ftmp; cvc_out[_GWI(11,idx,V4)+1] = ftmp;
    cvc_out[_GWI(12,idx,V4)  ] = ftmp; cvc_out[_GWI(12,idx,V4)+1] = ftmp;
    cvc_out[_GWI(13,idx,V4)  ] = ftmp; cvc_out[_GWI(13,idx,V4)+1] = ftmp;
    cvc_out[_GWI(14,idx,V4)  ] = ftmp; cvc_out[_GWI(14,idx,V4)+1] = ftmp;
    cvc_out[_GWI(15,idx,V4)  ] = ftmp; cvc_out[_GWI(15,idx,V4)+1] = ftmp;

    j0 = idx / L3;
    icount = idx - L3*j0;
    j1 = icount / L2;
    icount = icount - L2*j1;
    j2 = icount / L1;
    j3 = icount  - j2*L1;

    fTinv = 2. * _PI / (float)( devT );
    fLinv = 2. * _PI / (float)( devL );

    k[0] = (float)(j0) * fTinv;
    k[1] = (float)(j1) * fLinv;
    k[2] = (float)(j2) * fLinv;
    k[3] = (float)(j3) * fLinv;
/*
    khalf[0] = 0.5 * k[0];
    khalf[1] = 0.5 * k[1];
    khalf[2] = 0.5 * k[2];
    khalf[3] = 0.5 * k[3];
*/


    if(idx==102) {
      counter_term[0].x = fTinv; counter_term[0].y = fLinv;
      counter_term[1].x = k[0];  counter_term[1].y = k[1];
      counter_term[2].x = k[2];  counter_term[2].y = k[3];
      counter_term[3].x = (float)N; counter_term[3].y = (float)V4;
    } 


    // loop on internal momentum p (summation)
    i0=0; i1=0; i2=0; i3=0;
    for(icount=0; icount<V4; icount++) {

      p[0] = ((float)(i0)  + 0.5) * fTinv;
      phalf[0] = p[0] * 0.5;
      q[0] = ( ( (float)(i0) + (float)(j0) ) + 0.5 ) * fTinv;
      qhalf[0] = q[0] * 0.5;

      sinp[0]     = sin( p[0] );
      cosp[0]     = cos( p[0] );
      sinphalf[0] = sin( phalf[0] );
      sinq[0]     = sin( q[0] );
      sinqhalf[0] = sin( qhalf[0] );

      p[1] = (float)(i1)          * fLinv;
      phalf[1] = p[1] * 0.5;
      q[1] = ( (float)(i1) + (float)(j1) ) * fLinv;
      qhalf[1] = q[1] * 0.5;

      sinp[1]     = sin( p[1] );
      cosp[1]     = cos( p[1] );
      sinphalf[1] = sin( phalf[1] );
      sinq[1]     = sin( q[1] );
      sinqhalf[1] = sin( qhalf[1] );
  
      p[2] = (float)(i2)          * fLinv;
      phalf[2] = p[2] * 0.5;
      q[2] = ( (float)(i2) + (float)(j2) ) * fLinv;
      qhalf[2] = q[2] * 0.5;
  
      sinp[2]     = sin( p[2] );
      cosp[2]     = cos( p[2] );
      sinphalf[2] = sin( phalf[2] );
      sinq[2]     = sin( q[2] );
      sinqhalf[2] = sin( qhalf[2] );

      p[3] = (float)(i3) * fLinv;
      phalf[3] = p[3] * 0.5;
      q[3] = ( (float)(i3) + (float)(j3) ) * fLinv;
      qhalf[3] = q[3] * 0.5;

      sinp[3]     = sin( p[3] );
      cosp[3]     = cos( p[3] );
      sinphalf[3] = sin( phalf[3] );
      sinq[3]     = sin( q[3] );
      sinqhalf[3] = sin( qhalf[3] );
  
      aMp = devMq + 2. * (_SQR(sinphalf[0]) + _SQR(sinphalf[1]) + _SQR(sinphalf[2]) + _SQR(sinphalf[3]));

      aMq = devMq + 2. * (_SQR(sinqhalf[0]) + _SQR(sinqhalf[1]) + _SQR(sinqhalf[2]) + _SQR(sinqhalf[3]));

      aK2p = _SQR(sinp[0]) + _SQR(sinp[1]) + _SQR(sinp[2]) + _SQR(sinp[3]);

      aK2q = _SQR(sinq[0]) + _SQR(sinq[1]) + _SQR(sinq[2]) + _SQR(sinq[3]);

      denomp = 1. / ( aK2p + aMp*aMp + devMu*devMu );

      denomq = 1. / ( aK2q + aMq*aMq + devMu*devMu );

      sp[0].y = -sinp[0] * denomp;
      sp[1].y = -sinp[1] * denomp;
      sp[2].y = -sinp[2] * denomp;
      sp[3].y = -sinp[3] * denomp;
      sp[4].x =  aMp     * denomp;
      sp[5].y = -devMu   * denomp;

      sq[0].y = -sinq[0] * denomq;
      sq[1].y = -sinq[1] * denomq;
      sq[2].y = -sinq[2] * denomq;
      sq[3].y = -sinq[3] * denomq;
      sq[4].x =  aMq     * denomq;
      sq[5].y = -devMu   * denomq;

      _dev_set_phase(phase,p,k,0,0);
      cvc_tmp[0]. x += 1.;
      cvc_tmp[0].y += -2.;
/*
      _cvc_accum( cvc_tmp[ 0], 0, 0, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
*/
/*
      _dev_set_phase( phase, p, k, 0, 1 );
      _cvc_accum( cvc_tmp[ 1], 0, 1, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 0, 2 );
      _cvc_accum( cvc_tmp[ 2], 0, 2, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 0, 3 );
      _cvc_accum( cvc_tmp[ 3], 0, 3, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 1, 0 );
      _cvc_accum( cvc_tmp[ 4], 1, 0, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
*/
/*
      _dev_set_phase( phase, p, k, 1, 1 );
      _cvc_accum( cvc_tmp[ 5], 1, 1, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
*/
/*
      _dev_set_phase( phase, p, k, 1, 2 );
      _cvc_accum( cvc_tmp[ 6], 1, 2, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 1, 3 );
      _cvc_accum( cvc_tmp[ 7], 1, 3, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 2, 0 );
      _cvc_accum( cvc_tmp[ 8], 2, 0, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 2, 1 );
      _cvc_accum( cvc_tmp[ 9], 2, 1, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
*/
/*
      _dev_set_phase( phase, p, k, 2, 2 );
      _cvc_accum( cvc_tmp[10], 2, 2, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
*/
/*
      _dev_set_phase( phase, p, k, 2, 3 );
      _cvc_accum( cvc_tmp[11], 2, 3, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 3, 0 );
      _cvc_accum( cvc_tmp[12], 3, 0, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 3, 1 );
      _cvc_accum( cvc_tmp[13], 3, 1, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
      _dev_set_phase( phase, p, k, 3, 2 );
      _cvc_accum( cvc_tmp[14], 3, 2, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
*/
/*
      _dev_set_phase( phase, p, k, 3, 3 );
      _cvc_accum( cvc_tmp[15], 3, 3, dev_cvc_coeff, phase, sp, sq, f2tmp, f2tmp2 );
      //--------------------------------------------------------------------
*/
/*
      if( idx == 0 ){
        counter_term[0].x +=  sinp[0] * sp[0].y + sp[4].x * cosp[0];
        counter_term[0].y += -sinp[0] * sp[0].x + sp[4].y * cosp[0];
        counter_term[1].x +=  sinp[1] * sp[1].y + sp[4].x * cosp[1];
        counter_term[1].y += -sinp[1] * sp[1].x + sp[4].y * cosp[1];
        counter_term[2].x +=  sinp[2] * sp[2].y + sp[4].x * cosp[2];
        counter_term[2].y += -sinp[2] * sp[2].x + sp[4].y * cosp[2];
        counter_term[3].x +=  sinp[3] * sp[3].y + sp[4].x * cosp[3];
        counter_term[3].y += -sinp[3] * sp[3].x + sp[4].y * cosp[3];
      }
*/
      // increase the coordinates i0,...,i3
      i3 += 1;    rest = (i3==L1);   i3 -= rest*L1;
      i2 += rest; rest = (i2==L1);   i2 -= rest*L1;
      i1 += rest; rest = (i1==L1);   i1 -= rest*L1;
      i0 += rest; rest = (i0==devT); i0 -= rest*devT;

    }  // loop on icount

    // normalisation
    ftmp = 0.25 * _NSPIN * _NCOLOR / ( (float)(devT) * (float)(L1) * (float)(L1) * (float)(L1) );

    cvc_out[_GWI( 0,idx,V4)  ] = -cvc_tmp[ 0].x*ftmp; cvc_out[_GWI( 0,idx,V4)+1] = -cvc_tmp[ 0].y*ftmp;
    cvc_out[_GWI( 1,idx,V4)  ] = -cvc_tmp[ 1].x*ftmp; cvc_out[_GWI( 1,idx,V4)+1] = -cvc_tmp[ 1].y*ftmp;
    cvc_out[_GWI( 2,idx,V4)  ] = -cvc_tmp[ 2].x*ftmp; cvc_out[_GWI( 2,idx,V4)+1] = -cvc_tmp[ 2].y*ftmp;
    cvc_out[_GWI( 3,idx,V4)  ] = -cvc_tmp[ 3].x*ftmp; cvc_out[_GWI( 3,idx,V4)+1] = -cvc_tmp[ 3].y*ftmp;
    cvc_out[_GWI( 4,idx,V4)  ] = -cvc_tmp[ 4].x*ftmp; cvc_out[_GWI( 4,idx,V4)+1] = -cvc_tmp[ 4].y*ftmp;
    cvc_out[_GWI( 5,idx,V4)  ] = -cvc_tmp[ 5].x*ftmp; cvc_out[_GWI( 5,idx,V4)+1] = -cvc_tmp[ 5].y*ftmp;
    cvc_out[_GWI( 6,idx,V4)  ] = -cvc_tmp[ 6].x*ftmp; cvc_out[_GWI( 6,idx,V4)+1] = -cvc_tmp[ 6].y*ftmp;
    cvc_out[_GWI( 7,idx,V4)  ] = -cvc_tmp[ 7].x*ftmp; cvc_out[_GWI( 7,idx,V4)+1] = -cvc_tmp[ 7].y*ftmp;
    cvc_out[_GWI( 8,idx,V4)  ] = -cvc_tmp[ 8].x*ftmp; cvc_out[_GWI( 8,idx,V4)+1] = -cvc_tmp[ 8].y*ftmp;
    cvc_out[_GWI( 9,idx,V4)  ] = -cvc_tmp[ 9].x*ftmp; cvc_out[_GWI( 9,idx,V4)+1] = -cvc_tmp[ 9].y*ftmp;
    cvc_out[_GWI(10,idx,V4)  ] = -cvc_tmp[10].x*ftmp; cvc_out[_GWI(10,idx,V4)+1] = -cvc_tmp[10].y*ftmp;
    cvc_out[_GWI(11,idx,V4)  ] = -cvc_tmp[11].x*ftmp; cvc_out[_GWI(11,idx,V4)+1] = -cvc_tmp[11].y*ftmp;
    cvc_out[_GWI(12,idx,V4)  ] = -cvc_tmp[12].x*ftmp; cvc_out[_GWI(12,idx,V4)+1] = -cvc_tmp[12].y*ftmp;
    cvc_out[_GWI(13,idx,V4)  ] = -cvc_tmp[13].x*ftmp; cvc_out[_GWI(13,idx,V4)+1] = -cvc_tmp[13].y*ftmp;
    cvc_out[_GWI(14,idx,V4)  ] = -cvc_tmp[14].x*ftmp; cvc_out[_GWI(14,idx,V4)+1] = -cvc_tmp[14].y*ftmp;
    cvc_out[_GWI(15,idx,V4)  ] = -cvc_tmp[15].x*ftmp; cvc_out[_GWI(15,idx,V4)+1] = -cvc_tmp[15].y*ftmp;

    ftmp *= 4.;
    // if(idx==0)
    if(idx==102)
    {
      ct_out[0] = -counter_term[0].x * ftmp;
      ct_out[1] = -counter_term[0].y * ftmp;
      ct_out[2] = -counter_term[1].x * ftmp;
      ct_out[3] = -counter_term[1].y * ftmp;
      ct_out[4] = -counter_term[2].x * ftmp;
      ct_out[5] = -counter_term[2].y * ftmp;
      ct_out[6] = -counter_term[3].x * ftmp;
      ct_out[7] = -counter_term[3].y * ftmp;
    }


  }  // of if idx < N

}

/**********************************************************************
 * main program
 **********************************************************************/
int main(int argc, char **argv) {
    
  // int status;
  int c, filename_set=0, verbose=0;
  int mu, nu, x0, x1, x2, x3, ix;
  int imu, inu, isigma_mu, isigma_nu, ikappa, ilambda;
  unsigned int threadsPerBlock, blocksPerGrid;
  int i;

  double delta_mn, delta_mk, delta_nk, delta_ml, delta_nl, delta_lk;
  double sigma_mu, sigma_nu;
  float cvc_coeff[2304], phase[4];
  double *dptr = NULL;
  float *fptr  = NULL;
  // const double CVC_EPS = 5.e-15;

  void *cvc=NULL, *counter_term;
  // float WI_check;
  float ftmp;
  complex w, w1;

  char filename[80], contype[200];

  float *dev_cvc, *dev_ct;
  cudaDeviceProp prop;

  while ((c = getopt(argc, argv, "h?f:v:")) != -1) {
    switch (c) {
      case 'f':
        strcpy(filename, optarg);
        filename_set=1;
        break;
      case 'v':
        verbose = atoi( optarg );
        fprintf(stdout, "\n# [] using verbose mode %d\n", verbose);
        break;
      default:
        //usage();
        break;
    }
  }

  /* get the time stamp */
  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  T = T_global;
  L = LX;

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /***********************************************
   * device properties
   ***********************************************/
  HANDLE_ERROR( cudaGetDevice(&c) );
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, c) );
  fprintf(stdout, "\n--- General info for device no. %d\n", c);
  fprintf(stdout, "Name: %s\n", prop.name);
  fprintf(stdout, "Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Clock rate: %d\n", prop.clockRate);
  printf("Device copy overlap: ");
  if(prop.deviceOverlap) {
    printf("Enabled\n");
  } else {
    printf("Disabled\n");
  }
  printf("Kernel execution timeout: ");
  if(prop.kernelExecTimeoutEnabled) {
    printf("Enabled\n");
  } else {
    printf("Disabled\n");
  }
  printf("\n--- Memory info for device no. %d\n", c);
  printf("Total global mem: %ld\n", prop.totalGlobalMem);
  printf("Total constant mem: %ld\n", prop.totalConstMem);
  printf("Max mem pitch: %ld\n", prop.memPitch);
  printf("Texture alignment: %ld\n", prop.textureAlignment);
  printf("\n--- MP info for device no. %d\n", c);
  printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
  printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
  printf("Registers mem per mp: %d\n", prop.regsPerBlock);
  printf("Threads in warp: %d\n", prop.warpSize);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max thread dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0],
      prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max grid dimension: (%d, %d, %d)\n", prop.maxGridSize[0],
      prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("\n\n");

  /***********************************************
   * set number of threads and blocks
   ***********************************************/
  threadsPerBlock        = THREADS_PER_BLOCK;
  blocksPerGrid          = (VOLUME + threadsPerBlock-1)/threadsPerBlock;
  fprintf(stdout, "\n# [contractions] number threads per block: %u\n", threadsPerBlock);
  fprintf(stdout, "\n# [contractions] number blocks per grid  : %u\n", blocksPerGrid);

  // allocate memory for cvc
  cvc = calloc( 32*VOLUME, sizeof(double) );
  counter_term = calloc( 8, sizeof(double) );
  if( cvc == NULL || counter_term==NULL) {
    fprintf(stderr, "\nError, could not alloc cvc\n");
    exit(2);
  }

  /***************************
   * initialize on host
   ***************************/
 
  for(imu=0;imu<2304;imu++) cvc_coeff[imu] = 0.;
 
  // set the coefficients for the correlation functions
  for(imu=0; imu<4;imu++) {
  for(inu=0; inu<4;inu++) {
    delta_mn = (float)(imu==inu);

    for(isigma_mu=0; isigma_mu<2;isigma_mu++) {
    for(isigma_nu=0; isigma_nu<2;isigma_nu++) {

      sigma_mu =  2.*isigma_mu-1.;
      sigma_nu =  2.*isigma_nu-1.;

      // C_4_4
      cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, 4, 4) ] = delta_mn + sigma_mu*sigma_nu;

      // C_4_5, C_5_4, C_l_5, C_5_k
      // all 0
      
      // C_5_5
      cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, 5, 5) ] = -delta_mn + sigma_mu*sigma_nu;

      // C_4_k
      for(ikappa=0;ikappa<4;ikappa++) {
        delta_mk = (float)( imu == ikappa) ;
        delta_nk = (float)( inu == ikappa );
        cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, 4, ikappa) ] = delta_mk*sigma_nu + delta_nk*sigma_mu;
      }

      // C_l_4
      for( ilambda=0; ilambda<4;ilambda++) {
        delta_ml = (float)(imu==ilambda); 
        delta_nl = (float)(inu==ilambda);
        cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, ilambda, 4) ] = delta_ml*sigma_nu + delta_nl*sigma_mu;
      }

        // C_l_k
      for(ilambda=0; ilambda<4;ilambda++) {
      for(ikappa=0;  ikappa<4; ikappa++ ) {
        //*************************************
        //*************************************
        delta_ml = (float)(imu==ilambda);
        // ************************************
        delta_mk = (float)(imu==ikappa);
        // ************************************
        // ************************************
        delta_nl = (float)(inu==ilambda);
        // ************************************
        delta_nk = (float)(inu==ikappa);
        // ************************************
        // ************************************
        delta_lk = (float)(ilambda==ikappa);
        // ************************************
        // ************************************

        cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, ilambda, ikappa) ] = \
            delta_ml*delta_nk - delta_mn*delta_lk + delta_mk*delta_nl + delta_lk*sigma_mu*sigma_nu;

      }}

    }}  // of isigma_mu, isigma_nu

  }} // of imu, inu

  /**************************************************************************
   * test: print the matrix cvc_coeff
   **************************************************************************/
  if(verbose > 0) {
    for(imu=0;imu<4;imu++) {
    for(inu=0;inu<4;inu++) {
      fprintf(stdout, "# ---------------------------------------------------------------\n");
      fprintf(stdout, "# imu = %d; inu = %d\n", imu, inu);
      for(isigma_mu=0;isigma_mu<2;isigma_mu++) {
      for(isigma_nu=0;isigma_nu<2;isigma_nu++) {
        fprintf(stdout, "# ---------------------------------------------------------------\n");
        fprintf(stdout, "#\t sigma_mu = %e; sigma_nu = %e\n", 2.*isigma_mu-1., 2.*isigma_nu-1.);
        for(ilambda=0;ilambda<6;ilambda++) {
          fprintf(stdout, "%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n",
              cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, ilambda, 0) ],
              cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, ilambda, 1) ],
              cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, ilambda, 2) ],
              cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, ilambda, 3) ],
              cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, ilambda, 4) ],
              cvc_coeff[ _CVC_COEFF_IDX(imu, inu, isigma_mu, isigma_nu, ilambda, 5) ] );
        }
      }}
    }} 
  }
  /***************************************
   * allocate fields, initialize on device
   ***************************************/
  HANDLE_ERROR( cudaMalloc(&dev_cvc, 32*VOLUME*sizeof(float)) );
  HANDLE_ERROR( cudaMalloc(&dev_ct, 8*sizeof(float)) );

  HANDLE_ERROR( cudaMemcpyToSymbol( "devT", &T, sizeof(int), 0, cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpyToSymbol( "devL", &L, sizeof(int), 0, cudaMemcpyHostToDevice) );
  ftmp = (float)g_mu;
  fprintf(stdout, "# [] using mu = %f\n", ftmp);
  HANDLE_ERROR( cudaMemcpyToSymbol( "devMu", &ftmp, sizeof(float), 0, cudaMemcpyHostToDevice) );
  ftmp = (float)( 1. / (2. * g_kappa) - 4. );
  fprintf(stdout, "# [] using mq = %f\n", ftmp);
  HANDLE_ERROR( cudaMemcpyToSymbol( "devMq", &ftmp, sizeof(float), 0, cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpyToSymbol( "dev_cvc_coeff", cvc_coeff, 2304*sizeof(float), 0, cudaMemcpyHostToDevice) );
//  HANDLE_ERROR( cudaMemcpyToSymbol( dev_cvc_coeff, cvc_coeff, sizeof(cvc_coeff)) );

  /*************************
   * call kernel
   *************************/
  cvc_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_cvc, dev_ct, VOLUME);

  HANDLE_ERROR( cudaMemcpy(cvc, dev_cvc, 32*VOLUME*sizeof(float), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaMemcpy(counter_term, dev_ct, 8*sizeof(float), cudaMemcpyDeviceToHost) );

  fprintf(stdout, "\n# [] float counter terms:\n");
  fptr = (float*)counter_term; 
  fprintf(stdout, "\t%d\t%f\t%f\n", 0, fptr[0], fptr[1]);
  fprintf(stdout, "\t%d\t%f\t%f\n", 1, fptr[2], fptr[3]);
  fprintf(stdout, "\t%d\t%f\t%f\n", 2, fptr[4], fptr[5]);
  fprintf(stdout, "\t%d\t%f\t%f\n", 3, fptr[6], fptr[7]);

  // cast to double precision
  dptr = (double*)cvc;
  fptr = (float*)cvc;
/*
  for(ix=0;ix<VOLUME;ix++) {
    for(mu=0;mu<16;mu++) {
      fprintf(stdout, "%d\t%d\t%f\t%f\n", ix, mu, fptr[_GWI(mu,ix,VOLUME)], fptr[_GWI(mu,ix,VOLUME)+1]);
    }
  }
*/
  for(i=32*VOLUME-1;i>=0;i--) dptr[i] = (double)fptr[i];


  dptr = (double*)counter_term;
  fptr = (float*)counter_term;
  for(i=7;i>=0;i--) dptr[i] = (double)fptr[i];

  /*********************************************
   * add phase factors, subtract counter term
   *********************************************/
#ifdef _UNDEF
  for(mu=0; mu<4; mu++) {
    double *phi = (double*)cvc + _GWI(5*mu,0,VOLUME);

    for(x0=0; x0<T; x0++) {
      phase[0] = 2. * (double)(x0) * M_PI / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      phase[1] = 2. * (double)(x1) * M_PI / (double)LX;
    for(x2=0; x2<LY; x2++) {
      phase[2] = 2. * (double)(x2) * M_PI / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      phase[3] = 2. * (double)(x3) * M_PI / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      phi[2*ix  ] = - ((double*)counter_term)[2*mu  ];
      phi[2*ix+1] = - ((double*)counter_term)[2*mu+1];
    }}}}
  }  /* of mu */

  for(mu=0; mu<3; mu++) {
  for(nu=mu+1; nu<4; nu++) {
    double *phi = (double*)cvc + _GWI(4*mu+nu,0,VOLUME);
    double *chi = (double*)cvc + _GWI(4*nu+mu,0,VOLUME);

    for(x0=0; x0<T; x0++) {
      phase[0] =  (double)(x0) * M_PI / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      phase[1] =  (double)(x1) * M_PI / (double)LX;
    for(x2=0; x2<LY; x2++) {
      phase[2] =  (double)(x2) * M_PI / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      phase[3] =  (double)(x3) * M_PI / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re =  cos( phase[mu] - phase[nu] );
      w.im =  sin( phase[mu] - phase[nu] );
      _co_eq_co_ti_co(&w1,(complex*)( phi+2*ix ), &w);
      phi[2*ix  ] = w1.re;
      phi[2*ix+1] = w1.im;

      w.re =  cos( phase[nu] - phase[mu] );
      w.im =  sin( phase[nu] - phase[mu] );
      _co_eq_co_ti_co(&w1,(complex*)( chi+2*ix ), &w);
      chi[2*ix  ] = w1.re;
      chi[2*ix+1] = w1.im;
    }}}}
  }}  /* of mu and nu */
#endif
  // write to file
  sprintf(filename, "pi_L%.2dT%.2d_mu%6.4f", L, T, g_mu);
  sprintf(contype, "tree-level vacuum polarization");
  write_lime_contraction((double*)cvc, filename, 64, 16, contype, Nconf, 0);

  sprintf(filename, "pi_L%.2dT%.2d_mu%6.4f.ascii", L, T, g_mu);
  write_contraction((double*)cvc, NULL, filename, 16, 2, 0); 

  dptr = (double*)counter_term;
  fprintf(stdout, "\n# [] counter terms:\n");
  fprintf(stdout, "\t%d\t%e\t%e\n", 0, dptr[0], dptr[1]);
  fprintf(stdout, "\t%d\t%e\t%e\n", 0, dptr[2], dptr[3]);
  fprintf(stdout, "\t%d\t%e\t%e\n", 0, dptr[4], dptr[5]);
  fprintf(stdout, "\t%d\t%e\t%e\n", 0, dptr[6], dptr[7]);
#ifdef _UNDEF 
#endif
  /*************************************
   * free and finalize
   *************************************/
  if( cvc!=NULL ) free(cvc);
  if( counter_term!=NULL ) free(counter_term);
  cudaFree( dev_cvc );
  cudaFree( dev_ct );
  g_the_time = time(NULL);
  fprintf(stdout, "\n# [contractions] %s# [contractions] end of run\n", ctime(&g_the_time));
  fprintf(stderr, "\n# [contractions] %s# [contractions] end of run\n", ctime(&g_the_time));

  return(0);
}
