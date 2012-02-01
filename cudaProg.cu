#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#define MAIN_PROGRAM
extern "C" 
{
#include "lime.h"
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "contractions_io.h"
#include "read_input_parser.h"
}

#define THREADS_PER_BLOCK 256
void print_device_properties(struct cudaDeviceProp p, FILE*ofs);

__device__ void cm_eq_cm_ti_cm_2x2(float2*u, float2*v, float2*w);
__device__ void cm_eq_cm_ti_cm(float2*u, float2*v, float2*w);
__device__ void cm_eq_cm_ti_cm_dag(float2*u, float2*v, float2*w);
__device__ void cm_eq_zero(float2*u);
__device__ void cm_eq_id(float2*u);
__device__ void cm_reconstruct_gaugelink (float2*s_field, float4*g_field_1, float4*g_field_2);
__device__ void re_eq_tr_cm_ti_cm_dag(float *r,  float2*u, float2*v);
__device__ void re_eq_tr_cm(float *r,  float2*u);

__global__ void d_init_geometry(uint4*up, uint4*dn);
__global__ void plaquette(float*plaq, float4*g_field);

__global__ void reconstruct_gauge(float2*rec_gauge, float4*g_field);
__global__ void plaquette(float*plaq, float4*g_field);

__constant__ unsigned int devVolume, devVol3, devT, devL;
float4 *d_gauge_field;
__device__ uint4 *d_iup, *d_idn;
uint4 *d_iup_field, *d_idn_field;

/****************************************************************
 * initialize the next-neighbor fields
 ****************************************************************/
__global__ void d_init_geometry(uint4*up, uint4*dn) {
  unsigned int tid, gid;
  unsigned int x0, x1, x2, x3;
  unsigned int y0, y1, y2, y3;
  uint4 nn;
  unsigned int L, L2, L3, uitmp;
  tid = threadIdx.x;
  gid = blockIdx.x*blockDim.x + threadIdx.x;

  if(gid==0) {
    d_iup = up;
    d_idn = dn;
  }
  __syncthreads();

  L = devL;
  L2 = L*L;
  L3 = L2*L;

  /*************************************/

  x0 = gid / L3;
  uitmp = gid - x0*L3;
  x1 = uitmp / L2;
  uitmp = uitmp - x1*L2;
  x2 = uitmp / L;
  x3 = uitmp - x2*L;

  /*************************************/

  y0 = x0+1;
  y0 = (y0>=devT) ? y0-devT : y0;
  y1=x1; y2=x2; y3=x3;
  nn.x = y0*L3 + y1*L2 + y2*L + y3;

  y1 = x1+1;
  y1 = (y1>=devL) ? y1-devL : y1;
  y0=x0; y2=x2; y3=x3;
  nn.y = y0*L3 + y1*L2 + y2*L + y3;

  y2 = x2+1;
  y2 = (y2>=devL) ? y2-devL : y2;
  y0=x0; y1=x1; y3=x3;
  nn.z = y0*L3 + y1*L2 + y2*L + y3;

  y3 = x3+1;
  y3 = (y3>=devL) ? y3-devL : y3;
  y0=x0; y1=x1; y2=x2;
  nn.w = y0*L3 + y1*L2 + y2*L + y3;

  d_iup[gid] = nn;

  /*************************************/

  y0 = (x0+devT)-1;
  y0 = (y0>=devT) ? y0-devT : y0;
  y1=x1; y2=x2; y3=x3;
  nn.x = y0*L3 + y1*L2 + y2*L + y3;

  y1 = (x1+devL)-1;
  y1 = (y1>=devL) ? y1-devL : y1;
  y0=x0; y2=x2; y3=x3;
  nn.y = y0*L3 + y1*L2 + y2*L + y3;

  y2 = (x2+devL)-1;
  y2 = (y2>=devL) ? y2-devL : y2;
  y0=x0; y1=x1; y3=x3;
  nn.z = y0*L3 + y1*L2 + y2*L + y3;

  y3 = (x3+devL)-1;
  y3 = (y3>=devL) ? y3-devL : y3;
  y0=x0; y1=x1; y2=x2;
  nn.w = y0*L3 + y1*L2 + y2*L + y3;

  d_idn[gid] = nn;

  __syncthreads();
}


/***********************************
 * u = v x w for 2x2 matrices
 ***********************************/
__device__ void cm_eq_cm_ti_cm_2x2(float2*u, float2*v, float2*w) {
  u[0].x = v[0].x*w[0].x - v[0].y*w[0].y + v[1].x*w[2].x - v[1].y*w[2].y;
  u[0].y = v[0].x*w[0].y + v[0].y*w[0].x + v[1].x*w[2].y + v[1].y*w[2].x;
  u[1].x = v[0].x*w[1].x - v[0].y*w[1].y + v[1].x*w[3].x - v[1].y*w[3].y;
  u[1].y = v[0].x*w[1].y + v[0].y*w[1].x + v[1].x*w[3].y + v[1].y*w[3].x;
  u[2].x = v[2].x*w[0].x - v[2].y*w[0].y + v[3].x*w[2].x - v[3].y*w[2].y;
  u[2].y = v[2].x*w[0].y + v[2].y*w[0].x + v[3].x*w[2].y + v[3].y*w[2].x;
  u[3].x = v[2].x*w[1].x - v[2].y*w[1].y + v[3].x*w[3].x - v[3].y*w[3].y;
  u[3].y = v[2].x*w[1].y + v[2].y*w[1].x + v[3].x*w[3].y + v[3].y*w[3].x;
}
/***********************************
 * u = v x w
 ***********************************/
__device__ void cm_eq_cm_ti_cm(float2*u, float2*v, float2*w) {
  u[0].x = v[0].x*w[0].x - v[0].y*w[0].y + v[1].x*w[3].x - v[1].y*w[3].y + v[2].x*w[6].x - v[2].y*w[6].y;
  u[0].y = v[0].x*w[0].y + v[0].y*w[0].x + v[1].x*w[3].y + v[1].y*w[3].x + v[2].x*w[6].y + v[2].y*w[6].x;
  u[1].x = v[0].x*w[1].x - v[0].y*w[1].y + v[1].x*w[4].x - v[1].y*w[4].y + v[2].x*w[7].x - v[2].y*w[7].y;
  u[1].y = v[0].x*w[1].y + v[0].y*w[1].x + v[1].x*w[4].y + v[1].y*w[4].x + v[2].x*w[7].y + v[2].y*w[7].x;
  u[2].x = v[0].x*w[2].x - v[0].y*w[2].y + v[1].x*w[5].x - v[1].y*w[5].y + v[2].x*w[8].x - v[2].y*w[8].y;
  u[2].y = v[0].x*w[2].y + v[0].y*w[2].x + v[1].x*w[5].y + v[1].y*w[5].x + v[2].x*w[8].y + v[2].y*w[8].x;

  u[3].x = v[3].x*w[0].x - v[3].y*w[0].y + v[4].x*w[3].x - v[4].y*w[3].y + v[5].x*w[6].x - v[5].y*w[6].y;
  u[3].y = v[3].x*w[0].y + v[3].y*w[0].x + v[4].x*w[3].y + v[4].y*w[3].x + v[5].x*w[6].y + v[5].y*w[6].x;
  u[4].x = v[3].x*w[1].x - v[3].y*w[1].y + v[4].x*w[4].x - v[4].y*w[4].y + v[5].x*w[7].x - v[5].y*w[7].y;
  u[4].y = v[3].x*w[1].y + v[3].y*w[1].x + v[4].x*w[4].y + v[4].y*w[4].x + v[5].x*w[7].y + v[5].y*w[7].x;
  u[5].x = v[3].x*w[2].x - v[3].y*w[2].y + v[4].x*w[5].x - v[4].y*w[5].y + v[5].x*w[8].x - v[5].y*w[8].y;
  u[5].y = v[3].x*w[2].y + v[3].y*w[2].x + v[4].x*w[5].y + v[4].y*w[5].x + v[5].x*w[8].y + v[5].y*w[8].x;

  u[6].x = v[6].x*w[0].x - v[6].y*w[0].y + v[7].x*w[3].x - v[7].y*w[3].y + v[8].x*w[6].x - v[8].y*w[6].y;
  u[6].y = v[6].x*w[0].y + v[6].y*w[0].x + v[7].x*w[3].y + v[7].y*w[3].x + v[8].x*w[6].y + v[8].y*w[6].x;
  u[7].x = v[6].x*w[1].x - v[6].y*w[1].y + v[7].x*w[4].x - v[7].y*w[4].y + v[8].x*w[7].x - v[8].y*w[7].y;
  u[7].y = v[6].x*w[1].y + v[6].y*w[1].x + v[7].x*w[4].y + v[7].y*w[4].x + v[8].x*w[7].y + v[8].y*w[7].x;
  u[8].x = v[6].x*w[2].x - v[6].y*w[2].y + v[7].x*w[5].x - v[7].y*w[5].y + v[8].x*w[8].x - v[8].y*w[8].y;
  u[8].y = v[6].x*w[2].y + v[6].y*w[2].x + v[7].x*w[5].y + v[7].y*w[5].x + v[8].x*w[8].y + v[8].y*w[8].x;
}

/***********************************
 * u = v x w^dagger
 ***********************************/
__device__ void cm_eq_cm_ti_cm_dag(float2*u, float2*v, float2*w) {
  u[0].x =  v[0].x*w[0].x + v[0].y*w[0].y + v[1].x*w[3].x + v[1].y*w[3].y + v[2].x*w[6].x + v[2].y*w[6].y;
  u[0].y = -v[0].x*w[0].y + v[0].y*w[0].x - v[1].x*w[3].y + v[1].y*w[3].x - v[2].x*w[6].y + v[2].y*w[6].x;
  u[1].x =  v[0].x*w[1].x + v[0].y*w[1].y + v[1].x*w[4].x + v[1].y*w[4].y + v[2].x*w[7].x + v[2].y*w[7].y;
  u[1].y = -v[0].x*w[1].y + v[0].y*w[1].x - v[1].x*w[4].y + v[1].y*w[4].x - v[2].x*w[7].y + v[2].y*w[7].x;
  u[2].x =  v[0].x*w[2].x + v[0].y*w[2].y + v[1].x*w[5].x + v[1].y*w[5].y + v[2].x*w[8].x + v[2].y*w[8].y;
  u[2].y = -v[0].x*w[2].y + v[0].y*w[2].x - v[1].x*w[5].y + v[1].y*w[5].x - v[2].x*w[8].y + v[2].y*w[8].x;

  u[3].x =  v[3].x*w[0].x + v[3].y*w[0].y + v[4].x*w[3].x + v[4].y*w[3].y + v[5].x*w[6].x + v[5].y*w[6].y;
  u[3].y = -v[3].x*w[0].y + v[3].y*w[0].x - v[4].x*w[3].y + v[4].y*w[3].x - v[5].x*w[6].y + v[5].y*w[6].x;
  u[4].x =  v[3].x*w[1].x + v[3].y*w[1].y + v[4].x*w[4].x + v[4].y*w[4].y + v[5].x*w[7].x + v[5].y*w[7].y;
  u[4].y = -v[3].x*w[1].y + v[3].y*w[1].x - v[4].x*w[4].y + v[4].y*w[4].x - v[5].x*w[7].y + v[5].y*w[7].x;
  u[5].x =  v[3].x*w[2].x + v[3].y*w[2].y + v[4].x*w[5].x + v[4].y*w[5].y + v[5].x*w[8].x + v[5].y*w[8].y;
  u[5].y = -v[3].x*w[2].y + v[3].y*w[2].x - v[4].x*w[5].y + v[4].y*w[5].x - v[5].x*w[8].y + v[5].y*w[8].x;

  u[6].x =  v[6].x*w[0].x + v[6].y*w[0].y + v[7].x*w[3].x + v[7].y*w[3].y + v[8].x*w[6].x + v[8].y*w[6].y;
  u[6].y = -v[6].x*w[0].y + v[6].y*w[0].x - v[7].x*w[3].y + v[7].y*w[3].x - v[8].x*w[6].y + v[8].y*w[6].x;
  u[7].x =  v[6].x*w[1].x + v[6].y*w[1].y + v[7].x*w[4].x + v[7].y*w[4].y + v[8].x*w[7].x + v[8].y*w[7].y;
  u[7].y = -v[6].x*w[1].y + v[6].y*w[1].x - v[7].x*w[4].y + v[7].y*w[4].x - v[8].x*w[7].y + v[8].y*w[7].x;
  u[8].x =  v[6].x*w[2].x + v[6].y*w[2].y + v[7].x*w[5].x + v[7].y*w[5].y + v[8].x*w[8].x + v[8].y*w[8].y;
  u[8].y = -v[6].x*w[2].y + v[6].y*w[2].x - v[7].x*w[5].y + v[7].y*w[5].x - v[8].x*w[8].y + v[8].y*w[8].x;
}

/***********************************
 * set u to zero matrix
 ***********************************/
__device__ void cm_eq_zero(float2*u) {
  u[0].x = 0.; u[0].y = 0.;
  u[1].x = 0.; u[1].y = 0.;
  u[2].x = 0.; u[2].y = 0.;
  u[3].x = 0.; u[3].y = 0.;
  u[4].x = 0.; u[4].y = 0.;
  u[5].x = 0.; u[5].y = 0.;
  u[6].x = 0.; u[6].y = 0.;
  u[7].x = 0.; u[7].y = 0.;
  u[8].x = 0.; u[8].y = 0.;
}

/***********************************
 * set u to identity matrix
 ***********************************/
__device__ void cm_eq_id(float2*u) {
  u[0].x = 1.; u[0].y = 0.;
  u[1].x = 0.; u[1].y = 0.;
  u[2].x = 0.; u[2].y = 0.;
  u[3].x = 0.; u[3].y = 0.;
  u[4].x = 1.; u[4].y = 0.;
  u[5].x = 0.; u[5].y = 0.;
  u[6].x = 0.; u[6].y = 0.;
  u[7].x = 0.; u[7].y = 0.;
  u[8].x = 1.; u[8].y = 0.;
}

/*********************************************
 * kernel to reconstruct the gauge field
 *   from the compressed version
 *********************************************/
__device__ void cm_reconstruct_gaugelink (float2*s_field, float4*g_field_1, float4*g_field_2) {

  __shared__ float ftmp, ftmp2;
  __shared__ float v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y;
  __shared__ float a1x, a1y, c1x, c1y;
  __shared__ float g1x, g1y, g1z, g1w, g2x, g2y, g2z, g2w;

  g1x = g_field_1[0].x;
  g1y = g_field_1[0].y;
  g1z = g_field_1[0].z;
  g1w = g_field_1[0].w;
  g2x = g_field_2[0].x;
  g2y = g_field_2[0].y;
  g2z = g_field_2[0].z;
  g2w = g_field_2[0].w;

  ftmp = g1x*g1x + g1y*g1y +g1z*g1z +g1w*g1w; // this is N^2
  ftmp2 = sqrtf(1. - ftmp);

  a1x = ftmp2*cosf(g2z);                  
  a1y = ftmp2*sinf(g2z);
  ftmp = 1./ftmp;

  v0x = -g1z; 
  v0y =  g1w;  
  v1x =  g1x; 
  v1y = -g1y;
  v2x = -(a1x*g1x + a1y*g1y); 
  v2y = -(a1x*g1y - a1y*g1x);
  v3x = -(a1x*g1z + a1y*g1w); 
  v3y = -(a1x*g1w - a1y*g1z);

  ftmp2 = sqrtf( 1. - ( a1x*a1x + a1y*a1y + g2x*g2x + g2y*g2y ) );
  c1x = cosf(g2w)*ftmp2; 
  c1y = sinf(g2w)*ftmp2;

  s_field[0].x = a1x;
  s_field[0].y = a1y;
  s_field[1].x = g1x;
  s_field[1].y = g1y;
  s_field[2].x = g1z;
  s_field[2].y = g1w;
  s_field[3].x = g2x;
  s_field[3].y = g2y;
  s_field[4].x =  c1x*v0x + c1y*v0y + g2x*v2x - g2y*v2y;
  s_field[4].y =  c1x*v0y - c1y*v0x + g2x*v2y + g2y*v2x;
  s_field[4].x *= ftmp;
  s_field[4].y *= ftmp;
  s_field[5].x =  c1x*v1x + c1y*v1y + g2x*v3x - g2y*v3y;
  s_field[5].y =  c1x*v1y - c1y*v1x + g2x*v3y + g2y*v3x;
  s_field[5].x *= ftmp;
  s_field[5].y *= ftmp;
  s_field[6].x = c1x;
  s_field[6].y = c1y;
  s_field[7].x = -g2x*v0x - g2y*v0y + c1x*v2x - c1y*v2y;
  s_field[7].y = -g2x*v0y + g2y*v0x + c1x*v2y + c1y*v2x;
  s_field[7].x *= ftmp;
  s_field[7].y *= ftmp;
  s_field[8].x = -g2x*v1x - g2y*v1y + c1x*v3x - c1y*v3y;
  s_field[8].y = -g2x*v1y + g2y*v1x + c1x*v3y + c1y*v3x;
  s_field[8].x *= ftmp;
  s_field[8].y *= ftmp;
} 

/********************************************************************
 * calculate Re ( Tr [ u x v^dagger ] )
 ********************************************************************/
__device__ void re_eq_tr_cm_ti_cm_dag(float r[1],  float2 u[9], float2 v[9]) {
  float tmp;
  tmp  = u[0].x * v[0].x;
  tmp += u[0].y * v[0].y;
  tmp += u[1].x * v[1].x;
  tmp += u[1].y * v[1].y;
  tmp += u[2].x * v[2].x;
  tmp += u[2].y * v[2].y;
  /*
  tmp += u[3].x * v[3].x;
  tmp += u[3].y * v[3].y;
  tmp += u[4].x * v[4].x;
  tmp += u[4].y * v[4].y;
  tmp += u[5].x * v[5].x;
  tmp += u[5].y * v[5].y;
  tmp += u[6].x * v[6].x;
  tmp += u[6].y * v[6].y;
  tmp += u[7].x * v[7].x;
  tmp += u[7].y * v[7].y;
  tmp += u[8].x * v[8].x;
  tmp += u[8].y * v[8].y;
  */
  r[0] = tmp;
}

/********************************************************************
 * calculate Re ( Tr [ u x v^dagger ] )
 ********************************************************************/
__device__ void re_eq_tr_cm(float *r,  float2*u) {
  __shared__ float tmp;
  tmp  = u[0].x;
  tmp += u[4].x;
  tmp += u[8].x;
  r[0] = tmp;
}

/****************************************************************
 * reconstruct the gauge field to a global device memory
 ****************************************************************/
__global__ void reconstruct_gauge(float2*rec_gauge, float4*g_field) {
  unsigned int tid, gid;
  unsigned int uitmp;

  tid = threadIdx.x;
  gid = blockIdx.x*blockDim.x + threadIdx.x;
  uitmp = (devT-1)*devVol3;

  /* reconstruct the spatial links at x */
  cm_eq_id(rec_gauge+36*gid);
  cm_reconstruct_gaugelink(rec_gauge+36*gid+ 9, g_field+gid,             g_field+gid+  devVolume);
  cm_reconstruct_gaugelink(rec_gauge+36*gid+18, g_field+gid+2*devVolume, g_field+gid+3*devVolume);
  cm_reconstruct_gaugelink(rec_gauge+36*gid+27, g_field+gid+4*devVolume, g_field+gid+5*devVolume);
  if(gid >= uitmp) {
    cm_reconstruct_gaugelink(rec_gauge+36*gid,   (g_field+(6*devVolume + (gid-uitmp))), (g_field+(6*devVolume + (gid-uitmp)))+devVol3);
  }
}

/****************************************************************
 * calculate the plaquette
 ****************************************************************/
__global__ void plaquette(float*plaq, float4*g_field) {
  unsigned int tid, gid;
  unsigned int uitmp;
  unsigned int xp0, xp1, xp2, xp3;
  float ftmp[1];
  float2 g0[9], g1[9], g2[9], g3[9], g4[9], u[9], v[9], w[9];

  extern __shared__ float plaq_field[];

  tid = threadIdx.x;
  gid = blockIdx.x*blockDim.x + threadIdx.x;
  xp0 = d_iup[gid].x;
  xp1 = d_iup[gid].y;
  xp2 = d_iup[gid].z;
  xp3 = d_iup[gid].w;

  plaq_field[tid] = 0.;
  /* reconstruct the spatial links at x */
  cm_reconstruct_gaugelink(g0, g_field+gid,             g_field+gid+  devVolume);
  cm_reconstruct_gaugelink(g1, g_field+gid+2*devVolume, g_field+gid+3*devVolume);
  cm_reconstruct_gaugelink(g2, g_field+gid+4*devVolume, g_field+gid+5*devVolume);

  /* U_2 (x+1) */
  cm_reconstruct_gaugelink(g3, g_field+xp1+2*devVolume, g_field+xp1+3*devVolume);
  /* U_1 (x+3) */
  cm_reconstruct_gaugelink(g4, g_field+xp2,             g_field+xp3+  devVolume);
  cm_eq_cm_ti_cm(u, g0, g3);
  cm_eq_cm_ti_cm(v, g1, g4);
  cm_eq_cm_ti_cm(w, u, v);
  //re_eq_tr_cm(ftmp, w);
  plaq_field[tid] += ftmp[0];

//  re_eq_tr_cm(&ftmp, g3);
//  plaq_field[tid] += ftmp;

//  re_eq_tr_cm(&ftmp, g4);
//  plaq_field[tid] += ftmp;
//  re_eq_tr_cm(&ftmp, u);
//  plaq_field[tid] += ftmp;

/*
  cm_reconstruct_gaugelink(g3, g_field+xpn.y+4*devVolume, devVolume);
  cm_reconstruct_gaugelink(g4, g_field+xpn.w, devVolume);
  cm_eq_cm_ti_cm(u, g0, g3);
  cm_eq_cm_ti_cm(v, g2, g4);
  re_eq_tr_cm_ti_cm_dag(&ftmp, u, v);
  plaq_field[tid] += ftmp;

  cm_reconstruct_gaugelink(g3, g_field+xpn.z+4*devVolume, devVolume);
  cm_reconstruct_gaugelink(g4, g_field+xpn.w+2*devVolume, devVolume);
  cm_eq_cm_ti_cm(u, g1, g3);
  cm_eq_cm_ti_cm(v, g2, g4);
  re_eq_tr_cm_ti_cm_dag(&ftmp, u, v);
  plaq_field[tid] += ftmp;
*/
/*
  uitmp = (devT-1)*devVol3;
  if(gid>=uitmp) {

    cm_reconstruct_gaugelink(w, g_field+6*devVolume + gid - uitmp, devVol3);

    cm_reconstruct_gaugelink(g4, g_field+6*devVolume + xpn.y - uitmp, devVol3);
    cm_reconstruct_gaugelink(g3, g_field+xpn.x, devVolume);
    cm_eq_cm_ti_cm(u, w, g3);
    cm_eq_cm_ti_cm(v, g0, g4);
    re_eq_tr_cm_ti_cm_dag(&ftmp, u, v);
    plaq_field[tid] += ftmp;

    cm_reconstruct_gaugelink(g4, g_field+6*devVolume + xpn.z - uitmp, devVol3);
    cm_reconstruct_gaugelink(g3, g_field+xpn.x+2*devVolume, devVolume);
    cm_eq_cm_ti_cm(u, w, g3);
    cm_eq_cm_ti_cm(v, g1, g4);
    re_eq_tr_cm_ti_cm_dag(&ftmp, u, v);
    plaq_field[tid] += ftmp;

    cm_reconstruct_gaugelink(g4, g_field+6*devVolume + xpn.w - uitmp, devVol3);
    cm_reconstruct_gaugelink(g3, g_field+xpn.x+4*devVolume, devVolume);
    cm_eq_cm_ti_cm(u, w, g3);
    cm_eq_cm_ti_cm(v, g2, g4);
    re_eq_tr_cm_ti_cm_dag(&ftmp, u, v);
    plaq_field[tid] += ftmp;
  } else {
    cm_reconstruct_gaugelink(g3, g_field+xpn.x, devVolume);
    re_eq_tr_cm_ti_cm_dag(&ftmp, g3, g0);
    plaq_field[tid] += ftmp;

    cm_reconstruct_gaugelink(g3, g_field+xpn.x+2*devVolume, devVolume);
    re_eq_tr_cm_ti_cm_dag(&ftmp, g3, g1);
    plaq_field[tid] += ftmp;

    cm_reconstruct_gaugelink(g3, g_field+xpn.x+4*devVolume, devVolume);
    re_eq_tr_cm_ti_cm_dag(&ftmp, g3, g2);
    plaq_field[tid] += ftmp;
  }
*/

  __syncthreads();
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) { plaq_field[tid] += plaq_field[tid + s]; }
    __syncthreads();
  }
  //if (tid == 0) { plaq[blockIdx.x] = plaq_field[0]; } 
  if (tid == 0) { plaq[blockIdx.x] = (float)xp1; } 
}


/**********************************************************************************
 **********************************************************************************
 **
 ** end of device function declaration / begin of main function
 **
 **********************************************************************************
 **********************************************************************************/

void usage(void) {
  fprintf(stdout, "# Programme; exit\n");
  exit(0);
}

int main (int argc, char *argv[]) {

  int status, c;
  int num_fields = 0;
  int filename_set = 0;
  int it, ix, iix, count, itmp, itmp2, i, j;
  int VOL3;
  unsigned int uitmp, *nn_field;
  unsigned int threadsPerBlock, blocksPerGrid;
  float **spinor_field_flt=NULL, *gauge_field_flt=NULL;
  double *gauge_transform=NULL, *gauge_aux=NULL, U_[18];
  float *h_plaq_field;
  float *gauge_aux2=NULL;
  double plaq, dtmp;
  double ratime, retime;
  char filename[400];
  void *vptr;

  cudaError_t cuderr;
  int dev_num;
  struct cudaDeviceProp *dev_prop;
  float2 *d_gauge_rec;
  float *d_plaq_field;

  /****************************************
   * initialize the distance vectors
   ****************************************/

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

  g_the_time = time(NULL);


  mpi_init(argc, argv);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "Error, T and L's must be set\n");
    usage();
  }

  if(LX!=LY || LX!=LZ || LY!=LZ) {
    if(g_proc_id==0) fprintf(stderr, "Error, LX, LY and LZ must be mutually equal\n");
    usage();
  }

  VOL3 = LX*LY*LZ;
  T = T_global;

  status = init_geometry();
  if(status != 0) {
    fprintf(stderr, "Error from init_geometry, status was %d\n", status);
    exit(1);
  }

  geometry();

  /***************************************
   * try device management
   ***************************************/
  cuderr = cudaGetDeviceCount(&dev_num);
  fprintf(stdout, "\n# found %d devices\n", dev_num);
  dev_prop = (struct cudaDeviceProp*)malloc(dev_num*sizeof(struct cudaDeviceProp));
  if(dev_prop==NULL) {
    fprintf(stderr, "Error, could not alloc dev_prop\n");
    exit(109);
  }
  for(i=0; i<dev_num; i++) {
    cuderr = cudaGetDeviceProperties(dev_prop+i, i);
    print_device_properties(dev_prop[i], stdout);
  }
  free(dev_prop);

  cuderr = cudaSetDevice (0);
  if (cuderr == cudaErrorSetOnActiveProcess) {
    cudaGetDevice(&itmp);
    fprintf(stderr, "Error, could not set device 0, already using device %d\n", itmp);
  }

  /***********************************************
   * set number of threads and blocks
   ***********************************************/
  threadsPerBlock = THREADS_PER_BLOCK;
  blocksPerGrid   = (VOLUME+threadsPerBlock-1)/threadsPerBlock;
  fprintf(stdout, "# number threads per block: %u\n", threadsPerBlock);
  fprintf(stdout, "# number blocks per grid  : %u\n", blocksPerGrid);

  /************************************
   * initialise device constants
   ************************************/
  uitmp = (unsigned int)T;
  if( (cuderr = cudaMemcpyToSymbol("devT", &uitmp, sizeof(unsigned int))) != cudaSuccess) {
    fprintf(stderr, "Error, could not set devT\n");
    exit(113);
  }

  uitmp = (unsigned int)LX;
  if( (cuderr = cudaMemcpyToSymbol("devL", &uitmp, sizeof(unsigned int))) != cudaSuccess) {
    fprintf(stderr, "Error, could not set devL\n");
    exit(113);
  }
  uitmp =(unsigned int)VOLUME;
  if( (cuderr = cudaMemcpyToSymbol("devVolume", &uitmp, sizeof(unsigned int))) != cudaSuccess) {
    fprintf(stderr, "Error, could not set devVolume\n");
    exit(113);
  }
  uitmp =(unsigned int)VOL3;
  if( (cuderr = cudaMemcpyToSymbol("devVol3", &uitmp, sizeof(unsigned int))) != cudaSuccess) {
    fprintf(stderr, "Error, could not set devVol3\n");
    exit(113);
  }

  /************************************************
   * allocate memory for the nn fields on device
   ************************************************/
  uitmp = VOLUME*sizeof(uint4);
  cuderr = cudaMalloc(&d_iup_field, uitmp);
  if(cuderr != cudaSuccess) {
    fprintf(stderr, "Error, could not allocate mem on device\n");
    exit(110);
  }

  cuderr = cudaMalloc(&d_idn_field, uitmp);
  if(cuderr != cudaSuccess) {
    fprintf(stderr, "Error, could not allocate mem on device\n");
    exit(111);
  }

  d_init_geometry<<<blocksPerGrid, threadsPerBlock>>>(d_iup_field, d_idn_field);

  /*********************************************************************************
   **                        end of initialization part                           **
   *********************************************************************************/

  /* read the gauge field */
  alloc_gauge_field_dbl(&g_gauge_field, 72*VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
#ifdef MPI
  xchange_gauge();
#endif
  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

  /* alloc gauge transform field */
  alloc_gauge_field_dbl(&gauge_transform, VOLUME*18);
  if(gauge_transform == NULL) {
    fprintf(stderr, "Error, could not alloc gauge transform field\n");
    exit(16);
  }

  set_temporal_gauge(gauge_transform);
  alloc_gauge_field_dbl(&gauge_aux, VOLUME*72);
  apply_gauge_transform(gauge_aux, gauge_transform, g_gauge_field);

  plaquette2(&plaq, gauge_aux);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value after gauge transform: %25.16e\n", plaq);

  alloc_gauge_field_flt(&gauge_field_flt, 8*(3*T+1)*VOL3);
  compress_gauge(gauge_field_flt, gauge_aux);

  /************************************************
   * allocate memory for the gauge field on device
   ************************************************/
 
  uitmp = ( 6*(unsigned int)VOLUME+2*(unsigned int)VOL3 )*sizeof(float4);
  cuderr = cudaMalloc(&d_gauge_field, uitmp);
  if(cuderr != cudaSuccess) {
    fprintf(stderr, "Error, could not allocate mem on device\n");
    exit(112);
  }
  if( (cuderr = cudaMemcpy(d_gauge_field, gauge_field_flt, uitmp, cudaMemcpyHostToDevice)) != cudaSuccess ) {
    fprintf(stderr, "Error, could not memcpy gauge field to device\n");
    exit(115);
  }

  uitmp = blocksPerGrid * sizeof(float);
  if( (cuderr = cudaMalloc(&d_plaq_field, uitmp)) != cudaSuccess ) {
    fprintf(stderr, "Error, could not alloc field on device\n");
    exit(125);
  }
  if( (h_plaq_field = (float*)malloc(uitmp))==NULL ) {
    fprintf(stderr, "Error, could not alloc field on host\n");
    exit(16);
  }

  plaquette<<<blocksPerGrid, threadsPerBlock, uitmp>>>(d_plaq_field, d_gauge_field);
  if( (cuderr=cudaMemcpy(h_plaq_field, d_plaq_field, uitmp, cudaMemcpyDeviceToHost))!=cudaSuccess){
    fprintf(stderr, "Error, could not memcpy field from device to host\n");
    exit(127);
  }
  for(i=0; i<blocksPerGrid; i++) fprintf(stdout, "# plaq(%d) = %25.16e\n", i, h_plaq_field[i]);
  for(i=1; i<blocksPerGrid; i++) h_plaq_field[0] += h_plaq_field[i];
  fprintf(stdout, "# plaq as measured on device: %25.16e\n", h_plaq_field[0]);
  free(h_plaq_field);
  cudaFree(d_plaq_field);


  /********************************************************************************
   ********************************************************************************
   **
   ** free and finalize
   **
   ********************************************************************************
   ********************************************************************************/
  cudaFree(d_iup_field);
  cudaFree(d_idn_field);


  fprintf(stderr, "\n# %s# end of run\n", ctime(&g_the_time));
  fflush(stderr);

  fprintf(stdout, "\n# %s# end of run\n", ctime(&g_the_time));
  fflush(stdout);

  return(0);

}


void print_device_properties (struct cudaDeviceProp p, FILE*ofs) {

  fprintf(ofs, "\n# device properties:\n");
  fprintf(ofs, "# device name: %s\n", p.name);
  fprintf(ofs, "# device global memory: %u\n", p.totalGlobalMem);
  fprintf(ofs, "# device no. of shared memory per block: %u\n", p.sharedMemPerBlock);
  fprintf(ofs, "# device no. of registers per block: %d\n", p.regsPerBlock);
  fprintf(ofs, "# device warp size: %d\n", p.warpSize);
  fprintf(ofs, "# device memory pitch: %u\n", p.memPitch);
  fprintf(ofs, "# device max. no. of threads per block: %d\n", p.maxThreadsPerBlock);
  fprintf(ofs, "# device max. no. of thread dimensions: (%d, %d, %d)\n", 
      p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
  fprintf(ofs, "# device maximal grid size: (%d, %d, %d)\n\n", 
      p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
  fflush(ofs);

}
