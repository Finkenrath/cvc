/*********************************************************************************
 * jc_ud_p2.cu
 *
 * Sat Sep 25 16:21:00 CEST 2010
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
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "contractions_io.h"
#include "read_input_parser.h"
}

#define THREADS_PER_BLOCK 256

/**********************************************
 * reduce a float2 array of length n to one with length 
 *   equal to the number of blocks at launch time 
 *   by blockwise summation
 * - copied from reduce2 in SDK/C/src/reduction/reduction_kernel.cu
 **********************************************/
__global__ void reduce(float2*g_idata, float2*g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ float2 sdata[];

  sdata[tid].x = (i < n) ? g_idata[i].x : 0.;
  sdata[tid].y = (i < n) ? g_idata[i].y : 0.;

  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid].x += sdata[tid + s].x;
      sdata[tid].y += sdata[tid + s].y;
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x].x = sdata[0].x;
    g_odata[blockIdx.x].y = sdata[0].y;
  }
}
/**********************************************
 * calculate correlation for one lattice site
 * - some kernels will calculate zero
 **********************************************/
__global__ void build_correlator(float2*j_source, float2*j_sink, float2*corr, unsigned int T, unsigned int L, unsigned int Tsub, unsigned int Lsub, unsigned int shift_id, unsigned int mu, int ineg) {

  unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int thread_coords[4], shift_coords[4];
  __shared__ unsigned int sink_id;
  __shared__ int x_start[2];
  __shared__ unsigned int L1, L2, L3, Volume;
  __shared__ unsigned int Lsub1, Lsub2, Lsub3;
  __shared__ unsigned int count, ix, iy;
  __shared__ int x0_b, x0_e, x1_b, x1_e, x2_b, x2_e, x3_b, x3_e;
  __shared__ float2 j1x, j2y;

  L1 = L;
  L2 = L1*L1;
  L3 = L2*L1;
  Volume=L3*T;
  x_start[0] = (ineg==0) ? 0 : -Tsub+1;
  x_start[1] = (ineg==0) ? 0 : -Lsub+1;
  
  Lsub1 = Lsub - x_start[1];
  Lsub2 = Lsub1*Lsub1;
  Lsub3 = Lsub2*Lsub1;

  count = thread_id;
  thread_coords[0] = count / L3;
  count -= thread_coords[0]*L3;
  thread_coords[1] = count / L2;
  count -= thread_coords[1]*L2;
  thread_coords[2] = count / L;
  count -= thread_coords[2]*L;
  thread_coords[3] = count;

  count = shift_id;
  shift_coords[0] = count / Lsub3;
  count -= shift_coords[0]*Lsub3;
  shift_coords[1] = count / Lsub2;
  count -= shift_coords[1]*Lsub2;
  shift_coords[2] = count / Lsub1;
  count -= shift_coords[2]*Lsub1;
  shift_coords[3] = count;

  shift_coords[0] += x_start[0];
  shift_coords[1] += x_start[1];
  shift_coords[2] += x_start[1];
  shift_coords[3] += x_start[1];

  x0_b = (shift_coords[0]>=0) ? shift_coords[0] : 0;
  x0_e = T + shift_coords[0] - x0_b;

  x1_b = (shift_coords[1]>=0) ? shift_coords[1] : 0;
  x1_e = L + shift_coords[1] - x1_b;

  x2_b = (shift_coords[2]>=0) ? shift_coords[2] : 0;
  x2_e = L + shift_coords[2] - x2_b;

  x3_b = (shift_coords[3]>=0) ? shift_coords[3] : 0;
  x3_e = L + shift_coords[3] - x3_b;

  sink_id = (thread_coords[0] >= x0_b && thread_coords[0] < x0_e && thread_coords[1] >= x1_b && thread_coords[1] < x1_e && thread_coords[2] >= x2_b && thread_coords[2] < x2_e && thread_coords[3] >= x3_b && thread_coords[3] < x3_e) ? (((thread_coords[0]-shift_coords[0])*L1+ (thread_coords[1]-shift_coords[1]))*L1+ (thread_coords[2]-shift_coords[2]))*L1+ (thread_coords[3]-shift_coords[3]): Volume;

  ix = mu*(Volume+1) + thread_id;
  iy = mu*(Volume+1) + sink_id;
  j1x.x = j_source[ix].x; j1x.y = j_source[ix].y;
  j2y.x = j_sink[iy].x;   j2y.y = j_sink[iy].y;

  corr[thread_id].x = j1x.x * j2y.x - j1x.y * j2y.y;
  corr[thread_id].y = j1x.x * j2y.y + j1x.y * j2y.x;
}
/**********************************************
 * initialize a float2 vector of length V to 0.
 **********************************************/
__global__ void init_to_zero(float2*corr, unsigned int V) {

  unsigned int id_thread = blockIdx.x * blockDim.x + threadIdx.x;

  if (id_thread < V) {
    corr[id_thread].x = 0.;
    corr[id_thread].y = 0.;
  }
}

void usage() {
  fprintf(stdout, "Code to calculate correlation of quark-disconnected conserved vector current contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f <string> input filename [default cvc.input]\n");
  fprintf(stdout, "         -l <uint> spatial size of sublattice [default 2]\n");
  fprintf(stdout, "         -t <uint> temporal size of sublattice [default 2]\n");
  fprintf(stdout, "         -m allow negative entries in the shift vector [default no/0]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
 
  int Tsub = 2; 
  int Lsub = 2; 
  int c, i;
  unsigned int count, count2;
  int filename_set = 0;
  int ip;
  int it, ix, iy, iz, iix;
  int t_start, x_start, y_start, z_start;
  int sid1, sid2, gid;
  int include_negative=0;
  size_t nprop=0;
  float *h_sdata=NULL, *h_swork[48];
  //float *h_swork2=NULL;
  double *h_dwork=NULL, *h_ddata=NULL;
  float fnorm, r2;
  char filename[100];
  double ratime, retime;
  FILE *ofs=NULL;
  time_t the_time;

  unsigned int threadsPerBlock, blocksPerGrid;
  float2 *d_work1=NULL, *d_work2=NULL, *d_work3=NULL, *d_data=NULL;
  float2 *d_block_sum=NULL, *d_block_sum2=NULL, *d_block_sum_ptr1=NULL, *d_block_sum_ptr2=NULL, *d_block_sum_ptr3=NULL;
  unsigned int V4, mu;

  /****************************************
   * initialize the distance vectors
   ****************************************/

  while ((c = getopt(argc, argv, "h?f:l:t:m")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'l':
      Lsub = atoi(optarg);
      fprintf(stdout, "# using Lsub = %d\n", Lsub);
      break;
    case 't':
      Tsub = atoi(optarg);
      fprintf(stdout, "# using Tsub = %d\n", Tsub);
      break;
    case 'm':
      include_negative = 1;
      fprintf(stdout, "# will do negative R_i, too\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* get the time stamp */
  the_time = time(NULL);

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

  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stderr, "Error, kappa should be > 0.n");
    usage();
  }

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* jc_ud_p2\n* %s", ctime(&the_time));
  fprintf(stdout, "**************************************************\n\n");

  /* initialize fftw */
  T            = T_global;
  L            = LX;
  Tstart       = 0;
  if(!include_negative) {
    FFTW_LOC_VOLUME = Tsub * Lsub*Lsub*Lsub;
  } else {
    FFTW_LOC_VOLUME = (2*Tsub-1) * (2*Lsub-1) * (2*Lsub-1) * (2*Lsub-1);
  }
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "#       T            = %3d\n"\
		  "#       Tstart       = %3d\n"\
		  "#       FFTW_LOC_VOLUME = %8d\n",
		  g_cart_id, T, Tstart, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  V4 = (unsigned int)VOLUME;

  nprop = (size_t)(g_sourceid2 - g_sourceid) / (size_t)g_sourceid_step + 1;
  fprintf(stdout, "# number of stoch. propagators = %lu\n", nprop);

  /***********************************************
   * set number of threads and blocks
   ***********************************************/
  threadsPerBlock        = THREADS_PER_BLOCK;
  blocksPerGrid          = (V4+threadsPerBlock-1)/threadsPerBlock;
  fprintf(stdout, "# number threads per block: %u\n", threadsPerBlock);
  fprintf(stdout, "# number blocks per grid  : %u\n", blocksPerGrid);
 
  /****************************************
   * allocate host fields
   ****************************************/
  h_sdata = (float*)calloc(8*FFTW_LOC_VOLUME, sizeof(float));
  if( h_sdata==NULL ) { 
    fprintf(stderr, "could not allocate memory for h_sdata\n");
    exit(3);
  }

  h_ddata = (double*)calloc(8*FFTW_LOC_VOLUME, sizeof(double));
  if( h_ddata==NULL ) { 
    fprintf(stderr, "could not allocate memory for h_ddata\n");
    exit(3);
  }

  h_swork[0] = (float*)calloc(nprop*8*(VOLUME+1), sizeof(float));
  if( h_swork[0] == NULL ) { 
    fprintf(stderr, "could not allocate memory for h_swork field\n");
    exit(5);
  }
  for(i=1; i< nprop; i++) {
    h_swork[i] = h_swork[i-1] + 8*(VOLUME+1);
  }

  h_dwork = (double*)calloc(8*VOLUME, sizeof(double));
  if( h_dwork == NULL ) { 
    fprintf(stderr, "could not allocate memory for h_dwork\n");
    exit(15);
  }

  //h_swork2 = (float*)calloc(2*VOLUME, sizeof(float));
  //if( h_swork2 == NULL ) { 
  //  fprintf(stderr, "could not allocate memory for h_swork2 field\n");
  //  exit(5);
  //}
  /***********************************************
   * allocate GPU fields 
   ***********************************************/
  cudaMalloc(&d_work1, 4*(VOLUME+1)*sizeof(float2));
  cudaMalloc(&d_work2, 4*(VOLUME+1)*sizeof(float2));
  cudaMalloc(&d_work3, VOLUME*sizeof(float2));
  cudaMalloc(&d_data, 4*FFTW_LOC_VOLUME*sizeof(float2));
  cudaMalloc(&d_block_sum, blocksPerGrid*sizeof(float2));
  cudaMalloc(&d_block_sum2, blocksPerGrid*sizeof(float2));

  init_to_zero<<<4*blocksPerGrid, threadsPerBlock>>>(d_work1, 4*V4);
  init_to_zero<<<4*blocksPerGrid, threadsPerBlock>>>(d_work1, 4*V4);
  init_to_zero<<<blocksPerGrid, threadsPerBlock>>>(d_work3, V4);
  init_to_zero<<<(blocksPerGrid+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_block_sum, blocksPerGrid);
  init_to_zero<<<(blocksPerGrid+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_block_sum2, blocksPerGrid);
  init_to_zero<<<(4*FFTW_LOC_VOLUME+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_data, 4*FFTW_LOC_VOLUME);

  fnorm = 1. / ( (float)nprop * (float)(nprop-1));
  fprintf(stdout, "# fnorm = %16.7e\n", fnorm);

  /***********************************************
   * choose the start values for the entries
   *   of the R-vector
   ***********************************************/
  if(include_negative) {
    t_start = -Tsub+1;
    x_start = -Lsub+1;
    y_start = -Lsub+1;
    z_start = -Lsub+1;
  } else {
    t_start = 0;
    x_start = 0;
    y_start = 0;
    z_start = 0;
  }
  fprintf(stdout, "#\n# t_start=%d, x_start=%d, y_start=%d, z_start=%d\n", 
    t_start, x_start, y_start, z_start);

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    for(ix=0; ix<8*FFTW_LOC_VOLUME; ix++) h_ddata[ix] = 0.;

    /************************************************
     * read the contracted currents
     ************************************************/
    ratime = clock() / CLOCKS_PER_SEC;
    for(sid1=0; sid1<nprop; sid1++) {
      sprintf(filename, "jc_ud_x.%.4d.%.4d", gid, g_sourceid + sid1*g_sourceid_step);
      if(read_lime_contraction(h_dwork, filename, 4, 0) != 0) {
        fprintf(stderr, "Error, could not read field no. %d\n", sid1);
        exit(15);
      }
      count=0; iix=0;
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          h_swork[sid1][iix  ] = (float)(h_dwork[count  ]);
          h_swork[sid1][iix+1] = (float)(h_dwork[count+1]);
          count+=2; iix+=2;
        }
        h_swork[sid1][iix  ] = 0.;
        h_swork[sid1][iix+1] = 0.;
        iix+=2;
      }
      //sprintf(filename, "jc_ud_x.%.4d.%.4d.ascii", gid, g_sourceid+sid1*g_sourceid_step);
      //ofs = fopen(filename, "w");
      //for(i=0; i<4*V4; i++)
      //  fprintf(ofs, "%25.16e%25.16e\n", h_dwork[2*i], h_dwork[2*i+1]);
      //fclose(ofs);
    }
    retime = clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time for reading fields: %e seconds\n", retime-ratime);
    /***********************************************
     * start (double) loop on source id pairs
     ***********************************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(sid1=0; sid1<nprop-1; sid1++) {
      cudaMemcpy(d_work1, h_swork[sid1], (VOLUME+1)*8*sizeof(float), cudaMemcpyHostToDevice);
    for(sid2=sid1+1; sid2<nprop; sid2++) {
      cudaMemcpy(d_work2, h_swork[sid2], (VOLUME+1)*8*sizeof(float), cudaMemcpyHostToDevice);

      fprintf(stdout, "# processing sid pair (%3d,%3d)\n", sid1, sid2);

      for(mu=0; mu<4; mu++) {
        for(ip=0; ip<FFTW_LOC_VOLUME; ip++) {

          build_correlator<<<blocksPerGrid, threadsPerBlock>>>(d_work1, d_work2, d_work3, T, L, Tsub, Lsub, ip, mu, include_negative);
          //cudaMemcpy(h_swork2, d_work3, 2*V4*sizeof(float), cudaMemcpyDeviceToHost);
          //sprintf(filename, "j1xj2_2.%.2d.%.6d", mu, ip);
          //ofs = fopen(filename, "w");
          //for(i=0; i<V4; i++) {
          //  fprintf(ofs, "%18.9e%18.9e\n", h_swork2[2*i], h_swork2[2*i+1]);
          //}
          //fclose(ofs);

          reduce<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float2)>>>(d_work3, d_block_sum, V4);

          /*******************************************************************
           * stepwise reduction of d_block_sum
           *******************************************************************/
          d_block_sum_ptr1 = d_block_sum2;
          d_block_sum_ptr2 = d_block_sum;
          for(count=blocksPerGrid;;) {
            d_block_sum_ptr3 = d_block_sum_ptr1;
            d_block_sum_ptr1 = d_block_sum_ptr2;
            d_block_sum_ptr2 = d_block_sum_ptr3;
            count2 = (count+threadsPerBlock-1)/threadsPerBlock;
            if(count2==1) break;
            reduce<<<count2, threadsPerBlock, threadsPerBlock*sizeof(float2)>>>(d_block_sum_ptr1, d_block_sum_ptr2, count);
            count = count2;
          }
          d_block_sum_ptr2 = &(d_data[mu*FFTW_LOC_VOLUME+ip]);
          reduce<<<1, threadsPerBlock, threadsPerBlock*sizeof(float2)>>>(d_block_sum_ptr1, d_block_sum_ptr2, count);
        }
      }
      cudaMemcpy(h_sdata, d_data, 8*FFTW_LOC_VOLUME*sizeof(float), cudaMemcpyDeviceToHost);
      for(ip=0; ip<8*FFTW_LOC_VOLUME; ip++) h_ddata[ip] += (double)h_sdata[ip];
    }  /* of loop on sid2 */
    }  /* of loop on sid1 */

    //for(ix=0; ix<8*FFTW_LOC_VOLUME; ix++) h_ddata[ix] *= fnorm;
    retime = (double)clock() / CLOCKS_PER_SEC;
    if(g_cart_id == 0) fprintf(stdout, "# time for building correl.: %e seconds\n", retime-ratime);

    /************************************************
     * save results in position space
     ************************************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    sprintf(filename, "pi_ud_r2.%4d", gid);
    ofs = fopen(filename, "w");
    if (ofs==NULL) {
     fprintf(stderr, "Error, could not open file %s for writing\n", filename);
     exit(9);
    }
    iix = 0;
    for(mu=0; mu<4; mu++) {
      ip = 0;
      for(it=t_start; it<Tsub; it++) {
        for(ix=x_start; ix<Lsub; ix++) {
        for(iy=y_start; iy<Lsub; iy++) {
        for(iz=z_start; iz<Lsub; iz++) {
          r2 = (double)(ix*ix) + (double)(iy*iy) + (double)(iz*iz);
          fprintf(ofs, "%3d%3d%3d%3d%3d%16.7e%25.16e%25.16e\n", mu, it, ix, iy, iz, r2, h_ddata[iix], h_ddata[iix+1]);
          ip++; iix+=2;
        }}}
      }
    }
    fclose(ofs);
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time for writing correl. for file: %e seconds\n", retime-ratime);
  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  cudaFree(d_work1);
  cudaFree(d_work2);
  cudaFree(d_work3);
  cudaFree(d_data);
  cudaFree(d_block_sum);
  cudaFree(d_block_sum2);

  free_geometry();
  if(h_dwork     !=NULL) free(h_dwork);
  if(h_swork[0]  !=NULL) free(h_swork[0]);
  if(h_ddata     !=NULL) free(h_ddata);
  if(h_sdata     !=NULL) free(h_sdata);
  //if(h_swork2    !=NULL) free(h_swork2);
  return(0);

}
