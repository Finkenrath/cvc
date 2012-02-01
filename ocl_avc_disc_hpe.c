/****************************************************
 * ocl_avc_disc_hpe.c
 *
 * Sun Nov  7 20:14:46 CET 2010
 *
 * TODO: 
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <CL/opencl.h>

#ifndef EXIT_FAILURE
#  define EXIT_FAILURE 1
#endif
#ifndef EXIT_SUCCESS
#  define EXIT_SUCCESS 1
#endif

#define MAIN_PROGRAM
#include "global.h"

cl_context       cxGPUContext;
cl_command_queue cqCommandQueue;
cl_mem           cdSpinorfield, cdSpinorfield2, cdSpinorfield3, cdGaugefield;
cl_program       cpProgram;
cl_kernel        ckKernel;
char *cPathAndName=NULL, *cSourceC=NULL;

inline void checkErr(cl_int err, const char * name);
inline void cleanup (int icode);

void usage() {
  fprintf(stdout, "Code to perform quark-disc. contractions for conserved vector current\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {

  int c, verbose=0, filename_set=0;
  time_t g_the_time = time(NULL);

  char chBuffer[1024], filename[1024];
  cl_uint num_platforms, i;
  cl_platform_id *clPlatformIDs = NULL;
  cl_platform_id clSelectedPlatform = NULL, cpPlatform;
  cl_int ciErrNum, ciErr1, ciErr2;
  cl_uint ciDeviceCount;
  cl_device_id *devices = NULL, cdDevice;
  size_t szGlobalWorkSize, szLocalWorkSize, szParmDataBytes, szKernelLength;

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
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

  ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);
  checkErr(ciErrNum, "clGetPlatformIDs");
  if(num_platforms == 0) {
    fprintf(stderr, "[ocl_avc_disc_hpe] Error, number of platforms 0\n");
    exit(1);
  } else {
    fprintf(stdout, "# [ocl_avc_disc_hpe] number of platforms  = %u\n", num_platforms);
  }

  if( (clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id)) )==NULL ) {
    fprintf(stderr, "[ocl_avc_disc_hpe] Error from malloc\n");
    exit(2);
  }
  ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
  checkErr(ciErrNum, "clGetPlatformIDs");
  for(i=0; i<num_platforms; i++) {
    ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, sizeof(chBuffer), chBuffer, NULL);
    if(ciErrNum == CL_SUCCESS) {
      if(strstr(chBuffer, "NVIDIA")!=NULL) {
        clSelectedPlatform = clPlatformIDs[i];
        fprintf(stdout, "# [ocl_avc_disc_hpe] Found NVIDIA OpenCL platform; CL_PLATFORM_NAME = \"%s\"\n", chBuffer);
        break;
      }
    }
  }
  if(clSelectedPlatform == NULL) {
    fprintf(stderr, "[ocl_avc_disc_hpe] WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n");
    clSelectedPlatform = clPlatformIDs[0];
  }
  free(clPlatformIDs);

  ciErrNum = clGetPlatformInfo(clSelectedPlatform, CL_PLATFORM_VERSION, sizeof(chBuffer), chBuffer, NULL);
  checkErr(ciErrNum, "clGetPlatformInfo");
  fprintf(stdout, "# [ocl_avc_disc_hpe] CL_PLATFORM_VERSION= \"%s\"\n", chBuffer);

  ciErrNum = clGetDeviceIDs (clSelectedPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);
  checkErr(ciErrNum, "clGetDeviceIDs");
  if(ciDeviceCount==0) {
    fprintf(stderr, "[ocl_avc_disc_hpe] Error, no devices\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "# [ocl_avc_disc_hpe] found %u devices supporting OpenCL:\n\n", ciDeviceCount);
  if( (devices = (cl_device_id*)malloc(ciDeviceCount * sizeof(cl_device_id))) == NULL ) {
    fprintf(stderr, "[ocl_avc_disc_hpe] Error from malloc\n");
    exit(EXIT_FAILURE);
  }
  ciErrNum = clGetDeviceIDs(clSelectedPlatform, CL_DEVICE_TYPE_ALL, ciDeviceCount, devices, &ciDeviceCount);
  checkErr(ciErrNum, "clGetDeviceIDs");
  cxGPUContext = clCreateContext(0, ciDeviceCount, devices, NULL, NULL, &ciErrNum);
  checkErr(ciErrNum, "clCreateContext");
  for(i=0; i<ciDeviceCount; i++) {
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(chBuffer), &chBuffer, NULL);
    fprintf(stdout, "# [ocl_avc_disc_hpe] Device no. %u: %s\n", i, chBuffer);
    clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(chBuffer), &chBuffer, NULL);
    fprintf(stdout, "# [ocl_avc_disc_hpe] \tdevice vendor: %s\n", chBuffer);
    clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(chBuffer), &chBuffer, NULL);
    fprintf(stdout, "# [ocl_avc_disc_hpe] \tdriver version: %s\n", chBuffer);
    clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(chBuffer), &chBuffer, NULL);
    fprintf(stdout, "# [ocl_avc_disc_hpe] \tdevice version: %s\n", chBuffer);
    if(strncmp("OpenCL 1.0", chBuffer, 10)!=0) {
      clGetDeviceInfo(devices[i], CL_DEVICE_OPENCL_C_VERSION, sizeof(chBuffer), &chBuffer, NULL);
      fprintf(stdout, "# [ocl_avc_disc_hpe] \tdevice OpenCL version: %s\n", chBuffer);
    }
  }
  free(devices);
  clReleaseContext(cxGPUContext);


  cpPlatform = clSelectedPlatform;
  cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
  if(ciErr1 != CL_SUCCESS) {
    fprintf(stderr, "[ocl_avc_disc_hpe] Error, could not create context\n");
    cleanup(10);
  }
  
  cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
  checkErr(ciErr1, "clCreateCommandQueue");

  szLocalWorkSize = 256;
  szGlobalWorkSize = T*L*L*L;

  cdSpinorfield  = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float)*szGlobalWorkSize*24, NULL, &ciErr1);
  cdSpinorfield2 = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float)*szGlobalWorkSize*24, NULL, &ciErr2);
  ciErr1 |= ciErr2;
  cdSpinorfield3 = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float)*szGlobalWorkSize*24, NULL, &ciErr2);
  ciErr1 |= ciErr2;
  cdGaugefield = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float)*szGlobalWorkSize*72, NULL, &ciErr2);
  ciErr1 |= ciErr2;
  checkErr(ciErr1, "CreateBuffer");



  fprintf(stderr, "[ocl_avc_disc_hpe] end of run\t%s\n", ctime(&g_the_time));
  cleanup(EXIT_SUCCESS);
  return(0);

}


inline void checkErr(cl_int err, const char * name) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "[ocl_avc_disc_hpe] ERROR: %s\n", name);
    cleanup(EXIT_FAILURE);
  }
}

inline void cleanup (int icode) {
  if(cPathAndName) free(cPathAndName);
  if(cSourceCL) free(cSourceCL);
  if(ckKernel) clReleaseKernel(ckKernel);
  if(cpProgram) clReleaseProgram(cpProgram);
  if(cqCommandQueue) clReleaseCommandQueue(cqCommandQueue);
  if(cxGPUContext) clReleaseContext(cxGPUContext);
  if(cdSpinorfield) clReleaseMemObject(cdSpinorfield);
  if(cdSpinorfield2) clReleaseMemObject(cdSpinorfield2);
  if(cdGaugefield) clReleaseMemObject(cdGaugefield);
  if(g_spinor_field) {
    if(g_spinor_field[0]) free(g_spinor_field[0]);
    free(g_spinor_field);
  }
  if(g_gauge_field) free(g_gauge_field);
  exit(icode); 
}

