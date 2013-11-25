/************************************************************************
 * oclAvcDisc.c
 *
 * Sun Nov  7 18:26:07 CET 2010 
 ************************************************************************/

/* Standard utilities and common systems includes */
#include <oclUtils.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/*
#define _write_log(_text_ ) {\
  if(log_ofs != NULL) {\
    fprintf(log_ofs, "%s%s\n", my_name, _text_); \
  } else { \
    fprintf(stderr, "%sError, no log file opened; message was:\n", my_name, _text_); \
    fprintf(stderr, "%s\t%s\n", my_name, _text_); \
  }\
*/
#define _write_log fprintf

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv) {

  const char *my_name = "[oclAvcDisc]";

  int bPassed = 1;
  char filename[500], cBuffer[1024], sProfileString[2048];
  FILE *log_ofs=NULL;
  time_t g_the_time;

  /* OpenCL variables */
  cl_int ciErrNum;
  cl_platform_id clSelectedPlatformID = NULL; 
  cl_uint ciDeviceCount;
  cl_device_id *devices;

  sprintf(filename, "oclAvcDisc.txt");
  if( (log_ofs=fopen(filename, "a"))== NULL )  {
    fprintf(stderr, "[oclAvcDisc] Error, could not open file %s\n", filename);
    exit(1);
  }

  g_the_time = time(NULL);

  _write_log(log_ofs, "%s oclDeviceQuery.exe Starting...\n", my_name); 

  /* Get OpenCL platform ID for NVIDIA if avaiable, otherwise default */
  _write_log(log_ofs, "%s OpenCL SW Info:\n", my_name);
  ciErrNum = oclGetPlatformID (&clSelectedPlatformID);
  oclCheckError(ciErrNum, CL_SUCCESS);

  /* Get OpenCL platform name and version */
  ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
  if (ciErrNum == CL_SUCCESS) {
    _write_log(log_ofs, "%s CL_PLATFORM_NAME: \t%s\n", my_name, cBuffer);
  } else {
    _write_log(log_ofs, "%s Error %i in clGetPlatformInfo Call !!!\n\n", my_name, ciErrNum);
    bPassed = 0;
  }

  ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
  if (ciErrNum == CL_SUCCESS) {
    _write_log(log_ofs, "%s CL_PLATFORM_VERSION: \t%s\n", my_name, cBuffer);
  } else {
    _write_log(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
    bPassed = 0;
  }

  // Get and log OpenCL device info 
  _write_log(log_ofs, "%s OpenCL Device Info:\n\n", my_name);
  ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);

  // check for 0 devices found or errors... 
  if (ciDeviceCount == 0) {
    _write_log(log_ofs, "%s No devices found supporting OpenCL (return code %i)\n\n", my_name, ciErrNum);
    bPassed = false;
  } else if (ciErrNum != CL_SUCCESS) {
    _write_log(log_ofs, "%s Error %i in clGetDeviceIDs call !!!\n\n", my_name, ciErrNum);
    bPassed = false;
  } else {
    // Get and log the OpenCL device ID's
     char cTemp[2];
    _write_log(log_ofs, "%s %u devices found supporting OpenCL:\n\n", my_name , ciDeviceCount);
    sprintf(cTemp, "%u", ciDeviceCount);
    if ((devices = (cl_device_id*)malloc(sizeof(cl_device_id) * ciDeviceCount)) == NULL) {
      _write_log(log_ofs, "%s Failed to allocate memory for devices !!!\n\n", my_name);
      bPassed = false;
    }
    ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, ciDeviceCount, devices, &ciDeviceCount);
    if (ciErrNum == CL_SUCCESS) {
      //Create a context for the devices
      cl_context cxGPUContext = clCreateContext(0, ciDeviceCount, devices, NULL, NULL, &ciErrNum);
      if (ciErrNum != CL_SUCCESS) {
        _write_log(log_ofs, "%s Error %i in clCreateContext call !!!\n\n", my_name, ciErrNum);
        bPassed = false;
      } else {
        // show info for each device in the context
        for(unsigned int i = 0; i < ciDeviceCount; ++i ) {  
          _write_log(log_ofs, "%s ---------------------------------\n", my_name);
          clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
          _write_log(log_ofs, "%s Device %s\n", my_name, cBuffer);
          _write_log(log_ofs, "%s ---------------------------------\n", my_name);
          oclPrintDevInfo(LOGBOTH, devices[i]);
        }
        // Determine and show image format support 
        cl_uint uiNumSupportedFormats = 0;

        // 2D
        clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, NULL, NULL, &uiNumSupportedFormats);
        cl_image_format *ImageFormats = NULL;
        ImageFormats = (cl_image_format*)malloc(uiNumSupportedFormats*sizeof(cl_image_format));
        if(ImageFormats==NULL) {
          _write_log(log_ofs, "%s Error, could not alloc ImageFormats\n", my_name);
          exit(2);
        }

        clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, uiNumSupportedFormats, ImageFormats, NULL);
        _write_log(log_ofs, "%s  ---------------------------------\n", my_name);
        _write_log(log_ofs, "%s  2D Image Formats Supported (%u)\n", my_name, uiNumSupportedFormats); 
        _write_log(log_ofs, "%s  ---------------------------------\n", my_name);
        _write_log(log_ofs, "%s  %-6s%-16s%-22s\n\n", my_name, "#", "Channel Order", "Channel Type"); 
        for(unsigned int i = 0; i < uiNumSupportedFormats; i++) {  
           _write_log(log_ofs, "%s  %-6u%-16s%-22s\n", my_name, (i + 1), 
               oclImageFormatString(ImageFormats[i].image_channel_order), 
               oclImageFormatString(ImageFormats[i].image_channel_data_type));
        }
        _write_log(log_ofs, "%s\n", my_name); 
        free(ImageFormats); ImageFormats = NULL;

        // 3D
        clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE3D, NULL, NULL, &uiNumSupportedFormats);
        ImageFormats = (cl_image_format*)malloc(uiNumSupportedFormats*sizeof(cl_image_format));
        if(ImageFormats==NULL) {
          _write_log(log_ofs, "%s Error, could not alloc ImageFormats\n", my_name);
          exit(3);
        }
        clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE3D, uiNumSupportedFormats, ImageFormats, NULL);
        _write_log(log_ofs, "%s  ---------------------------------\n", my_name);
        _write_log(log_ofs, "%s  3D Image Formats Supported (%u)\n", my_name, uiNumSupportedFormats); 
        _write_log(log_ofs, "%s  ---------------------------------\n", my_name);
        _write_log(log_ofs, "%s  %-6s%-16s%-22s\n\n", my_name, "#", "Channel Order", "Channel Type"); 
        for(unsigned int i = 0; i < uiNumSupportedFormats; i++) {  
          _write_log(log_ofs, "%s  %-6u%-16s%-22s\n", my_name, (i + 1),
              oclImageFormatString(ImageFormats[i].image_channel_order), 
              oclImageFormatString(ImageFormats[i].image_channel_data_type));
        }
        write_log(log_ofs, "%s\n", my_name); 
        free(ImageFormats); ImageFormats=NULL;
      }
    } else {
      write_log(log_ofs, "%s Error %i in clGetDeviceIDs call !!!\n\n", my_name, ciErrNum);
      bPassed = 0;
    }
  }

  // finish
  _write_log(log_ofs, "%s %s\n\n", my_name, bPassed==1 ? "PASSED" : "FAILED"); 
  fflush(log_ofs);
  fclose(log_ofs);
  return(0);
}
