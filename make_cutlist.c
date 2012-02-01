/****************************************************
 * make_cutlist.c
 *
 * Thu Oct  8 20:04:05 CEST 2009
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 ****************************************************/

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
#include "Q_phi.h"
#include "make_cutlist.h"


int make_cutid_list(int *idlist, int *cutdir, double rad, double angle) {
/* the idlist has to be initialized externally */

  int ix, x0, x1, x2, x3;
  int y0, y1, y2, y3;
  double odist, norminv, rad2;

  if( (rad >= 0.) && (angle >= 0.) ) {
    fprintf(stdout, "Attention: Applying cylinder _AND_ cone cut\n");
  }
  if( (rad < 0.) && (angle < 0.) ) {
    fprintf(stdout, "Attention: No cut specified\n");
    return(0);
  }
  

  if(rad>=0.) {
    norminv = 1. / (cutdir[0]*cutdir[0] + cutdir[1]*cutdir[1] + cutdir[2]*cutdir[2] + 
                    cutdir[3]*cutdir[3]);
    rad2 = rad * rad;
    for(x0=0; x0<T;  x0++) {
      if(x0>T/2) { y0 = x0-T; }
      else       { y0 = x0; }
    for(x1=0; x1<LX; x1++) {
      if(x1>LX/2) { y1 = x1-LX; }
      else        { y1 = x1; }
    for(x2=0; x2<LY; x2++) {
      if(x2>LY/2) { y2 = x2-LY; }
      else        { y2 = x2; }
    for(x3=0; x3<LZ; x3++) {
      if(x3>LZ/2) { y3 = x3-LZ; }
      else        { y3 = x3; }
      ix = g_ipt[x0][x1][x2][x3];
      odist = y0*y0 + y1*y1 + y2*y2 + y3*y3 - 
        _sqr(y0*cutdir[0] + y1*cutdir[1] + y2*cutdir[2] + y3*cutdir[3]) * norminv;
      if(odist > rad2 + _Q2EPS) idlist[ix] = -1;
/*    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d, odist=%e, idlist=%4d\n", x0, x1, x2, x3, odist, idlist[ix]); */
    }
    }
    }
    }
  }
  
  if(angle>=0.) {
    norminv = 1. / sqrt(cutdir[0]*cutdir[0] + cutdir[1]*cutdir[1] + cutdir[2]*cutdir[2] + 
                        cutdir[3]*cutdir[3]);
    angle = cos(angle);
    for(x0=0; x0<T;  x0++) {
      if(x0>T/2) { y0 = x0-T; }
      else       { y0 = x0; }
    for(x1=0; x1<LX; x1++) {
      if(x1>LX/2) { y1 = x1-LX; }
      else        { y1 = x1; }
    for(x2=0; x2<LY; x2++) {
      if(x2>LY/2) { y2 = x2-LY; }
      else        { y2 = x2; }
    for(x3=0; x3<LZ; x3++) {
      if(x3>LZ/2) { y3 = x3-LZ; }
      else        { y3 = x3; }
      ix = g_ipt[x0][x1][x2][x3];
      odist = (y0*cutdir[0] + y1*cutdir[1] + y2*cutdir[2] + y3*cutdir[3]) / 
        sqrt(y0*y0 + y1*y1 + y2*y2 + y3*y3) * norminv;
      if(odist < angle+_Q2EPS) idlist[ix] = -1;
    }
    }
    }
    }
  }
 
  /*********************************
   * test: printf the list
   *********************************/
/*
  fprintf(stdout, "idlist after the cut with r=%12.5e, alpha=%12.5e:\n", rad, angle);
  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d\tid=%4d\n", x0, x1, x2, x3, idlist[ix]);
  } 
  } 
  } 
  } 
*/
  return(0);
}

/***********************************************************************/

int make_cutid_list2(int *idlist, int *cutdir, double rad, double angle) {
/* the idlist has to be initialized externally */

  int ix, x0, x1, x2, x3;
  double y0, y1, y2, y3;
  double odist, norminv, rad2;

  if( (rad >= 0.) && (angle >= 0.) ) {
    fprintf(stdout, "Attention: Applying cylinder _AND_ cone cut\n");
  }
  if( (rad < 0.) && (angle < 0.) ) {
    fprintf(stdout, "Attention: No cut specified\n");
    return(0);
  }
  

  if(rad>=0.) {
    norminv = 1. / (cutdir[0]*cutdir[0] + cutdir[1]*cutdir[1] + cutdir[2]*cutdir[2] + 
                    cutdir[3]*cutdir[3]);
    rad2 = rad * rad;
    for(x0=0; x0<T;  x0++) {
      if(x0>T/2) { y0 = (double)(x0-T) / (double)T; }
      else       { y0 = (double)x0 / (double)T; }
    for(x1=0; x1<LX; x1++) {
      if(x1>LX/2) { y1 = (double)(x1-LX) / (double)LX; }
      else        { y1 = (double)x1 / (double)LX; }
    for(x2=0; x2<LY; x2++) {
      if(x2>LY/2) { y2 = (double)(x2-LY) / (double)LY; }
      else        { y2 = (double)x2 / (double)LY; }
    for(x3=0; x3<LZ; x3++) {
      if(x3>LZ/2) { y3 = (double)(x3-LZ) / (double)LZ; }
      else        { y3 = (double)x3 / (double)LZ; }
      ix = g_ipt[x0][x1][x2][x3];
      odist = y0*y0 + y1*y1 + y2*y2 + y3*y3 - 
        _sqr(y0*cutdir[0] + y1*cutdir[1] + y2*cutdir[2] + y3*cutdir[3]) * norminv;
      if(odist > rad2 + _Q2EPS) idlist[ix] = -1;
/*    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d, odist=%e, idlist=%4d\n", x0, x1, x2, x3, odist, idlist[ix]); */
    }}}}
  }
  
  if(angle>=0.) {
    norminv = 1. / sqrt(cutdir[0]*cutdir[0] + cutdir[1]*cutdir[1] + cutdir[2]*cutdir[2] + 
                        cutdir[3]*cutdir[3]);
    angle = cos(angle);
    for(x0=0; x0<T;  x0++) {
      if(x0>T/2) { y0 = x0-T; }
      else       { y0 = x0; }
    for(x1=0; x1<LX; x1++) {
      if(x1>LX/2) { y1 = x1-LX; }
      else        { y1 = x1; }
    for(x2=0; x2<LY; x2++) {
      if(x2>LY/2) { y2 = x2-LY; }
      else        { y2 = x2; }
    for(x3=0; x3<LZ; x3++) {
      if(x3>LZ/2) { y3 = x3-LZ; }
      else        { y3 = x3; }
      ix = g_ipt[x0][x1][x2][x3];
      odist = (y0*cutdir[0] + y1*cutdir[1] + y2*cutdir[2] + y3*cutdir[3]) / 
        sqrt(y0*y0 + y1*y1 + y2*y2 + y3*y3) * norminv;
      if(odist < angle+_Q2EPS) idlist[ix] = -1;
    }
    }
    }
    }
  }
 
  /*********************************
   * test: printf the list
   *********************************/
/*
  fprintf(stdout, "idlist after the cut with r=%12.5e, alpha=%12.5e:\n", rad, angle);
  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(stdout, "t=%3d, x=%3d, y=%3d, z=%3d\tid=%4d\n", x0, x1, x2, x3, idlist[ix]);
  } 
  } 
  } 
  } 
*/
  return(0);
}

/***********************************************************************/

