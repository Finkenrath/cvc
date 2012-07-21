/*************************************************************
 * make_x_orbits.c
 *
 * Thu Jul 15 11:31:04 CEST 2010
 *
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
#include "Q_phi.h"
#include "make_H3orbits.h"
#include "make_x_orbits.h"

#define _IDet3x3(A_) { \
    (A_)[0]*(A_)[4]*(A_)[8]  \
  + (A_)[1]*(A_)[5]*(A_)[6]  \
  + (A_)[2]*(A_)[3]*(A_)[7]  \
  - (A_)[0]*(A_)[5]*(A_)[7]  \
  - (A_)[1]*(A_)[3]*(A_)[8]  \
  - (A_)[2]*(A_)[4]*(A_)[6]  }

#define _Dist4d(A_, B_) { \
  (A_)[0] = (double)_sqr((double)(B_)[0]) + (double)_sqr((double)(B_)[1])  \
          + (double)_sqr((double)(B_)[2]) + (double)_sqr((double)(B_)[3]); \
  (A_)[1] = (double)_qrt((double)(B_)[0]) + (double)_qrt((double)(B_)[1])  \
          + (double)_qrt((double)(B_)[2]) + (double)_qrt((double)(B_)[3]); \
  (A_)[2] = (double)_hex((double)(B_)[0]) + (double)_hex((double)(B_)[1])  \
          + (double)_hex((double)(B_)[2]) + (double)_hex((double)(B_)[3]); \
  (A_)[3] = (double)_oct((double)(B_)[0]) + (double)_oct((double)(B_)[1])  \
          + (double)_oct((double)(B_)[2]) + (double)_oct((double)(B_)[3]);}

#define _Dist4dSL(A_, B_, C_) { \
  (A_)[0] = _sqr((double)(B_)[0]-(double)(C_)[0]) + _sqr((double)(B_)[1]-(double)(C_)[1])  \
          + _sqr((double)(B_)[2]-(double)(C_)[2]) + _sqr((double)(B_)[3]-(double)(C_)[3]); \
  (A_)[1] = _qrt((double)(B_)[0]-(double)(C_)[0]) + _qrt((double)(B_)[1]-(double)(C_)[1])  \
          + _qrt((double)(B_)[2]-(double)(C_)[2]) + _qrt((double)(B_)[3]-(double)(C_)[3]); \
  (A_)[2] = _hex((double)(B_)[0]-(double)(C_)[0]) + _hex((double)(B_)[1]-(double)(C_)[1])  \
          + _hex((double)(B_)[2]-(double)(C_)[2]) + _hex((double)(B_)[3]-(double)(C_)[3]); \
  (A_)[3] = _oct((double)(B_)[0]-(double)(C_)[0]) + _oct((double)(B_)[1]-(double)(C_)[1])  \
          + _oct((double)(B_)[2]-(double)(C_)[2]) + _oct((double)(B_)[3]-(double)(C_)[3]);}

/********************************************************************************
 *
 ********************************************************************************/
int make_x_orbits_3d(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep) {

  int it, ix, iy, iz, iix, n;
  int x0;
  int isign, isign2, isign3;
  int Thp1  = T/2+1;
  int Thalf = T/2; 
  int Thm1  = T/2-1;
  int L     = LX;
  int Lhalf = L/2;
  int Lhp1  = L/2+1;
  int Lhm1  = L/2-1;
  int index_s, xcoords[4];
  int Nclasses, iclass, i, status;
  int new_flag;

  /************************************************
   * determine the number of classes
   * - at the moment: set to some large enough value
   ************************************************/
  Nclasses = T*L*L*L/4;
  fprintf(stdout, "# Nclasses = %d\n", Nclasses);
  
  status = init_x_orbits(xid, xid_count, xid_val, xid_rep, Nclasses);
  if(status != 0) {
    fprintf(stderr, "Error, could not intialise fields\n");
    return(1);
  }

  /***************************************
   * initialize the permutation tables
   ***************************************/
  init_perm_tabs();

  iclass = -1;
  for(it=0; it<T; it++) {
    if(it<=Thalf) { x0 = it;   }
    else          { x0 = it-T; }
    iclass++;
    (*xid_rep)[iclass][0] = x0;
    (*xid_rep)[iclass][1] = 0; 
    (*xid_rep)[iclass][2] = 0; 
    (*xid_rep)[iclass][3] = 0;
    iix = g_ipt[it][0][0][0];
    _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
    new_flag = 0;
    if((*xid)[iix]==-1) {
      (*xid)[iix] = iclass;
      (*xid_count)[iclass] += 1;
      new_flag = 1;
    }
    else {
      if((*xid)[iix] != iclass) fprintf(stderr, "(i) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
    }
/*    fprintf(stdout, "(i) representative: %3d%3d%3d%3d%6d\t%2d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3], iix, new_flag);  */
/*    fprintf(stdout, "(i) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
      (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */

/*
    iclass++;
    (*xid_rep)[iclass][0] = x0;
    (*xid_rep)[iclass][1] = Lhalf;
    (*xid_rep)[iclass][2] = 0;
    (*xid_rep)[iclass][3] = 0;
    iix = g_ipt[it][Lhalf][0][0];
    fprintf(stdout, "representative: %3d%3d%3d%3d%6d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3], iix); 
    _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
    if((*xid)[iix] == -1) {
      (*xid)[iix] = iclass;
      (*xid_count)[iclass] += 1;
    }
*/

    for(ix=1; ix<=Lhalf; ix++) {
      iclass++;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = 0;
      (*xid_rep)[iclass][3] = 0;
/*      fprintf(stdout, "(ii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]);  */
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(ii) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=1; i<4; i++) {
      for(isign=0; isign<=1; isign++) {
        xcoords[0] = it;
        xcoords[(i-1)%3+1] = (1-isign)*ix + isign*(L-ix);
        xcoords[(i  )%3+1] = 0;
        xcoords[(i+1)%3+1] = 0;
        iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
        new_flag = 0;
        if((*xid)[iix] == -1) {
          (*xid)[iix] = iclass;
          (*xid_count)[iclass] += 1;
          new_flag = 1;
        }
        else {
          if((*xid)[iix] != iclass) fprintf(stderr, "(ii) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
        }
/*        fprintf(stdout, "( ii)\tmember: %3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */
      }}
    }

    for(ix=1; ix<=Lhalf; ix++) {
    for(iy=1; iy<=ix; iy++) {
      iclass++;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = 0;
/*      fprintf(stdout, "(iii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], 
        (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(iii) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=0; i<6; i++) {
        for(isign=0; isign<2; isign++) {
        for(isign2=0; isign2<2; isign2++) {
          xcoords[0] = it;
          xcoords[perm_tab_3[i][0]+1] = (1-isign)*ix  + isign*(L-ix);
          xcoords[perm_tab_3[i][1]+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[perm_tab_3[i][2]+1] = 0;
          iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
          new_flag = 0;
          if((*xid)[iix] == -1) {
            (*xid)[iix] = iclass;
            (*xid_count)[iclass] += 1;
            new_flag = 1;
          }
          else {
            if((*xid)[iix] != iclass) fprintf(stderr, "(iii) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
          }
/*          fprintf(stdout, "(iii)\tmember: %3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */
        }}
      }
    }}

    ix = Lhalf;
    for(iy=1; iy<=ix; iy++) {
    for(iz=1; iz<=iy; iz++) {
      iclass++;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = iz;
/*      fprintf(stdout, "(iv) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(iv) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=0; i<6; i++) {
        for(isign=0; isign<2; isign++) {
        for(isign2=0; isign2<2; isign2++) {
          xcoords[0] = it;
          xcoords[perm_tab_3[i][0]+1] = ix;
          xcoords[perm_tab_3[i][1]+1] = (1-isign )*iy + isign *(L-iy);
          xcoords[perm_tab_3[i][2]+1] = (1-isign2)*iz + isign2*(L-iz);
          iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
          new_flag = 0;
          if((*xid)[iix] == -1) {
            (*xid)[iix] = iclass;
            (*xid_count)[iclass] += 1;
            new_flag = 1;
          }
          else {
            if((*xid)[iix] != iclass) fprintf(stderr, "(iv) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
          }
/*          fprintf(stdout, "( iv) \tmember: %3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */
        }}
      }
    }}

    for(ix=1; ix<Lhalf; ix++) {
    for(iy=1; iy<ix; iy++) {
    for(iz=1; iz<iy; iz++) {
      iclass++;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = iz;
/*      fprintf(stdout, "(v) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(v) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=0; i<3; i++) {
        for(isign=0; isign<2; isign++) {
        for(isign2=0; isign2<2; isign2++) {
          isign3 = (  (isign || isign2) && !(isign && isign2 ) );
          xcoords[0] = it;
          xcoords[perm_tab_3e[i][0]+1] = (1-isign)*ix  + isign*(L-ix);
          xcoords[perm_tab_3e[i][1]+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[perm_tab_3e[i][2]+1] = (1-isign3)*iz + isign3*(L-iz);
          iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
          new_flag = 0;
          if((*xid)[iix] == -1) {
            (*xid)[iix] = iclass;
            (*xid_count)[iclass] += 1;
            new_flag = 1;
          }
          else {
            if((*xid)[iix] != iclass) fprintf(stderr, "(v) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
          }
/*          fprintf(stdout, "(  v)\tmember%3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */

          isign3 = ( isign && isign2 ) || (!isign && !isign2);
          xcoords[0] = it;
          xcoords[perm_tab_3o[i][0]+1] = (1-isign)*ix  + isign*(L-ix);
          xcoords[perm_tab_3o[i][1]+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[perm_tab_3o[i][2]+1] = (1-isign3)*iz + isign3*(L-iz);
          iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
          new_flag = 0;
          if((*xid)[iix] == -1) {
            (*xid)[iix] = iclass;
            (*xid_count)[iclass] += 1;
            new_flag = 1;
          }
          else {
            if((*xid)[iix] != iclass) fprintf(stderr, "(v) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
          }
/*          fprintf(stdout, "(  v)\tmember%3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */
        }}
      }

      iclass++;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = -ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = iz;
/*      fprintf(stdout, "(vi) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(vi) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=0; i<3; i++) {
        for(isign=0; isign<2; isign++) {
        for(isign2=0; isign2<2; isign2++) {
          isign3 = (  (isign || isign2) && !(isign && isign2 ) );
          xcoords[0] = it;
          xcoords[perm_tab_3o[i][0]+1] = (1-isign)*ix  + isign*(L-ix);
          xcoords[perm_tab_3o[i][1]+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[perm_tab_3o[i][2]+1] = (1-isign3)*iz + isign3*(L-iz);
          iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
          new_flag = 0;
          if((*xid)[iix] == -1) {
            (*xid)[iix] = iclass;
            (*xid_count)[iclass] += 1;
            new_flag = 1;
          }
          else {
            if((*xid)[iix] != iclass) fprintf(stderr, "(vi) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
          }
/*          fprintf(stdout, "( vi)\tmember%3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */

          isign3 = ( isign && isign2 ) || (!isign && !isign2);
          xcoords[0] = it;
          xcoords[perm_tab_3e[i][0]+1] = (1-isign)*ix  + isign*(L-ix);
          xcoords[perm_tab_3e[i][1]+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[perm_tab_3e[i][2]+1] = (1-isign3)*iz + isign3*(L-iz);
          iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
          new_flag = 0;
          if((*xid)[iix] == -1) {
            (*xid)[iix] = iclass;
            (*xid_count)[iclass] += 1;
            new_flag = 1;
          }
          else {
            if((*xid)[iix] != iclass) fprintf(stderr, "(vi) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
          }
/*          fprintf(stdout, "( vi)\tmember%3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */
        }}
      }
    }}}

    for(ix=1; ix<Lhalf; ix++) {
    for(iy=1; iy<=ix; iy++) {

      iclass++;
      iz = iy;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = iz;
/*      fprintf(stdout, "(vii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(vii) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=1; i<4; i++) {
        for(isign=0; isign<=1; isign++) {
        for(isign2=0; isign2<=1; isign2++) {
        for(isign3=0; isign3<=1; isign3++) {
          xcoords[0] = it;
          xcoords[(i-1)%3+1] = (1-isign )*ix + isign *(L-ix);
          xcoords[(i  )%3+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[(i+1)%3+1] = (1-isign3)*iz + isign3*(L-iz);
          iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
          new_flag = 0;
          if((*xid)[iix] == -1) {
            (*xid)[iix] = iclass;
            (*xid_count)[iclass] += 1;
            new_flag = 1;
          }
          else {
            if((*xid)[iix] != iclass) fprintf(stderr, "(vii) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
          }
/*          fprintf(stdout, "(vii)\tmember%3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */
        }}}
      }

      if(ix==iy) continue;
      iclass++;
      iz = ix;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iz;
      (*xid_rep)[iclass][3] = iy;
/*      fprintf(stdout, "(viii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(viii) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=1; i<4; i++) {
        for(isign=0; isign<=1; isign++) {
        for(isign2=0; isign2<=1; isign2++) {
        for(isign3=0; isign3<=1; isign3++) {
          xcoords[0] = it;
          xcoords[(i-1)%3+1] = (1-isign )*ix + isign *(L-ix);
          xcoords[(i  )%3+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[(i+1)%3+1] = (1-isign3)*iz + isign3*(L-iz);
          iix = g_ipt[xcoords[0]][xcoords[1]][xcoords[2]][xcoords[3]];
          new_flag = 0;
          if((*xid)[iix] == -1) {
            (*xid)[iix] = iclass;
            (*xid_count)[iclass] += 1;
            new_flag = 1;
          }
          else {
            if((*xid)[iix] != iclass) fprintf(stderr, "(viii) Warning: xid %d <---> %d\n", (*xid)[iix], iclass);
          }
/*          fprintf(stdout, "(viii)\tmember%3d%3d%3d%3d%6d\t%2d\n", xcoords[0], xcoords[1], xcoords[2], xcoords[3], iix, new_flag); */
        }}}
      }

    }}  /* ix and iy */

  }  /* of loop on times */
  Nclasses = iclass+1;
  *xid_nc  = iclass+1;
  fprintf(stdout, "# number of orbits:%3d\n", *xid_nc);

  /************************
   * test: print the lists 
   ************************/
/*
  fprintf(stdout, "# t\tx\ty\tz\txid\n");
  for(it=0; it<T; it++) {
  for(ix=0; ix<L; ix++) {
  for(iy=0; iy<L; iy++) {
  for(iz=0; iz<L; iz++) {
    index_s = g_ipt[it][ix][iy][iz];
    fprintf(stdout, "%3d%3d%3d%3d%4d\n", it, ix, iy, iz, (*xid)[index_s]);
  }}}}
*/
/*
  fprintf(stdout, "# n\tt\tx\ty\tz\tx^[2]\tx^[4]\tx^[6]\tx^[8]\tmembers\n");
  for(n=0; n<Nclasses; n++) {
    fprintf(stdout, "%5d%5d%5d%5d%5d%16.7e%16.7e%16.7e%16.7e%4d\n",
      n, (*xid_rep)[n][0], (*xid_rep)[n][1], (*xid_rep)[n][2], (*xid_rep)[n][3],
      (*xid_val)[n][0], (*xid_val)[n][1], (*xid_val)[n][2],(*xid_val)[n][3], (*xid_count)[n]);
  }
*/
/*
  for(i=0; i<Nclasses; i++) {
    fprintf(stdout, "# class number %d; members: %d\n", i, (*xid_count)[i]);
    for(it=0; it<T; it++) {
    for(ix=0; ix<L; ix++) {
    for(iy=0; iy<L; iy++) {
    for(iz=0; iz<L; iz++) {
      iix = g_ipt[it][ix][iy][iz];
      if((*xid)[iix]==i) fprintf(stdout, "%5d\t%5d%5d%5d%5d\n", i, it, ix, iy, iz);
    }}}}
    fprintf(stdout, "-----------------------------------------------------------\n");
  }
*/


  return(0);
}

/***********************************************************************
 * deallocate memory for the fields used in H4 method analysis
 ***********************************************************************/
void finalize_x_orbits(int **xid, int **xid_count, double ***xid_val, int***xid_rep) {

  if( *xid != NULL ) {
    free(*xid);
    *xid = NULL;
  }
  if(*xid_count != NULL) {
    free(*xid_count); *xid_count = NULL;
  }
  if(*xid_val != NULL) {
    if(**xid_val != NULL) {
      free(**xid_val);
      **xid_val = NULL;
    }
    free(*xid_val);
    *xid_val = NULL;
  }
  if(*xid_rep != NULL) {
    if(**xid_rep != NULL) {
      free(**xid_rep);
      **xid_rep = NULL;
    }
    free(*xid_rep);
    *xid_rep = NULL;
  }
}

/*************************************************************************************
 *
 *************************************************************************************/
int init_x_orbits(int**xid, int**xid_count, double ***xid_val, int***xid_rep, int Nclasses) {
  int i;

  if( (*xid_count = (int *)calloc(Nclasses, sizeof(int)))==NULL ) {
    fprintf(stderr, "# Error, could not allocate mem for xid_count\n");
    return(1);
  }

  if( (*xid_val=(double **)calloc(Nclasses, sizeof(double*))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for xid_val\n");
    return(2);
  }
  if( (*(*xid_val) = (double * )calloc(4 * Nclasses, sizeof(double))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for xid_val[0]\n");
    return(3);
  }
  for(i=1; i<Nclasses; i++) (*xid_val)[i] = (*xid_val)[i-1] + 4;

  if( (*xid_rep=(int **)calloc(Nclasses, sizeof(int*))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for xid_rep\n");
    return(5);
  }
  if( (*(*xid_rep) = (int * )calloc(4 * Nclasses, sizeof(double))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for xid_rep[0]\n");
    return(6);
  }
  for(i=1; i<Nclasses; i++) (*xid_rep)[i] = (*xid_rep)[i-1] + 4;

  for(i=0; i<Nclasses; i++) {
    (*xid_count)[i]  =  0;
    (*xid_val)[i][0] = -1.;
    (*xid_val)[i][1] = -1.;
    (*xid_val)[i][2] = -1.;
    (*xid_val)[i][3] = -1.;
    (*xid_rep)[i][0] = L;
    (*xid_rep)[i][1] = L;
    (*xid_rep)[i][2] = L;
    (*xid_rep)[i][3] = L;
  }

  if( (*xid = (int*)calloc(VOLUME, sizeof(int))) == NULL) {
    fprintf(stderr, "# Error, could not allocate mem for xid\n");
    return(7);
  }
  for(i=0; i<VOLUME; i++) (*xid)[i] = -1;

  return(0);
}


/********************************************************************************
 * 4d x^2 orbits
 ********************************************************************************/
int make_x_orbits_4d(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep) {

  size_t ix;
  int x[4], y[4], z[4];
  int s0, s1, s2, s3;
  int L     = LX;
  int Lhalf = L/2;
  int Lhp1  = L/2+1;
  int Lhm1  = L/2-1;
  size_t Nclasses, iclass;
  int i, status, ip;

  
  /***************************************
   * number of classes 
   ***************************************/
  Nclasses = \
  /* 0 <= t,x,y,z <= L/2 */ \
  + (Lhalf+1) * (Lhalf) * (Lhalf-1) * (Lhalf-2) / 24 /* 1111 */ \
  + (Lhalf+1) * (Lhalf) * (Lhalf-1) / 4             /* 112  */ \
  + (Lhalf+1) * (Lhalf) / 6                          /* 22   */ \
  + (Lhalf+1) * (Lhalf) / 4                          /* 13   */ \
  + Lhp1                                             /* 4    */ \
  /* L/2+1 <= t < 2L */ \
  + Lhalf * ( (Lhalf+1)*Lhalf*(Lhalf-1) / 6 /* 111 */ \
            + (Lhalf+1)*Lhalf / 3           /* 12  */ \
            + (Lhalf+1) )                   /* 3   */;

  fprintf(stdout, "# [make_x_orbits_4d] Nclasses = %u\n", Nclasses);
  
  status = init_x_orbits(xid, xid_count, xid_val, xid_rep, Nclasses);
  if(status != 0) {
    fprintf(stderr, "Error, could not intialise fields\n");
    return(1);
  }

  /***************************************
   * initialize the permutation tables
   ***************************************/
  init_perm_tabs();

  /***************************************
   * characterize the classes
   ***************************************/
  iclass = 0;
  /******************************************************
   * (1) t = 0,...,L/2
   ******************************************************/
  // 1111
  for(x[0]=0;    x[0]<Lhp1-3; x[0]++) {
  for(x[1]=x[0]+1; x[1]<Lhp1-2; x[1]++) {
  for(x[2]=x[1]+1; x[2]<Lhp1-1; x[2]++) {
  for(x[3]=x[2]+1; x[3]<Lhp1;   x[3]++) {
    (*xid_rep)[iclass][0] = x[0];
    (*xid_rep)[iclass][1] = x[1]; 
    (*xid_rep)[iclass][2] = x[2]; 
    (*xid_rep)[iclass][3] = x[3];
    _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
    for(ip = 0; ip< 24;ip++) {
      y[perm_tab_4[ip][0]] = x[0];
      y[perm_tab_4[ip][1]] = x[1];
      y[perm_tab_4[ip][2]] = x[2];
      y[perm_tab_4[ip][3]] = x[3];

      for(s0=-1; s0<2; s0+=2) {
        z[0] = ((1+s0) * y[0] + (1-s0) * ((T-y[0])%T) ) / 2;
      for(s1=-1; s1<2; s1+=2) {
        z[1] = ((1+s1) * y[1] + (1-s1) * ((L-y[1])%L) ) / 2;
      for(s2=-1; s2<2; s2+=2) {
        z[2] = ((1+s2) * y[2] + (1-s2) * ((L-y[2])%L) ) / 2;
      for(s3=-1; s3<2; s3+=2) {
        z[3] = ((1+s3) * y[3] + (1-s3) * ((L-y[3])%L) ) / 2;
        ix = g_ipt[z[0]][z[1]][z[2]][z[3]];
        fprintf(stdout, "# [make_x_orbits_4d] z=(%2d,%2d,%2d,%2d); ix=%6u; id=%4d; class=%4d\n", z[0], z[1], z[2], z[3], ix, (*xid)[ix], iclass);
        if( (*xid)[ix] == -1 ) {
          (*xid)[ix] = iclass;
          (*xid_count)[iclass]++;
        } else if ((*xid)[ix] != iclass) {
          fprintf(stderr, "[make_x_orbits_4d] Error, class id of site %u != current class id %u\n", (*xid)[ix], iclass);
          return(2);
        }
      }}}}
    }
    iclass++;
  }}}}

  fprintf(stdout, "# [make_x_orbits_4d] finished classes 1111\n");

  // 112
  for(x[0]=0;    x[0]<Lhp1-2; x[0]++) {
  for(x[1]=x[0]+1; x[1]<Lhp1-1; x[1]++) {
  for(x[2]=x[1]+1; x[2]<Lhp1  ; x[2]++) {
  for(i=0;i<3;i++) {
    x[3] = x[i];
    (*xid_rep)[iclass][0] = x[0];
    (*xid_rep)[iclass][1] = x[1]; 
    (*xid_rep)[iclass][2] = x[2]; 
    (*xid_rep)[iclass][3] = x[3];
    _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
    for(ip = 0; ip< 24;ip++) {
      y[perm_tab_4[ip][0]] = x[0];
      y[perm_tab_4[ip][1]] = x[1];
      y[perm_tab_4[ip][2]] = x[2];
      y[perm_tab_4[ip][3]] = x[3];

      for(s0=-1; s0<2; s0+=2) {
        z[0] = ((1+s0) * y[0] + (1-s0) * ((T-y[0])%T) ) / 2;
      for(s1=-1; s1<2; s1+=2) {
        z[1] = ((1+s1) * y[1] + (1-s1) * ((L-y[1])%L) ) / 2;
      for(s2=-1; s2<2; s2+=2) {
        z[2] = ((1+s2) * y[2] + (1-s2) * ((L-y[2])%L) ) / 2;
      for(s3=-1; s3<2; s3+=2) {
        z[3] = ((1+s3) * y[3] + (1-s3) * ((L-y[3])%L) ) / 2;
        ix = g_ipt[z[0]][z[1]][z[2]][z[3]];
        fprintf(stdout, "# [make_x_orbits_4d] z=(%2d,%2d,%2d,%2d); ix=%6u; id=%4d; class=%4d\n", z[0], z[1], z[2], z[3], ix, (*xid)[ix], iclass);
        if( (*xid)[ix] == -1 ) {
          (*xid)[ix] = iclass;
          (*xid_count)[iclass]++;
        } else if ((*xid)[ix] != iclass) {
          fprintf(stderr, "[] Error, class id of site %u != current class id %u\n", (*xid)[ix], iclass);
          return(3);
        }
      }}}}
    }
    iclass++;
  }}}}
  fprintf(stdout, "# [make_x_orbits_4d] finished classes 112\n");
#if 0
  // 22
  for(x[0]=0;    x[0]<Lhp1-1; x[0]++) {
  for(x[2]=x[0]+1; x[2]<Lhp1  ; x[2]++) {
    (*xid_rep)[iclass][0] = x[0];
    (*xid_rep)[iclass][1] = x[0]; 
    (*xid_rep)[iclass][2] = x[2]; 
    (*xid_rep)[iclass][3] = x[2];
    _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
    for(ip = 0; ip< 24;ip++) {
      y[perm_tab_4[ip][0]] = x[0];
      y[perm_tab_4[ip][1]] = x[0];
      y[perm_tab_4[ip][2]] = x[2];
      y[perm_tab_4[ip][3]] = x[2];

      for(s0=-1; s0<2; s0+=2) {
        z[0] = ((1+s0) * y[0] + (1-s0) * ((T-y[0])%T) ) / 2;
      for(s1=-1; s1<2; s1+=2) {
        z[1] = ((1+s1) * y[1] + (1-s1) * ((L-y[1])%L) ) / 2;
      for(s2=-1; s2<2; s2+=2) {
        z[2] = ((1+s2) * y[2] + (1-s2) * ((L-y[2])%L) ) / 2;
      for(s3=-1; s3<2; s3+=2) {
        z[3] = ((1+s3) * y[3] + (1-s3) * ((L-y[3])%L) ) / 2;
        ix = g_ipt[z[0]][z[1]][z[2]][z[3]];
        fprintf(stdout, "# [make_x_orbits_4d] z=(%2d,%2d,%2d,%2d); ix=%6u; id=%4d; class=%4d\n", z[0], z[1], z[2], z[3], ix, (*xid)[ix], iclass);
        if( (*xid)[ix] == -1 ) {
          (*xid)[ix] = iclass;
          (*xid_count)[iclass]++;
        } else if ((*xid)[ix] != iclass) {
          fprintf(stderr, "[] Error, class id of site %u != current class id %u\n", (*xid)[ix], iclass);
          return(3);
        }
      }}}}
    }
    iclass++;
  }}
  fprintf(stdout, "# [make_x_orbits_4d] finished classes 22\n");

  // 13
  for(x[0]=0;    x[0]<Lhp1-1; x[0]++) {
  for(x[3]=x[0]+1; x[3]<Lhp1  ; x[3]++) {
    (*xid_rep)[iclass][0] = x[0];
    (*xid_rep)[iclass][1] = x[0]; 
    (*xid_rep)[iclass][2] = x[0]; 
    (*xid_rep)[iclass][3] = x[3];
    _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
    for(ip = 0; ip<4;ip++) {
      y[ip      ] = x[3];
      y[(ip+1)%4] = x[0];
      y[(ip+2)%4] = x[0];
      y[(ip+3)%4] = x[0]; 

      for(s0=-1; s0<2; s0+=2) {
        z[0] = ((1+s0) * y[0] + (1-s0) * ((T-y[0])%T) ) / 2;
      for(s1=-1; s1<2; s1+=2) {
        z[1] = ((1+s1) * y[1] + (1-s1) * ((L-y[1])%L) ) / 2;
      for(s2=-1; s2<2; s2+=2) {
        z[2] = ((1+s2) * y[2] + (1-s2) * ((L-y[2])%L) ) / 2;
      for(s3=-1; s3<2; s3+=2) {
        z[3] = ((1+s3) * y[3] + (1-s3) * ((L-y[3])%L) ) / 2;
        ix = g_ipt[z[0]][z[1]][z[2]][z[3]];
        fprintf(stdout, "# [make_x_orbits_4d] z=(%2d,%2d,%2d,%2d); ix=%6u; id=%4d; class=%4d\n", z[0], z[1], z[2], z[3], ix, (*xid)[ix], iclass);
        if( (*xid)[ix] == -1 ) {
          (*xid)[ix] = iclass;
          (*xid_count)[iclass]++;
        } else if ((*xid)[ix] != iclass) {
          fprintf(stderr, "[] Error, class id of site %u != current class id %u\n", (*xid)[ix], iclass);
          return(4);
        }
      }}}}
    }
    iclass++;
  }}
  fprintf(stdout, "# [make_x_orbits_4d] finished classes 13\n");

  // 4
  for(x[0]=0;    x[0]<Lhp1; x[0]++) {
    (*xid_rep)[iclass][0] = x[0];
    (*xid_rep)[iclass][1] = x[0]; 
    (*xid_rep)[iclass][2] = x[0]; 
    (*xid_rep)[iclass][3] = x[0];
    _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
    y[0] = x[0];
    y[1] = x[0];
    y[2] = x[0];
    y[3] = x[0];

    for(s0=-1; s0<2; s0+=2) {
      z[0] = ((1+s0) * y[0] + (1-s0) * ((T-y[0])%T) ) / 2;
    for(s1=-1; s1<2; s1+=2) {
      z[1] = ((1+s1) * y[1] + (1-s1) * ((L-y[1])%L) ) / 2;
    for(s2=-1; s2<2; s2+=2) {
      z[2] = ((1+s2) * y[2] + (1-s2) * ((L-y[2])%L) ) / 2;
    for(s3=-1; s3<2; s3+=2) {
      z[3] = ((1+s3) * y[3] + (1-s3) * ((L-y[3])%L) ) / 2;
      ix = g_ipt[z[0]][z[1]][z[2]][z[3]];
      fprintf(stdout, "# [make_x_orbits_4d] z=(%2d,%2d,%2d,%2d); ix=%6u; id=%4d; class=%4d\n", z[0], z[1], z[2], z[3], ix, (*xid)[ix], iclass);
      if( (*xid)[ix] == -1 ) {
        (*xid)[ix] = iclass;
        (*xid_count)[iclass]++;
      } else if ((*xid)[ix] != iclass) {
        fprintf(stderr, "[] Error, class id of site %u != current class id %u\n", (*xid)[ix], iclass);
        return(5);
      }
    }}}}
    iclass++;
  }
  fprintf(stdout, "# [make_x_orbits_4d] finished classes 4\n");

  /******************************************************
   * (2) t = L/2+1,...,2L-1
   ******************************************************/

  for(x[0]=Lhp1; x[0]<=L; x[0]++) {
    (*xid_rep)[iclass][0] = x[0];
    y[0] = x[0];
    // 111
    for(x[1]=0;    x[1]<Lhp1-2; x[1]++) {
    for(x[2]=x[1]+1; x[2]<Lhp1-1; x[2]++) {
    for(x[3]=x[2]+1; x[3]<Lhp1;   x[3]++) {
      (*xid_rep)[iclass][1] = x[1]; 
      (*xid_rep)[iclass][2] = x[2]; 
      (*xid_rep)[iclass][3] = x[3];
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
      for(ip = 0; ip< 6; ip++) {
        y[perm_tab_3[ip][0]+1] = x[1];
        y[perm_tab_3[ip][1]+1] = x[2];
        y[perm_tab_3[ip][2]+1] = x[3];
  
        for(s0=-1; s0<2; s0+=2) {
          z[0] = ((1+s0) * y[0] + (1-s0) * ((T-y[0])%T) ) / 2;
        for(s1=-1; s1<2; s1+=2) {
          z[1] = ((1+s1) * y[1] + (1-s1) * ((L-y[1])%L) ) / 2;
        for(s2=-1; s2<2; s2+=2) {
          z[2] = ((1+s2) * y[2] + (1-s2) * ((L-y[2])%L) ) / 2;
        for(s3=-1; s3<2; s3+=2) {
          z[3] = ((1+s3) * y[3] + (1-s3) * ((L-y[3])%L) ) / 2;
          ix = g_ipt[z[0]][z[1]][z[2]][z[3]];
          fprintf(stdout, "# [make_x_orbits_4d] z=(%2d,%2d,%2d,%2d); ix=%6u; id=%4d; class=%4d\n", z[0], z[1], z[2], z[3], ix, (*xid)[ix], iclass);
          if( (*xid)[ix] == -1 ) {
            (*xid)[ix] = iclass;
            (*xid_count)[iclass]++;
          } else if ((*xid)[ix] != iclass) {
            fprintf(stderr, "[] Error, class id of site %u != current class id %u\n", (*xid)[ix], iclass);
            return(3);
          }
        }}}}
      }
      iclass++;
    }}}
    fprintf(stdout, "# [make_x_orbits_4d] finished classes 1+111\n");
  
    // 12
    for(x[1]=0;    x[1]<Lhp1-1; x[1]++) {
    for(x[2]=x[1]+1; x[2]<Lhp1;   x[2]++) {
      (*xid_rep)[iclass][1] = x[1]; 
      (*xid_rep)[iclass][2] = x[2]; 
      (*xid_rep)[iclass][3] = x[2];
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
      for(ip = 0; ip<3; ip++) {
        y[ip      +1] = x[1];
        y[(ip+1)%3+1] = x[2];
        y[(ip+2)%3+1] = x[2];
  
        for(s0=-1; s0<2; s0+=2) {
          z[0] = ((1+s0) * y[0] + (1-s0) * ((T-y[0])%T) ) / 2;
        for(s1=-1; s1<2; s1+=2) {
          z[1] = ((1+s1) * y[1] + (1-s1) * ((L-y[1])%L) ) / 2;
        for(s2=-1; s2<2; s2+=2) {
          z[2] = ((1+s2) * y[2] + (1-s2) * ((L-y[2])%L) ) / 2;
        for(s3=-1; s3<2; s3+=2) {
          z[3] = ((1+s3) * y[3] + (1-s3) * ((L-y[3])%L) ) / 2;
          ix = g_ipt[z[0]][z[1]][z[2]][z[3]];
          fprintf(stdout, "# [make_x_orbits_4d] z=(%2d,%2d,%2d,%2d); ix=%6u; id=%4d; class=%4d\n", z[0], z[1], z[2], z[3], ix, (*xid)[ix], iclass);
          if( (*xid)[ix] == -1 ) {
            (*xid)[ix] = iclass;
            (*xid_count)[iclass]++;
          } else if ((*xid)[ix] != iclass) {
            fprintf(stderr, "[] Error, class id of site %u != current class id %u\n", (*xid)[ix], iclass);
            return(3);
          }
        }}}}
      }
      iclass++;
    }}
    fprintf(stdout, "# [make_x_orbits_4d] finished classes 1+12\n");
  
    // 3
    for(x[1]=0; x[1]<Lhp1; x[1]++) {
      (*xid_rep)[iclass][1] = x[1]; 
      (*xid_rep)[iclass][2] = x[1]; 
      (*xid_rep)[iclass][3] = x[1];
      _Dist4d( (*xid_val)[iclass], (*xid_rep)[iclass]);
      y[1] = x[1];
      y[2] = x[1];
      y[3] = x[1];
  
      for(s0=-1; s0<2; s0+=2) {
        z[0] = ((1+s0) * y[0] + (1-s0) * ((T-y[0])%T) ) / 2;
      for(s1=-1; s1<2; s1+=2) {
        z[1] = ((1+s1) * y[1] + (1-s1) * ((L-y[1])%L) ) / 2;
      for(s2=-1; s2<2; s2+=2) {
        z[2] = ((1+s2) * y[2] + (1-s2) * ((L-y[2])%L) ) / 2;
      for(s3=-1; s3<2; s3+=2) {
        z[3] = ((1+s3) * y[3] + (1-s3) * ((L-y[3])%L) ) / 2;
        ix = g_ipt[z[0]][z[1]][z[2]][z[3]];
        fprintf(stdout, "# [make_x_orbits_4d] z=(%2d,%2d,%2d,%2d); ix=%6u; id=%4d; class=%4d\n", z[0], z[1], z[2], z[3], ix, (*xid)[ix], iclass);
        if( (*xid)[ix] == -1 ) {
          (*xid)[ix] = iclass;
          (*xid_count)[iclass]++;
        } else if ((*xid)[ix] != iclass) {
          fprintf(stderr, "[] Error, class id of site %u != current class id %u\n", (*xid)[ix], iclass);
          return(5);
        }
      }}}}
      iclass++;
    }
    fprintf(stdout, "# [make_x_orbits_4d] finished classes 1+12\n");

  }  // of loop on times


  if(Nclasses != iclass) {
    fprintf(stderr, "[] Error, counted number of classes %u differs from calculated number %u\n", iclass, Nclasses);
    return(6);
  }
  *xid_nc = Nclasses;

   // TEST:
   // - print the lists 

  fprintf(stdout, "# t\tx\ty\tz\txid\n");
  for(x[0]=0; x[0]<T; x[0]++) {
  for(x[1]=0; x[1]<L; x[1]++) {
  for(x[2]=0; x[2]<L; x[2]++) {
  for(x[3]=0; x[3]<L; x[3]++) {
    ix = g_ipt[x[0]][x[1]][x[2]][x[3]];
    fprintf(stdout, "%3d%3d%3d%3d%4d\n", x[0], x[1], x[2], x[3], (*xid)[ix]);
  }}}}


  fprintf(stdout, "# n\tt\tx\ty\tz\tx^[2]\tx^[4]\tx^[6]\tx^[8]\tmembers\n");
  for(i=0; i<Nclasses; i++) {
    fprintf(stdout, "%5d%5d%5d%5d%5d%16.7e%16.7e%16.7e%16.7e%4d\n",
      i, (*xid_rep)[i][0], (*xid_rep)[i][1], (*xid_rep)[i][2], (*xid_rep)[i][3],
      (*xid_val)[i][0], (*xid_val)[i][1], (*xid_val)[i][2],(*xid_val)[i][3], (*xid_count)[i]);
  }
#endif  // of if 0
/*
  for(i=0; i<Nclasses; i++) {
    fprintf(stdout, "# class number %d; members: %d\n", i, (*xid_count)[i]);
    for(it=0; it<T; it++) {
    for(ix=0; ix<L; ix++) {
    for(iy=0; iy<L; iy++) {
    for(iz=0; iz<L; iz++) {
      iix = g_ipt[it][ix][iy][iz];
      if((*xid)[iix]==i) fprintf(stdout, "%5d\t%5d%5d%5d%5d\n", i, it, ix, iy, iz);
    }}}}
    fprintf(stdout, "-----------------------------------------------------------\n");
  }
*/
  return(0);
}
