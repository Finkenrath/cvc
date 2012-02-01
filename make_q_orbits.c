/*************************************************************
 * make_q_orbits.c
 *
 * Tue Jan 31 10:46:40 EET 2012
 * - changed the lattice size variables to global versions (LX_global, T_global)
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
#include "make_q_orbits.h"

#define _IDet3x3(A_) { \
    (A_)[0]*(A_)[4]*(A_)[8]  \
  + (A_)[1]*(A_)[5]*(A_)[6]  \
  + (A_)[2]*(A_)[3]*(A_)[7]  \
  - (A_)[0]*(A_)[5]*(A_)[7]  \
  - (A_)[1]*(A_)[3]*(A_)[8]  \
  - (A_)[2]*(A_)[4]*(A_)[6]  }

#define _QHATN(A_, B_) { \
  (A_)[0] =   4.*( _sqr( sin(M_PI*(double)(B_)[0] / (double)T_global)  ) + _sqr( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _sqr( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _sqr( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
  (A_)[1] =  16.*( _qrt( sin(M_PI*(double)(B_)[0] / (double)T_global)  ) + _qrt( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _qrt( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _qrt( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
  (A_)[2] =  64.*( _hex( sin(M_PI*(double)(B_)[0] / (double)T_global)  ) + _hex( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _hex( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _hex( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
  (A_)[3] = 256.*( _oct( sin(M_PI*(double)(B_)[0] / (double)T_global)  ) + _oct( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _oct( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _oct( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
}

#define _QHAT2(A_, B_) { \
  (A_)[0] =   4.*( _sqr( sin(M_PI*(double)(B_)[0] / (double)T_global)  ) + _sqr( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _sqr( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _sqr( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
}

#define _QHATN_3D(A_, B_) { \
  (A_)[0] =   4.*( _sqr( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _sqr( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _sqr( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
  (A_)[1] =  16.*( _qrt( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _qrt( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _qrt( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
  (A_)[2] =  64.*( _hex( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _hex( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _hex( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
  (A_)[3] = 256.*( _oct( sin(M_PI*(double)(B_)[1] / (double)LX_global) )  \
                 + _oct( sin(M_PI*(double)(B_)[2] / (double)LX_global) ) + _oct( sin(M_PI*(double)(B_)[3] / (double)LX_global) ) ); \
}

#define _QN_3D(A_, B_) { \
  (A_)[0] =   _sqr( 2.*M_PI/(double)LX_global ) * ( _sqr( (double)(B_)[1] ) +  _sqr( (double)(B_)[2] ) + _sqr( (double)(B_)[3] ) ); \
  (A_)[1] =   _qrt( 2.*M_PI/(double)LX_global ) * ( _qrt( (double)(B_)[1] ) +  _qrt( (double)(B_)[2] ) + _qrt( (double)(B_)[3] ) ); \
  (A_)[2] =   _hex( 2.*M_PI/(double)LX_global ) * ( _hex( (double)(B_)[1] ) +  _hex( (double)(B_)[2] ) + _hex( (double)(B_)[3] ) ); \
  (A_)[3] =   _oct( 2.*M_PI/(double)LX_global ) * ( _oct( (double)(B_)[1] ) +  _oct( (double)(B_)[2] ) + _oct( (double)(B_)[3] ) ); \
}

/*************************************************************************************
 *
 *************************************************************************************/
int init_q_orbits(int**xid, int**xid_count, double ***xid_val, int***xid_rep, int Nclasses) {
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

/***********************************************************************
 * deallocate memory for the fields used in H4 method analysis
 ***********************************************************************/
void finalize_q_orbits(int **xid, int **xid_count, double ***xid_val, int***xid_rep) {

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

/********************************************************************************
 *
 ********************************************************************************/
int make_q_orbits_3d(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep, double qmax) {

  int it, ix, iy, iz, iix, n;
  int x0;
  int isign, isign2, isign3;
  int Thp1  = T_global/2+1;
  int Thalf = T_global/2; 
  int Thm1  = T_global/2-1;
  int L     = LX_global;
  int Lhalf = L/2;
  int Lhp1  = L/2+1;
  int Lhm1  = L/2-1;
  int index_s, xcoords[4];
  int Nclasses, iclass, i, status;
  int new_flag;
  double qhat2;

  /************************************************
   * determine the number of classes
   * - at the moment: set to some large enough value
   ************************************************/
  Nclasses = T_global*L*L*L/4;
  fprintf(stdout, "# Nclasses = %d\n", Nclasses);
  
  status = init_q_orbits(xid, xid_count, xid_val, xid_rep, Nclasses);
  if(status != 0) {
    fprintf(stderr, "Error, could not intialise fields\n");
    return(1);
  }

  /***************************************
   * initialize the permutation tables
   ***************************************/
  init_perm_tabs();

  iclass = -1;
  for(it=0; it<T_global; it++) {
    if(it<=Thalf) { x0 = it;   }
    else          { x0 = it-T_global; }
    iclass++;
    (*xid_rep)[iclass][0] = x0;
    (*xid_rep)[iclass][1] = 0; 
    (*xid_rep)[iclass][2] = 0; 
    (*xid_rep)[iclass][3] = 0;
    iix = g_ipt[it][0][0][0];
    _QHATN( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QHATN( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QHATN( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QHATN( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QHATN( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QHATN( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QHATN( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QHATN( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
  for(it=0; it<T_global; it++) {
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
    fprintf(stdout, "# class number %d; members: %d; qhat2=%e\n", i, (*xid_count)[i], (*xid_val)[i][0]);
    for(it=0; it<T_global; it++) {
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

/********************************************************************************
 *
 ********************************************************************************/
int make_qlatt_orbits_3d_parity_avg(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep, int ***xmap) {

  int it, ix, iy, iz, iix, n;
  int x0;
  int isign, isign2, isign3;
  int Thp1  = T_global/2+1;
  int Thalf = T_global/2; 
  int Thm1  = T_global/2-1;
  int L     = LX_global;
  int Lhalf = L/2;
  int Lhp1  = L/2+1;
  int Lhm1  = L/2-1;
  int VOL3  = L*L*L;
  int index_s, xcoords[4];
  int Nclasses, iclass, i, status;
  int new_flag;
  double qhat2;

  /************************************************
   * determine the number of classes
   * - at the moment: set to some large enough value
   ************************************************/
  Nclasses = L*L*L/4;
  fprintf(stdout, "# Nclasses = %d\n", Nclasses);
  
  status = init_q_orbits(xid, xid_count, xid_val, xid_rep, Nclasses);
  if(status != 0) {
    fprintf(stderr, "Error, could not intialise fields\n");
    return(1);
  }

  /***************************************
   * initialize the permutation tables
   ***************************************/
  init_perm_tabs();

  iclass = -1;
//  for(it=0; it<T_global; it++) {
  it = 0;
  x0 = 0;
    iclass++;
    (*xid_rep)[iclass][0] = x0;
    (*xid_rep)[iclass][1] = 0; 
    (*xid_rep)[iclass][2] = 0; 
    (*xid_rep)[iclass][3] = 0;
    iix = g_ipt[it][0][0][0];
    _QHATN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      (*xid_rep)[iclass][0] = x0;  // should still be zero
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = 0;
      (*xid_rep)[iclass][3] = 0;
/*      fprintf(stdout, "(ii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]);  */
      _QHATN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      (*xid_rep)[iclass][0] = x0;  // still zero
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = 0;
/*      fprintf(stdout, "(iii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], 
        (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _QHATN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      (*xid_rep)[iclass][0] = x0;  // zero
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = iz;
/*      fprintf(stdout, "(iv) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _QHATN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QHATN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(v) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=0; i<6; i++) {
        for(isign =0; isign <2; isign++) {
        for(isign2=0; isign2<2; isign2++) {
        for(isign3=0; isign3<2; isign3++) {
          xcoords[0] = it;
          xcoords[perm_tab_3[i][0]+1] = (1-isign )*ix  + isign*(L-ix);
          xcoords[perm_tab_3[i][1]+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[perm_tab_3[i][2]+1] = (1-isign3)*iz + isign3*(L-iz);
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
        }}}
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
      _QHATN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      iz = ix;
      iclass++;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iz;
      (*xid_rep)[iclass][3] = iy;
      _QHATN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(viii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
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

//  }  /* of loop on times */
  Nclasses = iclass+1;
  *xid_nc  = iclass+1;
  fprintf(stdout, "# number of orbits:%3d\n", *xid_nc);

  (*xmap) = (int**)malloc(Nclasses*sizeof(int*));
  (*xmap)[0] = (int*)malloc(VOL3*sizeof(int));
  for(i=1; i<Nclasses; i++) (*xmap)[i] = (*xmap)[i-1] + (*xid_count)[i-1];
  
  
  for(i=0; i<Nclasses; i++) {
    it = 0;
    for(iix=0; iix<VOL3; iix++) {
      if((*xid)[iix]==i) {
        // fprintf(stdout, "\t match no %d for site %d and class %d\n", it, iix, i);
        // if(it>=(*xid_count)[i]) { fprintf(stderr, "\nError: too many instances for class %d\n", i);}
        (*xmap)[i][it] = iix;
        it++;
      }
    }
    if(it!=(*xid_count)[i]) { fprintf(stderr, "\nError: something wrong went wrong for class %d\n", i);}
  }

  // fprintf(stdout, "\n\n# complete xmap:\n");
  // for(ix=0;ix<VOL3;ix++) {fprintf(stdout, "\t%3d\t%3d\n", ix, (*xmap)[0][ix]); }

  /************************
   * test: print the lists 
   ************************/
/*
  fprintf(stdout, "\n\n# [] ===========================================================\n\n");
  fprintf(stdout, "# x\ty\tz\txid\n");
  for(ix=0; ix<L; ix++) {
  for(iy=0; iy<L; iy++) {
  for(iz=0; iz<L; iz++) {
    index_s = g_ipt[0][ix][iy][iz];
    fprintf(stdout, "\t%3d%3d%3d%8d\n", ix, iy, iz, (*xid)[index_s]);
  }}}
*/
/*
  fprintf(stdout, "\n\n# [] ===========================================================\n\n");
  fprintf(stdout, "# n\tx\ty\tz\tx^[2]\tx^[4]\tx^[6]\tx^[8]\tmembers\n");
  for(n=0; n<Nclasses; n++) {
    fprintf(stdout, "%5d%5d%5d%5d%16.7e%16.7e%16.7e%16.7e%4d\n",
      n, (*xid_rep)[n][1], (*xid_rep)[n][2], (*xid_rep)[n][3],
      (*xid_val)[n][0], (*xid_val)[n][1], (*xid_val)[n][2],(*xid_val)[n][3], (*xid_count)[n]);
  }
*/
/*
  fprintf(stdout, "\n\n# [] ===========================================================\n\n");
  for(i=0; i<Nclasses; i++) {
    fprintf(stdout, "# class number %d; members: %d; qhat2=%e\n", i, (*xid_count)[i], (*xid_val)[i][0]);
    for(it=0;it<(*xid_count)[i]; it++ ) {
      ix = (*xmap)[i][it];
      iz = ix % L;
      iy = ( ix%(L*L) ) / L;
      ix = ix / (L*L);
      fprintf(stdout, "\t%d ---> (%d, %d, %d)\n", (*xmap)[i][it], ix, iy, iz);
      if(ix>Lhalf) ix = L-ix;
      if(iy>Lhalf) iy = L-iy;
      if(iz>Lhalf) iz = L-iz;
      qhat2 = _sqr(2*M_PI/(double)L) * (double)( _sqr(ix) + _sqr(iy) + _sqr(iz) );
      if( fabs(qhat2-(*xid_val)[i][0]) > _Q2EPS ) {
        fprintf(stdout, "Warning: difference in q2 for class %d, member %d: %e vs %e\n", i, it, qhat2, (*xid_val)[i][0]);
      }
      qhat2 = _qrt(2*M_PI/(double)L) * (double)( _qrt(ix) + _qrt(iy) + _qrt(iz) );
      if( fabs(qhat2-(*xid_val)[i][1]) > _Q2EPS ) {
        fprintf(stdout, "Warning: difference in q4 for class %d, member %d: %e vs %e\n", i, it, qhat2, (*xid_val)[i][1]);
      }
      qhat2 = _hex(2*M_PI/(double)L) * (double)( _hex(ix) + _hex(iy) + _hex(iz) );
      if( fabs(qhat2-(*xid_val)[i][2]) > _Q2EPS ) {
        fprintf(stdout, "Warning: difference in q6 for class %d, member %d: %e vs %e\n", i, it, qhat2, (*xid_val)[i][2]);
      }
      qhat2 = _oct(2*M_PI/(double)L) * (double)( _oct(ix) + _oct(iy) + _oct(iz) );
      if( fabs(qhat2-(*xid_val)[i][3]) > _Q2EPS ) {
        fprintf(stdout, "Warning: difference in q8 for class %d, member %d: %e vs %e\n", i, it, qhat2, (*xid_val)[i][3]);
      }
    }
    fprintf(stdout, "# -----------------------------------------------------------\n");
  }
*/
  return(0);
}

/********************************************************************************
 * same as make_q_orbits_3d_parity_avg but with continuum momenta
 ********************************************************************************/
int make_qcont_orbits_3d_parity_avg(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep, int ***xmap) {

  int it, ix, iy, iz, iix, n;
  int x0;
  int isign, isign2, isign3;
  int Thp1  = T_global/2+1;
  int Thalf = T_global/2; 
  int Thm1  = T_global/2-1;
  int L     = LX_global;
  int Lhalf = L/2;
  int Lhp1  = L/2+1;
  int Lhm1  = L/2-1;
  int VOL3  = L*L*L;
  int index_s, xcoords[4];
  int Nclasses, iclass, i, status;
  int new_flag;
  double qhat2;

  /************************************************
   * determine the number of classes
   * - at the moment: set to some large enough value
   ************************************************/
  Nclasses = L*L*L/4;
  fprintf(stdout, "# Nclasses = %d\n", Nclasses);
  
  status = init_q_orbits(xid, xid_count, xid_val, xid_rep, Nclasses);
  if(status != 0) {
    fprintf(stderr, "Error, could not intialise fields\n");
    return(1);
  }

  /***************************************
   * initialize the permutation tables
   ***************************************/
  init_perm_tabs();

  iclass = -1;
//  for(it=0; it<T_global; it++) {
  it = 0;
  x0 = 0;
    iclass++;
    (*xid_rep)[iclass][0] = x0;
    (*xid_rep)[iclass][1] = 0; 
    (*xid_rep)[iclass][2] = 0; 
    (*xid_rep)[iclass][3] = 0;
    iix = g_ipt[it][0][0][0];
    _QN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      (*xid_rep)[iclass][0] = x0;  // should still be zero
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = 0;
      (*xid_rep)[iclass][3] = 0;
/*      fprintf(stdout, "(ii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]);  */
      _QN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      (*xid_rep)[iclass][0] = x0;  // still zero
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = 0;
/*      fprintf(stdout, "(iii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], 
        (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _QN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      (*xid_rep)[iclass][0] = x0;  // zero
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iy;
      (*xid_rep)[iclass][3] = iz;
/*      fprintf(stdout, "(iv) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
      _QN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      _QN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(v) dist: %16.7e%16.7e%16.7e%16.7e\n", (*xid_val)[iclass][0], (*xid_val)[iclass][1], 
        (*xid_val)[iclass][2], (*xid_val)[iclass][3]); */
      for(i=0; i<6; i++) {
        for(isign =0; isign <2; isign++) {
        for(isign2=0; isign2<2; isign2++) {
        for(isign3=0; isign3<2; isign3++) {
          xcoords[0] = it;
          xcoords[perm_tab_3[i][0]+1] = (1-isign )*ix  + isign*(L-ix);
          xcoords[perm_tab_3[i][1]+1] = (1-isign2)*iy + isign2*(L-iy);
          xcoords[perm_tab_3[i][2]+1] = (1-isign3)*iz + isign3*(L-iz);
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
        }}}
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
      _QN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
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
      iz = ix;
      iclass++;
      (*xid_rep)[iclass][0] = x0;
      (*xid_rep)[iclass][1] = ix;
      (*xid_rep)[iclass][2] = iz;
      (*xid_rep)[iclass][3] = iy;
      _QN_3D( (*xid_val)[iclass], (*xid_rep)[iclass]);
/*      fprintf(stdout, "(viii) representative: %3d%3d%3d%3d\n", (*xid_rep)[iclass][0], (*xid_rep)[iclass][1], (*xid_rep)[iclass][2], (*xid_rep)[iclass][3]); */
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

//  }  /* of loop on times */
  Nclasses = iclass+1;
  *xid_nc  = iclass+1;
  fprintf(stdout, "# number of orbits:%3d\n", *xid_nc);

  (*xmap) = (int**)malloc(Nclasses*sizeof(int*));
  (*xmap)[0] = (int*)malloc(VOL3*sizeof(int));
  for(i=1; i<Nclasses; i++) (*xmap)[i] = (*xmap)[i-1] + (*xid_count)[i-1];
  

  for(i=0; i<Nclasses; i++) {
    it = 0;
    for(iix=0; iix<VOL3; iix++) {
      if((*xid)[iix]==i) {
        // fprintf(stdout, "\t match no %d for site %d and class %d\n", it, iix, i);
        // fprintf(stdout, "\t class %3d: member[%2d] = %d\n", i, it, iix);
        // if(it>=(*xid_count)[i]) { fprintf(stderr, "\nError: too many instances for class %d\n", i);}
        (*xmap)[i][it] = iix;
        it++;
      }
    }
    if(it!=(*xid_count)[i]) { fprintf(stderr, "\nError: something wrong went wrong for class %d\n", i);}
  }

  // fprintf(stdout, "\n\n# complete xmap:\n");
  // for(ix=0;ix<VOL3;ix++) {fprintf(stdout, "\t%3d\t%3d\n", ix, (*xmap)[0][ix]); }

  /************************
   * test: print the lists 
   ************************/
/*
  fprintf(stdout, "\n\n# [] ===========================================================\n\n");
  fprintf(stdout, "# x\ty\tz\txid\n");
  for(ix=0; ix<L; ix++) {
  for(iy=0; iy<L; iy++) {
  for(iz=0; iz<L; iz++) {
    index_s = g_ipt[0][ix][iy][iz];
    fprintf(stdout, "\t%3d%3d%3d%8d\n", ix, iy, iz, (*xid)[index_s]);
  }}}
*/
/*
  fprintf(stdout, "\n\n# [] ===========================================================\n\n");
  fprintf(stdout, "# n\tx\ty\tz\tx^[2]\tx^[4]\tx^[6]\tx^[8]\tmembers\n");
  for(n=0; n<Nclasses; n++) {
    //fprintf(stdout, "%5d%5d%5d%5d%16.7e%16.7e%16.7e%16.7e%4d\n",
    //  n, (*xid_rep)[n][1], (*xid_rep)[n][2], (*xid_rep)[n][3],
    //  (*xid_val)[n][0], (*xid_val)[n][1], (*xid_val)[n][2],(*xid_val)[n][3], (*xid_count)[n]);
    fprintf(stdout, "%5d%5d%5d%5d\n", n, (*xid_rep)[n][1], (*xid_rep)[n][2], (*xid_rep)[n][3]);
  }
*/
/*
  fprintf(stdout, "\n\n# [] ===========================================================\n\n");
  for(i=0; i<Nclasses; i++) {
    fprintf(stdout, "# class number %d; members: %d; qhat2=%e\n", i, (*xid_count)[i], (*xid_val)[i][0]);
    for(it=0;it<(*xid_count)[i]; it++ ) {
      ix = (*xmap)[i][it];
      iz = ix % L;
      iy = ( ix%(L*L) ) / L;
      ix = ix / (L*L);
      fprintf(stdout, "\t%d ---> (%d, %d, %d)\n", (*xmap)[i][it], ix, iy, iz);
      if(ix>Lhalf) ix = L-ix;
      if(iy>Lhalf) iy = L-iy;
      if(iz>Lhalf) iz = L-iz;
      qhat2 = _sqr(2*M_PI/(double)L) * (double)( _sqr(ix) + _sqr(iy) + _sqr(iz) );
      if( fabs(qhat2-(*xid_val)[i][0]) > _Q2EPS ) {
        fprintf(stdout, "Warning: difference in q2 for class %d, member %d: %e vs %e\n", i, it, qhat2, (*xid_val)[i][0]);
      }
      qhat2 = _qrt(2*M_PI/(double)L) * (double)( _qrt(ix) + _qrt(iy) + _qrt(iz) );
      if( fabs(qhat2-(*xid_val)[i][1]) > _Q2EPS ) {
        fprintf(stdout, "Warning: difference in q4 for class %d, member %d: %e vs %e\n", i, it, qhat2, (*xid_val)[i][1]);
      }
      qhat2 = _hex(2*M_PI/(double)L) * (double)( _hex(ix) + _hex(iy) + _hex(iz) );
      if( fabs(qhat2-(*xid_val)[i][2]) > _Q2EPS ) {
        fprintf(stdout, "Warning: difference in q6 for class %d, member %d: %e vs %e\n", i, it, qhat2, (*xid_val)[i][2]);
      }
      qhat2 = _oct(2*M_PI/(double)L) * (double)( _oct(ix) + _oct(iy) + _oct(iz) );
      if( fabs(qhat2-(*xid_val)[i][3]) > _Q2EPS ) {
        fprintf(stdout, "Warning: difference in q8 for class %d, member %d: %e vs %e\n", i, it, qhat2, (*xid_val)[i][3]);
      }
    }
    fprintf(stdout, "# -----------------------------------------------------------\n");
  }
*/
  return(0);
}

