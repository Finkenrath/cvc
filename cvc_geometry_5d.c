#include <stdlib.h>
#include <stdio.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include <math.h>
#include "cvc_complex.h"
#include "global.h"
#include "cvc_utils.h"
#include "cvc_geometry.h"

int *iup_5d=NULL, *idn_5d=NULL, *ipt_5d=NULL , **ipt_5d_=NULL, ***ipt_5d__=NULL, ****ipt_5d___=NULL;

unsigned long int get_index_5d(const int s, const int t, const int x, const int y, const int z)
{

  unsigned long int tt, xx, yy, zz, ix, ss;
  unsigned long int V5 = LX*LY*LZ*T*L5;

  tt = (t + T) % T;
  xx = (x + LX) % LX;
  yy = (y + LY) % LY;
  zz = (z + LZ) % LZ;
  ss = (s + L5) % L5;
  ix = ((tt*LX+xx)*LY+yy)*LZ+zz + s*LX*LY*LZ*T ;

#ifdef MPI
  if(t==T) {
    ix = V5 + ss*RAND + (xx*LY+yy)*LZ+zz;
  }
  if(t==-1) {
    ix = V5 + ss*RAND + LX*LY*LZ + (xx*LY+yy)*LZ+zz;
  }
#  if defined PARALLELTX || defined PARALLELTXY
  if(x==LX) {
    ix = V5 + ss*RAND + 2*LX*LY*LZ +           (tt*LY+yy)*LZ+zz;
  }
  if(x==-1) {
    ix = V5 + ss*RAND + 2*LX*LY*LZ + T*LY*LZ + (tt*LY+yy)*LZ+zz;
  }
#  endif
#  if defined PARALLELTXY
  if(y==LY) {
    ix = V5 + ss*RAND + 2*(LX*LY*LZ + T*LY*LZ) + (tt*LX+xx)*LZ+zz;
  }
  if(y==-1) {
    ix = V5 + ss*RAND + 2*(LX*LY*LZ + T*LY*LZ) + T*LX*LZ + (tt*LX+xx)*LZ+zz;
  }
#  endif

#  if defined PARALLELTX || defined PARALLELTXY

  // x-t-edges
  if(x==LX) {
    if(t==T) {
      ix = V5 + L5*RAND + ss*EDGES + yy*LZ+zz;
    }
    if(t==-1) {
      ix = V5 + L5*RAND + ss*EDGES + 2*LY*LZ + yy*LZ+zz;
    }
  }
  if(x==-1) {
    if(t==T) {
      ix = V5 + L5*RAND + ss*EDGES + LY*LZ + yy*LZ+zz;
    }
    if(t==-1) {
      ix = V5 + L5*RAND + ss*EDGES + 3*LY*LZ + yy*LZ+zz;
    }
  }
#  if defined PARALLELTXY
  // y-t-edges
  if(t==T) {
    if(y==LY) {
      ix = V5 + L5*RAND + ss*EDGES + 4*LY*LZ           + xx*LZ+zz;
    }
    if(y==-1) {
      ix = V5 + L5*RAND + ss*EDGES + 4*LY*LZ +   LX*LZ + xx*LZ+zz;
    }
  }
  if(t==-1) {
    if(y==LY) {
      ix = V5 + L5*RAND + ss*EDGES + 4*LY*LZ + 2*LX*LZ + xx*LZ+zz;
    }
    if(y==-1) {
      ix = V5 + L5*RAND + ss*EDGES + 4*LY*LZ + 3*LX*LZ + xx*LZ+zz;
    }
  }

  // y-x-edges
  if(x==LX) {
    if(y==LY) {
      ix = V5 + L5*RAND + ss*EDGES + 4*(LY*LZ + LX*LZ)          + tt*LZ+zz;
    }
    if(y==-1) {
      ix = V5 + L5*RAND + ss*EDGES + 4*(LY*LZ + LX*LZ) +   T*LZ + tt*LZ+zz;
    }
  }
  if(x==-1) {
    if(y==LY) {
      ix = V5 + L5*RAND + ss*EDGES + 4*(LY*LZ + LX*LZ) + 2*T*LZ + tt*LZ+zz;
    }
    if(y==-1) {
      ix = V5 + L5*RAND + ss*EDGES + 4*(LY*LZ + LX*LZ) + 3*T*LZ + tt*LZ+zz;
    }
  }
#  endif  /* of if defined PARALLELTXY */
#  endif  /* of if defined PARALLELTX || defined PARALLELTXY */
#endif
  return(ix);
}

void geometry_5d() {

  int is;
  int x0, x1, x2, x3;
  int y0, y1, y2, y3, ix;
  int isboundary;
  int i_even, i_odd;
  int itzyx;
  unsigned int V5  = T * LX * LY * LZ * L5;
  unsigned int V5h = V5 / 2;

#ifdef MPI
  int start_valuet = 1;

#  if defined PARALLELTX || defined PARALLELTXY
  int start_valuex = 1;
#  else
  int start_valuex = 0;
#  endif

#  if defined PARALLELTXY
  int start_valuey = 1;
#  else
  int start_valuey = 0;
#  endif

#else
  int start_valuet = 0;
  int start_valuex = 0;
  int start_valuey = 0;
#endif

  for(is=0;is<L5;is++) {
  for(x0=-start_valuet; x0<T +start_valuet; x0++) {
  for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
  for(x2=-start_valuey; x2<LY+start_valuey; x2++) {

  for(x3=0; x3<LZ; x3++) {

    isboundary = 0;
    if(x0==-1 || x0== T) isboundary++;
    if(x1==-1 || x1==LX) isboundary++;
    if(x2==-1 || x2==LY) isboundary++;

    y0=x0; y1=x1; y2=x2; y3=x3;
    if(x0==-1) y0=T +1;
    if(x1==-1) y1=LX+1;
    if(x2==-1) y2=LY+1;

    if(isboundary > 2) {
      g_ipt_5d[is][y0][y1][y2][y3] = -1;
      continue;
    }

    ix = get_index_5d(is, x0, x1, x2, x3);

    //fprintf(stdout, "%d, %d, %d, %d, %d\n", is, y0, y1, y2, y3);
    //fflush(stdout);

    g_ipt_5d[is][y0][y1][y2][y3] = ix;

    g_iup_5d[ix][0] = get_index_5d(is, x0+1, x1, x2, x3);
    g_iup_5d[ix][1] = get_index_5d(is, x0, x1+1, x2, x3);
    g_iup_5d[ix][2] = get_index_5d(is, x0, x1, x2+1, x3);
    g_iup_5d[ix][3] = get_index_5d(is, x0, x1, x2, x3+1);

    g_idn_5d[ix][0] = get_index_5d(is, x0-1, x1, x2, x3);
    g_idn_5d[ix][1] = get_index_5d(is, x0, x1-1, x2, x3);
    g_idn_5d[ix][2] = get_index_5d(is, x0, x1, x2-1, x3);
    g_idn_5d[ix][3] = get_index_5d(is, x0, x1, x2, x3-1);

    // is even / odd
    g_iseven_5d[ix] = ( is + x0 + T *g_proc_coords[0] + x1 + LX*g_proc_coords[1] \
                   + x2 + LY*g_proc_coords[2] + x3 + LZ*g_proc_coords[3] ) % 2 == 0;

    // replace this by indext function
    itzyx = ( ( x0*LZ + x3 ) * LY + x2 ) * LX + x1 + is*T*LX*LY*LZ;
    g_isevent_5d[itzyx] = g_iseven_5d[ix];

  }}}} // of x3, x2, x1, x0
  }    // of is


  i_even = 0; i_odd = 0;
  for(ix=0; ix<(VOLUME+RAND)*L5; ix++) {
    if(g_iseven_5d[ix]) {
      g_lexic2eo_5d[ix    ] = i_even;
      g_eo2lexic_5d[i_even] = ix;
      i_even++;
    } else {
      g_lexic2eo_5d[ix         ] = i_odd + V5h;
      g_eo2lexic_5d[i_odd + V5h] = ix;
      i_odd++;
    }
  }

  // this will have to be changed if to be used with MPI
  itzyx = 0;
  i_even = 0; i_odd = 0;
  for(is=0;is<L5;is++) {
  for(x0=0;x0<T; x0++) {
  for(x3=0;x3<LZ;x3++) {
  for(x2=0;x2<LY;x2++) {
  for(x1=0;x1<LX;x1++) {
    ix = g_ipt_5d[is][x0][x1][x2][x3];
    if(ix == -1) continue;
    if(g_isevent_5d[itzyx]) {
      g_lexic2eot_5d[ix    ] = i_even;
      g_eot2lexic_5d[i_even] = ix;
      i_even++;
    } else {
      g_lexic2eot_5d[ix         ] = i_odd + V5h;
      g_eot2lexic_5d[i_odd + V5h] = ix;
      i_odd++;
    }
    itzyx++;
  }}}}
  }



/*
  if(g_cart_id==0) {

    for(x0=-start_valuet; x0< T+start_valuet; x0++) {
    for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
    for(x2=-start_valuey; x2<LY+start_valuey; x2++) {
    for(x3=0; x3<LZ; x3++) {
      y0=x0; y1=x1; y2=x2; y3=x3;
      if(x0==-1) y0 = T+1;
      if(x1==-1) y1 =LX+1;
      if(x2==-1) y2 =LY+1;
      fprintf(stdout, "[%2d geometry] %3d%3d%3d%3d%6d\n", g_cart_id, x0, x1, x2, x3, g_ipt_5d[is][y0][y1][y2][y3]);
    }}}}

  }
*/
/*
  if(g_cart_id==0) {
    for(is=0;is<L5;is++) {
    for(x0=-start_valuet; x0<T+start_valuet; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      y0=x0; y1=x1; y2=x2; y3=x3;
      if(x0==-1) y0=T+1;
      ix = g_ipt_5d[is][y0][y1][y2][y3];      
      fprintf(stdout, "%5d|%5d%5d%5d%5d||%8d%8d|%8d%8d|%8d%8d|%8d%8d\n", is, x0, x1, x2, x3, 
        g_iup_5d[ix][0], g_idn_5d[ix][0], g_iup_5d[ix][1], g_idn_5d[ix][1],
	g_iup_5d[ix][2], g_idn_5d[ix][2], g_iup_5d[ix][3], g_idn_5d[ix][3]);
    }}}}}
  }
  fflush(stdout);
*/
/*
  if(g_cart_id==0) {
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt_5d[is][x0][x1][x2][x3];      
      fprintf(stdout, "%6d%3d%3d%3d%3d%6d%6d\n", ix, x0, x1, x2, x3, 
        g_lexic2eo_5d[ix], g_lexic2eot_5d[ix] );      
    }}}}
  }
*/
}

int init_geometry_5d(void) {

  int ix = 0;
  unsigned int V, V5;
  int dx = 0, dy = 0;

  VOLUME         = T*LX*LY*LZ;
  VOLUMEPLUSRAND = VOLUME;
  RAND           = 0;
  EDGES          = 0;

#ifdef MPI
  RAND           += 2*LX*LY*LZ;
  VOLUMEPLUSRAND += 2*LX*LY*LZ;

#if defined PARALLELTX || defined PARALLELTXY
  RAND           += 2*T*LY*LZ;
  EDGES          +=             4*LY*LZ;
  VOLUMEPLUSRAND += 2*T*LY*LZ + 4*LY*LZ;
  dx = 2;
#endif

#if defined PARALLELTXY
  RAND           += 2*T*LX*LZ;
  EDGES          +=             4*LX*LZ + 4*T*LZ;
  VOLUMEPLUSRAND += 2*T*LX*LZ + 4*LX*LZ + 4*T*LZ;
  dy = 2;
#endif

#endif  /* of ifdef MPI */

  if(g_cart_id==0) fprintf(stdout, "# VOLUME = %d\n# RAND   = %d\n# EDGES  = %d\n# VOLUMEPLUSRAND = %d\n",
    VOLUME, RAND, EDGES, VOLUMEPLUSRAND);

  V = VOLUMEPLUSRAND;
  V5 = L5 * VOLUMEPLUSRAND;

  g_idn_5d  = (int**)calloc(V5, sizeof(int*));
  if((void*)g_idn_5d == NULL) return(1);

  idn_5d    = (int*)calloc(4*V5, sizeof(int));
  if((void*)idn_5d == NULL) return(2);

  g_iup_5d  = (int**)calloc(V5, sizeof(int*));
  if((void*)g_iup_5d==NULL) return(3);

  iup_5d    = (int*)calloc(4*V5, sizeof(int));
  if((void*)iup_5d==NULL) return(4);

  g_ipt_5d  = (int*****)calloc(L5, sizeof(int*));
  if((void*)g_ipt_5d == NULL) return(5);

  ipt_5d___ = (int****)calloc(L5*(T+2), sizeof(int*));
  if((void*)ipt_5d___ == NULL) return(6);

  ipt_5d__  = (int***)calloc(L5*(T+2)*(LX+dx), sizeof(int*));
  if((void*)ipt_5d__ == NULL) return(6);

  ipt_5d_   =  (int**)calloc(L5*(T+2)*(LX+dx)*(LY+dy), sizeof(int*));
  if((void*)ipt_5d_ == NULL) return(7);

  ipt_5d    =  (int*)calloc(L5*(T+2)*(LX+dx)*(LY+dy)*LZ, sizeof(int));
  if((void*)ipt_5d == NULL) return(8);

 
  g_iup_5d[0] = iup_5d;
  g_idn_5d[0] = idn_5d;
  for(ix=1; ix<V5; ix++) {
    g_iup_5d[ix] = g_iup_5d[ix-1] + 4;
    g_idn_5d[ix] = g_idn_5d[ix-1] + 4;
  }

  ipt_5d_[0]   = ipt_5d;
  ipt_5d__[0]  = ipt_5d_;
  ipt_5d___[0] = ipt_5d__;
  g_ipt_5d[0]  = ipt_5d___;
  for(ix=1; ix<L5*(T+2)*(LX+dx)*(LY+dy); ix++) ipt_5d_[ix]     = ipt_5d_[ix-1]     + LZ;

  for(ix=1; ix<L5*(T+2)*(LX+dx);         ix++) ipt_5d__[ix]    = ipt_5d__[ix-1]    + (LY+dy);

  for(ix=1; ix<L5*(T+2);                 ix++) ipt_5d___[ix]   = ipt_5d___[ix-1]   + (LX+dx);

  for(ix=1; ix<L5;                    ix++) g_ipt_5d[ix] = g_ipt_5d[ix-1] + (T+2);


  g_lexic2eo_5d  = (int*)calloc(V5, sizeof(int));
  if(g_lexic2eo_5d  == NULL) return(9);

  g_lexic2eot_5d = (int*)calloc(V5, sizeof(int));
  if(g_lexic2eot_5d == NULL) return(10);

  g_eo2lexic_5d  = (int*)calloc(V5, sizeof(int));
  if(g_eo2lexic_5d  == NULL) return(11);

  g_eot2lexic_5d = (int*)calloc(V5, sizeof(int));
  if(g_eot2lexic_5d == NULL) return(11);

  g_iseven_5d    = (int*)calloc(V5, sizeof(int));
  if(g_iseven_5d    == NULL) return(12);

  g_isevent_5d   = (int*)calloc(V5, sizeof(int));
  if(g_isevent_5d   == NULL) return(13);

  return(0);
}

void free_geometry_5d() {

  free(idn_5d);
  free(iup_5d);
  free(ipt_5d);
  free(ipt_5d_);
  free(ipt_5d__);
  free(ipt_5d___);
  free(g_ipt_5d);
  free(g_idn_5d);
  free(g_iup_5d);
  free(g_lexic2eo_5d);
  free(g_lexic2eot_5d);
  free(g_eo2lexic_5d);
  free(g_eot2lexic_5d);
  free(g_iseven_5d);
  free(g_isevent_5d);
}
