/*********************************************
 * Q_phi2_red.c
 *
 * Wed Nov 18 09:26:06 CET 2009
 *
 * PURPOSE:
 * - functions to calculate the n-th order contribution
 *   to the HPE with exact iterative method
 * - functions to calculate the n-th order contribution
 *   to the HPE with Monte-Carlo-Metho
 * TODO:
 * - finish+test numerical method
 * - implement+test MC-method
 * DONE:
 * CHANGES:
 *********************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include "cvc_complex.h"
#include "global.h"
#include "cvc_linalg.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "Q_phi.h"

/****************************************************************************
 * void init_trace_coeff_red()
 ****************************************************************************/

void init_trace_coeff_red(double **tcf, double **tcb, int ***loop_tab, int ***sigma_tab, int ***shift_start, int deg, int *N, int mudir) {

  int l, l1, mu, steps[4], steps2[4], count, isign, sid, nloop, ix;
  int *loop, Nloop_max, end_flag;
  int **lpath, **spath;
  double spinor1[24], spinor2[24], *sp1, *sp2, *sp3;
  double mutilde = 2.*g_kappa*g_mu;
  double norminv = 1. / (1. + mutilde*mutilde);
  double sigma;
  double kappatodeg=1;

  loop = (int*)malloc(deg*sizeof(int));
  Nloop_max=8; for(l=1; l<deg; l++) Nloop_max*=8;
  if(g_cart_id==0) fprintf(stdout, "deg=%d, Nloop_max=%d\n", deg, Nloop_max);
  lpath = (int**)malloc(Nloop_max*sizeof(int*));
  spath = (int**)malloc(Nloop_max*sizeof(int*));
  for(l=1; l<=deg; l++) kappatodeg *= -g_kappa; kappatodeg*=g_kappa;
  fprintf(stdout, "kappatodeg = %25.16e\n", kappatodeg);

  /********************************************************
   * (1) count the loops 
   ********************************************************/
  nloop = 0;
  for(l=0; l<deg; l++) loop[l]=0;
  for(ix=0; ix<Nloop_max; ix++) {
    steps[0]=0;  steps[1]=0;  steps[2]=0;  steps[3]=0;
    steps[mudir]++;
/*    fprintf(stdout, "ix = %d\n", ix); */
    for(l=0; l<deg; l++) {
      if(loop[l]<4) { 
        steps[loop[l]]++;
      } else {
        steps[loop[l]-4]--;
      }
      if(abs(steps[0]) + abs(steps[1]) + abs(steps[2]) + abs(steps[3]) < deg-l ) {
        if(l==deg-1) {
          nloop++;
/*          fprintf(stdout, "found new loop number %d\n", nloop); */
          lpath[nloop-1] = (int*)malloc((deg+1)*sizeof(int));
          spath[nloop-1] = (int*)malloc((deg+1)*sizeof(int));
          for(l1=0; l1<deg; l1++) lpath[nloop-1][l1+1] = loop[l1]%4;
          for(l1=0; l1<deg; l1++) spath[nloop-1][l1+1] = 1 - 2*(loop[l1]/4);
          lpath[nloop-1][0] = mudir;
          spath[nloop-1][0] = 1;
        }
      } else {
        break;
      }
    }
    l1=deg-1;
    while( l1>=0 && (loop[l1]=(loop[l1]+1)%8)==0 ) l1--;
  } 

/*
  if(g_cart_id==0) {
    fprintf(stdout, "nloop = %d\n", nloop);
    fprintf(stdout, "\nlpath table: \n");
    for(l=0; l<nloop; l++) {
      fprintf(stdout, "%3d: ", l);
      for(l1=0; l1<deg-1; l1++) {
        if(lpath[l][l1]<4) {
          fprintf(stdout, "\t+%1d", lpath[l][l1]);
        } else {
          fprintf(stdout, "\t-%1d", lpath[l][l1]-4);
        }
      }
      if(lpath[l][deg-1]<4) {
        fprintf(stdout, "\t+%1d\n", lpath[l][deg-1]);
      } else {
        fprintf(stdout, "\t-%1d\n", lpath[l][deg-1]-4);
      }
    }
  } 
*/
 
  /***************************************************
   * copy lpath to loop_tab
   ***************************************************/
  *loop_tab    = (int**)malloc(nloop *       sizeof(int*));
  (*loop_tab)[0] = (int* )malloc(nloop * (deg+2) * sizeof(int));
  for(l=1; l<nloop; l++) (*loop_tab)[l] = (*loop_tab)[l-1] + (deg+2);
  for(l=0; l<nloop; l++) memcpy((void*)(*loop_tab)[l], (void*)lpath[l], (deg+1)*sizeof(int));
  for(l=0; l<nloop; l++) (*loop_tab)[l][deg+1] = -1;

  *sigma_tab    = (int**)malloc(nloop *       sizeof(int*));
  (*sigma_tab)[0] = (int* )malloc(nloop * (deg+2) * sizeof(int));
  for(l=1; l<nloop; l++) (*sigma_tab)[l] = (*sigma_tab)[l-1] + (deg+2);
  for(l=0; l<nloop; l++) memcpy((void*)(*sigma_tab)[l], (void*)spath[l], (deg+1)*sizeof(int));

  *shift_start = (int**)malloc(nloop * sizeof(int*));
  (*shift_start)[0] = (int*)malloc(nloop*4*sizeof(int));
  for(l=1; l<nloop; l++) (*shift_start)[l] = (*shift_start)[l-1] + 4;
  
  /***************************************************
   * print loop_tab to stdout
   ***************************************************/
/*
  if(g_cart_id==0) {
    fprintf(stdout, "*********************************\n* mu = %d\n*********************************\n\n", mudir);
    fprintf(stdout, "nloop = %d\n", nloop);
    fprintf(stdout, "\nloop_tab: \n");
    for(l=0; l<nloop; l++) {
      fprintf(stdout, "%3d: ", l);
      for(l1=0; l1<deg; l1++) {
        if((*sigma_tab)[l][l1]==+1) {
          fprintf(stdout, "\t+%1d", (*loop_tab)[l][l1]);
        } else {
          fprintf(stdout, "\t-%1d", (*loop_tab)[l][l1]);
        }
      }
      if((*sigma_tab)[l][deg]==+1) {
        fprintf(stdout, "\t+%1d\n", (*loop_tab)[l][deg]);
      } else {
        fprintf(stdout, "\t-%1d\n", (*loop_tab)[l][deg]);
      }
    }
  }
*/
  *N = nloop;

  /****************************
   * check the loops
   ****************************/
/*
  for(l=0; l<nloop; l++) {
   mu=0;
   for(l1=0; l1<=deg; l1++) mu += (*sigma_tab)[l][l1]*(*loop_tab)[l][l1];
   fprintf(stdout, "checked loop no. %6d: sum = %6d\n", l, mu);
  }
*/
 
  free(loop);
  for(l=0; l<nloop; l++) free(lpath[l]);
  for(l=0; l<nloop; l++) free(spath[l]);
  free(lpath);
  free(spath);

  /********************************************************
   * (2) tcf/b 
   * _ATTENTION_ isign gives the direction on the lattice,
   *             i.e. +/-1 means 1 -/+ gamma_mu
   * - tcf/b have real and imaginary part   
   ********************************************************/
  *tcf = (double*)malloc(2*nloop*sizeof(double));
  *tcb = (double*)malloc(2*nloop*sizeof(double));
  
  for(l=0; l<nloop; l++) {
    (*tcb)[2*l  ] = 0.; (*tcb)[2*l+1] = 0.;
    (*tcf)[2*l  ] = 0.; (*tcf)[2*l+1] = 0.;
    for(sid=0; sid<4; sid++) {
      _fv_eq_zero(spinor1);
      spinor1[6*sid] = 1.;
      sp2 = spinor1; sp1 = spinor2;
      for(l1=0; l1<=deg; l1++) {
        sp3=sp1; sp1=sp2; sp2=sp3;
        _fv_eq_hpem_ti_fv(sp2, sp1, (*loop_tab)[l][l1], (*sigma_tab)[l][l1], mutilde, norminv);
      }
      (*tcb)[2*l  ] += sp2[6*sid  ];
      (*tcb)[2*l+1] += sp2[6*sid+1];

      _fv_eq_zero(spinor1);
      spinor1[6*sid] = 1.;
      sp2 = spinor1; sp1 = spinor2;
      for(l1=deg; l1>=0; l1--) {
        sp3=sp1; sp1=sp2; sp2=sp3;
        _fv_eq_hpem_ti_fv(sp2, sp1, (*loop_tab)[l][l1], (*sigma_tab)[l][l1], mutilde, norminv);
      }
      (*tcf)[2*l  ] += sp2[6*sid  ];
      (*tcf)[2*l+1] += sp2[6*sid+1];
    }
    (*tcf)[2*l  ] *=  kappatodeg;
    (*tcf)[2*l+1] *=  kappatodeg;
    (*tcb)[2*l  ] *= -kappatodeg;
    (*tcb)[2*l+1] *= -kappatodeg;
    
  }
/*
  for(l=0; l<nloop; l++) fprintf(stdout, "%4d%25.16e%25.16e%25.16e%25.16e\n", l, 
    (*tcf)[2*l], (*tcf)[2*l+1], (*tcb)[2*l], (*tcb)[2*l+1]);
*/

}

/****************************************************************************
 * void Hopping_iter_red()
 *
 ****************************************************************************/
void Hopping_iter_red(double *truf, double *trub, double *tcf, double *tcb, int xd, 
  int mu, int deg, int nloop, int **loop_tab, int **sigma_tab, int **shift_start) {

  int l, l1, isign, perm[4], perm_inv[4];
  int xd0, xd1, xd2, xd3;
  int y0, y1, y2, y3;
  double U1_[18], U2_[18], *u1, *u2, *u3;
  int xnew, xold;
  int *xloc[2], **xdir[2];
  complex w;

  xd0 = xd / (LX*LY*LZ);
  xd1 = ( xd % (LX*LY*LZ) ) / (LY*LZ);
  xd2 = ( xd % (LY*LZ) ) / LZ;
  xd3 = xd % LZ;

  xdir[0] = g_idn;
  xdir[1] = g_iup;
  xloc[0] = &xnew;
  xloc[1] = &xold;

/*  fprintf(stdout, "# xd = %d = (%3d,%3d,%3d,%3d)\n", xd, xd0, xd1, xd2, xd3); */
/*
  for(l=0; l<4; l++) perm[l]     = (l+mu)%4;
  for(l=0; l<4; l++) perm_inv[l] = (l-mu+4)%4;
  xd = xd;
*/

  for(l=0; l<nloop; l++) {
    _cm_eq_id(U1_);
    u2=U1_; u1=U2_;
/*
    y0 = ( xd0+shift_start[l][perm_inv[0]] + T  ) % T;
    y1 = ( xd1+shift_start[l][perm_inv[1]] + LX ) % LX;
    y2 = ( xd2+shift_start[l][perm_inv[2]] + LY ) % LY;
    y3 = ( xd3+shift_start[l][perm_inv[3]] + LZ ) % LZ;
*/
    y0 = ( xd0+shift_start[l][0] + T  ) % T;
    y1 = ( xd1+shift_start[l][1] + LX ) % LX;
    y2 = ( xd2+shift_start[l][2] + LY ) % LY;
    y3 = ( xd3+shift_start[l][3] + LZ ) % LZ;
    xnew = g_ipt[y0][y1][y2][y3];
    for(l1=0; loop_tab[l][l1] != -1; l1++) {
      u3=u1; u1=u2; u2=u3;
      isign = (sigma_tab[l][l1]+1)/2;
      xold=xnew;
/*      xnew = (xdir[isign])[xold][perm[loop_tab[l][l1]]]; */
      xnew = (xdir[isign])[xold][loop_tab[l][l1]];
      if(sigma_tab[l][l1]==+1) 
/*        {_cm_eq_cm_dag_ti_cm(u2, g_gauge_field+_GGI(*(xloc[isign]), perm[loop_tab[l][l1]]), u1);} */
        {_cm_eq_cm_dag_ti_cm(u2, g_gauge_field+_GGI(*(xloc[isign]), loop_tab[l][l1]), u1);}
      else
/*        {_cm_eq_cm_ti_cm(u2, g_gauge_field+_GGI(*(xloc[isign]), perm[loop_tab[l][l1]]), u1);} */
        {_cm_eq_cm_ti_cm(u2, g_gauge_field+_GGI(*(xloc[isign]), loop_tab[l][l1]), u1);}
    }
    _co_eq_tr_cm(&w, u2);
    _co_pl_eq_co_ti_co((complex*)trub, (complex*)(tcb+2*l), &w);

    w.im = -w.im;
    _co_pl_eq_co_ti_co((complex*)truf, (complex*)(tcf+2*l), &w);
  }
}

/****************************************************************************
 * void reduce_loop_tab()
 *
 ****************************************************************************/

void reduce_loop_tab(int **loop_tab, int **sigma_tab, int **shift_start, int deg, int nloop) {

  int l, l1, lpath[HPE_MAX_ORDER+2], spath[HPE_MAX_ORDER+2], \
      length, next, changed=1, start, end, i; 
  double ratime, retime;


  for(l=0; l<nloop; l++)
    for(l1=0; l1<4; l1++) shift_start[l][l1] = 0;

  fprintf(stdout, "deg = %d; nloop = %d\n", deg, nloop);
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  for(l=0; l<nloop; l++) {

/*    fprintf(stdout, "#------------------------\n");
    fprintf(stdout, "# reducing loop number %d\n", l); */
    length  = deg+1;
    changed = 1;
    start   = 0;
    end     = deg;
    memcpy((void*)lpath, (void*)(loop_tab[l]), (deg+1)*sizeof(int));
    memcpy((void*)spath, (void*)(sigma_tab[l]), (deg+1)*sizeof(int));

    while(length>0 && changed==1) {
      changed=0;
/*      fprintf(stdout, "# start=%d\tend=%d\tlength=%d\tchanged=%d\n", start, end, length, changed); */
      if(lpath[start] == lpath[end] && sigma_tab[l][start] == -sigma_tab[l][end]) {
        length -= 2;
        shift_start[l][lpath[start]] += sigma_tab[l][start];
        lpath[start] = -2;
        lpath[end]   = -2;
        while(lpath[start] ==-2 && start < end)   start++; 
        while(lpath[end]   ==-2 && end   > start) end--;
        changed=1;
/*        fprintf(stdout, "#\t new start=%d\tnew end=%d\tnew length=%d\tnew changed=%d\n", start, end, length, changed); */
        if(length==0) {
          lpath[0] = -1;
          break;
        }
      }
      for(l1=start; l1<end;) {
        next = l1+1;
        while(lpath[next]==-2 && next<end) next++;
/*        fprintf(stdout, "#\t\tl1=%d; next=%d\n", l1, next); */
        if(lpath[l1] == lpath[next] && sigma_tab[l][l1] == -sigma_tab[l][next]) {
          if(l1==start && next==end) {
            shift_start[l][lpath[l1]] += sigma_tab[l][l1];
          }
          lpath[l1]   = -2;
          lpath[next] = -2;
          length     -=  2;
          changed = 1;
          if(length==0) {
            lpath[0] = -1;
            break;
          }
        }
        l1 = next; while(lpath[l1]==-2 && l1<end) l1++;
/*        fprintf(stdout, "#\t\tnew l1=%d; new next=%d\n", l1, next); */
      }
      while(lpath[start]==-2 && start<end) start++;
      while(lpath[end]==-2 && end>start) end--;
/*      fprintf(stdout, "#\tend of while loop for l=%d: new start=%d; new end=%d\n", l, start, end); */
    }

/*    fprintf(stdout, "# final lpath:");
    for(l1=0; l1<deg; l1++) fprintf(stdout, "\t(%d, %d)", lpath[l1], sigma_tab[l][l1]);
    fprintf(stdout, "\t(%d, %d)\n", lpath[deg], sigma_tab[l][deg]); */
    l1=start;
/*    fprintf(stdout, "# transcription for loop %d: start=%d; length=%d\n", l, start, length); */

    for(i=0; i<length;) {
      loop_tab[l][i] = lpath[l1];
      spath[i] = sigma_tab[l][l1];
      i++; l1++;
      while(lpath[l1]==-2 && l1<end) l1++;
    }
    loop_tab[l][length] = -1;
    memcpy((void*)sigma_tab[l], (void*)spath, length*sizeof(int));
    sigma_tab[l][length] = 0;
  }

  /***************************************************
   * print reduced loop_tab to stdout
   ***************************************************/
/*
  if(g_cart_id==0) {
    fprintf(stdout, "\n\n--------------------------------\n");
    fprintf(stdout, "--------------------------------\n");
    fprintf(stdout, "\treduced loop_tab: \n");
    for(l=0; l<nloop; l++) {
      fprintf(stdout, "%3d: ", l);
      for(l1=0; l1<=deg && loop_tab[l][l1]!=-1; l1++) {
        if((sigma_tab)[l][l1]==+1) {
          fprintf(stdout, "\t+%1d", (loop_tab)[l][l1]);
        } else {
          fprintf(stdout, "\t-%1d", (loop_tab)[l][l1]);
        }
      }
      fprintf(stdout, "\tend of loop (%d, %d)\t", (loop_tab)[l][l1], sigma_tab[l][l1]);
      fprintf(stdout, "with shift = (%d,%d,%d,%d)\n", shift_start[l][0], shift_start[l][1], 
        shift_start[l][2], shift_start[l][3]);
    }
  }
*/

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time to reduce loops.: %e seconds\n", retime-ratime);


}




/****************************************************************************
 * void init_lvc_trace_coeff_red()
 ****************************************************************************/

void init_lvc_trace_coeff_red(double **tcf, int ***loop_tab, int ***sigma_tab, int ***shift_start, int deg, int *N, int mudir) {

  int l, l1, mu, steps[4], count, isign, sid, nloop, ix;
  int *loop, Nloop_max, end_flag;
  int **lpath, **spath;
  double spinor1[24], spinor2[24], *sp1, *sp2, *sp3, spinor3[24];
  double mutilde = 2.*g_kappa*g_mu;
  double norminv = 1. / (1. + mutilde*mutilde);
  double sigma;
  double kappatodeg=1;

  loop = (int*)malloc(deg*sizeof(int));
  Nloop_max=8; for(l=1; l<deg; l++) Nloop_max*=8;
  if(g_cart_id==0) fprintf(stdout, "deg=%d, Nloop_max=%d\n", deg, Nloop_max);
  lpath = (int**)malloc(Nloop_max*sizeof(int*));
  spath = (int**)malloc(Nloop_max*sizeof(int*));
  for(l=1; l<=deg; l++) kappatodeg *= g_kappa; kappatodeg*=2.*g_kappa;
  fprintf(stdout, "kappatodeg = %25.16e\n", kappatodeg);

  /********************************************************
   * (1) count the loops 
   ********************************************************/
  nloop = 0;
  for(l=0; l<deg; l++) loop[l]=0;
  for(ix=0; ix<Nloop_max; ix++) {
    steps[0]=0;  steps[1]=0;  steps[2]=0;  steps[3]=0;
/*    fprintf(stdout, "ix = %d\n", ix); */
    for(l=0; l<deg; l++) {
      if(loop[l]<4) { 
        steps[loop[l]]++;
      } else {
        steps[loop[l]-4]--;
      }
      if(abs(steps[0]) + abs(steps[1]) + abs(steps[2]) + abs(steps[3]) < deg-l ) {
        if(l==deg-1) {
          nloop++;
/*          fprintf(stdout, "found new loop number %d\n", nloop); */
          lpath[nloop-1] = (int*)malloc(deg*sizeof(int));
          spath[nloop-1] = (int*)malloc(deg*sizeof(int));
          for(l1=0; l1<deg; l1++) lpath[nloop-1][l1] = loop[l1]%4;
          for(l1=0; l1<deg; l1++) spath[nloop-1][l1] = 1 - 2*(loop[l1]/4);
        }
      } else {
        break;
      }
    }
    l1=deg-1;
    while( l1>=0 && (loop[l1]=(loop[l1]+1)%8)==0 ) l1--;
  } 

/*
  if(g_cart_id==0) {
    fprintf(stdout, "nloop = %d\n", nloop);
    fprintf(stdout, "\nlpath table: \n");
    for(l=0; l<nloop; l++) {
      fprintf(stdout, "%3d: ", l);
      for(l1=0; l1<deg-1; l1++) {
        if(lpath[l][l1]<4) {
          fprintf(stdout, "\t+%1d", lpath[l][l1]);
        } else {
          fprintf(stdout, "\t-%1d", lpath[l][l1]-4);
        }
      }
      if(lpath[l][deg-1]<4) {
        fprintf(stdout, "\t+%1d\n", lpath[l][deg-1]);
      } else {
        fprintf(stdout, "\t-%1d\n", lpath[l][deg-1]-4);
      }
    }
  } 
*/
 
  /***************************************************
   * copy lpath to loop_tab
   ***************************************************/
  *loop_tab = (int**)malloc(nloop * sizeof(int*));
  (*loop_tab)[0] = (int* )malloc(nloop * (deg+1) * sizeof(int));
  for(l=1; l<nloop; l++) (*loop_tab)[l] = (*loop_tab)[l-1] + (deg+1);
  for(l=0; l<nloop; l++) memcpy((void*)(*loop_tab)[l], (void*)lpath[l], deg*sizeof(int));
  for(l=0; l<nloop; l++) (*loop_tab)[l][deg] = -1;

  *sigma_tab = (int**)malloc(nloop * sizeof(int*));
  (*sigma_tab)[0] = (int* )malloc(nloop * (deg+1) * sizeof(int));
  for(l=1; l<nloop; l++) (*sigma_tab)[l] = (*sigma_tab)[l-1] + (deg+1);
  for(l=0; l<nloop; l++) memcpy((void*)(*sigma_tab)[l], (void*)spath[l], deg*sizeof(int));

  *shift_start = (int**)malloc(nloop * sizeof(int*));
  (*shift_start)[0] = (int*)malloc(nloop*4*sizeof(int));
  for(l=1; l<nloop; l++) (*shift_start)[l] = (*shift_start)[l-1] + 4;
  
  /***************************************************
   * print loop_tab to stdout
   ***************************************************/
/*
  if(g_cart_id==0) {
    fprintf(stdout, "*********************************\n* mu = %d\n*********************************\n\n", mudir);
    fprintf(stdout, "nloop = %d\n", nloop);
    fprintf(stdout, "\nloop_tab: \n");
    for(l=0; l<nloop; l++) {
      fprintf(stdout, "%3d: ", l);
      for(l1=0; l1<deg-1; l1++) {
        if((*sigma_tab)[l][l1]==+1) {
          fprintf(stdout, "\t+%1d", (*loop_tab)[l][l1]);
        } else {
          fprintf(stdout, "\t-%1d", (*loop_tab)[l][l1]);
        }
      }
      if((*sigma_tab)[l][deg-1]==+1) {
        fprintf(stdout, "\t+%1d\n", (*loop_tab)[l][deg-1]);
      } else {
        fprintf(stdout, "\t-%1d\n", (*loop_tab)[l][deg-1]);
      }
    }
  }
*/

  *N = nloop;

  /****************************
   * check the loops
   ****************************/
/*
  for(l=0; l<nloop; l++) {
   mu=0;
   for(l1=0; l1<deg; l1++) mu += (*sigma_tab)[l][l1]*(*loop_tab)[l][l1];
   fprintf(stdout, "# checked loop no. %6d: sum = %6d\n", l, mu);
  }
*/
 
  free(loop);
  for(l=0; l<nloop; l++) free(lpath[l]);
  for(l=0; l<nloop; l++) free(spath[l]);
  free(lpath);
  free(spath);

  /********************************************************
   * (2) tcf 
   ********************************************************/
  *tcf = (double*)malloc(2*nloop*sizeof(double));
  
  for(l=0; l<nloop; l++) {
    (*tcf)[2*l  ] = 0.; (*tcf)[2*l+1] = 0.;
    for(sid=0; sid<4; sid++) {
      _fv_eq_zero(spinor1);
      spinor1[6*sid] = 1.;
      sp2 = spinor1; sp1 = spinor2;
      for(l1=0; l1<deg; l1++) {
        sp3=sp1; sp1=sp2; sp2=sp3;
        _fv_eq_hpem_ti_fv(sp2, sp1, (*loop_tab)[l][l1], (*sigma_tab)[l][l1], mutilde, norminv);
      }

      _fv_eq_gamma_ti_fv(sp1, 5, sp2);
      _fv_eq_fv_ti_im(spinor3, sp1, -2.*g_kappa*g_mu);
      _fv_eq_fv_pl_fv(sp2, spinor3, sp2);
      _fv_ti_eq_re(sp2, 1./(1.+4.*g_kappa*g_kappa*g_mu*g_mu));

      _fv_eq_gamma_ti_fv(sp1, mudir, sp2);
      (*tcf)[2*l  ] += sp1[6*sid  ];
      (*tcf)[2*l+1] += sp1[6*sid+1];
    }
    (*tcf)[2*l  ] *= -kappatodeg;
    (*tcf)[2*l+1] *= -kappatodeg;
    
  }

/*
  if(g_cart_id==0) {
    fprintf(stdout, "# the trace coeff. for deg=%d and mu=%d\n", deg, mudir);
    for(l=0; l<nloop; l++) fprintf(stdout, "%4d%25.16e%25.16e\n", l, (*tcf)[2*l], (*tcf)[2*l+1]);
  }
*/

}

/****************************************************************************
 * void Hopping_lvc_iter_red()
 *
 ****************************************************************************/
void Hopping_lvc_iter_red(double *truf, double *tcf, int xd, 
  int mu, int deg, int nloop, int **loop_tab, int **sigma_tab, int **shift_start) {

  int l, l1, isign;
  int xd0, xd1, xd2, xd3;
  int y0, y1, y2, y3;
  double U1_[18], U2_[18], *u1, *u2, *u3;
  int xnew, xold;
  int *xloc[2], **xdir[2];
  complex w;

  xd0 = xd / (LX*LY*LZ);
  xd1 = ( xd % (LX*LY*LZ) ) / (LY*LZ);
  xd2 = ( xd % (LY*LZ) ) / LZ;
  xd3 = xd % LZ;

  xdir[0] = g_idn;
  xdir[1] = g_iup;
  xloc[0] = &xnew;
  xloc[1] = &xold;

/*  fprintf(stdout, "# xd = %d = (%3d,%3d,%3d,%3d)\n", xd, xd0, xd1, xd2, xd3); */

  for(l=0; l<nloop; l++) {
    _cm_eq_id(U1_);
    u2=U1_; u1=U2_;
    y0 = ( xd0+shift_start[l][0] + T  ) % T;
    y1 = ( xd1+shift_start[l][1] + LX ) % LX;
    y2 = ( xd2+shift_start[l][2] + LY ) % LY;
    y3 = ( xd3+shift_start[l][3] + LZ ) % LZ;
    xnew = g_ipt[y0][y1][y2][y3];
    for(l1=0; loop_tab[l][l1] != -1; l1++) {
      u3=u1; u1=u2; u2=u3;
      isign = (sigma_tab[l][l1]+1)/2;
      xold=xnew;
      xnew = (xdir[isign])[xold][loop_tab[l][l1]];
      if(sigma_tab[l][l1]==+1) 
        {_cm_eq_cm_dag_ti_cm(u2, g_gauge_field+_GGI(*(xloc[isign]), loop_tab[l][l1]), u1);}
      else
        {_cm_eq_cm_ti_cm(u2, g_gauge_field+_GGI(*(xloc[isign]), loop_tab[l][l1]), u1);}
    }
    _co_eq_tr_cm(&w, u2);
    _co_pl_eq_co_ti_co((complex*)truf, (complex*)(tcf+2*l), &w);
  }
}

/****************************************************************************
 * void Hopping_iter_mc_red()
 *
 ****************************************************************************/
void Hopping_iter_mc_red(double *truf, double *tcf, int xd,
  int mu, int deg, int nloop, int **loop_tab, int **sigma_tab, int **shift_start) {

  int l, l1, isign;
  int xd0, xd1, xd2, xd3;
  int y0, y1, y2, y3;
  double U1_[18], U2_[18], *u1, *u2, *u3;
  int xnew, xold;
  int *xloc[2], **xdir[2];
  complex w, w2;

  xd0 = xd / (LX*LY*LZ);
  xd1 = ( xd % (LX*LY*LZ) ) / (LY*LZ);
  xd2 = ( xd % (LY*LZ) ) / LZ;
  xd3 = xd % LZ;

  xdir[0] = g_idn;
  xdir[1] = g_iup;
  xloc[0] = &xnew;
  xloc[1] = &xold;

/*  fprintf(stdout, "# xd = %d = (%3d,%3d,%3d,%3d)\n", xd, xd0, xd1, xd2, xd3); */
  for(l=0; l<nloop; l++) {
    _cm_eq_id(U1_);
    u2=U1_; u1=U2_;
    y0 = ( xd0+shift_start[l][0] + T  ) % T;
    y1 = ( xd1+shift_start[l][1] + LX ) % LX;
    y2 = ( xd2+shift_start[l][2] + LY ) % LY;
    y3 = ( xd3+shift_start[l][3] + LZ ) % LZ;
    xnew = g_ipt[y0][y1][y2][y3];
    for(l1=0; loop_tab[l][l1] != -1; l1++) {
      u3=u1; u1=u2; u2=u3;
      isign = (sigma_tab[l][l1]+1)/2;
      xold=xnew;
      xnew = (xdir[isign])[xold][loop_tab[l][l1]];
      if(sigma_tab[l][l1]==+1)
        {_cm_eq_cm_dag_ti_cm(u2, g_gauge_field+_GGI(*(xloc[isign]), loop_tab[l][l1]), u1);}
      else
        {_cm_eq_cm_ti_cm(u2, g_gauge_field+_GGI(*(xloc[isign]), loop_tab[l][l1]), u1);}
    }
    _co_eq_tr_cm(&w, u2);
    w.re = -w.re;
    _co_eq_co_ti_co(&w2, (complex*)(tcf+2*l), &w);
    truf[1] += 2. * w2.im;
  }
}

