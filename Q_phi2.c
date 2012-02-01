/*********************************************
 * Q_phi2.c
 *
 * Wed Nov 18 09:26:06 CET 2009
 *
 * PURPOSE:
 * - function to recursively calculate 
 *   xi = (H^deg phi)(x) for point source phi
 * TODO:
 * DONE:
 * - tested HPE matrix calculation _fv_eq_hpem_ti_fv
 *   against the stepwise calculation
 * CHANGES:
 *********************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
 * void Hopping_rec(double *xi, double *phi, int xd, int xs, int l, int deg, int *steps)
 *
 * xi	destination (target spinor field at destination point)
 * phi	source (source spinor field at source location)
 * xd	destination lattice site
 * xs	source lattice site
 * l	level  (current exponent of H)
 * deg	degree (final   exponent of H)
 ****************************************************************************/

void Hopping_rec(double *xi, double *phi, int xd, int xs, int l, int deg, int *steps) {
  int steps_new[4];
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double norminv = 1. / (1. + 4. * g_kappa*g_kappa * g_mu*g_mu);
  double *xi_, *U_;

  if(l<deg) {
    /**************************************
     * call Hopping_rec with for two sites
     * at next level l+1
     **************************************/

    _fv_eq_zero(xi);

    /* Negative t-direction. */
    if( abs(steps[0]-1)+abs(steps[1])+abs(steps[2])+abs(steps[3]) <= (deg-l) ) {
      steps_new[0] = steps[0]-1; steps_new[1] = steps[1];
      steps_new[2] = steps[2];   steps_new[3] = steps[3];

      Hopping_rec(spinor1, phi, g_idn[xd][0], xs, l+1, deg, steps_new);

      _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
      _fv_pl_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(g_idn[xd][0], 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Positive t-direction. */
    if( abs(steps[0]+1)+abs(steps[1])+abs(steps[2])+abs(steps[3]) <= (deg-l) ) {
      steps_new[0] = steps[0]+1; steps_new[1] = steps[1];
      steps_new[2] = steps[2];   steps_new[3] = steps[3];

      Hopping_rec(spinor1, phi, g_iup[xd][0], xs, l+1, deg, steps_new);

      _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
      _fv_mi_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(xd, 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Negative x-direction. */
    if( abs(steps[0])+abs(steps[1]-1)+abs(steps[2])+abs(steps[3]) <= (deg-l) ) {
      steps_new[0] = steps[0]; steps_new[1] = steps[1]-1;
      steps_new[2] = steps[2]; steps_new[3] = steps[3];

      Hopping_rec(spinor1, phi, g_idn[xd][1], xs, l+1, deg, steps_new);

      _fv_eq_gamma_ti_fv(spinor2, 1, spinor1);
      _fv_pl_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(g_idn[xd][1], 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Positive x-direction. */
    if( abs(steps[0])+abs(steps[1]+1)+abs(steps[2])+abs(steps[3]) <= (deg-l) ) {
      steps_new[0] = steps[0]; steps_new[1] = steps[1]+1;
      steps_new[2] = steps[2]; steps_new[3] = steps[3];

      Hopping_rec(spinor1, phi, g_iup[xd][1], xs, l+1, deg, steps_new);

      _fv_eq_gamma_ti_fv(spinor2, 1, spinor1);
      _fv_mi_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(xd, 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Negative y-direction. */
    if( abs(steps[0])+abs(steps[1])+abs(steps[2]-1)+abs(steps[3]) <= (deg-l) ) {
      steps_new[0] = steps[0];   steps_new[1] = steps[1];
      steps_new[2] = steps[2]-1; steps_new[3] = steps[3];

      Hopping_rec(spinor1, phi, g_idn[xd][2], xs, l+1, deg, steps_new);

      _fv_eq_gamma_ti_fv(spinor2, 2, spinor1);
      _fv_pl_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(g_idn[xd][2], 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Positive y-direction. */
    if( abs(steps[0])+abs(steps[1])+abs(steps[2]+1)+abs(steps[3]) <= (deg-l) ) {
      steps_new[0] = steps[0];   steps_new[1] = steps[1];
      steps_new[2] = steps[2]+1; steps_new[3] = steps[3];

      Hopping_rec(spinor1, phi, g_iup[xd][2], xs, l+1, deg, steps_new);

      _fv_eq_gamma_ti_fv(spinor2, 2, spinor1);
      _fv_mi_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(xd, 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Negative z-direction. */
    if( abs(steps[0])+abs(steps[1])+abs(steps[2])+abs(steps[3]-1) <= (deg-l) ) {
      steps_new[0] = steps[0]; steps_new[1] = steps[1];
      steps_new[2] = steps[2]; steps_new[3] = steps[3]-1;

      Hopping_rec(spinor1, phi, g_idn[xd][3], xs, l+1, deg, steps_new);

      _fv_eq_gamma_ti_fv(spinor2, 3, spinor1);
      _fv_pl_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(g_idn[xd][3], 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Positive z-direction. */
    if( abs(steps[0])+abs(steps[1])+abs(steps[2])+abs(steps[3]+1) <= (deg-l) ) {
      steps_new[0] = steps[0]; steps_new[1] = steps[1];
      steps_new[2] = steps[2]; steps_new[3] = steps[3]+1;

      Hopping_rec(spinor1, phi, g_iup[xd][3], xs, l+1, deg, steps_new);

      _fv_eq_gamma_ti_fv(spinor2, 3, spinor1);
      _fv_mi_eq_fv(spinor1, spinor2);
 
      U_ = g_gauge_field + _GGI(xd, 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Multiplication with -kappa (hopping parameter repr.) */
    _fv_ti_eq_re(xi, -g_kappa);

    /* apply Inverse of 1+i gamma5 mu */
    _fv_eq_gamma_ti_fv(spinor1, 5, xi);
    _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
    _fv_eq_fv_pl_fv(spinor1, xi, spinor2);
    _fv_eq_fv_ti_re(xi, spinor1, norminv);
 
  }  /* of if l<deg ... */

  if(l==deg) {
    /**************************************
     * apply the Hopping matrix to the
     * source field
     **************************************/

    _fv_eq_zero(xi);

    /* Negative t-direction. */
    if(g_idn[xd][0] == xs) {
  
      _fv_eq_gamma_ti_fv(spinor1, 5, phi);
      _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
      _fv_pl_eq_fv(spinor2, phi);
      _fv_eq_fv_ti_re(spinor1, spinor2, norminv);

      _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
      _fv_pl_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(g_idn[xd][0], 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);

    }

    /* Positive t-direction. */
    if(g_iup[xd][0] == xs) {

      _fv_eq_gamma_ti_fv(spinor1, 5, phi);
      _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
      _fv_pl_eq_fv(spinor2, phi);
      _fv_eq_fv_ti_re(spinor1, spinor2, norminv);

      _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
      _fv_mi_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(xd, 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Negative x-direction. */
    if(g_idn[xd][1] == xs) {

      _fv_eq_gamma_ti_fv(spinor1, 5, phi);
      _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
      _fv_pl_eq_fv(spinor2, phi);
      _fv_eq_fv_ti_re(spinor1, spinor2, norminv);

      _fv_eq_gamma_ti_fv(spinor2, 1, spinor1);
      _fv_pl_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(g_idn[xd][1], 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Positive x-direction. */
    if(g_iup[xd][1] == xs) {

      _fv_eq_gamma_ti_fv(spinor1, 5, phi);
      _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
      _fv_pl_eq_fv(spinor2, phi);
      _fv_eq_fv_ti_re(spinor1, spinor2, norminv);


      _fv_eq_gamma_ti_fv(spinor2, 1, spinor1);
      _fv_mi_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(xd, 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);

    }

    /* Negative y-direction. */
    if(g_idn[xd][2] == xs) {

      _fv_eq_gamma_ti_fv(spinor1, 5, phi);
      _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
      _fv_pl_eq_fv(spinor2, phi);
      _fv_eq_fv_ti_re(spinor1, spinor2, norminv);

      _fv_eq_gamma_ti_fv(spinor2, 2, spinor1);
      _fv_pl_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(g_idn[xd][2], 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Positive y-direction. */
    if(g_iup[xd][2] == xs) {

      _fv_eq_gamma_ti_fv(spinor1, 5, phi);
      _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
      _fv_pl_eq_fv(spinor2, phi);
      _fv_eq_fv_ti_re(spinor1, spinor2, norminv);

      _fv_eq_gamma_ti_fv(spinor2, 2, spinor1);
      _fv_mi_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(xd, 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);

    }

    /* Negative z-direction. */
    if(g_idn[xd][3] == xs) {

      _fv_eq_gamma_ti_fv(spinor1, 5, phi);
      _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
      _fv_pl_eq_fv(spinor2, phi);
      _fv_eq_fv_ti_re(spinor1, spinor2, norminv);

      _fv_eq_gamma_ti_fv(spinor2, 3, spinor1);
      _fv_pl_eq_fv(spinor1, spinor2);

      U_ = g_gauge_field + _GGI(g_idn[xd][3], 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Positive z-direction. */
    if(g_iup[xd][3] == xs) {

      _fv_eq_gamma_ti_fv(spinor1, 5, phi);
      _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
      _fv_pl_eq_fv(spinor2, phi);
      _fv_eq_fv_ti_re(spinor1, spinor2, norminv);

      _fv_eq_gamma_ti_fv(spinor2, 3, spinor1);
      _fv_mi_eq_fv(spinor1, spinor2);
 
      U_ = g_gauge_field + _GGI(xd, 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi, spinor2);
    }

    /* Multiplication with -kappa (hopping parameter repr.) */
    _fv_ti_eq_re(xi, -g_kappa);

    /* apply Inverse of 1+i gamma5 mu */
    _fv_eq_gamma_ti_fv(spinor1, 5, xi);
    _fv_eq_fv_ti_im(spinor2, spinor1, -2.*g_kappa*g_mu);
    _fv_eq_fv_pl_fv(spinor1, xi, spinor2);
    _fv_eq_fv_ti_re(xi, spinor1, norminv);

  }  /* of l==deg */
}


/****************************************************************************
 * void init_trace_coeff()
 *
 ****************************************************************************/

#define MAX_ORD 6

void init_trace_coeff(double **tcf, double **tcb, int ***loop_tab, int deg, int *N) {

  int l, l1, mu, steps[4], steps2[4], count, isign, sid, nloop, ix;
  int *loop, Nloop_max, end_flag;
  int **lpath, dist;
  double spinor1[24], spinor2[24], *sp1, *sp2, *sp3;
  double mutilde = 2.*g_kappa*g_mu;
  double norminv = 1. / (1. + mutilde*mutilde);
  double sigma;
  double kappatodeg=1;

  loop = (int*)malloc(deg*sizeof(int));
  Nloop_max=8; for(l=2; l<=deg; l++) Nloop_max*=8;
  if(g_cart_id==0) fprintf(stdout, "deg=%d, Nloop_max=%d\n", deg, Nloop_max);
  lpath = (int**)malloc(Nloop_max*sizeof(int*));
  for(l=1; l<=deg; l++) kappatodeg *= -g_kappa; kappatodeg*=g_kappa;
  fprintf(stdout, "kappatodeg = %25.16e\n", kappatodeg);

  /********************************************************
   * (1) count the loops 
   ********************************************************/
  nloop = 0;
  for(l=0; l<deg; l++) loop[l]=0;
  for(ix=0; ix<Nloop_max; ix++) {
    steps[0]=0;  steps[1]=0;  steps[2]=0;  steps[3]=0;
    steps[0]++;

    for(l=0; l<deg; l++) {
      if(loop[l]<4) { 
        steps[loop[l]]++;
      } else {
        steps[loop[l]-4]--;
      }
      dist = abs(steps[0]) + abs(steps[1]) + abs(steps[2]) + abs(steps[3]);
/*      fprintf(stdout, "ix=%5d; l=%1d; steps= %1d, %1d, %1d, %1d; loop= %1d, %1d, %1d; dist=%d\n", \
        ix, l, steps[0], steps[1], steps[2], steps[3], loop[0], loop[1], loop[2], dist);
*/
      if(abs(steps[0]) + abs(steps[1]) + abs(steps[2]) + abs(steps[3]) < deg-l ) {
        if(l==deg-1) {
          nloop++;
 /*         fprintf(stdout, "found loop number %d\n", nloop); */
          lpath[nloop-1] = (int*)malloc(deg*sizeof(int));
          for(l1=0; l1<deg; l1++) lpath[nloop-1][l1] = loop[l1];
        }
      } else {
/*        fprintf(stdout, "dist too large ...\n"); */
/*
        l1 = l;
        while( l1>=0 && (loop[l1]+1)%8==0 ) l1--;
        l=l1;
        fprintf(stdout, "\tnew l = %d\n", l);
        loop[l] = (loop[l]+1)%8;
        for(l1=l+1; l1<deg; l1++) loop[l1]=0;
        fprintf(stdout, "\tnew loop = %1d, %1d, %1d\n", loop[0], loop[1], loop[2]); 
        l--;
*/
        break;
      }
    }
/*
    end_flag=1;
    for(l=0; l<deg; l++) end_flag *= (loop[l]==7);
    fprintf(stdout, "\tix=%d; end_flag=%d\n", ix, end_flag);
    if(end_flag==1) break;
*/
    l1=deg-1;
    while( l1>=0 && (loop[l1]=(loop[l1]+1)%8)==0 ) l1--;
/*    fprintf(stdout, "\tnew loop = %1d, %1d, %1d\n", loop[0], loop[1], loop[2]); */
  }  /* of ix */

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
  (*loop_tab)[0] = (int* )malloc(nloop * deg * sizeof(int));
  for(l=1; l<nloop; l++) (*loop_tab)[l] = (*loop_tab)[l-1] + deg;
  for(l=0; l<nloop; l++) memcpy((void*)(*loop_tab)[l], (void*)lpath[l], deg*sizeof(int));
  
  /***************************************************
   * print lpath to stdout
   ***************************************************/
  if(g_cart_id==0) {
    fprintf(stdout, "nloop = %d\n", nloop);
    fprintf(stdout, "\nloop_tab: \n");
    for(l=0; l<nloop; l++) {
      fprintf(stdout, "%3d: ", l);
      for(l1=0; l1<deg-1; l1++) {
        if( (*loop_tab)[l][l1]<4) {
          fprintf(stdout, "\t+%1d", (*loop_tab)[l][l1]);
        } else {
          fprintf(stdout, "\t-%1d", (*loop_tab)[l][l1]-4);
        }
      }
      if( (*loop_tab)[l][deg-1]<4) {
        fprintf(stdout, "\t+%1d\n", (*loop_tab)[l][deg-1]);
      } else {
        fprintf(stdout, "\t-%1d\n", (*loop_tab)[l][deg-1]-4);
      }
    }
  } 

  *N = nloop;
 
  free(loop);
  for(l=0; l<nloop; l++) free(lpath[l]);
  free(lpath);

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
      for(l1=0; l1<deg; l1++) {
        sp3=sp1; sp1=sp2; sp2=sp3;
        mu    = ((*loop_tab)[l][l1]) % 4;
        sigma = - ( 2. * (int)(((*loop_tab)[l][l1]) / 4) - 1. );
        _fv_eq_hpem_ti_fv(sp2, sp1, mu, sigma, mutilde, norminv);
      }
      _fv_eq_hpem_ti_fv(sp1, sp2, 0, +1., mutilde, norminv);
      (*tcb)[2*l  ] += sp1[6*sid  ];
      (*tcb)[2*l+1] += sp1[6*sid+1];

      _fv_eq_zero(spinor1);
      spinor1[6*sid] = 1.;
      sp2 = spinor1; sp1 = spinor2;
      for(l1=deg-1; l1>=0; l1--) {
        sp3=sp1; sp1=sp2; sp2=sp3;
        mu    = ((*loop_tab)[l][l1]) % 4;
        sigma =   ( 2. * (int)(((*loop_tab)[l][l1]) / 4) - 1. );
        _fv_eq_hpem_ti_fv(sp2, sp1, mu, sigma, mutilde, norminv);
      }
      _fv_eq_hpem_ti_fv(sp1, sp2, 0, -1., mutilde, norminv);
      (*tcf)[2*l  ] += sp1[6*sid  ];
      (*tcf)[2*l+1] += sp1[6*sid+1];
    }
    (*tcf)[2*l  ] *=  kappatodeg;
    (*tcf)[2*l+1] *=  kappatodeg;
    (*tcb)[2*l  ] *= -kappatodeg;
    (*tcb)[2*l+1] *= -kappatodeg;
    
  }

  for(l=0; l<nloop; l++) fprintf(stdout, "%4d%25.16e%25.16e%25.16e%25.16e\n", l, 
    (*tcf)[2*l], (*tcf)[2*l+1], (*tcb)[2*l], (*tcb)[2*l+1]);

}

/****************************************************************************
 * void Hopping_iter()
 *
 ****************************************************************************/

void Hopping_iter(double *truf, double *trub, double *tcf, double *tcb, int xd, 
  int mu, int deg, int nloop, int **loop_tab) {

  int l, l1, steps[4], mu1, isign, count, perm[4];
  double U1_[18], U2_[18], *u1, *u2, *u3;
  int sigma, xs, xnew, xold;
  complex w;

  for(l=0; l<4; l++) perm[l] = (l+mu)%4;
  fprintf(stdout, "perm = %2d, %2d, %2d, %2d\n", perm[0], perm[1], perm[2], perm[3]);
  
  xs = g_iup[xd][mu];
  fprintf(stdout, "xs = %d\n", xs);

  for(l=0; l<nloop; l++) {
 
/*    fprintf(stdout, "\nbackward loops:\n"); */

    _cm_eq_id(U1_);
    u2=U1_; u1=U2_;
    xnew=xs;
/*  fprintf(stdout, "before: xnew = %5d\n", xnew); */
    for(l1=0; l1<deg; l1++) {
      u3=u1; u1=u2; u2=u3;
      mu1   = perm[loop_tab[l][l1]%4];
      sigma = -(2*(loop_tab[l][l1]/4) - 1);
      if(sigma==+1) {
        xold=xnew;
        xnew = g_iup[xold][mu1];
/*        fprintf(stdout, "l=%3d, l1=%3d, step=%2d, sigma=%2d; xnew = %5d; applying U_%1d(%4d)^+\n", l, l1, perm[loop_tab[l][l1]%4], sigma, xnew, mu1, xold);
*/
/*        _cm_eq_cm_ti_cm_dag(u2, g_gauge_field+_GGI(xold, mu1), u1); */
        _cm_eq_cm_dag_ti_cm(u2, g_gauge_field+_GGI(xold, mu1), u1);
      } else {
        xold=xnew;
        xnew = g_idn[xold][mu1];
/*        fprintf(stdout, "l=%3d, l1=%3d, step=%2d, sigma=%2d; xnew = %5d, applying U_%1d(%4d)\n", l, l1, perm[loop_tab[l][l1]%4], sigma, xnew, mu1, xnew);
*/
        _cm_eq_cm_ti_cm(u2, g_gauge_field+_GGI(xnew, mu1), u1);
      }
    }
/*    _cm_eq_cm_ti_cm_dag(u1, g_gauge_field+_GGI(xd, mu), u2); */
    _cm_eq_cm_dag_ti_cm(u1, g_gauge_field+_GGI(xd, mu), u2); 
    _co_eq_tr_cm(&w, u1);
/*    fprintf(stdout, "current w = %25.16e +i %25.16e\n", w.re, w.im); */
    _co_pl_eq_co_ti_co((complex*)trub, (complex*)(tcb+2*l), &w);

/*    fprintf(stdout, "\nforward loops:\n"); */
 
    _cm_eq_id(U1_);
    u2=U1_; u1=U2_;
    xnew=xd;
/*    fprintf(stdout, "before: xnew = %5d\n", xnew); */
    for(l1=deg-1; l1>=0; l1--) {
      u3=u1; u1=u2; u2=u3;
      mu1   = perm[loop_tab[l][l1]%4];
      sigma = -(2*(loop_tab[l][l1]/4) - 1);
      if(sigma==-1) {
        xold=xnew;
        xnew = g_iup[xold][mu1];
/*        fprintf(stdout, "l=%3d, l1=%3d, step=%2d, sigma=%2d; xnew = %5d, applying U_%1d(%4d)^+\n", l, l1, perm[loop_tab[l][l1]%4], sigma, xnew, mu1, xold); */
/*        _cm_eq_cm_ti_cm_dag(u2, g_gauge_field+_GGI(xold, mu1), u1); */
        _cm_eq_cm_dag_ti_cm(u2, g_gauge_field+_GGI(xold, mu1), u1);
      } else {
        xold=xnew;
        xnew = g_idn[xold][mu1];
/*        fprintf(stdout, "l=%3d, l1=%3d, step=%2d, sigma=%2d; xnew = %5d, applying U_%1d(%4d)\n", l, l1, perm[loop_tab[l][l1]%4], sigma, xnew, mu1, xnew); */
        _cm_eq_cm_ti_cm(u2, g_gauge_field+_GGI(xnew, mu1), u1);
      }
    }
    _cm_eq_cm_ti_cm(u1, g_gauge_field+_GGI(xd, mu), u2);
    _co_eq_tr_cm(&w, u1);
    _co_pl_eq_co_ti_co((complex*)truf, (complex*)(tcf+2*l), &w);

  }

  fprintf(stdout, "%6d%3d%25.16e%25.16e%25.16e%25.16e\n", xd, mu, truf[0], truf[1], trub[0], trub[1]);

}

/*********************************************
 * functions for testing
 *********************************************/

void test_hpem(void) {

  int sid, mu, i, j;
  double spinor1[24], spinor2[24], spinor3[24], spinor4[24];
  double mutilde = 2.*g_kappa*g_mu;
/*  double mutilde = 0.; */
  double norminv = 1. / (1. + mutilde*mutilde);
/*  double norminv = 1.; */
  int sigma, count=-1;

  for(sigma=-1; sigma<2; sigma+=2) {
  for(mu=0; mu<4; mu++) {
    for(sid=0; sid<24; sid++) {
      _fv_eq_zero(spinor1);
      spinor1[sid] = 1.;

      /* 1st way */
      _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
      _fv_eq_fv_ti_im(spinor3, spinor2, -mutilde);
      _fv_eq_fv_pl_fv(spinor2, spinor1, spinor3);
      _fv_ti_eq_re(spinor2, norminv);
      _fv_eq_gamma_ti_fv(spinor3, mu, spinor2);
      if(sigma==-1) {
        _fv_mi_eq_fv(spinor2, spinor3);
      } else {
        _fv_pl_eq_fv(spinor2, spinor3);
      }
  
      /* 2nd way */
      _fv_eq_hpem_ti_fv(spinor4, spinor1, mu, sigma, mutilde, norminv);

      fprintf(stdout, "#----------------------------------------------------------------\n");
      for(i=0; i<12; i++) fprintf(stdout, "%5d%3d%2d%3d\t%25.16e%25.16e%25.16e%25.16e\n", \
        ++count, sigma, mu, sid, spinor2[2*i], spinor2[2*i+1], spinor4[2*i], spinor4[2*i+1]);
  
    }

  }
  }
}

void test_cm_eq_cm_dag_ti_cm(void) {

  int i, j;
  double U1[18], U2[18], U3[18];

  _cm_eq_cm_dag_ti_cm(U1, g_gauge_field+_GGI(50,3), g_gauge_field+_GGI(30,2));

  _cm_eq_cm_dag(U2, g_gauge_field+_GGI(50,3))
  _cm_eq_cm_ti_cm(U3, U2, g_gauge_field+_GGI(30,2));

  for(i=0; i<9; i++) {
    fprintf(stdout, "i=%d; %25.16e%25.16e%25.16e%25.16e\n", i, U1[2*i], U1[2*i+1], U3[2*i], U3[2*i+1]);
  }

  
  _cm_eq_cm_dag_ti_cm(U1, g_gauge_field+_GGI(50,3), g_gauge_field+_GGI(50,3));
  for(i=0; i<9; i++) {
    fprintf(stdout, "i=%d; %25.16e%25.16e\n", i, U1[2*i], U1[2*i+1]);
  }
}



  

