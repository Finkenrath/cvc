/********************
 * Q_h_phi.c
 ********************/

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
#include "Q_h_phi.h"

/***********************
 * Q applied to c-quark
 *
 ***********************/
void c_Q_h_phi(double *xi_c, double *phi_c, double *phi_s) {
  int ix;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_c_=NULL, *phi_c_=NULL, *phi_s_=NULL, *U_=NULL;

  double _1_2_kappa = 0.5 / g_kappa;

  for(ix = 0; ix < VOLUME; ix++) {

      index_s = ix;

      xi_c_ = xi_c + _GSI(index_s);

      _fv_eq_zero(xi_c_);

      /* Negative t-direction. */
      phi_c_ = phi_c + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_c_);
      _fv_pl_eq_fv(spinor1, phi_c_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][0], 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_c_, spinor2);

      /* Positive t-direction. */
      phi_c_ = phi_c + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_c_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_c_);

      U_ = g_gauge_field + _GGI(index_s, 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_c_, spinor2);

      /* Negative x-direction. */
      phi_c_ = phi_c + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_c_);
      _fv_pl_eq_fv(spinor1, phi_c_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][1], 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_c_, spinor2);

      /* Positive x-direction. */
      phi_c_ = phi_c + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_c_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_c_);

      U_ = g_gauge_field + _GGI(index_s, 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_c_, spinor2);


      /* Negative y-direction. */
      phi_c_ = phi_c + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_c_);
      _fv_pl_eq_fv(spinor1, phi_c_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][2], 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_c_, spinor2);


      /* Positive y-direction. */
      phi_c_ = phi_c + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_c_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_c_);

      U_ = g_gauge_field + _GGI(index_s, 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_c_, spinor2);


      /* Negative z-direction. */
      phi_c_ = phi_c + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_c_);
      _fv_pl_eq_fv(spinor1, phi_c_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][3], 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_c_, spinor2);

      /* Positive z-direction. */
      phi_c_ = phi_c + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_c_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_c_);
 
      U_ = g_gauge_field + _GGI(index_s, 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_c_, spinor2);

      /* Multiplication with -1/2. */

      _fv_ti_eq_re(xi_c_, -0.5);

      /* Diagonal elements. */

      phi_c_ = phi_c + _GSI(index_s);
      phi_s_ = phi_s + _GSI(index_s);
		    
      _fv_eq_fv_ti_re(spinor1, phi_c_, _1_2_kappa);
      _fv_pl_eq_fv(xi_c_, spinor1);
		    
      _fv_eq_gamma_ti_fv(spinor1, 5, phi_s_);
      _fv_eq_fv_ti_im(spinor2, spinor1, g_musigma);
      _fv_pl_eq_fv(xi_c_, spinor2);

      _fv_eq_fv_ti_re(spinor1, phi_c_, g_mudelta);
      _fv_pl_eq_fv(xi_c_, spinor1);

  }
/****************************************
 * call xchange_field in calling process
 *
#ifdef MPI
  xchange_field(xi);
#endif
 *
 ****************************************/
}

/****************************************
 * Q applied to the s-quark field
 *
 ****************************************/


void s_Q_h_phi(double *xi_s, double *phi_s, double *phi_c) {
  int ix;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_s_=NULL, *phi_s_=NULL, *phi_c_=NULL, *U_=NULL;

  double _1_2_kappa = 0.5 / g_kappa;

  for(ix = 0; ix < VOLUME; ix++) {

      index_s = ix;

      xi_s_ = xi_s + _GSI(index_s);

      _fv_eq_zero(xi_s_);

      /* Negative t-direction. */
      phi_s_ = phi_s + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_s_);
      _fv_pl_eq_fv(spinor1, phi_s_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][0], 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_s_, spinor2);

      /* Positive t-direction. */
      phi_s_ = phi_s + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_s_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_s_);

      U_ = g_gauge_field + _GGI(index_s, 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_s_, spinor2);

      /* Negative x-direction. */
      phi_s_ = phi_s + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_s_);
      _fv_pl_eq_fv(spinor1, phi_s_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][1], 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_s_, spinor2);

      /* Positive x-direction. */
      phi_s_ = phi_s + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_s_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_s_);

      U_ = g_gauge_field + _GGI(index_s, 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_s_, spinor2);


      /* Negative y-direction. */
      phi_s_ = phi_s + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_s_);
      _fv_pl_eq_fv(spinor1, phi_s_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][2], 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_s_, spinor2);


      /* Positive y-direction. */
      phi_s_ = phi_s + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_s_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_s_);

      U_ = g_gauge_field + _GGI(index_s, 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_s_, spinor2);


      /* Negative z-direction. */
      phi_s_ = phi_s + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_s_);
      _fv_pl_eq_fv(spinor1, phi_s_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][3], 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_s_, spinor2);

      /* Positive z-direction. */
      phi_s_ = phi_s + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_s_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_s_);
 
      U_ = g_gauge_field + _GGI(index_s, 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_s_, spinor2);

      /* Multiplication with -1/2. */

      _fv_ti_eq_re(xi_s_, -0.5);

      /* Diagonal elements. */

      phi_c_ = phi_c + _GSI(index_s);
      phi_s_ = phi_s + _GSI(index_s);
		    
      _fv_eq_fv_ti_re(spinor1, phi_s_, _1_2_kappa);
      _fv_pl_eq_fv(xi_s_, spinor1);
		    
      _fv_eq_gamma_ti_fv(spinor1, 5, phi_c_);
      _fv_eq_fv_ti_im(spinor2, spinor1, g_musigma);
      _fv_pl_eq_fv(xi_s_, spinor2);

      _fv_eq_fv_ti_re(spinor1, phi_s_, g_mudelta);
      _fv_mi_eq_fv(xi_s_, spinor1);
  }
/****************************************
 * call xchange_field in calling process
 *
#ifdef MPI
  xchange_field(xi);
#endif
 *
 ****************************************/
}

/********************
 * Q_h_phi 
 *
 ********************/

void Q_h_phi(double *xi_c, double *xi_s, double *phi_c, double *phi_s) {
  s_Q_h_phi(xi_s, phi_s, phi_c);
  c_Q_h_phi(xi_c, phi_c, phi_s);
}


/********************
 * B_h
 * - the diagornal part of the Dirac operator inverted
 ********************/

void s_B_h_phi(double *xi_s, double *phi_s, double *phi_c, double sign) {

  int ix;
  double spinor1[24], *xi_s_, *phi_s_, *phi_c_;
  double norm = 1. / (1. + 4.*g_kappa*g_kappa*(g_musigma*g_musigma-g_mudelta*g_mudelta) );

  for(ix=0; ix<VOLUME; ix++) {
    xi_s_  = xi_s  + _GSI(ix);
    phi_s_ = phi_s + _GSI(ix);
    phi_c_ = phi_c + _GSI(ix);

    _fv_eq_gamma_ti_fv(spinor1, 5, phi_c_);
    _fv_eq_fv_ti_im(xi_s_, spinor1, sign*2.*g_kappa*g_musigma);

// mp: plus g_mudelta here since it is the flavor tau^3
    _fv_eq_fv_ti_re(spinor1, phi_s_, (1+2.*g_kappa*g_mudelta));
    _fv_pl_eq_fv(xi_s_, spinor1);
    _fv_ti_eq_re(xi_s_, norm);
  }
}

void c_B_h_phi(double *xi_c, double *phi_c, double *phi_s, double sign) {

  int ix;
  double spinor1[24], *phi_s_, *phi_c_, *xi_c_;
  double norm = 1. / (1. + 4.*g_kappa*g_kappa*(g_musigma*g_musigma-g_mudelta*g_mudelta) );

  for(ix=0; ix<VOLUME; ix++) {
    xi_c_  = xi_c  + _GSI(ix);
    phi_s_ = phi_s + _GSI(ix);
    phi_c_ = phi_c + _GSI(ix);

    _fv_eq_gamma_ti_fv(spinor1, 5, phi_s_);
    _fv_eq_fv_ti_im(xi_c_, spinor1, sign*2.*g_kappa*g_musigma);

    _fv_eq_fv_ti_re(spinor1, phi_c_, (1-2.*g_kappa*g_mudelta));
    _fv_pl_eq_fv(xi_c_, spinor1);
    _fv_ti_eq_re(xi_c_, norm);
  }
}

void B_h_phi(double *xi_c, double *xi_s, double *phi_c, double *phi_s, double sign) {
  s_B_h_phi(xi_s, phi_s, phi_c, sign);
  c_B_h_phi(xi_c, phi_c, phi_s, sign);
}



/*****************************************************
 * gamma5_B_h_H4_gamma5 
 * - calculates xi = gamma5 (B^+ H)^4 gamma5 phi
 * - _NOTE_ : B^+H is indep. of repr.
 *****************************************************/

void gamma5_B_h_dagH4_gamma5 (double *xi_c, double *xi_s, double *phi_c, double *phi_s, double *work1, double *work2) {

  int ix, i;
  double spinor1[24];
  double sign = 1.;

  // multiply original source (phi) with gamma_5, save in xi 

  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(xi_c + _GSI(ix), 5, phi_c + _GSI(ix) );
  }

  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(xi_s + _GSI(ix), 5, phi_s + _GSI(ix) );
  }

  /************************************************************
   * apply B^dagger H four times 
   * - status: source = xi from last step, dest = work
   * - NOTE: sign = +1 for B^dagger
   ************************************************************/

  // 1st application 
  xchange_field(xi_c);
  Hopping(work2, xi_c);
  xchange_field(xi_s);
  Hopping(work1, xi_s);
  B_h_phi(xi_c, xi_s, work2, work1, sign);

  // 2nd application
  xchange_field(xi_c);
  Hopping(work1, xi_c);
  xchange_field(xi_s);
  Hopping(work2, xi_s);
  B_h_phi(xi_c, xi_s, work1, work2, sign);

  // 3rd application
  xchange_field(xi_c);
  Hopping(work1, xi_c);
  xchange_field(xi_s);
  Hopping(work2, xi_s);
  B_h_phi(xi_c, xi_s, work1, work2, sign);

  // 4th application
  xchange_field(xi_c);
  Hopping(work1, xi_c);
  xchange_field(xi_s);
  Hopping(work2, xi_s);
  B_h_phi(xi_c, xi_s, work1, work2, sign);


  /* final step: multiply with gamma_5 and */

  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(spinor1, 5, xi_c + _GSI(ix));
    _fv_eq_fv(xi_c + _GSI(ix), spinor1);
  }

  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(spinor1, 5, xi_s + _GSI(ix));
    _fv_eq_fv(xi_s + _GSI(ix), spinor1);
  }

}
