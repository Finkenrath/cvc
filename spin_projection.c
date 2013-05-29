/**********************************************************************
 * spin_projection.c
 *
 * Do 9. Mai 14:56:28 EEST 2013
 *
 * PURPOSE
 * - projection of spinor propagators to spin 1/2 and 3/2
 **********************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ilinalg.h>

/****************************************************************
 * s - result spinor propagator
 * t - input spinor propagator
 ****************************************************************/
void spin_projection_3_2_zero_momentum_tr (spinor_propagator_type *s, spinor_propagator_type *t) {

  const double _MINUS_ONE_THIRD = -1./3.;

  int i, k;

  spinor_propagator_type sp_aux, sp_aux2;

  create_sp(&sp_aux);
  create_sp(&sp_aux2);

  _sp_eq_zero(*s);
  for(i=1; i<4; i++) {
    _sp_pl_eq_sp(*s, t[5*i]);

    // TEST
    // printf_sp(*s, "s_diag", stdout);

    _sp_eq_zero(sp_aux2);
    for(k=1; k<4; k++) {
      // sp_aux = gamma_k t_ki
      _sp_eq_gamma_ti_sp(sp_aux, k, t[4*k+i]);

      // TEST
      // printf_sp(t[4*k+i], "t", stdout);
      // printf_sp(sp_aux, "gamma_ti_t", stdout);

      // sp_aux2 += sp_aux
      _sp_pl_eq_sp(sp_aux2, sp_aux);
    }
    
    // TEST
    // printf_sp(sp_aux2, "sum_gamma_t", stdout);

    // sp_aux = gamma_i sp_aux2 = gamma_i sum_k gamma_k t_ki
    _sp_eq_gamma_ti_sp(sp_aux, i, sp_aux2);

    // TEST
    // printf_sp(sp_aux, "s_gamma", stdout);

    _sp_pl_eq_sp_ti_re(*s, sp_aux, _MINUS_ONE_THIRD);

  }

  free_sp( &sp_aux );
  free_sp( &sp_aux2 );

}  // end of spin_projection_3_2_zero_momentum_tr


/****************************************************************
 * s - result spinor propagator
 * t - input spinor propagator
 ****************************************************************/
void spin_projection_3_2_zero_momentum (spinor_propagator_type *s, spinor_propagator_type *t) {

  const double _MINUS_ONE_THIRD = -1./3.;

  int i, j, k, icmp;
  char name[20];  

  spinor_propagator_type sp_aux, sp_aux2;

  create_sp(&sp_aux);
  create_sp(&sp_aux2);

  // (i) s_{0 nu} = 0 = s_{mu 0}
  for(i=0; i< 4; i++ ) { _sp_eq_zero(s[i]); }
  for(i=4; i<16; i+=4) { _sp_eq_zero(s[i]); }

  for(i=1; i<4; i++) {
  for(j=1; j<4; j++) {
    icmp = 4*i+j; 
    _sp_eq_sp(s[icmp], t[icmp]);

    // TEST
    // printf_sp(s[icmp], "s_diag", stdout);

    _sp_eq_zero(sp_aux2);
    for(k=1; k<4; k++) {
      // sp_aux = gamma_k t_ki
      _sp_eq_gamma_ti_sp(sp_aux, k, t[4*k+j]);

      // TEST
      // sprintf(name, "t_%d%d", k, j);
      // printf_sp(t[4*k+j], name, stdout);

      // TEST
      // sprintf(name, "g%dxt_%d%d", k, k, j);
      // printf_sp(sp_aux, name, stdout);

      // sp_aux2 += sp_aux
      _sp_pl_eq_sp(sp_aux2, sp_aux);
    }

    // TEST
    // printf_sp(sp_aux2, "spaux2", stdout);
    
    // sp_aux = gamma_i sp_aux2 = gamma_i sum_k gamma_k t_ki
    _sp_eq_gamma_ti_sp(sp_aux, i, sp_aux2);

    // TEST
    // printf_sp(sp_aux, "gixspaux2", stdout);

    _sp_pl_eq_sp_ti_re(s[icmp], sp_aux, _MINUS_ONE_THIRD);

  }}  // end of loop on components i,j

  free_sp( &sp_aux );
  free_sp( &sp_aux2 );

}  // end of spin_projection_3_2_zero_momentum

/****************************************************************
 * projection to spin 1/2
 * - s - result spinor propagator
 * - t - input spinor propagator
 ****************************************************************/
void spin_projection_1_2_zero_momentum (spinor_propagator_type *s, spinor_propagator_type *t) {

  const double _ONE_THIRD = -1./3.;

  int i, j, k, icmp;
  char name[20];  

  spinor_propagator_type sp_aux, sp_aux2;

  create_sp(&sp_aux);
  create_sp(&sp_aux2);

  // (i) s_{0 nu} = 0 = s_{mu 0}
  for(i=0; i< 4; i++ ) { _sp_eq_zero(s[i]); }
  for(i=4; i<16; i+=4) { _sp_eq_zero(s[i]); }

  for(i=1; i<4; i++) {
  for(j=1; j<4; j++) {
    icmp = 4*i+j; 
    _sp_eq_zero(s[icmp]);

    _sp_eq_zero(sp_aux2);
    for(k=1; k<4; k++) {
      // sp_aux = gamma_k t_ki
      _sp_eq_gamma_ti_sp(sp_aux, k, t[4*k+j]);

      // sp_aux2 += sp_aux
      _sp_pl_eq_sp(sp_aux2, sp_aux);
    }

    // sp_aux = gamma_i sp_aux2 = gamma_i sum_k gamma_k t_ki
    _sp_eq_gamma_ti_sp(sp_aux, i, sp_aux2);

    _sp_pl_eq_sp_ti_re(s[icmp], sp_aux, _ONE_THIRD);

  }}  // end of loop on components i,j

  free_sp( &sp_aux );
  free_sp( &sp_aux2 );

}  // end of spin_projection_1_2_zero_momentum


/****************************************************************
 * spin projection with general momentum
 * - s - result spinor propagator
 * - t - input spinor propagator
 ****************************************************************/
void spin_projection_3_2 (spinor_propagator_type *s, spinor_propagator_type *t, double e, double *pvec) {

  const double _MINUS_ONE_THIRD  = -1./3.;
  const double _MINUS_TWO_THIRDS = -2./3.;

  int icmp;
  int mu, nu, lambda;
  double p2 = -e*e + pvec[0]*pvec[0] + pvec[1]*pvec[1] + pvec[2]*pvec[2];
  double p2inv = 1./p2;
  double dtmp;

  spinor_propagator_type sp[4], sp_aux, sp_aux2;

  create_sp(&sp_aux);
  create_sp(&sp_aux2);
  create_sp(sp);
  create_sp(sp+1);
  create_sp(sp+2);
  create_sp(sp+3);

  // (i) components (mu,nu), delta_mu_nu - gamma_mu gamma_nu / 3 - 2 / 3 / p^2 p_mu p_nu
  for(nu = 0; nu < 4; nu++) {

    // sp[k] = gamma_k t[k,nu]
    _sp_eq_gamma_ti_sp(sp_aux , 0, t[   nu]);
    _sp_eq_gamma_ti_sp(sp_aux2, 1, t[ 4+nu]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    _sp_eq_gamma_ti_sp(sp_aux2, 2, t[ 8+nu]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    _sp_eq_gamma_ti_sp(sp_aux2, 3, t[12+nu]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);

    for(mu = 0; mu < 4; mu++) {
      icmp = 4*mu + nu;
      _sp_eq_sp(s[icmp], t[icmp]);

      _sp_eq_gamma_ti_sp(sp_aux2, mu, sp_aux);

      _sp_pl_eq_sp_ti_re(s[icmp], sp_aux2, _MINUS_ONE_THIRD);
    }  // end of loop on mu


    dtmp = _MINUS_TWO_THIRDS * p2inv;
    _sp_eq_sp_ti_im(sp_aux , t[   nu], e*dtmp);
    _sp_eq_sp_ti_re(sp_aux2, t[ 4+nu], pvec[0]*dtmp);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    _sp_eq_sp_ti_re(sp_aux2, t[ 8+nu], pvec[1]*dtmp);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    _sp_eq_sp_ti_re(sp_aux2, t[12+nu], pvec[2]*dtmp);
    _sp_pl_eq_sp(sp_aux, sp_aux2);

    // mu = 0
    _sp_pl_eq_sp_ti_im(s[nu], sp_aux, e);
    for(mu=1; mu<4; mu++) {
      icmp = 4*mu + nu;
      _sp_pl_eq_sp_ti_re(s[icmp], sp_aux, pvec[mu-1]);
    }

  }    // end of loop on nu


  // (ii) -1/3 / p^2  ( p_mu gamma_nu - gamma_mu p_nu ) slash(p)
  for(nu=0; nu<4; nu++) {
    // multiply each t_lambda_nu with slash(p)
    for(lambda=0; lambda<4; lambda++) {
      icmp = 4 * lambda + nu;
      // 0
      _sp_eq_gamma_ti_sp(sp_aux, 0, t[icmp]);
      _sp_eq_sp_ti_im(sp[lambda], sp_aux, e);
      // 1
      _sp_eq_gamma_ti_sp(sp_aux, 1, t[icmp]);
      _sp_pl_eq_sp_ti_re(sp[lambda], sp_aux, pvec[0]);
      // 2
      _sp_eq_gamma_ti_sp(sp_aux, 2, t[icmp]);
      _sp_pl_eq_sp_ti_re(sp[lambda], sp_aux, pvec[1]);
      // 3
      _sp_eq_gamma_ti_sp(sp_aux, 3, t[icmp]);
      _sp_pl_eq_sp_ti_re(sp[lambda], sp_aux, pvec[2]);
    }

    // sp_aux2 = sum_lambda gamma_lambda sp_lambda_nu
    _sp_eq_gamma_ti_sp(sp_aux2, 0, sp[0]);

    _sp_eq_gamma_ti_sp(sp_aux, 1, sp[1]);
    _sp_pl_eq_sp(sp_aux2, sp_aux);
    
    _sp_eq_gamma_ti_sp(sp_aux, 2, sp[2]);
    _sp_pl_eq_sp(sp_aux2, sp_aux);

    _sp_eq_gamma_ti_sp(sp_aux, 3, sp[3]);
    _sp_pl_eq_sp(sp_aux2, sp_aux);

    dtmp = _MINUS_ONE_THIRD * p2inv;
    // p_0
    _sp_pl_eq_sp_ti_im(s[nu], sp_aux2, dtmp*e);

    // p_mu, mu = 1, 2, 3
    for(mu=1; mu<4; mu++) {
      icmp = 4 * mu + nu;
      _sp_pl_eq_sp_ti_re(s[icmp], sp_aux2, dtmp*pvec[mu-1]);
    }

    dtmp *= -1.;
    // sp_aux2 = sum_lambda p_lambda sp_lambda_nu
    _sp_eq_sp_ti_im(sp_aux2, sp[0], dtmp*e);

    _sp_pl_eq_sp_ti_re(sp_aux2, sp[1], dtmp*pvec[0]);

    _sp_pl_eq_sp_ti_re(sp_aux2, sp[2], dtmp*pvec[1]);

    _sp_pl_eq_sp_ti_re(sp_aux2, sp[3], dtmp*pvec[2]);

    for(mu=0; mu<4; mu++) {
      icmp = 4 * mu + nu;
      _sp_eq_gamma_ti_sp(sp_aux, mu, sp_aux2);
      _sp_pl_eq_sp(s[icmp], sp_aux);
    }
  }  // of nu

  // TEST
/*
  // apply gamma_mu s_mu_nu
  for(nu=0; nu<4; nu++) {
    _sp_eq_gamma_ti_sp(sp_aux , 0, s[   nu]);
    _sp_eq_gamma_ti_sp(sp_aux2, 1, s[ 4+nu]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    _sp_eq_gamma_ti_sp(sp_aux2, 2, s[ 8+nu]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    _sp_eq_gamma_ti_sp(sp_aux2, 3, s[12+nu]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    norm2_sp(sp_aux, &dtmp);
    fprintf(stdout, "# [spin_projection_3_2] test g_mu s_mu_nu (%d) %16.7e\n", nu, sqrt(dtmp));
  }
  // apply p_mu s_mu_nu
  for(nu=0; nu<4; nu++) {
    _sp_eq_sp_ti_im(sp_aux , s[   nu], e);
    _sp_eq_sp_ti_re(sp_aux2, s[ 4+nu], pvec[0]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    _sp_eq_sp_ti_re(sp_aux2, s[ 8+nu], pvec[1]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    _sp_eq_sp_ti_re(sp_aux2, s[12+nu], pvec[2]);
    _sp_pl_eq_sp(sp_aux, sp_aux2);
    norm2_sp(sp_aux, &dtmp);
    fprintf(stdout, "# [spin_projection_3_2] test p_mu s_mu_nu (%d) %16.7e\n", nu, sqrt(dtmp));
  }
*/
  free_sp(sp);
  free_sp(sp+1);
  free_sp(sp+2);
  free_sp(sp+3);

  free_sp( &sp_aux );
  free_sp( &sp_aux2 );
}  // end of spin_projection_3_2

/****************************************************************
 * spin projection with general momentum like spin_projection_3_2_field
 * - s and t are fields, loop over entries
 ****************************************************************/
void spin_projection_3_2_field (spinor_propagator_type *sfield, spinor_propagator_type *tfield, double e, double *pvec, unsigned int N) {

  const double _MINUS_ONE_THIRD  = -1./3.;
  const double _MINUS_TWO_THIRDS = -2./3.;

  int icmp;
  int mu, nu, lambda;
  double p2 = -e*e + pvec[0]*pvec[0] + pvec[1]*pvec[1] + pvec[2]*pvec[2];
  double p2inv = 1./p2;
  double dtmp;
  unsigned int ix;

  spinor_propagator_type sp[4], sp_aux, sp_aux2, *s=NULL, *t=NULL;

  create_sp(&sp_aux);
  create_sp(&sp_aux2);
  create_sp(sp);
  create_sp(sp+1);
  create_sp(sp+2);
  create_sp(sp+3);

  
  for(ix=0; ix<N; ix++) {
    s = sfield + ix;
    t = tfield + ix;

    // (i) components (mu,nu), delta_mu_nu - gamma_mu gamma_nu / 3 - 2 / 3 / p^2 p_mu p_nu
    for(nu = 0; nu < 4; nu++) {
  
      // sp[k] = gamma_k t[k,nu]
      _sp_eq_gamma_ti_sp(sp_aux , 0, t[   nu]);
      _sp_eq_gamma_ti_sp(sp_aux2, 1, t[ 4+nu]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      _sp_eq_gamma_ti_sp(sp_aux2, 2, t[ 8+nu]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      _sp_eq_gamma_ti_sp(sp_aux2, 3, t[12+nu]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
  
      for(mu = 0; mu < 4; mu++) {
        icmp = 4*mu + nu;
        _sp_eq_sp(s[icmp], t[icmp]);
  
        _sp_eq_gamma_ti_sp(sp_aux2, mu, sp_aux);
  
        _sp_pl_eq_sp_ti_re(s[icmp], sp_aux2, _MINUS_ONE_THIRD);
      }  // end of loop on mu
  
  
      dtmp = _MINUS_TWO_THIRDS * p2inv;
      _sp_eq_sp_ti_im(sp_aux , t[   nu], e*dtmp);
      _sp_eq_sp_ti_re(sp_aux2, t[ 4+nu], pvec[0]*dtmp);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      _sp_eq_sp_ti_re(sp_aux2, t[ 8+nu], pvec[1]*dtmp);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      _sp_eq_sp_ti_re(sp_aux2, t[12+nu], pvec[2]*dtmp);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
  
      // mu = 0
      _sp_pl_eq_sp_ti_im(s[nu], sp_aux, e);
      for(mu=1; mu<4; mu++) {
        icmp = 4*mu + nu;
        _sp_pl_eq_sp_ti_re(s[icmp], sp_aux, pvec[mu-1]);
      }
  
    }    // end of loop on nu
  
  
    // (ii) -1/3 / p^2  ( p_mu gamma_nu - gamma_mu p_nu ) slash(p)
    for(nu=0; nu<4; nu++) {
      // multiply each t_lambda_nu with slash(p)
      for(lambda=0; lambda<4; lambda++) {
        icmp = 4 * lambda + nu;
        // 0
        _sp_eq_gamma_ti_sp(sp_aux, 0, t[icmp]);
        _sp_eq_sp_ti_im(sp[lambda], sp_aux, e);
        // 1
        _sp_eq_gamma_ti_sp(sp_aux, 1, t[icmp]);
        _sp_pl_eq_sp_ti_re(sp[lambda], sp_aux, pvec[0]);
        // 2
        _sp_eq_gamma_ti_sp(sp_aux, 2, t[icmp]);
        _sp_pl_eq_sp_ti_re(sp[lambda], sp_aux, pvec[1]);
        // 3
        _sp_eq_gamma_ti_sp(sp_aux, 3, t[icmp]);
        _sp_pl_eq_sp_ti_re(sp[lambda], sp_aux, pvec[2]);
      }
  
      // sp_aux2 = sum_lambda gamma_lambda sp_lambda_nu
      _sp_eq_gamma_ti_sp(sp_aux2, 0, sp[0]);
  
      _sp_eq_gamma_ti_sp(sp_aux, 1, sp[1]);
      _sp_pl_eq_sp(sp_aux2, sp_aux);
      
      _sp_eq_gamma_ti_sp(sp_aux, 2, sp[2]);
      _sp_pl_eq_sp(sp_aux2, sp_aux);
  
      _sp_eq_gamma_ti_sp(sp_aux, 3, sp[3]);
      _sp_pl_eq_sp(sp_aux2, sp_aux);
  
      dtmp = _MINUS_ONE_THIRD * p2inv;
      // p_0
      _sp_pl_eq_sp_ti_im(s[nu], sp_aux2, dtmp*e);
  
      // p_mu, mu = 1, 2, 3
      for(mu=1; mu<4; mu++) {
        icmp = 4 * mu + nu;
        _sp_pl_eq_sp_ti_re(s[icmp], sp_aux2, dtmp*pvec[mu-1]);
      }
  
      dtmp *= -1.;
      // sp_aux2 = sum_lambda p_lambda sp_lambda_nu
      _sp_eq_sp_ti_im(sp_aux2, sp[0], dtmp*e);
  
      _sp_pl_eq_sp_ti_re(sp_aux2, sp[1], dtmp*pvec[0]);
  
      _sp_pl_eq_sp_ti_re(sp_aux2, sp[2], dtmp*pvec[1]);
  
      _sp_pl_eq_sp_ti_re(sp_aux2, sp[3], dtmp*pvec[2]);
  
      for(mu=0; mu<4; mu++) {
        icmp = 4 * mu + nu;
        _sp_eq_gamma_ti_sp(sp_aux, mu, sp_aux2);
        _sp_pl_eq_sp(s[icmp], sp_aux);
      }
    }  // of nu
  
    // TEST
/*
    // apply gamma_mu s_mu_nu
    for(nu=0; nu<4; nu++) {
      _sp_eq_gamma_ti_sp(sp_aux , 0, s[   nu]);
      _sp_eq_gamma_ti_sp(sp_aux2, 1, s[ 4+nu]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      _sp_eq_gamma_ti_sp(sp_aux2, 2, s[ 8+nu]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      _sp_eq_gamma_ti_sp(sp_aux2, 3, s[12+nu]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      norm2_sp(sp_aux, &dtmp);
      fprintf(stdout, "# [spin_projection_3_2] test g_mu s_mu_nu (%d) %16.7e\n", nu, sqrt(dtmp));
    }
    // apply p_mu s_mu_nu
    for(nu=0; nu<4; nu++) {
      _sp_eq_sp_ti_im(sp_aux , s[   nu], e);
      _sp_eq_sp_ti_re(sp_aux2, s[ 4+nu], pvec[0]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      _sp_eq_sp_ti_re(sp_aux2, s[ 8+nu], pvec[1]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      _sp_eq_sp_ti_re(sp_aux2, s[12+nu], pvec[2]);
      _sp_pl_eq_sp(sp_aux, sp_aux2);
      norm2_sp(sp_aux, &dtmp);
      fprintf(stdout, "# [spin_projection_3_2] test p_mu s_mu_nu (%d) %16.7e\n", nu, sqrt(dtmp));
    }
*/
  }  // of loop on ix

  free_sp(sp);
  free_sp(sp+1);
  free_sp(sp+2);
  free_sp(sp+3);

  free_sp( &sp_aux );
  free_sp( &sp_aux2 );
}  // end of spin_projection_3_2_field
