#ifndef _Q_H_PHI_H
#define _Q_H_PHI_H

void Q_h_phi(double *xi_c, double *xi_s, double *phi_c, double *phi_s);
void c_Q_h_phi(double *xi_c, double *phi_c, double *phi_s);
void s_Q_h_phi(double *xi_s, double *phi_s, double *phi_c);
//
void B_h_phi(double *xi_c, double *xi_s, double *phi_c, double *phi_s, double sign);
void c_B_h_phi(double *xi_c, double *phi_c, double *phi_s, double sign);
void s_B_h_phi(double *xi_s, double *phi_s, double *phi_c, double sign);

void gamma5_B_h_dagH4_gamma5(double *xi_c, double *xi_s, double *phi_c, double *phi_s, double *work1, double *work2);
#endif

