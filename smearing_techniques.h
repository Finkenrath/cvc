/**********************************
 * smearing_techniques.h  
 *
 * Author: Marc Wagner
 * Date: September 2007
 *
 * February 2010
 * taken over to cvc_parallel
 *
 **********************************/
#ifndef _SMEARING_TECHNIQUES_H
#define _SMEARING_TECHNIQUES_H

#include "cvc_linalg.h"

int Fat_Time_Links(double *gauge_field, double *smeared_gauge_field, double time_link_epsilon);

int APE_Smearing_Step(double *smeared_gauge_field, double APE_smearing_alpha);

int APE_Smearing_Step_Timeslice(double *smeared_gauge_field, double APE_smearing_alpha);

int Jacobi_Smearing_Steps(double *smeared_gauge_field, double *psi, int N, double kappa, int timeslice);

int Jacobi_Smearing_Step_one(double *smeared_gauge_field, double *psi, double *psi_old, double kappa);
int Jacobi_Smearing_Step_one_Timeslice(double *smeared_gauge_field, double *psi, double *psi_old, double kappa);
#ifdef OPENMP
int APE_Smearing_Step_threads(double *smeared_gauge_field, int nstep, double APE_smearing_alpha);
int APE_Smearing_Step_Timeslice_threads(double *smeared_gauge_field, int nstep, double APE_smearing_alpha);
// int Jacobi_Smearing_Step_one_threads(double *smeared_gauge_field, double *psi, double *psi_old, double kappa);
int Jacobi_Smearing_Step_one_threads(double *smeared_gauge_field, double *psi, double *psi_old, int nstep, double kappa);
int Jacobi_Smearing_Step_one_Timeslice_threads(double *smeared_gauge_field, double *psi, double *psi_old, int nstep, double kappa);
int Jacobi_Smearing_threaded(double *smeared_gauge_field, double *psi, double *psi_old, double kappa, int nstep, int threadid, int nthreads);
#endif
#endif
