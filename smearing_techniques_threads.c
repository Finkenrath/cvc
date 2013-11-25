/******************************************
 * smearing_techniques_threads.c
 *
 * functions taken from: 
 *   smearing_techniques.cc
 *   Author: Marc Wagner
 *   Date: September 2007
 ******************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef MPI
#include <mpi.h>  
#endif
#ifdef OPENMP
#include <omp.h>
#endif
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "mpi_init.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "smearing_techniques.h"

/************************************************************
 * Performs an APE smearing step.
 *
 * - the smeared_gauge_field is exchanged at the end
 * ATTENTION: _NO_ exchange of smeared_gauge_field at the beginning
 *            (has to be done by calling process)
 *            _NO_ exchange of smeared gauge field at the end
 *            (has to be done in calling process)
 ************************************************************/
int APE_Smearing_Step_threads(double *smeared_gauge_field, double APE_smearing_alpha) {
  int idx;
  double M1[18], M2[18];
  double *smeared_gauge_field_old = NULL;
  int index;
  int index_mx_1, index_mx_2, index_mx_3;
  int index_px_1, index_px_2, index_px_3;
  int index_my_1, index_my_2, index_my_3;
  int index_py_1, index_py_2, index_py_3;
  int index_mz_1, index_mz_2, index_mz_3;
  int index_pz_1, index_pz_2, index_pz_3;
  double *U = NULL;

  alloc_gauge_field(&smeared_gauge_field_old, VOLUMEPLUSRAND);
  memcpy((void*)smeared_gauge_field_old, (void*)smeared_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));

#ifdef OPENMP
#pragma omp parallel for private (idx,U,M1,M2, index_mx_1, index_mx_2, index_mx_3,\
    index_px_1, index_px_2,\
    index_px_3, index_my_1, index_my_2, index_my_3, index_py_1, index_py_2, index_py_3, index_mz_1,\
    index_mz_2, index_mz_3, index_pz_1, index_pz_2, index_pz_3) shared (smeared_gauge_field_old)
#endif
  for(idx = 0; idx < VOLUME; idx++) {

    /************************
     * Links in x-direction.
     ************************/
    index = _GGI(idx, 1);

    index_my_1 = _GGI(g_idn[idx][2], 2);
    index_my_2 = _GGI(g_idn[idx][2], 1);
    index_my_3 = _GGI(g_idn[g_iup[idx][1]][2], 2);

    index_py_1 = _GGI(idx, 2);
    index_py_2 = _GGI(g_iup[idx][2], 1);
    index_py_3 = _GGI(g_iup[idx][1], 2);

    index_mz_1 = _GGI(g_idn[idx][3], 3);
    index_mz_2 = _GGI(g_idn[idx][3], 1);
    index_mz_3 = _GGI(g_idn[g_iup[idx][1]][3], 3);

    index_pz_1 = _GGI(idx, 3);
    index_pz_2 = _GGI(g_iup[idx][3], 1);
    index_pz_3 = _GGI(g_iup[idx][1], 3);


    U = smeared_gauge_field + index;
    _cm_eq_zero(U);

    /* negative y-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);

    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive y-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);

    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* negative z-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);

    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive z-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);

    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
    _cm_pl_eq_cm(U, M2);

    _cm_ti_eq_re(U, APE_smearing_alpha);

    /* center */
    _cm_pl_eq_cm(U, smeared_gauge_field_old + index);

    /* Projection to SU(3). */
    cm_proj(U);


    /***********************
     * Links in y-direction.
     ***********************/

    index = _GGI(idx, 2);

    index_mx_1 = _GGI(g_idn[idx][1], 1);
    index_mx_2 = _GGI(g_idn[idx][1], 2);
    index_mx_3 = _GGI(g_idn[g_iup[idx][2]][1], 1);

    index_px_1 = _GGI(idx, 1);
    index_px_2 = _GGI(g_iup[idx][1], 2);
    index_px_3 = _GGI(g_iup[idx][2], 1);

    index_mz_1 = _GGI(g_idn[idx][3], 3);
    index_mz_2 = _GGI(g_idn[idx][3], 2);
    index_mz_3 = _GGI(g_idn[g_iup[idx][2]][3], 3);

    index_pz_1 = _GGI(idx, 3);
    index_pz_2 = _GGI(g_iup[idx][3], 2);
    index_pz_3 = _GGI(g_iup[idx][2], 3);

    U = smeared_gauge_field + index;
    _cm_eq_zero(U);

    /* negative x-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive x-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* negative z-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive z-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
    _cm_pl_eq_cm(U, M2);

    _cm_ti_eq_re(U, APE_smearing_alpha);

    /* center */
    _cm_pl_eq_cm(U, smeared_gauge_field_old + index);

    /* Projection to SU(3). */
    cm_proj(U);

    /**************************
     * Links in z-direction.
     **************************/

    index = _GGI(idx, 3);

    index_mx_1 = _GGI(g_idn[idx][1], 1);
    index_mx_2 = _GGI(g_idn[idx][1], 3);
    index_mx_3 = _GGI(g_idn[g_iup[idx][3]][1], 1);

    index_px_1 = _GGI(idx, 1);
    index_px_2 = _GGI(g_iup[idx][1], 3);
    index_px_3 = _GGI(g_iup[idx][3], 1);

    index_my_1 = _GGI(g_idn[idx][2], 2);
    index_my_2 = _GGI(g_idn[idx][2], 3);
    index_my_3 = _GGI(g_idn[g_iup[idx][3]][2], 2);

    index_py_1 = _GGI(idx, 2);
    index_py_2 = _GGI(g_iup[idx][2], 3);
    index_py_3 = _GGI(g_iup[idx][3], 2);

    U = smeared_gauge_field + index;
    _cm_eq_zero(U);

    /* negative x-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive x-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* negative y-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive y-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
    _cm_pl_eq_cm(U, M2);

    _cm_ti_eq_re(U, APE_smearing_alpha);

    /* center */
    _cm_pl_eq_cm(U, smeared_gauge_field_old + index);

    /* Projection to SU(3). */
    cm_proj(U);
  }  // of idx

  free(smeared_gauge_field_old);
  return(0);
}


/*********************************************************
 *
 * Performs an APE smearing step on a given timeslice.
 *
 * - like APE_Smearing_Step_Timeslice_threads
 *********************************************************/
int APE_Smearing_Step_Timeslice_threads(double *smeared_gauge_field, double APE_smearing_alpha) {
  int idx;
  int VOL3 = LX*LY*LZ;
  int index, index_;
  int index_mx_1, index_mx_2, index_mx_3;
  int index_px_1, index_px_2, index_px_3;
  int index_my_1, index_my_2, index_my_3;
  int index_py_1, index_py_2, index_py_3;
  int index_mz_1, index_mz_2, index_mz_3;
  int index_pz_1, index_pz_2, index_pz_3;
  double M1[18], M2[18];
  double *smeared_gauge_field_old=NULL, *U=NULL;

  smeared_gauge_field_old = (double*)malloc(72*VOL3*sizeof(double));
  if(smeared_gauge_field_old == (double*)NULL) {
    if(g_cart_id==0) fprintf(stderr, "Error, could not allocate mem for smeared_gauge_field_old\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif    
    return(1);
  }
  memcpy((void*)smeared_gauge_field_old, (void*)smeared_gauge_field, 72*VOL3*sizeof(double));

#ifdef OPENMP
#pragma omp parallel for private (idx,U,M1,M2, index, index_, \
    index_mx_1, index_mx_2, index_mx_3, index_px_1, index_px_2,\
    index_px_3, index_my_1, index_my_2, index_my_3, index_py_1, index_py_2, index_py_3, index_mz_1,\
    index_mz_2, index_mz_3, index_pz_1, index_pz_2, index_pz_3) shared (smeared_gauge_field_old)
#endif
  for(idx = 0; idx < VOL3; idx++) {

    /********************************
     * Links in x-direction.
     ********************************/
    index      = _GGI(idx, 1);
    index_     = _GGI(idx, 1);

    index_my_1 = _GGI(g_idn[idx][2], 2);
    index_my_2 = _GGI(g_idn[idx][2], 1);
    index_my_3 = _GGI(g_idn[g_iup[idx][1]][2], 2);

    index_py_1 = _GGI(idx, 2);
    index_py_2 = _GGI(g_iup[idx][2], 1);
    index_py_3 = _GGI(g_iup[idx][1], 2);

    index_mz_1 = _GGI(g_idn[idx][3], 3);
    index_mz_2 = _GGI(g_idn[idx][3], 1);
    index_mz_3 = _GGI(g_idn[g_iup[idx][1]][3], 3);

    index_pz_1 = _GGI(idx, 3);
    index_pz_2 = _GGI(g_iup[idx][3], 1);
    index_pz_3 = _GGI(g_iup[idx][1], 3);

    U = smeared_gauge_field + index;
    _cm_eq_zero(U);

    /* negative y-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive y-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* negative z-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive z-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
    _cm_pl_eq_cm(U, M2);

    _cm_ti_eq_re(U, APE_smearing_alpha);

    /* center */
    _cm_pl_eq_cm(U, smeared_gauge_field_old + index_);

    /* Projection to SU(3). */
    cm_proj(U);

    /******************************
     * Links in y-direction.
     ******************************/

    index      = _GGI(idx, 2);
    index_     = _GGI(idx, 2);

    index_mx_1 = _GGI(g_idn[idx][1], 1);
    index_mx_2 = _GGI(g_idn[idx][1], 2);
    index_mx_3 = _GGI(g_idn[g_iup[idx][2]][1], 1);

    index_px_1 = _GGI(idx, 1);
    index_px_2 = _GGI(g_iup[idx][1], 2);
    index_px_3 = _GGI(g_iup[idx][2], 1);

    index_mz_1 = _GGI(g_idn[idx][3], 3);
    index_mz_2 = _GGI(g_idn[idx][3], 2);
    index_mz_3 = _GGI(g_idn[g_iup[idx][2]][3], 3);

    index_pz_1 = _GGI(idx, 3);
    index_pz_2 = _GGI(g_iup[idx][3], 2);
    index_pz_3 = _GGI(g_iup[idx][2], 3);

    U = smeared_gauge_field + index;
    _cm_eq_zero(U);

    /* negative x-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive x-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* negative z-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive z-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
    _cm_pl_eq_cm(U, M2);

    _cm_ti_eq_re(U, APE_smearing_alpha);

    /* center */
    _cm_pl_eq_cm(U, smeared_gauge_field_old + index_);

    /* Projection to SU(3). */
    cm_proj(U);

    /***********************************
     * Links in z-direction.
     ***********************************/

    index      = _GGI(idx, 3);
    index_     = _GGI(idx, 3);

    index_mx_1 = _GGI(g_idn[idx][1], 1);
    index_mx_2 = _GGI(g_idn[idx][1], 3);
    index_mx_3 = _GGI(g_idn[g_iup[idx][3]][1], 1);

    index_px_1 = _GGI(idx, 1);
    index_px_2 = _GGI(g_iup[idx][1], 3);
    index_px_3 = _GGI(g_iup[idx][3], 1);

    index_my_1 = _GGI(g_idn[idx][2], 2);
    index_my_2 = _GGI(g_idn[idx][2], 3);
    index_my_3 = _GGI(g_idn[g_iup[idx][3]][2], 2);

    index_py_1 = _GGI(idx, 2);
    index_py_2 = _GGI(g_iup[idx][2], 3);
    index_py_3 = _GGI(g_iup[idx][3], 2);

    U = smeared_gauge_field + index;
    _cm_eq_zero(U);

    /* negative x-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive x-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* negative y-direction */
    _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
    _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
    _cm_pl_eq_cm(U, M2);

    /* positive y-direction */
    _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
    _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
    _cm_pl_eq_cm(U, M2);

    _cm_ti_eq_re(U, APE_smearing_alpha);

    /* center */
    _cm_pl_eq_cm(U, smeared_gauge_field_old + index_);


    /* Projection to SU(3). */
    cm_proj(U);
  }  // of idx

  free(smeared_gauge_field_old);
/*  xchange_gauge_field(smeared_gauge_field); */
  return(0);
}

/********************************************************************
 *
 * Performs on Jacobi smearing step on a full spinor field
 *
 * psi       = quark spinor
 * kappa     = Jacobi smearing parameter
 * timeslice = the timeslice, on which the smearing is performed
 *
 ********************************************************************/
int Jacobi_Smearing_Step_one_threads(double *smeared_gauge_field, double *psi, double *psi_old, double kappa) {
  int ix, iy, iz, idx, idy;
  int timeslice;
  int VOL3 = LX*LY*LZ;
  int index_s, index_s_mx, index_s_px, index_s_my, index_s_py, index_s_mz, index_s_pz, index_g_mx;
  int index_g_px, index_g_my, index_g_py, index_g_mz, index_g_pz; 
  double *s=NULL, spinor[24];
  double norm = 1.0 / (1.0 + 6.0*kappa);

  /* Copy the timeslice of interest to psi_old. */
  memcpy((void*)psi_old, (void*)psi, 24*(VOLUME+RAND)*sizeof(double));

#ifdef OPENMP
#pragma omp parallel for private(timeslice,ix,idx,idy,index_s,spinor,s) firstprivate(VOL3,norm,kappa)
#endif
  for(timeslice=0; timeslice<T; timeslice++) {

  for(ix = 0; ix < VOL3; ix++) {
    idx = timeslice*VOL3 + ix;

    /* Get indices. */
    index_s = _GSI(idx);

    index_s_mx = _GSI(g_idn[idx][1]);
    index_s_px = _GSI(g_iup[idx][1]);
    index_s_my = _GSI(g_idn[idx][2]);
    index_s_py = _GSI(g_iup[idx][2]);
    index_s_mz = _GSI(g_idn[idx][3]);
    index_s_pz = _GSI(g_iup[idx][3]);

    idy = idx;
    index_g_mx = _GGI(g_idn[idy][1], 1);
    index_g_px = _GGI(idy, 1);
    index_g_my = _GGI(g_idn[idy][2], 2);
    index_g_py = _GGI(idy, 2);
    index_g_mz = _GGI(g_idn[idy][3], 3);
    index_g_pz = _GGI(idy, 3);

    s = psi + _GSI(idy);
    _fv_eq_zero(s);

    /* negative x-direction */
    _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mx, psi_old + index_s_mx);
    _fv_pl_eq_fv(s, spinor);

    /* positive x-direction */
    _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_px, psi_old + index_s_px);
    _fv_pl_eq_fv(s, spinor);

    /* negative y-direction */
    _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_my, psi_old + index_s_my);
    _fv_pl_eq_fv(s, spinor);

    /* positive y-direction */
    _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_py, psi_old + index_s_py);
    _fv_pl_eq_fv(s, spinor);

    /* negative z-direction */
    _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mz, psi_old + index_s_mz);
    _fv_pl_eq_fv(s, spinor);

    /* positive z-direction */
    _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_pz, psi_old + index_s_pz);
    _fv_pl_eq_fv(s, spinor);


    /* Put everything together; normalization. */
    _fv_ti_eq_re(s, kappa);
    _fv_pl_eq_fv(s, psi_old + index_s);
    _fv_ti_eq_re(s, norm);

  }  // of ix
  }  // of timeslice
  return(0);
}

/********************************************************************
 *
 * Performs on Jacobi smearing step on a given timeslice.
 *
 * psi       = quark spinor
 * kappa     = Jacobi smearing parameter
 * timeslice = the timeslice, on which the smearing is performed
 *
 ********************************************************************/
int Jacobi_Smearing_Step_one_Timeslice_threads(double *smeared_gauge_field, double *psi, double *psi_old, double kappa) {
  int idx, idy;
  int VOL3 = LX*LY*LZ;
  int index_s, index_s_mx, index_s_px, index_s_my, index_s_py, index_s_mz, index_s_pz, index_g_mx;
  int index_g_px, index_g_my, index_g_py, index_g_mz, index_g_pz; 
  double *s=NULL, spinor[24];
  double norm = 1.0 / (1.0 + 6.0*kappa);

  /* Copy the timeslice of interest to psi_old. */
  memcpy((void*)psi_old, (void*)psi, 24*VOL3*sizeof(double));

#ifdef OPENMP
#pragma omp parallel for private(idx,idy,index_s,spinor,s) firstprivate(VOL3,norm,kappa)
#endif
  for(idx = 0; idx < VOL3; idx++) {

    /* Get indices. */
    index_s = _GSI(idx);

    index_s_mx = _GSI(g_idn[idx][1]);
    index_s_px = _GSI(g_iup[idx][1]);
    index_s_my = _GSI(g_idn[idx][2]);
    index_s_py = _GSI(g_iup[idx][2]);
    index_s_mz = _GSI(g_idn[idx][3]);
    index_s_pz = _GSI(g_iup[idx][3]);

    idy = idx;
    index_g_mx = _GGI(g_idn[idy][1], 1);
    index_g_px = _GGI(idy, 1);
    index_g_my = _GGI(g_idn[idy][2], 2);
    index_g_py = _GGI(idy, 2);
    index_g_mz = _GGI(g_idn[idy][3], 3);
    index_g_pz = _GGI(idy, 3);

    s = psi + _GSI(idy);
    _fv_eq_zero(s);

    /* negative x-direction */
    _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mx, psi_old + index_s_mx);
    _fv_pl_eq_fv(s, spinor);

    /* positive x-direction */
    _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_px, psi_old + index_s_px);
    _fv_pl_eq_fv(s, spinor);

    /* negative y-direction */
    _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_my, psi_old + index_s_my);
    _fv_pl_eq_fv(s, spinor);

    /* positive y-direction */
    _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_py, psi_old + index_s_py);
    _fv_pl_eq_fv(s, spinor);

    /* negative z-direction */
    _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mz, psi_old + index_s_mz);
    _fv_pl_eq_fv(s, spinor);

    /* positive z-direction */
    _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_pz, psi_old + index_s_pz);
    _fv_pl_eq_fv(s, spinor);


    /* Put everything together; normalization. */
    _fv_ti_eq_re(s, kappa);
    _fv_pl_eq_fv(s, psi_old + index_s);
    _fv_ti_eq_re(s, norm);

  }  // of idx
  return(0);
}

