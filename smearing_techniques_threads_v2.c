/******************************************
 * smearing_techniques_threads_v2.c
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
#include <time.h>
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
int APE_Smearing_Step_threads(double *smeared_gauge_field, int nstep, double APE_smearing_alpha) {
  int idx, istep, it, ix;
  double *M1=NULL, *M2=NULL, *M1_buffer=NULL,*M2_buffer=NULL;
  double *smeared_gauge_field_old = NULL;
  int index;
  int index_mx_1, index_mx_2, index_mx_3;
  int index_px_1, index_px_2, index_px_3;
  int index_my_1, index_my_2, index_my_3;
  int index_py_1, index_py_2, index_py_3;
  int index_mz_1, index_mz_2, index_mz_3;
  int index_pz_1, index_pz_2, index_pz_3;
  int threadid,nthreads;
  unsigned int VOL3 = LX*LY*LZ;
  double *U = NULL;

  alloc_gauge_field(&smeared_gauge_field_old, VOLUMEPLUSRAND);

#ifdef OPENMP
  nthreads = g_num_threads;
  omp_set_num_threads(nthreads);
#else
  nthreads = 1;
#endif
  M1_buffer = (double*)malloc(nthreads*18*sizeof(double));
  M2_buffer = (double*)malloc(nthreads*18*sizeof(double));

#ifdef OPENMP
#pragma omp parallel private (threadid,istep,it,ix,idx,U,M1,M2,\
    index_mx_1, index_mx_2, index_mx_3, index_px_1, index_px_2, index_px_3, \
    index_my_1, index_my_2, index_my_3, index_py_1, index_py_2, index_py_3, \
    index_mz_1, index_mz_2, index_mz_3, index_pz_1, index_pz_2, index_pz_3) \
  firstprivate (smeared_gauge_field_old,M1_buffer,M2_buffer,smeared_gauge_field,APE_smearing_alpha)
{
  threadid = omp_get_thread_num();
  M1 = M1_buffer + 18*threadid; M2 = M2_buffer + 18*threadid;  
#else
  threadid = 0;
  M1 = M1_buffer; M2 = M2_buffer;
#endif
  for(it = threadid; it < T; it+=nthreads) {

    for(istep=0;istep<nstep;istep++) {

      memcpy((void*)(smeared_gauge_field_old+_GGI(it*VOL3,0)), (void*)(smeared_gauge_field+_GGI(it*VOL3,0)), 72*VOL3*sizeof(double));

      for(ix = 0; ix < VOL3; ix++) {
        idx = it*VOL3 + ix;

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
      }  // of ix
    }    // of it
  }      // of istep
#ifdef OPENMP
}
#endif

  free(smeared_gauge_field_old);
  free(M1_buffer);
  free(M2_buffer);
  return(0);
}


/*********************************************************
 *
 * Performs an APE smearing step on a given timeslice.
 *
 * - like APE_Smearing_Step_Timeslice_threads
 *********************************************************/
int APE_Smearing_Step_Timeslice_threads(double *smeared_gauge_field, int nstep, double APE_smearing_alpha) {
  int idx, istep;
  int VOL3 = LX*LY*LZ;
  int index, index_;
  int index_mx_1, index_mx_2, index_mx_3;
  int index_px_1, index_px_2, index_px_3;
  int index_my_1, index_my_2, index_my_3;
  int index_py_1, index_py_2, index_py_3;
  int index_mz_1, index_mz_2, index_mz_3;
  int index_pz_1, index_pz_2, index_pz_3;
  double *M1=NULL, *M2=NULL, *M1_buffer=NULL, *M2_buffer=NULL;
  int threadid, nthreads;
  double *smeared_gauge_field_old=NULL, *U=NULL;
  unsigned int offset, bytes;

  smeared_gauge_field_old = (double*)malloc(72*VOL3*sizeof(double));
  if(smeared_gauge_field_old == (double*)NULL) {
    if(g_cart_id==0) fprintf(stderr, "Error, could not allocate mem for smeared_gauge_field_old\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif    
    return(1);
  }

#ifdef OPENMP
  nthreads = g_num_threads;
  omp_set_num_threads(nthreads);
#else
  nthreads = 1;
#endif
  M1_buffer = (double*)malloc(nthreads*18*sizeof(double));
  M2_buffer = (double*)malloc(nthreads*18*sizeof(double));

#ifdef OPENMP
#pragma omp parallel private (threadid, idx,U,M1,M2, index, index_, istep, offset, bytes,\
    index_mx_1, index_mx_2, index_mx_3, index_px_1, index_px_2, index_px_3,\
    index_my_1, index_my_2, index_my_3, index_py_1, index_py_2, index_py_3,\
    index_mz_1, index_mz_2, index_mz_3, index_pz_1, index_pz_2, index_pz_3) \
  firstprivate (smeared_gauge_field_old,M1_buffer,M2_buffer,smeared_gauge_field,APE_smearing_alpha)
{
  threadid = omp_get_thread_num();
  M1 = M1_buffer + 18*threadid; M2 = M2_buffer + 18*threadid;
  offset = threadid * (VOL3/nthreads);
  bytes = VOL3/nthreads;
  if(offset+bytes>VOL3) bytes = VOL3-offset;
  offset *= 72;
  bytes *= 72*sizeof(double);
#else
  threadid = 0;
  M1 = M1_buffer; M2 = M2_buffer;
  offset = 0;
  bytes = 72*VOL3*sizeof(double);
#endif
  for(istep=0;istep<nstep;istep++) {

    memcpy((void*)(smeared_gauge_field_old+offset), (void*)(smeared_gauge_field+offset), bytes);

#pragma omp barrier
    for(idx = threadid; idx < VOL3; idx+=nthreads) {
  
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
  
      // negative y-direction
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // positive y-direction
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // negative z-direction
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // positive z-direction
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      // center 
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index_);
  
      // Projection to SU(3). 
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
  
      // negative x-direction 
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // positive x-direction 
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // negative z-direction 
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // positive z-direction 
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      // center 
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index_);
  
      // Projection to SU(3). 
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
  
      // negative x-direction 
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // positive x-direction 
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // negative y-direction 
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      // positive y-direction 
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      // center 
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index_);
  
  
      // Projection to SU(3). 
      cm_proj(U);
    }  // of idx
#ifdef OPENMP
#pragma omp barrier
#endif
  //xchange_gauge_field(smeared_gauge_field);
  }    // of istep
#ifdef OPENMP
}
#endif
  free(smeared_gauge_field_old);
  free(M1_buffer);
  free(M2_buffer);
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
int Jacobi_Smearing_Step_one_threads(double *smeared_gauge_field, double *psi, double *psi_old, int nstep, double kappa) {
  int idx, it, ix, istep;
  int index_s, index_s_mx, index_s_px, index_s_my, index_s_py, index_s_mz, index_s_pz, index_g_mx;
  int index_g_px, index_g_my, index_g_py, index_g_mz, index_g_pz; 
  double *s=NULL, *s_old=NULL, *spinor=NULL, *spinor_buffer=NULL, *Up=NULL, *Um=NULL, *U_buffer=NULL;
  double norm = 1.0 / (1.0 + 6.0*kappa);
  int idx_min, idx_max, threadid, nthreads;
  double ratime, retime;
  unsigned int VOL3 = LX*LY*LZ;

  // Copy the timeslice of interest to psi_old.
//  memcpy((void*)psi_old, (void*)psi, 24*(VOLUME+RAND)*sizeof(double));
#ifdef OPENMP
  nthreads = g_num_threads;
#else
  nthreads = 1;
#endif
  //fprintf(stdout, "# [Jacobi_Smearing_Step_one_threads] number of threads = %d\n", nthreads);
  spinor_buffer = (double*)malloc(3*nthreads*24*sizeof(double));
  U_buffer = (double*)malloc(2*nthreads*54*sizeof(double));

#ifdef OPENMP
  omp_set_num_threads(nthreads);
#pragma omp parallel private(istep,it,ix,idx,index_s,spinor,s,threadid,            \
    index_s_mx, index_s_px, index_s_my, index_s_py, index_s_mz, index_s_pz,        \
    index_g_mx,index_g_px, index_g_my, index_g_py, index_g_mz, index_g_pz,         \
    Up,Um,s_old)                                                                   \
  firstprivate(VOLUME,norm,kappa,spinor_buffer,g_idn,g_iup,psi_old,smeared_gauge_field,nstep,psi,U_buffer,VOL3)
{
  threadid = omp_get_thread_num();
  spinor = spinor_buffer + 24*threadid;
  s      = spinor_buffer + 24*(nthreads + threadid );
  s_old  = spinor_buffer + 24*(2*nthreads + threadid );
  Up     = U_buffer + 54*threadid;
  Um     = U_buffer + 54*(threadid+nthreads);
  //fprintf(stdout, "# [Jacobi_Smearing_Step_one_threads] thread%2d range %d %d\n", threadid, idx_min, idx_max);
//  ratime = omp_get_wtime();
#else
  threadid=0;
  spinor = spinor_buffer;
  s      = spinor_buffer + 24;
  s_old  = spinor_buffer + 48;
  Up     = U_buffer;
  Um     = U_buffer + 54;
//  ratime = (double)clock() / (double)CLOCKS_PER_SEC;
#endif
  for(it = threadid; it < T; it+=nthreads) {

    for(istep = 0; istep < nstep; istep++) {

      memcpy(psi_old+_GSI(it*VOL3), psi+_GSI(it*VOL3), 24*VOL3*sizeof(double) );

      for(ix = 0; ix < VOL3; ix++) {
        idx = it*VOL3 + ix;

        // get indices of neighbours
        index_s = _GSI(idx);
    
        index_s_mx = _GSI(g_idn[idx][1]);
        index_s_px = _GSI(g_iup[idx][1]);
        index_s_my = _GSI(g_idn[idx][2]);
        index_s_py = _GSI(g_iup[idx][2]);
        index_s_mz = _GSI(g_idn[idx][3]);
        index_s_pz = _GSI(g_iup[idx][3]);
    
        index_g_mx = _GGI(g_idn[idx][1], 1);
        index_g_px = _GGI(idx, 1);
        index_g_my = _GGI(g_idn[idx][2], 2);
        index_g_py = _GGI(idx, 2);
        index_g_mz = _GGI(g_idn[idx][3], 3);
        index_g_pz = _GGI(idx, 3);
    
        memcpy(Up, smeared_gauge_field + index_g_px, 54*sizeof(double) );
        memcpy(Um   , smeared_gauge_field + index_g_mx, 18*sizeof(double) );
        memcpy(Um+18, smeared_gauge_field + index_g_my, 18*sizeof(double) );
        memcpy(Um+36, smeared_gauge_field + index_g_mz, 18*sizeof(double) );
    
        //s = psi + _GSI(idx);
        _fv_eq_zero(s);
    
        // negative x-direction
        memcpy(s_old, psi_old + index_s_mx, 24*sizeof(double));
        _fv_eq_cm_dag_ti_fv(spinor, Um, s_old);
        _fv_pl_eq_fv(s, spinor);
    
        // positive x-direction
        memcpy(s_old, psi_old + index_s_px, 24*sizeof(double));
        _fv_eq_cm_ti_fv(spinor, Up, s_old);
        _fv_pl_eq_fv(s, spinor);
    
        // negative y-direction
        memcpy(s_old, psi_old + index_s_my, 24*sizeof(double));
        _fv_eq_cm_dag_ti_fv(spinor, Um+18, s_old);
        _fv_pl_eq_fv(s, spinor);
    
        // positive y-direction
        memcpy(s_old, psi_old + index_s_py, 24*sizeof(double));
        _fv_eq_cm_ti_fv(spinor, Up+18, s_old);
        _fv_pl_eq_fv(s, spinor);
    
        // negative z-direction
        memcpy(s_old, psi_old + index_s_mz, 24*sizeof(double));
        _fv_eq_cm_dag_ti_fv(spinor, Um+36, s_old);
        _fv_pl_eq_fv(s, spinor);
    
        // positive z-direction
        memcpy(s_old, psi_old + index_s_pz, 24*sizeof(double));
        _fv_eq_cm_ti_fv(spinor, Up+36, s_old);
        _fv_pl_eq_fv(s, spinor);
    
        // Put everything together; normalization.
        memcpy(s_old, psi_old + index_s, 24*sizeof(double));
        _fv_ti_eq_re(s, kappa);
        _fv_pl_eq_fv(s, s_old);
        _fv_eq_fv_ti_re(psi+_GSI(idx), s, norm);
    
      }  // of ix
    }    // of istep
  }      // of it
#ifdef OPENMP
//  retime = omp_get_wtime();
#else
//  retime = (double)clock() / (double)CLOCKS_PER_SEC;
#endif
//  if(threadid==0) fprintf(stdout, "# [Jacobi_Smearing_Step_one_threads] time for smearing iterations = %e\n", retime-ratime);
#ifdef OPENMP
}
#endif
  free(spinor_buffer);
  free(U_buffer);

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
int Jacobi_Smearing_Step_one_Timeslice_threads(double *smeared_gauge_field, double *psi, double *psi_old, int nstep, double kappa) {
  int idx, istep;
  int VOL3 = LX*LY*LZ;
  int index_s, index_s_mx, index_s_px, index_s_my, index_s_py, index_s_mz, index_s_pz, index_g_mx;
  int index_g_px, index_g_my, index_g_py, index_g_mz, index_g_pz; 
  double *s=NULL, *spinor=NULL, *spinor_buffer=NULL;
  double norm = 1.0 / (1.0 + 6.0*kappa);
  int threadid, nthreads;
  unsigned offset, bytes;

#ifdef OPENMP
  nthreads = g_num_threads;
  omp_set_num_threads(nthreads);
#else
  nthreads = 1;
#endif
  spinor_buffer = (double*)malloc(nthreads*24*sizeof(double));

#ifdef OPENMP
#pragma omp parallel private(threadid,istep, idx,index_s,spinor,s,\
    index_s_mx, index_s_px, index_s_my, index_s_py, index_s_mz, index_s_pz,\
    index_g_mx,index_g_px, index_g_my, index_g_py, index_g_mz, index_g_pz,offset,bytes) \
  firstprivate(VOL3,norm,kappa,spinor_buffer,psi_old,psi,smeared_gauge_field,nstep)
{
  threadid = omp_get_thread_num();
  spinor = spinor_buffer + 24*threadid;
  offset = threadid * (VOL3/nthreads);
  bytes = (VOL3/nthreads);
  if(offset+bytes>VOL3) bytes = VOL3-offset;
  offset *= 24;
  bytes *= 24*sizeof(double);
#else
  threadid=0;
  spinor = spinor_buffer;
  offset = 0;
  bytes = 24 * VOL3 * sizeof(double);
#endif
  for(istep=0;istep<nstep;istep++) {
    // Copy the timeslice of interest to psi_old
    memcpy( (void*)(psi_old+offset), (void*)(psi+offset), bytes);
    //for(idx=threadid; idx<VOL3; idx+=nthreads) {
    //  _fv_eq_fv(psi_old+_GSI(idx), psi+_GSI(idx));
    //}
#pragma omp barrier
    for(idx=threadid; idx<VOL3; idx+=nthreads) {
  
      // Get indices
      index_s = _GSI(idx);
  
      index_s_mx = _GSI(g_idn[idx][1]);
      index_s_px = _GSI(g_iup[idx][1]);
      index_s_my = _GSI(g_idn[idx][2]);
      index_s_py = _GSI(g_iup[idx][2]);
      index_s_mz = _GSI(g_idn[idx][3]);
      index_s_pz = _GSI(g_iup[idx][3]);
  
      index_g_mx = _GGI(g_idn[idx][1], 1);
      index_g_px = _GGI(idx, 1);
      index_g_my = _GGI(g_idn[idx][2], 2);
      index_g_py = _GGI(idx, 2);
      index_g_mz = _GGI(g_idn[idx][3], 3);
      index_g_pz = _GGI(idx, 3);
  
      s = psi + _GSI(idx);
      _fv_eq_zero(s);
  
      // negative x-direction
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mx, psi_old + index_s_mx);
      _fv_pl_eq_fv(s, spinor);
  
      // positive x-direction
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_px, psi_old + index_s_px);
      _fv_pl_eq_fv(s, spinor);
  
      // negative y-direction
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_my, psi_old + index_s_my);
      _fv_pl_eq_fv(s, spinor);
  
      // positive y-direction
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_py, psi_old + index_s_py);
      _fv_pl_eq_fv(s, spinor);
  
      // negative z-direction
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mz, psi_old + index_s_mz);
      _fv_pl_eq_fv(s, spinor);
  
      // positive z-direction
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_pz, psi_old + index_s_pz);
      _fv_pl_eq_fv(s, spinor);
  
  
      // Put everything together; normalization
      _fv_ti_eq_re(s, kappa);
      _fv_pl_eq_fv(s, psi_old + index_s);
      _fv_ti_eq_re(s, norm);
  
    }  // of idx
#ifdef OPENMP
#pragma omp barrier
#endif
  }    // of istep
#ifdef OPENMP
}
#endif
  free(spinor_buffer);

  return(0);
}

