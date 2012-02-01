#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "mpi_init.h"
#include "cvc_utils.h"
#include "fuzz.h"

/************************************************************
 *
 * written by Chris Michael
 *
 * PURPOSE:
 * - Combines Nlong APE-smeared links to one straight fuzzed link for 1 timeslice
 * - as written assumes Nlong > 1  
 * - if Nlong==1 -- needs to copy timeslice of smeared to fuzzed
 *
 * ATTENTION: _NO_ field exchange here
 *            smeared_gauge_field is a _TIMESLICE_, whereas fuzzed_gauge_field
 *            points to a full VOLUME-field
 ************************************************************/

int fuzzed_links_Timeslice(double *fuzzed_gauge_field, double *smeared_gauge_field, const int Nlong, const int timeslice) {
  unsigned long int index, index_;
  unsigned long int index_px , index_py , index_pz ;
  int ir, ix, iy, iz;
  int VOL3 = LX*LY*LZ;
  int status=0;
  double *fuzzed_gauge_field_old = NULL;
  
  if( ( fuzzed_gauge_field_old = (double*)malloc(72*VOL3*sizeof(double)) ) == (double*)NULL ) {
    fprintf(stderr, "Error, could not allocate memory for fuzzed_gauge_field\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    return(301);
  }

  memcpy((void*)fuzzed_gauge_field_old, (void*)smeared_gauge_field, 72*VOL3*sizeof(double));

  for(ir = 1; ir < Nlong; ir++) {
    for(ix = 0; ix < LX; ix++) {
    for(iy = 0; iy < LY; iy++) {
    for(iz = 0; iz < LZ; iz++) {
      index_   = _GGI(g_ipt[0][ix][iy][iz],            1);
      index    = _GGI(g_ipt[timeslice][ix][iy][iz],    1);
      index_px = _GGI(g_ipt[0][(ix+ir)%LX][iy][iz], 1);
 
      _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, smeared_gauge_field+index_px);
 
      index_   = _GGI(g_ipt[0][ix][iy][iz],            2);
      index    = _GGI(g_ipt[timeslice][ix][iy][iz],    2);
      index_py = _GGI(g_ipt[0][ix][(iy+ir)%LY][iz], 2);
	  
      _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, smeared_gauge_field+index_py);
 
      index_   = _GGI(g_ipt[0][ix][iy][iz],            3);
      index    = _GGI(g_ipt[timeslice][ix][iy][iz],    3);
      index_pz = _GGI(g_ipt[0][ix][iy][(iz+ir)%LZ], 3);
	  
      _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, smeared_gauge_field+index_pz);
	  
    }}}
    if (ir < (Nlong-1)) {
      memcpy((void*)fuzzed_gauge_field_old, 
        (void*)(fuzzed_gauge_field + _GGI(g_ipt[timeslice][0][0][0],0)), 72*VOL3*sizeof(double));
    }
  }
  free(fuzzed_gauge_field_old);
  return(0);
}

/************************************************************
 *
 * written by Chris Michael - based on Jacobi smearing code
 *
 * PURPOSE:
 * - Creates a fuzzed spinor - overwrites input spinor
 * - fuzzed gauge field must have length Nlong
 * - input:
 * psi                = quark spinor
 * Nlong              = Fuzzing  parameter
 * fuzzed_gauge_field = the fuzzed gauge field
 *
 * ATTENTION: _NO_ field exchange here
 *************************************************************/

int Fuzz_prop(double *fuzzed_gauge_field, double *psi, const int Nlong) { 

  int VOL3 = LX*LY*LZ;
  int ix, iy, iz, timeslice;
  int index_s, index_s_mx, index_s_px, index_s_my, index_s_py,  index_s_mz, index_s_pz;
  int index_g_mx, index_g_px, index_g_my, index_g_py,  index_g_mz, index_g_pz;
  double *psi_old=(double*)NULL, *s=(double*)NULL, spinor[24];
  double norm = 1.0 / 6.0;
  
  if ( ( psi_old = (double*)malloc(VOL3*24*sizeof(double)) ) ==(double*)NULL) {
    fprintf(stderr, "Error, could not allocate memory for psi_old\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    return(401);
  }
 
  for(timeslice=0; timeslice<T; timeslice++) { 
    /* Copy the timeslice of interest to psi_old. */
    memcpy((void*)psi_old, (void*)(psi+_GSI(g_ipt[timeslice][0][0][0])), 24*VOL3*sizeof(double));
  
    for(ix = 0; ix < LX; ix++) {
    for(iy = 0; iy < LY; iy++) {
    for(iz = 0; iz < LZ; iz++) {
	
      /* Get indices */
      index_s    = _GSI(g_ipt[0][ix][iy][iz]);
      index_s_mx = _GSI(g_ipt[0][(ix-Nlong+LX)%LX][iy][iz]);
      index_s_px = _GSI(g_ipt[0][(ix+Nlong+LX)%LX][iy][iz]);
      index_s_my = _GSI(g_ipt[0][ix][(iy-Nlong+LY)%LY][iz]);
      index_s_py = _GSI(g_ipt[0][ix][(iy+Nlong+LY)%LY][iz]);
      index_s_mz = _GSI(g_ipt[0][ix][iy][(iz-Nlong+LZ)%LZ]);
      index_s_pz = _GSI(g_ipt[0][ix][iy][(iz+Nlong+LZ)%LZ]);
	
      index_g_mx = _GGI(g_ipt[timeslice][(ix-Nlong+LX)%LX][iy][iz], 1);
      index_g_px = _GGI(g_ipt[timeslice][ix][iy][iz], 1);
      index_g_my = _GGI(g_ipt[timeslice][ix][(iy-Nlong+LY)%LY][iz], 2);
      index_g_py = _GGI(g_ipt[timeslice][ix][iy][iz], 2);
      index_g_mz = _GGI(g_ipt[timeslice][ix][iy][(iz-Nlong+LZ)%LZ], 3);
      index_g_pz = _GGI(g_ipt[timeslice][ix][iy][iz], 3);
	
      s = psi + _GSI(g_ipt[timeslice][ix][iy][iz]);
      _fv_eq_zero(s);

      /* negative x-direction */
      _fv_eq_cm_dag_ti_fv(spinor, fuzzed_gauge_field + index_g_mx, psi_old + index_s_mx);
      _fv_pl_eq_fv(s, spinor);
	
      /* positive x-direction */
      _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_px, psi_old + index_s_px);
      _fv_pl_eq_fv(s, spinor);	

      /* negative y-direction */
      _fv_eq_cm_dag_ti_fv(spinor, fuzzed_gauge_field + index_g_my, psi_old + index_s_my);
      _fv_pl_eq_fv(s, spinor);	
	
      /* positive y-direction */
      _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_py, psi_old + index_s_py);
      _fv_pl_eq_fv(s, spinor);	
	
      /* negative z-direction */
      _fv_eq_cm_dag_ti_fv(spinor, fuzzed_gauge_field + index_g_mz, psi_old + index_s_mz);
      _fv_pl_eq_fv(s, spinor);	
	
      /* positive z-direction */
      _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_pz, psi_old + index_s_pz);
      _fv_pl_eq_fv(s, spinor);	
	
/* 
      conventionally CM did not normalise here though 6.0 makes numbers nicer 
      _fv_ti_eq_re(s, norm);
*/
	
    }}}
  } /* of loop on timeslices */
  free(psi_old);
  return(0);
}

/************************************************************
 *
 * written by Chris Michael - based on Jacobi smearing code
 *
 * PURPOSE:
 * - Creates a fuzzed spinor - overwrites input spinor
 * - fuzzed gauge field must have length Nlong
 * - input:
 * psi                = quark spinor
 * Nlong              = Fuzzing  parameter
 * fuzzed_gauge_field = the fuzzed gauge field
 *
 * ATTENTION: _NO_ field exchange here
 *************************************************************/

int Fuzz_prop2(double *fuzzed_gauge_field, double *psi, double *psi_old, const int Nlong) { 

  int VOL3 = LX*LY*LZ;
  int ix, iy, iz, timeslice, iixm, iixp;
  int index_s, index_s_mx, index_s_px, index_s_my, index_s_py,  index_s_mz, index_s_pz;
  int index_g_mx, index_g_px, index_g_my, index_g_py,  index_g_mz, index_g_pz;
  double *s=(double*)NULL, spinor[24];
  double norm = 1.0 / 6.0;
  double *psi_x_m=NULL, *psi_x_p=NULL, *psi_y_m=NULL, *psi_y_p=NULL, *psi_z_m=NULL, *psi_z_p=NULL;
  double *gauge_x_p=NULL, *gauge_x_m=NULL;
#ifdef PARALLELTX
  int pid_up, pid_dn, cntr, xid_up, xid_dn, d_up, d_dn;
  double *xslice_up=NULL, *xslice_dn=NULL;
  double *gaugeslice_up=NULL, *gaugeslice_dn=NULL;
  MPI_Status status[12];
  MPI_Request request[12];
#endif 
 
  /* Copy psi_old <- psi. */
  memcpy((void*)psi_old, (void*)psi, 24*VOLUMEPLUSRAND*sizeof(double));

  psi_y_p = psi_old;
  psi_y_m = psi_old;
  psi_z_p = psi_old;
  psi_z_m = psi_old;
  gauge_x_p = fuzzed_gauge_field;
 
#ifdef PARALLELTX
    xslice_up = (double*)calloc(24*T*LY*LZ,sizeof(double));
    xslice_dn = (double*)calloc(24*T*LY*LZ,sizeof(double));
 /*   gaugeslice_up = (double*)calloc(72*T*LY*LZ,sizeof(double)); */
    gaugeslice_dn = (double*)calloc(72*T*LY*LZ,sizeof(double));
#endif

  for(ix = 0; ix < LX; ix++) {
#ifdef PARALLELTX
    d_up = ix + Nlong - LX;
    d_dn = -(ix - Nlong + 1);
/*    fprintf(stdout, "[%d] ir=%d, ix=%d, d_up=%d, d_dn=%d\n", g_cart_id, Nlong, ix, d_up, d_dn); */
    cntr = 0;
    if(d_up>0) { /* xchange the upper x-slice */
      pid_up = (g_ts_id + (d_up)/LX + 1) % g_nproc_x;
      xid_up = ((LXstart+ix+Nlong) % LX_global) % LX;
      pid_dn = (g_ts_id - (d_up)/LX - 1 + g_nproc_x) % g_nproc_x;
/*      fprintf(stdout, "[%d] g_ts_id=%d, pid_up=%d, pid_dn=%d, xid_up=%d\n", g_cart_id, g_ts_id, pid_up, pid_dn, xid_up); */
      MPI_Isend(&psi_old[_GSI(g_ipt[0][xid_up][0][0])], 1, spinor_x_slice_vector, pid_dn, 83, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(xslice_up,                              1, spinor_x_slice_cont,   pid_up, 83, g_ts_comm, &request[cntr]);
      cntr++;
/*
      MPI_Isend(&fuzzed_gauge_field[_GGI(g_ipt[0][xid_up][0][0],0)], 1, gauge_x_slice_vector, pid_dn, 85, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(gaugeslice_up,                                       1, gauge_x_slice_cont,   pid_up, 85, g_ts_comm, &request[cntr]);
      cntr++;
*/
      psi_x_p = xslice_up;
      iixp = 0;
    } else {
      psi_x_p = psi_old;
      iixp = ix + Nlong;
    }
    if(d_dn>0) { /* xchange the lower x-slice */
      pid_dn = (g_ts_id - (d_dn)/LX - 1 + g_nproc_x) % g_nproc_x;
      xid_dn = ((LXstart+ix-Nlong+LX_global) % LX_global) % LX;
      pid_up = (g_ts_id + (d_dn)/LX + 1) % g_nproc_x;
/*      fprintf(stdout, "[%d] g_ts_id=%d, pid_up=%d, pid_dn=%d, xid_dn=%d\n", g_cart_id, g_ts_id, pid_up, pid_dn, xid_dn); */
      MPI_Isend(&psi_old[_GSI(g_ipt[0][xid_dn][0][0])], 1, spinor_x_slice_vector, pid_up, 84, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(xslice_dn,                              1, spinor_x_slice_cont,   pid_dn, 84, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Isend(&fuzzed_gauge_field[_GGI(g_ipt[0][xid_dn][0][0],0)], 1, gauge_x_slice_vector, pid_up, 86, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(gaugeslice_dn,                                       1, gauge_x_slice_cont,   pid_dn, 86, g_ts_comm, &request[cntr]);
      cntr++;
      psi_x_m = xslice_dn;
      iixm = 0;
      gauge_x_m = gaugeslice_dn;
    } else {
      psi_x_m = psi_old;
      iixm = ix - Nlong;
      if(iixm==-1) iixm = LX+1;
      gauge_x_m = fuzzed_gauge_field;
    }
#else
    psi_x_p = psi_old;
    psi_x_m = psi_old;
    iixp = (ix + Nlong + LX) % LX;
    iixm = (ix - Nlong + LX) % LX;
    gauge_x_m = fuzzed_gauge_field;
#endif

    for(timeslice=0; timeslice<T; timeslice++) { 
    for(iy = 0; iy < LY; iy++) {
    for(iz = 0; iz < LZ; iz++) {
	
      /* Get indices */
      index_s    = _GSI(g_ipt[timeslice][ix][iy][iz]);
      index_s_my = _GSI(g_ipt[timeslice][ix][(iy-Nlong+LY)%LY][iz]);
      index_s_py = _GSI(g_ipt[timeslice][ix][(iy+Nlong+LY)%LY][iz]);
      index_s_mz = _GSI(g_ipt[timeslice][ix][iy][(iz-Nlong+LZ)%LZ]);
      index_s_pz = _GSI(g_ipt[timeslice][ix][iy][(iz+Nlong+LZ)%LZ]);
	
      index_g_my = _GGI(g_ipt[timeslice][ix][(iy-Nlong+LY)%LY][iz], 2);
      index_g_py = _GGI(g_ipt[timeslice][ix][iy][iz], 2);
      index_g_mz = _GGI(g_ipt[timeslice][ix][iy][(iz-Nlong+LZ)%LZ], 3);
      index_g_pz = _GGI(g_ipt[timeslice][ix][iy][iz], 3);
	
      s = psi + _GSI(g_ipt[timeslice][ix][iy][iz]);
      _fv_eq_zero(s);

      /* negative y-direction */
      _fv_eq_cm_dag_ti_fv(spinor, fuzzed_gauge_field + index_g_my, psi_y_m + index_s_my);
      _fv_pl_eq_fv(s, spinor);	
	
      /* positive y-direction */
      _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_py, psi_y_p + index_s_py);
      _fv_pl_eq_fv(s, spinor);	
	
      /* negative z-direction */
      _fv_eq_cm_dag_ti_fv(spinor, fuzzed_gauge_field + index_g_mz, psi_z_m + index_s_mz);
      _fv_pl_eq_fv(s, spinor);	
	
      /* positive z-direction */
      _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_pz, psi_z_p + index_s_pz);
      _fv_pl_eq_fv(s, spinor);	
    }}}

#ifdef PARALLELTX
    if(cntr>0) MPI_Waitall(cntr, request, status);
#endif
    for(timeslice=0; timeslice<T; timeslice++) { 
    for(iy = 0; iy < LY; iy++) {
    for(iz = 0; iz < LZ; iz++) {
	
      index_s    = _GSI(g_ipt[timeslice][ix][iy][iz]);
      s = psi + _GSI(g_ipt[timeslice][ix][iy][iz]);

      /* Get indices */
#ifdef PARALLELTX
      if(d_dn>0) {
        index_s_mx = _GSI((timeslice*LY+iy)*LZ+iz);
        index_g_mx = _GGI((timeslice*LY+iy)*LZ+iz , 1);
      } else {
        index_s_mx = _GSI(g_ipt[timeslice][iixm][iy][iz]);
        index_g_mx = _GGI(g_ipt[timeslice][iixm][iy][iz], 1);
      }

      if(d_up>0) {
        index_s_px = _GSI((timeslice*LY+iy)*LZ+iz);
      } else {
        index_s_px = _GSI(g_ipt[timeslice][iixp][iy][iz]);
      }
      index_g_px = _GGI(g_ipt[timeslice][ix][iy][iz], 1);
#else
      index_s_mx = _GSI(g_ipt[timeslice][(ix-Nlong+LX)%LX][iy][iz]);
      index_s_px = _GSI(g_ipt[timeslice][(ix+Nlong+LX)%LX][iy][iz]);
	
      index_g_mx = _GGI(g_ipt[timeslice][(ix-Nlong+LX)%LX][iy][iz], 1);
      index_g_px = _GGI(g_ipt[timeslice][ix][iy][iz], 1);
#endif	

      /* negative x-direction */
      _fv_eq_cm_dag_ti_fv(spinor, gauge_x_m + index_g_mx, psi_x_m + index_s_mx);
      _fv_pl_eq_fv(s, spinor);
	
      /* positive x-direction */
      _fv_eq_cm_ti_fv(spinor, gauge_x_p + index_g_px, psi_x_p + index_s_px);
      _fv_pl_eq_fv(s, spinor);	
/* 
      conventionally CM did not normalise here though 6.0 makes numbers nicer 
      _fv_ti_eq_re(s, norm);
*/
    }}}
  } /* of loop on ix */
#ifdef PARALLELTX
  free(xslice_up); 
  free(xslice_dn);
/*  free(gaugeslice_up); */
  free(gaugeslice_dn);
#endif
  return(0);
}

/************************************************************
 *
 * written by Chris Michael
 *
 * PURPOSE:
 * - Combines Nlong APE-smeared links to one straight fuzzed link for 1 timeslice
 * - as written assumes Nlong > 1  
 * - if Nlong==1 -- needs to copy timeslice of smeared to fuzzed
 *
 * ATTENTION: _NO_ field exchange here
 *            smeared_gauge_field is a _TIMESLICE_, whereas fuzzed_gauge_field
 *            points to a full VOLUME-field
 ************************************************************/

int fuzzed_links(double *fuzzed_gauge_field, double *smeared_gauge_field, const int Nlong) {
  unsigned long int index, index_;
  unsigned long int index_px , index_py , index_pz ;
  int ir, ix, iy, iz, it, iixp;
  int VOL3 = LX*LY*LZ;
  double *fuzzed_gauge_field_old = NULL, *gauge_x_p=NULL;
#ifdef PARALLELTX
  int pid_up, pid_dn, xid_up, d_up, cntr;
  double *gaugeslice_up=NULL;
  MPI_Status status[12];
  MPI_Request request[12];
#endif
  
  if( ( fuzzed_gauge_field_old = (double*)calloc(72*VOLUMEPLUSRAND, sizeof(double)) ) == (double*)NULL ) {
    fprintf(stderr, "Error, could not allocate memory for fuzzed_gauge_field\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    return(301);
  }

#ifdef PARALLELTX
  gaugeslice_up = (double*)calloc(72*T*LY*LZ, sizeof(double));
#endif
  
  memcpy((void*)fuzzed_gauge_field_old, (void*)smeared_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));

  for(ir = 1; ir < Nlong; ir++) {
    for(ix = 0; ix < LX; ix++) {
#ifdef PARALLELTX
      d_up = ix + ir - LX;
/*      fprintf(stdout, "[%d] ir=%d, ix=%d, d_up=%d\n", g_cart_id, ir, ix, d_up); */
      cntr = 0;
      if(d_up>0) { /* xchange the upper x-slice */
        pid_up = (g_ts_id + (d_up)/LX + 1) % g_nproc_x;
        xid_up = ((LXstart+ix+ir) % LX_global) % LX;
        pid_dn = (g_ts_id - (d_up)/LX - 1 + g_nproc_x) % g_nproc_x;
/*        fprintf(stdout, "[%d] g_ts_id=%d, pid_up=%d, pid_dn=%d, xid_up=%d\n", g_cart_id, g_ts_id, pid_up, pid_dn, xid_up); */
        MPI_Isend(&smeared_gauge_field[_GGI(g_ipt[0][xid_up][0][0],0)], 1, gauge_x_slice_vector, pid_dn, 83, g_ts_comm, &request[cntr]);
        cntr++;
        MPI_Irecv(gaugeslice_up,                                        1, gauge_x_slice_cont,   pid_up, 83, g_ts_comm, &request[cntr]);
        cntr++;
        iixp = 0;
        gauge_x_p = gaugeslice_up;
      } else {
        iixp = ix + ir;
        gauge_x_p = smeared_gauge_field;
      }
#else
      iixp = (ix+ir)%LX;
      gauge_x_p = smeared_gauge_field;
#endif
      for(it = 0; it < T; it++) {
      for(iy = 0; iy < LY; iy++) {
      for(iz = 0; iz < LZ; iz++) {
        index_   = _GGI(g_ipt[it][ix][iy][iz],         2);
        index    = _GGI(g_ipt[it][ix][iy][iz],         2);
        index_py = _GGI(g_ipt[it][ix][(iy+ir)%LY][iz], 2);
	  
        _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, smeared_gauge_field+index_py);
 
        index_   = _GGI(g_ipt[it][ix][iy][iz],         3);
        index    = _GGI(g_ipt[it][ix][iy][iz],         3);
        index_pz = _GGI(g_ipt[it][ix][iy][(iz+ir)%LZ], 3);
	  
        _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, smeared_gauge_field+index_pz);
      }}}
#ifdef PARALLELTX
      if(cntr>0) MPI_Waitall(cntr, request, status);
#endif
      for(it = 0; it < T; it++) {
      for(iy = 0; iy < LY; iy++) {
      for(iz = 0; iz < LZ; iz++) {
        index_   = _GGI(g_ipt[it][ix][iy][iz], 1);
        index    = _GGI(g_ipt[it][ix][iy][iz], 1);
#ifdef PARALLELTX
        if(d_up>0) {
          index_px = _GGI((it*LY+iy)*LZ+iz, 1);
        } else {
#endif
          index_px = _GGI(g_ipt[it][iixp][iy][iz], 1);
#ifdef PARALLELTX
        }
#endif
        _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, gauge_x_p+index_px);
      }}}
    }

    xchange_gauge_field_timeslice(fuzzed_gauge_field);
    if (ir < (Nlong-1)) {
      memcpy((void*)fuzzed_gauge_field_old, (void*)fuzzed_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));
    }
  }  /* of ir = 1,...,Nlong-1 */
  free(fuzzed_gauge_field_old);
#ifdef PARALLELTX
  free(gaugeslice_up);
#endif
  return(0);
}
