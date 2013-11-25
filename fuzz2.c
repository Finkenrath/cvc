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
#include "fuzz2.h"

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

int Fuzz_prop3(double *fuzzed_gauge_field, double *psi, double *psi_old, const int Nlong) { 

  int ix, iy, iz, it, iixp, iiyp;
  int index_s, index_s_px, index_s_py,  index_s_pz;
  int index_g_px, index_g_py, index_g_pz;
  int d_up_prev;
  double *s=(double*)NULL, spinor[24];
  double norm = 1.0 / 6.0;
  double *psi_x_p=NULL, *psi_y_p=NULL, *psi_z_p=NULL;
  double *gauge_x_p=NULL, *gauge_y_p=NULL;
#if (defined PARALLELTX) || (defined PARALLELTXY)
  int pid_up, pid_dn, cntr, xid_up, yid_up, d_up, shift_up;
  double *xslice=NULL, *xslice2=NULL;
  double *gaugeslice=NULL, *gaugeslice2=NULL;
  MPI_Status status[12];
  MPI_Request request[12];
#endif 
 
  /* Copy psi_old <- psi. */
  memcpy((void*)psi_old, (void*)psi, 24*(VOLUME+RAND)*sizeof(double));

  psi_x_p = psi_old;
  psi_y_p = psi_old;
  psi_z_p = psi_old;
  gauge_x_p = fuzzed_gauge_field;
  gauge_y_p = fuzzed_gauge_field;
 
#if (defined PARALLELTX) || (defined PARALLELTXY)
#ifdef PARALLELTXY
    ix = (LX > LY) ? LX : LY;
#else
    ix = LY;
#endif
    ix *= T * LZ;
    xslice      = (double*)calloc(24*ix,sizeof(double));
    xslice2     = (double*)calloc(24*ix,sizeof(double));
    gaugeslice  = (double*)calloc(72*ix,sizeof(double));
    gaugeslice2 = (double*)calloc(72*ix,sizeof(double));
    if(xslice==NULL || xslice2==NULL || gaugeslice==NULL || gaugeslice2==NULL) {
      fprintf(stderr, "Error, could not allocate memory for slices\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
      exit(401);
    }
#endif

  /****************************************************
   *
   * the forward directions
   *
   ****************************************************/

  ix = 0;
#if (defined PARALLELTX) || (defined PARALLELTXY)
  d_up = ix + Nlong - LX;
  cntr = 0;
  if(d_up > 0) {
    shift_up = d_up / LX + 1;
    MPI_Cart_shift(g_ts_comm, 0, shift_up, &pid_dn, &pid_up);
    xid_up = ( (LXstart + ix + Nlong) % LX_global ) % LX;

    MPI_Isend(&psi_old[_GSI(g_ipt[0][xid_up][0][0])], 1, spinor_x_slice_vector, pid_dn, 83, g_ts_comm, &request[cntr]);
    cntr++;
    MPI_Irecv(xslice2,                                1, spinor_x_slice_cont,   pid_up, 83, g_ts_comm, &request[cntr]);
    cntr++;
    iixp    = 0;
    psi_x_p = xslice;
  } else {
    iixp    = ix + Nlong;
    psi_x_p = psi_old;
  }
  d_up_prev = d_up;
#else
  iixp      = (ix + Nlong) % LX;
  psi_x_p   = psi_old;
  d_up_prev = 0;
#endif

  /* calculation for positive z-direction */
  for(it=0; it<T;  it++) {
  for(ix=0; ix<LX; ix++) {
  for(iy=0; iy<LY; iy++) {
  for(iz=0; iz<LZ; iz++) {
    index_s    = _GSI(g_ipt[it][ix][iy][iz]);
    index_s_pz = _GSI(g_ipt[it][ix][iy][(iz+Nlong+LZ)%LZ]);
    index_g_pz = _GGI(g_ipt[it][ix][iy][iz], 3);
    s = psi + index_s;
    _fv_eq_zero(s);
    _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_pz, psi_z_p + index_s_pz);
    _fv_pl_eq_fv(s, spinor);	
  }}}}

#if (defined PARALLELTX) || (defined PARALLELTXY)
  if(cntr > 0) MPI_Waitall(cntr, request, status);
#endif

  for(ix = 1; ix < LX; ix++) {
#if (defined PARALLELTX) || (defined PARALLELTXY)
    memcpy((void*)xslice, (void*)xslice2, 24*T*LY*LZ*sizeof(double));
    d_up = ix + Nlong - LX;
    cntr = 0;
    if(d_up > 0) {
      shift_up = d_up / LX + 1;
      MPI_Cart_shift(g_ts_comm, 0, shift_up, &pid_dn, &pid_up);
      xid_up = ((LXstart+ix+Nlong) % LX_global) % LX;

      MPI_Isend(&psi_old[_GSI(g_ipt[0][xid_up][0][0])], 1, spinor_x_slice_vector, pid_dn, 84, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(xslice2,                                1, spinor_x_slice_cont,   pid_up, 84, g_ts_comm, &request[cntr]);
      cntr++;
    }
#endif

    /* positive x-direction for _PREVIOUS_ x-value */
    for(it=0; it<T;  it++) {
    for(iy=0; iy<LY; iy++) {
    for(iz=0; iz<LZ; iz++) {
      index_s    = _GSI(g_ipt[it][ix-1][iy][iz]);
      index_g_px = _GGI(g_ipt[it][ix-1][iy][iz], 1);
      if(d_up_prev > 0) {
        index_s_px = _GSI( (it *LY + iy) * LZ + iz);
      } else {
        index_s_px = _GSI(g_ipt[it][iixp][iy][iz]);
      }
      s = psi + index_s;
      _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_px, psi_x_p + index_s_px);
      _fv_pl_eq_fv(s, spinor);	
    }}}

#if (defined PARALLELTX) || (defined PARALLELTXY)
    if(cntr > 0) MPI_Waitall(cntr, request, status);
    if(d_up > 0) {
      iixp = 0;
      psi_x_p = xslice;
    } else {
      iixp = ix + Nlong;
      psi_x_p = psi_old;
    }
    d_up_prev = d_up;
#else
    iixp = ( ix + Nlong ) % LX;
    psi_x_p = psi_old;
    d_up_prev = 0;
#endif

  }  /* of ix=1,...,LX-1 */

#if (defined PARALLELTX) || (defined PARALLELTXY)
  memcpy((void*)xslice, (void*)xslice2, 24*T*LY*LZ*sizeof(double));
#endif

  /* exchange for first y-value */
  iy = 0;
#if (defined PARALLELTXY)
  d_up = iy + Nlong - LY;
  cntr = 0;
  if(d_up > 0) {
    shift_up = d_up / LY + 1;
    MPI_Cart_shift(g_ts_comm, 1, shift_up, &pid_dn, &pid_up);
    yid_up = ((LYstart + iy + Nlong) % LY_global) % LY;
    
    MPI_Isend(&psi_old[_GSI(g_ipt[0][0][yid_up][0])], 1, spinor_y_slice_vector, pid_dn, 85, g_ts_comm, &request[cntr]);
    cntr++;
    MPI_Irecv(xslice2,                                1, spinor_y_slice_cont,   pid_up, 85, g_ts_comm, &request[cntr]);
    cntr++;
  }
#endif

  /* positive x-direction for last x-value */
  ix = LX-1;
  for(it=0; it<T;  it++) {
  for(iy=0; iy<LY; iy++) {
  for(iz=0; iz<LZ; iz++) {
    index_s    = _GSI(g_ipt[it][ix][iy][iz]);
    index_g_px = _GGI(g_ipt[it][ix][iy][iz], 1);
    if(d_up_prev > 0) {
      index_s_px = _GSI( (it *LY + iy) * LZ + iz);
    } else {
      index_s_px = _GSI(g_ipt[it][iixp][iy][iz]);
    }
    s = psi + index_s;
    _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_px, psi_x_p + index_s_px);
    _fv_pl_eq_fv(s, spinor);	
  }}}

  iy = 0;
#if (defined PARALLELTXY)
  if(cntr > 0) MPI_Waitall(cntr, request, status);
  if(d_up > 0) {
    iiyp = 0;
    psi_y_p = xslice;
  } else {
    iiyp    = iy + Nlong;
    psi_y_p = psi_old;
  }
  d_up_prev = d_up;
#else
  iiyp      = ( iy + Nlong ) % LY;
  psi_y_p   = psi_old;
  d_up_prev = 0;
#endif

  for(iy=1; iy<LY; iy++) {
#ifdef PARALLELTXY
    memcpy((void*)xslice, (void*)xslice2, 24*T*LX*LZ*sizeof(double));
    d_up = iy + Nlong - LY;
    cntr = 0;
    if(d_up > 0) {
      shift_up = d_up / LY + 1;
      yid_up = ( (LYstart + iy + Nlong) % LY_global ) % LY;
      MPI_Cart_shift(g_ts_comm, 1, shift_up, &pid_dn, &pid_up);

      MPI_Isend(&psi_old[_GSI(g_ipt[0][0][yid_up][0])], 1, spinor_y_slice_vector, pid_dn, 86, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(xslice2,                                1, spinor_y_slice_cont,   pid_up, 86, g_ts_comm, &request[cntr]);
      cntr++;
    }
#endif

    /* positive y-direction for _PREVIOUS_ y-value */
    for(it=0; it<T;  it++) {
    for(ix=0; ix<LX; ix++) {
    for(iz=0; iz<LZ; iz++) {
      index_s    = _GSI(g_ipt[it][ix][iy-1][iz]);
      index_g_py = _GGI(g_ipt[it][ix][iy-1][iz], 2);
      if(d_up_prev > 0) {
        index_s_py = _GSI( (it * LX + ix) * LZ + iz);
      } else {
        index_s_py = _GSI(g_ipt[it][ix][iiyp][iz]);
      }
      s = psi + index_s;
      _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_py, psi_y_p + index_s_py);
      _fv_pl_eq_fv(s, spinor);	
    }}}

#ifdef PARALLELTXY
    if(cntr > 0) MPI_Waitall(cntr, request, status);
    if(d_up > 0) {
      iiyp    = 0;
      psi_y_p = xslice;
    } else {
      iiyp    = iy + Nlong;
      psi_y_p = psi_old;
    }
    d_up_prev = d_up;
#else
    iiyp      = (iy + Nlong) % LY;
    psi_y_p   = psi_old;
    d_up_prev = 0;
#endif
  }  /* of iy=1,...,LY-1 */

  /* positive y-direction for last y-value */
  iy = LY-1;
#ifdef PARALLELTXY
  memcpy((void*)xslice, (void*)xslice2, 24*T*LX*LZ*sizeof(double));
#endif
  for(it=0; it<T;  it++) {
  for(ix=0; ix<LX; ix++) {
  for(iz=0; iz<LZ; iz++) {
    index_s    = _GSI(g_ipt[it][ix][iy][iz]);
    index_g_py = _GGI(g_ipt[it][ix][iy][iz], 2);
    if(d_up_prev > 0) {
      index_s_py = _GSI( (it * LX + ix) * LZ + iz);
    } else {
      index_s_py = _GSI(g_ipt[it][ix][iiyp][iz]);
    }
    s = psi + index_s;
    _fv_eq_cm_ti_fv(spinor, fuzzed_gauge_field + index_g_py, psi_y_p + index_s_py);
    _fv_pl_eq_fv(s, spinor);	
  }}}


  /****************************************************
   *
   * the backward directions
   *
   ****************************************************/

  ix = 0;
#if (defined PARALLELTX) || (defined PARALLELTXY)
  d_up = -(ix - Nlong + 1);
  cntr = 0;
  if(d_up > 0) {
    shift_up = -(d_up / LX + 1);
    MPI_Cart_shift(g_ts_comm, 0, shift_up, &pid_dn, &pid_up);
    xid_up = ( (LXstart + ix - Nlong + LX_global) % LX_global ) % LX;

    MPI_Isend(&psi_old[_GSI(g_ipt[0][xid_up][0][0])], 1, spinor_x_slice_vector, pid_dn, 87, g_ts_comm, &request[cntr]);
    cntr++;
    MPI_Irecv(xslice2,                                1, spinor_x_slice_cont,   pid_up, 87, g_ts_comm, &request[cntr]);
    cntr++;
    MPI_Isend(&fuzzed_gauge_field[_GGI(g_ipt[0][xid_up][0][0], 0)], 1, gauge_x_slice_vector, pid_dn, 88, g_ts_comm, &request[cntr]);
    cntr++;
    MPI_Irecv(gaugeslice2,                                          1, gauge_x_slice_cont,   pid_up, 88, g_ts_comm, &request[cntr]);
    cntr++;

    iixp      = 0;
    psi_x_p   = xslice;
    gauge_x_p = gaugeslice;
  } else {
    if( (iixp = ix - Nlong) == -1 ) iixp = LX+1;
    psi_x_p   = psi_old;
    gauge_x_p = fuzzed_gauge_field;
  }
  d_up_prev   = d_up;
#else
  iixp        = (ix - Nlong + LX) % LX;
  psi_x_p     = psi_old;
  gauge_x_p   = fuzzed_gauge_field;
  d_up_prev   = 0;
#endif

  /* calculation for negative z-direction */
  for(it=0; it<T;  it++) {
  for(ix=0; ix<LX; ix++) {
  for(iy=0; iy<LY; iy++) {
  for(iz=0; iz<LZ; iz++) {
    index_s    = _GSI(g_ipt[it][ix][iy][iz]);
    index_s_pz = _GSI(g_ipt[it][ix][iy][(iz-Nlong+LZ)%LZ]);
    index_g_pz = _GGI(g_ipt[it][ix][iy][(iz-Nlong+LZ)%LZ], 3);
    s = psi + index_s;
    _fv_eq_cm_dag_ti_fv(spinor, fuzzed_gauge_field + index_g_pz, psi_z_p + index_s_pz);
    _fv_pl_eq_fv(s, spinor);	
  }}}}

#if (defined PARALLELTX) || (defined PARALLELTXY)
  if(cntr > 0) MPI_Waitall(cntr, request, status);
#endif

  for(ix = 1; ix < LX; ix++) {
#if (defined PARALLELTX) || (defined PARALLELTXY)
    memcpy((void*)xslice,     (void*)xslice2,     24*T*LY*LZ*sizeof(double));
    memcpy((void*)gaugeslice, (void*)gaugeslice2, 72*T*LY*LZ*sizeof(double));
    d_up = -(ix - Nlong + 1);
    cntr = 0;
    if(d_up > 0) {
      shift_up = -(d_up / LX + 1);
      MPI_Cart_shift(g_ts_comm, 0, shift_up, &pid_dn, &pid_up);
      xid_up = ( (LXstart + ix - Nlong + LX_global ) % LX_global) % LX;

      MPI_Isend(&psi_old[_GSI(g_ipt[0][xid_up][0][0])], 1, spinor_x_slice_vector, pid_dn, 89, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(xslice2,                                1, spinor_x_slice_cont,   pid_up, 89, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Isend(&fuzzed_gauge_field[_GGI(g_ipt[0][xid_up][0][0], 0)], 1, gauge_x_slice_vector, pid_dn, 90, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(gaugeslice2,                                          1, gauge_x_slice_cont,   pid_up, 90, g_ts_comm, &request[cntr]);
      cntr++;
    }
#endif

    /* negative x-direction for _PREVIOUS_ x-value */
    for(it=0; it<T;  it++) {
    for(iy=0; iy<LY; iy++) {
    for(iz=0; iz<LZ; iz++) {
      index_s    = _GSI(g_ipt[it][ix-1][iy][iz]);
      if(d_up_prev > 0) {
        index_s_px = _GSI( (it *LY + iy) * LZ + iz);
        index_g_px = _GGI( (it *LY + iy) * LZ + iz, 1);
      } else {
        index_s_px = _GSI(g_ipt[it][iixp][iy][iz]);
        index_g_px = _GGI(g_ipt[it][iixp][iy][iz], 1);
      }
      s = psi + index_s;
      _fv_eq_cm_dag_ti_fv(spinor, gauge_x_p + index_g_px, psi_x_p + index_s_px);
      _fv_pl_eq_fv(s, spinor);	
    }}}

#if (defined PARALLELTX) || (defined PARALLELTXY)
    if(cntr > 0) MPI_Waitall(cntr, request, status);
    if(d_up > 0) {
      iixp      = 0;
      psi_x_p   = xslice;
      gauge_x_p = gaugeslice;
    } else {
      if( (iixp = ix - Nlong) == -1 ) iixp = LX+1;
      psi_x_p   = psi_old;
      gauge_x_p = fuzzed_gauge_field;
    }
    d_up_prev = d_up;
#else
    iixp      = ( ix - Nlong + LX ) % LX;
    psi_x_p   = psi_old;
    gauge_x_p = fuzzed_gauge_field;
    d_up_prev = 0;
#endif

  }  /* of ix=1,...,LX-1 */

#if (defined PARALLELTX) || (defined PARALLELTXY)
  memcpy((void*)xslice,     (void*)xslice2,     24*T*LY*LZ*sizeof(double));
  memcpy((void*)gaugeslice, (void*)gaugeslice2, 72*T*LY*LZ*sizeof(double));
#endif

  /* exchange for first y-value */
  iy = 0;
#if (defined PARALLELTXY)
  d_up = -(iy - Nlong + 1);
  cntr = 0;
  if(d_up > 0) {
    shift_up = -(d_up / LY + 1);
    MPI_Cart_shift(g_ts_comm, 1, shift_up, &pid_dn, &pid_up);
    yid_up = ((LYstart + iy - Nlong + LY_global) % LY_global) % LY;
    
    MPI_Isend(&psi_old[_GSI(g_ipt[0][0][yid_up][0])], 1, spinor_y_slice_vector, pid_dn, 91, g_ts_comm, &request[cntr]);
    cntr++;
    MPI_Irecv(xslice2,                                1, spinor_y_slice_cont,   pid_up, 91, g_ts_comm, &request[cntr]);
    cntr++;
    MPI_Isend(&fuzzed_gauge_field[_GGI(g_ipt[0][0][yid_up][0], 0)], 1, gauge_y_slice_vector, pid_dn, 92, g_ts_comm, &request[cntr]);
    cntr++;
    MPI_Irecv(gaugeslice2,                                          1, gauge_y_slice_cont,   pid_up, 92, g_ts_comm, &request[cntr]);
    cntr++;
  }
#endif

  /* negative x-direction for last x-value */
  ix = LX-1;
  for(it=0; it<T;  it++) {
  for(iy=0; iy<LY; iy++) {
  for(iz=0; iz<LZ; iz++) {
    index_s    = _GSI(g_ipt[it][ix][iy][iz]);
    if(d_up_prev > 0) {
      index_s_px = _GSI( (it *LY + iy) * LZ + iz);
      index_g_px = _GGI( (it *LY + iy) * LZ + iz, 1);
    } else {
      index_s_px = _GSI(g_ipt[it][iixp][iy][iz]);
      index_g_px = _GGI(g_ipt[it][iixp][iy][iz], 1);
    }
    s = psi + index_s;
    _fv_eq_cm_dag_ti_fv(spinor, gauge_x_p + index_g_px, psi_x_p + index_s_px);
    _fv_pl_eq_fv(s, spinor);	
  }}}

  iy = 0;
#if (defined PARALLELTXY)
  if(cntr > 0) MPI_Waitall(cntr, request, status);
  if(d_up > 0) {
    iiyp      = 0;
    psi_y_p   = xslice;
    gauge_y_p = gaugeslice;
  } else {
    if( (iiyp = iy - Nlong) == -1 ) iiyp = LY+1;
    psi_y_p   = psi_old;
    gauge_y_p = fuzzed_gauge_field;
  }
  d_up_prev   = d_up;
#else
  iiyp        = (iy - Nlong + LY) % LY;
  psi_y_p     = psi_old;
  gauge_y_p   = fuzzed_gauge_field;
  d_up_prev   = 0;
#endif

  for(iy=1; iy<LY; iy++) {
#ifdef PARALLELTXY
    memcpy((void*)xslice,     (void*)xslice2,     24*T*LX*LZ*sizeof(double));
    memcpy((void*)gaugeslice, (void*)gaugeslice2, 72*T*LX*LZ*sizeof(double));
    d_up = -(iy - Nlong + 1);
    cntr = 0;
    if(d_up > 0) {
      shift_up = -(d_up / LY + 1);
      yid_up = ( (LYstart + iy - Nlong + LY_global) % LY_global ) % LY;
      MPI_Cart_shift(g_ts_comm, 1, shift_up, &pid_dn, &pid_up);

      MPI_Isend(&psi_old[_GSI(g_ipt[0][0][yid_up][0])], 1, spinor_y_slice_vector, pid_dn, 93, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(xslice2,                                1, spinor_y_slice_cont,   pid_up, 93, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Isend(&fuzzed_gauge_field[_GGI(g_ipt[0][0][yid_up][0], 0)], 1, gauge_y_slice_vector, pid_dn, 94, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(gaugeslice2,                                          1, gauge_y_slice_cont,   pid_up, 94, g_ts_comm, &request[cntr]);
      cntr++;
    }
#endif

    /* negative y-direction for _PREVIOUS_ y-value */
    for(it=0; it<T;  it++) {
    for(ix=0; ix<LX; ix++) {
    for(iz=0; iz<LZ; iz++) {
      index_s    = _GSI(g_ipt[it][ix][iy-1][iz]);
      if(d_up_prev > 0) {
        index_s_py = _GSI( (it * LX + ix) * LZ + iz);
        index_g_py = _GGI( (it * LX + ix) * LZ + iz, 2);
      } else {
        index_s_py = _GSI(g_ipt[it][ix][iiyp][iz]);
        index_g_py = _GGI(g_ipt[it][ix][iiyp][iz], 2);
      }
      s = psi + index_s;
      _fv_eq_cm_dag_ti_fv(spinor, gauge_y_p + index_g_py, psi_y_p + index_s_py);
      _fv_pl_eq_fv(s, spinor);	
    }}}

#ifdef PARALLELTXY
    if(cntr > 0) MPI_Waitall(cntr, request, status);
    if(d_up > 0) {
      iiyp      = 0;
      psi_y_p   = xslice;
      gauge_y_p = gaugeslice;
    } else {
      if( (iiyp = iy - Nlong) == -1) iiyp = LY+1;;
      psi_y_p   = psi_old;
      gauge_y_p = fuzzed_gauge_field;
    }
    d_up_prev   = d_up;
#else
    iiyp        = (iy - Nlong + LY) % LY;
    psi_y_p     = psi_old;
    gauge_y_p   = fuzzed_gauge_field;
    d_up_prev   = 0;
#endif
  }  /* of iy=1,...,LY-1 */

  /* negative y-direction for last y-value */
#ifdef PARALLELTXY
  memcpy((void*)xslice,     (void*)xslice2,     24*T*LX*LZ*sizeof(double));
  memcpy((void*)gaugeslice, (void*)gaugeslice2, 72*T*LX*LZ*sizeof(double));
#endif
  iy = LY-1;
  for(it=0; it<T;  it++) {
  for(ix=0; ix<LX; ix++) {
  for(iz=0; iz<LZ; iz++) {
    index_s    = _GSI(g_ipt[it][ix][iy][iz]);
    if(d_up_prev > 0) {
      index_s_py = _GSI( (it * LX + ix) * LZ + iz);
      index_g_py = _GGI( (it * LX + ix) * LZ + iz, 2);
    } else {
      index_s_py = _GSI(g_ipt[it][ix][iiyp][iz]);
      index_g_py = _GGI(g_ipt[it][ix][iiyp][iz], 2);
    }
    s = psi + index_s;
    _fv_eq_cm_dag_ti_fv(spinor, gauge_y_p + index_g_py, psi_y_p + index_s_py);
    _fv_pl_eq_fv(s, spinor);	
  }}}

#if (defined PARALLELTX) || (defined PARALLELTXY)
  free(xslice); 
  free(xslice2);
  free(gaugeslice);
  free(gaugeslice2);
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

int fuzzed_links2(double *fuzzed_gauge_field, double *smeared_gauge_field, const int Nlong) {
  unsigned long int index, index_;
  unsigned long int index_px , index_py , index_pz ;
  int ir, ix, iy, iz, it, iixp, iiyp;
  int d_up_prev=0;
  double *fuzzed_gauge_field_old = NULL, *gauge_x_p=NULL, *gauge_y_p=NULL;
#if (defined PARALLELTX) || (defined PARALLELTXY)
  int pid_up, pid_dn, xid_up, yid_up, d_up, cntr, shift_up;
  double *gaugeslice_up=NULL, *gaugeslice_up2=NULL;
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

#if (defined PARALLELTX) || (defined PARALLELTXY)
#ifdef PARALLELTXY
  ix = (LX > LY) ? LX : LY;
#else
  ix = LY;
#endif
  ix *= (72 * T * LZ);
  if( (gaugeslice_up = (double*)calloc(ix, sizeof(double))) == NULL ) {
    fprintf(stderr, "Error, could not allocate mem for gaugeslice_up\n");
    return(302);
  }
  if( (gaugeslice_up2 = (double*)calloc(ix, sizeof(double))) == NULL ) {
    fprintf(stderr, "Error, could not allocate mem for gaugeslice_up2\n");
    return(303);
  }
#endif
  
  memcpy((void*)fuzzed_gauge_field_old, (void*)smeared_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));

  for(ir = 1; ir < Nlong; ir++) {

    /* 1st x-value */
    ix = 0;
#if (defined PARALLELTX) || (defined PARALLELTXY)
    d_up = ix + ir - LX;
    cntr = 0;
    if(d_up>0) { /* xchange the upper x-slice */
      shift_up = d_up / LX + 1;
      MPI_Cart_shift(g_ts_comm, 0, shift_up, &pid_dn, &pid_up);
      xid_up = ((LXstart+ir) % LX_global) % LX;
/*      fprintf(stdout, "[%d] g_ts_id=%d, pid_up=%d, pid_dn=%d, xid_up=%d\n", g_cart_id, g_ts_id, pid_up, pid_dn, xid_up); */
      MPI_Isend(&smeared_gauge_field[_GGI(g_ipt[0][xid_up][0][0],0)], 1, gauge_x_slice_vector, pid_dn, 83, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(gaugeslice_up2,                                       1, gauge_x_slice_cont,   pid_up, 83, g_ts_comm, &request[cntr]);
      cntr++;
      iixp = 0;
      gauge_x_p = gaugeslice_up;
    } else {
      iixp = ix + ir;
      gauge_x_p = smeared_gauge_field;
    }
    d_up_prev = d_up;
#else
    iixp = (ix + ir)%LX;
    gauge_x_p = smeared_gauge_field;
    d_up_prev = 0;
#endif

    for(it = 0; it < T;  it++) {
    for(ix = 0; ix < LX; ix++) {
    for(iy = 0; iy < LY; iy++) {
    for(iz = 0; iz < LZ; iz++) {
      index_   = _GGI(g_ipt[it][ix][iy][iz],         3);
      index    = _GGI(g_ipt[it][ix][iy][iz],         3);
      index_pz = _GGI(g_ipt[it][ix][iy][(iz+ir)%LZ], 3);
      _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, smeared_gauge_field+index_pz);
    }}}}

#if (defined PARALLELTX) || (defined PARALLELTXY)
    if(cntr>0) MPI_Waitall(cntr, request, status);
#endif

    for(ix = 1; ix < LX; ix++) {
#if (defined PARALLELTX) || (defined PARALLELTXY)
      memcpy((void*)gaugeslice_up, (void*)gaugeslice_up2, 72*T*LY*LZ*sizeof(double));
      d_up = ix + ir - LX;
      cntr = 0;
      if(d_up>0) {
        shift_up = d_up / LX + 1;
        MPI_Cart_shift(g_ts_comm, 0, shift_up, &pid_dn, &pid_up);
        xid_up = ((LXstart+ix+ir) % LX_global) % LX;
       
        MPI_Isend(&smeared_gauge_field[_GGI(g_ipt[0][xid_up][0][0],0)], 1, gauge_x_slice_vector, pid_dn, 84, g_ts_comm, &request[cntr]);
        cntr++;
        MPI_Irecv(gaugeslice_up2,                                       1, gauge_x_slice_cont,   pid_up, 84, g_ts_comm, &request[cntr]);
        cntr++;
      }
#endif
      for(it = 0; it < T; it++) {
      for(iy = 0; iy < LY; iy++) {
      for(iz = 0; iz < LZ; iz++) {
        index_   = _GGI(g_ipt[it][ix-1][iy][iz],   1);
        index    = _GGI(g_ipt[it][ix-1][iy][iz],   1);
        if(d_up_prev > 0) {
          index_px = _GGI( (it*LY+iy) * LZ+iz ,    1);
        } else {
          index_px = _GGI(g_ipt[it][iixp][iy][iz], 1);
        }
        _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, gauge_x_p+index_px);
      }}}

#if (defined PARALLELTX) || (defined PARALLELTXY)
      if(cntr>0) MPI_Waitall(cntr, request, status);
      if(d_up>0) {
        iixp = 0;
        gauge_x_p = gaugeslice_up;
      } else {
        iixp = ix + ir;
        gauge_x_p = smeared_gauge_field;
      }
      d_up_prev = d_up;
#else
      iixp = (ix + ir)%LX;
      gauge_x_p = smeared_gauge_field;
      d_up_prev = 0;
#endif
    }  /* of ix = 1,...,LX-1 */

#if (defined PARALLELTX) || (defined PARALLELTXY)
    memcpy((void*)gaugeslice_up, (void*)gaugeslice_up2, 72*T*LY*LZ*sizeof(double));
#endif

    /* 1st y-value */
    iy = 0;
#if (defined PARALLELTXY)
    d_up = (iy + ir) - LY;
    cntr = 0;
    if(d_up > 0) {
      shift_up = d_up / LY + 1;
      MPI_Cart_shift(g_ts_comm, 1, shift_up, &pid_dn, &pid_up);
      yid_up = ((LYstart+iy+ir) % LY_global) % LY;
       
      MPI_Isend(&smeared_gauge_field[_GGI(g_ipt[0][0][yid_up][0],0)], 1, gauge_y_slice_vector, pid_dn, 85, g_ts_comm, &request[cntr]);
      cntr++;
      MPI_Irecv(gaugeslice_up2,                                       1, gauge_y_slice_cont,   pid_up, 85, g_ts_comm, &request[cntr]);
      cntr++;
    }
#endif

    /* the last x value */
    ix = LX-1;
    for(it = 0; it < T; it++) {
    for(iy = 0; iy < LY; iy++) {
    for(iz = 0; iz < LZ; iz++) {
      index_   = _GGI(g_ipt[it][ix][iy][iz],     1);
      index    = _GGI(g_ipt[it][ix][iy][iz],     1);
      if(d_up_prev > 0) {
        index_px = _GGI( (it*LY+iy) * LZ+iz ,    1);
      } else {
        index_px = _GGI(g_ipt[it][iixp][iy][iz], 1);
      }
      _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, gauge_x_p+index_px);
    }}}
    
    iy = 0;
#if (defined PARALLELTXY)
    if(cntr>0) MPI_Waitall(cntr, request, status);
    if(d_up > 0){
      iiyp = 0;
      gauge_y_p = gaugeslice_up;
    } else {
      iiyp = iy + ir;
      gauge_y_p = smeared_gauge_field;
    }
    d_up_prev = d_up;
#else
    iiyp = (iy + ir) % LY;
    gauge_y_p = smeared_gauge_field;
    d_up_prev = 0;
#endif

    for(iy=1; iy<LY; iy++) {
#if (defined PARALLELTXY)
      memcpy((void*)gaugeslice_up, (void*)gaugeslice_up2, 72*T*LX*LZ*sizeof(double));
      d_up = iy + ir - LY;
      cntr = 0;
      if(d_up > 0) {
        shift_up = d_up / LY + 1;
        yid_up = ( (LYstart + iy + ir) % LY_global ) % LY;
        MPI_Cart_shift(g_ts_comm, 1, shift_up, &pid_dn, &pid_up);
        MPI_Isend(&smeared_gauge_field[_GGI(g_ipt[0][0][yid_up][0],0)], 1, gauge_y_slice_vector, pid_dn, 86, g_ts_comm, &request[cntr]);
        cntr++;
        MPI_Irecv(gaugeslice_up2,                                       1, gauge_y_slice_cont,   pid_up, 86, g_ts_comm, &request[cntr]);
        cntr++;
      }
#endif

      for(it = 0; it < T; it++) {
      for(ix = 0; ix < LX; ix++) {
      for(iz = 0; iz < LZ; iz++) {
        index_   = _GGI(g_ipt[it][ix][iy-1][iz],   2);
        index    = _GGI(g_ipt[it][ix][iy-1][iz],   2);
        if(d_up_prev > 0) {
          index_py = _GGI( (it*LX+ix)*LZ+iz,       2);
        } else {
          index_py = _GGI(g_ipt[it][ix][iiyp][iz], 2);
        }
        _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, gauge_y_p+index_py);
      }}}

#ifdef PARALLELTXY
      if(cntr > 0) MPI_Waitall(cntr, request, status);
      if(d_up > 0) {
        iiyp = 0;
        gauge_y_p = gaugeslice_up;
      } else {
        iiyp = iy + ir;
        gauge_y_p = smeared_gauge_field;
      }
      d_up_prev = d_up;
#else
      iiyp = ( iy + ir ) % LY;
      gauge_y_p = smeared_gauge_field;
      d_up_prev = 0;
#endif
    }  /* of iy = 1,...,LY-2 */

    /* last y-value */
    iy = LY-1;
#ifdef PARALLELTXY
    memcpy((void*)gaugeslice_up, (void*)gaugeslice_up2, 72*T*LX*LZ*sizeof(double));
#endif
    for(it = 0; it < T; it++) {
    for(ix = 0; ix < LX; ix++) {
    for(iz = 0; iz < LZ; iz++) {
      index_   = _GGI(g_ipt[it][ix][iy][iz],     2);
      index    = _GGI(g_ipt[it][ix][iy][iz],     2);
      if(d_up_prev > 0) {
        index_py = _GGI( (it*LX+ix)*LZ+iz,       2);
      } else {
        index_py = _GGI(g_ipt[it][ix][iiyp][iz], 2);
      }
      _cm_eq_cm_ti_cm(fuzzed_gauge_field+index, fuzzed_gauge_field_old+index_, gauge_y_p+index_py);
    }}}

    xchange_gauge_field_timeslice(fuzzed_gauge_field);
    if (ir < (Nlong-1)) {
      memcpy((void*)fuzzed_gauge_field_old, (void*)fuzzed_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));
    }

  }  /* of ir = 1,...,Nlong-1 */

  free(fuzzed_gauge_field_old);
#if (defined PARALLELTX) || (defined PARALLELTXY)
  free(gaugeslice_up);
  free(gaugeslice_up2);
#endif
  return(0);
}
