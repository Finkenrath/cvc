/*********************************************************************************
 * jc_u_tp0.c
 *
 * Wed Mar 16 14:50:36 CET 2011
 *
 * PURPOSE:
 * - calculate the local and non-local ViVi t-dep. correlator for use
 *   in moment calculation
 * TODO:
 * DONE:
 * CHANGES:
 *********************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include <getopt.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "contractions_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"

void usage() {
  fprintf(stdout, "Code to perform quark-disconnected conserved vector current contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, j, mu, nu;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, it;
  int sid, status, gid;
  double **corr=NULL, **corr2=NULL;
  double *tcorr=NULL, *tcorr2=NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  int nsource=0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq; 
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_trafo=(double*)NULL;
  double mom2, mom4;
  complex w, w1, *cp1, *cp2, *cp3;
  FILE *ofs; 


#ifdef MPI
//  MPI_Init(&argc, &argv);
  fprintf(stderr, "[jc_ud_x] Error, only non-mpi version implemented\n");
  exit(1);
#endif

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* jc_ud_x\n");
  fprintf(stdout, "**************************************************\n\n");

  /*********************************
   * initialize MPI parameters 
   *********************************/
  // mpi_init(argc, argv);

  /* initialize */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /*************************************************
   * allocate mem for gauge field and spinor fields
   *************************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  nsource = (g_sourceid2 - g_sourceid + 1) / g_sourceid_step;
  if(g_cart_id==0) fprintf(stdout, "# nsource = %d\n", nsource);

  corr     = (double**)calloc( nsource, sizeof(double*));
  corr[0]  = (double*)calloc( nsource*T*8, sizeof(double));
  for(i=1;i<nsource;i++) corr[i] = corr[i-1] + 8*T;

  corr2    = (double**)calloc( nsource, sizeof(double*));
  corr2[0] = (double*)calloc( nsource*8*T, sizeof(double));
  for(i=1;i<nsource;i++) corr2[i] = corr2[i-1] + 8*T;

  tcorr  = (double*)calloc(T*8, sizeof(double));
  tcorr2 = (double*)calloc(T*8, sizeof(double));

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    sprintf(filename, "%s.%.4d", gaugefilename_prefix, gid);
    if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
    xchange_gauge();
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

    /* reset disc to zero */
    for(ix=0; ix<nsource*8*T; ix++) corr[0][ix]  = 0.;
    for(ix=0; ix<nsource*8*T; ix++) corr2[0][ix] = 0.;

    count=0;
    /***********************************************
     * start loop on source id.s 
     ***********************************************/
    for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {

      /* read the new propagator to g_spinor_field[0] */
      ratime = (double)clock() / CLOCKS_PER_SEC;
      if(format==0) {
        sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, gid, sid);
        if(read_lime_spinor(g_spinor_field[0], filename, 0) != 0) break;
      }
      else if(format==1) {
        sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, gid, sid);
        if(read_cmi(g_spinor_field[0], filename) != 0) {
          fprintf(stderr, "\nError from read_cmi\n");
          break;
        }
      }
      xchange_field(g_spinor_field[0]);
      retime = (double)clock() / CLOCKS_PER_SEC;
      if(g_cart_id==0) fprintf(stdout, "# time to read prop.: %e seconds\n", retime-ratime);

      ratime = (double)clock() / CLOCKS_PER_SEC;

      /* apply [1] = D_tm [0] */
      Q_phi_tbc(g_spinor_field[1], g_spinor_field[0]);
      xchange_field(g_spinor_field[1]);

      retime = (double)clock() / CLOCKS_PER_SEC;
      if(g_cart_id==0) fprintf(stdout, "# time to apply D_W: %e seconds\n", retime-ratime);

      ratime = (double)clock() / CLOCKS_PER_SEC;
      /* calculate real and imaginary part */
      for(mu=0; mu<4; mu++) {
        for(x0=0; x0<T; x0++) {
          for(x1=0; x1<LX; x1++) {
          for(x2=0; x2<LY; x2++) {
          for(x3=0; x3<LZ; x3++) {
            ix = g_ipt[x0][x1][x2][x3];
            _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]));
            _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[0][_GSI(g_iup[ix][mu])]);
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
            _fv_mi_eq_fv(spinor2, spinor1);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[1][_GSI(ix)], spinor2);

            corr[count][2*(mu*T+x0)  ] -= 0.5*w.re;
            corr[count][2*(mu*T+x0)+1] -= 0.5*w.im;

            _fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[0][_GSI(ix)]);
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
            _fv_pl_eq_fv(spinor2, spinor1);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[1][_GSI(g_iup[ix][mu])], spinor2);

            corr[count][2*(mu*T+x0)  ] -= 0.5*w.re;
            corr[count][2*(mu*T+x0)+1] -= 0.5*w.im;

            _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[0][_GSI(ix)]);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[1][_GSI(ix)], spinor1);
            corr2[count][2*(mu*T+x0)  ] -= w.re;
            corr2[count][2*(mu*T+x0)+1] -= w.im;
            
          }}}
        }
      }  // of mu

      count++;
    }  // of sid
    retime = (double)clock() / CLOCKS_PER_SEC;
    if(g_cart_id==0) fprintf(stdout, "# time to calculate contractions: %e seconds\n", retime-ratime);

    for(ix=0;ix<8*T;ix++) tcorr[ix] = 0.;
    for(ix=0;ix<8*T;ix++) tcorr2[ix] = 0.;
    
    for(i=0;i<nsource-1;i++) {
    for(j=i+1;j<nsource;j++)   {
      for(mu=0;mu<4;mu++) {
        for(x0=0;x0<T;x0++) {  // times at source
        for(x1=0;x1<T;x1++) {  // times at sink
          it = (x1 - x0 + T) % T;
          // conserved current
          tcorr[2*(mu*T+it)  ] += corr[i][2*(mu*T+x1)] * corr[j][2*(mu*T+x0)  ] - corr[i][2*(mu*T+x1)+1] * corr[j][2*(mu*T+x0)+1];
          tcorr[2*(mu*T+it)+1] += corr[i][2*(mu*T+x1)] * corr[j][2*(mu*T+x0)+1] + corr[i][2*(mu*T+x1)+1] * corr[j][2*(mu*T+x0)  ];
          tcorr[2*(mu*T+it)  ] += corr[j][2*(mu*T+x1)] * corr[i][2*(mu*T+x0)  ] - corr[j][2*(mu*T+x1)+1] * corr[i][2*(mu*T+x0)+1];
          tcorr[2*(mu*T+it)+1] += corr[j][2*(mu*T+x1)] * corr[i][2*(mu*T+x0)+1] + corr[j][2*(mu*T+x1)+1] * corr[i][2*(mu*T+x0)  ];

          // local current
          tcorr2[2*(mu*T+it)  ] += corr2[i][2*(mu*T+x1)] * corr2[j][2*(mu*T+x0)  ] - corr2[i][2*(mu*T+x1)+1] * corr2[j][2*(mu*T+x0)+1];
          tcorr2[2*(mu*T+it)+1] += corr2[i][2*(mu*T+x1)] * corr2[j][2*(mu*T+x0)+1] + corr2[i][2*(mu*T+x1)+1] * corr2[j][2*(mu*T+x0)  ];
          tcorr2[2*(mu*T+it)  ] += corr2[j][2*(mu*T+x1)] * corr2[i][2*(mu*T+x0)  ] - corr2[j][2*(mu*T+x1)+1] * corr2[i][2*(mu*T+x0)+1];
          tcorr2[2*(mu*T+it)+1] += corr2[j][2*(mu*T+x1)] * corr2[i][2*(mu*T+x0)+1] + corr2[j][2*(mu*T+x1)+1] * corr2[i][2*(mu*T+x0)  ];
        }}
      }
    }}

    fnorm = 1. / ( g_prop_normsqr * g_prop_normsqr * (double)(LX*LY*LZ) * (double)(LX*LY*LZ) * nsource * (nsource-1));
    if(g_cart_id==0) fprintf(stdout, "X-fnorm = %e\n", fnorm);
    for(ix=0;ix<8*T;ix++) tcorr[ix]  *= fnorm;
    for(ix=0;ix<8*T;ix++) tcorr2[ix] *= fnorm;

    /************************************************
     * save results
     ************************************************/
    if(g_cart_id == 0) fprintf(stdout, "# save results for gauge id %d and sid %d\n", gid, sid);

    /* save the result in position space */
    sprintf(filename, "jc_u_tp0.%.4d.%.4d", gid, sid);
    ofs = fopen(filename, "w");
    for(x0=0;x0<T;x0++) fprintf(ofs, "%d%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e\n", x0,
       tcorr[2*(0*T+x0)], tcorr[2*(0*T+x0)+1],
       tcorr[2*(1*T+x0)], tcorr[2*(1*T+x0)+1],
       tcorr[2*(2*T+x0)], tcorr[2*(2*T+x0)+1],
       tcorr[2*(3*T+x0)], tcorr[2*(3*T+x0)+1]);
    
    fclose(ofs);

  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  free(corr);
  free(corr2);
  free(tcorr);
  free(tcorr2);

  return(0);

}
