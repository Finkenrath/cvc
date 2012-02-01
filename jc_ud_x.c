/*********************************************************************************
 * jc_ud_x.c
 *
 * Mon Aug 30 14:27:40 CEST 2010
 *
 * PURPOSE:
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
#include "ifftw.h"
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
  
  int c, i, mu, nu;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int sid, status, gid;
  double *disc  = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq; 
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_trafo=(double*)NULL;
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

  /* initialize fftw */
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
  disc  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
    exit(3);
  }

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;

    sprintf(filename, "%s.%.4d", gaugefilename_prefix, gid);
    if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
    xchange_gauge();
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

    /***********************************************
     * start loop on source id.s 
     ***********************************************/
    for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
      /* reset disc to zero */
      for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;

      /* read the new propagator to g_spinor_field[0] */
      ratime = (double)clock() / CLOCKS_PER_SEC;
      if(format==0) {
        sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, gid, sid);
        if(read_lime_spinor(g_spinor_field[0], filename, 0) != 0) break;
      }
      else if(format==1) {
        sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, gid, sid);
        if(read_cmi(g_spinor_field[0], filename) != 0) break;
      }
      xchange_field(g_spinor_field[0]);
      retime = (double)clock() / CLOCKS_PER_SEC;
      if(g_cart_id==0) fprintf(stdout, "# time to read prop.: %e seconds\n", retime-ratime);

      ratime = (double)clock() / CLOCKS_PER_SEC;

      /* apply D_W once, save in g_spinor_field[1] */
      Hopping(g_spinor_field[1], g_spinor_field[0]);
      for(ix=0; ix<VOLUME; ix++) {
        _fv_pl_eq_fv(g_spinor_field[1]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
        _fv_ti_eq_re(g_spinor_field[1]+_GSI(ix),  1./(2.*g_kappa));
      }
      xchange_field(g_spinor_field[1]);

      retime = (double)clock() / CLOCKS_PER_SEC;
      if(g_cart_id==0) fprintf(stdout, "# time to apply D_W: %e seconds\n", retime-ratime);

      ratime = (double)clock() / CLOCKS_PER_SEC;
      /* calculate real and imaginary part */
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]));
          _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[0]+_GSI(g_iup[ix][mu]));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_pl_eq_fv(spinor2, spinor1);
          _fv_eq_cm_ti_fv(spinor1, U_, spinor2);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor1);
          disc[_GWI(mu,ix,VOLUME)  ] = g_mu * w.im;

          _fv_eq_gamma_ti_fv(spinor1, mu, g_spinor_field[1]+_GSI(g_iup[ix][mu]));
          _fv_pl_eq_fv(spinor1, g_spinor_field[1]+_GSI(g_iup[ix][mu]));
          _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor2);
          disc[_GWI(mu,ix,VOLUME)+1] = w.im / 3.;
        }
      }
      retime = (double)clock() / CLOCKS_PER_SEC;
      if(g_cart_id==0) fprintf(stdout, "# time to calculate contractions: %e seconds\n", retime-ratime);

      /************************************************
       * save results
       ************************************************/
      if(g_cart_id == 0) fprintf(stdout, "# save results for gauge id %d and sid %d\n", gid, sid);

      /* save the result in position space */
      fnorm = 1. / g_prop_normsqr;
      if(g_cart_id==0) fprintf(stdout, "X-fnorm = %e\n", fnorm);
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          disc[_GWI(mu,ix,VOLUME)  ] *= fnorm;
          disc[_GWI(mu,ix,VOLUME)+1] *= fnorm;
        }
      }
      sprintf(filename, "jc_ud_x.%.4d.%.4d", gid, sid);
      sprintf(contype, "jc-u_and_d-X");
      write_lime_contraction(disc, filename, 64, 4, contype, gid, sid);

      //sprintf(filename, "jc_ud_x.%.4d.%.4d.ascii", gid, sid);
      //write_contraction (disc, NULL, filename, 4, 2, 0);
 
    }  /* of loop on sid */
  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  free(disc);

  return(0);

}
