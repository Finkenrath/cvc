/*********************************************************************************
 * jc_u_x.c
 *
 * Tue May  3 10:24:45 CEST 2011
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

#ifdef MPI
#  define _TAKE_TIME MPI_Wtime()
#else
#  define _TAKE_TIME (double)clock() / CLOCKS_PER_SEC
#endif

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
  int mms_id = -1;
  int exitstatus;
  int write_ascii = 0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq; 
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_trafo=(double*)NULL;
  double *spinor_work = NULL;
  complex w, w1, *cp1, *cp2, *cp3;
  FILE *ofs; 

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ah?f:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'm':
      mms_id = atoi(optarg);
      fprintf(stdout, "\n# [jc_u_x] will read propagators in MMS format with mass id %d\n", mms_id);
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [jc_u_x] will will write contraction data in ASCII format too\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // global time stamp
  g_the_time = time(NULL);
  if(g_proc_id == 0) fprintf(stdout, "\n# [jc_u_x] using global time_stamp %s", ctime(&g_the_time));


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [jc_u_x] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[jc_u_x] Error: T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stderr, "[jc_u_x] Error: kappa should be > 0.n");
    usage();
  }

  /*********************************
   * initialize MPI parameters 
   *********************************/
  mpi_init(argc, argv);

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* jc_u_x\n");
  fprintf(stdout, "* %s", ctime(&g_the_time));
  fprintf(stdout, "**************************************************\n\n");

  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifndef MPI
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
#endif
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n",
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at);

  if(init_geometry() != 0) {
    fprintf(stderr, "[jc_u_x] Error from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  /*************************************************
   * allocate mem for gauge field and spinor fields
   *************************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  no_fields = 2;
  if(mms_id != -1) no_fields++;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);
  if(mms_id != -1) spinor_work = g_spinor_field[no_fields-1];

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  disc  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "[jc_u_x] Error: could not allocate memory for disc\n");
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 3);
      MPI_Finalize();
#endif
    exit(3);
  }

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid++) {

    sprintf(filename, "%s.%.4d", gaugefilename_prefix, gid);
    if(g_cart_id==0) fprintf(stdout, "# [jc_u_x] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
    if(exitstatus != 0) {
      fprintf(stderr, "\n[jc_u_x] Error, could not read from file %s\n", filename);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 2);
      MPI_Finalize();
#endif
      exit(2);
    }
    xchange_gauge();
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# [jc_u_x] measured plaquette value: %25.16e\n", plaq);

    /***********************************************
     * start loop on source id.s 
     ***********************************************/
    for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
      /* reset disc to zero */
      for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;

      /* read the new propagator to g_spinor_field[0] */
      ratime = _TAKE_TIME;
      if( mms_id == -1 ) {
        if(format==0) {
          sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, gid, sid);
          if(read_lime_spinor(g_spinor_field[0], filename, 0) != 0) {
            fprintf(stderr, "[jc_u_x] Error: could not read from file %s\n", filename);
#ifdef MPI
            MPI_Abort(MPI_COMM_WORLD, 5);
            MPI_Finalize();
#endif
            exit(5);
          };
        }
        else if(format==1) {
          sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, gid, sid);
          if(read_cmi(g_spinor_field[0], filename) != 0) {
            fprintf(stderr, "[jc_u_x] Error: could not read from file %s\n", filename);
#ifdef MPI
            MPI_Abort(MPI_COMM_WORLD, 5);
            MPI_Finalize();
#endif
            exit(5);
          };
        }
      } else {
        sprintf(filename, "%s.%.4d.%.5d.cgmms.%.2d.inverted", filename_prefix, gid, sid, mms_id);
        if(read_lime_spinor(spinor_work, filename, 0) != 0) {
          fprintf(stderr, "[jc_u_x] Error: could not read from file %s\n", filename);
#ifdef MPI
          MPI_Abort(MPI_COMM_WORLD, 5);
          MPI_Finalize();
#endif
          exit(5);
        };
        xchange_field(spinor_work);
        Qf5(g_spinor_field[0], spinor_work, -g_mu);
      }
      xchange_field(g_spinor_field[0]);
      retime = _TAKE_TIME;
      if(g_cart_id==0) fprintf(stdout, "# [jc_u_x] time to read prop.: %e seconds\n", retime-ratime);

      ratime = _TAKE_TIME;

      // recover source
      Q_phi_tbc(g_spinor_field[1], g_spinor_field[0]);
      xchange_field(g_spinor_field[1]);

      retime = _TAKE_TIME;
      if(g_cart_id==0) fprintf(stdout, "# [jc_u_x] time to apply D_tm: %e seconds\n", retime-ratime);

      ratime = _TAKE_TIME;
      /* calculate real and imaginary part */
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]));

          // 1st contribution U_mu_x ( gamma_mu - 1 )
          _fv_eq_cm_ti_fv(spinor1, U_, g_spinor_field[0]+_GSI(g_iup[ix][mu]));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_mi_eq_fv(spinor2, spinor1);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[1]+_GSI(ix), spinor2);
          disc[_GWI(mu,ix,VOLUME)  ] = -0.5 * w.re;
          disc[_GWI(mu,ix,VOLUME)+1] = -0.5 * w.im;

          // 2nd contribution U_mu_x^dagger ( gamma_mu + 1 )
          _fv_eq_cm_dag_ti_fv(spinor1, U_, g_spinor_field[0]+_GSI(ix));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_pl_eq_fv(spinor2, spinor1);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[1]+_GSI(g_iup[ix][mu]), spinor2);
          disc[_GWI(mu,ix,VOLUME)  ] += -0.5 * w.re;
          disc[_GWI(mu,ix,VOLUME)+1] += -0.5 * w.im;
        }
      }
      retime = _TAKE_TIME;
      if(g_cart_id==0) fprintf(stdout, "# [jc_u_x] time to calculate contractions: %e seconds\n", retime-ratime);

      /************************************************
       * save results
       ************************************************/
      if(g_cart_id == 0) fprintf(stdout, "# [jc_u_x] save results for gauge id %d and sid %d\n", gid, sid);

      /* save the result in position space */
      fnorm = 1. / g_prop_normsqr;
      if(g_cart_id==0) fprintf(stdout, "# [jc_u_x] X-fnorm = %e\n", fnorm);
      for(ix=0; ix<8*VOLUME; ix++) disc[ix] *= fnorm;
      sprintf(filename, "jc_u_x.%.4d.%.4d", gid, sid);
      sprintf(contype, "jc^u in position space, re,im, no imp, normalized");
      write_lime_contraction(disc, filename, 32, 4, contype, gid, sid);

      if(write_ascii) {
        sprintf(filename, "jc_u_x.%.4d.%.4d.ascii", gid, sid);
        write_contraction (disc, NULL, filename, 4, 2, 0);
      }
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

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [jc_u_x] %s# [jc_u_x] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [jc_u_x] %s# [jc_u_x] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
