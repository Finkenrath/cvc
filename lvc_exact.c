/****************************************************
 * lvc_exact.c
 *
 * Fri Sep 18 15:38:51 CEST 2009
 *
 * PURPOSE:
 * - implement the vacuum polarization tensor by
 *   contraction of the local vector current correlator
 * - _NOT_ TESTED
 * DONE:
 * TODO:
 * - implementation
 * - checks
 ****************************************************/

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
#include "Q_phi.h"

#ifdef QMAT
#  define _CVC_RE_FACT (5./9.)
#  define _CVC_IM_FACT (1./3.)
#else
#  define _CVC_RE_FACT (1.)
#  define _CVC_IM_FACT (1.)
#endif

#ifdef AVCMAT
#  define _AVC_RE_FACT  (2.)
#  define _AVC_IM_FACT  (0.)
#else
#  define _AVC_RE_FACT  (1.)
#  define _AVC_IM_FACT  (1.)
#endif

void usage() {
  fprintf(stdout, "Code to perform local vector current correlator conn. contractions\n");
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
  
  int c, i, j, mu, nu, ir, is, ia, ib;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int source_location, have_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int sx0, sx1, sx2, sx3;
  double *conn = (double*)NULL;
  double *conn2 = (double*)NULL;
  double A[4][12][24];
  double V[4][12][24];
  double phase[4];
  int verbose = 0;
  int do_gt   = 0;
  char filename[800];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24];
  double *gauge_trafo=(double*)NULL;
  complex w, w1;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p;
  int *status;
#else
  fftwnd_plan plan_p;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vgf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'g':
      do_gt = 1;
      break;
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

  /**************************
   * set the default values *
   **************************/
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");

  /***********************
   * read the input file *
   ***********************/
  read_input(filename);

  /*********************************
   * some checks on the input data *
   *********************************/
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  /*****************************
   * initialize MPI parameters *
   *****************************/
  mpi_init(argc, argv);
#ifdef MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

  /*******************
   * initialize fftw *
   *******************/
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
#endif
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

#ifdef MPI
  if(T==0) {
    fprintf(stderr, "[%2d] local T is zero; exit\n", g_cart_id);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(2);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  /************************
   * read the gauge field *
   ************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  /* xchange_gauge(); */

  /*************************
   * measure the plaquette *
   *************************/
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /*****************************************
   * allocate memory for the spinor fields *
   *****************************************/
  no_fields = 24;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  conn  = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  conn2 = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  if( (conn==(double*)NULL) || (conn2==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] = 0.;
  for(ix=0; ix<32*VOLUME; ix++) conn2[ix] = 0.;

  /*****************************************
   * prepare Fourier transformation arrays *
   *****************************************/
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  if(do_gt==1) {
    /********************************
     * prepare gauge transformation *
     ********************************/
    init_gauge_trafo(&gauge_trafo, 1.0);
    apply_gt_gauge(gauge_trafo);
    /* measure the plaquette */
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);
  }

  /*********************************************************************************
   * determine source coordinates, find out, if source_location is in this process *
   *********************************************************************************/
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  if(have_source_flag==1) { 
    fprintf(stdout, "local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
  }
#ifdef MPI
  MPI_Gather(&have_source_flag, 1, MPI_INT, status, 1, MPI_INT, 0, g_cart_grid);
  if(g_cart_id==0) {
    for(mu=0; mu<g_nproc; mu++) fprintf(stdout, "status[%1d]=%d\n", mu,status[mu]);
  }
  if(g_cart_id==0) {
    for(have_source_flag=0; status[have_source_flag]!=1; have_source_flag++);
    fprintf(stdout, "have_source_flag= %d\n", have_source_flag);
  }
  MPI_Bcast(&have_source_flag, 1, MPI_INT, 0, g_cart_grid);
  fprintf(stdout, "[%2d] have_source_flag = %d\n", g_cart_id, have_source_flag);
#else
  have_source_flag = 0;
#endif

#ifdef QMAT
  if(g_cart_id==0) {
    fprintf(stdout, "Using matrix of electric charges for cvc\n");
    fprintf(stdout, "_CVC_RE_FACT = %f; _CVC_IM_FACT = %f\n", _CVC_RE_FACT, _CVC_IM_FACT);
  }
#else
  if(g_cart_id==0) fprintf(stdout, "Contracting single, up-type flavour case for cvc\n");
#endif

#ifdef AVCMAT
  if(g_cart_id==0) {
    fprintf(stdout, "Using PP-defined matrix  for avc\n");
    fprintf(stdout, "_AVC_RE_FACT = %f; _AVC_IM_FACT = %f\n", _AVC_RE_FACT, _AVC_IM_FACT);
  }
#else
  if(g_cart_id==0) fprintf(stdout, "Contracting single, up-type flavour case for avc\n");
#endif

  /********************************
   * contractions
   ********************************/

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  /* read 12 up-type propagators */
  if(do_gt==0) {
    for(ia=0; ia<12; ia++) {
      get_filename(filename, 4, ia, 1);
      read_lime_spinor(g_spinor_field[ia], filename, 0);
/*      xchange_field(g_spinor_field[ia]); */
    }
  }
  else {
    for(ia=0; ia<12; ia++) {
      apply_gt_prop(gauge_trafo, g_spinor_field[ia], ia/3, ia%3, 4, filename_prefix, source_location);
/*      xchange_field(g_spinor_field[ia]); */
    }
  }

  /* read 12 dn-type propagators */
  if(do_gt==0) {
    for(ia=0; ia<12; ia++) {
      get_filename(filename, 4, ia, -1);
      read_lime_spinor(g_spinor_field[12+ia], filename, 0);
/*      xchange_field(g_spinor_field[12+ia]); */
    }
  }
  else {
    for(ia=0; ia<12; ia++) {
      apply_gt_prop(gauge_trafo, g_spinor_field[12+ia], ia/3, ia%3, 4, filename_prefix2, source_location);
/*      xchange_field(g_spinor_field[12+ia]); */
    }
  }

  for(ix=0; ix<VOLUME; ix++) {
    for(ir=0; ir<4; ir++) {
    for(ia=0; ia<3; ia++) {
      for(is=0; is<4; is++) {
      for(ib=0; ib<3; ib++) {
     
        for(mu=0; mu<4; mu++) {

          /* contrib to lavc */
	  _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[3*ir+ia][_GSI(ix)]);
	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
	  A[mu][3*ir+ia][2*(3*is+ib)  ] = w.re;
	  A[mu][3*ir+ia][2*(3*is+ib)+1] = w.im;

          /* contrib to lvc */
	  _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor2);
	  V[mu][3*ir+ia][2*(3*is+ib)  ] = -w.re;
	  V[mu][3*ir+ia][2*(3*is+ib)+1] = -w.im;

        }/* of mu */

      } /* of ib */
      } /* of is */
    } /* of ia */
    } /* of ir */

    for(nu=0; nu<4; nu++) {
     
      /* take the trace of product gamma_nu A */
      for(mu=0; mu<4; mu++) {
        _co_eq_tr_gamma_sm(&w,nu,A[mu]);
        conn[_GWI(4*mu+nu, ix, VOLUME)  ] += _AVC_RE_FACT * w.re;
        conn[_GWI(4*mu+nu, ix, VOLUME)+1] += _AVC_IM_FACT * w.im;
      }

      /* take the trace of product gamma_nu V */
      for(mu=0; mu<4; mu++) {
        _co_eq_tr_gamma_sm(&w,6+nu,V[mu]);
        conn2[_GWI(4*mu+nu, ix, VOLUME)  ] += _CVC_RE_FACT * w.re;
        conn2[_GWI(4*mu+nu, ix, VOLUME)+1] += _CVC_IM_FACT * w.im;
      }
    }

  }/* of ix */

#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "contractions in %e seconds\n", retime-ratime);

  if(do_gt==0) free(gauge_trafo);

  /****************
   * save results *
   ****************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  sprintf(filename, "lvc_a_x.%.4d", Nconf);
  write_contraction(conn, (int*)NULL, filename, 16, 0, 0);
  sprintf(filename, "lvc_v_x.%.4d", Nconf);
  write_contraction(conn2, (int*)NULL, filename, 16, 0, 0);
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved position space results in %e seconds\n", retime-ratime);

  /**************************
   * Fourier transformation *
   **************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)&conn[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&conn[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)&conn2[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&conn2[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "Fourier transform in %e seconds\n", retime-ratime);

  /*******************************
   * save momentum space results *
   *******************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  sprintf(filename, "lvc_a_p.%.4d", Nconf);
  write_contraction(conn, (int*)NULL, filename, 16, 0, 0);
  sprintf(filename, "lvc_v_p.%.4d", Nconf);
  write_contraction(conn2, (int*)NULL, filename, 16, 0, 0);

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved momentum space results in %e seconds\n", retime-ratime);

  /* free the allocated memory, finalize */
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(conn);
  free(conn2);
#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
#endif

  return(0);

}
