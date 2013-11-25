/****************************************************
 * 
 * avc_exact.c
 *
 * Thu Aug 13 18:33:16 MEST 2009
 *
 * DONE:
 * - checked _co_eq_gammaU[dag]_sm in check_traceop
 *   and $WORK/test_avc/1.0/
 * - checked serial/parallel version in free/non-free case
 * - results of checks in $WORK/test_avc/1.x
 * TODO:
 * - add flavour structure factors for AVC, SCAL, XAVC, PSEU
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
#include "contractions_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"

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
#  define _PSEU_RE_FACT (2.)
#  define _PSEU_IM_FACT (0.)
#  define _SCAL_RE_FACT (0.)
#  define _SCAL_IM_FACT (2.)
#  define _XAVC_RE_FACT (2.)
#  define _XAVC_IM_FACT (0.)
#else
#  define _AVC_RE_FACT  (1.)
#  define _AVC_IM_FACT  (1.)
#  define _PSEU_RE_FACT (1.)
#  define _PSEU_IM_FACT (1.)
#  define _SCAL_RE_FACT (1.)
#  define _SCAL_IM_FACT (1.)
#  define _XAVC_RE_FACT (1.)
#  define _XAVC_IM_FACT (1.)
#endif

void usage() {
  fprintf(stdout, "Code to perform AV current correlator conn. contractions\n");
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
  double *scal = (double*)NULL;
  double *pseu = (double*)NULL;
  double *xavc = (double*)NULL;
  double *conn2 = (double*)NULL;
  double C13[4][12][24], C24[4][12][24];
  double CT[12][24], X[12][24], P[12][24], S[12][24];
  double V[4][12][24];
  double contact_term[8], contact_term2[8];
  double phase[4];
  int verbose = 0;
  int do_gt   = 0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_trafo=(double*)NULL;
  complex w, w1;
  double Usourcebuff[72], *Usource[4];
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

  /* initialize MPI parameters */
  mpi_init(argc, argv);
#ifdef MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

  /* initialize fftw */
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

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(strcmp(gaugefilename_prefix, "identity") == 0) {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [avc_exact] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  } else {
    /* read the gauge field from file */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "\n# [avc_exact] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  }
#ifdef MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /* allocate memory for the spinor fields */
  no_fields = 24;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  conn = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  scal = (double*)calloc(2 *  4 * VOLUME, sizeof(double));
  pseu = (double*)calloc(2 *  4 * VOLUME, sizeof(double));
  xavc = (double*)calloc(2 *  4 * VOLUME, sizeof(double));
  conn2 = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  if( (conn==(double*)NULL) || (scal==(double*)NULL) ||
    (pseu==(double*)NULL) || (xavc==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] = 0.;
  for(ix=0; ix< 8*VOLUME; ix++) scal[ix] = 0.;
  for(ix=0; ix< 8*VOLUME; ix++) pseu[ix] = 0.;
  for(ix=0; ix< 8*VOLUME; ix++) xavc[ix] = 0.;
  for(ix=0; ix<32*VOLUME; ix++) conn2[ix] = 0.;

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  if(do_gt==1) {
    /* prepare gauge transformation */
    init_gauge_trafo(&gauge_trafo, 1.0);
    apply_gt_gauge(gauge_trafo);
    /* measure the plaquette */
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);
  }

  /* determine source coordinates, find out, if source_location is in this process */
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  Usource[0] = Usourcebuff;
  Usource[1] = Usourcebuff+18;
  Usource[2] = Usourcebuff+36;
  Usource[3] = Usourcebuff+54;
  if(have_source_flag==1) { 
    fprintf(stdout, "local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
    _cm_eq_cm_ti_co(Usource[0], &g_gauge_field[_GGI(source_location,0)], &co_phase_up[0]);
    _cm_eq_cm_ti_co(Usource[1], &g_gauge_field[_GGI(source_location,1)], &co_phase_up[1]);
    _cm_eq_cm_ti_co(Usource[2], &g_gauge_field[_GGI(source_location,2)], &co_phase_up[2]);
    _cm_eq_cm_ti_co(Usource[3], &g_gauge_field[_GGI(source_location,3)], &co_phase_up[3]);
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
  MPI_Bcast(Usourcebuff, 72, MPI_DOUBLE, have_source_flag, g_cart_grid);
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


#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  /* loop on right Lorentz index nu */
  for(nu=0; nu<4; nu++) {


    /********************************
     * first and third contribution
     ********************************/

    /* read 12 up-type propagators */
    if(do_gt==0) {
      for(ia=0; ia<12; ia++) {
/*
        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted",
          filename_prefix, Nconf, 4, ia);
*/
        get_filename(filename, 4, ia, 1);
        read_lime_spinor(g_spinor_field[ia], filename, 0);
        xchange_field(g_spinor_field[ia]);
      }
    }
    else {
      for(ia=0; ia<12; ia++) {
        apply_gt_prop(gauge_trafo, g_spinor_field[ia], ia/3, ia%3, 4, filename_prefix, source_location);
        xchange_field(g_spinor_field[ia]);
      }
    }

    /* read 12 dn-type propagators */
    if(do_gt==0) {
      for(ia=0; ia<12; ia++) {
/*
        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted",
          filename_prefix2, Nconf, nu, ia);
*/
        get_filename(filename, nu, ia, -1);
        read_lime_spinor(g_spinor_field[12+ia], filename, 0);
        xchange_field(g_spinor_field[12+ia]);
      }
    }
    else {
      for(ia=0; ia<12; ia++) {
        apply_gt_prop(gauge_trafo, g_spinor_field[12+ia], ia/3, ia%3, nu, filename_prefix2, source_location);
        xchange_field(g_spinor_field[12+ia]);
      }
    }

    for(ix=0; ix<VOLUME; ix++) {
      /* set X to zero */
      for(i=0; i<12; i++) {
        for(j=0; j<24; j++) X[i][j] = 0.;
      }  

      for(ir=0; ir<4; ir++) {
      for(ia=0; ia<3; ia++) {
        for(is=0; is<4; is++) {
        for(ib=0; ib<3; ib++) {
     
          for(mu=0; mu<4; mu++) {

          
            _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

	    _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[3*ir+ia][_GSI(g_iup[ix][mu])]);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
	    X[3*ir+ia][2*(3*is+ib)  ] += w.re;
	    X[3*ir+ia][2*(3*is+ib)+1] += w.im;

	    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor2);
	    C13[mu][3*ir+ia][2*(3*is+ib)  ] = w.re;
	    C13[mu][3*ir+ia][2*(3*is+ib)+1] = w.im;

            /* contrib to cvc */
	    _fv_mi_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
	    V[mu][3*ir+ia][2*(3*is+ib)  ] = w.re;
	    V[mu][3*ir+ia][2*(3*is+ib)+1] = w.im;
            if(ix==0) fprintf(stdout, "[1] %3d%3d%3d%3d\t%e\t%e\n", ir, is, ia, ib, w.re, w.im);

	    _fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[3*ir+ia][_GSI(ix)]);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(g_iup[ix][mu])], spinor1);
	    X[3*ir+ia][2*(3*is+ib)  ] += w.re;
	    X[3*ir+ia][2*(3*is+ib)+1] += w.im;

	    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(g_iup[ix][mu])], spinor2);
	    C13[mu][3*ir+ia][2*(3*is+ib)  ] += w.re;
	    C13[mu][3*ir+ia][2*(3*is+ib)+1] += w.im;

            /* contrib to cvc */
	    _fv_pl_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(g_iup[ix][mu])], spinor1);
	    V[mu][3*ir+ia][2*(3*is+ib)  ] += w.re;
	    V[mu][3*ir+ia][2*(3*is+ib)+1] += w.im;
            if(ix==0) fprintf(stdout, "[3] %3d%3d%3d%3d\t%e\t%e\n", ir, is, ia, ib, w.re, w.im);

            _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(g_idn[ix][mu],mu)], &co_phase_up[mu]);

	    _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[3*ir+ia][_GSI(ix)]);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(g_idn[ix][mu])], spinor1);
	    X[3*ir+ia][2*(3*is+ib)  ] += w.re;
	    X[3*ir+ia][2*(3*is+ib)+1] += w.im;

	    _fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[3*ir+ia][_GSI(g_idn[ix][mu])]);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
	    X[3*ir+ia][2*(3*is+ib)  ] += w.re;
	    X[3*ir+ia][2*(3*is+ib)+1] += w.im;

          }/* of mu */

          /* contribution to pseu */
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], &g_spinor_field[3*ir+ia][_GSI(ix)]);
          P[3*ir+ia][2*(3*is+ib)  ] = w.re;
          P[3*ir+ia][2*(3*is+ib)+1] = w.im;

          X[3*ir+ia][2*(3*is+ib)  ] -= 16. * w.re;
          X[3*ir+ia][2*(3*is+ib)+1] -= 16. * w.im;

          /* contribution to scal */
          _fv_eq_gamma_ti_fv(spinor1, 5, &g_spinor_field[3*ir+ia][_GSI(ix)]);
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
          S[3*ir+ia][2*(3*is+ib)  ] = w.re;
          S[3*ir+ia][2*(3*is+ib)+1] = w.im;

	} /* of ib */
	} /* of is */
      } /* of ia */
      } /* of ir */


      /* take the trace of product gamma_nu U_nu (y) C13 */
      for(mu=0; mu<4; mu++) {
        _co_eq_tr_gammaU_sm(&w,nu,Usource[nu],C13[mu]);
	conn[_GWI(4*mu+nu, ix, VOLUME)  ] += _AVC_RE_FACT * w.re;
	conn[_GWI(4*mu+nu, ix, VOLUME)+1] += _AVC_IM_FACT * w.im;
      }
      _co_eq_tr_gammaU_sm(&w,nu,Usource[nu],P);
      pseu[_GWI(nu, ix, VOLUME)  ] += _PSEU_RE_FACT * w.re;
      pseu[_GWI(nu, ix, VOLUME)+1] += _PSEU_IM_FACT * w.im;
      
      _co_eq_tr_gammaU_sm(&w,nu,Usource[nu],S);
      scal[_GWI(nu, ix, VOLUME)  ] += _SCAL_RE_FACT * w.re;
      scal[_GWI(nu, ix, VOLUME)+1] += _SCAL_IM_FACT * w.im;
      
      _co_eq_tr_gammaU_sm(&w,nu,Usource[nu],X);
      xavc[_GWI(nu, ix, VOLUME)  ] += _XAVC_RE_FACT * w.re;
      xavc[_GWI(nu, ix, VOLUME)+1] += _XAVC_IM_FACT * w.im;

      for(mu=0; mu<4; mu++) {
/*
        if(ix==0) {
          for(i=0;i<12;i++) {
          for(j=0;j<12;j++) {
            fprintf(stdout, "1_3 V %d %d %d %d  %e %e\n", i/3,i%3, j/3,j%3, V[mu][i][2*j], V[mu][i][2*j+1]);
          }}
        }


        fprintf(stdout, "contrib 1/3\tix=%6d\tnu=%6d\tmu=%6d\n", ix, nu, mu);
	for(i=0; i<12; i++) {
	for(j=0; j<12; j++) {
	  fprintf(stdout, "%3d%3d%21.12e%21.12e\n", i, j, V[mu][i][2*j], V[mu][i][2*j+1]);
	}
	}
*/
        _co_eq_tr_gammaU_sm(&w,6+nu,Usource[nu],V[mu]);
	conn2[_GWI(4*mu+nu, ix, VOLUME)  ] += _CVC_RE_FACT * w.re;
	conn2[_GWI(4*mu+nu, ix, VOLUME)+1] += _CVC_IM_FACT * w.im;
        //if(ix==0) fprintf(stdout, "[%d, %d] 1/3 nu5 = %e +I %e\n", mu, nu, w.re, w.im);
        _co_eq_tr_gammaU_sm(&w,5,Usource[nu],V[mu]);
	conn2[_GWI(4*mu+nu, ix, VOLUME)  ] -= _CVC_RE_FACT * w.re;
	conn2[_GWI(4*mu+nu, ix, VOLUME)+1] -= _CVC_IM_FACT * w.im;
        //if(ix==0) fprintf(stdout, "[%d, %d] 1/3 5 = %e +I %e\n", mu, nu, w.re, w.im);
      }
    }/* of ix */

    /* contribution to contact term */
    if(have_source_flag==g_cart_id) {
      for(ir=0; ir<12; ir++) 
        memcpy((void*)CT[ir], (void*)&g_spinor_field[ir][_GSI(g_iup[source_location][nu])], 24*sizeof(double));
      _co_eq_tr_gammaU_sm(&w, nu, Usource[nu], CT);
      contact_term[2*nu  ] = _AVC_RE_FACT * w.re;
      contact_term[2*nu+1] = _AVC_IM_FACT * w.im;

      _co_eq_tr_gammaU_sm(&w, nu, Usource[nu], CT);
      contact_term2[2*nu  ] = _CVC_RE_FACT * w.re;
      contact_term2[2*nu+1] = _CVC_IM_FACT * w.im; 
      _co_eq_tr_gammaU_sm(&w, 4, Usource[nu], CT);
      contact_term2[2*nu  ] -= _CVC_RE_FACT * w.re;
      contact_term2[2*nu+1] -= _CVC_IM_FACT * w.im; 
    }

    /*********************************
     * second and fourth contribution
     *********************************/

    /* read 12 up-type propagators */
    if(do_gt==0) {
      for(ia=0; ia<12; ia++) {
/*
        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted",
          filename_prefix, Nconf, nu, ia);
*/
        get_filename(filename, nu, ia, 1);
        read_lime_spinor(g_spinor_field[ia], filename, 0);
        xchange_field(g_spinor_field[ia]);
      }
    }
    else {
      for(ia=0; ia<12; ia++) {
        apply_gt_prop(gauge_trafo, g_spinor_field[ia], ia/3, ia%3, nu, filename_prefix, source_location);
        xchange_field(g_spinor_field[ia]);
      }
    }

    /* read 12 dn-type propagators */
    if(do_gt==0) {
      for(ia=0; ia<12; ia++) {
/*
        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted",
          filename_prefix2, Nconf, 4, ia);
*/
        get_filename(filename, 4, ia, -1);
        read_lime_spinor(g_spinor_field[12+ia], filename, 0);
        xchange_field(g_spinor_field[12+ia]);
      }
    }
    else {
      for(ia=0; ia<12; ia++) {
        apply_gt_prop(gauge_trafo, g_spinor_field[12+ia], ia/3, ia%3, 4, filename_prefix2, source_location);
        xchange_field(g_spinor_field[12+ia]);
      }
    }

    for(ix=0; ix<VOLUME; ix++) {
      /* set X to zero */
      for(i=0; i<12; i++) {
        for(j=0; j<24; j++) X[i][j] = 0.;
      }

      for(ir=0; ir<4; ir++) {
      for(ia=0; ia<3; ia++) {
        for(is=0; is<4; is++) {
        for(ib=0; ib<3; ib++) {
      
          for(mu=0; mu<4; mu++) {

            _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

	    _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[3*ir+ia][_GSI(g_iup[ix][mu])]);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
	    X[3*ir+ia][2*(3*is+ib)  ] += w.re;
	    X[3*ir+ia][2*(3*is+ib)+1] += w.im;

	    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor2);
	    C24[mu][3*ir+ia][2*(3*is+ib)  ] = w.re;
	    C24[mu][3*ir+ia][2*(3*is+ib)+1] = w.im;

            /* contrib to cvc */
	    _fv_mi_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
	    V[mu][3*ir+ia][2*(3*is+ib)  ] = w.re;
	    V[mu][3*ir+ia][2*(3*is+ib)+1] = w.im;
            if(ix==0) fprintf(stdout, "[2] %3d%3d%3d%3d\t%e\t%e\n", ir, is, ia, ib, w.re, w.im);

	    _fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[3*ir+ia][_GSI(ix)]);
            _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(g_iup[ix][mu])], spinor1);
	    X[3*ir+ia][2*(3*is+ib)  ] += w.re;
	    X[3*ir+ia][2*(3*is+ib)+1] += w.im;

	    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(g_iup[ix][mu])], spinor2);
	    C24[mu][3*ir+ia][2*(3*is+ib)  ] += w.re;
	    C24[mu][3*ir+ia][2*(3*is+ib)+1] += w.im;

            /* contrib to cvc */
	    _fv_pl_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(g_iup[ix][mu])], spinor1);
	    V[mu][3*ir+ia][2*(3*is+ib)  ] += w.re;
	    V[mu][3*ir+ia][2*(3*is+ib)+1] += w.im;
            if(ix==0) fprintf(stdout, "[4] %3d%3d%3d%3d\t%e\t%e\n", ir, is, ia, ib, w.re, w.im);

	    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(g_idn[ix][mu], mu)], &co_phase_up[mu]);

	    _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[3*ir+ia][_GSI(ix)]);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(g_idn[ix][mu])], spinor1);
	    X[3*ir+ia][2*(3*is+ib)  ] += w.re;
	    X[3*ir+ia][2*(3*is+ib)+1] += w.im;
             
	    _fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[3*ir+ia][_GSI(g_idn[ix][mu])]);
	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
	    X[3*ir+ia][2*(3*is+ib)  ] += w.re;
	    X[3*ir+ia][2*(3*is+ib)+1] += w.im;

          }/* of mu */
          
          /* contribution to pseu */
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], &g_spinor_field[3*ir+ia][_GSI(ix)]);
          P[3*ir+ia][2*(3*is+ib)  ] = w.re;
          P[3*ir+ia][2*(3*is+ib)+1] = w.im;

          X[3*ir+ia][2*(3*is+ib)  ] -= 16. * w.re;
          X[3*ir+ia][2*(3*is+ib)+1] -= 16. * w.im;

          /* contribution to scal */
          _fv_eq_gamma_ti_fv(spinor1, 5, &g_spinor_field[3*ir+ia][_GSI(ix)]);
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[12+3*is+ib][_GSI(ix)], spinor1);
          S[3*ir+ia][2*(3*is+ib)  ] = w.re;
          S[3*ir+ia][2*(3*is+ib)+1] = w.im;

	}
	}
      }
      }

      /* take the trace of product gamma_nu U_nu (y)^+ C24 */
      for(mu=0; mu<4; mu++) {
        _co_eq_tr_gammaUdag_sm(&w,nu,Usource[nu],C24[mu]);
	conn[_GWI(4*mu+nu, ix, VOLUME)  ] += _AVC_RE_FACT * w.re;
	conn[_GWI(4*mu+nu, ix, VOLUME)+1] += _AVC_IM_FACT * w.im;
      }
      _co_eq_tr_gammaUdag_sm(&w, nu, Usource[nu], P);
      pseu[_GWI(nu, ix, VOLUME)  ] += _PSEU_RE_FACT * w.re;
      pseu[_GWI(nu, ix, VOLUME)+1] += _PSEU_IM_FACT * w.im;
      
      _co_eq_tr_gammaUdag_sm(&w, nu, Usource[nu], S);
      scal[_GWI(nu, ix, VOLUME)  ] += _SCAL_RE_FACT * w.re;
      scal[_GWI(nu, ix, VOLUME)+1] += _SCAL_IM_FACT * w.im;
      
      _co_eq_tr_gammaUdag_sm(&w, nu, Usource[nu], X);
      xavc[_GWI(nu, ix, VOLUME)  ] += _XAVC_RE_FACT * w.re;
      xavc[_GWI(nu, ix, VOLUME)+1] += _XAVC_IM_FACT * w.im;

      for(mu=0; mu<4; mu++) {


/*      
        fprintf(stdout, "contrib 2/4\tix=%6d\tnu=%6d\tmu=%6d\n", ix, nu, mu);
	for(i=0; i<12; i++) {
	for(j=0; j<12; j++) {
	  fprintf(stdout, "%3d%3d%21.12e%21.12e\n", i, j, V[mu][i][2*j], V[mu][i][2*j+1]);
	}
	}

        if(ix==0) {
          for(i=0;i<12;i++) {
          for(j=0;j<12;j++) {
            fprintf(stdout, "2_4 V %d %d %d %d  %e %e\n", i/3, i%3, j/3, j%3, V[mu][i][2*j], V[mu][i][2*j+1]);
          }}
        }
*/
        _co_eq_tr_gammaUdag_sm(&w,6+nu,Usource[nu],V[mu]);
	conn2[_GWI(4*mu+nu, ix, VOLUME)  ] += _CVC_RE_FACT * w.re;
	conn2[_GWI(4*mu+nu, ix, VOLUME)+1] += _CVC_IM_FACT * w.im;
        //if(ix==0) fprintf(stdout, "[%d, %d] 2/4 nu5 = %e +I %e\n", mu, nu, w.re, w.im);
        _co_eq_tr_gammaUdag_sm(&w,5,Usource[nu],V[mu]);
	conn2[_GWI(4*mu+nu, ix, VOLUME)  ] += _CVC_RE_FACT * w.re;
	conn2[_GWI(4*mu+nu, ix, VOLUME)+1] += _CVC_IM_FACT * w.im;
        //if(ix==0) fprintf(stdout, "[%d, %d] 2/4 5 = %e +I %e\n", mu, nu, w.re, w.im);
      }
      
    }/* of ix */

    /* contribution to contact term */
    if(have_source_flag==g_cart_id) {
      for(ir=0; ir<12; ir++) 
        memcpy((void*)CT[ir], (void*)&g_spinor_field[ir][_GSI(source_location)], 24*sizeof(double));
      _co_eq_tr_gammaUdag_sm(&w,nu,Usource[nu],CT);
      contact_term[2*nu  ] -= _AVC_RE_FACT * w.re;
      contact_term[2*nu+1] -= _AVC_IM_FACT * w.im;

      _co_eq_tr_gammaUdag_sm(&w, nu, Usource[nu], CT);
      contact_term2[2*nu  ] -= _CVC_RE_FACT * w.re;
      contact_term2[2*nu+1] -= _CVC_IM_FACT * w.im;
      _co_eq_tr_gammaUdag_sm(&w, 4, Usource[nu], CT);
      contact_term2[2*nu  ] -= _CVC_RE_FACT * w.re;
      contact_term2[2*nu+1] -= _CVC_IM_FACT * w.im;
    }

  }/* of loop on nu*/

  /* normalisation of contractions */
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] /= 4.;
  for(ix=0; ix< 8*VOLUME; ix++) {
    pseu[ix] /= -2.;
    scal[ix] /= -2.;
    xavc[ix] /= 4.;
  }
  for(ix=0; ix< 8; ix++) contact_term[ix] /= 2.;

  for(ix=0; ix<32*VOLUME; ix++) conn2[ix] /= -4.;
  for(ix=0; ix< 8; ix++) contact_term2[ix] /= 2.;

#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "contractions in %e seconds\n", retime-ratime);

  if(do_gt==0) free(gauge_trafo);
  
  /* save results */
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

#ifndef MPI
  /* save the result in position space */
/*
  sprintf(filename, "avc_a_x.%.4d", Nconf);
  if((ofs = fopen(filename, "w"))==(FILE*)NULL) {
    fprintf(stderr, "could not open file %s for writing\n", filename);
    exit(8);
  }
  fprintf(ofs, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, LX, LY, LZ, T, g_kappa, g_mu);
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(ofs, "#x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", x0, x1, x2, x3);
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<16; mu++) {
      fprintf(ofs, "%3d%3d%25.16e%25.16e\n", mu/4, mu%4, conn[_GWI(mu, ix, VOLUME)], conn[_GWI(mu, ix, VOLUME)+1]);
    }
  }
  }
  }
  }
  fprintf(ofs, "#========================================\n");
  fprintf(ofs, "#contact term:\n");
  for(i=0; i<4; i++) fprintf(ofs, "#%3d%25.16e%25.16e\n", i, contact_term[2*i], contact_term[2*i+1]);
  fclose(ofs);

  sprintf(filename, "avc_v_x.%.4d", Nconf);
  if((ofs = fopen(filename, "w"))==(FILE*)NULL) {
    fprintf(stderr, "could not open file %s for writing\n", filename);
    exit(8);
  }
  fprintf(ofs, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, LX, LY, LZ, T, g_kappa, g_mu);
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(ofs, "#x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", x0, x1, x2, x3);
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<16; mu++) {
      fprintf(ofs, "%3d%3d%25.16e%25.16e\n", mu/4, mu%4, conn2[_GWI(mu, ix, VOLUME)], conn2[_GWI(mu, ix, VOLUME)+1]);
    }
  }
  }
  }
  }
  fprintf(ofs, "#========================================\n");
  fprintf(ofs, "#contact term:\n");
  for(i=0; i<4; i++) fprintf(ofs, "#%3d%25.16e%25.16e\n", i, contact_term2[2*i], contact_term2[2*i+1]);
  fclose(ofs);

  sprintf(filename, "avc_psx_x.%.4d", Nconf);
  if((ofs = fopen(filename, "w"))==(FILE*)NULL) {
    fprintf(stderr, "could not open file %s for writing\n", filename);
    exit(9);
  }
  fprintf(ofs, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, LX, LY, LZ, T, g_kappa, g_mu);
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "#%3d%3d%3d%3d\n", x0, x1, x2, x3);
    for(mu=0; mu<4; mu++) {
      fprintf(ofs, "#%3d%21.12e%21.12e%21.12e%21.12e%21.12e%21.12e\n", mu,
        pseu[_GWI(mu, ix, VOLUME)], pseu[_GWI(mu, ix, VOLUME)+1],
	scal[_GWI(mu, ix, VOLUME)], scal[_GWI(mu, ix, VOLUME)+1],
	xavc[_GWI(mu, ix, VOLUME)], xavc[_GWI(mu, ix, VOLUME)+1]);
    }
  }
  }
  }
  }
  fclose(ofs);
*/
#endif

  //sprintf(filename, "avc_a_x.%.4d", Nconf);
  //write_contraction(conn, (int*)NULL, filename, 16, 0, 0);

  sprintf(filename, "avc_v_x.%.4d", Nconf);
  sprintf(contype, "cvc");
  //write_contraction(conn2, (int*)NULL, filename, 16, 0, 0);
  write_lime_contraction(conn2, filename, 64, 16, contype, Nconf, 0);

  sprintf(filename, "avc_v_x.%.4d.ascii", Nconf);
  write_contraction(conn2, NULL, filename, 16, 2, 0);

  //sprintf(filename, "avc_p_x.%.4d", Nconf);
  //write_contraction(pseu, (int*)NULL, filename, 4, 0, 0);
  //sprintf(filename, "avc_s_x.%.4d", Nconf);
  //write_contraction(scal, (int*)NULL, filename, 4, 0, 0);
  //sprintf(filename, "avc_x_x.%.4d", Nconf);
  //write_contraction(xavc, (int*)NULL, filename, 4, 0, 0);

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved position space results in %e seconds\n", retime-ratime);



#ifndef MPI
  /* check the Ward identity in position space */
/*
  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(stdout, "#a %3d%3d%3d%3d\n", x0, x1, x2, x3);
    ix=g_ipt[x0][x1][x2][x3];
    for(nu=0; nu<4; nu++) {
      w.re = conn[_GWI(4*0+nu,ix,VOLUME)] + conn[_GWI(4*1+nu,ix,VOLUME)]
           + conn[_GWI(4*2+nu,ix,VOLUME)] + conn[_GWI(4*3+nu,ix,VOLUME)]
	   - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)]
	   - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)];

      w.im = conn[_GWI(4*0+nu,ix,VOLUME)+1] + conn[_GWI(4*1+nu,ix,VOLUME)+1]
           + conn[_GWI(4*2+nu,ix,VOLUME)+1] + conn[_GWI(4*3+nu,ix,VOLUME)+1]
	   - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)+1] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)+1]
	   - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)+1] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)+1];
      
      w1.re = 2.*(1/(2.*g_kappa)-4.)*pseu[_GWI(nu,ix,VOLUME)] - 2.*g_mu*scal[_GWI(nu,ix,VOLUME)+1]
              + xavc[_GWI(nu,ix,VOLUME)];
      w1.im = 2.*(1/(2.*g_kappa)-4.)*pseu[_GWI(nu,ix,VOLUME)+1] + 2.*g_mu*scal[_GWI(nu,ix,VOLUME)]
              + xavc[_GWI(nu,ix,VOLUME)+1];

      fprintf(stdout, "#a %3d%25.16e%25.16e%25.16e%25.16e\n", nu, w.re, w.im, w1.re, w1.im);

    }
  }
  }
  }
  }
*/
  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(stdout, "#v %3d%3d%3d%3d\n", x0, x1, x2, x3);
    ix=g_ipt[x0][x1][x2][x3];
    for(nu=0; nu<4; nu++) {
      w.re = conn2[_GWI(4*0+nu,ix,VOLUME)] + conn2[_GWI(4*1+nu,ix,VOLUME)]
           + conn2[_GWI(4*2+nu,ix,VOLUME)] + conn2[_GWI(4*3+nu,ix,VOLUME)]
	   - conn2[_GWI(4*0+nu,g_idn[ix][0],VOLUME)] - conn2[_GWI(4*1+nu,g_idn[ix][1],VOLUME)]
	   - conn2[_GWI(4*2+nu,g_idn[ix][2],VOLUME)] - conn2[_GWI(4*3+nu,g_idn[ix][3],VOLUME)];

      w.im = conn2[_GWI(4*0+nu,ix,VOLUME)+1] + conn2[_GWI(4*1+nu,ix,VOLUME)+1]
           + conn2[_GWI(4*2+nu,ix,VOLUME)+1] + conn2[_GWI(4*3+nu,ix,VOLUME)+1]
	   - conn2[_GWI(4*0+nu,g_idn[ix][0],VOLUME)+1] - conn2[_GWI(4*1+nu,g_idn[ix][1],VOLUME)+1]
	   - conn2[_GWI(4*2+nu,g_idn[ix][2],VOLUME)+1] - conn2[_GWI(4*3+nu,g_idn[ix][3],VOLUME)+1];
      
      fprintf(stdout, "#v %3d%25.16e%25.16e\n", nu, w.re, w.im);

    }
  }
  }
  }
  }
#endif

  /* Fourier transformation */
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

  for(mu=0; mu<4; mu++) {
    memcpy((void*)in, (void*)&pseu[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&pseu[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

  for(mu=0; mu<4; mu++) {
    memcpy((void*)in, (void*)&scal[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&scal[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

  for(mu=0; mu<4; mu++) {
    memcpy((void*)in, (void*)&xavc[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&xavc[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

#ifdef MPI
  MPI_Bcast(contact_term, 8, MPI_DOUBLE, have_source_flag, g_cart_grid);
  fprintf(stdout, "[%2d] contact term = (%12.5e+%12.5eI,%12.5e+%12.5eI,%12.5e+%12.5eI,%12.5e+%12.5eI)\n",
    g_cart_id, contact_term[0], contact_term[1], contact_term[2], contact_term[3], 
    contact_term[4], contact_term[5], contact_term[6], contact_term[7]);
  MPI_Bcast(contact_term2, 8, MPI_DOUBLE, have_source_flag, g_cart_grid);
  fprintf(stdout, "[%2d] contact_term2 = (%12.5e+%12.5eI,%12.5e+%12.5eI,%12.5e+%12.5eI,%12.5e+%12.5eI)\n",
    g_cart_id, contact_term2[0], contact_term2[1], contact_term2[2], contact_term2[3], 
    contact_term2[4], contact_term2[5], contact_term2[6], contact_term2[7]);
#endif

  /* add phase factors */
  for(x0=0; x0<T; x0++) {
    phase[0] = (double)(Tstart+x0) * M_PI / (double)T_global;
  for(x1=0; x1<LX; x1++) {
    phase[1] = (double)(x1) * M_PI / (double)LX;
  for(x2=0; x2<LY; x2++) {
    phase[2] = (double)(x2) * M_PI / (double)LY;
  for(x3=0; x3<LZ; x3++) {
    phase[3] = (double)(x3) * M_PI / (double)LZ;
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      w.re = cos( phase[mu]-phase[nu]-2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      w.im = sin( phase[mu]-phase[nu]-2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      _co_eq_co_ti_co(&w1,(complex*)&conn[_GWI(4*mu+nu,ix,VOLUME)],&w);
      conn[_GWI(4*mu+nu,ix,VOLUME)  ] = w1.re;
      conn[_GWI(4*mu+nu,ix,VOLUME)+1] = w1.im;
      if(mu==nu) {
        conn[_GWI(5*mu,ix,VOLUME)  ] += contact_term[2*mu  ];
        conn[_GWI(5*mu,ix,VOLUME)+1] += contact_term[2*mu+1];
      }

      _co_eq_co_ti_co(&w1,(complex*)&conn2[_GWI(4*mu+nu,ix,VOLUME)],&w);
      conn2[_GWI(4*mu+nu,ix,VOLUME)  ] = w1.re;
      conn2[_GWI(4*mu+nu,ix,VOLUME)+1] = w1.im;
      if(mu==nu) {
        conn2[_GWI(5*mu,ix,VOLUME)  ] += contact_term2[2*mu  ];
        conn2[_GWI(5*mu,ix,VOLUME)+1] += contact_term2[2*mu+1];
      }
    }  /* of nu */
      /* phase factors for pseu and scal */
      w.re = cos( -phase[mu]-2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      w.im = sin( -phase[mu]-2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      _co_eq_co_ti_co(&w1,(complex*)&pseu[_GWI(mu,ix,VOLUME)],&w);
      pseu[_GWI(mu,ix,VOLUME)  ] = w1.re;
      pseu[_GWI(mu,ix,VOLUME)+1] = w1.im;

      _co_eq_co_ti_co(&w1,(complex*)&scal[_GWI(mu,ix,VOLUME)],&w);
      scal[_GWI(mu,ix,VOLUME)  ] = w1.re;
      scal[_GWI(mu,ix,VOLUME)+1] = w1.im;

      _co_eq_co_ti_co(&w1,(complex*)&xavc[_GWI(mu,ix,VOLUME)],&w);
      xavc[_GWI(mu,ix,VOLUME)  ] = w1.re;
      xavc[_GWI(mu,ix,VOLUME)+1] = w1.im;

    }  /* of mu */
  }
  }
  }
  }

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "Fourier transform in %e seconds\n", retime-ratime);

  /* save momentum space results */
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

#ifndef MPI
  /* save the result in momentum space */
/*
  sprintf(filename, "avc_a_p.%.4d", Nconf);
  if((ofs = fopen(filename, "w"))==(FILE*)NULL) {
    fprintf(stderr, "could not open file %s for writing\n", filename);
    exit(8);
  }
  fprintf(ofs, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, LX, LY, LZ, T, g_kappa, g_mu);
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(ofs, "#x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", x0, x1, x2, x3);
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<16; mu++) {
      fprintf(ofs, "%3d%3d%25.16e%25.16e\n", mu/4, mu%4, conn[_GWI(mu, ix, VOLUME)], conn[_GWI(mu, ix, VOLUME)+1]);
    }
  }
  }
  }
  }
  fclose(ofs);

  sprintf(filename, "avc_v_p.%.4d", Nconf);
  if((ofs = fopen(filename, "w"))==(FILE*)NULL) {
    fprintf(stderr, "could not open file %s for writing\n", filename);
    exit(8);
  }
  fprintf(ofs, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, LX, LY, LZ, T, g_kappa, g_mu);
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(ofs, "#x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", x0, x1, x2, x3);
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<16; mu++) {
      fprintf(ofs, "%3d%3d%25.16e%25.16e\n", mu/4, mu%4, conn2[_GWI(mu, ix, VOLUME)], conn2[_GWI(mu, ix, VOLUME)+1]);
    }
  }
  }
  }
  }
  fclose(ofs);

  sprintf(filename, "avc_psx_p.%.4d", Nconf);
  if((ofs = fopen(filename, "w"))==(FILE*)NULL) {
    fprintf(stderr, "could not open file %s for writing\n", filename);
    exit(8);
  }
  fprintf(ofs, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, LX, LY, LZ, T, g_kappa, g_mu);
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(ofs, "#x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", x0, x1, x2, x3);
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<4; mu++) {
      fprintf(ofs, "%3d%21.12e%21.12e%21.12e%21.12e%21.12e%21.12e\n", mu,
        pseu[_GWI(mu, ix, VOLUME)], pseu[_GWI(mu, ix, VOLUME)+1],
        scal[_GWI(mu, ix, VOLUME)], scal[_GWI(mu, ix, VOLUME)+1],
        xavc[_GWI(mu, ix, VOLUME)], xavc[_GWI(mu, ix, VOLUME)+1]);
    }
  }
  }
  }
  }
  fclose(ofs);
*/
#endif

  //sprintf(filename, "avc_a_p.%.4d", Nconf);
  //sprintf(contype, "axial_current-with-axial_current");
  /* write_contraction(conn, (int*)NULL, filename, 16, 0, 0); */
  //write_lime_contraction(conn,  filename, 64, 16, contype, Nconf, 0);

  sprintf(filename, "avc_v_p.%.4d", Nconf);
  sprintf(contype, "vector_current-with-vector_current");
  /* write_contraction(conn2, (int*)NULL, filename, 16, 0, 0); */
  write_lime_contraction(conn2, filename, 64, 16, contype, Nconf, 0);
#ifndef MPI
  //sprintf(filename, "avc_v_p.%.4d.ascii", Nconf);
  //write_contraction(conn2, (int*)NULL, filename, 16, 2, 0);
#endif

  //sprintf(filename, "avc_p_p.%.4d", Nconf);
  //sprintf(contype, "pseudoscalar-with-axial_current");
  /* write_contraction(pseu, (int*)NULL, filename, 4, 0, 0); */
  //write_lime_contraction(pseu,  filename, 64,  4, contype, Nconf, 0);

  //sprintf(filename, "avc_s_p.%.4d", Nconf);
  //sprintf(contype, "scalar-with-axial_current");
  /* write_contraction(scal, (int*)NULL, filename, 4, 0, 0); */
  //write_lime_contraction(scal,  filename, 64,  4, contype, Nconf, 0);

  //sprintf(filename, "avc_x_p.%.4d", Nconf);
  //sprintf(contype, "xavc-with-axial_current");
  /* write_contraction(xavc, (int*)NULL, filename, 4, 0, 0); */
  //write_lime_contraction(xavc,  filename, 64,  4, contype, Nconf, 0);

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved momentum space results in %e seconds\n", retime-ratime);


  /*****************************************
   * free the allocated memory, finalize
   *****************************************/
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(conn);
  free(pseu);
  free(scal);
  free(xavc);
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
