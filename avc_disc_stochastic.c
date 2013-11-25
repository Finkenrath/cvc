/****************************************************
 * avc_disc_stochastic.c
 *
 * Wed Sep  9 16:20:52 CEST 2009 
 *
 * TODO: 
 * - solve the potential problem of having
 *   one (or several) local T=0
 * CHANGES:
 * - included PP switches AVC, AVC_WI, CVC, QMAT
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
#include "read_input_parser.h"

void usage() {
  fprintf(stdout, "Code to perform quark-disc. contractions for axial and/or conserved vector current\n");
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
  int sl0, sl1, sl2, sl3, have_source_flag=0;
  int sid;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
#ifdef AVC_WI
  double *pseu=(double*)NULL, *scal=(double*)NULL, *xavc=(double*)NULL; 
#endif
  double *work = (double*)NULL;
  double *disc_diag = (double*)NULL;
  double *disc_diag2 = (double*)NULL;
  double q[4], fnorm;
  double unit_trace[2], shift_trace[2], D_trace[2];
  int do_gt   = 0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1, *cp1, *cp2, *cp3;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
  int *status;
#else
  fftwnd_plan plan_p, plan_m;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vgf:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
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

  /**************************************************
   * read input parameters / set the default values 
   **************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
/*
  set_default_input_values();
  read_input(filename);
*/
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
  plan_m = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_FORWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD,  FFTW_MEASURE | FFTW_IN_PLACE);
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
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
#ifdef MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /* allocate memory for the spinor fields */
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
#ifdef CVC
  disc  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
  for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
#endif

#ifdef AVC
  disc2 = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc2 == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc2\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
  for(ix=0; ix<8*VOLUME; ix++) disc2[ix] = 0.;
#endif

  work  = (double*)calloc(48*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

#ifdef AVC_WI
  pseu = (double*)calloc(2*VOLUME, sizeof(double));
  scal = (double*)calloc(2*VOLUME, sizeof(double));
  xavc = (double*)calloc(2*VOLUME, sizeof(double));
  if( (pseu==(double*)NULL) || (scal==(double*)NULL) || (xavc==(double*)NULL) ) { 
    fprintf(stderr, "could not allocate memory for pseu/scal/xavc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
  for(ix=0; ix<2*VOLUME; ix++) pseu[ix] = 0.;
  for(ix=0; ix<2*VOLUME; ix++) scal[ix] = 0.;
  for(ix=0; ix<2*VOLUME; ix++) xavc[ix] = 0.;
#endif

  if(g_subtract == 1) {
    /* allocate memory for disc_diag */
#ifdef CVC
    disc_diag  = (double*)calloc(32*VOLUME, sizeof(double));
    if( disc_diag==(double*)NULL ) {
      fprintf(stderr, "could not allocate memory for disc_diag\n");
#  ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#  endif
      exit(8);
    }
    for(ix=0; ix<32*VOLUME; ix++) disc_diag[ix] = 0.;
#endif /* of CVC */

#ifdef AVC
    disc_diag2 = (double*)calloc(32*VOLUME, sizeof(double));
    if( disc_diag2==(double*)NULL ) {
      fprintf(stderr, "could not allocate memory for disc_diag2\n");
#  ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#  endif
      exit(8);
    }
    for(ix=0; ix<32*VOLUME; ix++) disc_diag2[ix] = 0.;
#endif /* of AVC */
  }

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  if(g_resume==1) { /* read current disc from file */
#ifdef CVC
    sprintf(filename, ".outcvc_current.%.4d", Nconf);
    c = read_contraction(disc, &count, filename, 4);
    if( (g_subtract==1) && (c==0) ) {
      sprintf(filename, ".outcvc_diag_current.%.4d", Nconf);
      c = read_contraction(disc_diag, (int*)NULL, filename, 16);
    }
#else
    c = 1;
#endif

#ifdef AVC
    if(c!=0) {
      sprintf(filename, ".outavc_current.%.4d", Nconf);
      c = read_contraction(disc2, &count, filename, 4);
    }
    if( (g_subtract==1) && (c==0) ) {
      sprintf(filename, ".outavc_diag_current.%.4d", Nconf);
      c = read_contraction(disc_diag2, (int*)NULL, filename, 16);
    }
#endif

#ifdef MPI
    MPI_Gather(&c, 1, MPI_INT, status, 1, MPI_INT, 0, g_cart_grid);
    if(g_cart_id==0) {
      /* check the entries in status */
      for(i=0; i<g_nproc; i++) 
        if(status[i]!=0) { status[0] = 1; break; }
    }
    MPI_Bcast(status, 1, MPI_INT, 0, g_cart_grid);
    if(status[0]==1) {
      for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
      count = 0;
    }
#else
    if(c != 0) {
      fprintf(stdout, "could not read current disc; start new\n");
#  ifdef CVC
      for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
      if(g_subtract==1) for(ix=0; ix<32*VOLUME; ix++) disc_diag[ix] = 0.;
#  endif
#  ifdef AVC
      for(ix=0; ix<8*VOLUME; ix++) disc2[ix] = 0.;
      if(g_subtract==1) for(ix=0; ix<32*VOLUME; ix++) disc_diag2[ix] = 0.;
#  endif
      count = 0;
    }
#endif
    if(g_cart_id==0) fprintf(stdout, "starting with count = %d\n", count);
  }  /* of g_resume ==  1 */
 
  /***********************************************
   * the source location / source loc. coord.
   ***********************************************/
  sl0 = g_source_location / (LX*LY*LZ);
  sl1 = ( g_source_location % (LX*LY*LZ) ) / (LY*LZ);
  sl2 = ( g_source_location % (LY*LZ) ) / LZ;
  sl3 = g_source_location % LZ;
  have_source_flag = sl0-Tstart>=0 && sl0-Tstart<T;
  if(have_source_flag==1) {
    fprintf(stdout, "# [%d] have source\n", g_cart_id);
    fprintf(stdout, "# [%d] source at (%d,%d,%d,%d)\n", g_cart_id, sl0, sl1, sl2, sl3);
    sl0 -= Tstart;
  }
 
  /***********************************************
   * start loop on source id.s 
   ***********************************************/
  for(sid=g_sourceid; sid<=g_sourceid2; sid++) {

    /* read the new propagator */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(format==0) {
      sprintf(filename, "%s.%.4d.%.2d.%.5d.inverted", filename_prefix, Nconf, g_source_timeslice, sid);
      // sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid);
      // sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, sid);
      if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) break;
    }
    else if(format==1) {
      sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid);
      if(read_cmi(g_spinor_field[1], filename) != 0) break;
    }
    xchange_field(g_spinor_field[1]);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    fprintf(stdout, "time to read prop.: %e seconds\n", retime-ratime);

    count++;

    /* calculate the source: apply Q_phi_tbc */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);
    xchange_field(g_spinor_field[0]); 
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "time to calculate source: %e seconds\n", retime-ratime);

#ifdef CVC
    /* add new contractions to (existing) disc */
#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    for(mu=0; mu<4; mu++) { /* loop on Lorentz index of the current */
      iix = _GWI(mu,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {    /* loop on lattice sites */
        _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix, mu)], &co_phase_up[mu]);

        /* first contribution */
        _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[1][_GSI(g_iup[ix][mu])]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_mi_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor2);
#ifndef QMAT
	disc[iix  ] -= 0.5 * w.re;
	disc[iix+1] -= 0.5 * w.im;
#else
	disc[iix  ] -= 0.5 * w.re;
	disc[iix+1] -= 0.5 * w.im / 3.;
#endif
        if(g_subtract==1) {
#ifndef QMAT
	  work[iix  ] = -0.5 * w.re;
	  work[iix+1] = -0.5 * w.im;
#else
	  work[iix  ] = -0.5 * w.re;
	  work[iix+1] = -0.5 * w.im / 3.;
#endif
	}

        // fprintf(stdout, "mu=%d, ix=%5d: L^- = %25.16e +I %25.16e; ", mu, ix, -0.5*w.re, -0.5*w.im);

        /* second contribution */
	_fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[1][_GSI(ix)]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_pl_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(g_iup[ix][mu])], spinor2);
#ifndef QMAT
	disc[iix  ] -= 0.5 * w.re;
	disc[iix+1] -= 0.5 * w.im;
#else
	disc[iix  ] -= 0.5 * w.re;
	disc[iix+1] -= 0.5 * w.im / 3.;
#endif
        if(g_subtract==1) {
#ifndef QMAT
	  work[iix  ] -= 0.5 * w.re;
	  work[iix+1] -= 0.5 * w.im;
#else
	  work[iix  ] -= 0.5 * w.re;
	  work[iix+1] -= 0.5 * w.im / 3.;
#endif
	}

        // fprintf(stdout, "L^+ = %25.16e +I %25.16e\n", -0.5*w.re, -0.5*w.im);

	iix += 2;
      }  /* of ix */
    }    /* of mu */

#  ifdef MPI
    retime = MPI_Wtime();
#  else
    retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    fprintf(stdout, "[%2d] contractions in %e seconds\n", g_cart_id, retime-ratime);

    if(g_subtract==1) {
      /* add current contribution to disc_diag */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
        fftwnd_one(plan_m, in, NULL);
#  endif
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
        fftwnd_one(plan_p, in, NULL);
#  endif
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }  /* of mu */

      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        for(ix=0; ix<VOLUME; ix++) {
	  _co_pl_eq_co_ti_co((complex*)(disc_diag+_GWI(4*mu+nu,ix,VOLUME)), (complex*)(work+_GWI(mu,ix,VOLUME)), (complex*)(work+_GWI(4+nu,ix,VOLUME)));
	}
      }
      }
    } /* of g_subtract == 1 */
#endif /* of CVC */


#ifdef AVC
    /* add new contractions to (existing) disc2 */
#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    for(mu=0; mu<4; mu++) { /* loop on Lorentz index of the current */
      iix = _GWI(mu,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {    /* loop on lattice sites */
        _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix, mu)], &co_phase_up[mu]);

        /* first contribution */
        _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[1][_GSI(g_iup[ix][mu])]);
	_fv_eq_gamma_ti_fv(spinor2, 6+mu, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor2);
	disc2[iix  ] -= 0.5 * w.re;
	disc2[iix+1] -= 0.5 * w.im; 
        if(g_subtract==1) {
	  work[iix  ] = -0.5 * w.re;
	  work[iix+1] = -0.5 * w.im; 
	}

        /* second contribution */
	_fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[1][_GSI(ix)]);
	_fv_eq_gamma_ti_fv(spinor2, 6+mu, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(g_iup[ix][mu])], spinor2);
	disc2[iix  ] -= 0.5 * w.re;
	disc2[iix+1] -= 0.5 * w.im;
        if(g_subtract==1) {
	  work[iix  ] -= 0.5 * w.re;
	  work[iix+1] -= 0.5 * w.im;
	}

#  ifdef AVC_WI
        /* mu-dep. contrib. to xavc */
	_fv_eq_cm_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(g_iup[ix][mu]));
	_fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
	_co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor2);
	xavc[2*ix  ] += 0.5 * w.re;
	xavc[2*ix+1] += 0.5 * w.im;
	_fv_eq_cm_dag_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(ix));
	_fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
	_co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_iup[ix][mu]), spinor2);
	xavc[2*ix  ] += 0.5 * w.re; 
	xavc[2*ix+1] += 0.5 * w.im;
	_cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(g_idn[ix][mu],mu), &co_phase_up[mu]);
	_fv_eq_cm_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(ix));
	_fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
	_co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_idn[ix][mu]), spinor2);
	xavc[2*ix  ] += 0.5 * w.re;
	xavc[2*ix+1] += 0.5 * w.im;
	_fv_eq_cm_dag_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(g_idn[ix][mu]));
	_fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
	_co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor2);
	xavc[2*ix  ] += 0.5 * w.re;
	xavc[2*ix+1] += 0.5 * w.im;
#  endif

	iix += 2;
      }  /* of ix */
    }    /* of mu */

#  ifdef AVC_WI
    /* contributions to pseu/scal/xavc to check WI */
    for(ix=0; ix<VOLUME; ix++) {
      iix = 2*ix;
      _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), g_spinor_field[1]+_GSI(ix));
      scal[iix  ] -= w.re;
      scal[iix+1] -= w.im;
      _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[1]+_GSI(ix));
      _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor1);
      pseu[iix  ] -= w.re;
      pseu[iix+1] -= w.im;
      xavc[iix  ] -= 8. * w.re;
      xavc[iix+1] -= 8. * w.im;
    }
#  endif

#  ifdef MPI
    retime = MPI_Wtime();
#  else
    retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    fprintf(stdout, "[%2d] contractions in %e seconds\n", g_cart_id, retime-ratime);

    if(g_subtract==1) {
      /* add current contribution to disc_diag2 */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
        fftwnd_one(plan_m, in, NULL);
#  endif
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
        fftwnd_one(plan_p, in, NULL);
#  endif
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }  /* of mu */

      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        for(ix=0; ix<VOLUME; ix++) {
	  _co_pl_eq_co_ti_co((complex*)(disc_diag2+_GWI(4*mu+nu,ix,VOLUME)), (complex*)(work+_GWI(mu,ix,VOLUME)), (complex*)(work+_GWI(4+nu,ix,VOLUME)));
	}
      }
      }
    } /* of g_subtract == 1 */
#endif /* of AVC */

  if(have_source_flag==1) {
    /***************************************************
     * check: convergence of trace of unit matrix
     ***************************************************/
     ix = g_ipt[sl0][sl1][sl2][sl3];
     _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
     fprintf(stdout, "# [%d] unit_trace: %4d%25.16e%25.16e\n", g_cart_id, count, w.re, w.im);
     _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), g_spinor_field[0]+_GSI(g_iup[ix][0]));
     fprintf(stdout, "# [%d] shift_trace: %4d%25.16e%25.16e\n", g_cart_id, count, w.re, w.im);
  }

  /***************************************************
   * check: convergence of trace D_u(source_location, source_location)
   ***************************************************/
  Q_phi_tbc(g_spinor_field[1], g_spinor_field[0]);
  if(have_source_flag==1) {
     ix = g_ipt[sl0][sl1][sl2][sl3];
     _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), g_spinor_field[1]+_GSI(ix));
     fprintf(stdout, "# [%d] D_trace: %4d%25.16e%25.16e\n", g_cart_id, count, w.re, w.im); 
  }

    /***************************************************
     * save results for count = multiple of Nsave
     ***************************************************/
    if(count%Nsave == 0) {
      if(g_cart_id == 0) fprintf(stdout, "# save results for count = %d\n", count);

#ifdef CVC
#  ifdef MPI
      ratime = MPI_Wtime();
#  else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      /* save the result in position space */
      fnorm = 1. / ( (double)count * g_prop_normsqr );
      if(g_cart_id==0) fprintf(stdout, "# X-fnorm = %e\n", fnorm);
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          work[_GWI(mu,ix,VOLUME)  ] = disc[_GWI(mu,ix,VOLUME)  ] * fnorm;
          work[_GWI(mu,ix,VOLUME)+1] = disc[_GWI(mu,ix,VOLUME)+1] * fnorm;
        }
      }      
      sprintf(filename, "outcvc_X.%.4d.%.4d", Nconf, count);
      sprintf(contype, "cvc-disc-stoch-X");
      write_lime_contraction(work, filename, 64, 4, contype, Nconf, count);

      sprintf(filename, "outcvc_X.%.4d.%.4d.ascii", Nconf, count);
      write_contraction(work, NULL, filename, 4, 2, 0);

      /* Fourier transform data, copy to work */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
        fftwnd_one(plan_m, in, NULL);
#  endif
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
        fftwnd_one(plan_p, in, NULL);
#  endif
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }  /* of mu =0 ,..., 3*/

      fnorm = 1. / (double)(T_global*LX*LY*LZ);
      if(g_cart_id==0) fprintf(stdout, "# P-fnorm = %e\n", fnorm);
      /* save results in momentum space */
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        cp1 = (complex*)(work+_GWI(mu,0,VOLUME));
        cp2 = (complex*)(work+_GWI(4+nu,0,VOLUME));
        cp3 = (complex*)(work+_GWI(8+4*mu+nu,0,VOLUME));
     
        for(x0=0; x0<T; x0++) {
	  q[0] = (double)(x0+Tstart) / (double)T_global;
        for(x1=0; x1<LX; x1++) {
	  q[1] = (double)(x1) / (double)LX;
        for(x2=0; x2<LY; x2++) {
	  q[2] = (double)(x2) / (double)LY;
        for(x3=0; x3<LZ; x3++) {
	  q[3] = (double)(x3) / (double)LZ;
	  ix = g_ipt[x0][x1][x2][x3];
	  w.re = cos( M_PI * (q[mu]-q[nu]) );
	  w.im = sin( M_PI * (q[mu]-q[nu]) );
	  _co_eq_co_ti_co(&w1, cp1, cp2);
	  if(g_subtract==1) {
	    _co_mi_eq_co(&w1, (complex*)(disc_diag+_GWI(4*mu+nu,ix,VOLUME)));
	  }
	  _co_eq_co_ti_co(cp3, &w1, &w);
	  _co_ti_eq_re(cp3, fnorm);
	  cp1++; cp2++; cp3++;
	}}}}
      }}
  
      /* save the result in momentum space */
      sprintf(filename, "outcvc_P.%.4d.%.4d", Nconf, count);
      sprintf(contype, "cvc-disc-stoch-P");
      write_lime_contraction(work+_GWI(8,0,VOLUME), filename, 64, 16, contype, Nconf, count);
#  ifdef MPI
      retime = MPI_Wtime();
#  else
      retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      if(g_cart_id==0) fprintf(stdout, "time to cvc save results: %e seconds\n", retime-ratime);
#endif /* of CVC */

#ifdef AVC  /*************** save AVC **********************/
#  ifdef MPI
      ratime = MPI_Wtime();
#  else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      /* save the result in position space */
      fnorm = 1. / ( (double)count * g_prop_normsqr );
      if(g_cart_id==0) fprintf(stdout, "# X-fnorm = %e\n", fnorm);
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          work[_GWI(mu,ix,VOLUME)  ] = disc2[_GWI(mu,ix,VOLUME)  ] * fnorm;
          work[_GWI(mu,ix,VOLUME)+1] = disc2[_GWI(mu,ix,VOLUME)+1] * fnorm;
        }
      }      
      sprintf(filename, "outavc_X.%.4d.%.4d", Nconf, count);
      sprintf(contype, "avc-disc-stoch-X");
      write_lime_contraction(work, filename, 64, 4, contype, Nconf, count);

      /* Fourier transform data, copy to work */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
        fftwnd_one(plan_m, in, NULL);
#  endif
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
        fftwnd_one(plan_p, in, NULL);
#  endif
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }  /* of mu =0 ,..., 3*/

      fnorm = 1. / (double)(T_global*LX*LY*LZ);
      if(g_cart_id==0) fprintf(stdout, "# P-fnorm = %e\n", fnorm);
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        cp1 = (complex*)(work+_GWI(mu,0,VOLUME));
        cp2 = (complex*)(work+_GWI(4+nu,0,VOLUME));
        cp3 = (complex*)(work+_GWI(8+4*mu+nu,0,VOLUME));

        for(x0=0; x0<T; x0++) {
	  q[0] = (double)(x0+Tstart) / (double)T_global;
        for(x1=0; x1<LX; x1++) {
	  q[1] = (double)(x1) / (double)LX;
        for(x2=0; x2<LY; x2++) {
	  q[2] = (double)(x2) / (double)LY;
        for(x3=0; x3<LZ; x3++) {
	  q[3] = (double)(x3) / (double)LZ;
	  ix = g_ipt[x0][x1][x2][x3];
	  w.re = cos( M_PI * (q[mu]-q[nu]) );
	  w.im = sin( M_PI * (q[mu]-q[nu]) );
	  _co_eq_co_ti_co(&w1, cp1, cp2);
	  if(g_subtract==1) {
	    _co_mi_eq_co(&w1, (complex*)(disc_diag2+_GWI(4*mu+nu,ix,VOLUME)));
	  }
	  _co_eq_co_ti_co(cp3, &w1, &w);
	  _co_ti_eq_re(cp3, fnorm);
	  cp1++; cp2++; cp3++;
	}}}}
      }}
      /* save the result in momentum space */
      sprintf(filename, "outavc_P.%.4d.%.4d", Nconf, count);
      sprintf(contype, "avc-disc-stoch-P");
      write_lime_contraction(work+_GWI(8,0,VOLUME), filename, 64, 16, contype, Nconf, count);
#  ifdef MPI
      retime = MPI_Wtime();
#  else
      retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      if(g_cart_id==0) fprintf(stdout, "time to save avc results: %e seconds\n", retime-ratime);
#endif /* of AVC */

#ifdef AVC_WI /******************* save AVC_WI *******************************/
#  ifdef MPI
      ratime = MPI_Wtime();
#  else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
       
      fnorm = 1. / ( (double)count * g_prop_normsqr );
      if(g_cart_id=0) fprintf(stdout, "# X-fnorm = %e\n", fnorm);
      for(ix=0; ix<VOLUME; ix++) {
        work[_GWI(0,ix,VOLUME)  ] = pseu[_GWI(0,ix,VOLUME)  ] * fnorm;
        work[_GWI(0,ix,VOLUME)+1] = pseu[_GWI(0,ix,VOLUME)+1] * fnorm;
        work[_GWI(2,ix,VOLUME)  ] = scal[_GWI(0,ix,VOLUME)  ] * fnorm;
        work[_GWI(2,ix,VOLUME)+1] = scal[_GWI(0,ix,VOLUME)+1] * fnorm;
        work[_GWI(4,ix,VOLUME)  ] = xavc[_GWI(0,ix,VOLUME)  ] * fnorm;
        work[_GWI(4,ix,VOLUME)+1] = xavc[_GWI(0,ix,VOLUME)+1] * fnorm;
      }
      sprintf(filename, "outpsx_X.%.4d.%.4d", Nconf, count);
      if( (ofs = fopen(filename, "w")) == (FILE*)NULL ) exit(115);
      for(ix=0; ix<VOLUME; ix++) {
        fprintf(ofs, "%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e\n",
	  work[_GWI(0,ix,VOLUME)], work[_GWI(0,ix,VOLUME)+1], 
          work[_GWI(2,ix,VOLUME)], work[_GWI(2,ix,VOLUME)+1], 
          work[_GWI(4,ix,VOLUME)], work[_GWI(4,ix,VOLUME)+1]);
      }
      fclose(ofs);

      memcpy((void*)in, (void*)work, 2*VOLUME*sizeof(double));
#  ifdef MPI
      fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
      fftwnd_one(plan_m, in, NULL);
#  endif
      memcpy((void*)(work+2*VOLUME), (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)work, 2*VOLUME*sizeof(double));
#  ifdef MPI
      fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
      fftwnd_one(plan_p, in, NULL);
#  endif
      memcpy((void*)work, (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)(work+4*VOLUME), 2*VOLUME*sizeof(double));
#  ifdef MPI
      fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
      fftwnd_one(plan_m, in, NULL);
#  endif
      memcpy((void*)(work+6*VOLUME), (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)(work+4*VOLUME), 2*VOLUME*sizeof(double));
#  ifdef MPI
      fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
      fftwnd_one(plan_p, in, NULL);
#  endif
      memcpy((void*)(work+4*VOLUME), (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)(work+8*VOLUME), 2*VOLUME*sizeof(double));
#  ifdef MPI
      fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
      fftwnd_one(plan_m, in, NULL);
#  endif
      memcpy((void*)(work+10*VOLUME), (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)(work+8*VOLUME), 2*VOLUME*sizeof(double));
#  ifdef MPI
      fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
      fftwnd_one(plan_p, in, NULL);
#  endif
      memcpy((void*)(work+8*VOLUME), (void*)in, 2*VOLUME*sizeof(double));

      fnorm = 1. / (double)(T_global*LX*LY*LZ);
      if(g_cart_id=0) fprintf(stdout, "# P-fnorm = %e\n", fnorm);
      for(ix=0; ix<VOLUME; ix++) {
        work[_GWI(6,ix,VOLUME)  ] = 2.*(1./(2.*g_kappa)-4.)*work[_GWI(0,ix,VOLUME)  ] -
	  2.*g_mu*work[_GWI(2,ix,VOLUME)+1] + work[_GWI(4,ix,VOLUME)  ];
        work[_GWI(6,ix,VOLUME)+1] = 2.*(1./(2.*g_kappa)-4.)*work[_GWI(0,ix,VOLUME)+1] +
	  2.*g_mu*work[_GWI(2,ix,VOLUME)  ] + work[_GWI(4,ix,VOLUME)+1];

        work[_GWI(7,ix,VOLUME)  ] = 2.*(1./(2.*g_kappa)-4.)*work[_GWI(1,ix,VOLUME)  ] -
	  2.*g_mu*work[_GWI(3,ix,VOLUME)+1] + work[_GWI(5,ix,VOLUME)];
        work[_GWI(7,ix,VOLUME)+1] = 2.*(1./(2.*g_kappa)-4.)*work[_GWI(1,ix,VOLUME)+1] +
	  2.*g_mu*work[_GWI(3,ix,VOLUME)  ] + work[_GWI(5,ix,VOLUME)+1];
	
	_co_eq_co_ti_co((complex*)(work+_GWI(8,ix,VOLUME)), (complex*)(work+_GWI(6,ix,VOLUME)), (complex*)(work+_GWI(7,ix,VOLUME)));
	_co_ti_eq_re((complex*)(work+_GWI(8,ix,VOLUME)), fnorm);
      }

      sprintf(filename, "outWI_P.%.4d.%.4d", Nconf, count);
      if((ofs = fopen(filename, "w"))==(FILE*)NULL) return(-5);
      for(ix=0; ix<VOLUME; ix++) {
        fprintf(ofs, "%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e\n", 
	  work[_GWI(6,ix,VOLUME)], work[_GWI(6,ix,VOLUME)+1],
	  work[_GWI(7,ix,VOLUME)], work[_GWI(7,ix,VOLUME)+1],
	  work[_GWI(8,ix,VOLUME)], work[_GWI(8,ix,VOLUME)+1]);
      }
      fclose(ofs);
#  ifdef MPI
      retime = MPI_Wtime();
#  else
      retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      if(g_cart_id==0) fprintf(stdout, "time to save avc WI results: %e seconds\n", retime-ratime);
#endif /* of ifdef AVC_WI */

    }  /* of count % Nsave == 0 */
  }  /* of loop on sid */

  if(g_resume==1) {
    /* write current disc to file */
#ifdef CVC
    sprintf(filename, ".outcvc_current.%.4d", Nconf);
    write_contraction(disc, &count, filename, 4, 0, 0);
    if(g_subtract == 1) {
      /* write current disc_diag to file */
      sprintf(filename, ".outcvc_diag_current.%.4d", Nconf);
      write_contraction(disc_diag, (int*)NULL, filename, 16, 0, 0);
    }
#endif

#ifdef AVC
    sprintf(filename, ".outavc_current.%.4d", Nconf);
    write_contraction(disc2, &count, filename, 4, 0, 0);
    if(g_subtract == 1) {
      /* write current disc_diag to file */
      sprintf(filename, ".outavc_diag_current.%.4d", Nconf);
      write_contraction(disc_diag2, (int*)NULL, filename, 16, 0, 0);
    }
#endif

  }

  /* free the allocated memory, finalize */
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
#ifdef CVC
  free(disc);
  if(g_subtract==1) free(disc_diag);
#endif

#ifdef AVC
  free(disc2);
  if(g_subtract==1) free(disc_diag2);
#endif

  free(work);
#ifdef AVC_WI
  free(pseu); free(scal); free(xavc); 
#endif

#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  fftwnd_mpi_destroy_plan(plan_m);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);
#endif

  return(0);

}
