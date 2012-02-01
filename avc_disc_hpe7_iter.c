/*********************************************************************************
 * avc_disc_hpe7_iter.c
 *
 * Fri Nov 20 00:50:53 CET 2009 
 *
 * PURPOSE:
 * - calculate the disconnected contractions of the vacuum polarization 
 *   and apply the Hopping-parameter expansion
 * - use iterative method for 5th order contribution (cf. avc_disc_hpe7 with
 *   recursive method)
 *   expansion
 * TODO:
 * - current version _DOES NOT WORK WITH MPI_
 * - the allocations of fields must be relocated, not all fields 
 *   are needed simultaneously
 * - free the fields allocated for the 5th order calculation
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
#include "Q_phi.h"

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
  int dxm[4], dxn[4], ixpm, ixpn;
  int sid, steps[4], nloop;
  int **loop_tab;
  double *tcf, *tcb, r1[2], r2[2];
  double *disc  = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq, _2kappamu, hpe3_coeff, onepmutilde2, mutilde2;
  double spinor1[24], spinor2[24], spinor3[24], U_[18], U1_[18], U2_[18];
  double *gauge_trafo=(double*)NULL;
  complex w, w1, w2, *cp1, *cp2, *cp3;
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
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");

  /* read the input file */
  read_input(filename);

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
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

  if(do_gt==1) {
    /***********************************
     * initialize gauge transformation
     ***********************************/
    init_gauge_trafo(&gauge_trafo, 1.);
    apply_gt_gauge(gauge_trafo);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value after gauge trafo: %25.16e\n", plaq);
  }

  /****************************************
   * allocate memory for the spinor fields
   ****************************************/
  no_fields = 3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /****************************************
   * allocate memory for the contractions
   ****************************************/
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

  work  = (double*)calloc(48*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  /****************************************
   * prepare Fourier transformation arrays
   ****************************************/
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  /************************************************
   * HPE: calculate coeff. of 3rd order term
   ************************************************/
  _2kappamu    = 2. * g_kappa * g_mu;
  onepmutilde2 = 1. + _2kappamu * _2kappamu;
  mutilde2     = _2kappamu * _2kappamu;

  hpe3_coeff   = 16. * g_kappa*g_kappa*g_kappa*g_kappa * (1. + 6. * mutilde2 + mutilde2*mutilde2) / onepmutilde2 / onepmutilde2 / onepmutilde2 / onepmutilde2;

/*
  hpe3_coeff = 8. * g_kappa*g_kappa*g_kappa * \
        (1. + 6.*_2kappamu*_2kappamu + _2kappamu*_2kappamu*_2kappamu*_2kappamu) / (1. + _2kappamu*_2kappamu) / (1. + _2kappamu*_2kappamu) / (1. + _2kappamu*_2kappamu) / (1. + _2kappamu*_2kappamu);
*/
  fprintf(stdout, "# hpe3_coeff = %25.16e\n", hpe3_coeff);

  /************************************************
   * HPE: calculate the 3rd order plaquette terms 
   ************************************************/

  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) { 
      for(i=1; i<4; i++) {
        nu = (mu+i)%4;
        _cm_eq_cm_ti_cm(U1_, g_gauge_field+_GGI(ix,mu), g_gauge_field+_GGI(g_iup[ix][mu],nu) );
        _cm_eq_cm_ti_cm(U2_, g_gauge_field+_GGI(ix,nu), g_gauge_field+_GGI(g_iup[ix][nu],mu) );
        _cm_eq_cm_ti_cm_dag(U_, U1_, U2_);
        _co_eq_tr_cm(&w1, U_);

        iix = g_idn[ix][nu];
        _cm_eq_cm_ti_cm(U1_, g_gauge_field+_GGI(iix,mu), g_gauge_field+_GGI(g_iup[iix][mu],nu) );
        _cm_eq_cm_ti_cm(U2_, g_gauge_field+_GGI(iix,nu), g_gauge_field+_GGI(g_iup[iix][nu],mu) );
        _cm_eq_cm_ti_cm_dag(U_, U1_, U2_);
        _co_eq_tr_cm(&w2, U_);
        disc[_GWI(mu,ix,VOLUME)+1] += hpe3_coeff * (w1.im - w2.im);
/*
        _cm_eq_cm_ti_cm(U1_, g_gauge_field+_GGI(g_idn[ix][nu],nu), g_gauge_field+_GGI(ix,mu) );
        _cm_eq_cm_ti_cm(U2_, g_gauge_field+_GGI(g_idn[ix][nu],mu), g_gauge_field+_GGI(g_iup[g_idn[ix][nu]][mu], nu) );
        _cm_eq_cm_ti_cm_dag(U_, U1_, U2_);
        _co_eq_tr_cm(&w2, U_);
        disc[_GWI(mu,ix,VOLUME)+1] += hpe3_coeff * (w1.im + w2.im);
*/
/*        fprintf(stdout, "mu=%1d, ix=%5d, nu=%1d, w1=%25.16e +i %25.16e; w2=%25.16e +i %25.16e\n", 
            mu, ix, nu, w1.re, w1.im, w2.re, w2.im); 
*/
      }  /* of nu */

      /****************************************
       * - in case lattice size equals 4 
       *   calculate additional loop term
       * - _NOTE_ the possible minus sign from
       *   the fermionic boundary conditions
       ****************************************/
      if(dims[mu]==4) {
        wilson_loop(&w, ix, mu, dims[mu]);
        fnorm = -64. * g_kappa*g_kappa*g_kappa*g_kappa / onepmutilde2 / onepmutilde2 / onepmutilde2 / onepmutilde2; 
        disc[_GWI(mu,ix,VOLUME)+1] += fnorm * w.im;
/*        fprintf(stdout, "loop contribution: ix=%5d, mu=%2d, fnorm=%25.16e, w=%25.16e\n", ix, mu, fnorm, w.im); */
      }
/*
      fprintf(stdout, "-------------------------------------------\n");
      fprintf(stdout, "disc[ix=%d,mu=%d] = %25.16e +i %25.16e\n", ix, mu, disc[_GWI(mu,ix,VOLUME)], disc[_GWI(mu,ix,VOLUME)+1]);
      fprintf(stdout, "-------------------------------------------\n");
*/
    }  /* of mu = 0 to 3 */
  }    /* of ix = 0 to VOLUME-1 */

  sprintf(filename, "avc_disc_hpe7_iter_3rd.%.4d", Nconf);
  ofs = fopen(filename, "w");
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) { 
      fprintf(ofs, "%6d%3d%25.16e\t%25.16e\n", ix, mu, disc[_GWI(mu,ix,VOLUME)], \
              disc[_GWI(mu,ix,VOLUME)+1]);
    }
  }
  fclose(ofs);

  for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;


  /*******************************************************************
   * HPE: calculate the 5th order term
   * - NOTE: the 5th order contribution is purely imaginary
   *******************************************************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  init_trace_coeff(&tcf, &tcb, &loop_tab, 5, &nloop);
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
/*
      r1[0] = 0.; r1[1] = 0.;
      r2[0] = 0.; r2[1] = 0.;
*/
      Hopping_iter( disc+_GWI(mu,ix,VOLUME), disc+_GWI(mu,ix,VOLUME), tcf, tcb, ix, mu, 5, nloop, loop_tab);

/*      fprintf(stdout, "%6d, %3d; r1=%25.16e, %25.16e; r2=%25.16e, %25.16e\n", ix, mu, r1[0], r1[1], r2[0], r2[1]); */
/*      disc[_GWI(mu,ix,VOLUME)  ] += w1.re + w2.re; */
/*      disc[_GWI(mu,ix,VOLUME)+1] += r1[1] + r2[1]; */
    }
  }
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# time to calculate 5th order contribution: %e seconds\n", retime-ratime);


  sprintf(filename, "avc_disc_hpe7_iter_5th.%.4d", Nconf);
  ofs = fopen(filename, "w");
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) { 
      fprintf(ofs, "%6d%3d%25.16e\t%25.16e\n", ix, mu, disc[_GWI(mu,ix,VOLUME)], \
              disc[_GWI(mu,ix,VOLUME)+1]);
    }
  }
  fclose(ofs);

  for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;


  /***********************************************
   * From here: everything as before except for
   *            the change BH5 --> BH7
   *
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
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
      if(read_lime_spinor(g_spinor_field[2], filename, 0) != 0) break;
    }
    else if(format==1) {
      sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid);
      if(read_cmi(g_spinor_field[2], filename) != 0) break;
    }
    xchange_field(g_spinor_field[2]);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    fprintf(stdout, "time to read prop.: %e seconds\n", retime-ratime);

    if(do_gt==1) {
      /******************************************
       * gauge transform the propagators for sid
       ******************************************/
      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_cm_ti_fv(spinor1, gauge_trafo+18*ix, g_spinor_field[2]+_GSI(ix));
        _fv_eq_fv(g_spinor_field[2]+_GSI(ix), spinor1);
      }
      xchange_field(g_spinor_field[2]);
    }

    count++;

    /************************************************
     * calculate the source: apply Q_phi_tbc 
     ************************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    Q_phi_tbc(g_spinor_field[0], g_spinor_field[2]);
    xchange_field(g_spinor_field[0]); 
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "time to calculate source: %e seconds\n", retime-ratime);


    /************************************************
     * HPE: apply BH7 
     ************************************************/
    BH7(g_spinor_field[1], g_spinor_field[2]);

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
	disc[iix  ] -= 0.5 * w.re;
	disc[iix+1] -= 0.5 * w.im;

        /* second contribution */
	_fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[1][_GSI(ix)]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_pl_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(g_iup[ix][mu])], spinor2);
	disc[iix  ] -= 0.5 * w.re;
	disc[iix+1] -= 0.5 * w.im;

	iix += 2;
      }  /* of ix */
    }    /* of mu */

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "[%2d] time to contract cvc: %e seconds\n", g_cart_id, retime-ratime);


    /************************************************
     * save results for count = multiple of Nsave 
     ************************************************/
    if(count%Nsave == 0) {

      if(g_cart_id == 0) fprintf(stdout, "save results for count = %d\n", count);

      /* save the result in position space */
      sprintf(filename, "cvc_hpe7_iter_X.%.4d.%.4d", Nconf, count);
      write_contraction(disc, NULL, filename, 4, 2, 0);

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

      /* Fourier transform data, copy to work */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }  /* of mu =0 ,..., 3*/

      fnorm = 1. / ((double)(T_global*LX*LY*LZ) * (double)(count*count));
      fprintf(stdout, "fnorm = %e\n", fnorm);
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
	  _co_eq_co_ti_co(cp3, &w1, &w);
	  _co_ti_eq_re(cp3, fnorm);
	  cp1++; cp2++; cp3++;
	}
	}
	}
	}

      }
      }
  
      /* save the result in momentum space */
      sprintf(filename, "cvc_hpe7_iter_P.%.4d.%.4d", Nconf, count);
      write_contraction(work+_GWI(8,0,VOLUME), NULL, filename, 16, 2, 0);

#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "time to save cvc results: %e seconds\n", retime-ratime);

    }  /* of count % Nsave == 0 */

  }  /* of loop on sid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(disc);

  free(work);

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
