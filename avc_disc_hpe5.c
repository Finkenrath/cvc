/****************************************************
 * avc_disc_hpe5.c
 *
 * Wed Nov 11 09:00:53 CET 2009
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * - corrected sample average over stochastic sources
 * CHANGES:
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

void usage() {
  fprintf(stdout, "Code to perform cvc quark-disc, contractions\n");
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
  int sid;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq, _2kappamu, hpe3_coeff, onepmutilde2, mutilde2;
  double spinor1[24], spinor2[24], U_[18], U1_[18], U2_[18];
  double *gauge_trafo=(double*)NULL;
  complex w, w1, w2, *cp1, *cp2, *cp3;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
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

  if(do_gt==1) {
    /***********************************
     * initialize gauge transformation
     ***********************************/
    init_gauge_trafo(&gauge_trafo, 1.);
    apply_gt_gauge(gauge_trafo);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "measured plaquette value after gauge trafo: %25.16e\n", plaq);
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
  fprintf(stdout, "hpe3_coeff = %25.16e\n", hpe3_coeff);

  /************************************************
   * HPE: calculate the plaquette terms 
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
        disc2[_GWI(mu,ix,VOLUME)+1] += hpe3_coeff * (w1.im - w2.im);

/*
        _cm_eq_cm_ti_cm(U1_, g_gauge_field+_GGI(g_idn[ix][nu],nu), g_gauge_field+_GGI(ix,mu) );
        _cm_eq_cm_ti_cm(U2_, g_gauge_field+_GGI(g_idn[ix][nu],mu), g_gauge_field+_GGI(g_iup[g_idn[ix][nu]][mu], nu) );
        _cm_eq_cm_ti_cm_dag(U_, U1_, U2_);
        _co_eq_tr_cm(&w2, U_);
        disc2[_GWI(mu,ix,VOLUME)+1] += hpe3_coeff * (w1.im + w2.im);
*/


/*        fprintf(stdout, "mu=%1d, ix=%5d, nu=%1d, w1=%25.16e +i %25.16e; w2=%25.16e +i %25.16e\n", 
            mu, ix, nu, w1.re, w1.im, w2.re, w2.im); */
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
        disc2[_GWI(mu,ix,VOLUME)+1] += fnorm * w.im;
/*        fprintf(stdout, "loop contribution: ix=%5d, mu=%2d, fnorm=%25.16e, w=%25.16e\n", ix, mu, fnorm, w.im); */
      }
/*
      fprintf(stdout, "-------------------------------------------\n");
      fprintf(stdout, "disc2[ix=%d,mu=%d] = %25.16e +i %25.16e\n", ix, mu, disc2[_GWI(mu,ix,VOLUME)], disc2[_GWI(mu,ix,VOLUME)+1]);
      fprintf(stdout, "-------------------------------------------\n");
*/
    }
  }
/*
  sprintf(filename, "avc_disc_hpe5_3rd.%.4d", Nconf);
  ofs = fopen(filename, "w");
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) { 
      fprintf(ofs, "%6d%3d%25.16e\t%25.16e\n", ix, mu, disc[_GWI(mu,ix,VOLUME)], disc[_GWI(mu,ix,VOLUME)+1]);
    }
  }
  fclose(ofs);
  for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
*/
/*
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<4; mu++) {
      dxm[0]=0; dxm[1]=0; dxm[2]=0; dxm[3]=0; dxm[mu]=1;

      for(i=1; i<4; i++) {
        nu = (mu+i)%4;
        dxn[0]=0; dxn[1]=0; dxn[2]=0; dxn[3]=0; dxn[nu]=1;

        ixpm = g_ipt[(x0+dxm[0]+T)%T][(x1+dxm[1]+LX)%LX][(x2+dxm[2]+LY)%LY][(x3+dxm[3]+LZ)%LZ];
        ixpn = g_ipt[(x0+dxn[0]+T)%T][(x1+dxn[1]+LX)%LX][(x2+dxn[2]+LY)%LY][(x3+dxn[3]+LZ)%LZ];

        _cm_eq_cm_ti_cm(U1_, g_gauge_field + 72*ix+18*mu, g_gauge_field + 72*ixpm+18*nu );
        _cm_eq_cm_ti_cm(U2_, g_gauge_field + 72*ix+18*nu, g_gauge_field + 72*ixpn+18*mu );
        _cm_eq_cm_ti_cm_dag(U_, U1_, U2_);
        _co_eq_tr_cm(&w1, U_);

        ixpm = g_ipt[(x0+dxm[0]-dxn[0]+T)%T][(x1+dxm[1]-dxn[1]+LX)%LX][(x2+dxm[2]-dxn[2]+LY)%LY][(x3+dxm[3]-dxn[3]+LZ)%LZ];
        ixpn = g_ipt[(x0-dxn[0]+T)%T][(x1-dxn[1]+LX)%LX][(x2-dxn[2]+LY)%LY][(x3-dxn[3]+LZ)%LZ];

        _cm_eq_cm_ti_cm(U1_, g_gauge_field + 72*ixpn+18*nu, g_gauge_field + 72*ix+18*mu);
        _cm_eq_cm_ti_cm(U2_, g_gauge_field + 72*ixpn+18*mu, g_gauge_field + 72*ixpm+18*nu);
        _cm_eq_cm_ti_cm_dag(U_, U1_, U2_);
        _co_eq_tr_cm(&w2, U_);

        disc2[_GWI(mu,ix,VOLUME)+1] += hpe3_coeff * (w1.im + w2.im);
        fprintf(stdout, "mu=%1d, ix=%5d, nu=%1d, w1=%25.16e; w2=%25.16e\n", mu, ix, nu, w1.im, w2.im); 
      }
      fprintf(stdout, "-------------------------------------------\n");
      fprintf(stdout, "disc2[ix=%d,mu=%d] = %25.16e +i %25.16e\n", ix, mu, disc2[_GWI(mu,ix,VOLUME)], disc2[_GWI(mu,ix,VOLUME)+1]);
      fprintf(stdout, "-------------------------------------------\n");
    }
  }
  }
  }
  }
*/

  /***********************************************
   * start loop on source id.s 
   ***********************************************/
  for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {

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
     * HPE: apply BH5 
     ************************************************/
    BH5(g_spinor_field[1], g_spinor_field[2]);

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
    if(g_cart_id==0) fprintf(stdout, "# time to contract cvc: %e seconds\n", retime-ratime);


    /************************************************
     * save results for count = multiple of Nsave 
     ************************************************/
    if(count%Nsave == 0) {

      if(g_cart_id == 0) fprintf(stdout, "save results for count = %d\n", count);

      fnorm = 1. / ( (double)count * g_prop_normsqr );
      if(g_cart_id==0) fprintf(stdout, "# X-fnorm = %e\n", fnorm);
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          work[_GWI(mu,ix,VOLUME)  ] = disc[_GWI(mu,ix,VOLUME)  ] * fnorm + disc2[_GWI(mu,ix,VOLUME)  ];
          work[_GWI(mu,ix,VOLUME)+1] = disc[_GWI(mu,ix,VOLUME)+1] * fnorm + disc2[_GWI(mu,ix,VOLUME)+1];
        }
      }

      /* save the result in position space */
      sprintf(filename, "cvc_hpe5_X.%.4d.%.4d", Nconf, count);
      sprintf(contype, "cvc-disc-all-hpe-05-X");
      write_lime_contraction(work, filename, 64, 4, contype, Nconf, count);

/*
      sprintf(filename, "cvc_hpe5_Xascii.%.4d.%.4d", Nconf, count);
      write_contraction(work, NULL, filename, 4, 2, 0);
*/

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

      /* Fourier transform data, copy to work */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));


        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
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
      sprintf(filename, "cvc_hpe5_P.%.4d.%.4d", Nconf, count);
      sprintf(contype, "cvc-disc-all-hpe-05-P");
      write_lime_contraction(work+_GWI(8,0,VOLUME), filename, 64, 16, contype, Nconf, count);
/*
      sprintf(filename, "cvc_hpe5_Pascii.%.4d.%.4d", Nconf, count);
      write_contraction(work+_GWI(8,0,VOLUME), NULL, filename, 16, 2, 0);
*/
#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "# time to save cvc results: %e seconds\n", retime-ratime);

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
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);
#endif

  return(0);

}
