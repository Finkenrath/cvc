/****************************************************
 * correct_measurements_hpe03.c
 *
 * Wed Nov 11 09:00:53 CET 2009
 *
 * PURPOSE:
 * TODO:
 * DONE:
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
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, nu;
  int status, gid;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int dxm[4], dxn[4], ixpm, ixpn;
  double *disc2 = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq, _2kappamu, hpe3_coeff, onepmutilde2, mutilde2;
  double U_[18], U1_[18], U2_[18];
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
   * allocate memory for the contractions
   ****************************************/
  disc2  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc2 == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc2\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

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
  if(g_cart_id==0) fprintf(stdout, "# hpe3_coeff = %25.16e\n", hpe3_coeff);


  /************************************************
   * loop on gauge configurations
   ************************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    for(ix=0; ix<8*VOLUME; ix++) disc2[ix] = 0.;
    for(ix=0; ix<48*VOLUME; ix++) work[ix] = 0.;
    /************************************************
     * HPE: calculate the plaquette terms 
     ************************************************/
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, gid);
    if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
    if( read_lime_gauge_field_doubleprec(filename) != 0 ) {
      if(g_cart_id==0) fprintf(stderr, "Error, could not read file %s\n", filename);
      exit(115);
    }
    xchange_gauge();
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

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
        }

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
        }
      }
    }
 
    sprintf(filename, "%s_X.%.4d.%.4d", filename_prefix, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Reading contractions from file %s\n", filename);
    status = read_lime_contraction(work, filename, 4, 0);
    if(status != 0) {
      fprintf(stderr, "Error, could not read from file %s; status was %d\n", filename, status);
      exit(116);
    }
 
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        work[_GWI(mu,ix,VOLUME)  ] += disc2[_GWI(mu,ix,VOLUME)  ];
        work[_GWI(mu,ix,VOLUME)  ] /= 2.;
        work[_GWI(mu,ix,VOLUME)+1] += disc2[_GWI(mu,ix,VOLUME)+1];
        work[_GWI(mu,ix,VOLUME)+1] /= 2.;
      }
    }

    sprintf(filename, "%cvc_hpe5_loops_X.%.4d", gid) ;
    if(g_cart_id==0) fprintf(stdout, "# Writing ascii loops to file %s\n", filename);
    write_contraction(disc2, NULL, filename, 4, 2, 0);

    sprintf(filename, "%s_X.%.4d.%.4d", filename_prefix2, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Reading contractions from file %s\n", filename);
    sprintf(contype, "%s_X", filename_prefix);
    write_lime_contraction(work, filename, 64, 4, contype, gid, Nsave);
 
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
      }}}}
    }}
  
   /* save the result in momentum space */
    sprintf(filename, "%s_P.%.4d.%.4d", filename_prefix2, gid, Nsave);
    sprintf(contype, "%s_P", filename_prefix);
    write_lime_contraction(work+_GWI(8,0,VOLUME), filename, 64, 16, contype, gid, Nsave);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to save cvc results: %e seconds\n", retime-ratime);
  } /* of loop on gauge id*/

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  free_geometry();
  fftw_free(in);
  free(disc2);

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
