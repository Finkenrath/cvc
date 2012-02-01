/****************************************************
 * avc_stochastic.c
 *
 * Wed Aug 12 13:29:48 MEST 2009
 *
 * TODO:
 * - implement contact term
 * - tests serial/parallel
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

void usage() {
  fprintf(stdout, "Code to perform AV current correlator disc. contractions\n");
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
  int count_disc   = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int sid, sid2;
  double *conn  = (double*)NULL;
  double *conn2 = (double*)NULL;
  double *Arr = (double*)NULL;
  double *Ass = (double*)NULL;
  double *Ars = (double*)NULL;
  double *Asr = (double*)NULL;
  double *Vrr = (double*)NULL;
  double *Vss = (double*)NULL;
  double *Vrs = (double*)NULL;
  double *Vsr = (double*)NULL;
  double contact_term[8], contact_term2[8];
  double q[4], fnorm, *buff;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1;
  complex *vp1, *vp2, *vp3, *vp4, *vp5;
  complex *ap1, *ap2, *ap3, *ap4, *ap5;
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
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /* allocate memory for the spinor fields */
  no_fields = 20;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  conn  = (double*)calloc(32*VOLUME, sizeof(double));
  conn2 = (double*)calloc(32*VOLUME, sizeof(double));
  Arr  = (double*)calloc(16*VOLUME, sizeof(double));
  Ass  = (double*)calloc(16*VOLUME, sizeof(double));
  Ars  = (double*)calloc(16*VOLUME, sizeof(double));
  Asr  = (double*)calloc(16*VOLUME, sizeof(double));
  Vrr  = (double*)calloc(16*VOLUME, sizeof(double));
  Vss  = (double*)calloc(16*VOLUME, sizeof(double));
  Vrs  = (double*)calloc(16*VOLUME, sizeof(double));
  Vsr  = (double*)calloc(16*VOLUME, sizeof(double));

  if( (conn==(double*)NULL) || (conn2==(double*)NULL) || 
      (Arr==(double*)NULL)  || (Ass==(double*)NULL)   || 
      (Asr==(double*)NULL)  || (Ars==(double*)NULL)   ||
      (Vrr==(double*)NULL)  || (Vss==(double*)NULL)   || 
      (Vsr==(double*)NULL)  || (Vrs==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) conn[ix]  = 0.;
  for(ix=0; ix<32*VOLUME; ix++) conn2[ix] = 0.;
  for(ix=0; ix<8; ix++) contact_term[ix]  = 0.;
  for(ix=0; ix<8; ix++) contact_term2[ix] = 0.;

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  /* start loop on source id.s */
  for(sid=g_sourceid; sid<g_sourceid2; sid++) {

    /* read the new propagator */
    if(format==0) {
      sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, sid);
      if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) break;
    }
    else if(format==1) {
      sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, sid);
      if(read_cmi(g_spinor_field[1], filename) != 0) break;
    }
    xchange_field(g_spinor_field[1]); 

    /* calculate the source: apply Q_phi_tbc */
    Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);
    xchange_field(g_spinor_field[0]);

    /* calculate U phi */
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
	_fv_eq_cm_ti_fv(&g_spinor_field[2+mu][_GSI(ix)], U_, &g_spinor_field[1][_GSI(g_iup[ix][mu])]);
      }
    }

    /* calculate U xi */
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
	_fv_eq_cm_ti_fv(&g_spinor_field[6+mu][_GSI(ix)], U_, &g_spinor_field[0][_GSI(g_iup[ix][mu])]);
      }
    }

    /* calculate Vrr */
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        iix = _GSI(ix);
        _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[2+mu][iix]);
        _fv_mi_eq_fv(spinor1, &g_spinor_field[2+mu][iix]);
	_co_eq_fv_dag_ti_fv((complex*)&Vrr[_GWI(mu,ix,VOLUME)], &g_spinor_field[0][iix], spinor1);

        _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[6+mu][iix]);
        _fv_pl_eq_fv(spinor1,  &g_spinor_field[6+mu][iix]);
	_co_eq_fv_dag_ti_fv(&w, spinor1, &g_spinor_field[1][iix]);
	Vrr[_GWI(mu,ix,VOLUME)]   += w.re;
	Vrr[_GWI(mu,ix,VOLUME)+1] += w.im;
      }
    }

    /* FT Vrr */
    for(mu=0; mu<4; mu++) {
      memcpy((void*)in, (void*)&Vrr[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
      memcpy((void*)&Vrr[_GWI(4+mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)&Vrr[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
      memcpy((void*)&Vrr[_GWI(mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
    }

    /* calculate Arr */
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        iix = _GSI(ix);
        _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[2+mu][iix]);
	_co_eq_fv_dag_ti_fv((complex*)&Arr[_GWI(mu,ix,VOLUME)], &g_spinor_field[0][iix], spinor1);

        _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[1][iix]);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[6+mu][iix], spinor1);
	Arr[_GWI(mu,ix,VOLUME)]   += w.re;
	Arr[_GWI(mu,ix,VOLUME)+1] += w.im;
      }
    }

    /* FT Arr */
    for(mu=0; mu<4; mu++) {
      memcpy((void*)in, (void*)&Arr[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
      memcpy((void*)&Arr[_GWI(4+mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)&Arr[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
      memcpy((void*)&Arr[_GWI(mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
    }

    for(sid2=sid+1; sid2<=g_sourceid2; sid2++) {

      /* read the new propagator */
      if(format==0) {
        sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, sid);
        if(read_lime_spinor(g_spinor_field[11], filename, 0) != 0) break;
      }
      else if(format==1) {
        sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, sid);
        if(read_cmi(g_spinor_field[11], filename) != 0) break;
      }
      xchange_field(g_spinor_field[11]); 

      /* calculate the source: apply Q_phi_tbc */
      Q_phi_tbc(g_spinor_field[10], g_spinor_field[11]);
      xchange_field(g_spinor_field[10]); 

      /* calculate U phi */
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
          _fv_eq_cm_ti_fv(&g_spinor_field[12+mu][_GSI(ix)], U_, &g_spinor_field[11][_GSI(g_iup[ix][mu])]);
        }
      }

      /* calculate U xi */
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
	  _fv_eq_cm_ti_fv(&g_spinor_field[16+mu][_GSI(ix)], U_, &g_spinor_field[10][_GSI(g_iup[ix][mu])]);
        }
      }


      /* calculate Vss */
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          iix = _GSI(ix);
          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[12+mu][iix]);
          _fv_mi_eq_fv(spinor1, &g_spinor_field[12+mu][iix]);
          _co_eq_fv_dag_ti_fv((complex*)&Vss[_GWI(mu,ix,VOLUME)], &g_spinor_field[0][iix], spinor1);

          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[16+mu][iix]);
          _fv_pl_eq_fv(spinor1,  &g_spinor_field[16+mu][iix]);
          _co_eq_fv_dag_ti_fv(&w, spinor1, &g_spinor_field[1][iix]);
	  Vss[_GWI(mu,ix,VOLUME)]   += w.re;
          Vss[_GWI(mu,ix,VOLUME)+1] += w.im;
        }
      }

      /* FT Vss */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)&Vss[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)&Vss[_GWI(4+mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)&Vss[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)&Vss[_GWI(mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
      }

      /* calculate Ass */
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          iix = _GSI(ix);
          _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[12+mu][iix]);
          _co_eq_fv_dag_ti_fv((complex*)&Ass[_GWI(mu,ix,VOLUME)], &g_spinor_field[10][iix], spinor1);

          _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[11][iix]);
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[16+mu][iix], spinor1);
          Ass[_GWI(mu,ix,VOLUME)]   += w.re;
 	  Ass[_GWI(mu,ix,VOLUME)+1] += w.im;
        }
      }

      /* FT Ass */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)&Ass[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)&Ass[_GWI(4+mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)&Ass[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)&Ass[_GWI(mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
      }

      count++;

      /* calculate Ars, Asr, Vrs, Vsr */
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          iix = _GSI(ix);

          /* first contribution */
          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[12+mu][iix]);
	  _fv_mi_eq_fv(spinor1, &g_spinor_field[12+mu][iix]);
	  _co_eq_fv_dag_ti_fv((complex*)&Vrs[_GWI(mu,ix,VOLUME)], &g_spinor_field[0][iix], spinor1);

          /* second contribution */
          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[16+mu][iix]);
	  _fv_pl_eq_fv(spinor1, &g_spinor_field[16+mu][iix]);
	  _co_eq_fv_dag_ti_fv(&w, spinor1, &g_spinor_field[1][iix]);
	  Vrs[_GWI(mu,ix,VOLUME)  ] += w.re;
	  Vrs[_GWI(mu,ix,VOLUME)+1] += w.im;

          /* first contribution */
          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[2+mu][iix]);
	  _fv_mi_eq_fv(spinor1, &g_spinor_field[2+mu][iix]);
	  _co_eq_fv_dag_ti_fv((complex*)&Vsr[_GWI(mu,ix,VOLUME)], &g_spinor_field[10][iix], spinor1);

          /* second contribution */
          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[6+mu][iix]);
	  _fv_pl_eq_fv(spinor1, &g_spinor_field[6+mu][iix]);
	  _co_eq_fv_dag_ti_fv(&w, spinor1, &g_spinor_field[11][iix]);
	  Vsr[_GWI(mu,ix,VOLUME)  ] += w.re;
	  Vsr[_GWI(mu,ix,VOLUME)+1] += w.im;
        }
      }

      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          iix = _GSI(ix);

          /* first contribution */
          _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[12+mu][iix]);
	  _co_eq_fv_dag_ti_fv((complex*)&Ars[_GWI(mu,ix,VOLUME)], &g_spinor_field[0][iix], spinor1);

          /* second contribution */
          _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[1][iix]);
	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[16+mu][iix], spinor1);
	  Ars[_GWI(mu,ix,VOLUME)  ] += w.re;
	  Ars[_GWI(mu,ix,VOLUME)+1] += w.im;

          /* first contribution */
          _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[2+mu][iix]);
	  _co_eq_fv_dag_ti_fv((complex*)&Asr[_GWI(mu,ix,VOLUME)], &g_spinor_field[10][iix], spinor1);

          /* second contribution */
          _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[11][iix]);
	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[6+mu][iix], spinor1);
	  Asr[_GWI(mu,ix,VOLUME)  ] += w.re;
	  Asr[_GWI(mu,ix,VOLUME)+1] += w.im;
        }
      }

      /* FT Vrs, Vsr */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)&Vrs[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)&Vrs[_GWI(4+mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)&Vrs[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)&Vrs[_GWI(mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
      }

      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)&Vsr[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)&Vsr[_GWI(4+mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)&Vsr[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)&Vsr[_GWI(mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
      }

      /* FT Ars, Asr */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)&Ars[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)&Ars[_GWI(4+mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)&Ars[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)&Ars[_GWI(mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
      }

      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)&Asr[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)&Asr[_GWI(4+mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)&Asr[_GWI(mu,ix,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)&Asr[_GWI(mu,ix,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
      }

      /* add contribution to conn, conn2 */
      for(mu=0; mu<4; mu++) {
        vp1 = (complex*)(Vrs + _GWI(mu,0,VOLUME));
        vp3 = (complex*)(Vrr + _GWI(mu,0,VOLUME));
        ap1 = (complex*)(Ars + _GWI(mu,0,VOLUME));
        ap3 = (complex*)(Arr + _GWI(mu,0,VOLUME));
	for(nu=0; nu<4; nu++) {
          vp2 = (complex*)(Vsr + _GWI(nu,0,VOLUME));
          vp4 = (complex*)(Vss + _GWI(nu,0,VOLUME));
          ap2 = (complex*)(Asr + _GWI(nu,0,VOLUME));
          ap4 = (complex*)(Ass + _GWI(nu,0,VOLUME));
        
          vp5 = (complex*)(conn2 + _GWI(4*mu+nu,0,VOLUME));
          ap5 = (complex*)(conn  + _GWI(4*mu+nu,0,VOLUME));

	  for(ix=0; ix<VOLUME; ix++) {
            _co_mi_eq_co_ti_co(vp5, vp1, vp2);
            _co_pl_eq_co_ti_co(vp5, vp3, vp4);

            _co_mi_eq_co_ti_co(ap5, ap1, ap2);
            _co_pl_eq_co_ti_co(ap5, ap3, ap4);

	    vp1++; vp2++; vp3++; vp4++; vp5++;
	    ap1++; ap2++; ap3++; ap4++; ap5++;
          }
        }  /* of nu */
      }    /* of mu */

      for(mu=0; mu<4; mu++) {
        vp1 = (complex*)(Vsr + _GWI(mu,0,VOLUME));
        vp3 = (complex*)(Vss + _GWI(mu,0,VOLUME));
        ap1 = (complex*)(Asr + _GWI(mu,0,VOLUME));
        ap3 = (complex*)(Ass + _GWI(mu,0,VOLUME));
	for(nu=0; nu<4; nu++) {
          vp2 = (complex*)(Vrs + _GWI(nu,0,VOLUME));
          vp4 = (complex*)(Vrr + _GWI(nu,0,VOLUME));
          ap2 = (complex*)(Ars + _GWI(nu,0,VOLUME));
          ap4 = (complex*)(Arr + _GWI(nu,0,VOLUME));
        
          vp5 = (complex*)(conn2 + _GWI(4*mu+nu,0,VOLUME));
          ap5 = (complex*)(conn  + _GWI(4*mu+nu,0,VOLUME));

	  for(ix=0; ix<VOLUME; ix++) {
            _co_mi_eq_co_ti_co(vp5, vp1, vp2);
            _co_pl_eq_co_ti_co(vp5, vp3, vp4);

            _co_mi_eq_co_ti_co(ap5, ap1, ap2);
            _co_pl_eq_co_ti_co(ap5, ap3, ap4);

	    vp1++; vp2++; vp3++; vp4++; vp5++;
	    ap1++; ap2++; ap3++; ap4++; ap5++;
          }
        }  /* of nu */
      }    /* of mu */


    }  /* of sid2 */

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "contractions in %e seconds\n", retime-ratime);

    /* add contribution to contact terms */
    count_disc++;
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        /* A */
        _fv_eq_gamma_ti_fv(spinor1, 6+mu, g_spinor_field[2+mu]+_GSI(ix));
	_co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor1);
        _co_pl_eq_co((complex*)(contact_term+2*mu), &w);

        _fv_eq_gamma_ti_fv(spinor1, 6+mu, g_spinor_field[1]+_GSI(ix));
	_co_eq_fv_dag_ti_fv(&w, g_spinor_field[6+mu]+_GSI(ix), spinor1);
        _co_mi_eq_co((complex*)(contact_term+2*mu), &w);

        _fv_eq_gamma_ti_fv(spinor1, mu, g_spinor_field[2+mu]+_GSI(ix));
	_fv_mi_eq_fv(spinor1, g_spinor_field[2+mu]+_GSI(ix));
	_co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor1);
        _co_pl_eq_co((complex*)(contact_term2+2*mu), &w);

        _fv_eq_gamma_ti_fv(spinor1, mu, g_spinor_field[1]+_GSI(ix));
	_fv_pl_eq_fv(spinor1, g_spinor_field[1]+_GSI(ix));
	_co_eq_fv_dag_ti_fv(&w, g_spinor_field[6+mu]+_GSI(ix), spinor1);
        _co_mi_eq_co((complex*)(contact_term2+2*mu), &w);
      }
    }
  
    if(sid==g_sourceid-1) {
      /* include also contribution from sid2==g_sourceid2 */
      count_disc++;
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          /* A */
          _fv_eq_gamma_ti_fv(spinor1, 6+mu, g_spinor_field[12+mu]+_GSI(ix));
  	  _co_eq_fv_dag_ti_fv(&w, g_spinor_field[10]+_GSI(ix), spinor1);
          _co_pl_eq_co((complex*)(contact_term+2*mu), &w);

          _fv_eq_gamma_ti_fv(spinor1, 6+mu, g_spinor_field[11]+_GSI(ix));
	  _co_eq_fv_dag_ti_fv(&w, g_spinor_field[16+mu]+_GSI(ix), spinor1);
          _co_mi_eq_co((complex*)(contact_term+2*mu), &w);

          _fv_eq_gamma_ti_fv(spinor1, mu, g_spinor_field[12+mu]+_GSI(ix));
	  _fv_mi_eq_fv(spinor1, g_spinor_field[12+mu]+_GSI(ix));
	  _co_eq_fv_dag_ti_fv(&w, g_spinor_field[10]+_GSI(ix), spinor1);
          _co_pl_eq_co((complex*)(contact_term2+2*mu), &w);

          _fv_eq_gamma_ti_fv(spinor1, mu, g_spinor_field[11]+_GSI(ix));
	  _fv_pl_eq_fv(spinor1, g_spinor_field[11]+_GSI(ix));
	  _co_eq_fv_dag_ti_fv(&w, g_spinor_field[16+mu]+_GSI(ix), spinor1);
          _co_mi_eq_co((complex*)(contact_term2+2*mu), &w);
        }
      }
    }

  }    /* of sid */

  /* normalisation */
  if(g_cart_id==0) fprintf(stdout,"count = %d\ncount_disc = %d\n", count, count_disc);

  /* contact_term */
#ifdef MPI
  /* sum up the contact terms */
  buff = (double*)malloc(8*sizeof(double));
  fnorm = 1. / ((double)(T_global*LX*LY*LZ) * 2. * (double)(g_sourceid2-g_sourceid+1));
  MPI_Allreduce(contact_term, buff, 8, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  for(mu=0; mu<8; mu++) contact_term[mu]  = buff[mu]*fnorm;
  MPI_Allreduce(contact_term2, buff, 8, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  for(mu=0; mu<8; mu++) contact_term2[mu]  = buff[mu]*fnorm;
  free(buff);
#else
  fnorm = 1. / ((double)(T*LX*LY*LZ) * 2. * (double)count_disc);
  for(mu=0; mu<8; mu++) {
    contact_term[mu]  *= fnorm;
    contact_term2[mu] *= fnorm;
  }
#endif

  /* conn, conn2; add phase factor and contact_term */
  fnorm = 1. / ((double)(T_global*LX*LY*LZ) * 4. * (double)count);
  for(mu=0; mu<4; mu++) {
  for(nu=0; nu<4; nu++) {
    for(x0=0; x0<T;  x0++) {
      q[0] = (double)(Tstart+x0) / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      q[1] = (double)(x1) / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = (double)(x2) / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = (double)(x3) / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re = cos(M_PI*(q[mu]-q[nu]));
      w.im = sin(M_PI*(q[mu]-q[nu]));
      _co_eq_co_ti_co(&w1, (complex*)(conn+_GWI(4*mu+nu,ix,VOLUME)), &w);
      _co_eq_co_ti_re((complex*)(conn+_GWI(4*mu+nu,ix,VOLUME)), &w1, fnorm);

      _co_eq_co_ti_co(&w1, (complex*)(conn2+_GWI(4*mu+nu,ix,VOLUME)), &w);
      _co_eq_co_ti_re((complex*)(conn2+_GWI(4*mu+nu,ix,VOLUME)), &w1, fnorm);
    }
    }
    }
    }
    if(mu==nu) {
      for(ix=0; ix<VOLUME; ix++) {
        _co_pl_eq_co((complex*)(conn+_GWI(5*mu,ix,VOLUME)), (complex*)(contact_term+2*mu));
        _co_pl_eq_co((complex*)(conn2+_GWI(5*mu,ix,VOLUME)), (complex*)(contact_term2+2*mu));
      }
    }
  }
  }

  /* write conn/2 to file */
  sprintf(filename, "avc_p.%.4d.%.2d.%.2d", Nconf, g_sourceid, g_sourceid2);
  write_contraction(conn, (int*)NULL, filename, 16, 2, 0);
  sprintf(filename, "cvc_p.%.4d.%.2d.%.2d", Nconf, g_sourceid, g_sourceid2);
  write_contraction(conn2, (int*)NULL, filename, 16, 2, 0);

  /* free the allocated memory, finalize */
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(conn);
  free(conn2);
  free(Arr);
  free(Ass);
  free(Ars);
  free(Asr);
  free(Vrr);
  free(Vss);
  free(Vrs);
  free(Vsr);
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
