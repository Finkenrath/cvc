/****************************************************
 * trjmux_stochastic.c
 *
 * Wed Oct  7 09:54:57 CEST 2009
 *
 * PURPOSE:
 * - calculate the contractions for Tr[J_\mu(x)]
 * - uses series of prop.s from stochastic volume
 *   sources
 * - needs output of lmux.c to construct full
 *   quark-disconnected contribution for _FIXED_
 *   source location
 * - trace at source location from output of lmux.c
 *   (file cvc_lnuy.Nconf)
 * TODO:
 * - convergence check
 * DONE:
 * - checked result for source location against
 *   lmux.c and avc_disc_stochastic.c
 * - checked gauge invariance
 * CHANGES:
 * - added stoch. estim. of the spin-colour unit matrix
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
  fprintf(stdout, "Code to perform light neutral contractions\n");
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
  int sx0, sx1, sx2, sx3;
  int sid;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  double cvc_lnuy[8];
  double *gauge_trafo=(double*)NULL;
  double unit_trace[2], D_trace[2];
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1, *cp1, *cp2, *cp3;
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
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /* get the source location coordinates */
  sx0 =   g_source_location / (LX*LY*LZ  );
  sx1 = ( g_source_location % (LX*LY*LZ) ) / (LY*LZ);
  sx2 = ( g_source_location % (LY*LZ)    ) / LZ;
  sx3 = ( g_source_location %  LZ        );

  /* read the data for lnuy */
  sprintf(filename, "cvc_lnuy_X.%.4d", Nconf);
  ofs = fopen(filename, "r");
  fprintf(stdout, "reading cvc lnuy from file %s\n", filename);
  for(mu=0; mu<4; mu++) {
    fscanf(ofs, "%lf%lf", cvc_lnuy+2*mu, cvc_lnuy+2*mu+1);
  }
  fclose(ofs);

  /* allocate memory for the spinor fields */
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /****************************************
   * allocate memory for the contractions 
   ****************************************/
  disc  = (double*)calloc(8*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;

  disc2 = (double*)calloc(8*VOLUME, sizeof(double));
  if( disc2 == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc2\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<8*VOLUME; ix++) disc2[ix] = 0.;

  work  = (double*)calloc(48*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
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

  if(g_resume==1) { /* read current disc from file */
    sprintf(filename, ".outcvc_current.%.4d", Nconf);
    c = read_contraction(disc, &count, filename, 8);

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
      for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
      count = 0;
    }
#endif
    if(g_cart_id==0) fprintf(stdout, "starting with count = %d\n", count);
  }  /* of g_resume ==  1 */

  if(do_gt==1) {
    /***********************************
     * initialize gauge transformation 
     ***********************************/
    init_gauge_trafo(&gauge_trafo,1.0);
    fprintf(stdout, "applying gauge trafo to gauge field\n");
    apply_gt_gauge(gauge_trafo);
     plaquette(&plaq);
     if(g_cart_id==0) fprintf(stdout, "plaquette plaq = %25.16e\n", plaq);
  } 
  unit_trace[0] = 0.;
  unit_trace[1] = 0.;
  D_trace[0] = 0.;
  D_trace[1] = 0.;
  
  /****************************************
   * start loop on source id.s
   ****************************************/
  for(sid=g_sourceid; sid<=g_sourceid2; sid++) {

    /****************************************
     * read the new propagator
     ****************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

    /****************************************
     * check: write source before D-appl.
     ****************************************/
/*
    if(format==0) {
      sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, sid);
      read_lime_spinor(g_spinor_field[0], filename, 0);
    }
    for(ix=0; ix<12*VOLUME; ix++) {
      fprintf(stdout, "source: %6d%25.16e%25.16e\n", ix, g_spinor_field[0][2*ix], g_spinor_field[0][2*ix+1]);
    }
*/

    if(format==0) {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
      /* sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, sid); */
      if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) break;
    }
    else if(format==1) {
      sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid);
      if(read_cmi(g_spinor_field[1], filename) != 0) break;
    }
    xchange_field(g_spinor_field[1]);

    if(do_gt==1) {
      fprintf(stdout, "applying gt on propagators\n");
      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_cm_ti_fv(spinor1, gauge_trafo+18*ix, g_spinor_field[1]+_GSI(ix));
        _fv_eq_fv(g_spinor_field[1]+_GSI(ix), spinor1);
      }
    }

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    fprintf(stdout, "time to read prop.: %e seconds\n", retime-ratime);

    count++;

    /****************************************
     * calculate the source: apply Q_phi_tbc
     ****************************************/
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

    /****************************************
     * check: write source after D-appl.
     ****************************************/
/*
     for(ix=0; ix<12*VOLUME; ix++) {
       fprintf(stdout, "D_source: %6d%25.16e%25.16e\n", ix, g_spinor_field[0][2*ix], g_spinor_field[0][2*ix+1]);
     }
*/

    /****************************************
     * add new contractions to (existing) disc
     ****************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
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
	disc2[iix  ] -= 0.5 * w.re;
	disc2[iix+1] -= 0.5 * w.im;

        iix += 2;
      }  /* of ix */
    }    /* of mu */

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    fprintf(stdout, "[%2d] contractions for CVC in %e seconds\n", g_cart_id, retime-ratime);

    /***************************************************
     * check: convergence of trace of unit matrix
     ***************************************************/
     _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_source_location), g_spinor_field[0]+_GSI(g_source_location));
     unit_trace[0] += w.re;
     unit_trace[1] += w.im;
     fprintf(stdout, "unit_trace: %4d%25.16e%25.16e\n", count, w.re, w.im);
     _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_source_location), g_spinor_field[0]+_GSI(g_iup[g_source_location][0]));
     fprintf(stdout, "shift_trace: %4d%25.16e%25.16e\n", count, w.re, w.im);

    /***************************************************
     * check: convergence of trace D_u(source_location, source_location)
     ***************************************************/
     Q_phi_tbc(g_spinor_field[1], g_spinor_field[0]);
     _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_source_location), g_spinor_field[1]+_GSI(g_source_location));
     D_trace[0] += w.re;
     D_trace[1] += w.im;
/*     fprintf(stdout, "D_trace: %4d%25.16e%25.16e\n", count, D_trace[0]/(double)count, D_trace[1]/(double)count); */
     fprintf(stdout, "D_trace: %4d%25.16e%25.16e\n", count, w.re, w.im); 


    /***************************************************
     * save results for count = multiple of Nsave 
     ***************************************************/
    if(count%Nsave == 0) {

      if(g_cart_id == 0) fprintf(stdout, "save results for count = %d\n", count);

      /* save the result in position space */

      /* divide by number of propagators */
      for(ix=0; ix<8*VOLUME; ix++) work[ix] = disc[ix]  / (double)count;
      sprintf(filename, "outcvc_Xm.%.4d.%.4d", Nconf, count);
      write_contraction(work, NULL, filename, 4, 2, 0);
      for(ix=0; ix<8*VOLUME; ix++) work[ix] = disc2[ix] / (double)count;
      sprintf(filename, "outcvc_Xp.%.4d.%.4d", Nconf, count);
      write_contraction(work, NULL, filename, 4, 2, 0);
      for(ix=0; ix<8*VOLUME; ix++) work[ix] = (disc[ix] + disc2[ix]) / (double)count;
      sprintf(filename, "outcvc_X.%.4d.%.4d", Nconf, count);
      write_contraction(work, NULL, filename, 4, 2, 0);

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

      /****************************************
       * Fourier transform data, copy to work
       ****************************************/
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }  /* of mu =0 ,..., 3*/

      /* fnorm = 1. / ((double)count); */
      fprintf(stdout, "fnorm = %e\n", fnorm);
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        cp1 = (complex*)(work+_GWI(mu,0,VOLUME));
        cp2 = (complex*)(cvc_lnuy+2*nu);
        cp3 = (complex*)(work+_GWI(4+4*mu+nu,0,VOLUME));
     
        for(x0=0; x0<T; x0++) {
	  q[0] = (double)(x0+Tstart) / (double)T_global;
        for(x1=0; x1<LX; x1++) {
	  q[1] = (double)(x1) / (double)LX;
        for(x2=0; x2<LY; x2++) {
	  q[2] = (double)(x2) / (double)LY;
        for(x3=0; x3<LZ; x3++) {
	  q[3] = (double)(x3) / (double)LZ;
	  ix = g_ipt[x0][x1][x2][x3];
	  w.re = cos( M_PI * ( q[mu] - q[nu] - 2.*(sx0*q[0]+sx1*q[1]+sx2*q[2]+sx3*q[3])) );
	  w.im = sin( M_PI * ( q[mu] - q[nu] - 2.*(sx0*q[0]+sx1*q[1]+sx2*q[2]+sx3*q[3])) );
/*          fprintf(stdout, "mu=%3d, nu=%3d, t=%3d, x=%3d, y=%3d, z=%3d, phase= %21.12e + %21.12ei\n", \
              mu, nu, x0, x1, x2, x3, w.re, w.im); */
	  _co_eq_co_ti_co(&w1, cp1, cp2);
	  _co_eq_co_ti_co(cp3, &w1, &w);
          /* _co_ti_eq_re(cp3, fnorm); */
	  cp1++; cp3++;
	}
	}
	}
	}

      }
      }
  
      /* save the result in momentum space */
      sprintf(filename, "outcvc_P.%.4d.%.4d", Nconf, count);
      write_contraction(work+_GWI(4,0,VOLUME), NULL, filename, 16, 2, 0);

#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "time to cvc save results: %e seconds\n", retime-ratime);

    }  /* of count % Nsave == 0 */

  }  /* of loop on sid */

  if(g_resume==1) {
    /* write current disc to file */
    sprintf(filename, ".outcvc_current.%.4d", Nconf);
    write_contraction(disc, &count, filename, 4, 0, 0);

  }

  /**************************************
   * free the allocated memory, finalize
   **************************************/
  free(g_gauge_field);
  if(do_gt==1) free(gauge_trafo);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(disc); free(disc2);
  free(work);
#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
#endif

  return(0);

}
