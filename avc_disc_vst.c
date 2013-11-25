/****************************************************
 * avc_disc_vst.c
 *
 * Fri Oct 16 16:57:12 CEST 2009
 *
 * PURPOSE:
 * - calculate contractions for disconnected contributions
 *   to the vacuum polarization using the volume
 *   source technique
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
#include "Q_phi.h"

void usage() {
  fprintf(stdout, "Code to perform VP VST contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:\n");
/*  fprintf(stdout, " -g apply a random gauge transformation\n"); */
  fprintf(stdout, " -f input filename [default cvc.input]\n");
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
  int sid;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  double unit_trace[2], shift_trace[2], D_trace[2];
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
  fftwnd_mpi_plan plan_p, plan_m;
  int *status;
#else
  fftwnd_plan plan_p, plan_m;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?gf:")) != -1) {
    switch (c) {
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
  no_fields = 1;
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

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  /***********************************************
   * start loop on source id.s 
   ***********************************************/
  for(sid=0; sid<12; sid++) {

    /* read the new propagator */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(format==0) {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
      /* sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, sid); */
      if(read_lime_spinor(g_spinor_field[0], filename, 0) != 0) break;
    }
    else if(format==1) {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
      if(read_cmi(g_spinor_field[0], filename) != 0) break;
    }
    xchange_field(g_spinor_field[0]);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    fprintf(stdout, "time to read prop.: %e seconds\n", retime-ratime);

    count++;

    /* calculate the source: apply Q_phi_tbc */
/*
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
    if(g_cart_id==0) 
      fprintf(stdout, "time to calculate source: %e seconds\n", retime-ratime);
*/


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
        _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[0][_GSI(g_iup[ix][mu])]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_mi_eq_fv(spinor2, spinor1);
/*
        _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor2);
	disc[iix  ] -= 0.5 * w.re;
	disc[iix+1] -= 0.5 * w.im;
*/
	disc[iix  ] -= 0.5 * spinor2[2*sid  ];
	disc[iix+1] -= 0.5 * spinor2[2*sid+1];

        /* second contribution */
	_fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[0][_GSI(ix)]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_pl_eq_fv(spinor2, spinor1);
/*
        _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(g_iup[ix][mu])], spinor2);
	disc[iix  ] -= 0.5 * w.re;
	disc[iix+1] -= 0.5 * w.im;
*/
	disc[iix  ] -= 0.5 * spinor2[2*sid  ];
	disc[iix+1] -= 0.5 * spinor2[2*sid+1];

	iix += 2;
      }  /* of ix */
    }    /* of mu */

#  ifdef MPI
    retime = MPI_Wtime();
#  else
    retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    fprintf(stdout, "[%2d] contractions in %e seconds\n", g_cart_id, retime-ratime);
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
        _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[0][_GSI(g_iup[ix][mu])]);
	_fv_eq_gamma_ti_fv(spinor2, 6+mu, spinor1);
	disc2[iix  ] -= 0.5 * spinor2[2*sid  ];
	disc2[iix+1] -= 0.5 * spinor2[2*sid+1]; 

        /* second contribution */
	_fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[0][_GSI(ix)]);
	_fv_eq_gamma_ti_fv(spinor2, 6+mu, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(g_iup[ix][mu])], spinor2);
	disc2[iix  ] -= 0.5 * spinor2[2*sid  ];
	disc2[iix+1] -= 0.5 * spinor2[2*sid+1];

	iix += 2;
      }  /* of ix */
    }    /* of mu */

#  ifdef MPI
    retime = MPI_Wtime();
#  else
    retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    fprintf(stdout, "[%2d] contractions in %e seconds\n", g_cart_id, retime-ratime);
#endif /* of AVC */

    if(g_cart_id==0) {
      /***************************************************
       * check: convergence of trace of unit matrix
       ***************************************************/
       _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_source_location), g_spinor_field[0]+_GSI(g_source_location));
       fprintf(stdout, "unit_trace: %4d%25.16e%25.16e\n", count, w.re, w.im);
       _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_source_location), g_spinor_field[0]+_GSI(g_iup[g_source_location][0]));
       fprintf(stdout, "shift_trace: %4d%25.16e%25.16e\n", count, w.re, w.im);
    }

    /***************************************************
     * check: convergence of trace D_u(source_location, source_location)
     ***************************************************/
/*
    Q_phi_tbc(g_spinor_field[1], g_spinor_field[0]);
    xchange_field(g_spinor_field[1]);
    if(g_cart_id==0) {
       _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_source_location), g_spinor_field[1]+_GSI(g_source_location));
       fprintf(stdout, "D_trace: %4d%25.16e%25.16e\n", count, w.re, w.im); 
    }
*/

  }  /* of loop on source id */

  /************************************************
   * save results
   ************************************************/

  if(g_cart_id == 0) fprintf(stdout, "save results\n");

  /* save the result in position space */
#ifdef CVC
  sprintf(filename, "outcvc_X_vst.%.4d", Nconf);
  write_contraction(disc, NULL, filename, 4, 2, 0);
#endif

#ifdef AVC
  sprintf(filename, "outavc_X_vst.%.4d", Nconf);
  write_contraction(disc2, NULL, filename, 4, 2, 0);
#endif

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

#ifdef CVC
  /* Fourier transform data, copy to work */
  for(mu=0; mu<4; mu++) {
    memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
    fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
    fftwnd_one(plan_m, in, NULL);
#  endif
    memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

    memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
    fftwnd_one(plan_p, in, NULL);
#  endif
    memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
  }  /* of mu =0 ,..., 3*/

  fnorm = 1. / ((double)(T_global*LX*LY*LZ));
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
  sprintf(filename, "outcvc_P_vst.%.4d", Nconf);
  write_contraction(work+_GWI(8,0,VOLUME), NULL, filename, 16, 0, 0);
#endif /* of CVC */

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "time to cvc save results: %e seconds\n", retime-ratime);

#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

#ifdef AVC
  /* Fourier transform data, copy to work */
  for(mu=0; mu<4; mu++) {
    memcpy((void*)in, (void*)(disc2+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
    fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
    fftwnd_one(plan_m, in, NULL);
#  endif
    memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

    memcpy((void*)in, (void*)(disc2+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#  ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#  else
    fftwnd_one(plan_p, in, NULL);
#  endif
    memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
  }  /* of mu =0 ,..., 3*/

  fnorm = 1. / ((double)(T_global*LX*LY*LZ));
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
  sprintf(filename, "outavc_P_vst.%.4d", Nconf);
  write_contraction(work+_GWI(8,0,VOLUME), NULL, filename, 16, 0, 0);
#endif /* of AVC */

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "time to save avc results: %e seconds\n", retime-ratime);


  /* free the allocated memory, finalize */
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
#ifdef CVC
  free(disc);
#endif

#ifdef AVC
  free(disc2);
#endif

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
