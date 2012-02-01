/****************************************************
 * lvc_disc_stochastic.c
 *
 * Fri Sep 18 17:24:06 CEST 2009
 *
 * PURPOSE:
 * - calculate quark-disc. contribution to vacuum polarization
 *   from local (axial) vector current
 * TODO:
 * - implementation
 * - checks
 * DONE:
 * CHANGES:
 * 
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
  int sid;
  double *disc       = (double*)NULL;
  double *disc2      = (double*)NULL;
  double *work       = (double*)NULL;
  double *disc_diag  = (double*)NULL;
  double *disc_diag2 = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24];
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
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

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

  /************************
   * read the gauge field *
   ************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();

  /*************************
   * measure the plaquette *
   *************************/
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /*****************************************
   * allocate memory for the spinor fields *
   *****************************************/
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
#ifdef CVC
  disc2  = (double*)calloc( 8*VOLUME, sizeof(double));
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

#ifdef AVC
  disc = (double*)calloc( 8*VOLUME, sizeof(double));
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

  work  = (double*)calloc(48*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  if(g_subtract == 1) {
    /* allocate memory for disc_diag */
#ifdef AVC
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
#endif /* of AVC */

#ifdef CVC
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
#endif /* of CVC */
  }

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

  if(g_resume==1) { /* read current disc2 from file */
#ifdef CVC
    sprintf(filename, ".outcvc_current.%.4d", Nconf);
    c = read_contraction(disc2, &count, filename, 4);
    if( (g_subtract==1) && (c==0) ) {
      sprintf(filename, ".outcvc_diag_current.%.4d", Nconf);
      c = read_contraction(disc_diag2, (int*)NULL, filename, 16);
    }
#else
    c = 1;
#endif

#ifdef AVC
    if(c!=0) {
      sprintf(filename, ".outavc_current.%.4d", Nconf);
      c = read_contraction(disc, &count, filename, 4);
    }
    if( (g_subtract==1) && (c==0) ) {
      sprintf(filename, ".outavc_diag_current.%.4d", Nconf);
      c = read_contraction(disc_diag, (int*)NULL, filename, 16);
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
      for(ix=0; ix<8*VOLUME; ix++) disc2[ix] = 0.;
      if(g_subtract==1) for(ix=0; ix<32*VOLUME; ix++) disc_diag2[ix] = 0.;
#  endif
#  ifdef AVC
      for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
      if(g_subtract==1) for(ix=0; ix<32*VOLUME; ix++) disc_diag[ix] = 0.;
#  endif
      count = 0;
    }
#endif
    if(g_cart_id==0) fprintf(stdout, "starting with count = %d\n", count);
  }  /* of g_resume ==  1 */
  
  /*****************************
   * start loop on source id.s *
   *****************************/
  for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {

    /*******************************************
     * read the new propagator 
     *******************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
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
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    fprintf(stdout, "time to read prop.: %e seconds\n", retime-ratime);

    count++;

    /*******************************************
     * calculate the source: apply Q_phi_tbc 
     *******************************************/
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
    /********************************************
     * add new contractions to (existing) disc2 
     ********************************************/
#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif

    for(mu=0; mu<4; mu++) { // loop on Lorentz index of the current
      iix = _GWI(mu,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {    // loop on lattice sites

	_fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[1][_GSI(ix)]);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1);
#ifndef QMAT
	disc2[iix  ] -= w.re;
	disc2[iix+1] -= w.im;
#else
	disc2[iix  ] -= w.re;
	disc2[iix+1] -= w.im / 3.;
#endif
        if(g_subtract==1) {
#ifndef QMAT
	  work[iix  ] = -w.re;
	  work[iix+1] = -w.im;
#else
	  work[iix  ] = -w.re;
	  work[iix+1] = -w.im / 3.;
#endif
	}

	iix += 2;
      }  // of ix
    }    // of mu

    // for the purpose of testing
/*
      iix = _GWI(0,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {
	_fv_eq_gamma_ti_fv(spinor1, 5, &g_spinor_field[1][_GSI(ix)]);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1);
	disc2[iix  ] -= w.re;
	disc2[iix+1] -= w.im;
	iix += 2;
      }
      iix = _GWI(1,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {
	_fv_eq_gamma_ti_fv(spinor1, 4, &g_spinor_field[1][_GSI(ix)]);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1);
	disc2[iix  ] -= w.re;
	disc2[iix+1] -= w.im;
	iix += 2;
      }
      iix = _GWI(2,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {
	_fv_eq_gamma_ti_fv(spinor1, 5, &g_spinor_field[1][_GSI(ix)]);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[1][_GSI(ix)], spinor1);
	disc2[iix  ] -= w.re;
	disc2[iix+1] -= w.im;
	iix += 2;
      }
      iix = _GWI(3,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {
	_fv_eq_gamma_ti_fv(spinor1, 4, &g_spinor_field[1][_GSI(ix)]);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[1][_GSI(ix)], spinor1);
	disc2[iix  ] -= w.re;
	disc2[iix+1] -= w.im;
	iix += 2;
      }
*/

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
#endif /* of CVC */


#ifdef AVC
    /* add new contractions to (existing) disc */
#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    for(mu=0; mu<4; mu++) { /* loop on Lorentz index of the current */
      iix = _GWI(mu,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {    /* loop on lattice sites */

        /* first contribution */
	_fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[1][_GSI(ix)]);
	_co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1);
	disc[iix  ] -= w.re;
	disc[iix+1] -= w.im; 
        if(g_subtract==1) {
	  work[iix  ] = -w.re;
	  work[iix+1] = -w.im; 
	}

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
#endif /* of AVC */


    /*********************************************
     * save results for count = multiple of Nsave 
     *********************************************/
    if(count%Nsave == 0) {

      if(g_cart_id == 0) fprintf(stdout, "save results for count = %d\n", count);

      /* save the result in position space */

#ifdef CVC
#  ifdef MPI
      ratime = MPI_Wtime();
#  else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      fnorm = 1. / ( (double)count * g_prop_normsqr );
      if(g_cart_id==0) fprintf(stdout, "# X-fnorm = %e\n", fnorm);
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          work[_GWI(mu,ix,VOLUME)  ] = disc2[_GWI(mu,ix,VOLUME)  ] * fnorm;
          work[_GWI(mu,ix,VOLUME)+1] = disc2[_GWI(mu,ix,VOLUME)+1] * fnorm;
        }
      }
      sprintf(filename, "outlvc_X.%.4d.%.4d", Nconf, count);
      sprintf(contype, "lvc_disc_stochastic_X");
      write_lime_contraction(work, filename, 64, 4, contype, Nconf, count);

      sprintf(filename, "outlvc_X.%.4d.%.4d.ascii", Nconf, count);
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
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        cp1 = (complex*)(work+_GWI(mu,0,VOLUME));
        cp2 = (complex*)(work+_GWI(4+nu,0,VOLUME));
        cp3 = (complex*)(work+_GWI(8+4*mu+nu,0,VOLUME));
     
        for(x0=0; x0<T; x0++) {
        for(x1=0; x1<LX; x1++) {
        for(x2=0; x2<LY; x2++) {
        for(x3=0; x3<LZ; x3++) {
	  ix = g_ipt[x0][x1][x2][x3];
	  _co_eq_co_ti_co(&w1, cp1, cp2);
	  if(g_subtract==1) {
	    _co_mi_eq_co(&w1, (complex*)(disc_diag2+_GWI(4*mu+nu,ix,VOLUME)));
	  }
          cp3->re = w1.re * fnorm;
          cp3->im = w1.im * fnorm;
	  cp1++; cp2++; cp3++;
	}}}}
      }}
  
      /* save the result in momentum space */
      sprintf(filename, "outlvc_P.%.4d.%.4d", Nconf, count);
      sprintf(contype, "lvc_disc_stochastic_P");
      write_lime_contraction(work+_GWI(8,0,VOLUME), filename, 64, 16, contype, Nconf, count);
#  ifdef MPI
      retime = MPI_Wtime();
#  else
      retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      if(g_cart_id==0) fprintf(stdout, "# time to cvc save results: %e seconds\n", retime-ratime);
#endif /* of CVC */



#ifdef AVC
#  ifdef MPI
      ratime = MPI_Wtime();
#  else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      fnorm = 1. / ( (double)count * g_prop_normsqr );
      if(g_cart_id==0) fprintf(stdout, "# X-fnorm = %e\n", fnorm);
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          work[_GWI(mu,ix,VOLUME)  ] = disc[_GWI(mu,ix,VOLUME)  ] * fnorm;
          work[_GWI(mu,ix,VOLUME)+1] = disc[_GWI(mu,ix,VOLUME)+1] * fnorm;
        }
      }
      sprintf(filename, "outlavc_X.%.4d.%.4d", Nconf, count);
      sprintf(contype, "lavc_disc_stochastic_X");
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
      sprintf(filename, "outlavc_P.%.4d.%.4d", Nconf, count);
      sprintf(contype, "lavc_disc_stochastic_P");
      write_lime_contraction(work+_GWI(8,0,VOLUME), filename, 64, 16, contype, Nconf, count);
#  ifdef MPI
      retime = MPI_Wtime();
#  else
      retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
      if(g_cart_id==0) fprintf(stdout, "# time to save avc results: %e seconds\n", retime-ratime);
#endif /* of AVC */

    }  /* of count % Nsave == 0 */
  }  /* of loop on sid */

  if(g_resume==1) {
    /* write current disc to file */
#ifdef CVC
    sprintf(filename, ".outlvc_current.%.4d", Nconf);
    write_contraction(disc2, &count, filename, 4, 0, 0);
    if(g_subtract == 1) {
      /* write current disc_diag2 to file */
      sprintf(filename, ".outlvc_diag_current.%.4d", Nconf);
      write_contraction(disc_diag2, (int*)NULL, filename, 16, 0, 0);
    }
#endif

#ifdef AVC
    sprintf(filename, ".outlavc_current.%.4d", Nconf);
    write_contraction(disc, &count, filename, 4, 0, 0);
    if(g_subtract == 1) {
      /* write current disc_diag to file */
      sprintf(filename, ".outlavc_diag_current.%.4d", Nconf);
      write_contraction(disc_diag, (int*)NULL, filename, 16, 0, 0);
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
  free(disc2);
  if(g_subtract==1) free(disc_diag2);
#endif

#ifdef AVC
  free(disc);
  if(g_subtract==1) free(disc_diag);
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
