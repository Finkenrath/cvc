/*********************************************************************************
 * vp_disc_ud_xspace.c
 *
 * Fri Jul 16 19:12:18 CEST 2010 
 *
 * PURPOSE:
 * TODO:
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
#include "contractions_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"

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
  int sid, status, gid;
  double *disc = (double*)NULL;
  double *data = (double*)NULL;
  double *work = (double*)NULL;
  double *bias = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq; 
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1, *cp1=NULL, *cp2=NULL, *cp3=NULL, *cp4=NULL;
/*  FILE *ofs; */

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

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* vp_disc_ud_stoch\n");
  fprintf(stdout, "**************************************************\n\n");

  /*********************************
   * initialize MPI parameters 
   *********************************/
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

  /*************************************************
   * allocate mem for gauge field and spinor fields
   *************************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /****************************************
   * allocate memory for the contractions
   ****************************************/
  disc  = (double*)calloc(16*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  data  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( data == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for data\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  work  = (double*)calloc(32*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  bias  = (double*)calloc(32*VOLUME, sizeof(double));
  if( bias == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for bias\n");
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

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    for(ix=0; ix< 8*VOLUME; ix++) data[ix] = 0.;
    for(ix=0; ix<32*VOLUME; ix++) work[ix] = 0.;
    for(ix=0; ix<32*VOLUME; ix++) bias[ix] = 0.;

    sprintf(filename, "%s.%.4d", gaugefilename_prefix, gid);
    if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
    xchange_gauge();
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);
    count = 0;
    /***********************************************
     * start loop on source id.s 
     ***********************************************/
    for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
      for(ix=0; ix<16*VOLUME; ix++) disc[ix] = 0.;

      /* read the new propagator to g_spinor_field[0] */
#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(format==0) {
        sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, gid, sid);
        if(read_lime_spinor(g_spinor_field[0], filename, 0) != 0) break;
      }
      else if(format==1) {

        sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, gid, sid);
        if(read_cmi(g_spinor_field[0], filename) != 0) break;

/*
        sprintf(filename, "%s.%.5d.%.6d", filename_prefix, sid, gid);
        if(read_cmi_3(g_spinor_field[0], filename) != 0) break;
*/
      }
      xchange_field(g_spinor_field[0]);
#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "# time to read prop.: %e seconds\n", retime-ratime);

      count++;

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
      /* apply D_W once, save in g_spinor_field[1] */
      Hopping(g_spinor_field[1], g_spinor_field[0]);
      for(ix=0; ix<VOLUME; ix++) {
        _fv_pl_eq_fv(g_spinor_field[1]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
        _fv_ti_eq_re(g_spinor_field[1]+_GSI(ix),  1./(2.*g_kappa));
      }
      xchange_field(g_spinor_field[1]);
#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "# time to apply D_W: %e seconds\n", retime-ratime);

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
      /* calculate real and imaginary part */
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]));
          _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[0]+_GSI(g_iup[ix][mu]));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_pl_eq_fv(spinor2, spinor1);
          _fv_eq_cm_ti_fv(spinor1, U_, spinor2);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor1);
          disc[_GWI(mu,ix,VOLUME)  ]  = g_mu * w.im;
          data[_GWI(mu,ix,VOLUME)  ] += g_mu * w.im;

          _fv_eq_gamma_ti_fv(spinor1, mu, g_spinor_field[1]+_GSI(g_iup[ix][mu]));
          _fv_pl_eq_fv(spinor1, g_spinor_field[1]+_GSI(g_iup[ix][mu]));
          _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor2);
          disc[_GWI(mu,ix,VOLUME)+1]  = w.im / 3.;
          data[_GWI(mu,ix,VOLUME)+1] += w.im / 3.;
        }
      }
#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "# time to calculate contractions: %e seconds\n", retime-ratime);

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)(disc+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

        memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)(disc+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        cp1 = (complex*)(disc+_GWI(mu,     0,VOLUME));
        cp2 = (complex*)(disc+_GWI(4+nu,   0,VOLUME));
        cp3 = (complex*)(bias+_GWI(4*mu+nu,0,VOLUME));
        for(ix=0; ix<VOLUME; ix++) {
	  _co_eq_co_ti_co(&w, cp1, cp2);
          cp3->re += w.re;
          cp3->im += w.im;
	  cp1++; cp2++; cp3++;
 	}
      }}

      if(count==Nsave) {
        for(mu=0; mu<4; mu++) {
          memcpy((void*)in, (void*)(data+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
          fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
          fftwnd_one(plan_m, in, NULL);
#endif
          memcpy((void*)(disc+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

          memcpy((void*)in, (void*)(data+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
          fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
          fftwnd_one(plan_p, in, NULL);
#endif
          memcpy((void*)(disc+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
        }

        /************************************************
         * save results for count = multiple of Nsave 
         ************************************************/
        if(g_cart_id == 0) fprintf(stdout, "# save results for gauge id %d and count = %d\n", gid, count);
        fnorm = 1. / ( (double)(T_global*LX*LY*LZ) * (double)(count*(count-1)) * g_prop_normsqr*g_prop_normsqr );
        if(g_cart_id==0) fprintf(stdout, "# P-fnorm = %25.16e\n", fnorm);
        for(mu=0; mu<4; mu++) {
        for(nu=0; nu<4; nu++) {
          cp1 = (complex*)(disc+_GWI(mu,0,VOLUME));
          cp2 = (complex*)(disc+_GWI(4+nu,0,VOLUME));
          cp3 = (complex*)(work+_GWI(4*mu+nu,0,VOLUME));
          cp4 = (complex*)(bias+_GWI(4*mu+nu,0,VOLUME));
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
            w1.re -= cp4->re;
            w1.im -= cp4->im;
	    _co_eq_co_ti_co(cp3, &w1, &w);
            cp3->re *= fnorm;
            cp3->im *= fnorm;
	    cp1++; cp2++; cp3++; cp4++;
 	  }}}}
        }}
  
        /* save the result in momentum space */
        sprintf(filename, "vp_disc_ud_subtracted_P.%.4d.%.4d", gid, count);
        sprintf(contype, "cvc-disc-u_and_d-stoch-subtracted-P");
        write_lime_contraction(work, filename, 64, 16, contype, gid, count);
#ifdef MPI
        retime = MPI_Wtime();
#else
        retime = (double)clock() / CLOCKS_PER_SEC;
#endif
        if(g_cart_id==0) fprintf(stdout, "# time to save cvc results: %e seconds\n", retime-ratime);
        break;
      }  /* of count % Nsave == 0 */
 
    }  /* of loop on sid */
  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(disc);
  free(data);
  free(work);
  free(bias);
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
