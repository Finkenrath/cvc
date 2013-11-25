/*********************************************************************************
 * pi_ud_p.c 
 *
 * Wed Dec 22 18:54:36 CET 2010
 *
 * PURPOSE:
 * - use the out put fields of jc_ud_x to construct the disconn. correlator
 *   of the vacuum polarization tensor
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
  int use_real_part = 1;
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

  while ((c = getopt(argc, argv, "h?vgf:r:")) != -1) {
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
    case 'r':
      use_real_part = atoi(optarg);
      if(use_real_part==1) fprintf(stdout, "# will use real part of el. magn. current estimator\n");
      else fprintf(stdout, "# will use imaginary part of el. magn. current estimator\n"); 
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

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* pi_ud_p\n");
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
      sprintf(filename, "jc_ud_x.%.4d.%.4d", gid, sid);
      status = read_lime_contraction(disc, filename, 4, 0);

      if(status!=0) {
        fprintf(stderr, "Error, could not read contraction data from file %s\n", filename);
        exit(121);
      }

#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "# time to read contractions: %e seconds\n", retime-ratime);

      count++;

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
     
      if(use_real_part==1) { /* use only the real part of the current estimator */
          for(ix=0; ix<4*VOLUME; ix++) disc[2*ix+1] = 0.;
      } else if(use_real_part==-1) { /* use only imaginary part */
          for(ix=0; ix<4*VOLUME; ix++) disc[2*ix] = 0.;
      }
      for(ix=0; ix<8*VOLUME; ix++) data[ix] += disc[ix];

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
        fnorm = 1. / ( (double)(T_global*LX*LY*LZ) * (double)(count*(count-1)) );
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
        sprintf(filename, "pi_ud_P.%.4d.%.4d", gid, count);
        sprintf(contype, "cvc-disc-u_and_d-stoch-subtracted-P");
        write_lime_contraction(work, filename, 64, 16, contype, gid, count);
        sprintf(filename, "pi_ud_P.%.4d.%.4d.ascii", gid, count);
        write_contraction(work, NULL, filename, 16, 2, 0);
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
