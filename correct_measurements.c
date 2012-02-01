/*********************************************************************************
 * correct_measurements.c
 *
 * Fri Jan 22 17:28:23 CET 2010 
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
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int gid, status;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  char filename[100], contype[200];
  double ratime, retime;
  complex w, w1, *cp1, *cp2, *cp3;
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

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
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

  if(hpe_order%2==0 && hpe_order>0) {
    hpe_order--;
    fprintf(stdout, "Attention: HPE order reset to %d\n", hpe_order);
  }

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* correct_measurements\n");
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
  if( disc2== (double*)NULL ) { 
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

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    /****************************************
     * read the loop part
     ****************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(hpe_order>0) {
      sprintf(filename, "%s_X.%.4d", gaugefilename_prefix, gid);
      if(g_cart_id==0) fprintf(stdout, "# reading loop part from file %s\n", filename);
      if( (status = read_lime_contraction(disc2, filename, 4, 0)) == 106 ) {
/*
#ifdef MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
#endif
        exit(101);
*/
        fprintf(stderr, "Error, could not read from file %s, continue\n");
        continue;
      }
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to read loop contributions: %e seconds\n", retime-ratime);

    sprintf(filename, "%s_X.%.4d.%.4d", filename_prefix, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Reading old X-data from file %s\n", filename);
    if(read_lime_contraction(disc, filename, 4, 0) == 106 ) {
      fprintf(stderr, "Error, could not read from file %s; continue\n", filename);
      continue;
    }
   
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        work[_GWI(mu,ix,VOLUME)  ] = disc[_GWI(mu,ix,VOLUME)  ] - disc2[_GWI(mu,ix,VOLUME)  ];
        work[_GWI(mu,ix,VOLUME)+1] = disc[_GWI(mu,ix,VOLUME)+1] - disc2[_GWI(mu,ix,VOLUME)+1];
      }
    }
    
    fnorm = 1. / g_prop_normsqr;
    if(g_cart_id==0) fprintf(stdout, "# X-fnorm = %e\n", fnorm);
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        work[_GWI(mu,ix,VOLUME)  ] = work[_GWI(mu,ix,VOLUME)  ] * fnorm + disc2[_GWI(mu,ix,VOLUME)  ];
        work[_GWI(mu,ix,VOLUME)+1] = work[_GWI(mu,ix,VOLUME)+1] * fnorm + disc2[_GWI(mu,ix,VOLUME)+1];
      }
    }

    sprintf(filename, "%s_X.%.4d.%.4d", filename_prefix2, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Writing new X-data to file %s\n", filename);
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
    if(g_cart_id==0) fprintf(stdout, "P-fnorm = %e\n", fnorm);
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
    if(g_cart_id==0) fprintf(stdout, "# Writing new P-data to file %s\n", filename);
    sprintf(contype, "%s_P", filename_prefix);
    write_lime_contraction(work+_GWI(8,0,VOLUME), filename, 64, 16, contype, gid, Nsave);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to save cvc results: %e seconds\n", retime-ratime);
  }  /* of loop on gid */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  fftw_free(in);
  free(disc);
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
