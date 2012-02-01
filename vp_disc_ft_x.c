/*********************************************************************************
 * vp_disc_ft_x.c
 *
 * Sun Mar 28 09:34:52 CEST 2010
 *
 * PURPOSE:
 * - calculate the r-dependence of Pi^dis for each t
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
#include "pidisc_model.h"
#include "make_q2orbits.h"
#include "make_H3orbits.h"

void usage() {
  fprintf(stdout, "Code to perform quark-disconnected conserved vector current contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, nu, mode=0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3;
  unsigned long int ix, iix, iiy;
  int gid, status;
  int model_type = -1;
  fftw_complex *disc  = NULL;
  fftw_complex *disc2 = NULL;
  fftw_complex *work  = NULL;
  double q[4], fnorm;
  int *oh_count=(int*)NULL, *oh_id=(int*)NULL, oh_nc=0;
  int *rid = (int*)NULL, rcount, *picount=(int*)NULL;
  double *rlist = (double*)NULL;
  double *pir   = (double*)NULL;
  double **oh_val = (double**)NULL;
  char filename[100], contype[200];
  double ratime, retime;
  double rmin2, rmax2, rsqr;
  complex w, w1;
  FILE *ofs=NULL;

  int mu_nu_comb[4][2] = {{0,0},{0,1},{1,2},{1,1}};
  int mu_nu_tab[16] = {0, 1, 1, 1, 1, 3, 2, 2, 1, 2, 3, 2, 1, 2, 2, 3};

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
#else
  fftwnd_plan plan_p, plan_m;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:t:m:")) != -1) {
    switch (c) {
    case 't':
      model_type = atoi(optarg);
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'm':
      mode = atoi(optarg);
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
  fprintf(stdout, "* vp_disc_ft_2\n");
  fprintf(stdout, "**************************************************\n\n");


  /*********************************
   * initialize MPI parameters 
   *********************************/
  mpi_init(argc, argv);

  /* initialize fftw */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  plan_m = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_FORWARD,  FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
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

  disc  = (fftw_complex*)fftw_malloc(16*VOLUME*sizeof(fftw_complex));
  if( disc == (fftw_complex*)NULL ) {
    fprintf(stderr, "could not allocate memory for disc\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  disc2 = (fftw_complex*)fftw_malloc(16*VOLUME*sizeof(fftw_complex));
  if( disc2 == (fftw_complex*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc2\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(4);
  }

  work = (fftw_complex*)fftw_malloc(16*VOLUME*sizeof(fftw_complex));
  if( work == (fftw_complex*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(5);
  }


#ifndef MPI
  plan_p = fftwnd_create_plan_specific(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE, work, 16, work, 16);
  plan_m = fftwnd_create_plan_specific(4, dims, FFTW_FORWARD,  FFTW_MEASURE | FFTW_IN_PLACE, work, 16, work, 16);
#endif

  /***************************************
   * initialize the r=const list
   ***************************************/
  make_rid_list(&rid, &rlist, &rcount, g_rmin, g_rmax);

  /***************************************
   * initialize the O-h orbits
   ***************************************/
  make_Oh_orbits_r(&oh_id, &oh_count, &oh_val, &oh_nc, g_rmin, g_rmax);

  if( (picount = (int*)malloc(rcount*sizeof(int)))==(int*)NULL) exit(6);

  /***********************************************
   * start loop on gauge id.s 
   ***********************************************/

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

    if(g_cart_id==0) fprintf(stdout, "# Start working on gauge id %d\n", gid);

#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    sprintf(filename, "%s_P.%.4d.%.4d", filename_prefix, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Reading contraction data from file %s\n", filename);
    if(read_lime_contraction((double*)disc, filename, 16, 0) == 106) {
      if(g_cart_id==0) fprintf(stderr, "Error, could not read from file %s, continue\n", filename);
      continue;
    }
    sprintf(filename, "%s_P.%.4d.%.4d", filename_prefix2, gid, Nsave);
    if(g_cart_id==0) fprintf(stdout, "# Reading contraction data from file %s\n", filename);
    if(read_lime_contraction((double*)disc2, filename, 16, 0) == 106) {
      if(g_cart_id==0) fprintf(stderr, "Error, could not read from file %s, continue\n", filename);
      continue;
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to read contraction: %e seconds\n", retime-ratime);

    /****************************************************
     * prepare \Pi_\mu\nu (q) for Fourier transformation
     * - subtract bias from biased operator
     * - multiply with inverse phase factor
     ****************************************************/

#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    for(mu=0; mu<16; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        iix = mu * VOLUME + ix;
        iiy = 16 * ix + mu;
        work[iiy].re = (disc[iix].re - disc2[iix].re) * (double)Nsave / (double)(Nsave-1);
        work[iiy].im = (disc[iix].im - disc2[iix].im) * (double)Nsave / (double)(Nsave-1);
      }
    }
    for(x0=0; x0<T; x0++) {
      q[0] = (double)(x0+Tstart) / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      q[1] = (double)x1 / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = (double)x2 / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = (double)x3 / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      i=0;
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        iix = 16 * ix + i;
        w.re =  cos(M_PI * (q[mu] - q[nu]));
        w.im = -sin(M_PI * (q[mu] - q[nu]));
        _co_eq_co_ti_co(&w1, work+iix, &w);
        work[iix].re = w1.re;
        work[iix].im = w1.im;
        i++;
      }}
    }}}}

    /************************************************
     * inverse Fourier transformation
     ************************************************/

#ifdef MPI
    fftwnd_mpi(plan_m, 16, work, disc, FFTW_NORMAL_ORDER);
#else
    fftwnd(plan_m, 16, work, 16, 1, work, 16, 1);
#endif
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for inverse Fourier transform: %e seconds\n", retime-ratime);


    /************************************************
     * prepare \Pi_\mu\nu (z)
     * - separate according to the radii g_rmin and
     *   g_rmax
     ************************************************/
  if(mode==1) {
#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    for(ix=0; ix<16*VOLUME; ix++) {
      disc[ix].re = 0.; disc[ix].im = 0.;
    }
    for(ix=0; ix<rcount; ix++) picount[ix] = 0;

    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];

      rsqr = (double)(x1*x1) + (double)(x2*x2) + (double)(x3*x3);
      
      iix = g_ipt[0][x1][x2][x3];
      if(rid[iix] == -1) continue;
      for(mu=0; mu<16; mu++) {
        disc[mu_nu_tab[mu]*T*rcount + x0*rcount + rid[iix]].re += work[16*ix+mu].re;
        disc[mu_nu_tab[mu]*T*rcount + x0*rcount + rid[iix]].im += work[16*ix+mu].im;
      }
      if(x0==0) picount[rid[iix]]++;
    }}}}

    for(x0=0; x0<T; x0++) {
      for(ix=0; ix<rcount; ix++) {
        disc[0*T*rcount + x0*rcount + ix].re /=      picount[ix];
        disc[0*T*rcount + x0*rcount + ix].im /=      picount[ix];

        disc[1*T*rcount + x0*rcount + ix].re /= 6. * picount[ix];
        disc[1*T*rcount + x0*rcount + ix].im /= 6. * picount[ix];

        disc[2*T*rcount + x0*rcount + ix].re /= 6. * picount[ix];
        disc[2*T*rcount + x0*rcount + ix].im /= 6. * picount[ix];

        disc[3*T*rcount + x0*rcount + ix].re /= 3. * picount[ix];
        disc[3*T*rcount + x0*rcount + ix].im /= 3. * picount[ix];
      }
    }

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to average over |r|=const.: %e seconds\n", retime-ratime);

    /***********************************************
     * save results
     ***********************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    sprintf(filename, "%s_R.%.4d.%.4d", gaugefilename_prefix, Nconf, Nsave);
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error, could not open file %s for writing.\n", filename);
      exit(7);
    }
    fprintf(stdout, "# writing results to file %s\n", filename);
    for(mu=0; mu<4; mu++) {
      for(x0=0; x0<T; x0++) {
        for(ix=0; ix<rcount; ix++) {
          fprintf(ofs, "%3d%3d%16.7e%25.16e%25.16e%3d\n", mu, x0, rlist[ix],
            disc[mu*T*rcount + x0*rcount + ix].re, disc[mu*T*rcount + x0*rcount + ix].im,
            picount[ix]);
        }
      }
    }
    fclose(ofs);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for saving results: %e seconds\n", retime-ratime);
  }  /* of mode==1 */

  if(mode==2) {
#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    for(ix=0; ix<16*VOLUME; ix++) {
      disc[ix].re = 0.; disc[ix].im = 0.;
    }
    for(ix=0; ix<rcount; ix++) picount[ix] = 0;

    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
      for(mu=0; mu<16; mu++) {
        disc[mu_nu_tab[mu]*T*LX + x0*LX + x1].re += 
          work[16*g_ipt[x0][x1][0][0] + mu].re +
          work[16*g_ipt[x0][0][x1][0] + mu].re + 
          work[16*g_ipt[x0][0][0][x1] + mu].re;

        disc[mu_nu_tab[mu]*T*LX + x0*LX + x1].im += 
          work[16*g_ipt[x0][x1][0][0] + mu].im +
          work[16*g_ipt[x0][0][x1][0] + mu].im +
          work[16*g_ipt[x0][0][0][x1] + mu].im;
      }
    }}

    for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
        disc[0*T*LX + x0*LX + x1].re /=      3.;
        disc[0*T*LX + x0*LX + x1].im /=      3.;

        disc[1*T*LX + x0*LX + x1].re /= 6. * 3.;
        disc[1*T*LX + x0*LX + x1].im /= 6. * 3.;

        disc[2*T*LX + x0*LX + x1].re /= 6. * 3.;
        disc[2*T*LX + x0*LX + x1].im /= 6. * 3.;

        disc[3*T*LX + x0*LX + x1].re /= 3. * 3.;
        disc[3*T*LX + x0*LX + x1].im /= 3. * 3.;
      }
    }

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to average over |r|=const.: %e seconds\n", retime-ratime);

    /***********************************************
     * save results
     ***********************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    sprintf(filename, "%s_L.%.4d.%.4d", gaugefilename_prefix, Nconf, Nsave);
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error, could not open file %s for writing.\n", filename);
      exit(7);
    }
    fprintf(stdout, "# writing results to file %s\n", filename);
    for(mu=0; mu<4; mu++) {
      for(x0=0; x0<T; x0++) {
        for(x1=0; x1<LX; x1++) {
          fprintf(ofs, "%3d%3d%3d%25.16e%25.16e\n", mu, x0, x1,
            disc[mu*T*LX + x0*LX + x1].re, disc[mu*T*LX + x0*LX + x1].im);
        }
      }
    }
    fclose(ofs);
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for saving results: %e seconds\n", retime-ratime);
  }  /* of mode==2 */

    if(g_cart_id==0) fprintf(stdout, "# Finished working on gauge id %d\n", gid);
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();

  fftw_free(disc);
  fftw_free(disc2);
  fftw_free(work);
  if(rid       != (int*)   NULL) free(rid);
  if(oh_id     != (int*)   NULL) free(oh_id);
  if(oh_count  != (int*)   NULL) free(oh_count);
  if(picount   != (int*)   NULL) free(picount);
  if(rlist     != (double*)NULL) free(rlist);
  if(oh_val    != (double**) NULL) {
    free(*oh_val); free(oh_val);
  }

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
