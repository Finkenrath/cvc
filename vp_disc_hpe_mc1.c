/*********************************************************************************
 * vp_disc_hpe_mc1.c
 *
 * Fri Nov 20 00:50:53 CET 2009 
 *
 * PURPOSE:
 * - calculate the the Hopping-parameter expansion (HPE) contribution to 
 *   i Im(Tr[(1-gamma_mu)U_mu(x) S_u(x+mu,x)])  to nth order
 * - calculate the loop contributions
 * - use iterative method for HPE and loop reduction
 * TODO:
 * - current version _DOES NOT_ work with MPI
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
#include "Q_phi2_red.h"
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
  
  int c, mu;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int ix;
  int gid, nloop[HPE_MAX_ORDER];
  int **loop_tab[HPE_MAX_ORDER];
  int **sigma_tab[HPE_MAX_ORDER];
  int **shift_start[HPE_MAX_ORDER];
  double *tcf[HPE_MAX_ORDER], *tcb[HPE_MAX_ORDER];
  double *disc  = (double*)NULL;
  int verbose = 0;
  int do_gt   = 0;
  int deg;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq;
  double *gauge_trafo=(double*)NULL;
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
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
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

  if(hpe_order_min==-1 && hpe_order_max==-1) {
    fprintf(stdout, "# min/max order is -1; exit\n");
    exit(0);
  }
  else if(hpe_order_min==-1 && hpe_order_max>= 0) {hpe_order_min=3;}
  else if(hpe_order_min>= 0 && hpe_order_max==-1) {hpe_order_max=hpe_order_min;}
  if(hpe_order_min%2==0 && hpe_order_min>0) {
    hpe_order_min--;
    fprintf(stdout, "Attention: HPE min order reset to %d\n", hpe_order_min);
  }
  if(hpe_order_max%2==0 && hpe_order_max>0) {
    hpe_order_max--;
    fprintf(stdout, "Attention: HPE max order reset to %d\n", hpe_order_max);
  }

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* vp_disc_hpe_loops with HPE of order %d to %d\n", hpe_order_min, hpe_order_max);
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

  /*********************************************
   * allocate memory for the gauge field
   *********************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  if(do_gt==1) {
    /***********************************
     * initialize gauge transformation
     ***********************************/
    init_gauge_trafo(&gauge_trafo, 1.);
  }

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


  /******************************************************************************************
   * init HPE fields
   ******************************************************************************************/
  init_hpe_fields(loop_tab, sigma_tab, shift_start, tcf, tcb);

  /************************************************
   * loop on gauge configurations
   ************************************************/
  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, gid);
    if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
    if( read_lime_gauge_field_doubleprec(filename) != 0 ) {
      if(g_cart_id==0) fprintf(stderr, "Error could not read gauge field %s\n", filename);
      continue;
    }
    xchange_gauge();
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

    if(do_gt==1) {
      apply_gt_gauge(gauge_trafo);
      plaquette(&plaq);
      if(g_cart_id==0) fprintf(stdout, "# measured plaquette value after gauge trafo: %25.16e\n", plaq);
    }

    for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;

    for(mu=0; mu<4; mu++) {
      for(deg=hpe_order_min; deg<=hpe_order_max; deg+=2) {  
#ifdef MPI
        ratime = MPI_Wtime();
#else
        ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
        init_trace_coeff_red(&(tcf[deg]), &(tcb[deg]), &(loop_tab[deg]), &(sigma_tab[deg]), &(shift_start[deg]), deg, nloop+deg, mu);
        reduce_loop_tab(loop_tab[deg], sigma_tab[deg], shift_start[deg], deg, nloop[deg]);
#ifdef MPI
        retime = MPI_Wtime();
#else
        retime = (double)clock() / CLOCKS_PER_SEC;
#endif
        if(g_cart_id==0) fprintf(stdout, "# time initialize trace coefficients of order %d: %e seconds\n", deg, retime-ratime);
      }
#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

      for(deg=hpe_order_min; deg<=hpe_order_max; deg+=2) {
        for(ix=0; ix<VOLUME; ix++) {
          Hopping_iter_mc_red( disc+_GWI(mu,ix,VOLUME), tcf[deg], ix, mu, deg, nloop[deg], loop_tab[deg], sigma_tab[deg], shift_start[deg]);
        }
      }
#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "# time to calculate loops for gauge id %d: %e seconds\n", gid, retime-ratime);
    }

/*
    sprintf(filename, "vp_disc_hpe%.2d_mc_Xascii.%.4d", hpe_order_max, gid);
    if( (ofs = fopen(filename, "w")) == (FILE*)NULL ) exit(114);
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<4; mu++) { 
        fprintf(ofs, "%6d%3d%25.16e\t%25.16e\n", ix, mu, disc[_GWI(mu,ix,VOLUME)], \
                disc[_GWI(mu,ix,VOLUME)+1]);
      }
    }
    fclose(ofs);
*/

    sprintf(filename, "vp_disc_hpe%.2d_mc_X.%.4d", hpe_order_max, gid);
    sprintf(contype, "cvc-disc-hpe-loops-%.2d-to-%.2d-iter", hpe_order_min, hpe_order_max);
    write_lime_contraction(disc, filename, 64, 4, contype, gid, -1);

  }


  /******************************************************************************************
   * free HPE fields
   ******************************************************************************************/
  free_hpe_fields(loop_tab, sigma_tab, shift_start, tcf, tcb);

  /****************************************
   * prepare Fourier transformation arrays
   ****************************************/
/*
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }
*/

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(g_gauge_field);
  free_geometry();
  free(disc);
/*  free(in); */

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
