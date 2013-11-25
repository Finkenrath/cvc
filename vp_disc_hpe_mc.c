/*********************************************************************************
 * vp_disc_hpe_mc.c
 *
 * Tue Dec  1 09:08:17 CET 2009
 *
 * PURPOSE:
 * - calculate the disconnected contractions of the vacuum polarization 
 *   by Hopping-parameter expansion (HPE) to nth order
 * - estimate the loop contributions
 * - use stochastic method for estimation of the trace
 * TODO:
 * - current version tested against avc_disc_stochastic for HPE order
 *   0, 3 and 5 (together with vp_disc_hpe_mc2)
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
  
  int c, mu, i, count;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int ix;
  unsigned int seed=123456;
  int sid, nloop[HPE_MAX_ORDER];
  double *disc  = (double*)NULL;
  double *work  = (double*)NULL;
  double *sp1, *sp2, *sp3, spinor1[24], spinor2[24];
  int verbose = 0;
  int do_gt   = 0;
  int deg;
  char filename[100], contype[200];
  double ratime, retime;
  double plaq;
  double *gauge_trafo=(double*)NULL, U_[18];
  complex w;
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

  if(hpe_order_min==-1 && hpe_order_max==-1) {hpe_order_min=0; hpe_order_max=0;}
  else if(hpe_order_min==-1 && hpe_order_max>=0) {hpe_order_min=3;}
  else if(hpe_order_min>=0 && hpe_order_max==-1) {hpe_order_max=hpe_order_min;}
  if(hpe_order_min%2==0 && hpe_order_min>0) {
    hpe_order_min--;
    fprintf(stdout, "Attention: HPE min order reset to %d\n", hpe_order_min);
  }
  if(hpe_order_max%2==0 && hpe_order_max>0) {
    hpe_order_max--;
    fprintf(stdout, "Attention: HPE max order reset to %d\n", hpe_order_max);
  }

  fprintf(stdout, "\n**************************************************\n");
  fprintf(stdout, "* vp_disc_hpe_mc with HPE of order %d to %d\n", hpe_order_min, hpe_order_max);
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
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

  if(do_gt==1) {
    /***********************************
     * initialize gauge transformation
     ***********************************/
    init_gauge_trafo(&gauge_trafo, 1.);
    apply_gt_gauge(gauge_trafo);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value after gauge trafo: %25.16e\n", plaq);
  }

  /*********************************************
   * allocate memory for the spinor fields
   *********************************************/
  no_fields = 3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);


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

  work  = (double*)calloc(8*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

  /************************************************
   * loop on sources
   ************************************************/
  count = 0;
  if(g_cart_id==0) fprintf(stdout, "# Using seed %u", g_seed);
  srand(g_seed);
  for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {

    /******************************************************************************************
     * initialize the HPE calculation (xchange for working field in BHn)
     ******************************************************************************************/
    count++;

/*
    ranz2(g_spinor_field[0], 24*VOLUME); 
    xchange_field(g_spinor_field[0]);
*/

   
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_zero(g_spinor_field[0]+_GSI(ix));
    }
/*
    mu = sid/12;
    ix = sid%12;
    fprintf(stdout, "mu = %d; ix=%d\n", mu, ix);
    if(mu<4) {
      g_spinor_field[0][_GSI(g_iup[0][mu])+2*mu] = 1.;
    } else {
      g_spinor_field[0][_GSI(0)+2*mu] = 1.;
    }
    fprintf(stdout, "set component %d to 1.\n", 2*sid);
*/
    g_spinor_field[0][2*sid] = 1.;

 
    memcpy((void*)g_spinor_field[2], (void*)g_spinor_field[0], 24*VOLUME*sizeof(double));

#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    mul_one_pm_imu_inv(g_spinor_field[0], +1., VOLUME);
    BHn(g_spinor_field[1], g_spinor_field[0], hpe_order_min);
    sp2 = g_spinor_field[1]; sp1 = g_spinor_field[0];
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]));
        _fv_eq_cm_ti_fv(spinor1, U_, sp2+_GSI(g_iup[ix][mu]));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_mi_eq_fv(spinor1, spinor2);
        _co_eq_fv_dag_ti_fv(&w, g_spinor_field[2]+_GSI(ix), spinor1);
        disc[_GWI(mu,ix,VOLUME)+1] -= 2. * g_kappa * w.im;
      }
    }
 
    for(deg=hpe_order_min+2; deg<=hpe_order_max; deg+=2) {
      BHn(sp1, sp2, 1);
      BHn(sp2, sp1, 1);
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]));
          _fv_eq_cm_ti_fv(spinor1, U_, sp2+_GSI(g_iup[ix][mu]));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_mi_eq_fv(spinor1, spinor2);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[2]+_GSI(ix), spinor1);
          disc[_GWI(mu,ix,VOLUME)+1] -= 2. * g_kappa * w.im;
        }
      }

    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to calculate contribution for sid %d: %e seconds\n", sid, retime-ratime);

    if(count%Nsave==0) {
#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          work[_GWI(mu,ix,VOLUME)  ] = 0.;
          work[_GWI(mu,ix,VOLUME)+1] = disc[_GWI(mu,ix,VOLUME)+1] / (double)count; 
        }
      }
      sprintf(filename, "vp_disc_hpe%.2d_mc.%.4d.%.4d", hpe_order_max, Nconf, count);
      sprintf(contype, "cvc-disc-hpe-loops-%.2d_to.%.2d-se", hpe_order_min, hpe_order_max);
      write_lime_contraction (work, filename, 64, 4, contype, Nconf, count);

/*
      sprintf(filename, "vp_disc_hpe%.2d_mcascii.%.4d.%.4d", hpe_order_max, Nconf, count);
      ofs = fopen(filename, "w");
      for(ix=0; ix<VOLUME; ix++) {
        for(mu=0; mu<4; mu++) {
          fprintf(ofs, "%6d%3d%25.16e%25.16e\n", ix, mu, work[_GWI(mu,ix,VOLUME)], work[_GWI(mu,ix,VOLUME)+1]);
        }
      }
      fclose(ofs);
*/     


#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "# time to save results for count=%d: %e seconds\n", count, retime-ratime);
    }
  }

  /******************************************************************************************
   * END OF HPE CALCULATION
   ******************************************************************************************/




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
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free(disc);
  free(work);
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
