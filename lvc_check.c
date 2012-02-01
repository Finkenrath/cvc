/****************************************************
 * lvc_check.c
 *
 * Mon Sep 21 09:57:05 CEST 2009
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
  int sid;
  double *disc       = (double*)NULL;
  double *disc2      = (double*)NULL;
  double *work       = (double*)NULL;
  double *disc_diag  = (double*)NULL;
  double *disc_diag2 = (double*)NULL;
  double q[4], fnorm;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24];
  complex w, w1, *cp1, *cp2, *cp3;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

  fftwnd_plan plan_p, plan_m;

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
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");

  /***********************
   * read the input file *
   ***********************/
  read_input(filename);

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

  dims[0] = T_global;
  dims[1] = LX;
  dims[2] = LY;
  dims[3] = LZ;
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD,  FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

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
  disc2  = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc2 == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc2\n");
    exit(3);
  }
  for(ix=0; ix<8*VOLUME; ix++) disc2[ix] = 0.;

  disc = (double*)calloc( 8*VOLUME, sizeof(double));
  if( disc == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for disc\n");
    exit(3);
  }
  for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;

  work  = (double*)calloc(48*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
    exit(3);
  }

  if(g_subtract == 1) {
    /* allocate memory for disc_diag */
    disc_diag  = (double*)calloc(32*VOLUME, sizeof(double));
    if( disc_diag==(double*)NULL ) {
      fprintf(stderr, "could not allocate memory for disc_diag\n");
      exit(8);
    }
    for(ix=0; ix<32*VOLUME; ix++) disc_diag[ix] = 0.;

    disc_diag2 = (double*)calloc(32*VOLUME, sizeof(double));
    if( disc_diag2==(double*)NULL ) {
      fprintf(stderr, "could not allocate memory for disc_diag2\n");
      exit(8);
    }
    for(ix=0; ix<32*VOLUME; ix++) disc_diag2[ix] = 0.;
  }

  /*****************************************
   * prepare Fourier transformation arrays *
   *****************************************/
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
    exit(4);
  }

  /*****************************
   * start loop on source id.s *
   *****************************/
  for(sid=g_sourceid; sid<=g_sourceid2; sid++) {

    /* read the new propagator */
    ratime = (double)clock() / CLOCKS_PER_SEC;
    if(format==0) {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
      /* sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, sid); */
      if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) break;
    }
    else if(format==1) {
      sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid);
      if(read_cmi(g_spinor_field[1], filename) != 0) break;
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "time to read prop.: %e seconds\n", retime-ratime);

    count++;

    /* calculate the source: apply Q_phi_tbc */
    ratime = (double)clock() / CLOCKS_PER_SEC;
    Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);
    retime = (double)clock() / CLOCKS_PER_SEC;
    if(g_cart_id==0) fprintf(stdout, "time to calculate source: %e seconds\n", retime-ratime);

    /* contractions for lvc */
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[1][_GSI(ix)]);
        _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1);
        disc2[_GWI(mu,ix,VOLUME)  ] -= w.re;
        disc2[_GWI(mu,ix,VOLUME)+1] -= w.im;
        if(g_subtract==1) {
          work[_GWI(mu,ix,VOLUME)  ] -= w.re;
          work[_GWI(mu,ix,VOLUME)+1] -= w.im;
        }
      }
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "[%2d] contractions in %e seconds\n", g_cart_id, retime-ratime);

    if(g_subtract==1) {
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
        fftwnd_one(plan_m, in, NULL);
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
        
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
        fftwnd_one(plan_p, in, NULL);
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _co_eq_co_ti_co((complex*)(disc_diag2+_GWI(4*mu+nu,ix,VOLUME)), (complex*)(work+_GWI(mu,ix,VOLUME)), (complex*)(work+_GWI(4+nu,ix,VOLUME)));
        }
      }
      }
    }

    /* contractions for lavc */
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_gamma_ti_fv(spinor1, 6+mu, &g_spinor_field[1][_GSI(ix)]);
        _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1);
        disc[_GWI(mu,ix,VOLUME)  ] -= w.re;
        disc[_GWI(mu,ix,VOLUME)+1] -= w.im;
        if(g_subtract==1) {
          work[_GWI(mu,ix,VOLUME)  ] -= w.re;
          work[_GWI(mu,ix,VOLUME)+1] -= w.im;
        }
      }
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "[%2d] contractions in %e seconds\n", g_cart_id, retime-ratime);

    if(g_subtract==1) {
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
        fftwnd_one(plan_m, in, NULL);
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
        
        memcpy((void*)in, (void*)(work+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
        fftwnd_one(plan_p, in, NULL);
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _co_eq_co_ti_co((complex*)(disc_diag+_GWI(4*mu+nu,ix,VOLUME)), (complex*)(work+_GWI(mu,ix,VOLUME)), (complex*)(work+_GWI(4+nu,ix,VOLUME)));
        }
      }
      }
    }


    /* save results for count = multiple of Nsave */
    if(count%Nsave == 0) {

      if(g_cart_id == 0) fprintf(stdout, "save results for count = %d\n", count);

      /* save the result in position space */
      fprintf(stdout, "Save the result in position space\n");
      sprintf(filename, "lvc_check_X.%.4d.%.4d", Nconf, count);
      write_contraction(disc2, NULL, filename, 4, 2, 0);

      sprintf(filename, "lavc_check_X.%.4d.%.4d", Nconf, count);
      write_contraction(disc, NULL, filename, 4, 2, 0);

      fprintf(stdout, "Save the result in momentum space\n");
      ratime = (double)clock() / CLOCKS_PER_SEC;
      /* Fourier transform data, copy to work */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(disc2+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
        fftwnd_one(plan_m, in, NULL);
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
        
        memcpy((void*)in, (void*)(disc2+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
        fftwnd_one(plan_p, in, NULL);
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }
 
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _co_eq_co_ti_co((complex*)(work+_GWI(8+4*mu+nu,ix,VOLUME)), (complex*)(work+_GWI(mu,ix,VOLUME)),(complex*)(work+_GWI(4+nu,ix,VOLUME)));
        }
      }
      }
      sprintf(filename, "lvc_check_P.%.4d.%.4d", Nconf, sid);
      write_contraction(work+_GWI(8,0,VOLUME), NULL, filename, 16, 2, 0);
      retime = (double)clock() / CLOCKS_PER_SEC;
      fprintf(stdout, "[%2d] contractions in %e seconds\n", g_cart_id, retime-ratime);
       
      ratime = (double)clock() / CLOCKS_PER_SEC;
      /* Fourier transform data, copy to work */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
        fftwnd_one(plan_m, in, NULL);
        memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
        
        memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
        fftwnd_one(plan_p, in, NULL);
        memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
      }
 
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _co_eq_co_ti_co((complex*)(work+_GWI(8+4*mu+nu,ix,VOLUME)), (complex*)(work+_GWI(mu,ix,VOLUME)),(complex*)(work+_GWI(4+nu,ix,VOLUME)));
        }
      }
      }
      sprintf(filename, "lavc_check_P.%.4d.%.4d", Nconf, sid);
      write_contraction(work+_GWI(8,0,VOLUME), NULL, filename, 16, 2, 0);
      retime = (double)clock() / CLOCKS_PER_SEC;
      fprintf(stdout, "[%2d] contractions in %e seconds\n", g_cart_id, retime-ratime);
       

    }  /* of count % Nsave == 0 */

  }  /* of loop on sid */

  /* free the allocated memory, finalize */
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(disc2);
  if(g_subtract==1) free(disc_diag2);

  free(disc);
  if(g_subtract==1) free(disc_diag);

  free(work);
  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);

  return(0);

}
