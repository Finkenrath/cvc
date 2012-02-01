/****************************************************
 * hl_conn_5.c
 *
 * Fri Apr 16 15:26:00 CET 2010
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#  include <fftw_mpi.h>
#else
#  include <fftw.h>
#endif
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
#include "fuzz.h"
#include "fuzz2.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"

void usage() {
  fprintf(stdout, "\n\nCode to perform contractions for connected contributions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -h, -? this help and exit\n");
  fprintf(stdout, "         -v verbose [no effect, lots of stdout output anyway]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -l lightest mass _NOT_ in mms file [default lightest is mms]\n");
  fprintf(stdout, "         -L other light mass prop. (if any) are _NOT_ mms files [default other lights are mms]\n");
  fprintf(stdout, "         -H heavy mass prop. (if any) are _NOT_ mms files [default heavy masses are mms]\n");
  fprintf(stdout, "         -m number of first heavy mass to be _USED_ [default max( 1 , g_no_light_masses)]\n");
  fprintf(stdout, "         -p <n> number of colours [default 1]\n\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}
/*
#ifdef _UNDEF
#undef _UNDEF
#endif
*/
#ifndef _UNDEF
#define _UNDEF
#endif


int main(int argc, char **argv) {
  
  int c, i, j, k, k2, ll, sl, t, j1;
  int count;
  long unsigned int VOL3;
  int filename_set = 0;
  int mms1=0, mms2=0, mms2_min=1;
  int light_mms=1;
  int heavy_mms=1;
  int other_light_mms=1;
  int x0, x1, x2, ix, idx;
  int n_c=1, n_s=4;
  int K=32;
  int *xgindex1=NULL, *xgindex2=NULL, *xisimag=NULL;
  int non_mms_one_file = 0;
  double *xvsign=NULL;
  double *cconn  = (double*)NULL;
  double *buffer = (double*)NULL, *buffer2 = (double*)NULL;
  double *work=NULL;
  int sigmalight=0, sigmaheavy=0;
  int index_min=0, index_smeared_start=0;
  double *mms_masses=NULL;
  double correlator_norm = 1.;
  char *mms_extra_masses_file="cvc.extra_masses.input";

  int src0, src1, src2, src3;
 
  int verbose = 0;
  char filename[200];
  double ratime, retime;
  double plaq;
  double *gauge_field_timeslice=NULL, *gauge_field_f=NULL;
  double **chi=NULL, **psi=NULL;
  double *Ctmp=NULL;
  FILE *ofs=NULL;
  double c_conf_gamma_sign[]  = {1., 1., 1., -1., -1., -1., -1., 1., 1., 1., -1., -1.,  1.,  1., 1., 1.};
  double n_conf_gamma_sign[]  = {1., 1., 1., -1., -1., -1., -1., 1., 1., 1.,  1.,  1., -1., -1., 1., 1.};
  double *conf_gamma_sign=NULL;

  /**************************************************************************************************
   * charged stuff
   *
   * (pseudo-)scalar:
   * g5 - g5,	g5   - g0g5,	g0g5 - g5,	g0g5 - g0g5,
   * g0 - g0,	g5   - g0,	g0   - g5,	g0g5 - g0,
   * g0 - g0g5,	1    - 1,	1    - g5,	g5   - 1,
   * 1  - g0g5,	g0g5 - 1,	1    - g0,	g0   - 1
   *
   * (pseudo-)vector:
   * gig0 - gig0,	gi     - gi,		gig5 - gig5,	gig0   - gi,
   * gi   - gig0,	gig0   - gig5,		gig5 - gig0,	gi     - gig5,
   * gig5 - gi,		gig0g5 - gig0g5,	gig0 - gig0g5,	gig0g5 - gig0,
   * gi   - gig0g5,	gig0g5 - gi,		gig5 - gig0g5,	gig0g5 - gig5
   **************************************************************************************************/
  int gindex1[] = {5, 5, 6, 6, 0, 5, 0, 6, 0, 4, 4, 5, 4, 6, 4, 0,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 10, 11, 12, 1, 2, 3, 10, 11, 12, 7, 8, 9, 1, 2, 3, 7, 8, 9,
                   13, 14, 15, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13, 7, 8, 9, 15, 14, 13};

  int gindex2[] = {5, 6, 5, 6, 0, 0, 5, 0, 6, 4, 5, 4, 6, 4, 0, 4,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 1, 2, 3, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 1, 2, 3,
                   13, 14, 15, 15, 14, 13, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13, 7, 8, 9};

  /* due to twisting we have several correlators that are purely imaginary */
  int isimag[]  = {0, 0, 0, 0, 
                   0, 1, 1, 1, 
                   1, 0, 1, 1, 
                   1, 1, 0, 0,

                   0, 0, 0, 0, 
                   0, 1, 1, 1, 
                   1, 0, 1, 1, 
                   1, 1, 0, 0};

  double vsign[]  = {1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1., -1., 1.,   1., -1., 1., 
                     1., -1., 1.,   1., -1., 1.,   1., -1., 1.,   1., -1., 1.};


  /**************************************************************************************************
   * neutral stuff 
   *
   * (pseudo-)scalar:
   * g5 - g5,	g5   - g0g5,	g0g5 - g5,	g0g5 - g0g5,
   * 1  - 1,	g5   - 1,	1    - g5,	g0g5 - 1,
   * 1  - g0g5,	g0   - g0,	g0   - g5,	g5   - g0,
   * g0 - g0g5,	g0g5 - g0,	g0   - 1,	1    - g0
   *
   * (pseudo-)vector:
   * gig0   - gig0,		gi   - gi,	gig0g5 - gig0g5,	gig0   - gi, 
   * gi     - gig0,		gig0 - gig0g5,	gig0g5 - gig0,		gi     - gig0g5,
   * gig0g5 - gi		gig5 - gig5,	gig5   - gi,		gi     - gig5,
   * gig5   - gig0,		gig0 - gig5,	gig5   - gig0g5,	gig0g5 - gig5
   **************************************************************************************************/
  int ngindex1[] = {5, 5, 6, 6, 4, 5, 4, 6, 4, 0, 0, 5, 0, 6, 0, 4,
                    10, 11, 12, 1, 2, 3, 13, 14, 15, 10, 11, 12,  1,  2,  3, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13,
                     7,  8,  9, 7, 8, 9,  1,  2,  3,  7,  8,  9, 10, 11, 12,  7,  8,  9, 15, 14, 13};

  int ngindex2[] = {5, 6, 5, 6, 4, 4, 5, 4, 6, 0, 5, 0, 6, 0, 4, 0,
                    10, 11, 12, 1, 2, 3, 13, 14, 15,  1,  2,  3, 10, 11, 12, 15, 14, 13, 10, 11, 12, 15, 14, 13, 1, 2, 3,
                     7,  8,  9, 1, 2, 3,  7,  8,  9, 10, 11, 12,  7,  8,  9, 15, 14, 13,  7,  8, 9};

  int nisimag[]  = {0, 0, 0, 0,
                    0, 1, 1, 1,
                    1, 0, 1, 1,
                    1, 1, 0, 0,

                    0, 0, 0, 0,
                    0, 1, 1, 1, 
                    1, 0, 1, 1,
                    1, 1, 0, 0};

  double nvsign[] = {1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1., 
                     1.,  1., 1.,   1., -1., 1.,   1., -1., 1.,   1., -1., 1.,
                     1., -1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1., -1., 1.,   1., -1., 1. };
 

/*
  double isneg_std[]=    {+1., -1., +1., -1., +1., +1., +1., +1., -1., +1., +1., +1., +1., +1., +1., +1.,    
                          -1., +1., -1., -1., +1., +1., +1., -1., +1., -1., +1., +1., +1., +1., +1., +1.}; 
*/
  double isneg_std[]=    {+1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1.,    
                          +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1.};

  double isneg[32];


#ifdef MPI
  MPI_Status status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "oh?vlHLf:p:m:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'p':
      n_c = atoi(optarg);
      fprintf(stdout, "\n# [hl_conn_5] using number of colors = %d\n", n_c);
      break;
    case 'm':
      mms2_min = atoi(optarg);
      break;
    case 'l':
      light_mms = 0;
      fprintf(stdout, "\n# [hl_conn_5] light fermion propagators _NOT_ in mms file format\n");
      break;
    case 'H':
      heavy_mms = 0;
      break;
    case 'L':
      other_light_mms = 0;
      break;
    case 'o':
      non_mms_one_file = 1;
      fprintf(stdout, "\n# [hl_conn_5] reading non-mms fermion propagators from same file at different positions\n");
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
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  if( Nlong > 0 ) {
    if(g_proc_id==0) fprintf(stdout, "Fuzzing not available in this version.\n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  VOL3 = LX*LY*LZ;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",
		  g_cart_id, g_cart_id, T_global,  g_cart_id, T,  g_cart_id, Tstart, 
                             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
                             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  /*********************************************
   * set the isneg field
   *********************************************/
  for(i = 0; i < K; i++) isneg[i] = isneg_std[i];

  /*********************************************
   * read the gauge field 
   *********************************************/
  if(light_mms || other_light_mms || heavy_mms || (N_Jacobi > 0) ) {

    alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(strcmp(gaugefilename_prefix,"identity")==0) {
      if(g_cart_id==0) fprintf(stdout, "\n# [hl_conn_5] initializing unit matrices\n");
      for(ix=0;ix<VOLUME;ix++) {
        _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
      }
    } else {
      read_lime_gauge_field_doubleprec(filename);
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for reading gauge field: %e seconds\n", retime-ratime);

#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    xchange_gauge();
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for exchanging gauge field: %e seconds\n", retime-ratime);

    /* measure the plaquette */
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

  } else {
    g_gauge_field = (double*)NULL;
  }


  if( (N_Jacobi>0) || (Nlong>0) ) {

    if(g_cart_id==0) fprintf(stdout, "# apply APE smearing of gauge field with parameters:\n"\
                                     "# N_ape = %d\n# alpha_ape = %f\n", N_ape, alpha_ape);
    
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

#if !( (defined PARALLELTX) || (defined PARALLELTXY) )
    alloc_gauge_field(&gauge_field_f, VOLUME);
    if( (gauge_field_timeslice = (double*)malloc(72*VOL3*sizeof(double))) == (double*)NULL  ) {
      fprintf(stderr, "Error, could not allocate mem for gauge_field_timeslice\n");
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 3);
      MPI_Finalize();
#endif
      exit(2);
    }
    for(x0=0; x0<T; x0++) {
      memcpy((void*)gauge_field_timeslice, (void*)(g_gauge_field+_GGI(g_ipt[x0][0][0][0],0)), 72*VOL3*sizeof(double));
      for(i=0; i<N_ape; i++) {
        APE_Smearing_Step_Timeslice(gauge_field_timeslice, alpha_ape);
      }
      if(Nlong > -1) {
        fuzzed_links_Timeslice(gauge_field_f, gauge_field_timeslice, Nlong, x0);
      } else {
        memcpy((void*)(gauge_field_f+_GGI(g_ipt[x0][0][0][0],0)), (void*)gauge_field_timeslice, 72*VOL3*sizeof(double));
      }
    }
    free(gauge_field_timeslice);
#else
    for(i=0; i<N_ape; i++) {
      APE_Smearing_Step(g_gauge_field, alpha_ape);
      xchange_gauge_field_timeslice(g_gauge_field);
    }

    alloc_gauge_field(&gauge_field_f, VOLUMEPLUSRAND);

    if(Nlong > 0) {
      fuzzed_links2(gauge_field_f, g_gauge_field, Nlong);
    } else {
      memcpy((void*)gauge_field_f, (void*)g_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));
    }
    xchange_gauge_field(gauge_field_f);

    if(strcmp(gaugefilename_prefix,"identity")==0) {
      if(g_cart_id==0) fprintf(stdout, "\n# [hl_conn_5] re-initializing unit matrices\n");
      for(ix=0;ix<VOLUME;ix++) {
        _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
      }
    } else {
      read_lime_gauge_field_doubleprec(filename);
    }
    xchange_gauge();
#endif

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for APE smearing gauge field: %e seconds\n", retime-ratime);

  }

  /*********************************************************
   * allocate memory for the spinor fields 
   *********************************************************/
  no_fields = 4*( g_no_light_masses + (g_no_extra_masses+1-g_no_light_masses>0) )*n_s*n_c + 1;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  if(g_cart_id==0) fprintf(stdout, "# no. of spinor fields is %d\n", no_fields);
#if !( (defined PARALLELTX) || (defined PARALLELTXY) )
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUME + RAND);
#else
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME + RAND);
#endif

  /* allocate memory for the contractions */
/*
#if (defined PARALLELTX) || (defined PARALLELTXY)
  idx = 8 * K * T_global;
#else
  if(g_cart_id==0) { idx = 8*K*T_global; } 
  else             { idx = 8*K*T; }
#endif
*/
  cconn = (double*)calloc(8*K*T, sizeof(double));
  if( cconn==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for cconn\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 4);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<8*K*T; ix++) cconn[ix] = 0.;

  buffer  = (double*)calloc(8*K*T,        sizeof(double));
  buffer2 = (double*)calloc(8*K*T_global, sizeof(double));
  if( buffer==(double*)NULL || buffer2==(double*)NULL) {
    fprintf(stderr, "could not allocate memory for buffers\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 5);
    MPI_Finalize();
#endif
    exit(4);
  }

  if( (Ctmp = (double*)calloc(2*T, sizeof(double))) == NULL ) {
    fprintf(stderr, "Error, could not allocate mem for Ctmp\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 6);
    MPI_Finalize();
#endif
    exit(5);
  }

  /*********************************************
   * read the extra masses for mms
   * - if mms is not used, use g_mu as the mass 
   *********************************************/
  if( (mms_masses = (double*)calloc(g_no_extra_masses+1, sizeof(double))) == NULL ) {
    fprintf(stderr, "Error, could allocate mms_masses\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 7);
    MPI_Finalize();
#endif
    exit(6);
  }
  mms_masses[0] = g_mu;
  if(g_no_extra_masses>0) {
    if( (ofs=fopen(mms_extra_masses_file, "r"))==NULL ) {
      fprintf(stderr, "Error, could not open file %s for reading\n", mms_extra_masses_file);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 8);
      MPI_Finalize();
#endif
      exit(7);
    }
    for(i=0; i<g_no_extra_masses; i++) fscanf(ofs, "%lf", mms_masses+i+1);
    fclose(ofs);
  }
  if(g_cart_id==0) {
    fprintf(stdout, "# mms masses:\n");
    for(i=0; i<=g_no_extra_masses; i++) fprintf(stdout, "# mass[%2d] = %e\n",
      i, mms_masses[i]);
  } else {
    if(g_cart_id==0) fprintf(stdout, "# no extra masses specified\n");
  }

  /******************************************************************
   * get the source coordinates 
   ******************************************************************/

  
  /******************************************************************
   * final normalization of the correlators
   ******************************************************************/
  correlator_norm = 1. / ( 2. * g_kappa * g_kappa * (double)(LX_global*LY_global*LZ) );
  if(g_cart_id==0) fprintf(stdout, "# correlator_norm = %12.5e\n", correlator_norm);

  /******************************************************************
   ******************************************************************
   **                                                              **
   **  local - local and local - smeared                           **
   **                                                              **
   ******************************************************************
   ******************************************************************/
if(g_local_local || g_local_smeared) {
  if(g_cart_id==0) fprintf(stdout, "# Starting LL and LS contractions\n");
  /****************************************************************
   * (1.0) the light masses
   ****************************************************************/
  work = g_spinor_field[no_fields-1];
  index_min = 0;
  index_smeared_start = 0;

  for(k=0; k<g_no_light_masses; k++) {
    mms1 = k;
    if(!light_mms) mms1--;
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(i=0; i<n_s*n_c; i++) {

      if(k==0) {
        prepare_propagator(g_source_timeslice, i+index_smeared_start, light_mms, mms1, -1., mms_masses[k], i+index_min, work, 0);
        if(light_mms) {
          Qf5(g_spinor_field[i+2*n_s*n_c+index_min], work, mms_masses[k]);
        } else {
          if(non_mms_one_file) {
	    prepare_propagator(g_source_timeslice, i+index_smeared_start, light_mms, mms1, -1., mms_masses[k], i+2*n_s*n_c+index_min, work, 1);
          } else {
	    prepare_propagator(g_source_timeslice, i+index_smeared_start, light_mms, mms1, 1., mms_masses[k], i+2*n_s*n_c+index_min, work, 0);
          }
        }
      } else {
        prepare_propagator(g_source_timeslice, i+index_smeared_start, other_light_mms, mms1, -1., mms_masses[k], i+index_min, work, 0);
        if(other_light_mms) {
          Qf5(g_spinor_field[i+2*n_s*n_c+index_min], work, mms_masses[k]);
        } else {
          if(non_mms_one_file) {
            prepare_propagator(g_source_timeslice, i+index_smeared_start, other_light_mms, mms1, -1., mms_masses[k], i+2*n_s*n_c+index_min, work, 1);
          } else {
            prepare_propagator(g_source_timeslice, i+index_smeared_start, other_light_mms, mms1, 1., mms_masses[k], i+2*n_s*n_c+index_min, work, 0);
          }
        }
      }
      if(N_Jacobi>0) {
        memcpy((void*)g_spinor_field[i+n_s*n_c+index_min], (void*)g_spinor_field[i+index_min], 24*VOLUME*sizeof(double));
        xchange_field_timeslice(g_spinor_field[i+n_s*n_c+index_min]);
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i+n_s*n_c+index_min], work, kappa_Jacobi);
          xchange_field_timeslice(g_spinor_field[i+n_s*n_c+index_min]);
        }
        memcpy((void*)g_spinor_field[i+3*n_s*n_c+index_min], (void*)g_spinor_field[i+2*n_s*n_c+index_min], 24*VOLUME*sizeof(double));
        xchange_field_timeslice(g_spinor_field[i+3*n_s*n_c+index_min]);
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i+3*n_s*n_c+index_min], work, kappa_Jacobi);
          xchange_field_timeslice(g_spinor_field[i+3*n_s*n_c+index_min]);
        }
      }
    }  /* of i=0, ... , n_s*n_c */
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for preparing light prop.: %e seconds\n", retime-ratime);
  
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

    for(mms2=k; mms2>=0; mms2--) {
      count=-1;
      for(sigmalight=-1; sigmalight<=1; sigmalight+=2) {
      for(sigmaheavy=-1; sigmaheavy<=1; sigmaheavy+=2) {
        count++;
        for(idx=0; idx<8*K*T; idx++) cconn[idx] = 0.;
        for(j=0; j<2; j++) {
          ll = j;
          chi = &g_spinor_field[( sigmalight+1 + 4*mms2 + j )*n_s*n_c];
          psi = &g_spinor_field[( sigmaheavy+1 + 4*k    + j )*n_s*n_c];
          if(sigmalight == sigmaheavy) {
            xgindex1 = gindex1;  xgindex2 = gindex2;  xisimag=isimag;  xvsign=vsign;  conf_gamma_sign = c_conf_gamma_sign;
          } else {
            xgindex1 = ngindex1; xgindex2 = ngindex2; xisimag=nisimag; xvsign=nvsign; conf_gamma_sign = n_conf_gamma_sign;
          }

          sl = 2*ll*K*T;

          /* (pseudo-)scalar sector */
          for(idx=0; idx<16; idx++) {
            contract_twopoint(&cconn[sl], xgindex1[idx], xgindex2[idx], chi, psi, n_c);
            sl += (2*T);
          }
          /* (pseudo-)vector sector */
          for(idx = 16; idx < 64; idx+=3) {
            for(i = 0; i < 3; i++) {
              for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
              contract_twopoint(Ctmp, xgindex1[idx+i], xgindex2[idx+i], chi, psi, n_c);
              for(x0=0; x0<T; x0++) {
                cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0  ]);
                cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0+1]);
              }
            }
            sl += (2*T);
          }
        }

#ifdef MPI
#if (defined PARALLELTX) || (defined PARALLELTXY)
/*        if(g_xs_id==0) fprintf(stdout, "# [%2d] collecting results\n", g_cart_id); */
        for(ix=0; ix<8*K*T; ix++) buffer[ix] = 0.;
        for(ix=0; ix<8*K*T_global; ix++) buffer2[ix] = 0.;
        MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
        MPI_Allgather(buffer, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, g_xs_comm);
/*
        memcpy((void*)cconn, (void*)buffer2, 8*K*T_global);
        MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
        MPI_Allgather(buffer, 8*K*T, MPI_DOUBLE, cconn, 8*K*T, MPI_DOUBLE, 0, g_xs_comm);
*/
#else
        MPI_Gather(cconn, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, 0, g_cart_grid);
/*        if(g_cart_id==0) memcpy((void*)cconn, (void*)buffer, 8*K*T_global*sizeof(double)); */
#endif
#else
        memcpy((void*)buffer2, (void*)cconn, 8*K*T_global*sizeof(double));
#endif
        if(g_cart_id==0) {
          sprintf(filename, "correl.%.4d.%.2d.%.2d.%.2d", Nconf, g_source_timeslice, mms2, k);
          if(count==0) {
            ofs=fopen(filename, "w");
          } else {
            ofs=fopen(filename, "a");
          }
          if( ofs == (FILE*)NULL ) {
            fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
            MPI_Abort(MPI_COMM_WORLD, 9);
            MPI_Finalize();
#endif
            exit(8);
          }
          for(idx=0; idx<8*K*T_global; idx++) buffer2[idx] *= correlator_norm;
  
          fprintf(ofs, "# %5d%3d%3d%3d%3d%15.8e%15.8e%15.8e%3d%3d\tLL und LS\n",
            Nconf, T_global, LX_global, LY_global, LZ, g_kappa, mms_masses[mms2], mms_masses[k], -sigmalight, -sigmaheavy);
          for(idx=0; idx<K; idx++) {
            for(ll=0; ll<2; ll++) {
              x1 = (0+g_source_timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0, isneg[idx]*buffer2[i], 0.);
              for(x0=1; x0<T_global/2; x0++) {
                x1 = ( x0+g_source_timeslice) % T_global;
                x2 = (-x0+g_source_timeslice+T_global) % T_global;
                i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
                j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + xisimag[idx];
                fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], isneg[idx]*buffer2[j]);
              }
              x0 = T_global/2;
              x1 = (x0+g_source_timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], 0.);
            }
          }
          fclose(ofs);
        }
      }}  /* sigmalight, sigmaheavy */

    }  /* loop on mms2 = k, ..., 0 */
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for light-light contractions: %e seconds\n", retime-ratime);

    index_min += 4*n_s*n_c;
  }

  
  /****************************************************************
   * (1.1) the heavy masses
   ****************************************************************/
  if(mms2_min<g_no_light_masses) mms2_min = g_no_light_masses;
  index_min = g_no_light_masses * 4*n_s*n_c;

  for(k=mms2_min; k<=g_no_extra_masses; k++) {
    mms2 = k;
    if(!light_mms) mms2--;

    /****************************************************************
     * (1.1.0) read the heavy-mass propagators
     ****************************************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(i=0; i<n_s*n_c; i++) {
      prepare_propagator(g_source_timeslice, i+index_smeared_start, heavy_mms, mms2, -1., mms_masses[k], i+index_min, work, 0);
      if(heavy_mms) {
        Qf5(g_spinor_field[i+2*n_s*n_c+index_min], work, mms_masses[k]);
      } else {
        if(non_mms_one_file) {
          prepare_propagator(g_source_timeslice, i+index_smeared_start, heavy_mms, mms2, -1., mms_masses[k], i+2*n_s*n_c+index_min, work, 1);
        } else {
          prepare_propagator(g_source_timeslice, i+index_smeared_start, heavy_mms, mms2, 1., mms_masses[k], i+2*n_s*n_c+index_min, work, 0);
        }
      }
      if(N_Jacobi>0) {
        memcpy((void*)g_spinor_field[i+n_s*n_c+index_min], (void*)g_spinor_field[i+index_min], 24*VOLUME*sizeof(double));
        xchange_field_timeslice(g_spinor_field[i+n_s*n_c+index_min]);
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i+n_s*n_c+index_min], work, kappa_Jacobi);
          xchange_field_timeslice(g_spinor_field[i+n_s*n_c+index_min]);
        }
        memcpy((void*)g_spinor_field[i+3*n_s*n_c+index_min], (void*)g_spinor_field[i+2*n_s*n_c+index_min], 24*VOLUME*sizeof(double));
        xchange_field_timeslice(g_spinor_field[i+3*n_s*n_c+index_min]);
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i+3*n_s*n_c+index_min], work, kappa_Jacobi);
          xchange_field_timeslice(g_spinor_field[i+3*n_s*n_c+index_min]);
        }
      }
    }  
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for preparing heavy prop.: %e seconds\n", retime-ratime);

  
    /****************************************************************
     * (1.1.1) heavy - heavy contractions (mass-diagonal)
     ****************************************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    count=-1; 
    for(sigmalight=-1; sigmalight<=1; sigmalight+=2) {
    for(sigmaheavy=-1; sigmaheavy<=1; sigmaheavy+=2) {
      count++;
      for(ix=0; ix<8*K*T; ix++) cconn[ix] = 0.;
      for(j=0; j<2; j++) {
        ll = j;
        chi  = &g_spinor_field[( sigmalight+1 + j)*n_s*n_c + index_min];
        psi  = &g_spinor_field[( sigmaheavy+1 + j)*n_s*n_c + index_min];
        if(sigmaheavy==sigmalight) {
          xgindex1 = gindex1;  xgindex2 = gindex2;  xisimag=isimag;  xvsign=vsign;  conf_gamma_sign = c_conf_gamma_sign;
        } else {
          xgindex1 = ngindex1; xgindex2 = ngindex2; xisimag=nisimag; xvsign=nvsign; conf_gamma_sign = n_conf_gamma_sign;
        }

        sl = 2*ll*K*T;

        /* (pseudo-)scalar sector */
        for(idx=0; idx<16; idx++) {
          contract_twopoint(&cconn[sl], xgindex1[idx], xgindex2[idx], chi, psi, n_c);
          sl += (2*T);
        }
        /* (pseudo-)vector sector */
        for(idx = 16; idx < 64; idx+=3) {
          for(i = 0; i < 3; i++) {
            for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
            contract_twopoint(Ctmp, xgindex1[idx+i], xgindex2[idx+i], chi, psi, n_c);
            for(x0=0; x0<T; x0++) {
              cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0  ]);
              cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0+1]);
            }
          }
          sl += (2*T);
        }
      }

#ifdef MPI
#if (defined PARALLELTX) || (defined PARALLELTXY)
      for(ix=0; ix<8*K*T; ix++) buffer[ix] = 0.;
      for(ix=0; ix<8*K*T_global; ix++) buffer2[ix] = 0.;
      MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
      MPI_Allgather(buffer, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, g_xs_comm);
/*
      MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
      MPI_Gather(buffer, 8*K*T, MPI_DOUBLE, cconn, 8*K*T, MPI_DOUBLE, 0, g_xs_comm);
*/
#else
      MPI_Gather(cconn, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, 0, g_cart_grid);
/*      if(g_cart_id==0) memcpy((void*)cconn, (void*)buffer, 8*K*T_global*sizeof(double)); */
#endif
#else
        memcpy((void*)buffer2, (void*)cconn, 8*K*T_global*sizeof(double));
#endif
      if(g_cart_id==0) {
        sprintf(filename, "correl.%.4d.%.2d.%.2d.%.2d", Nconf, g_source_timeslice, k, k);
        if(count==0) {
          ofs=fopen(filename, "w");
        } else {
          ofs=fopen(filename, "a");
        }
        if( ofs == (FILE*)NULL ) {
          fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
          MPI_Abort(MPI_COMM_WORLD, 9);
          MPI_Finalize();
#endif
          exit(8);
        }

        for(idx=0; idx<8*K*T_global; idx++) buffer2[idx] *= correlator_norm;

        fprintf(ofs, "# %5d%3d%3d%3d%3d%15.8e%15.8e%15.8e%3d%3d\tLL und LS\n",
          Nconf, T_global, LX_global, LY_global, LZ, g_kappa, mms_masses[k], mms_masses[k], -sigmalight, -sigmaheavy);
        for(idx=0; idx<K; idx++) {
          for(ll=0; ll<2; ll++) {
            x1 = (0+g_source_timeslice) % T_global;
            i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0, isneg[idx]*buffer2[i], 0.);
            for(x0=1; x0<T_global/2; x0++) {
              x1 = ( x0+g_source_timeslice) % T_global;
              x2 = (-x0+g_source_timeslice+T_global) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], isneg[idx]*buffer2[j]);
            }
            x0 = T_global/2;
            x1 = (x0+g_source_timeslice) % T_global;
            i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], 0.);
          }
        }
        fclose(ofs);
      }
    }}
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for heavy-heavy contractions: %e seconds\n", retime-ratime);

    /****************************************************************
     * (1.1.2) light - heavy contractions
     ****************************************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(mms1=0; mms1<g_no_light_masses; mms1++) {
      
      count=-1;
      for(sigmalight=-1; sigmalight<=1; sigmalight+=2) {
      for(sigmaheavy=-1; sigmaheavy<=1; sigmaheavy+=2) {
        count++;
        for(ix=0; ix<8*K*T; ix++) cconn[ix] = 0.;
        for(j=0; j<2; j++) {
          ll = j;
          chi = &g_spinor_field[( sigmalight+1 + j + mms1*4 ) * n_s*n_c];
          psi = &g_spinor_field[( sigmaheavy+1 + j ) * n_s*n_c + index_min];
          if(sigmalight == sigmaheavy) {
            xgindex1 = gindex1;  xgindex2 = gindex2;  xisimag=isimag;  xvsign=vsign;  conf_gamma_sign = c_conf_gamma_sign;
          } else {
            xgindex1 = ngindex1; xgindex2 = ngindex2; xisimag=nisimag; xvsign=nvsign; conf_gamma_sign = n_conf_gamma_sign;
          }

          sl = 2*ll*K*T;

          /* (pseudo-)scalar sector */
          for(idx=0; idx<16; idx++) {
            contract_twopoint(&cconn[sl], xgindex1[idx], xgindex2[idx], chi, psi, n_c);
            sl += (2*T);
          }
          /* (pseudo-)vector sector */
          for(idx = 16; idx < 64; idx+=3) {
            for(i = 0; i < 3; i++) {
              for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
              contract_twopoint(Ctmp, xgindex1[idx+i], xgindex2[idx+i], chi, psi, n_c);
              for(x0=0; x0<T; x0++) {
                cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0  ]);
                cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0+1]);
              }
            }
            sl += (2*T);
          }
        }
#ifdef MPI
#if (defined PARALLELTX) || (defined PARALLELTXY)
        for(ix=0; ix<8*K*T; ix++) buffer[ix] = 0.;
        for(ix=0; ix<8*K*T_global; ix++) buffer2[ix] = 0.;
        MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
        MPI_Allgather(buffer, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, g_xs_comm);
/*
        MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
        MPI_Gather(buffer, 8*K*T, MPI_DOUBLE, cconn, 8*K*T, MPI_DOUBLE, 0, g_xs_comm);
*/
#else
        MPI_Gather(cconn, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, 0, g_cart_grid);
/*        if(g_cart_id==0) memcpy((void*)cconn, (void*)buffer, 8*K*T_global*sizeof(double)); */
#endif
#else
        memcpy((void*)buffer2, (void*)cconn, 8*K*T_global*sizeof(double));
#endif
        if(g_cart_id==0) {
          sprintf(filename, "correl.%.4d.%.2d.%.2d.%.2d", Nconf, g_source_timeslice, mms1, k);
          if(count==0) {
            ofs=fopen(filename, "w");
          } else {
            ofs=fopen(filename, "a");
          }
          if( ofs == (FILE*)NULL ) {
            fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
            MPI_Abort(MPI_COMM_WORLD, 9);
            MPI_Finalize();
#endif
            exit(8);
          }

          for(idx=0; idx<8*K*T_global; idx++) buffer2[idx] *= correlator_norm;

          fprintf(ofs, "# %5d%3d%3d%3d%3d%15.8e%15.8e%15.8e%3d%3d\tLL und LS\n",
            Nconf, T_global, LX_global, LY_global, LZ, g_kappa, mms_masses[mms1], mms_masses[k], -sigmalight, -sigmaheavy);
          for(idx=0; idx<K; idx++) {
            for(ll=0; ll<2; ll++) {
              x1 = (0+g_source_timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0, isneg[idx]*buffer2[i], 0.);
              for(x0=1; x0<T_global/2; x0++) {
                x1 = ( x0+g_source_timeslice) % T_global;
                x2 = (-x0+g_source_timeslice+T_global) % T_global;
                i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
                j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + xisimag[idx];
                fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], isneg[idx]*buffer2[j]);
              }
              x0 = T_global/2;
              x1 = (x0+g_source_timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], 0.);
            }
          }
          fclose(ofs);
        }
      }}
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for light-heavy contractions: %e seconds\n", retime-ratime);

  }  /* loop over heavy extra masses */
  if(g_cart_id==0) fprintf(stdout, "# finished LL and LS contractions\n");
}

  /******************************************************************
   ******************************************************************
   **                                                              **
   **  smeared/fuzzed - local and smeared/fuzzed - smeared/fuzzed  **
   **                                                              **
   ******************************************************************
   ******************************************************************/

if(g_smeared_smeared || g_smeared_local) {
  if(g_cart_id==0) fprintf(stdout, "# Starting SL and SS contractions\n");
  /****************************************************************
   * (2.0) the light masses
   ****************************************************************/
  work = g_spinor_field[no_fields-1];
  index_min = 0;
  if(g_local_local || g_local_smeared) index_smeared_start = n_s*n_c;

  for(k=0; k<g_no_light_masses; k++) {
    mms1 = k;
    if(!light_mms) mms1--;

#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(i=0; i<n_s*n_c; i++) {

      if(k==0) {
        prepare_propagator(g_source_timeslice, i+index_smeared_start, light_mms, mms1, -1., mms_masses[k], i+index_min, work, 0);
        if(light_mms) {
          Qf5(g_spinor_field[i+2*n_s*n_c+index_min], work, mms_masses[k]);
        } else {
          if(non_mms_one_file) {
            prepare_propagator(g_source_timeslice, i+index_smeared_start, light_mms, mms1, -1., mms_masses[k], i+2*n_s*n_c+index_min, work, 1);
          } else {
            prepare_propagator(g_source_timeslice, i+index_smeared_start, light_mms, mms1, 1., mms_masses[k], i+2*n_s*n_c+index_min, work, 0);
          }
        }
      } else {
        prepare_propagator(g_source_timeslice, i+index_smeared_start, other_light_mms, mms1, -1., mms_masses[k], i+index_min, work, 0);
        if(other_light_mms) {
          Qf5(g_spinor_field[i+2*n_s*n_c+index_min], work, mms_masses[k]);
        } else {
          if(non_mms_one_file) {
            prepare_propagator(g_source_timeslice, i+index_smeared_start, other_light_mms, mms1, -1., mms_masses[k], i+2*n_s*n_c+index_min, work, 1);
          } else {
            prepare_propagator(g_source_timeslice, i+index_smeared_start, other_light_mms, mms1, 1., mms_masses[k], i+2*n_s*n_c+index_min, work, 0);
          }
        }
      }
      if(N_Jacobi>0) {
        memcpy((void*)g_spinor_field[i+n_s*n_c+index_min], (void*)g_spinor_field[i+index_min], 24*VOLUME*sizeof(double));
        xchange_field_timeslice(g_spinor_field[i+n_s*n_c+index_min]);
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i+n_s*n_c+index_min], work, kappa_Jacobi);
          xchange_field_timeslice(g_spinor_field[i+n_s*n_c+index_min]);
        }
        memcpy((void*)g_spinor_field[i+3*n_s*n_c+index_min], (void*)g_spinor_field[i+2*n_s*n_c+index_min], 24*VOLUME*sizeof(double));
        xchange_field_timeslice(g_spinor_field[i+3*n_s*n_c+index_min]);
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i+3*n_s*n_c+index_min], work, kappa_Jacobi);
          xchange_field_timeslice(g_spinor_field[i+3*n_s*n_c+index_min]);
        }
      }
    }  /* of i=0, ... , n_s*n_c */
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for preparing prop.: %e seconds\n", retime-ratime);
  
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(mms2=k; mms2>=0; mms2--) {
      count=-1; 
      for(sigmalight=-1; sigmalight<=1; sigmalight+=2) {
      for(sigmaheavy=-1; sigmaheavy<=1; sigmaheavy+=2) {
        count++;
        for(idx=0; idx<8*K*T; idx++) cconn[idx] = 0.;
        for(j=0; j<2; j++) {
          ll = j+2;
/*          if(g_cart_id==0) fprintf(stdout, "# chi[%2d] and psi[%2d]\n", ( sigmalight+1 + j + 4*mms2 )*n_s*n_c, ( sigmaheavy+1 + j + 4*k )*n_s*n_c); */
          chi = &(g_spinor_field[( sigmalight+1 + j + 4*mms2 )*n_s*n_c]);
          psi = &(g_spinor_field[( sigmaheavy+1 + j + 4*k    )*n_s*n_c]);
          if(sigmalight==sigmaheavy) {
            xgindex1 = gindex1;  xgindex2 = gindex2;  xisimag=isimag;  xvsign=vsign;  conf_gamma_sign = c_conf_gamma_sign;
          } else {
            xgindex1 = ngindex1; xgindex2 = ngindex2; xisimag=nisimag; xvsign=nvsign; conf_gamma_sign = n_conf_gamma_sign;
          }

          sl = 2*ll*K*T;

          /* (pseudo-)scalar sector */
          for(idx=0; idx<16; idx++) {
/*            if(g_cart_id==0) fprintf(stdout, "# xgindex=(%2d, %2d)\n", xgindex1[idx], xgindex2[idx]); */
            contract_twopoint(&cconn[sl], xgindex1[idx], xgindex2[idx], chi, psi, n_c);
/*            for(t=0; t<T; t++) fprintf(stdout, "[%2d] %3d%3d%3d%25.16e%25.16e\n", g_cart_id, j, idx, t, cconn[sl+2*t], cconn[sl+2*t+1]); */
            sl += (2*T);
          }
          /* (pseudo-)vector sector */
          for(idx = 16; idx < 64; idx+=3) {
            for(i = 0; i < 3; i++) {
/*              if(g_cart_id==0) fprintf(stdout, "# xgindex=(%2d, %2d)\n", xgindex1[idx+i], xgindex2[idx+i]); */
              for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
              contract_twopoint(Ctmp, xgindex1[idx+i], xgindex2[idx+i], chi, psi, n_c);
              for(x0=0; x0<T; x0++) {
                cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0  ]);
                cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0+1]);
/*                for(t=0; t<T; t++) fprintf(stdout, "[%2d] %3d%3d%3d%25.16e%25.16e\n", g_cart_id, j, idx+i, t, cconn[sl+2*t], cconn[sl+2*t+1]); */
              }
            }
            sl += (2*T);
          }
        }

#ifdef MPI
#if (defined PARALLELTX) || (defined PARALLELTXY)
/*
        for(ix=0; ix<32; ix++) {
          for(j1=0; j1<2; j1++) {
            for(t=0; t<T; t++) {
              fprintf(stdout, "[%2d,%2d,%2d-before] %3d%3d%3d%25.16e%25.16e\n", 
                g_cart_id, g_ts_id, g_xs_id, 
                ix, j1+2, t, cconn[2*(K*T*(j1+2) +ix*T+t)], cconn[2*(K*T*(j1+2) + ix*T+t)+1]);
            }
          }
        }
        if(g_xs_id==0) fprintf(stdout, "# [%2d] collecting results\n", g_cart_id);
*/

        for(ix=0; ix<8*K*T; ix++) buffer[ix] = 0.;
        for(ix=0; ix<8*K*T_global; ix++) buffer2[ix] = 0.;
        MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
        MPI_Allgather(buffer, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, g_xs_comm);
/*
        memcpy((void*)cconn, (void*)buffer2, 8*K*T_global);
        MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
        MPI_Gather(buffer, 8*K*T, MPI_DOUBLE, cconn, 8*K*T, MPI_DOUBLE, 0, g_xs_comm);
*/
/*
        for(c=0; c<g_nproc_t; c++) {
          for(ix=0; ix<32; ix++) {
            for(j1=0; j1<2; j1++) {
              for(t=0; t<T; t++) {
                fprintf(stdout, "[%2d,%2d,%2d-after] %3d%3d%3d%25.16e%25.16e\n", 
                  g_cart_id, g_ts_id, g_xs_id, 
                  ix, j1+2, t, cconn[2*(c*4*K*T + K*T*(j1+2) + ix*T + t)  ], 
                               cconn[2*(c*4*K*T + K*T*(j1+2) + ix*T + t)+1]);
              }
            }
          }
        } 
*/         
#else
        MPI_Gather(cconn, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, 0, g_cart_grid);
/*        if(g_cart_id==0) memcpy((void*)cconn, (void*)buffer, 8*K*T_global*sizeof(double)); */
#endif
#else
        memcpy((void*)buffer2, (void*)cconn, 8*K*T_global*sizeof(double));
#endif
        if(g_cart_id==0) {
          sprintf(filename, "correl.%.4d.%.2d.%.2d.%.2d", Nconf, g_source_timeslice, mms2, k);
          if(count==0 && !(g_local_local || g_local_smeared)) {
            ofs=fopen(filename, "w");
          } else {
            ofs=fopen(filename, "a");
          }
          if( ofs == (FILE*)NULL ) {
            fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
            MPI_Abort(MPI_COMM_WORLD, 9);
            MPI_Finalize();
#endif
            exit(8);
          }
          for(idx=0; idx<8*K*T_global; idx++) buffer2[idx] *= correlator_norm;
  
          fprintf(ofs, "# %5d%3d%3d%3d%3d%15.8e%15.8e%15.8e%3d%3d\tSL und SS\n",
            Nconf, T_global, LX_global, LY_global, LZ, g_kappa, mms_masses[mms2], mms_masses[k], -sigmalight, -sigmaheavy);
          for(idx=0; idx<K; idx++) {
            for(ll=2; ll<4; ll++) {
              x1 = (0+g_source_timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0, isneg[idx]*buffer2[i], 0.);
              for(x0=1; x0<T_global/2; x0++) {
                x1 = ( x0+g_source_timeslice) % T_global;
                x2 = (-x0+g_source_timeslice+T_global) % T_global;
                i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
                j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + xisimag[idx];
                fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], isneg[idx]*buffer2[j]);
              }
              x0 = T_global/2;
              x1 = (x0+g_source_timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], 0.);
            }
          }
          fclose(ofs);
        }
      }}
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for light-light contractions: %e seconds\n", retime-ratime);

    index_min += 4*n_s*n_c;
  }

  
  /****************************************************************
   * (2.1) the heavy masses
   ****************************************************************/
  if(mms2_min<g_no_light_masses) mms2_min = g_no_light_masses;
  index_min = g_no_light_masses * 4*n_s*n_c;

  for(k=mms2_min; k<=g_no_extra_masses; k++) {
    mms2 = k;
    if(!light_mms) mms2--;

    /****************************************************************
     * (2.1.0) read the heavy-mass propagators
     ****************************************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(i=0; i<n_s*n_c; i++) {
      prepare_propagator(g_source_timeslice, i+index_smeared_start, heavy_mms, mms2, -1., mms_masses[k], i+index_min, work, 0);
      if(heavy_mms) {
        Qf5(g_spinor_field[i+2*n_s*n_c+index_min], work, mms_masses[k]);
      } else {
        if(non_mms_one_file) {
          prepare_propagator(g_source_timeslice, i+index_smeared_start, heavy_mms, mms2, -1., mms_masses[k], i+2*n_s*n_c+index_min, work, 1);
        } else {
          prepare_propagator(g_source_timeslice, i+index_smeared_start, heavy_mms, mms2, 1., mms_masses[k], i+2*n_s*n_c+index_min, work, 0);
        }
      }
      if(N_Jacobi>0) {
        memcpy((void*)g_spinor_field[i+n_s*n_c+index_min], (void*)g_spinor_field[i+index_min], 24*VOLUME*sizeof(double));
        xchange_field_timeslice(g_spinor_field[i+n_s*n_c+index_min]);
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i+n_s*n_c+index_min], work, kappa_Jacobi);
          xchange_field_timeslice(g_spinor_field[i+n_s*n_c+index_min]);
        }
        memcpy((void*)g_spinor_field[i+3*n_s*n_c+index_min], (void*)g_spinor_field[i+2*n_s*n_c+index_min], 24*VOLUME*sizeof(double));
        xchange_field_timeslice(g_spinor_field[i+3*n_s*n_c+index_min]);
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i+3*n_s*n_c+index_min], work, kappa_Jacobi);
          xchange_field_timeslice(g_spinor_field[i+3*n_s*n_c+index_min]);
        }
      }
    }  
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for preparing heavy prop.: %e seconds\n", retime-ratime);

  
    /****************************************************************
     * (2.1.1) heavy - heavy contractions (mass-diagonal)
     ****************************************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    count=-1; 
    for(sigmalight=-1; sigmalight<=1; sigmalight+=2) {
    for(sigmaheavy=-1; sigmaheavy<=1; sigmaheavy+=2) {
      count++;
      for(ix=0; ix<8*K*T; ix++) cconn[ix] = 0.;
      for(j=0; j<2; j++) {
        ll = j+2;
        chi  = &g_spinor_field[( sigmalight+1 + j )*n_s*n_c+index_min];
        psi  = &g_spinor_field[( sigmaheavy+1 + j )*n_s*n_c+index_min];
        if(sigmaheavy==sigmalight) {
          xgindex1 = gindex1;  xgindex2 = gindex2;  xisimag=isimag;  xvsign=vsign;  conf_gamma_sign = c_conf_gamma_sign;
        } else {
          xgindex1 = ngindex1; xgindex2 = ngindex2; xisimag=nisimag; xvsign=nvsign; conf_gamma_sign = n_conf_gamma_sign;
        }

        sl = 2*ll*K*T;

        /* (pseudo-)scalar sector */
        for(idx=0; idx<16; idx++) {
          contract_twopoint(&cconn[sl], xgindex1[idx], xgindex2[idx], chi, psi, n_c);
          sl += (2*T);
        }
        /* (pseudo-)vector sector */
        for(idx = 16; idx < 64; idx+=3) {
          for(i = 0; i < 3; i++) {
            for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
            contract_twopoint(Ctmp, xgindex1[idx+i], xgindex2[idx+i], chi, psi, n_c);
            for(x0=0; x0<T; x0++) {
              cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0  ]);
              cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0+1]);
            }
          }
          sl += (2*T);
        }
      }

#ifdef MPI
#if (defined PARALLELTX) || (defined PARALLELTXY)
      for(ix=0; ix<8*K*T; ix++) buffer[ix] = 0.;
      for(ix=0; ix<8*K*T_global; ix++) buffer2[ix] = 0.;
      MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
      MPI_Allgather(buffer, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, g_xs_comm);
/*
      MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
      MPI_Gather(buffer, 8*K*T, MPI_DOUBLE, cconn, 8*K*T, MPI_DOUBLE, 0, g_xs_comm);
*/
#else
      MPI_Gather(cconn, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, 0, g_cart_grid);
/*      if(g_cart_id==0) memcpy((void*)cconn, (void*)buffer, 8*K*T_global*sizeof(double)); */
#endif
#else
      memcpy((void*)buffer2, (void*)cconn, 8*K*T_global*sizeof(double));
#endif
      if(g_cart_id==0) {
        sprintf(filename, "correl.%.4d.%.2d.%.2d.%.2d", Nconf, g_source_timeslice, k, k);
        if(count==0 && !(g_local_local || g_local_smeared)) {
          ofs=fopen(filename, "w");
        } else {
          ofs=fopen(filename, "a");
        }
        if( ofs == (FILE*)NULL ) {
          fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
          MPI_Abort(MPI_COMM_WORLD, 9);
          MPI_Finalize();
#endif
          exit(8);
        }

        for(idx=0; idx<8*K*T_global; idx++) buffer2[idx] *= correlator_norm;

        fprintf(ofs, "# %5d%3d%3d%3d%3d%15.8e%15.8e%15.8e%3d%3d\tSL und SS\n",
          Nconf, T_global, LX_global, LY_global, LZ, g_kappa, mms_masses[k], mms_masses[k], -sigmalight, -sigmaheavy);
        for(idx=0; idx<K; idx++) {
          for(ll=2; ll<4; ll++) {
            x1 = (0+g_source_timeslice) % T_global;
            i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0, isneg[idx]*buffer2[i], 0.);
            for(x0=1; x0<T_global/2; x0++) {
              x1 = ( x0+g_source_timeslice) % T_global;
              x2 = (-x0+g_source_timeslice+T_global) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], isneg[idx]*buffer2[j]);
            }
            x0 = T_global/2;
            x1 = (x0+g_source_timeslice) % T_global;
            i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], 0.);
          }
        }
        fclose(ofs);
      }
    }}
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for heavy-heavy contractions: %e seconds\n", retime-ratime);

    /****************************************************************
     * (2.1.2) light - heavy contractions
     ****************************************************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(mms1=0; mms1<g_no_light_masses; mms1++) {
      
      count=-1;
      for(sigmalight=-1; sigmalight<=1; sigmalight+=2) {
      for(sigmaheavy=-1; sigmaheavy<=1; sigmaheavy+=2) {
        count++;
        for(ix=0; ix<8*K*T; ix++) cconn[ix] = 0.;
        for(j=0; j<2; j++) {
          ll = j+2;
          chi  = &g_spinor_field[( sigmalight+1 + j + 4*mms1 )*n_s*n_c];
          psi  = &g_spinor_field[( sigmaheavy+1 + j )*n_s*n_c + index_min];
          if(sigmaheavy==sigmalight) {
            xgindex1 = gindex1;  xgindex2 = gindex2;  xisimag=isimag;  xvsign=vsign;  conf_gamma_sign = c_conf_gamma_sign;
          } else {
            xgindex1 = ngindex1; xgindex2 = ngindex2; xisimag=nisimag; xvsign=nvsign; conf_gamma_sign = n_conf_gamma_sign;
          }

          sl = 2*ll*K*T;

          /* (pseudo-)scalar sector */
          for(idx=0; idx<16; idx++) {
            contract_twopoint(&cconn[sl], xgindex1[idx], xgindex2[idx], chi, psi, n_c);
            sl += (2*T);
          }
          /* (pseudo-)vector sector */
          for(idx = 16; idx < 64; idx+=3) {
            for(i = 0; i < 3; i++) {
              for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
              contract_twopoint(Ctmp, xgindex1[idx+i], xgindex2[idx+i], chi, psi, n_c);
              for(x0=0; x0<T; x0++) {
                cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0  ]);
                cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-16)/3]*xvsign[idx-16+i]*Ctmp[2*x0+1]);
              }
            }
            sl += (2*T);
          }
        }
#ifdef MPI
#if (defined PARALLELTX) || (defined PARALLELTXY)
        for(ix=0; ix<8*K*T; ix++) buffer[ix] = 0.;
        for(ix=0; ix<8*K*T_global; ix++) buffer2[ix] = 0.;
        MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
        MPI_Allgather(buffer, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, g_xs_comm);
/*
        MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
        MPI_Gather(buffer, 8*K*T, MPI_DOUBLE, cconn, 8*K*T, MPI_DOUBLE, 0, g_xs_comm);
*/
#else
        MPI_Gather(cconn, 8*K*T, MPI_DOUBLE, buffer2, 8*K*T, MPI_DOUBLE, 0, g_cart_grid);
/*        if(g_cart_id==0) memcpy((void*)cconn, (void*)buffer, 8*K*T_global*sizeof(double)); */
#endif
#else
        memcpy((void*)buffer2, (void*)cconn, 8*K*T_global*sizeof(double));
#endif
        if(g_cart_id==0) {
          sprintf(filename, "correl.%.4d.%.2d.%.2d.%.2d", Nconf, g_source_timeslice, mms1, k);
          if(count==0 && !(g_local_local || g_local_smeared)) {
            ofs=fopen(filename, "w");
          } else {
            ofs=fopen(filename, "a");
          }
          if( ofs == (FILE*)NULL ) {
            fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
            MPI_Abort(MPI_COMM_WORLD, 9);
            MPI_Finalize();
#endif
            exit(8);
          }

          for(idx=0; idx<8*K*T_global; idx++) buffer2[idx] *= correlator_norm;

          fprintf(ofs, "# %5d%3d%3d%3d%3d%15.8e%15.8e%15.8e%3d%3d\tSL und SS\n",
            Nconf, T_global, LX_global, LY_global, LZ, g_kappa, mms_masses[mms1], mms_masses[k], -sigmalight, -sigmaheavy);
          for(idx=0; idx<K; idx++) {
            for(ll=2; ll<4; ll++) {
              x1 = (0+g_source_timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0, isneg[idx]*buffer2[i], 0.);
              for(x0=1; x0<T_global/2; x0++) {
                x1 = ( x0+g_source_timeslice) % T_global;
                x2 = (-x0+g_source_timeslice+T_global) % T_global;
                i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
                j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + xisimag[idx];
                fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], isneg[idx]*buffer2[j]);
              }
              x0 = T_global/2;
              x1 = (x0+g_source_timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + xisimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*buffer2[i], 0.);
            }
          }
          fclose(ofs);
        }
      }}
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for light-heavy contractions: %e seconds\n", retime-ratime);

  }  /* loop over heavy extra masses */
  if(g_cart_id==0) fprintf(stdout, "# finished SL and SS contractions\n");
}
  /**************************************************
   * free the allocated memory, finalize 
   **************************************************/
  if(g_gauge_field != (double*)NULL) free(g_gauge_field); 
  if(no_fields>0) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field); 
  }
  free_geometry(); 
  free(cconn);
  free(Ctmp);
  if(gauge_field_f != (double*)NULL) free(gauge_field_f);
  if(mms_masses!=NULL) free(mms_masses);
  free(buffer); 
  free(buffer2); 
#ifdef MPI
  MPI_Finalize();
#endif
  return(0);

}
