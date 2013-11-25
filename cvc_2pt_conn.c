/****************************************************
 * cvc_2pt_conn.c
 *
 * Mon Feb 08 23:02:00 MEST 2010
 *
 * PURPOSE:
 * - major change: source momentum must be negative of sink momentum ( i.e. correlation of the form
 *   <G(x) G(y)^\dagger> exp(i \vec{k} (\vec{x} - \vec{y}) )
 * - propagators are assumed to have been generated with correctly chosen source momenta
 * - multiplication with momentum factor exp( i \vec{k} ( \vec{x}-\vec{x_src} ) ) in contract_twopoint_snk_momentum
 * TODO:
 * - test with smearing
 * - tests with point sources (n_c = 3)
 * - (maybe) integrate additional switch in case of no mms to
 *   enable reading propagators from either two files at position 0 or
 *   one file at position 0 and 1
 * - add OpenMP
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
#endif
#ifdef OPENMP
#  include <omp.h>
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
#include "gauge_io.h"
#include "Q_phi.h"
#include "fuzz.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "make_q_orbits.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for connected contributions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose [no effect, lots of stdout output it]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -l Nlong for fuzzing [default -1, no fuzzing]\n");
  fprintf(stdout, "         -a no of steps for APE smearing [default -1, no smearing]\n");
  fprintf(stdout, "         -k alpha for APE smearing [default 0.]\n");
  EXIT(0);
}

int n_c=1, n_s=4;

static inline void get_propagator_filename(char*filename, char*prefix, int*sc, int i, int*source_momentum, int conf) {
  int isc;
  switch(g_source_type) {
    case 0:  // point source
      isc = i % (n_s*n_c);
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", prefix, conf, sc[0], sc[1], sc[2], sc[3], isc);
      break;
    case 2:  // timeslice source
    case 3:  // timeslice source
      if(g_sink_momentum_set) {
        sprintf(filename, "%s.%.4d.%.2d.%.2d.qx%.2dqy%.2dqz%.2d.inverted", prefix, conf, g_source_timeslice, i,
            source_momentum[0], source_momentum[1], source_momentum[2]);
      } else {
        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", prefix, conf, g_source_timeslice, i);
      }
      break;
  }
  fprintf(stdout, "# [get_propagator_filename] filename = %s\n", filename);
  return;
}

/****************************************************************************************
 *
 * main program
 *
 ****************************************************************************************/
int main(int argc, char **argv) {
 
  const int src_momentum_zero[] = {0,0,0};

  int c, i, j, ll, sl, status;
  int filename_set = 0;
  int timeslice=2, mms1=0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, ix, idx;
  int VOL3, icol;
  int K=20, nK=20, itype;
  int use_mms=0;
  int full_orbit=0;
  int do_shifts = 0, shifts_num=1, ishift;
  int source_coords[4], source_coords_orig[4], source_proc_coords[4], source_proc_id, source_location, lsource_coords[4];
  double *cconn = (double*)NULL;
  double *nconn = (double*)NULL;
  double *work=NULL;
  int verbose = 0;
  int fermion_type = 0;  // twisted mass fermion
  char filename[200];
  double ratime, retime;
  double plaq_r, plaq_m;
  double *gauge_field_timeslice=NULL, *gauge_field_f=NULL;
  double **chi=NULL, **chi2=NULL, **psi=NULL, **psi2=NULL;
  double *Ctmp;
  double correlator_norm;
  FILE *ofs;
/*  double sign_adj5[] = {-1., -1., -1., -1., +1., +1., +1., +1., +1., +1., -1., -1., -1., 1., -1., -1.}; */
  double conf_gamma_sign[] = {1., 1., 1., 1., 1., -1., -1., -1., -1.};
  int *qlatt_id=NULL, *qlatt_count=NULL, **qlatt_rep=NULL, **qlatt_map=NULL, qlatt_nclass;
  double **qlatt_list=NULL;
  int snk_momentum_runs = 1, snk_momentum_id=0, snk_momentum[3], src_momentum[3], imom;
  int shift_vector[5][4] =  {{0,0,0,0}, {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  size_t nconn_length=0, cconn_length=0;

  /**************************************************************************************************
   * charged stuff
   * here we loop over ll, ls, sl, ss (order source-sink)
   * pion:
   * g5-g5, g5-g0g5, g0g5-g5, g0g5-g0g5, g0-g0, g5-g0, g0-g5, g0g5-g0, g0-g0g5
   * rho:
   * gig0-gig0, gi-gi, gig5-gig5, gig0-gi, gi-gig0, gig0-gig5, gig5-gig0, gi-gig5, gig5-gi
   * a0, b1:
   * 1-1, gig0g5-gig0g5
   **************************************************************************************************/
  int gindex1[] = {5, 5, 6, 6, 0, 5, 0, 6, 0,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 10, 11, 12, 1, 2, 3, 10, 11, 12, 7, 8, 9, 1, 2, 3, 7, 8, 9,
                   4, 13, 14, 15};

  int gindex2[] = {5, 6, 5, 6, 0, 0, 5, 0, 6,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 1, 2, 3, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 1, 2, 3,
                   4, 13, 14, 15};

  /* due to twisting we have several correlators that are purely imaginary */
  int isimag[]  = {0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0};

  /**************************************************************************************************
   * neutral stuff
   * here we loop over ll, ls, sl, ss (order source-sink)
   * pion:
   * g5-g5, g5-g0g5, g0g5-g5, g0g5-g0g5, 1-1, g5-1, 1-g5, g0g5-1, 1-g0g5
   * rho:
   * gig0-gig0, gi-gi, gig0g5-gig0g5, gig0-gi, gi-gig0, gig0-gig0g5, gig0g5-gig0, gi-gig0g5, gig0g5-gi
   * a0, b1:
   * g0-g0, gig5-gig5
   **************************************************************************************************/
  int ngindex1[] = {5, 5, 6, 6, 4, 5, 4, 6, 4,
                    10, 11, 12, 1, 2, 3, 13, 14, 15, 10, 11, 12, 1, 2, 3, 10, 11, 12, 13, 14, 15, 1, 2, 3, 13, 14, 15,
                    0, 7, 8, 9};
  int ngindex2[] = {5, 6, 5, 6, 4, 4, 5, 4, 6,
                    10, 11, 12, 1, 2, 3, 13, 14, 15, 1, 2, 3, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 1, 2, 3,
                    0, 7, 8, 9};
  int nisimag[]  = {0, 0, 0, 0, 0, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 1, 1, 1, 1,
                    0, 0};
  double isneg_std[]=    {+1., -1., +1., -1., +1., +1., +1., +1., -1.,
                          -1., +1., -1., -1., +1., +1., +1., -1., +1.,
                          +1., -1.};
  double isneg[20];

  /* every correlator for the rho part including gig0 either at source
   * or at sink has a different relative sign between the 3 contributions */
  double vsign[]= {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1.,
                   1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
  double nvsign[] ={1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1.,
                    1., 1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1.};

#ifdef MPI
  MPI_Status status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "soh?vguf:p:m:F:")) != -1) {
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
      fprintf(stdout, "# [] will use number of colors = %d\n", n_c);
      break;
    case 'm':
      mms1 = atoi(optarg);
      break;
    case 'u':
      use_mms = 1;
      break;
    case 'F':
      if(strcmp(optarg, "Wilson") == 0) {
        fermion_type = _WILSON_FERMION;
      } else if(strcmp(optarg, "tm") == 0) {
        fermion_type = _TM_FERMION;
      } else {
        fprintf(stderr, "[] Error, unrecognized fermion type\n");
        EXIT(145);
      }
      fprintf(stdout, "# [] will use fermion type %s ---> no. %d\n", optarg, fermion_type);
      break;
    case 'o':
      full_orbit=1;
      fprintf(stdout, "# [] will loop over full orbit\n");
      break;
    case 's':
      do_shifts=1;
      fprintf(stdout, "# [] will include shifts +e_\\mu of source location\n");
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
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  if(!g_sink_momentum_set && full_orbit) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, full orbit but no sink momentum specified\n");
    EXIT(123);
  }


  /* initialize MPI parameters */
  mpi_init(argc, argv);

#ifdef MPI
  T = T_global / g_nproc;
  Tstart = g_cart_id * T;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  VOL3 = LX*LY*LZ;
#else
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  VOL3 = LX*LY*LZ;
#endif
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    EXIT(1);
  }

  geometry();

  // prepare momentum lists
  if(g_sink_momentum_set && full_orbit) {
    status = make_qcont_orbits_3d_parity_avg( &qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
    if(status != 0) {
      fprintf(stderr, "\n[baryon_corr_qdep] Error while creating O_3-lists\n");
      EXIT(4);
    }
    fprintf(stdout, "# [baryon_corr_qdep] number of classes = %d\n", qlatt_nclass);
  }

  /**********************************
   * check for shifts
   **********************************/
  if(do_shifts && g_source_type == 0) {
    shifts_num = 5;
  } else {
    shifts_num = 1;
  }

  /**********************************
   * source coordinates
   **********************************/
  if(g_source_type==0) {
    source_coords[0] = g_source_location / (LX_global * LY_global * LZ_global);
    source_coords[1] = ( g_source_location % (LX_global * LY_global * LZ_global) ) / (LY_global * LZ_global);
    source_coords[2] = ( g_source_location % (LY_global * LZ_global) ) / LZ_global;
    source_coords[3] = g_source_location % LZ_global;
    fprintf(stdout, "# [] global source_location %d ---> global source coordinates = (%d, %d, %d, %d)\n", g_source_location,
        source_coords[0],source_coords[1], source_coords[2], source_coords[3]);
    g_source_timeslice = source_coords[0];
    memcpy(source_coords_orig, source_coords, 4*sizeof(int));
    source_proc_coords[0] = source_coords[0] / T;  
    source_proc_coords[1] = source_coords[1] / LX;  
    source_proc_coords[2] = source_coords[2] / LY;  
    source_proc_coords[3] = source_coords[3] / LZ;
#ifdef MPI
    MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
#else
    source_proc_id = 0;
#endif 
    lsource_coords[0] = source_coords[0] % T;
    lsource_coords[1] = source_coords[1] % LX;
    lsource_coords[2] = source_coords[2] % LY;
    lsource_coords[3] = source_coords[3] % LZ;
    if(g_proc_id == source_proc_id) {
      source_location = g_ipt[lsource_coords[0]][lsource_coords[1]][lsource_coords[2]][lsource_coords[3]];
      fprintf(stdout, "# [] local source_location %d ---> local source coordinates = (%d, %d, %d, %d)\n", source_location,
          lsource_coords[0], lsource_coords[1], lsource_coords[2], lsource_coords[3]);
    }
  } else {
    source_coords[0] = 0;
    source_coords[1] = 0;
    source_coords[2] = 0;
    source_coords[3] = 0;
    memcpy(source_coords_orig, source_coords, 4*sizeof(int));
    memcpy(lsource_coords, source_coords, 4*sizeof(int));
  }


  for(i = 0; i < 20; i++) isneg[i] = isneg_std[i];

  // allocate memory for the contractions
  if(g_cart_id==0) { idx = 8*K*T_global; } 
  else             { idx = 8*K*T; }
  cconn_length = idx;
  cconn = (double*)calloc(idx, sizeof(double));
  if( cconn==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for cconn\n");
    EXIT(3);
  }
  for(ix=0; ix<idx; ix++) cconn[ix] = 0.;
  
  if(g_cart_id==0) { idx = 8*nK*T_global; } 
  else             { idx = 8*nK*T; }
  nconn_length = idx;
  nconn = (double*)calloc(idx, sizeof(double));
  if( nconn==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for nconn\n");
    EXIT(5);
  }
  for(ix=0; ix<idx; ix++) nconn[ix] = 0.;

  if( (Ctmp = (double*)calloc(2*T, sizeof(double))) == NULL ) {
    fprintf(stderr, "Error, could not allocate mem for Ctmp\n");
    EXIT(4);
  }
  

  if( N_Jacobi>0) {
    alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
 
    // read the gauge field
    switch(g_gauge_file_format) {
      case 0:
        sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
        if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
        status = read_lime_gauge_field_doubleprec(filename);
        break;
      case 1:
        sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
        if(g_cart_id==0) fprintf(stdout, "\n# [] reading gauge field from file %s\n", filename);
        status = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
      break;
    }
    if(status != 0) {
      fprintf(stderr, "[] Error, could not read gauge field\n");
      EXIT(21);
    }
    xchange_gauge();
  
    // measure the plaquette
    plaquette(&plaq_m);
    if(g_cart_id==0) fprintf(stdout, "# plaquette values measures = %25.16e; read = %25.16e\n", plaq_m, plaq_r);
  
    if(g_cart_id==0) fprintf(stdout, "# apply fuzzing of gauge field and propagators with parameters:\n"\
                                     "# Nlong = %d\n# N_ape = %d\n# alpha_ape = %f\n", Nlong, N_ape, alpha_ape);
    alloc_gauge_field(&gauge_field_f, VOLUMEPLUSRAND);
    // copy the gauge field to smear it
    memcpy(gauge_field_f, g_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));
    if(g_cart_id==0) fprintf(stdout, "# APE-smearing / fuzzing gauge field with Nlong=%d, N_APE=%d, alpha_APE=%f\n", Nlong, N_ape, alpha_ape);
#ifdef OPENMP
    if(N_ape>0) {
      APE_Smearing_Step_threads(gauge_field_f, N_ape, alpha_ape);
    }
    if(Nlong>0) {
      if(g_cart_id==0) fprintf(stdout, "[] Warning, no threaded version of fuzzing\n");
      alloc_gauge_field(&gauge_field_timeslice, VOL3);
      for(x0=0; x0<T; x0++) {
        memcpy( gauge_field_timeslice, gauge_field_f + _GGI(g_ipt[x0][0][0][0],0), 72*VOL3*sizeof(double));
        fuzzed_links_Timeslice(gauge_field_f, gauge_field_timeslice, Nlong, x0);
      }
      free(gauge_field_timeslice);
    }
#else
    alloc_gauge_field(&gauge_field_timeslice, VOL3);
    for(x0=0; x0<T; x0++) {
      memcpy((void*)gauge_field_timeslice, (void*)(g_gauge_field+_GGI(g_ipt[x0][0][0][0],0)), 72*VOL3*sizeof(double));
      for(i=0; i<N_ape; i++) {
        APE_Smearing_Step_Timeslice(gauge_field_timeslice, alpha_ape);
      }
      if(Nlong > 0) {
        fuzzed_links_Timeslice(gauge_field_f, gauge_field_timeslice, Nlong, x0);
      } else {
        memcpy((void*)(gauge_field_f+_GGI(g_ipt[x0][0][0][0],0)), (void*)gauge_field_timeslice, 72*VOL3*sizeof(double));
      }
    }
    free(gauge_field_timeslice);
#endif
  // test: print the fuzzed APE smeared gauge field to stdout
  //  for(ix=0; ix<36*VOLUME; ix++) {
  //    fprintf(stdout, "%6d%25.16e%25.16e\n", ix, g_gauge_field[2*ix], g_gauge_field[2*ix+1]);
  //  }
  } else {
    g_gauge_field = NULL;
  }
  
  // allocate memory for the spinor fields
  no_fields = n_s;
  if( fermion_type==0 || (g_sink_momentum_set && ( g_source_type==2  || g_source_type==3 || g_source_type==4) ) ) no_fields+=n_s;
  if(Nlong>0) no_fields += n_s;
  no_fields *= n_c;
  no_fields++;
  if(g_cart_id==0) fprintf(stdout, "# [] total number of fields = %d\n", no_fields);
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUMEPLUSRAND);
  
  timeslice = g_source_timeslice;
  
  // check source/sink momentum
  if(g_sink_momentum_set) {
  /*
    if(g_source_momentum[0]<0) g_source_momentum[0] += LX_global;
    if(g_source_momentum[1]<0) g_source_momentum[1] += LY_global;
    if(g_source_momentum[2]<0) g_source_momentum[2] += LZ_global;
  */
    if(g_sink_momentum[0]<0) g_sink_momentum[0] += LX_global;
    if(g_sink_momentum[1]<0) g_sink_momentum[1] += LY_global;
    if(g_sink_momentum[2]<0) g_sink_momentum[2] += LZ_global;
    fprintf(stdout, "# [] using final sink momentum ( %d, %d, %d )\n", g_sink_momentum[0], g_sink_momentum[1], g_sink_momentum[2]);
  
    g_source_momentum[0] = (-g_sink_momentum[0] + LX_global ) % LX_global;
    g_source_momentum[1] = (-g_sink_momentum[1] + LY_global ) % LY_global;
    g_source_momentum[2] = (-g_sink_momentum[2] + LZ_global ) % LZ_global;
    fprintf(stdout, "# [] using final source momentum ( %d, %d, %d )\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
  
    if(full_orbit) {
      snk_momentum_id   = qlatt_id[g_ipt[0][g_sink_momentum[0]][g_sink_momentum[1]][g_sink_momentum[2]]];
      snk_momentum_runs = qlatt_count[snk_momentum_id] + 1;
    }
  }
  fprintf(stdout, "# [] number of runs = %d\n", snk_momentum_runs);
  
  // set the correlator norm
  //  correlator_norm = 1. / ( (double)VOL3 * g_kappa * g_kappa * 2.);
  correlator_norm = 1.;
  fprintf(stdout, "# [] using correlator norm %e\n", correlator_norm);

  /*************************************
   * loop on shifts of source location
   *************************************/
  for(ishift=0; ishift<shifts_num; ishift++) {
 
    memset(cconn, 0, cconn_length*sizeof(double));
    memset(nconn, 0, nconn_length*sizeof(double));

    if(ishift>0) {
      source_coords[0] = (source_coords_orig[0] + shift_vector[ishift][0] ) % T_global;
      source_coords[1] = (source_coords_orig[1] + shift_vector[ishift][1] ) % LX_global;
      source_coords[2] = (source_coords_orig[2] + shift_vector[ishift][2] ) % LY_global;
      source_coords[3] = (source_coords_orig[3] + shift_vector[ishift][3] ) % LZ_global;
      source_proc_coords[0] = source_coords[0] / T;
      source_proc_coords[1] = source_coords[1] / LX;
      source_proc_coords[2] = source_coords[2] / LY;
      source_proc_coords[3] = source_coords[3] / LZ;
#ifdef MPI
      MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
#else
      source_proc_id = 0;
#endif
      lsource_coords[0] = source_coords[0] % T;
      lsource_coords[1] = source_coords[1] % LX;
      lsource_coords[2] = source_coords[1] % LY;
      lsource_coords[3] = source_coords[1] % LZ;
      source_location = g_ipt[lsource_coords[0]][lsource_coords[1]][lsource_coords[2]][lsource_coords[3]];
    }
  
    /*************************************
     * loop on sink momentum runs
     *************************************/
      for(imom=0; imom<snk_momentum_runs;imom++) {
    
        if(imom == 0) {
          if(full_orbit) {
            snk_momentum[0] = 0;
            snk_momentum[1] = 0;
            snk_momentum[2] = 0;
            src_momentum[0] = 0;
            src_momentum[1] = 0;
            src_momentum[2] = 0;
          } else {
            snk_momentum[0] =  g_sink_momentum[0];
            snk_momentum[1] =  g_sink_momentum[1];
            snk_momentum[2] =  g_sink_momentum[2];
            src_momentum[0] = (-g_sink_momentum[0] + LX_global) % LX_global;
            src_momentum[1] = (-g_sink_momentum[1] + LY_global) % LY_global;
            src_momentum[2] = (-g_sink_momentum[2] + LZ_global) % LZ_global;
          }
        } else {
          snk_momentum[0] = qlatt_map[snk_momentum_id][imom-1] / (LY_global*LZ_global);
          snk_momentum[1] = ( qlatt_map[snk_momentum_id][imom-1] % (LY_global*LZ_global) ) / LZ_global;
          snk_momentum[2] = qlatt_map[snk_momentum_id][imom-1] % LZ_global;
          src_momentum[0] = (-snk_momentum[0] + LX_global ) % LX_global; 
          src_momentum[1] = (-snk_momentum[1] + LY_global ) % LY_global;
          src_momentum[2] = (-snk_momentum[2] + LZ_global ) % LZ_global;
        }
        fprintf(stdout, "# [] run no. %d with source momentum (%d, %d, %d) and sink momentum (%d, %d, %d)\n", imom,
            src_momentum[0], src_momentum[1], src_momentum[2],\
            snk_momentum[0], snk_momentum[1], snk_momentum[2]);
    
    
        if(g_cart_id==0) {
          for(ix=0; ix<8*K*T_global; ix++) cconn[ix] = 0.;
          for(ix=0; ix<8*K*T_global; ix++) nconn[ix] = 0.;
        } else {
          for(ix=0; ix<8*K*T; ix++) cconn[ix] = 0.;
          for(ix=0; ix<8*K*T; ix++) nconn[ix] = 0.;
        }
    
        /*************************************
         * begin loop on LL, LS, SL, SS
         *************************************/
          ll = 0;
    //      for(j=0; j<4; j++)
          for(j=0; j<1; j++)
          {
            work = g_spinor_field[no_fields-1];
            if(j==0) {
              // local-local (source-sink) -> phi[0-3]^dagger.p[0-3] -> p.p
              ll = 0;
              for(i=0; i<n_s*n_c; i++) {
                if(use_mms) {
                  sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, timeslice, i, mms1);
                  read_lime_spinor(work, filename, 0);
                  xchange_field(work);
                  Qf5(g_spinor_field[i], work, -g_mu);
                  if(fermion_type == 0) {
                    Qf5(g_spinor_field[i+n_s*n_c], work, g_mu);
                  }
                } else {
                  get_propagator_filename(filename, filename_prefix, source_coords, i, src_momentum_zero, Nconf);
                  check_error(read_lime_spinor(g_spinor_field[i], filename, 0), "read_lime_spinor", NULL, 15);
                  if(g_sink_momentum_set) {
                    get_propagator_filename(filename, filename_prefix, source_coords, i, src_momentum, Nconf);
                    check_error(read_lime_spinor(g_spinor_field[i+n_s*n_c], filename, 0), "read_lime_spinor", NULL, 15);
                  }
                  if(fermion_type == 0) { // read down propagators
                    // get_propagator_filename(filename, filename_prefix2, source_coords, i, src_momentum, Nconf);
                    // check_error(read_lime_spinor(g_spinor_field[i+n_s*n_c*(1+g_sink_momentum_set)], filename, 0), "read_lime_spinor", NULL, 16);
                    check_error(read_lime_spinor(g_spinor_field[i+n_s*n_c*(1+g_sink_momentum_set)], filename, 1), "read_lime_spinor", NULL, 16);
                  }
                }  // of if use_mms
              }    // of loop on isc
              chi  = &g_spinor_field[0];
              psi  = &g_spinor_field[g_sink_momentum_set*n_s*n_c];
              if(fermion_type==0) {
                chi2 = &g_spinor_field[0];
                psi2 = &g_spinor_field[n_s*n_c*(1+g_sink_momentum_set)];
              } else {
                chi2 = NULL;
                psi2 = NULL;
              }
            } else if(j==1) {
              if(Nlong>0) { // fuzzed-local -> phi[0-3]^dagger.phi[4-7] -> p.f
                ll = 2;
                chi  = &g_spinor_field[0];
                psi  = &g_spinor_field[n_s*n_c];
                if(fermion_type == 0) {
                  chi2 = &g_spinor_field[0];
                  psi2 = &g_spinor_field[2*n_s*n_c];
                } else {
                  chi2 = NULL;
                  psi2 = NULL;
                }
                for(i=n_s*n_c; i<2*n_s*n_c; i++) {
                  if(use_mms) {
                    sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, timeslice, i, mms1);
                    read_lime_spinor(work, filename, 0);
                    Qf5(g_spinor_field[i], work, -g_mu);
                    if(fermion_type==0) {
                      Qf5(g_spinor_field[i+n_s*n_c], work, g_mu);
                    }
                  } else {
                    get_propagator_filename(filename, filename_prefix, source_coords, i, src_momentum, Nconf);
                    check_error( read_lime_spinor(g_spinor_field[i], filename, 0), "read_lime_spinor", NULL, 17);
      
                    if(fermion_type==0) {
                      get_propagator_filename(filename, filename_prefix2, source_coords, i, src_momentum, Nconf);
                      check_error( read_lime_spinor(g_spinor_field[i+n_s*n_c], filename, 0), "read_lime_spinor", NULL, 17);
      /*              read_lime_spinor(g_spinor_field[i+n_s*n_c], filename, 1); */
                    }
                  }
                }
              } else {
              // local-smeared
                ll = 1; 
                chi  = &g_spinor_field[0];
                psi  = &g_spinor_field[g_sink_momentum_set*n_s*n_c];
                if(fermion_type==0) {
                  chi2 = &g_spinor_field[0];
                  psi2 = &g_spinor_field[n_s*n_c*(1+g_sink_momentum_set)];
                } else {
                  chi2 = NULL;
                  psi2 = NULL;
                }
                for(i = 0; i < ( (fermion_type==0) + (g_sink_momentum_set) + 1)*n_s*n_c; i++) {
    #ifdef OPENMP
                  Jacobi_Smearing_Step_one_threads(gauge_field_f, g_spinor_field[i], work, N_Jacobi, kappa_Jacobi);
    #else
                  for(x0 = 0; x0 < T; x0++) {
                    Jacobi_Smearing_Steps(gauge_field_f, g_spinor_field[i], N_Jacobi, kappa_Jacobi, x0);
                  }
    #endif
                }
              }
            } else if(j==2) {
              if(Nlong>0) {
              // local-fuzzed -> phi[0-3]^dagger.phi[4-7] -> p.pf
                ll = 1;
                chi  = &g_spinor_field[0];
                psi  = &g_spinor_field[  n_s*n_c];
                chi2 = fermion_type == 0 ? &g_spinor_field[0] : NULL;
                psi2 = fermion_type == 0 ? &g_spinor_field[2*n_s*n_c] : NULL;
                for(i=0; i<n_s*n_c; i++) {
                  if(use_mms) {
                    sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, timeslice, i, mms1);
                    read_lime_spinor(work, filename, 0);
                    xchange_field(work);
                    Qf5(g_spinor_field[i+n_s*n_c], work, -g_mu);
                    if(fermion_type==0) {
                      Qf5(g_spinor_field[i+2*n_s*n_c], work, g_mu);
                    }
                  } else {
                    get_propagator_filename(filename, filename_prefix, source_coords, i, src_momentum, Nconf);
                    check_error( read_lime_spinor(g_spinor_field[i+n_s*n_c], filename, 0), "read_lime_spinor", NULL, 18);
                    if(fermion_type==0) {
                      get_propagator_filename(filename, filename_prefix2, source_coords, i, src_momentum, Nconf);
                      check_error(read_lime_spinor(g_spinor_field[i+2*n_s*n_c], filename, 0), "read_lime_spinor", NULL, 19);
                      //status = read_lime_spinor(g_spinor_field[i+2*n_s*n_c], filename, 1);
                    }
                  }
                  if(g_cart_id==0) fprintf(stdout, "# fuzzing prop. with Nlong=%d\n", Nlong);
                  Fuzz_prop(gauge_field_f, g_spinor_field[i+n_s*n_c], Nlong);
                  if(fermion_type==0) {
                    Fuzz_prop(gauge_field_f, g_spinor_field[i+2*n_s*n_c], Nlong);
                  }
                }
              } else {
              // smeared-local
                ll = 2;
                //if(g_cart_id==0) fprintf(stdout, "# [] processing ll = 2\n");
                chi  = &g_spinor_field[0];
                psi  = &g_spinor_field[g_sink_momentum_set*n_s*n_c];
                chi2 = fermion_type == 0 ? &g_spinor_field[0] : NULL;
                psi2 = fermion_type == 0 ? &g_spinor_field[n_s*n_c*(1+g_sink_momentum_set)] : NULL;
                for(i=0; i<n_s*n_c; i++) {
                  if(use_mms) {
                    sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, timeslice, i+n_s*n_c, mms1);
                    read_lime_spinor(work, filename, 0);
                    xchange_field(work);
                    Qf5(g_spinor_field[i], work, -g_mu);
                    if(fermion_type==0) {
                      Qf5(g_spinor_field[i+n_s*n_c], work, g_mu);
                    }
                  } else {
                    get_propagator_filename(filename, filename_prefix, source_coords, i+n_s*n_c, src_momentum_zero, Nconf);
                    check_error( read_lime_spinor(g_spinor_field[i], filename, 0), "read_lime_spinor", NULL, 20);
                    if(g_sink_momentum_set) {
                      get_propagator_filename(filename, filename_prefix, source_coords, i+n_s*n_c, src_momentum, Nconf);
                      check_error( read_lime_spinor(g_spinor_field[i+n_s*n_c], filename, 0), "read_lime_spinor", NULL, 20);
                    }
                    if(fermion_type==0) {
                      get_propagator_filename(filename, filename_prefix2, source_coords, i+n_s*n_c, src_momentum, Nconf);
                      check_error( read_lime_spinor(g_spinor_field[i+n_s*n_c*(1+g_sink_momentum_set)], filename, 0), "read_lime_spinor", NULL, 21);
                      //status = read_lime_spinor(g_spinor_field[i+n_s*n_c], filename, 1);
                    }
                  }
                }
              }
            } else if(j==3) {
            // smeared-smeared -> phi[0-3]^dagger.phi[4-7] -> f.pf
              ll = 3;
              //if(g_cart_id==0) fprintf(stdout, "# [] processing ll = 3\n");
              if(Nlong>0) {
                chi  = &g_spinor_field[0];
                psi  = &g_spinor_field[  n_s*n_c];
                chi2 = fermion_type == 0 ? &g_spinor_field[0] : NULL;
                psi2 = fermion_type == 0 ? &g_spinor_field[2*n_s*n_c]: NULL;
                for(i=0; i<n_s*n_c; i++) {
                  if(use_mms) {
                    sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, timeslice, i+n_s*n_c, mms1);
                    read_lime_spinor(work, filename, 0);
                    xchange_field(work);
                    Qf5(g_spinor_field[i], work, -g_mu);
                  } else {
                    get_propagator_filename(filename, filename_prefix, source_coords, i+n_s*n_c, src_momentum_zero, Nconf);
                    check_error( read_lime_spinor(g_spinor_field[i], filename, 0), "read_lime_spinor", NULL, 22);
                  }
                }
              } else {
                chi  = &g_spinor_field[0];
                psi  = &g_spinor_field[g_sink_momentum_set*n_s*n_c];
                chi2 = fermion_type==0 ? &g_spinor_field[0]: NULL;
                psi2 = fermion_type==0 ? &g_spinor_field[n_s*n_c*(1+g_sink_momentum_set)]: NULL;
                for(i = 0; i < ( (fermion_type==0) + g_sink_momentum_set + 1)*n_s*n_c; i++) {
    #ifdef OPENMP
                  Jacobi_Smearing_Step_one_threads(gauge_field_f, g_spinor_field[i], work, N_Jacobi, kappa_Jacobi);
    #else
                  for(x0 = 0; x0 < T; x0++) {
                    Jacobi_Smearing_Steps(gauge_field_f, g_spinor_field[i], N_Jacobi, kappa_Jacobi, x0);
                  }
    #endif
                }
              }
            }
      
            /************************************************************
             * the charged contractions
             ************************************************************/
            sl = 2*ll*T*K;
            itype = 1; 
            // pion sector
            for(idx=0; idx<9; idx++)
            {
              contract_twopoint_snk_momentum(&cconn[sl], gindex1[idx], gindex2[idx], chi, psi, n_c, snk_momentum);
              //for(x0=0; x0<T; x0++) fprintf(stdout, "pion: %3d%25.16e%25.16e\n", x0, 
              //    cconn[sl+2*x0]/(double)VOL3/2./g_kappa/g_kappa, cconn[sl+2*x0+1]/(double)VOL3/2./g_kappa/g_kappa);
              sl += (2*T);        
              itype++; 
            }
      
            // rho sector
            for(idx = 9; idx < 36; idx+=3) {
              for(i = 0; i < 3; i++) {
                for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
                contract_twopoint_snk_momentum(Ctmp, gindex1[idx+i], gindex2[idx+i], chi, psi, n_c, snk_momentum);
                for(x0=0; x0<T; x0++) {
                  cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-9)/3]*vsign[idx-9+i]*Ctmp[2*x0  ]);
                  cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-9)/3]*vsign[idx-9+i]*Ctmp[2*x0+1]);
                }
                for(x0=0; x0<T; x0++) {
                  x1 = (x0+timeslice)%T_global;
                  //fprintf(stdout, "rho: %3d%25.16e%25.16e\n", x0, 
                  //  vsign[idx-9+i]*Ctmp[2*x1  ]/(double)VOL3/2./g_kappa/g_kappa, 
                  //  vsign[idx-9+i]*Ctmp[2*x1+1]/(double)VOL3/2./g_kappa/g_kappa);
                }
              }
              sl += (2*T); 
              itype++;
            }
      
            // the a0
            contract_twopoint_snk_momentum(&cconn[sl], gindex1[36], gindex2[36], chi, psi, n_c, snk_momentum);
            sl += (2*T);
            itype++;
      
            // the b1
            for(i=0; i<3; i++) {
              for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
              idx = 37;
              contract_twopoint_snk_momentum(Ctmp, gindex1[idx+i], gindex2[idx+i], chi, psi, n_c, snk_momentum);
              for(x0=0; x0<T; x0++) { 
                cconn[sl+2*x0  ] += (vsign[idx-9+i]*Ctmp[2*x0  ]);
                cconn[sl+2*x0+1] += (vsign[idx-9+i]*Ctmp[2*x0+1]);
              }
            }

            /************************************************************
             * the neutral contractions
             ************************************************************/
            if(fermion_type == 0) {
              sl = 2*ll*nK*T;
              itype = 1;
              // pion sector first
              for(idx=0; idx<9; idx++) {
                contract_twopoint_snk_momentum(&nconn[sl], ngindex1[idx], ngindex2[idx], chi2, psi2, n_c, snk_momentum);
                sl += (2*T);
                itype++;
              }
        
              // the neutral rho
              for(idx=9; idx<36; idx+=3) {
                for(i=0; i<3; i++) {
                  for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
                  contract_twopoint_snk_momentum(Ctmp, ngindex1[idx+i], ngindex2[idx+i], chi2, psi2, n_c, snk_momentum);
                  for(x0=0; x0<T; x0++) {
                    nconn[sl+2*x0  ] += (nvsign[idx-9+i]*Ctmp[2*x0  ]);
                    nconn[sl+2*x0+1] += (nvsign[idx-9+i]*Ctmp[2*x0+1]);
                  }
                }
                sl += (2*T);
                itype++;
              }
        
              // the X (JPC=0+- with no experimental candidate known)
              contract_twopoint_snk_momentum(&nconn[sl], ngindex1[36], ngindex2[36], chi2, psi2, n_c, snk_momentum);
              sl += (2*T);
              itype++;
        
              // the a1/f1
              for(i = 0; i < 3; i++) {
                for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
                idx = 37;
                contract_twopoint_snk_momentum(Ctmp, ngindex1[idx+i], ngindex2[idx+i], chi2, psi2, n_c, snk_momentum);
                for(x0=0; x0<T; x0++) {
                  nconn[sl+2*x0  ] += (nvsign[idx-9+i]*Ctmp[2*x0  ]);
                  nconn[sl+2*x0+1] += (nvsign[idx-9+i]*Ctmp[2*x0+1]);
                }
              }
            }  // of if fermion_type == 0
          }    // of j=0,...,3
      
    #ifdef MPI
        // collect the results
        if(g_cart_id==0) {
          for(i=1; i<g_nproc; i++) {
            MPI_Recv(&cconn[i*8*K*T], 8*K*T, MPI_DOUBLE, i, 100+i, g_cart_grid, &status);
            MPI_Recv(&nconn[i*8*nK*T], 8*nK*T, MPI_DOUBLE, i, 400+i, g_cart_grid, &status);
          }
        } else {
          for(i=1; i<g_nproc; i++) {
            if(g_cart_id==i) {
              MPI_Send(cconn, 8*K*T, MPI_DOUBLE, 0, 100+i, g_cart_grid);
              MPI_Send(nconn, 8*nK*T, MPI_DOUBLE, 0, 400+i, g_cart_grid);
            }
          }
        }
    #endif
        // write to file
        if(g_cart_id==0) {
          if(g_source_type == 0) {
            sprintf(filename, "charged.t%.2dx%.2dy%.2dz%.2d.%.4d", source_coords[0], source_coords[1], source_coords[2],
             source_coords[3], Nconf);
          } else {
            sprintf(filename, "charged.%.2d.%.4d", timeslice, Nconf);
          }
          if(imom==0) {
            ofs=fopen(filename, "w");
          } else {
            ofs=fopen(filename, "a");
          }
          if( ofs == (FILE*)NULL ) {
            fprintf(stderr, "Error, could not open file %s for writing\n", filename);
            EXIT(6);
          }
          fprintf(stdout, "# writing charged correlators to file %s\n", filename);
          fprintf(ofs, "# %3d%3d%3d%3d%10.6f%8.4f (%d,%d,%d) (%d,%d,%d)\n", T, LX, LY, LZ, g_kappa, g_mu,
              src_momentum[0], src_momentum[1], src_momentum[2], snk_momentum[0], snk_momentum[1], snk_momentum[2]);
          for(idx=0; idx<K; idx++)
          {
            // for(ll=0; ll<4; ll++)
            for(ll=0; ll<1; ll++)
            {
              x1 = (0+timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0, isneg[idx]*cconn[i]*correlator_norm, 0.);
              for(x0=1; x0<T_global/2; x0++) {
                x1 = ( x0+timeslice) % T_global;
                x2 = (-x0+timeslice+T_global) % T_global;
                i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
                j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + isimag[idx];
                //fprintf(stdout, "idx=%d; x0=%d, x1=%d, x2=%d, i=%d, j=%d\n", idx, x0, x1, x2, i, j);
                fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*cconn[i]*correlator_norm, isneg[idx]*cconn[j]*correlator_norm); 
              }
              x0 = T_global/2;
              x1 = (x0+timeslice) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*cconn[i]*correlator_norm, 0.);
              //for(x0=0; x0<T_global; x0++) {
              //  x1 = x0;
              //  i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T );
              //  fprintf(ofs, "%3d%3d%4d%25.16e%25.16e%3d%3d%3d\n", idx+1, 2*ll+1, x0, isneg[idx]*cconn[i]*correlator_norm, isneg[idx]*cconn[i+1]*correlator_norm,
              //      snk_momentum[0], snk_momentum[1], snk_momentum[2]); 
              //}
            }
          }   
          fclose(ofs);
      
          if(fermion_type==0) {
            if(g_source_type == 0) {
              sprintf(filename, "neutral.t%.2dx%.2dy%.2dz%.2d.%.4d", source_coords[0], source_coords[1], source_coords[2],
               source_coords[3], Nconf);
            } else {
              sprintf(filename, "neutral.%.2d.%.4d", timeslice, Nconf);
            }
            if(imom==0) {
              ofs=fopen(filename, "w");
            } else {
              ofs=fopen(filename, "a");
            }
            if( ofs  == (FILE*)NULL ) {
              fprintf(stderr, "Error, could not open file %s for writing\n", filename);
              EXIT(7);
            }
            fprintf(stdout, "# writing neutral correlators to file %s\n", filename);
            fprintf(ofs, "# %3d%3d%3d%3d%10.6f%8.4f (%d,%d,%d) (%d,%d,%d)\n", T, LX, LY, LZ, g_kappa, g_mu,
                src_momentum[0], src_momentum[1], src_momentum[2], snk_momentum[0], snk_momentum[1], snk_momentum[2]);
            for(idx=0; idx<nK; idx++)
            {
              // for(ll=0; ll<4; ll++)
              for(ll=0; ll<1; ll++)
              {
                x1 = (0+timeslice) % T_global;
                i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + nisimag[idx];
                fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0, isneg[idx]*nconn[i]*correlator_norm, 0.);
                for(x0=1; x0<T_global/2; x0++) {
                  x1 = ( x0+timeslice) % T_global;
                  x2 = (-x0+timeslice+T_global) % T_global;
                  i = 2* ( (x1/T)*4*nK*T + ll*nK*T + idx*T + x1%T ) + nisimag[idx];
                  j = 2* ( (x2/T)*4*nK*T + ll*nK*T + idx*T + x2%T ) + nisimag[idx];
                  fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*nconn[i]*correlator_norm, isneg[idx]*nconn[j]*correlator_norm); 
                }
                x0 = T_global/2;
                x1 = (x0+timeslice) % T_global;
                i = 2* ( (x1/T)*4*nK*T + ll*nK*T + idx*T + x1%T ) + nisimag[idx];
                fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0, isneg[idx]*nconn[i]*correlator_norm, 0.);
                //for(x0=0; x0<T_global; x0++) {
                //  x1 = x0;
                //  i = 2* ( (x1/T)*4*nK*T + ll*nK*T + idx*T + x1%T );
                //  fprintf(ofs, "%3d%3d%4d%25.16e%25.16e%3d%3d%3d\n", idx+1, 2*ll+1, x0, isneg[idx]*nconn[i]*correlator_norm, isneg[idx]*nconn[i+1]*correlator_norm,
                //    snk_momentum[0], snk_momentum[1], snk_momentum[2]); 
                //}
              }
            }    
            fclose(ofs);
          }
        }  // of if g_cart_id == 0
      }    // of loop on sink momenta
    }      // of loop on shift vectors

  /****************************************************
   * free the allocated memory, finalize
   ****************************************************/
  free(g_gauge_field); g_gauge_field=(double*)NULL;
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); g_spinor_field=(double**)NULL;
  free_geometry();
  free(cconn);
  free(nconn);
  free(Ctmp);
  free(gauge_field_f);

  finalize_q_orbits(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_rep);
  if(qlatt_map != NULL) {
    free(qlatt_map[0]);
    free(qlatt_map);
  }


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [cvc_2pt_conn] %s# [cvc_2pt_conn] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "[cvc_2pt_conn] %s[cvc_2pt_conn] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
