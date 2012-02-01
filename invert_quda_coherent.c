/****************************************************
 * invert_quda_coherent.c
 *
 * Mi 2. Nov 07:45:11 EET 2011
 *
 * PURPOSE:
 * - invert using the QUDA library provided by
 *   M. A. Clark, R. Babich, K. Barros, R. Brower, and C. Rebbi, "Solving
 *   Lattice QCD systems of equations using mixed precision solvers on GPUs,"
 *   Comput. Phys. Commun. 181, 1517 (2010) [arXiv:0911.3191 [hep-lat]].
 * - produce propagators from single 4-coherent timeslice sources with same random numbers in single and 4-coherent
 *   source
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include <getopt.h>
#ifdef OPENMP
#include <omp.h>
#endif

#define MAIN_PROGRAM

#ifdef __cplusplus
extern "C" {
#endif
#include "ifftw.h"
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "invert_Qtm.h"
#include "gauge_io.h"
#include "smearing_techniques.h"
#include "prepare_source.h"
#ifdef __cplusplus
}
#endif

// quda library
#include "quda.h"

void usage() {
  fprintf(stdout, "Code to invert D_tm\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  printf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, status;
  int ispin, icol, isc, isample;
  int n_c = 3;
  int n_s = 4;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, iy;
  int sl0, sl1, sl2, sl3, have_source_flag=0;
  int source_proc_coords[4], lsl0, lsl1, lsl2, lsl3, source_proc_id;
  int check_residuum = 0;
  int num_timeslices=0;
  int rng_continue=0;
  unsigned int VOL3;
  int do_gt   = 0;
  char filename[200];
  double ratime, retime;
  double plaq_r=0., plaq_m=0., norm, norm2;
  // double spinor1[24], spinor2[24];
  double *gauge_qdp[4], *gauge_field_timeslice=NULL, *gauge_field_smeared=NULL;
  double *coherent_source = NULL;
  double _1_2_kappa, _2_kappa;
  // FILE *ofs;
  int mu_trans[4] = {3, 0, 1, 2};
  int threadid, nthreads;
  int timeslice;
  char rng_file_in[100], rng_file_out[100];

  /***********************************************
   * QUDA parameters
   ***********************************************/
  QudaPrecision cpu_prec         = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec        = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec_sloppy = QUDA_SINGLE_PRECISION;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();


#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?vgf:r:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
    case 'g':
      do_gt = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_residuum = 1;
      fprintf(stdout, "# [] will check residuum again\n");
      break;
    case 'r':
      if(strcmp(optarg, "new")==0) {
        rng_continue = 0;
      } else if(strcmp(optarg, "continue")==0) {
        rng_continue = 1;
      }
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // get the time stamp
  g_the_time = time(NULL);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#endif

  /**************************************
   * set the default values, read input
   **************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_proc_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[invert_quda] Error, T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stderr, "[invert_quda] Error, kappa should be > 0.n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);
  
  // the volume of a timeslice
  VOL3 = LX*LY*LZ;

  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n",\
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart);

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


  /**************************************
   * initialize the QUDA library
   **************************************/
  fprintf(stdout, "# [invert_quda] initializing quda\n");
  initQuda(g_gpu_device_number);
  
  /**************************************
   * prepare the gauge field
   **************************************/
  // read the gauge field from file
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(strcmp( gaugefilename_prefix, "identity")==0 ) {
    if(g_cart_id==0) fprintf(stdout, "# [invert_quda] Setting up unit gauge field\n");
    for(ix=0;ix<VOLUME; ix++) {
      for(mu=0;mu<4;mu++) {
        _cm_eq_id(g_gauge_field+_GGI(ix,mu));
      }
    }
  } else {
    if(g_gauge_file_format == 0) {
      // ILDG
      sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "# Reading gauge field from file %s\n", filename);
      status = read_lime_gauge_field_doubleprec(filename);
    } else if(g_gauge_file_format == 1) {
      // NERSC
      sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "# Reading gauge field from file %s\n", filename);
      status = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
    }
    if(status != 0) {
      fprintf(stderr, "[invert_quda] Error, could not read gauge field");
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 12);
      MPI_Finalize();
#endif
      exit(12);
    }
  }
#ifdef MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq_m);
  if(g_cart_id==0) fprintf(stdout, "# Measured plaquette value: %25.16e\n", plaq_m);
  if(g_cart_id==0) fprintf(stdout, "# Read plaquette value    : %25.16e\n", plaq_r);

  // allocate the smeared / qdp ordered gauge field
  alloc_gauge_field(&gauge_field_smeared, VOLUME);
  for(i=0;i<4;i++) {
    gauge_qdp[i] = gauge_field_smeared + i*18*VOLUME;
  }


  // transcribe the gauge field
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel for private(ix,iy,mu)
#endif
  for(ix=0;ix<VOLUME;ix++) {
    iy = g_lexic2eot[ix];
    for(mu=0;mu<4;mu++) {
      _cm_eq_cm(gauge_qdp[mu_trans[mu]]+18*iy, g_gauge_field+_GGI(ix,mu));
    }
  }
  // multiply timeslice T-1 with factor of -1 (antiperiodic boundary condition)
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel for private(ix,iy)
#endif
  for(ix=0;ix<VOL3;ix++) {
    iix = (T-1)*VOL3 + ix;
    iy = g_lexic2eot[iix];
    _cm_ti_eq_re(gauge_qdp[mu_trans[0]]+18*iy, -1.);
  }


  // QUDA gauge parameters
  gauge_param.X[0] = LX_global;
  gauge_param.X[1] = LY_global;
  gauge_param.X[2] = LZ_global;
  gauge_param.X[3] = T_global;

  gauge_param.anisotropy  = 1.0;
  gauge_param.type        = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary  = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec           = cpu_prec;
  gauge_param.cuda_prec          = cuda_prec;
  gauge_param.reconstruct        = QUDA_RECONSTRUCT_12;
  gauge_param.cuda_prec_sloppy   = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
  gauge_param.gauge_fix          = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;

  // load the gauge field
  fprintf(stdout, "# [invert_quda] loading gauge field\n");
  loadGaugeQuda((void*)gauge_qdp, &gauge_param);
  gauge_qdp[0] = NULL; 
  gauge_qdp[1] = NULL; 
  gauge_qdp[2] = NULL; 
  gauge_qdp[3] = NULL; 

  /*********************************************
   * APE smear the gauge field
   *********************************************/
  memcpy(gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));
  if(N_ape>0) {
    fprintf(stdout, "# [] APE smearing gauge field with paramters N_APE=%d, alpha_APE=%e\n", N_ape, alpha_ape);
#ifdef OPENMP
     APE_Smearing_Step_threads(gauge_field_smeared, N_ape, alpha_ape);
#else
    for(i=0; i<N_ape; i++) {
       APE_Smearing_Step(gauge_field_smeared, alpha_ape);
     }
#endif
  }

  /* allocate memory for the spinor fields */
  no_fields = 4;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);
  coherent_source = g_spinor_field[no_fields - 1];

  /* the source locaton */
  sl0 =   g_source_location                              / (LX_global*LY_global*LZ);
  sl1 = ( g_source_location % (LX_global*LY_global*LZ) ) / (          LY_global*LZ);
  sl2 = ( g_source_location % (          LY_global*LZ) ) / (                    LZ);
  sl3 =   g_source_location %                      LZ;
  if(g_cart_id==0) fprintf(stdout, "# [invert_quda] global sl = (%d, %d, %d, %d)\n", sl0, sl1, sl2, sl3);
  source_proc_coords[0] = sl0 / T;
  source_proc_coords[1] = sl1 / LX;
  source_proc_coords[2] = sl2 / LY;
  source_proc_coords[3] = sl3 / LZ;
#ifdef MPI
  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
#else
  source_proc_id = 0;
#endif
  have_source_flag = source_proc_id == g_cart_id;

  lsl0 = sl0 % T;
  lsl1 = sl1 % LX;
  lsl2 = sl2 % LY;
  lsl3 = sl3 % LZ;
  if(have_source_flag) {
    fprintf(stdout, "# [invert_quda] process %d has the source at (%d, %d, %d, %d)\n", g_cart_id, lsl0, lsl1, lsl2, lsl3);
  }

  // QUDA inverter parameters
  inv_param.dslash_type    = QUDA_WILSON_DSLASH;
  inv_param.inv_type       = QUDA_BICGSTAB_INVERTER;
//  inv_param.inv_type       = QUDA_CG_INVERTER;
  inv_param.kappa          = g_kappa;
  inv_param.tol            = solver_precision;
  inv_param.maxiter        = niter_max;
  inv_param.reliable_delta = reliable_delta;

  inv_param.solution_type      = QUDA_MAT_SOLUTION;
  inv_param.solve_type         = QUDA_DIRECT_PC_SOLVE;
//  inv_param.solve_type         = QUDA_NORMEQ_PC_SOLVE;
  inv_param.matpc_type         = QUDA_MATPC_EVEN_EVEN; // QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger             = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION; //;QUDA_MASS_NORMALIZATION;

  inv_param.cpu_prec         = cpu_prec;
  inv_param.cuda_prec        = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source  = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order      = QUDA_DIRAC_ORDER;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.verbosity = QUDA_VERBOSE;

  strcpy(rng_file_out, g_rng_filename);
  strcpy(rng_file_in,  g_rng_filename);
  if(rng_continue == 0) {
    // write initial rng state to file
    // if(g_source_type==2 && g_coherent_source==2) {
      fprintf(stdout, "# [] writing initial rng state for file %s\n", rng_file_out);
      if( init_rng_stat_file (g_seed, rng_file_out) != 0 ) {
        fprintf(stderr, "[] Error, could not write rng status\n");
        exit(210);
      }
    // }
  }

  // number of time slices
  num_timeslices = T / g_coherent_source_delta;
  fprintf(stdout, "# [] number of time slices of coherent source = %d\n", num_timeslices);

  /***********************************************
   * loop on spin-color-index
   ***********************************************/
  // for(isample=0; isample<g_nsample; isample++) {
  for(isample=g_sourceid; isample<=g_sourceid2; isample+=g_sourceid_step) {
    // initialize the coherent source
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel for private (ix) shared(VOLUME)
#endif
    for(ix=0;ix<VOLUME;ix++) { _fv_eq_zero(coherent_source + _GSI(ix) ); }

    for(isc=0; isc <= num_timeslices; isc++) {
    /***********************************************
     * prepare the source
     ***********************************************/
      if(g_read_source==0) {
        switch(g_source_type) {
          case 2:
            if(isc<num_timeslices) {
              // timeslice source
              // strcpy(rng_file_in, rng_file_out);
              // if(isc == g_nsample) { strcpy(rng_file_out, g_rng_filename); }
              // else                 { sprintf(rng_file_out, "%s.%d", g_rng_filename, isc+1); }
              timeslice = (g_coherent_source_base+isc*g_coherent_source_delta)%T_global;
              fprintf(stdout, "# [] Creating timeslice source\n");
              status = prepare_timeslice_source(g_spinor_field[0], gauge_field_smeared, timeslice, VOLUME, rng_file_in, rng_file_out);
              if(status != 0) {
                fprintf(stderr, "[] Error from prepare source, status was %d\n", status);
                exit(123);
              }
              // add current source to coherent source
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel for private (ix) shared(VOLUME)
#endif
              for(ix=0;ix<VOLUME;ix++) { _fv_pl_eq_fv(coherent_source + _GSI(ix), g_spinor_field[0]+_GSI(ix) ); }
              if(g_coherent_source>1) {
                fprintf(stdout, "# [] will not invert single-timeslice sources; continue\n");
                continue;
              }
            } else {
              if(g_coherent_source > 0) {
                memcpy(g_spinor_field[0], coherent_source, 24*VOLUME*sizeof(double));
              } else {
                fprintf(stdout, "# [] will not produce coherent source; continue\n");
                continue;
              }
            }
            break;
          default:
            fprintf(stderr, "\nError, unrecognized source type\n");
            exit(32);
            break;
        }

        // printf_spinor_field(g_spinor_field[0], stdout);

        // smearing
        if(N_Jacobi > 0) {
#ifdef OPENMP
          Jacobi_Smearing_Step_one_threads(gauge_field_smeared, g_spinor_field[0], g_spinor_field[1], N_Jacobi, kappa_Jacobi);
#else
          for(c=0; c<N_Jacobi; c++) {
            Jacobi_Smearing_Step_one(gauge_field_smeared, g_spinor_field[0], g_spinor_field[1], kappa_Jacobi);
          }
#endif
        }

        if(g_write_source==1) {
          // write the source 
          switch(g_source_type) {
            case 2:
              if(isc<num_timeslices) {
                sprintf(filename, "%s.%.4d.%.2d.%.5d", filename_prefix, Nconf, timeslice, isample);
              } else {
                sprintf(filename, "%s.%.4d.%.2d.%.5d", filename_prefix2, Nconf, g_coherent_source_base, isample);
              }
              break;
          } 
          status = write_propagator(g_spinor_field[0], filename, 0, g_propagator_precision);
          if(status != 0) { 
            fprintf(stderr, "Error from write_propagator, status was %d\n", status);
            exit(23);
          }
        }
      } else { // of if read source == 0
        switch(g_source_type) {
          case 2:
            if(isc<num_timeslices) {
              timeslice = (g_coherent_source_base+isc*g_coherent_source_delta)%T_global;
              sprintf(filename, "%s.%.4d.%.2d.%.5d", filename_prefix, Nconf, timeslice, isample);
            } else {
              if(g_coherent_source > 0) {
                sprintf(filename, "%s.%.4d.%.2d.%.5d", filename_prefix2, Nconf, g_coherent_source_base, isample);
              } else {
                fprintf(stdout, "# [] will not read coherent source; continue\n");
                continue;
              }
            }
            break;
        }
        fprintf(stdout, "# [] reading source from file %s\n", filename);
        status = read_lime_spinor(g_spinor_field[0], filename, 0);
        if(status != 0) { 
          fprintf(stderr, "Error from write_propagator, status was %d\n", status);
          exit(23);
        }
      } // else of if read source == 0

      // multiply with g2
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel for private (ix) shared(VOLUME)
#endif
      for(ix=0;ix<VOLUME;ix++) {
        _fv_eq_gamma_ti_fv(g_spinor_field[1]+_GSI(ix), 2, g_spinor_field[0]+_GSI(ix));
      }

      // transcribe the spinor field to even-odd ordering with coordinates (x,y,z,t)
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel for private (ix,iy) shared(VOLUME)
#endif
      for(ix=0;ix<VOLUME;ix++) {
        iy = g_lexic2eot[ix];
        _fv_eq_fv(g_spinor_field[2]+_GSI(iy), g_spinor_field[1]+_GSI(ix));
      }


      /***********************************************
       * perform the inversion
      ***********************************************/
      fprintf(stdout, "# [invert_quda] starting inversion\n");
      ratime = (double)clock() / CLOCKS_PER_SEC;

      for(ix=0;ix<VOLUME;ix++) {
        _fv_eq_zero(g_spinor_field[1]+_GSI(ix) );
      }

      invertQuda(g_spinor_field[1], g_spinor_field[2], &inv_param);

      retime = (double)clock() / CLOCKS_PER_SEC;
      fprintf(stdout, "# [invert_quda] inversion done in %e seconds\n", retime-ratime);
      fprintf(stdout, "# [invert_quda] Device memory used:\n\tSpinor: %f GiB\n\tGauge: %f GiB\n",
        inv_param.spinorGiB, gauge_param.gaugeGiB);

      if(inv_param.mass_normalization == QUDA_KAPPA_NORMALIZATION) {
        _2_kappa = 2. * g_kappa;
        for(ix=0;ix<VOLUME;ix++) {
          _fv_ti_eq_re(g_spinor_field[1]+_GSI(ix), _2_kappa );
        }
      }

      // transcribe the spinor field to lexicographical order with (t,x,y,z)
      for(ix=0;ix<VOLUME;ix++) {
        iy = g_lexic2eot[ix];
        _fv_eq_fv(g_spinor_field[2]+_GSI(ix), g_spinor_field[1]+_GSI(iy));
      }
      // multiply with g2
      for(ix=0;ix<VOLUME;ix++) {
        _fv_eq_gamma_ti_fv(g_spinor_field[1]+_GSI(ix), 2, g_spinor_field[2]+_GSI(ix));
      }

      /***********************************************
       * check residuum
       ***********************************************/
      if(check_residuum) {
        // apply the Wilson Dirac operator in the gamma-basis defined in cvc_linalg,
        //   which uses the tmLQCD conventions (same as in contractions)
        //   without explicit boundary conditions
        Q_Wilson_phi(g_spinor_field[2], g_spinor_field[1]);

        for(ix=0;ix<VOLUME;ix++) {
          _fv_mi_eq_fv(g_spinor_field[2]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
        }

        spinor_scalar_product_re(&norm, g_spinor_field[2], g_spinor_field[2], VOLUME);
        spinor_scalar_product_re(&norm2, g_spinor_field[0], g_spinor_field[0], VOLUME);
        fprintf(stdout, "\n# [] absolut residuum squared: %e; relative residuum %e\n", norm, sqrt(norm/norm2) );
      }

      // write the solution
      switch(g_source_type) {
        case 2:
          if(isc<num_timeslices) {
            sprintf(filename, "%s.%.4d.%.2d.%.5d.inverted", filename_prefix, Nconf, timeslice, isample);
          } else {
            sprintf(filename, "%s.%.4d.%.2d.%.5d.inverted", filename_prefix2, Nconf, g_coherent_source_base, isample);
          }
          break;
      } 
      status = write_propagator(g_spinor_field[1], filename, 0, g_propagator_precision);
      if(status != 0) {
        fprintf(stderr, "Error from write_propagator, status was %d\n", status);
        exit(22);
      }

    }  // of isc
  }    // of isample

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  // finalize the QUDA library
  fprintf(stdout, "# [invert_quda] finalizing quda\n");
  endQuda();

  free(g_gauge_field);
  free(gauge_field_smeared);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();

#ifdef MPI
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [invert_quda] %s# [invert_quda] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [invert_quda] %s# [invert_quda] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
