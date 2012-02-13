/****************************************************
 *
 * invert_dw_quda.c
 *
 * Mon Jan 30 20:13:38 EET 2012
 *
 * PURPOSE:
 * - invert using the QUDA library provided by
 *   M. A. Clark, R. Babich, K. Barros, R. Brower, and C. Rebbi, "Solving
 *   Lattice QCD systems of equations using mixed precision solvers on GPUs,"
 *   Comput. Phys. Commun. 181, 1517 (2010) [arXiv:0911.3191 [hep-lat]].
 * - use QUDA inverter for domain wall fermions
 * - ATTENTION: for MPI usage the parameter kernelPackT must be set to true
 * TODO:
 * -finish and test MPI implementation with code from Alexei
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

#ifndef EXIT
#  ifdef MPI
#  define EXIT(_i) { MPI_Abort(MPI_COMM_WORLD, (_i)); MPI_Finalize(); exit((_i)); }
#  else
#  define EXIT(_i) { exit(_i); }
#  endif
#endif

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
#include "make_q_orbits.h"

#ifdef __cplusplus
}
#endif
#ifdef HAVE_QUDA
// quda library
#include "quda.h"
#endif


#ifdef MPI
#define CLOCK MPI_Wtime()
#else
#define CLOCK ((double)clock() / CLOCKS_PER_SEC)
#endif

void usage() {
  fprintf(stdout, "Code to invert D_tm\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  EXIT(0);
}

//extern bool kernelPackT;

int main(int argc, char **argv) {
  
  int c, i, mu, status;
  int ispin, icol, isc;
  int n_c = 3;
  int n_s = 4;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int grid_size[4];
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, iy, is;
  int sl0, sl1, sl2, sl3, have_source_flag=0;
  int source_proc_coords[4], lsl0, lsl1, lsl2, lsl3;
  int check_residuum = 0;
  unsigned int VOL3, V5;
  int do_gt   = 0;
  int full_orbit = 0;
  int smear_source = 0;
  char filename[200], source_filename[200];
  double ratime, retime;
  double plaq_r=0., plaq_m=0., norm, norm2;
  // double spinor1[24], spinor2[24];
  double *gauge_qdp[4], *gauge_field_timeslice=NULL, *gauge_field_smeared=NULL;
  double _1_2_kappa, _2_kappa, phase;
  FILE *ofs;
  int mu_trans[4] = {3, 0, 1, 2};
  int threadid, nthreads;
  int timeslice, source_timeslice;
  char rng_file_in[100], rng_file_out[100];
  int *source_momentum=NULL;
  int source_momentum_class = -1;
  int source_momentum_no = 0;
  int source_momentum_runs = 1;
  int imom;
  int num_gpu_on_node=0, rank;
  int source_location_5d_iseven;

  /****************************************************************************/
#if (defined HAVE_QUDA) && (defined MULTI_GPU)
  int x_face_size, y_face_size, z_face_size, t_face_size, pad_size;
#endif
  /****************************************************************************/

  /************************************************/
  int qlatt_nclass;
  int *qlatt_id=NULL, *qlatt_count=NULL, **qlatt_rep=NULL, **qlatt_map=NULL;
  double **qlatt_list=NULL;
  /************************************************/
//#ifdef MPI       
//  kernelPackT = true;
//#endif

  /***********************************************
   * QUDA parameters
   ***********************************************/
#ifdef HAVE_QUDA
  QudaPrecision cpu_prec         = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec        = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec_sloppy = QUDA_SINGLE_PRECISION;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
#endif

  while ((c = getopt(argc, argv, "soch?vgf:p:")) != -1) {
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
      fprintf(stdout, "# [invert_dw_quda] will check residuum again\n");
      break;
    case 'p':
      n_c = atoi(optarg);
      fprintf(stdout, "# [invert_dw_quda] will use number of colors = %d\n", n_c);
      break;
    case 'o':
      full_orbit = 1;
      fprintf(stdout, "# [invert_dw_quda] will invert for full orbit, if source momentum set\n");
    case 's':
      smear_source = 1;
      fprintf(stdout, "# [invert_dw_quda] will smear the sources if they are read from file\n");
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



#ifdef MPI
#ifdef HAVE_QUDA
  grid_size[0] = g_nproc_x;
  grid_size[1] = g_nproc_y;
  grid_size[2] = g_nproc_z;
  grid_size[3] = g_nproc_t;
  fprintf(stdout, "# [] g_nproc = (%d,%d,%d,%d)\n", g_nproc_x, g_nproc_y, g_nproc_z, g_nproc_t);
  initCommsQuda(argc, argv, grid_size, 4);
#else
  MPI_Init(&argc, &argv);
#endif
#endif


  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[invert_dw_quda] Error, T and L's must be set\n");
    usage();
  }

  // initialize MPI parameters
  mpi_init(argc, argv);
  
  // the volume of a timeslice
  VOL3 = LX*LY*LZ;
  V5   = T*LX*LY*LZ*L5;
  g_kappa5d = 0.5 / (5. + g_m5);
  if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] kappa5d = %e\n", g_kappa5d);

  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] L5           = %3d\n",\
                  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, L5);


#ifdef MPI
  if(T==0) {
    fprintf(stderr, "[%2d] local T is zero; exit\n", g_cart_id);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(2);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[invert_dw_quda] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();

  if( init_geometry_5d() != 0 ) {
    fprintf(stderr, "[invert_dw_quda] Error from init_geometry_5d\n");
    EXIT(2);
  }
  geometry_5d();

  /**************************************
   * initialize the QUDA library
   **************************************/
  if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] initializing quda\n");
#ifdef HAVE_QUDA
  // cudaGetDeviceCount(&num_gpu_on_node);
  if(g_gpu_per_node<0) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, number of GPUs per node not set\n");
    EXIT(106);
  } else {
    num_gpu_on_node = g_gpu_per_node;
  }
#ifdef MPI
  rank = comm_rank();
#else
  rank = 0;
#endif
  g_gpu_device_number = rank % num_gpu_on_node;
  fprintf(stdout, "# [] process %d/%d uses device %d\n", rank, g_cart_id, g_gpu_device_number);

  initQuda(g_gpu_device_number);

#endif
 
  /**************************************
   * prepare the gauge field
   **************************************/
  // read the gauge field from file
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(strcmp( gaugefilename_prefix, "identity")==0 ) {
    if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] Setting up unit gauge field\n");
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
      fprintf(stderr, "[invert_dw_quda] Error, could not read gauge field");
      EXIT(12);
    }
  }
#ifdef MPI
  xchange_gauge();
#endif

  // measure the plaquette
  plaquette(&plaq_m);
  if(g_cart_id==0) fprintf(stdout, "# Measured plaquette value: %25.16e\n", plaq_m);
  if(g_cart_id==0) fprintf(stdout, "# Read plaquette value    : %25.16e\n", plaq_r);

#ifndef HAVE_QUDA
  if(N_Jacobi>0) {
#endif
    // allocate the smeared / qdp ordered gauge field
    alloc_gauge_field(&gauge_field_smeared, VOLUMEPLUSRAND);
    for(i=0;i<4;i++) {
      gauge_qdp[i] = gauge_field_smeared + i*18*VOLUME;
    }
#ifndef HAVE_QUDA
  }
#endif

#ifdef HAVE_QUDA
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
  if(g_proc_coords[0]==g_nproc_t-1) {
    fprintf(stdout, "# [] process %d multiplies gauge-field timeslice T_global-1 with -1\n", g_cart_id);
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel for private(ix,iy)
#endif
    for(ix=0;ix<VOL3;ix++) {
      iix = (T-1)*VOL3 + ix;
      iy = g_lexic2eot[iix];
      _cm_ti_eq_re(gauge_qdp[mu_trans[0]]+18*iy, -1.);
    }
  }

  // QUDA precision parameters
  switch(g_cpu_prec) {
    case 0: cpu_prec = QUDA_HALF_PRECISION; break;
    case 1: cpu_prec = QUDA_SINGLE_PRECISION; break;
    case 2: cpu_prec = QUDA_DOUBLE_PRECISION; break;
    default: cpu_prec = QUDA_DOUBLE_PRECISION; break;
  }
  switch(g_gpu_prec) {
    case 0: cuda_prec = QUDA_HALF_PRECISION; break;
    case 1: cuda_prec = QUDA_SINGLE_PRECISION; break;
    case 2: cuda_prec = QUDA_DOUBLE_PRECISION; break;
    default: cuda_prec = QUDA_DOUBLE_PRECISION; break;
  }
  switch(g_gpu_prec_sloppy) {
    case 0: cuda_prec_sloppy = QUDA_HALF_PRECISION; break;
    case 1: cuda_prec_sloppy = QUDA_SINGLE_PRECISION; break;
    case 2: cuda_prec_sloppy = QUDA_DOUBLE_PRECISION; break;
    default: cuda_prec_sloppy = QUDA_SINGLE_PRECISION; break;
  }

  // QUDA gauge parameters
  gauge_param.X[0] = LX;
  gauge_param.X[1] = LY;
  gauge_param.X[2] = LZ;
  gauge_param.X[3] = T;
  inv_param.Ls = L5;

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
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  x_face_size = inv_param.Ls * gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  y_face_size = inv_param.Ls * gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  z_face_size = inv_param.Ls * gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  t_face_size = inv_param.Ls * gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  pad_size = _MAX(x_face_size, y_face_size);
  pad_size = _MAX(pad_size, z_face_size);
  pad_size = _MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;
#endif

  // load the gauge field
  if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] loading gauge field\n");
  loadGaugeQuda((void*)gauge_qdp, &gauge_param);
  gauge_qdp[0] = NULL; 
  gauge_qdp[1] = NULL; 
  gauge_qdp[2] = NULL; 
  gauge_qdp[3] = NULL; 

#endif

  /*********************************************
   * APE smear the gauge field
   *********************************************/
  if(N_Jacobi>0) {
    memcpy(gauge_field_smeared, g_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));
    fprintf(stdout, "# [invert_dw_quda] APE smearing gauge field with paramters N_APE=%d, alpha_APE=%e\n", N_ape, alpha_ape);
#ifdef OPENMP
     APE_Smearing_Step_threads(gauge_field_smeared, N_ape, alpha_ape);
#else
    for(i=0; i<N_ape; i++) {
       APE_Smearing_Step(gauge_field_smeared, alpha_ape);
     }
#endif
    xchange_gauge_field(gauge_field_smeared);
  }

  // allocate memory for the spinor fields
#ifdef HAVE_QUDA
  no_fields = 3;
#else
  no_fields = 9;
#endif
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND*L5);

  switch(g_source_type) {
    case 0:
      // the source locaton
      sl0 =   g_source_location                              / (LX_global*LY_global*LZ);
      sl1 = ( g_source_location % (LX_global*LY_global*LZ) ) / (          LY_global*LZ);
      sl2 = ( g_source_location % (          LY_global*LZ) ) / (                    LZ);
      sl3 =   g_source_location %                      LZ;
      if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] global sl = (%d, %d, %d, %d)\n", sl0, sl1, sl2, sl3);
      source_proc_coords[0] = sl0 / T;
      source_proc_coords[1] = sl1 / LX;
      source_proc_coords[2] = sl2 / LY;
      source_proc_coords[3] = sl3 / LZ;
    #ifdef MPI
      MPI_Cart_rank(g_cart_grid, source_proc_coords, &g_source_proc_id);
    #else
      g_source_proc_id = 0;
    #endif
      have_source_flag = g_source_proc_id == g_cart_id;
    
      lsl0 = sl0 % T;
      lsl1 = sl1 % LX;
      lsl2 = sl2 % LY;
      lsl3 = sl3 % LZ;
      if(have_source_flag) {
        fprintf(stdout, "# [invert_dw_quda] process %d has the source at (%d, %d, %d, %d)\n", g_cart_id, lsl0, lsl1, lsl2, lsl3);
      }
      break;
    case 2:
    case 3:
    case 4:
      // the source timeslice
#ifdef MPI
      source_proc_coords[0] = g_source_timeslice / T;
      source_proc_coords[1] = 0;
      source_proc_coords[2] = 0;
      source_proc_coords[3] = 0;
      MPI_Cart_rank(g_cart_grid, source_proc_coords, &g_source_proc_id);
      have_source_flag = ( g_source_proc_id == g_cart_id );
      source_timeslice = have_source_flag ? g_source_timeslice % T : -1;
#else
      g_source_proc_id = 0;
      have_source_flag = 1;
      source_timeslice = g_source_timeslice;
#endif
      break;
  }

#ifdef HAVE_QUDA
  /*************************************************************
   * QUDA inverter parameters
   *************************************************************/
  inv_param.dslash_type    = QUDA_DOMAIN_WALL_DSLASH;

  if(strcmp(g_inverter_type_name, "cg") == 0) {
    inv_param.inv_type       = QUDA_CG_INVERTER;
    if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] using cg inverter\n"); 
  } else if(strcmp(g_inverter_type_name, "bicgstab") == 0) {
    inv_param.inv_type       = QUDA_BICGSTAB_INVERTER;
    if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] using bicgstab inverter\n"); 
  } else if(strcmp(g_inverter_type_name, "gcr") == 0) {
    inv_param.inv_type       = QUDA_GCR_INVERTER;
    if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] using gcr inverter\n"); 
  } else {
    if(g_cart_id==0) fprintf(stderr, "[invert_dw_quda] Error, unrecognized inverter type %s\n", g_inverter_type_name);
    EXIT(123);
  }


  if(inv_param.inv_type == QUDA_CG_INVERTER) {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.solve_type    = QUDA_NORMEQ_PC_SOLVE;
  } else if(inv_param.inv_type == QUDA_BICGSTAB_INVERTER) {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.solve_type    = QUDA_DIRECT_PC_SOLVE;
  } else {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
    inv_param.solve_type    = QUDA_DIRECT_PC_SOLVE;
  }

  inv_param.m5             = g_m5;
  inv_param.kappa          = 0.5 / (5. + inv_param.m5);
  inv_param.mass           = g_m0;

  inv_param.tol            = solver_precision;
  inv_param.maxiter        = niter_max;
  inv_param.reliable_delta = reliable_delta;

#ifdef MPI
  // domain decomposition preconditioner parameters
  if(inv_param.inv_type == QUDA_GCR_INVERTER) {
    inv_param.gcrNkrylov     = 15;
    inv_param.inv_type_precondition = QUDA_MR_INVERTER;
    inv_param.tol_precondition = 1e-6;
    inv_param.maxiter_precondition = 200;
    inv_param.verbosity_precondition = QUDA_VERBOSE;
    inv_param.prec_precondition = cuda_prec_sloppy;
    inv_param.omega = 0.7;
  }
#endif

  inv_param.matpc_type         = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger             = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION; //;QUDA_MASS_NORMALIZATION;

  inv_param.cpu_prec         = cpu_prec;
  inv_param.cuda_prec        = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.verbosity = QUDA_VERBOSE;

  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
#ifdef MPI
  inv_param.preserve_dirac = QUDA_PRESERVE_DIRAC_YES;
  inv_param.prec_precondition = cuda_prec_sloppy;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_tune = QUDA_TUNE_NO;
#endif
 
  //set the T dimension partitioning flag
//  commDimPartitionedSet(3);
#endif

  // write initial rng state to file
  if( g_source_type==2 && g_coherent_source==2 ) {
    sprintf(rng_file_out, "%s.0", g_rng_filename);
    status = init_rng_stat_file (g_seed, rng_file_out);
    if( status != 0 ) {
      fprintf(stderr, "[invert_dw_quda] Error, could not write rng status\n");
      EXIT(210);
    }
  } else if( (g_source_type==2 && g_coherent_source==1) || g_source_type==3 || g_source_type==4) {
    if( init_rng_state(g_seed, &g_rng_state) != 0 ) {
      fprintf(stderr, "[invert_dw_quda] Error, could initialize rng state\n");
      EXIT(211);
    }
  }

  // check the source momenta
  if(g_source_momentum_set) {
    source_momentum = (int*)malloc(3*sizeof(int));

    if(g_source_momentum[0]<0) g_source_momentum[0] += LX_global;
    if(g_source_momentum[1]<0) g_source_momentum[1] += LY_global;
    if(g_source_momentum[2]<0) g_source_momentum[2] += LZ_global;
    fprintf(stdout, "# [invert_dw_quda] using final source momentum ( %d, %d, %d )\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);


    if(full_orbit) {
      status = make_qcont_orbits_3d_parity_avg( &qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
      if(status != 0) {
        if(g_cart_id==0) fprintf(stderr, "\n[invert_dw_quda] Error while creating O_3-lists\n");
        EXIT(4);
      }
      source_momentum_class = qlatt_id[g_ipt[0][g_source_momentum[0]][g_source_momentum[1]][g_source_momentum[2]]];
      source_momentum_no    = qlatt_count[source_momentum_class];
      source_momentum_runs  = source_momentum_class==0 ? 1 : source_momentum_no + 1;
      if(g_cart_id==0) fprintf(stdout, "# [] source momentum belongs to class %d with %d members, which means %d runs\n",
          source_momentum_class, source_momentum_no, source_momentum_runs);
    }
  }


  /***********************************************
   * loop on spin-color-index
   ***********************************************/
  for(isc=g_source_index[0]; isc<=g_source_index[1]; isc++)
//  for(isc=g_source_index[0]; isc<=g_source_index[0]; isc++)
  {
    ispin = isc / n_c;
    icol  = isc % n_c;

    for(imom=0; imom<source_momentum_runs; imom++) {

      /***********************************************
       * set source momentum
       ***********************************************/
      if(g_source_momentum_set) {
        if(imom == 0) {
          if(full_orbit) {
            source_momentum[0] = 0;
            source_momentum[1] = 0;
            source_momentum[2] = 0;
          } else {
            source_momentum[0] = g_source_momentum[0];
            source_momentum[1] = g_source_momentum[1];
            source_momentum[2] = g_source_momentum[2];
          }
        } else {
          source_momentum[0] = qlatt_map[source_momentum_class][imom-1] / (LY_global*LZ_global);
          source_momentum[1] = ( qlatt_map[source_momentum_class][imom-1] % (LY_global*LZ_global) ) / LZ_global;
          source_momentum[2] = qlatt_map[source_momentum_class][imom-1] % LZ_global;
        }
        if(g_cart_id==0) fprintf(stdout, "# [] run no. %d, source momentum (%d, %d, %d)\n",
            imom, source_momentum[0], source_momentum[1], source_momentum[2]);
      
      }
 
      /***********************************************
       * prepare the souce
       ***********************************************/
      if(g_read_source == 0) {  // create source
        switch(g_source_type) {
          case 0:
            // point source
            if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] Creating point source\n");
            for(ix=0;ix<L5*VOLUME;ix++) { _fv_eq_zero(g_spinor_field[0]+ix); }
            if(have_source_flag) {
              if(g_source_momentum_set) {
                phase = 2*M_PI*( source_momentum[0]*sl1/(double)LX_global + source_momentum[1]*sl2/(double)LY_global + source_momentum[2]*sl3/(double)LZ_global );
                g_spinor_field[0][_GSI(g_source_location) + 2*(n_c*ispin+icol)  ] = cos(phase);
                g_spinor_field[0][_GSI(g_source_location) + 2*(n_c*ispin+icol)+1] = sin(phase);
              } else {
                g_spinor_field[0][_GSI(g_source_location) + 2*(n_c*ispin+icol)  ] = 1.;
              }
            }
            if(g_source_momentum_set) {
              sprintf(source_filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d",
                  filename_prefix, Nconf, sl0, sl1, sl2, sl3, n_c*ispin+icol, source_momentum[0], source_momentum[1], source_momentum[2]);
            } else {
              sprintf(source_filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d", filename_prefix, Nconf, sl0, sl1, sl2, sl3, n_c*ispin+icol);
            }
#ifdef HAVE_QUDA
            // set matpc_tpye
            source_location_5d_iseven = ( (g_iseven[g_source_location] && ispin<n_s/2) || (!g_iseven[g_source_location] && ispin>=n_s/2) ) ? 1 : 0;
            if(source_location_5d_iseven) {
              inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
              if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] matpc type is MATPC_EVEN_EVEN\n");
            } else {
              inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
              if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] matpc type is MATPC_ODD_ODD\n");
            }
#endif
            break;
          case 2:
            // timeslice source
            if(g_coherent_source==1) {
              if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] Creating coherent timeslice source\n");
              status = prepare_coherent_timeslice_source(g_spinor_field[0], gauge_field_smeared, g_coherent_source_base, g_coherent_source_delta, VOLUME, g_rng_state, 1);
              if(status != 0) {
                fprintf(stderr, "[invert_dw_quda] Error from prepare source, status was %d\n", status);
#ifdef MPI
                MPI_Abort(MPI_COMM_WORLD, 123);
                MPI_Finalize();
#endif
                exit(123);
              }
              check_error(prepare_coherent_timeslice_source(g_spinor_field[0], gauge_field_smeared, g_coherent_source_base, g_coherent_source_delta, VOLUME, g_rng_state, 1),
                  "prepare_coherent_timeslice_source", NULL, 123);
              timeslice = g_coherent_source_base;
            } else {
              if(g_coherent_source==2) {
                timeslice = (g_coherent_source_base+isc*g_coherent_source_delta)%T_global;
                fprintf(stdout, "# [invert_dw_quda] Creating timeslice source\n");
                check_error(prepare_timeslice_source(g_spinor_field[0], gauge_field_smeared, timeslice, VOLUME, g_rng_state, 1),
                    "prepare_timeslice_source", NULL, 123);
              } else {
                if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] Creating timeslice source\n");
                check_error(prepare_timeslice_source(g_spinor_field[0], gauge_field_smeared, g_source_timeslice, VOLUME, g_rng_state, 1),
                    "prepare_timeslice_source", NULL, 124);
                timeslice = g_source_timeslice;
              }
            }
            if(g_source_momentum_set) {
              sprintf(source_filename, "%s.%.4d.%.2d.%.5d.qx%.2dqy%.2dqz%.2d", filename_prefix, Nconf, 
                  timeslice, isc, source_momentum[0], source_momentum[1], source_momentum[2]);
            } else {
              sprintf(source_filename, "%s.%.4d.%.2d.%.5d", filename_prefix, Nconf, timeslice, isc);
            }
            break;
          case 3:
            // timeslice sources for one-end trick (spin dilution)
            fprintf(stdout, "# [invert_dw_quda] Creating timeslice source for one-end-trick\n");
            check_error( prepare_timeslice_source_one_end(g_spinor_field[0], gauge_field_smeared, source_timeslice, source_momentum, isc%n_s, g_rng_state, \
                ( isc%n_s==(n_s-1) && imom==source_momentum_runs-1 )), "prepare_timeslice_source_one_end", NULL, 125 );
            c = N_Jacobi > 0 ? isc%n_s + n_s : isc%n_s;
            if(g_source_momentum_set) {
              sprintf(source_filename, "%s.%.4d.%.2d.%.2d.qx%.2dqy%.2dqz%.2d", filename_prefix, Nconf, 
                  g_source_timeslice, c, source_momentum[0], source_momentum[1], source_momentum[2]);
            } else {
              sprintf(source_filename, "%s.%.4d.%.2d.%.2d", filename_prefix, Nconf, g_source_timeslice, c);
            }
            break;
          case 4:
            // timeslice sources for one-end trick (spin and color dilution )
            fprintf(stdout, "# [invert_dw_quda] Creating timeslice source for one-end-trick\n");
            check_error(prepare_timeslice_source_one_end_color(g_spinor_field[0], gauge_field_smeared, source_timeslice, source_momentum,\
                isc%(n_s*n_c), g_rng_state, ( isc%(n_s*n_c)==(n_s*n_c-1)  && imom==source_momentum_runs-1 )), "prepare_timeslice_source_one_end_color", NULL, 126);
            c = N_Jacobi > 0 ? isc%(n_s*n_c) + (n_s*n_c) : isc%(n_s*n_c);
            if(g_source_momentum_set) {
              sprintf(source_filename, "%s.%.4d.%.2d.%.2d.qx%.2dqy%.2dqz%.2d", filename_prefix, Nconf, 
                  g_source_timeslice, c, source_momentum[0], source_momentum[1], source_momentum[2]);
            } else {
              sprintf(source_filename, "%s.%.4d.%.2d.%.2d", filename_prefix, Nconf, g_source_timeslice, c);
            }
            break;
          default:
            fprintf(stderr, "\nError, unrecognized source type\n");
            exit(32);
            break;
        }
      } else { // read source
        switch(g_source_type) {
          case 0:  // point source
            if(g_source_momentum_set) {
              sprintf(source_filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d", \
                  filename_prefix2, Nconf, sl0, sl1, sl2, sl3, isc, source_momentum[0], source_momentum[1], source_momentum[2]);
            } else  {
              sprintf(source_filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d", filename_prefix2, Nconf, sl0, sl1, sl2, sl3, isc);
            }
            fprintf(stdout, "# [invert_dw_quda] reading source from file %s\n", source_filename);
            check_error(read_lime_spinor(g_spinor_field[0], source_filename, 0), "read_lime_spinor", NULL, 115);
            break;
          case 2:  // timeslice source
            if(g_source_momentum_set) {
              sprintf(source_filename, "%s.%.4d.%.2d.%.5d.qx%.2dqy%.2dqz%.2d", filename_prefix2, Nconf, g_source_timeslice,
                  isc, source_momentum[0], source_momentum[1], source_momentum[2]);
            } else {
              sprintf(source_filename, "%s.%.4d.%.2d.%.5d", filename_prefix2, Nconf, g_source_timeslice, isc);
            }
            fprintf(stdout, "# [invert_dw_quda] reading source from file %s\n", source_filename);
            check_error(read_lime_spinor(g_spinor_field[0], source_filename, 0), "read_lime_spinor", NULL, 115);
            break;
          default:
            check_error(1, "source type", NULL, 104);
            break;
        }
      }  // of if g_read_source
  
      //sprintf(filename, "%s.ascii.%.2d", source_filename, g_cart_id);
      //ofs = fopen(filename, "w");
      //printf_spinor_field(g_spinor_field[0], ofs);
      //fclose(ofs);
  
      if(g_write_source) {
        check_error(write_propagator(g_spinor_field[0], source_filename, 0, g_propagator_precision), "write_propagator", NULL, 27);
      }
  
      // smearing
      if(!g_read_source || (g_read_source && smear_source ) ) {
        if(g_cart_id==0) fprintf(stdout, "#  [invert_dw_quda] smearing source with N_Jacobi=%d, kappa_Jacobi=%e\n", N_Jacobi, kappa_Jacobi);
        if(N_Jacobi > 0) {
#ifdef OPENMP
          Jacobi_Smearing_Step_one_threads(gauge_field_smeared, g_spinor_field[0], g_spinor_field[1], N_Jacobi, kappa_Jacobi);
#else
          for(c=0; c<N_Jacobi; c++) {
            Jacobi_Smearing_Step_one(gauge_field_smeared, g_spinor_field[0], g_spinor_field[1], kappa_Jacobi);
          }
#endif
        }
      }
      xchange_field(g_spinor_field[0]);

      /***********************************************
       * create the 5-dim. source field
       ***********************************************/
      spinor_4d_to_5d(g_spinor_field[0], g_spinor_field[0]);
      xchange_field_5d(g_spinor_field[0]);

      //sprintf(filename, "%s.ascii.5d.%.2d", source_filename, g_cart_id);
      //ofs = fopen(filename, "w");
      //printf_spinor_field_5d(g_spinor_field[0], ofs);
      //fclose(ofs);

      
#ifdef HAVE_QUDA  
      // reorder, multiply with g2
      for(ix=0;ix<L5*VOLUME;ix++) {
        _fv_eq_zero(g_spinor_field[2]+_GSI(ix));
      }
      for(ix=0;ix<VOLUME;ix++) {
        iy = lexic2eot_5d(0, ix);
        _fv_eq_gamma_ti_fv(g_spinor_field[2]+_GSI(iy), 2, g_spinor_field[0]+_GSI(ix));
      }
      for(ix=0;ix<VOLUME;ix++) {
        iy = lexic2eot_5d(L5-1, ix);
        _fv_eq_gamma_ti_fv(g_spinor_field[2]+_GSI(iy), 2, g_spinor_field[0]+_GSI(ix+(L5-1)*VOLUME));
      }
#endif

      /***********************************************
       * perform the inversion
       ***********************************************/
      if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] starting inversion\n");

#ifdef HAVE_QUDA 
      //for(ix=0;ix<24*VOLUME*L5;ix++) g_spinor_field[2][ix] = 1.;
      //sprintf(filename, "source.ascii.5d.%.2d.%.2d", g_nproc, g_cart_id);
      //ofs = fopen(filename, "w");
      //printf_spinor_field_5d(g_spinor_field[2], ofs);
      //fclose(ofs);

      for(ix=0;ix<VOLUME*L5;ix++) {
        _fv_eq_zero(g_spinor_field[1]+_GSI(ix) );
      }
      xchange_field_5d(g_spinor_field[1]);
      ratime = CLOCK;
#ifdef MPI
      if(inv_param.inv_type == QUDA_BICGSTAB_INVERTER  || inv_param.inv_type == QUDA_GCR_INVERTER) {
        if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] calling invertQuda\n");
        invertQuda(g_spinor_field[1], g_spinor_field[2], &inv_param);
      } else if(inv_param.inv_type == QUDA_CG_INVERTER) {
        if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] calling testCG\n");
        testCG(g_spinor_field[1], g_spinor_field[2], &inv_param);
      } else {
        if(g_cart_id==0) fprintf(stderr, "# [invert_dw_quda] unrecognized inverter\n");
      }
#else
      invertQuda(g_spinor_field[1], g_spinor_field[2], &inv_param);
#endif
      retime = CLOCK;

      if(g_cart_id==0) {
        fprintf(stdout, "# [invert_dw_quda] inversion done in %e seconds\n", retime-ratime);
        fprintf(stdout, "# [invert_dw_quda] Device memory used:\n\tSpinor: %f GiB\n\tGauge: %f GiB\n",
        inv_param.spinorGiB, gauge_param.gaugeGiB);
      }
  
      //sprintf(filename, "prop.ascii.5d.%.2d.%.2d", g_nproc, g_cart_id);
      //ofs = fopen(filename, "w");
      //printf_spinor_field_5d(g_spinor_field[1], ofs);
      //fclose(ofs);

      if(inv_param.mass_normalization == QUDA_KAPPA_NORMALIZATION) {
        _2_kappa = 2. * g_kappa5d;
        for(ix=0;ix<VOLUME*L5;ix++) {
          _fv_ti_eq_re(g_spinor_field[1]+_GSI(ix), _2_kappa );
        }
      }
  
      // reorder, multiply with g2
      for(is=0;is<L5;is++) {
      for(ix=0;ix<VOLUME;ix++) {
        iy  = lexic2eot_5d(is, ix);
        iix = is*VOLUME + ix;
        _fv_eq_fv(g_spinor_field[2]+_GSI(iix), g_spinor_field[1]+_GSI(iy));
      }}
      for(ix=0;ix<VOLUME*L5;ix++) {
        _fv_eq_gamma_ti_fv(g_spinor_field[1]+_GSI(ix), 2, g_spinor_field[2]+_GSI(ix));
      }
#ifdef MPI
      xchange_field_5d(g_spinor_field[1]);
#endif
#else
      for(ix=0;ix<VOLUME*L5;ix++) { _fv_eq_zero(g_spinor_field[1]+_GSI(ix) ); }
      //g_spinor_field[1][_GSI(g_ipt[lsl0][lsl1][lsl2][lsl3])] = 1.;
      xchange_field_5d(g_spinor_field[1]);
//      if(strcmp(gaugefilename_prefix, "identity")==0) {
//        if(g_cart_id==0) fprintf(stdout, "# [] calling invert_Q_DW_Wilson_her\n");
//        status = invert_Q_DW_Wilson_her(g_spinor_field[1], g_spinor_field[0], 2)
//      } else {
        if(g_cart_id==0) fprintf(stdout, "# [] calling invert_Q_Wilson\n");
        // status = invert_Q_DW_Wilson(g_spinor_field[1], g_spinor_field[0], 2);
        status = invert_Q_DW_Wilson_her(g_spinor_field[1], g_spinor_field[0], 2);
//      }
        if(status < 0) {
          if(g_cart_id==0) fprintf(stderr, "[invert_dw_quda] Error from inversion, status was %d\n", status);
          EXIT(5);
        }
#endif
      retime = CLOCK;
      if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] inversion done in %e seconds\n", retime-ratime);


      /***********************************************
       * check residuum
       ***********************************************/
      if(check_residuum) {
        // apply the Wilson Dirac operator in the gamma-basis defined in cvc_linalg,
        //   which uses the tmLQCD conventions (same as in contractions)
        //   without explicit boundary conditions
#ifdef MPI
        xchange_field_5d(g_spinor_field[1]);
#endif
        for(ix=0;ix<L5*VOLUME;ix++) { _fv_eq_zero(g_spinor_field[2]); }
        //xchange_field_5d(g_spinor_field[2]);

        //sprintf(filename, "%s.inverted.ascii.%.2d", source_filename, g_cart_id);
        //ofs = fopen(filename, "w");
        //printf_spinor_field_5d(g_spinor_field[1], ofs);
        //fclose(ofs);

        Q_DW_Wilson_phi(g_spinor_field[2], g_spinor_field[1]);
  
        for(ix=0;ix<VOLUME*L5;ix++) {
          _fv_mi_eq_fv(g_spinor_field[2]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
        }
  
        spinor_scalar_product_re(&norm, g_spinor_field[2], g_spinor_field[2], VOLUME*L5);
        spinor_scalar_product_re(&norm2, g_spinor_field[0], g_spinor_field[0], VOLUME*L5);
        if(g_cart_id==0) fprintf(stdout, "\n# [invert_dw_quda] absolut residuum squared: %e; relative residuum %e\n", norm, sqrt(norm/norm2) );
      }
  
      /***********************************************
       * create 4-dim. propagator
       ***********************************************/
      spinor_5d_to_4d(g_spinor_field[1], g_spinor_field[1]);

      /***********************************************
       * write the solution 
       ***********************************************/
      sprintf(filename, "%s.inverted", source_filename);
      if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] writing propagator to file %s\n", filename);
      check_error(write_propagator(g_spinor_field[1], filename, 0, g_propagator_precision), "write_propagator", NULL, 22);
      
 
    }  // of loop on momenta

  }  // of isc

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
#ifdef HAVE_QUDA
  // finalize the QUDA library
  if(g_cart_id==0) fprintf(stdout, "# [invert_dw_quda] finalizing quda\n");
#ifdef MPI
  freeGaugeQuda();
#endif
  endQuda();
#endif
  if(g_gauge_field != NULL) free(g_gauge_field);
  if(gauge_field_smeared != NULL) free(gauge_field_smeared);
  if(no_fields>0) {
    if(g_spinor_field!=NULL) {
      for(i=0; i<no_fields; i++) if(g_spinor_field[i]!=NULL) free(g_spinor_field[i]);
      free(g_spinor_field);
    }
  }
  free_geometry();

  if(g_source_momentum_set && full_orbit) {
    finalize_q_orbits(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_rep);
    if(qlatt_map != NULL) {
      free(qlatt_map[0]);
      free(qlatt_map);
    }
  }
  if(source_momentum != NULL) free(source_momentum);

#ifdef MPI
#ifdef HAVE_QUDA
  endCommsQuda();
#else
  MPI_Finalize();
#endif
#endif
  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [invert_dw_quda] %s# [invert_dw_quda] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [invert_dw_quda] %s# [invert_dw_quda] end of run\n", ctime(&g_the_time));
  }
  return(0);
}
