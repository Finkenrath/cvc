/****************************************************
 * check_residuum.c
 *
 * Mo 14. Nov 22:26:16 EET 2011
 *
 * PURPOSE:
 * - check residuum of inverted source; Wilson fermion with Sign boundary condition
 * - reconstruct the source field (APE smear gauge field, Jacobi smear source field)
 * - only point source so far; for stochastic sources one needs the ranlxd state file
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

void usage() {
  fprintf(stdout, "Code to apply D to propagator, reconstruct source, check residuum\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, status;
  int isc;
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
  int num_threads = 1;
  unsigned int VOL3;
  int do_gt   = 0;
  char filename[200];
  double ratime, retime;
  double plaq_r=0., plaq_m=0., norm, norm2;
  // double spinor1[24], spinor2[24];
  double *gauge_qdp[4], *gauge_field_timeslice=NULL, *gauge_field_smeared=NULL;
  double _1_2_kappa, _2_kappa;
  // FILE *ofs;
  int mu_trans[4] = {3, 0, 1, 2};
  int threadid, nthreads;
  int timeslice;
  int spin_color_index = -1;
  char rng_file_in[100], rng_file_out[100];

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?vgf:t:s:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
    case 'g':
      do_gt = 1;
      break;
    case 't':
      num_threads = atoi(optarg);
      fprintf(stdout, "\n# [] will use %d threads in spacetime loops\n", num_threads);
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 's':
      spin_color_index = atoi(optarg);
      fprintf(stdout, "# [] will use spin-color index %d\n", spin_color_index);
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
  omp_set_num_threads(num_threads);
#endif

  /**************************************
   * set the default values, read input
   **************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_proc_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[check_residuum] Error, T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stderr, "[check_residuum] Error, kappa should be > 0.\n");
    usage();
  }

  // check the spin-color index
  if(g_source_type == 0 && spin_color_index == -1) {
    if(g_proc_id==0) fprintf(stderr, "[check_residuum] Error, spin-color index must be non-negative\n");
    usage();
  }

  // initialize MPI parameters
  mpi_init(argc, argv);
  
  // the volume of a timeslice
  VOL3 = LX*LY*LZ;

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
   * prepare the gauge field
   **************************************/
  // read the gauge field from file
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(strcmp( gaugefilename_prefix, "identity")==0 ) {
    if(g_cart_id==0) fprintf(stdout, "# [check_residuum] Setting up unit gauge field\n");
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
      fprintf(stderr, "[check_residuum] Error, could not read gauge field");
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

  /*********************************************
   * APE smear the gauge field
   *********************************************/
  if(N_ape>0) {
    alloc_gauge_field(&gauge_field_smeared, VOLUMEPLUSRAND);
    memcpy(gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));
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
  no_fields = 3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* the source locaton */
  sl0 =   g_source_location                              / (LX_global*LY_global*LZ);
  sl1 = ( g_source_location % (LX_global*LY_global*LZ) ) / (          LY_global*LZ);
  sl2 = ( g_source_location % (          LY_global*LZ) ) / (                    LZ);
  sl3 =   g_source_location %                      LZ;
  if(g_cart_id==0) fprintf(stdout, "# [check_residuum] global sl = (%d, %d, %d, %d)\n", sl0, sl1, sl2, sl3);
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
    fprintf(stdout, "# [check_residuum] process %d has the source at (%d, %d, %d, %d)\n", g_cart_id, lsl0, lsl1, lsl2, lsl3);
  }

  /***********************************************
   * prepare the souce
   ***********************************************/
    switch(g_source_type) {
      case 0:
        // point source
        fprintf(stdout, "# [] Creating point source\n");
        for(ix=0;ix<VOLUME;ix++) { _fv_eq_zero( g_spinor_field[0]+_GSI(ix) ); }
        if(have_source_flag) {
          g_spinor_field[0][_GSI(g_source_location) + 2*spin_color_index  ] = 1.;
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

    // write the solution
    switch(g_source_type) {
      case 0:
        sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted",
            filename_prefix, Nconf, sl0, sl1, sl2, sl3, spin_color_index);
        break;
    } 
    status = read_lime_spinor(g_spinor_field[1], filename, 0);
    if(status != 0) {
      fprintf(stderr, "Error from read_lime_spinor, status was %d\n", status);
      exit(22);
    }

    /***********************************************
     * check residuum
     ***********************************************/
    Q_Wilson_phi(g_spinor_field[2], g_spinor_field[1]);

    for(ix=0;ix<VOLUME;ix++) {
      _fv_mi_eq_fv(g_spinor_field[2]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
    }

    spinor_scalar_product_re(&norm, g_spinor_field[2], g_spinor_field[2], VOLUME);
    spinor_scalar_product_re(&norm2, g_spinor_field[0], g_spinor_field[0], VOLUME);
    fprintf(stdout, "\n# [] absolut residuum squared: %e; relative residuum %e\n", norm, sqrt(norm/norm2) );

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

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
    fprintf(stdout, "\n# [check_residuum] %s# [check_residuum] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [check_residuum] %s# [check_residuum] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
