/****************************************************
 * cvc_2pt_conn_qdep.c
 *
 * Fri Jan 18 09:47:58 EET 2013
 *
 * PURPOSE:
 * - originally copied from cvc_2pt_conn
 * - read timeslices of propergators, contract them to form meson 2-pt. functions and Fourier transform in spatial
 *   momentum
 * - focus now on (smeared) propagators from (smeared) point sources and on charged contractions
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
#include "gauge_io.h"
#include "Q_phi.h"
#include "fuzz.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "make_q_orbits.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for connected contributions to meson 2-pt. functions\n");
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
      if(g_sink_momentum_set) {
        sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.inverted", prefix, conf, sc[0], sc[1], sc[2], sc[3], isc,
            source_momentum[0], source_momentum[1], source_momentum[2]);
      } else  {
        sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", prefix, conf, sc[0], sc[1], sc[2], sc[3], isc);
      }
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

  int c, i, j, ll, sl, status, itmp[3], count;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, ix, idx, it;
  int VOL3, icol;
  int K=20, nK=20, itype;
//  int use_mms=0, mms1=0;
  int full_orbit=0;
  int source_coords[4], source_coords_orig[4], source_proc_coords[4], source_proc_id, source_location, lsource_coords[4];
  size_t items, bytes;
  double *cconnx = NULL, *cconnq=NULL;
  double *nconnx = NULL, *nconnq=NULL;
  double *work=NULL;
  int verbose = 0;
  int fermion_type = -1;
  char filename[200], line[200], gauge_field_filename[200];
  double ratime, retime;
  double plaq_r, plaq_m;
  double *gauge_field_timeslice=NULL;
  double **chi=NULL, **chi2=NULL, **psi=NULL, **psi2=NULL;
  double *Ctmp, dtmp[2], cosphase, sinphase, phase;
  double correlator_norm;
  FILE *ofs;
/*  double sign_adj5[] = {-1., -1., -1., -1., +1., +1., +1., +1., +1., +1., -1., -1., -1., 1., -1., -1.}; */
  double conf_gamma_sign[] = {1., 1., 1., 1., 1., -1., -1., -1., -1.};
  int snk_momentum_runs = 1, snk_momentum_id=0, snk_momentum[3], src_momentum[3], imom;
  size_t nconnx_length=0, nconnq_length=0, cconnx_length=0, cconnq_length=0;


//  int do_shifts = 0, shifts_num=1, ishift;
//  int shift_vector[5][4] =  {{0,0,0,0}, {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}};


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

/**************************************************************************/
  int momentum_filename_set = 0, momentum_no=0;
  char momentum_filename[200];
  int **momentum_list=NULL, *momentum_id=NULL;
/**************************************************************************/
  int *qlatt_id=NULL, *qlatt_count=NULL, **qlatt_rep=NULL, **qlatt_map=NULL, qlatt_nclass;
  double **qlatt_list=NULL;
/**************************************************************************/
  DML_Checksum *spinor_cks=NULL, ildg_gauge_field_checksum;
  uint32_t nersc_gauge_field_checksum;
/**************************************************************************/
  fftw_complex *in=NULL;
#ifdef MPI
  fftwnd_mpi_plan plan_p;
#else
  fftwnd_plan plan_p;
#endif
  int dims[3];
/**************************************************************************/




#ifdef MPI
  MPI_Status status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "soh?vguf:p:m:F:P:")) != -1) {
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
      fprintf(stdout, "# [cvc_2pt_conn_qdep] will use number of colors = %d\n", n_c);
      break;
//    case 'm':
//      mms1 = atoi(optarg);
//      break;
//    case 'u':
//      use_mms = 1;
//      break;
    case 'F':
      if(strcmp(optarg, "Wilson") == 0) {
        fermion_type = _WILSON_FERMION;
      } else if(strcmp(optarg, "tm") == 0) {
        fermion_type = _TM_FERMION;
      } else {
        fprintf(stderr, "[cvc_2pt_conn_qdep] Error, unrecognized fermion type\n");
        EXIT(145);
      }
      fprintf(stdout, "# [cvc_2pt_conn_qdep] will use fermion type %s ---> no. %d\n", optarg, fermion_type);
      break;
    case 'o':
      full_orbit=1;
      fprintf(stdout, "# [cvc_2pt_conn_qdep] will loop over full orbit\n");
      break;
//    case 's':
//      do_shifts=1;
//      fprintf(stdout, "# [cvc_2pt_conn_qdep] will include shifts +e_\\mu of source location\n");
//      break;
    case 'P':
      momentum_filename_set = 1;
      strcpy(momentum_filename, optarg);
      fprintf(stdout, "# [baryon_corr_qdep] will use momentum file %s\n", momentum_filename);
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // time
  g_the_time = time(NULL);

  // set the default values
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

#ifdef OPENMP
  omp_set_num_threads(g_num_threads);

  status = fftw_threads_init();
  if(status != 0) {
    fprintf(stderr, "[cvc_2pt_conn_qdep] Error from fftw_init_threads; status was %d\n", status);
    EXIT(120);
  }
#else
  fprintf(stdout, "[cvc_2pt_conn_qdep] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif



  // set g_sink_momentum_set
  //if(momentum_filename_set) { g_sink_momentum_set = 1; }

  //if(!g_sink_momentum_set && full_orbit) {
  //  if(g_cart_id==0) fprintf(stderr, "[cvc_2pt_conn_qdep] Error, full orbit but no sink momentum specified\n");
  //  EXIT(123);
  //}

  if(!fermion_type == -1) {
    if(g_cart_id==0) fprintf(stderr, "[cvc_2pt_conn_qdep] Error, fermion type has not been set\n");
    EXIT(124);
  }

  // initialize MPI parameters
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
  status = make_qcont_orbits_3d_parity_avg( &qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
  if(status != 0) {
    fprintf(stderr, "\n[cvc_2pt_conn_qdep] Error while creating O_3-lists\n");
    EXIT(4);
  }
  fprintf(stdout, "# [cvc_2pt_conn_qdep] number of classes = %d\n", qlatt_nclass);

  if(momentum_filename_set) {
    /***************************************************************************
     * read the momentum list to be used
     ***************************************************************************/
    ofs = fopen(momentum_filename, "r");
    if(ofs == NULL) {
      fprintf(stderr, "[cvc_2pt_conn_qdep] Error, could not open file %s for reading\n", momentum_filename);
      exit(6);
    }
    // (1) count number of momentum classes and total number of momenta
    momentum_no = 0;
    count = 0;
    while( fgets(line, 199, ofs) != NULL) {
      if(line[0] != '#') {
        momentum_no++;
        sscanf(line, "%d%d%d", itmp, itmp+1, itmp+2);
        itmp[0] += itmp[0] < 0 ? LX : 0;
        itmp[1] += itmp[1] < 0 ? LY : 0;
        itmp[2] += itmp[2] < 0 ? LZ : 0;
        ix = g_ipt[0][itmp[0]][itmp[1]][itmp[2]];
        idx = qlatt_id[ix];
        fprintf(stdout, "# [] itmp = (%2d, %2d, %2d), ixlexic=%3d, id=%3d, number of members %3d\n",
            itmp[0], itmp[1], itmp[2], ix, idx, qlatt_count[idx]);
        count += full_orbit ? qlatt_count[idx] : 1;
      }
    }
    if(momentum_no == 0) {
      fprintf(stderr, "[cvc_2pt_conn_qdep] Error, number of momenta is zero\n");
      exit(7);
    } else {
      fprintf(stdout, "# [cvc_2pt_conn_qdep] number of momenta = %d / %d\n", momentum_no, count);
      fflush(stdout);
    }
    rewind(ofs);
    momentum_list    = (int**)malloc(count * sizeof(int*));
    momentum_list[0] = (int*)malloc(3*count * sizeof(int));
    for(i=1;i<count;i++) momentum_list[i] = momentum_list[i-1] + 3;
    count = 0;
    while( fgets(line, 199, ofs) != NULL) {
      if(line[0] != '#') {
        sscanf(line, "%d%d%d", itmp, itmp+1, itmp+2);
        itmp[0] += itmp[0] < 0 ? LX : 0;
        itmp[1] += itmp[1] < 0 ? LY : 0;
        itmp[2] += itmp[2] < 0 ? LZ : 0;
        idx = qlatt_id[ g_ipt[0][itmp[0]][itmp[1]][itmp[2]] ];

        if(full_orbit) {
          for(i=0; i<qlatt_count[idx]; i++) {
            x0 = qlatt_map[idx][i];
            momentum_list[count + i][0] = x0 / (LY * LZ);
            momentum_list[count + i][1] = (x0 % (LY * LZ) ) / LZ;
            momentum_list[count + i][2] = x0 % LZ;
          }
          count += qlatt_count[idx];
        } else {
          memcpy(momentum_list[count], itmp, 3*sizeof(int));
          count++;
        }
      }
    }
    fclose(ofs);

    momentum_no = count;
    momentum_id = (int*)malloc(momentum_no * sizeof(int));
    for(i=0;i<momentum_no;i++) {
      momentum_id[i] = g_ipt[0][momentum_list[i][0]][momentum_list[i][1]][momentum_list[i][2]];
    }
    // TEST
    fprintf(stdout, "# [cvc_2pt_conn_qdep] momentum id list\n");
    for(i=0;i<momentum_no;i++) {
      fprintf(stdout, "\t%3d%6d\t%3d%3d%3d\n", i, momentum_id[i], momentum_list[i][0],  momentum_list[i][1], momentum_list[i][2]);
    }

    // check for multiple occurences
    for(i=0;i<momentum_no-1;i++) {
      for(j=i+1;j<momentum_no;j++) {
        if(momentum_id[i] == momentum_id[j]) {
          fprintf(stderr, "[cvc_2pt_conn_qdep] Error, multiple occurence of momentum no. %d: %6d = (%d, %d, %d)\n",
              i, momentum_id[i], momentum_list[i][0], momentum_list[i][1], momentum_list[i][2]);
          exit(127);
        }
      }
    }
  }  // of if momentum_filename_set

  /**********************************
   * source coordinates
   **********************************/
  if(g_source_type==0) {
    source_coords[0] = g_source_location / (LX_global * LY_global * LZ_global);
    source_coords[1] = ( g_source_location % (LX_global * LY_global * LZ_global) ) / (LY_global * LZ_global);
    source_coords[2] = ( g_source_location % (LY_global * LZ_global) ) / LZ_global;
    source_coords[3] = g_source_location % LZ_global;
    fprintf(stdout, "# [cvc_2pt_conn_qdep] global source_location %d ---> global source coordinates = (%d, %d, %d, %d)\n", g_source_location,
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
      fprintf(stdout, "# [cvc_2pt_conn_qdep] local source_location %d ---> local source coordinates = (%d, %d, %d, %d)\n", source_location,
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
  cconnx = (double*)calloc(8*K*VOL3, sizeof(double));
  if( cconnx == NULL ) {
    fprintf(stderr, "could not allocate memory for cconnx\n");
    EXIT(3);
  }
  cconnx_length = 8*K*VOL3;

  items = 8 * K * momentum_no * ( (g_cart_id==0) ? T_global : T );
  cconnq_length = items;
  cconnq = (double*)calloc( items, sizeof(double));
  if( cconnq == NULL ) {
    fprintf(stderr, "could not allocate memory for cconnq\n");
    EXIT(3);
  }

  memset(cconnq, 0, items*sizeof(double));

/*
  nconnx = (double*)calloc(8*K*VOL3, sizeof(double));
  if( nconnx == NULL ) {
    fprintf(stderr, "could not allocate memory for cconnx\n");
    EXIT(3);
  }
  nconnx_length = 8*nK*VOL3;

  nconnq = (double*)calloc( items, sizeof(double));
  if( nconnq == NULL ) {
    fprintf(stderr, "could not allocate memory for cconnq\n");
    EXIT(3);
  }
  nconnq_length = 8 * nK * momentum_no * ( (g_cart_id==0) ? T_global : T );

  memset(nconnq, 0, items*sizeof(double));
*/


  if( (Ctmp = (double*)calloc(2*VOL3, sizeof(double))) == NULL ) {
    fprintf(stderr, "Error, could not allocate mem for Ctmp\n");
    EXIT(4);
  }
 
  // intialize FFTW
  items = 4 * K * VOL3;
  bytes = sizeof(fftw_complex);

  in  = (fftw_complex*)malloc(items * bytes);
  if(in == NULL) {
    fprintf(stderr, "[] Error, could not malloc in for FFTW\n");
    EXIT(155);
  }
  dims[0]=LX; dims[1]=LY; dims[2]=LZ;
#ifdef MPI
  EXIT(129);
#else
  plan_p = fftwnd_create_plan_specific(3, dims, FFTW_BACKWARD, FFTW_MEASURE, in, 1, (fftw_complex*)cconnx, 1);
#endif


  // prepare the gauge filed
  if( N_Jacobi>0) {
    alloc_gauge_field(&g_gauge_field, VOL3);
 
    // set filename
    switch(g_gauge_file_format) {
      case 0:
        sprintf(gauge_field_filename, "%s.%.4d", gaugefilename_prefix, Nconf);
        if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_conn_qdep]reading gauge field from file %s\n", gauge_field_filename);
        break;
      case 1:
        sprintf(gauge_field_filename, "%s.%.5d", gaugefilename_prefix, Nconf);
      break;
    }
    if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_conn_qdep] reading gauge field from file %s\n", gauge_field_filename);
  
  } else {
    g_gauge_field = NULL;
  }
 
  // allocate memory for the spinor fields
  no_fields = n_s;
  if( g_sink_momentum_set ) no_fields+=n_s;
  if( fermion_type==0 ) no_fields+=n_s;
  if(Nlong>0) no_fields += n_s;
  no_fields *= n_c;
  no_fields++;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_conn_qdep] total number of fields = %d\n", no_fields);
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOL3);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOL3);
  
  
  spinor_cks = (DML_Checksum*)malloc( n_s*n_c*(1 + (fermion_type==0) + (g_sink_momentum_set==1) ) * sizeof(DML_Checksum));
  if(spinor_cks == NULL) {
    fprintf(stderr, "[] Error, could not allocate spinor_cks\n");
    EXIT(114);
  }

  // check source/sink momentum
  if(g_sink_momentum_set) {
    if(g_sink_momentum[0]<0) g_sink_momentum[0] += LX_global;
    if(g_sink_momentum[1]<0) g_sink_momentum[1] += LY_global;
    if(g_sink_momentum[2]<0) g_sink_momentum[2] += LZ_global;
    if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_conn_qdep] using final sink momentum ( %d, %d, %d )\n",
        g_sink_momentum[0], g_sink_momentum[1], g_sink_momentum[2]);
  
    g_source_momentum[0] = (-g_sink_momentum[0] + LX_global ) % LX_global;
    g_source_momentum[1] = (-g_sink_momentum[1] + LY_global ) % LY_global;
    g_source_momentum[2] = (-g_sink_momentum[2] + LZ_global ) % LZ_global;
    fprintf(stdout, "# [cvc_2pt_conn_qdep] using final source momentum ( %d, %d, %d )\n",
        g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    memcpy(src_momentum, g_source_momentum, 3*sizeof(int));
  
  }

  // set the correlator norm
  correlator_norm = 1.;

  if(g_cart_id == 0) fprintf(stdout, "# [cvc_2pt_conn_qdep] using correlator norm %e\n", correlator_norm);

  for(it=0; it<T; it++) {
    if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_conn_qdep] start processing timeslice %d\n", it);

    // read timeslice of the gauge field
    if( N_Jacobi>0) {
      switch(g_gauge_file_format) {
        case 0:
          status = read_lime_gauge_field_doubleprec_timeslice(g_gauge_field, gauge_field_filename, it, &ildg_gauge_field_checksum);
          break;
        case 1:
          status = read_nersc_gauge_field_timeslice(g_gauge_field, gauge_field_filename, it, &nersc_gauge_field_checksum);
          break;
      }
      if(status != 0) {
        fprintf(stderr, "[cvc_2pt_conn_qdep] Error, could not read gauge field\n");
        exit(21);
      }
#ifdef OPENMP
      status = APE_Smearing_Step_Timeslice_threads(g_gauge_field, N_ape, alpha_ape);
#else
      status = 0;
      for(i=0; i<N_ape; i++) { status |= APE_Smearing_Step_Timeslice(g_gauge_field, alpha_ape); }
#endif
      if(status != 0) {
        fprintf(stderr, "[cvc_2pt_conn_qdep] Error, from APE smearing function\n");
        EXIT(130);
      }
    }

    // TEST
    // write smeared gauge field
/*
    ofs = it==0 ? fopen("gauge.sm", "w") : fopen("gauge.sm", "a");
    for(ix=0; ix<VOL3; ix++) {
      for(i=0; i<4; i++) {
        for(j=0;j<9;j++) {
          fprintf(ofs, "%8d%3d%3d%25.16e%25.16e\n", it*VOL3 + ix, i, j, g_gauge_field[(ix*4+i)*18+2*j], g_gauge_field[(ix*4+i)*18+2*j+1]);
        }
      }
    }
    fclose(ofs);
*/


    memset(cconnx, 0, cconnx_length*sizeof(double));
//    memset(nconn, 0, nconnx_length*sizeof(double));



    /*************************************
     * begin loop on LL, LS, SL, SS
     *************************************/
    ll = 0;
    for(j=2; j<4; j++)
    {
      work = g_spinor_field[no_fields-1];
      if(j==0) {
        // local-local
        ll = 0;
        for(i=0; i<n_s*n_c; i++) {
          get_propagator_filename(filename, filename_prefix, source_coords, i, src_momentum_zero, Nconf);

          check_error(read_lime_spinor_timeslice(g_spinor_field[i], it, filename, 0, spinor_cks+i),
              "read_lime_spinor_timeslice", NULL, 15);
          if(g_sink_momentum_set) {
            get_propagator_filename(filename, filename_prefix, source_coords, i, src_momentum, Nconf);

            check_error(read_lime_spinor_timeslice(g_spinor_field[i+n_s*n_c], it, filename, 0, spinor_cks+n_s*n_c+i),
                "read_lime_spinor_timeslice", NULL, 15);
          }
          if(fermion_type == 0) { // read down propagators from position 1
            check_error(read_lime_spinor_timeslice(g_spinor_field[i+n_s*n_c*(1+g_sink_momentum_set)], it, filename, 1,
                  spinor_cks+i+n_s*n_c*(1+g_sink_momentum_set)), "read_lime_spinor", NULL, 16);
          }
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
          Jacobi_Smearing_Step_one_Timeslice_threads(g_gauge_field, g_spinor_field[i], work, N_Jacobi, kappa_Jacobi);
#else
          for(c=0; c<N_Jacobi; c++) {
            Jacobi_Smearing_Step_one_Timeslice(g_gauge_field, g_spinor_field[i], work, kappa_Jacobi);
          }
#endif
        }
      } else if(j==2) {
        // smeared-local
        ll = 2;
        chi  = &g_spinor_field[0];
        psi  = &g_spinor_field[g_sink_momentum_set*n_s*n_c];
        chi2 = fermion_type == 0 ? &g_spinor_field[0] : NULL;
        psi2 = fermion_type == 0 ? &g_spinor_field[n_s*n_c*(1+g_sink_momentum_set)] : NULL;
        for(i=0; i<n_s*n_c; i++) {
          get_propagator_filename(filename, filename_prefix, source_coords, i+n_s*n_c, src_momentum_zero, Nconf);

          check_error( read_lime_spinor_timeslice(g_spinor_field[i], it, filename, 0, spinor_cks+i),
              "read_lime_spinor", NULL, 20);
          if(g_sink_momentum_set) {
            get_propagator_filename(filename, filename_prefix, source_coords, i+n_s*n_c, src_momentum, Nconf);

            check_error( read_lime_spinor_timeslice(g_spinor_field[i+n_s*n_c], it, filename, 0, spinor_cks+i+n_s*n_c),
                "read_lime_spinor", NULL, 20);
          }
          if(fermion_type==0) {
            get_propagator_filename(filename, filename_prefix2, source_coords, i+n_s*n_c, src_momentum, Nconf);

            check_error( read_lime_spinor_timeslice(g_spinor_field[i+n_s*n_c*(1+g_sink_momentum_set)], it, filename, 0,
                  spinor_cks+i+n_s*n_c*(1+g_sink_momentum_set)), "read_lime_spinor", NULL, 21);
          }
        }
      } else if(j==3) {
        // smeared-smeared
        ll = 3;
        chi  = &g_spinor_field[0];
        psi  = &g_spinor_field[g_sink_momentum_set*n_s*n_c];
        chi2 = fermion_type==0 ? &g_spinor_field[0]: NULL;
        psi2 = fermion_type==0 ? &g_spinor_field[n_s*n_c*(1+g_sink_momentum_set)]: NULL;
        for(i = 0; i < ( (fermion_type==0) + g_sink_momentum_set + 1)*n_s*n_c; i++) {
#ifdef OPENMP
          Jacobi_Smearing_Step_one_Timeslice_threads(g_gauge_field, g_spinor_field[i], work, N_Jacobi, kappa_Jacobi);
#else
          for(c=0; c<N_Jacobi; c++) {
            Jacobi_Smearing_Step_one_Timeslice(g_gauge_field, g_spinor_field[i], work, kappa_Jacobi);
          }
#endif
          // TEST
/*
          sprintf(filename, "spinor.sm.%.2d", i);
          ofs = it==0 ? fopen(filename, "w") : fopen(filename, "a");
          for(ix=0; ix<VOL3; ix++) {
            for(c=0;c<12;c++) {
              fprintf(ofs, "%8d%3d%25.16e%25.16e\n", it*VOL3 + ix, c, g_spinor_field[i][_GSI(ix)+2*c], g_spinor_field[i][_GSI(ix)+2*c+1]);
            }
          }
          fclose(ofs);
*/

        }
      }

      /************************************************************
       * the charged contractions
       ************************************************************/
      sl = 2*ll*VOL3*K;
      itype = 1; 
      // pion sector
      for(idx=0; idx<9; idx++)
      {
        contract_twopoint_xdep_timeslice(cconnx+sl, gindex1[idx],  gindex2[idx], chi, psi, n_c, 1, 1., 64);
        sl += (2*VOL3);
        itype++; 
      }

      // rho sector
      for(idx = 9; idx < 36; idx+=3) {
        for(i = 0; i < 3; i++) {
          memset(Ctmp, 0, 2*VOL3*sizeof(double));
          contract_twopoint_xdep_timeslice(Ctmp, gindex1[idx+i],  gindex2[idx+i], chi, psi, n_c, 1, 1., 64);

          for(x0=0; x0<VOL3; x0++) {
            cconnx[sl+2*x0  ] += (conf_gamma_sign[(idx-9)/3]*vsign[idx-9+i]*Ctmp[2*x0  ]);
            cconnx[sl+2*x0+1] += (conf_gamma_sign[(idx-9)/3]*vsign[idx-9+i]*Ctmp[2*x0+1]);
          }
        }
        sl += (2*VOL3); 
        itype++;
      }

      // the a0
      contract_twopoint_xdep_timeslice(cconnx+sl, gindex1[36],  gindex2[36], chi, psi, n_c, 1, 1., 64);
      sl += (2*VOL3);
      itype++;

      // the b1
      for(i=0; i<3; i++) {
        memset(Ctmp, 0, 2*VOL3*sizeof(double));
        idx = 37;
        contract_twopoint_xdep_timeslice(Ctmp, gindex1[idx+i],  gindex2[idx+i], chi, psi, n_c, 1, 1., 64);

        for(x0=0; x0<VOL3; x0++) { 
          cconnx[sl+2*x0  ] += (vsign[idx-9+i]*Ctmp[2*x0  ]);
          cconnx[sl+2*x0+1] += (vsign[idx-9+i]*Ctmp[2*x0+1]);
        }
      }

      /************************************************************
       * the neutral contractions
       ************************************************************/

/*
      if(fermion_type == 0) {
        sl = 2*ll*nK*VOL3;
        itype = 1;
        // pion sector first
        for(idx=0; idx<9; idx++) {
          contract_twopoint_xdep_timeslice(nconnx+sl, ngindex1[idx],  ngindex2[idx], chi2, psi2, n_c, 1, 1., 64);
          sl += (2*VOL3);
          itype++;
        }
  
        // the neutral rho
        for(idx=9; idx<36; idx+=3) {
          for(i=0; i<3; i++) {
            memset(Ctmp, 0, 2*VOL3*sizeof(double));
            contract_twopoint_xdep_timeslice(Ctmp, ngindex1[idx+i], ngindex2[idx+i], chi2, psi2, n_c, 1, 1., 64);

            for(x0=0; x0<VOL3; x0++) {
              nconn[sl+2*x0  ] += (nvsign[idx-9+i]*Ctmp[2*x0  ]);
              nconn[sl+2*x0+1] += (nvsign[idx-9+i]*Ctmp[2*x0+1]);
            }
          }
          sl += (2*VOL3);
          itype++;
        }
  
        // the X (JPC=0+- with no experimental candidate known)
        contract_twopoint_xdep_timeslice(nconnx+sl, ngindex1[36], ngindex2[36], chi2, psi2, n_c, 1, 1., 64);
        sl += (2*VOL3);
        itype++;
  
        // the a1/f1
        for(i = 0; i < 3; i++) {
          memset(Ctmp, 0, 2*VOL3*sizeof(double));
          idx = 37;
          contract_twopoint_xdep_timeslice(Ctmp, ngindex1[idx+i], ngindex2[idx+i], chi2, psi2, n_c, 1, 1., 64);
          for(x0=0; x0<VOL3; x0++) {
            nconn[sl+2*x0  ] += (nvsign[idx-9+i]*Ctmp[2*x0  ]);
            nconn[sl+2*x0+1] += (nvsign[idx-9+i]*Ctmp[2*x0+1]);
          }
        }
      }  // of if fermion_type == 0

*/

    }    // of j=0,...,3

    // TEST
    //for(i=0; i<4*K*VOL3; i++) {
    //  fprintf(stdout, "\t%6d%25.16e%25.16e\n", i, cconnx[2*i], cconnx[2*i+1]);
    //}

    // Fourier transform
    items =  2 * K * 4 * VOL3;
    bytes = sizeof(double);

    memcpy(in, cconnx, items * bytes);
#ifdef MPI
    EXIT(129);
#else
#  ifdef OPENMP
    fftwnd_threads(g_num_threads, plan_p, 4*K, in, 1, VOL3, (fftw_complex*)cconnx, 1, VOL3);
#  else
    fftwnd(plan_p, 4*K, in, 1, VOL3, (fftw_complex*)cconnx, 1, VOL3);
#  endif
#endif

    // TEST
/*
    sl=0;
    for(ll=0; ll<4; ll++) {
      for(idx=0; idx<K; idx++) {
        for(i=0; i<VOL3; i++) {
          fprintf(stdout, "\t%3d%3d%6d%25.16e%25.16e\n", ll, idx, i, cconnx[sl], cconnx[sl+1]);
          sl += 2;
    }}}
*/

/*
    items =  2 * nK * 4 * VOL3;
    memcpy(in, nconnx, items * bytes);
#ifdef MPI
    EXIT(129);
#else
#  ifdef OPENMP
    fftwnd_threads(g_num_threads, plan_p, 4*K, in, 1, VOL3, (fftw_complex*)nconnx, 1, VOL3);
#  else
    fftwnd(plan_p, 4*K, in, 1, VOL3, (fftw_complex*)nconnx, 1, VOL3);
#  endif
#endif
*/


    // select momenta
    for(ll=0; ll<4; ll++) {
      sl    = 2 * ll * K * momentum_no;
      count = 2 * ll * K * VOL3;
      for(idx=0; idx<K; idx++) {
        for(imom=0; imom<momentum_no; imom++) {
          cconnq[sl + 2*imom  ] = cconnx[count + 2*momentum_id[imom]  ];
          cconnq[sl + 2*imom+1] = cconnx[count + 2*momentum_id[imom]+1];

          // TEST
          //fprintf(stdout, "\t%3d%3d%6d%25.16e%25.16e\n", ll, idx, imom,  cconnq[sl + 2*imom],  cconnq[sl + 2*imom+1]);
        }
        sl    += 2 * momentum_no;
        count += 2 * VOL3;
      }
    }
    if(g_source_type == 0) {
      // add phase factors
      fprintf(stdout, "# [cvc_2pt_conn_qdep] adding phase factors from source location\n");
      for(imom=0; imom<momentum_no; imom++) {
        phase = 2. * M_PI * ( \
                (double)momentum_list[imom][0] / (double)LX * source_coords[1] \
              + (double)momentum_list[imom][1] / (double)LY * source_coords[2] \
              + (double)momentum_list[imom][2] / (double)LZ * source_coords[3] \
            );
        cosphase =  cos(phase);
        sinphase = -sin(phase);
        for(ll=0; ll<4; ll++) {
          for(idx=0; idx<K; idx++) {
            sl = 2 * ( (ll * K + idx) * momentum_no + imom );
            dtmp[0] = cconnq[sl  ];
            dtmp[1] = cconnq[sl+1];
            cconnq[sl  ] = dtmp[0] * cosphase - dtmp[1] * sinphase;
            cconnq[sl+1] = dtmp[1] * cosphase + dtmp[0] * sinphase;
          }
        }
      }
    }  // of if g_source_type == 0
  

    // write to file
    if(g_cart_id==0) {
      if(g_source_type == 0) {
        sprintf(filename, "charged.t%.2dx%.2dy%.2dz%.2d.%.4d", source_coords[0], source_coords[1], source_coords[2],
            source_coords[3], Nconf);
      } else {
        sprintf(filename, "charged.%.2d.%.4d", g_source_timeslice, Nconf);
      }
      if(it==0) {
        ofs=fopen(filename, "w");
      } else {
        ofs=fopen(filename, "a");
      }
      if( ofs == (FILE*)NULL ) {
        fprintf(stderr, "Error, could not open file %s for writing\n", filename);
        EXIT(6);
      }
      fprintf(stdout, "# [cvc_2pt_conn_qdep] writing charged correlators to file %s\n", filename);
      fprintf(ofs, "# %3d%3d%3d%3d%10.6f%8.4f%7d\n", T, LX, LY, LZ, g_kappa, g_mu, it);
      sl = 0;
      for(ll=0; ll<4; ll++)
      {
        for(idx=0; idx<K; idx++)
        {
          for(imom=0; imom<momentum_no; imom++) {
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e%3d%3d%3d\n", idx+1, 2*ll+1, it,
                isneg[idx]*cconnq[sl]*correlator_norm, isneg[idx]*cconnq[sl+1]*correlator_norm,
                momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2]);
            sl += 2;
          }
        }
      }  // end of loop on ll 
      fclose(ofs);

/*
      if(fermion_type==0) {
      }
*/

    }  // of if g_cart_id == 0

    if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_conn_qdep] finished processing timeslice %d\n", it);
  }    // of loop on timeslices


  /****************************************************
   * free the allocated memory, finalize
   ****************************************************/
  free(g_gauge_field); g_gauge_field=(double*)NULL;
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); g_spinor_field=(double**)NULL;
  free_geometry();
  if(cconnx != NULL) free(cconnx);
  if(cconnq != NULL) free(cconnq);
  if(nconnx != NULL) free(nconnx);
  if(nconnq != NULL) free(nconnq);
  if(Ctmp != NULL) free(Ctmp);
  if(spinor_cks != NULL) free(spinor_cks);

  finalize_q_orbits(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_rep);
  if(qlatt_map != NULL) {
    free(qlatt_map[0]);
    free(qlatt_map);
  }

  if(g_cart_id==0) {
    fprintf(stdout, "# [cvc_2pt_conn_qdep] %s# [cvc_2pt_conn_qdep] end fo run\n", ctime(&g_the_time));
    fprintf(stderr, "[cvc_2pt_conn_qdep] %s[cvc_2pt_conn_qdep] end fo run\n", ctime(&g_the_time));
  }
#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}
