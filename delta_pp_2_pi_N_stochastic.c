/****************************************************
 * delta_pp_2_pi_N_stochastic.c
 *
 * Thu Jan 19 11:30:08 EET 2012
 *
 * PURPOSE:
 * - delta^++ to pi^+ N^+ 3-pt. function using one stochastic propagator
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
#endif
#ifdef OPENMP
#include <omp.h>
#endif
#include <getopt.h>

#define MAIN_PROGRAM

#include "ifftw.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "dml.h"
#include "gauge_io.h"
#include "Q_phi.h"
#include "fuzz.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "make_H3orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for proton 2-pt. function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose [no effect, lots of stdout output]\n");
  fprintf(stdout, "         -f input filename [default proton.input]\n");
  fprintf(stdout, "         -p number of colors [default 1]\n");
  fprintf(stdout, "         -a write ascii output too [default no ascii output]\n");
  fprintf(stdout, "         -F fermion type [default Wilson fermion, id 1]\n");
  fprintf(stdout, "         -t number of threads for OPENMP [default 1]\n");
  fprintf(stdout, "         -g do random gauge transformation [default no gauge transformation]\n");
  fprintf(stdout, "         -h? this help\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  const int n_c=3;
  const int n_s=4;
  const char outfile_prefix[] = "delta_pp_2pt_v3";

  int c, i, icomp;
  int filename_set = 0;
  int append, status;
  int l_LX_at, l_LXstart_at;
  int ix, it, iix, x1,x2,x3;
  int ir, ir2, is;
  int VOL3;
  int do_gt=0;
  int dims[3];
  double *connt=NULL;
  spinor_propagator_type *connq=NULL;
  int verbose = 0;
  int sx0, sx1, sx2, sx3;
  int write_ascii=0;
  int fermion_type = 1;  // Wilson fermion type
  int num_threads=1;
  int pos;
  char filename[200], contype[200], gauge_field_filename[200];
  double ratime, retime;
  //double plaq_m, plaq_r;
  double *work=NULL;
  fermion_propagator_type fp1=NULL, fp2=NULL, fp3=NULL, fp4=NULL, fpaux=NULL, uprop=NULL, dprop=NULL, *stochastic_fp=NULL;
  spinor_propagator_type sp1, sp2;
  double q[3], phase, *gauge_trafo=NULL;
  double *stochastic_source=NULL, *stochastic_prop=NULL;
  complex w, w1;
  size_t items, bytes;
  FILE *ofs;
  int timeslice;
  DML_Checksum ildg_gauge_field_checksum, *spinor_field_checksum=NULL, connq_checksum;
  uint32_t nersc_gauge_field_checksum;

/***********************************************************/
  int *qlatt_id=NULL, *qlatt_count=NULL, **qlatt_rep=NULL, **qlatt_map=NULL, qlatt_nclass=0;
  int use_lattice_momenta = 0;
  double **qlatt_list=NULL;
/***********************************************************/

/***********************************************************/
  int rel_momentum_filename_set = 0, rel_momentum_no=0;
  int **rel_momentum_list=NULL;
  char rel_momentum_filename[200];
/***********************************************************/

/***********************************************************/
  int snk_momentum_no = 1;
  int **snk_momentum_list = NULL;
  int snk_momentum_filename_set = 0;
  char snk_momentum_filename[200];
/***********************************************************/

/*******************************************************************
 * Gamma components for the Delta:
 */
  //const int num_component = 16;
  //int gamma_component[2][16] = { {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3}, \
  //                               {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3}};
  //double gamma_component_sign[16] = {1., 1.,-1., 1., 1., 1.,-1., 1.,-1.,-1., 1.,-1., 1., 1.,-1., 1.};
  const int num_component = 4;
  int gamma_component[2][4] = { {0, 1, 2, 3},
                                {0, 1, 2, 3} };
  double gamma_component_sign[4] = {+1.,+1.,+1.,+1.};
/*
 *******************************************************************/
  fftw_complex *in=NULL;
#ifdef MPI
   fftwnd_mpi_plan plan_p;
#else
   fftwnd_plan plan_p;
#endif 

#ifdef MPI
  MPI_Status status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ah?vgf:t:F:p:P:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [] will write in ascii format\n");
      break;
    case 'F':
      if(strcmp(optarg, "Wilson") == 0) {
        fermion_type = _WILSON_FERMION;
      } else if(strcmp(optarg, "tm") == 0) {
        fermion_type = _TM_FERMION;
      } else {
        fprintf(stderr, "[] Error, unrecognized fermion type\n");
        exit(145);
      }
      fprintf(stdout, "# [] will use fermion type %s ---> no. %d\n", optarg, fermion_type);
      break;
    case 't':
      num_threads = atoi(optarg);
      fprintf(stdout, "# [] number of threads set to %d\n", num_threads);
      break;
    case 's':
      use_lattice_momenta = 1;
      fprintf(stdout, "# [] will use lattice momenta\n");
      break;
    case 'p':
      rel_momentum_filename_set = 1;
      strcpy(rel_momentum_filename, optarg);
      fprintf(stdout, "# [] will use current momentum file %s\n", rel_momentum_filename);
      break;
    case 'P':
      snk_momentum_filename_set = 1;
      strcpy(snk_momentum_filename, optarg);
      fprintf(stdout, "# [] will use nucleon momentum file %s\n", snk_momentum_filename);
      break;
    case 'g':
      do_gt = 1;
      fprintf(stdout, "# [] will perform gauge transform\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

#ifdef OPENMP
  omp_set_num_threads(num_threads);
#endif

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

  /* initialize MPI parameters */
  mpi_init(argc, argv);

#ifdef OPENMP
  status = fftw_threads_init();
  if(status != 0) {
    fprintf(stderr, "\n[] Error from fftw_init_threads; status was %d\n", status);
    exit(120);
  }
#endif

  /******************************************************
   *
   ******************************************************/
  VOL3 = LX*LY*LZ;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] parameters:\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  if(N_Jacobi>0) {

    // alloc the gauge field
    alloc_gauge_field(&g_gauge_field, VOL3);
    switch(g_gauge_file_format) {
      case 0:
        sprintf(gauge_field_filename, "%s.%.4d", gaugefilename_prefix, Nconf);
        break;
      case 1:
        sprintf(gauge_field_filename, "%s.%.5d", gaugefilename_prefix, Nconf);
        break;
    }
  } else {
    g_gauge_field = NULL;
  }


  /*********************************************************************
   * gauge transformation
   *********************************************************************/
  if(do_gt) { init_gauge_trafo(&gauge_trafo, 1.); }

  // determine the source location
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
//  g_source_time_slice = sx0;
  fprintf(stdout, "# [] source location %d = (%d,%d,%d,%d)\n", g_source_location, sx0, sx1, sx2, sx3);
  source_timeslice = sx0;


  if(!use_lattice_momenta) {
    status = make_qcont_orbits_3d_parity_avg(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
  } else {
    status = make_qlatt_orbits_3d_parity_avg(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
  }
  if(status != 0) {
    fprintf(stderr, "\n[] Error while creating h4-lists\n");
    exit(4);
  }
  fprintf(stdout, "# [] number of classes = %d\n", qlatt_nclass);


  /***************************************************************************
   * read the relative momenta q to be used
   ***************************************************************************/
/*
  ofs = fopen(rel_momentum_filename, "r");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for reading\n", rel_momentum_filename);
    exit(6);
  }
  rel_momentum_no = 0;
  while( fgets(line, 199, ofs) != NULL) {
    if(line[0] != '#') {
      rel_momentum_no++;
    }
  }
  if(rel_momentum_no == 0) {
    fprintf(stderr, "[] Error, number of momenta is zero\n");
    exit(7);
  } else {
    fprintf(stdout, "# [] number of current momenta = %d\n", rel_momentum_no);
  }
  rewind(ofs);
  rel_momentum_list = (int**)malloc(rel_momentum_no * sizeof(int*));
  rel_momentum_list[0] = (int*)malloc(3*rel_momentum_no * sizeof(int));
  for(i=1;i<rel_momentum_no;i++) { rel_momentum_list[i] = rel_momentum_list[i-1] + 3; }
  count=0;
  while( fgets(line, 199, ofs) != NULL) {
    if(line[0] != '#') {
      sscanf(line, "%d%d%d", rel_momentum_list[count], rel_momentum_list[count]+1, rel_momentum_list[count]+2);
      count++;
    }
  }
  fclose(ofs);
  fprintf(stdout, "# [] current momentum list:\n");
  for(i=0;i<rel_momentum_no;i++) {
    fprintf(stdout, "\t%3d%3d%3d%3d\n", i, rel_momentum_list[i][0], rel_momentum_list[i][1], rel_momentum_list[i][2]);
  }
*/

  /***************************************************************************
   * read the nucleon final momenta to be used
   ***************************************************************************/
  ofs = fopen(snk_momentum_filename, "r");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for reading\n", snk_momentum_filename);
    exit(6);
  }
  snk_momentum_no = 0;
  while( fgets(line, 199, ofs) != NULL) {
    if(line[0] != '#') {
      snk_momentum_no++;
    }
  }
  if(snk_momentum_no == 0) {
    fprintf(stderr, "[] Error, number of momenta is zero\n");
    exit(7);
  } else {
    fprintf(stdout, "# [] number of nucleon final momenta = %d\n", snk_momentum_no);
  }
  rewind(ofs);
  snk_momentum_list = (int**)malloc(snk_momentum_no * sizeof(int*));
  snk_momentum_list[0] = (int*)malloc(3*snk_momentum_no * sizeof(int));
  for(i=1;i<snk_momentum_no;i++) { snk_momentum_list[i] = snk_momentum_list[i-1] + 3; }
  count=0;
  while( fgets(line, 199, ofs) != NULL) {
    if(line[0] != '#') {
      sscanf(line, "%d%d%d", snk_momentum_list[count], snk_momentum_list[count]+1, snk_momentum_list[count]+2);
      count++;
    }
  }
  fclose(ofs);
  fprintf(stdout, "# [] the nucleon final momentum list:\n");
  for(i=0;i<snk_momentum_no;i++) {
    fprintf(stdout, "\t%3d%3d%3d%3d\n", i, snk_momentum_list[i][0], snk_momentum_list[i][1], snk_momentum_list[i][1], snk_momentum_list[i][2]);
  }



  /***********************************************************
   * allocate memory for the spinor fields
   ***********************************************************/
  g_spinor_field = NULL;
  if(fermion_type == _TM_FERMION) {
    no_fields = 2*n_s*n_c+3;
  } else {
    no_fields =   n_s*n_c+3;
  }
  if(N_Jacobi>0) no_fields++;

  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-2; i++) alloc_spinor_field(&g_spinor_field[i], VOL3);
  // work
  if(N_Jacobi>0) work = g_spinor_field[no_fields-4];
  // stochastic_fv
  stochastic_fv = g_spinor_field[no_fields-3];
  // stochastic source and propagator
  alloc_spinor_field(&g_spinor_field[no_fields-2], VOLUME);
  stochastic_source = g_spinor_field[no_fields-2];
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUME);
  stochastic_prop   = g_spinor_field[no_fields-1];


  spinor_field_checksum = (DML_Checksum*)malloc(no_fields * sizeof(DML_Checksum) );
  if(spinor_field_checksum == NULL ) {
    fprintf(stderr, "[] Error, could not alloc checksums for spinor fields\n");
    exit(73);
  }
  
  /*************************************************
   * allocate memory for the contractions
   *************************************************/
  items = 4* num_component*T;
  bytes = sizeof(double);
  connt = (double*)malloc(items*bytes);
  if(connt == NULL) {
    fprintf(stderr, "\n[] Error, could not alloc connt\n");
    exit(2);
  }
  for(ix=0; ix<items; ix++) connt[ix] = 0.;

  items = num_component * (size_t)VOL3;
  connq = create_sp_field( items );
  if(connq == NULL) {
    fprintf(stderr, "\n[] Error, could not alloc connq\n");
    exit(2);
  }

  items = (size_t)VOL3;
  stochastic_fp = create_sp_field( items );
  if(stochastic_fp== NULL) {
    fprintf(stderr, "\n[] Error, could not alloc stochastic_fp\n");
    exit(22);
  }

  /******************************************************
   * initialize FFTW
   ******************************************************/
  items = g_fv_dim * (size_t)VOL3;
  bytes = sizeof(fftw_complex);
  in  = (fftw_complex*)malloc( items * bytes );
  if(in == NULL) {
    fprintf(stderr, "[] Error, could not malloc in for FFTW\n");
    exit(155);
  }
  dims[0]=LX; dims[1]=LY; dims[2]=LZ;
  //plan_p = fftwnd_create_plan(3, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_p = fftwnd_create_plan_specific(3, dims, FFTW_FORWARD, FFTW_MEASURE, in, g_fv_dim, (fftw_complex*)( stochastic_fv ), g_fv_dim);

  // create the fermion propagator points
  create_fp(&uprop);
  create_fp(&dprop);
  create_fp(&fp1);
  create_fp(&fp2);
  create_fp(&fp3);
  create_fp(&stochastic_fp);
  create_sp(&sp1);
  create_sp(&sp2);


  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // !! implement twisting for _TM_FERMION
  // !!
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef OPENMP
#pragma omp parallel for private(ix) shared(stochastic_prop)
#endif
  for(ix=0;ix<VOLUME;ix++) { _fv_eq_zero(stochastic_prop+_GSI(ix)); }

  for(sid=g_sourceid; sid<=g_sourceid2;sid+=g_sourceid_step) {
    switch(g_soruce_type) {
      case 2:  // timeslice source
        sprintf(filename, "%s.%.4d.%.2d.%.5d.inverted", filename_prefix, Nconf, source_timeslice, sid);
        break;
      default:
        fprintf(stderr, "# [] source type %d not implented; exit\n", g_source_type);
        exit(100);
    }
    fprintf(stdout, "# [] trying to read sample up-prop. from file %s\n", filename);
    read_lime_spinor(stochastic_source, filename, 0);
#ifdef OPENMP
#pragma omp parallel for private(ix) shared(stochastic_prop, stochastic_source)
#endif
    for(ix=0;ix<VOLUME;ix++) { _fv_pl_eq_fv(stochastic_prop+_GSI(ix), stochastic_source+_GSI(ix)); }
  }
#ifdef OPENMP
#pragma omp parallel for private(ix) shared(stochastic_prop, stochastic_source)
#endif
  fnorm = 1. / ( (double)(g_sourceid2 - g_sourceid + 1) * g_prop_normsqr );
  for(ix=0;ix<VOLUME;ix++) { _fv_ti_eq_re(stochastic_prop+_GSI(ix), fnorm); }
  //  calculate the source
  if(fermion_type && g_propagator_bc_type == 1) {
    Q_Wilson_phi(stochastic_source, stochastic_prop);
  } else {
    Q_phi_tbc(stochastic_source, stochastic_prop);
  }

  /******************************************************
   * prepare the stochastic fermion field
   ******************************************************/
  // read timeslice of the gauge field
  if( N_Jacobi>0) {
    switch(g_gauge_file_format) {
      case 0:
        status = read_lime_gauge_field_doubleprec_timeslice(g_gauge_field, gauge_field_filename, source_timeslice, &ildg_gauge_field_checksum);
        break;
      case 1:
        status = read_nersc_gauge_field_timeslice(g_gauge_field, gauge_field_filename, source_timeslice, &nersc_gauge_field_checksum);
        break;
    }
    if(status != 0) {
      fprintf(stderr, "[] Error, could not read gauge field\n");
      exit(21);
    }
    for(i=0; i<N_ape; i++) {
#ifdef OPENMP
      status = APE_Smearing_Step_Timeslice_threads(g_gauge_field, alpha_ape);
#else
      status = APE_Smearing_Step_Timeslice(g_gauge_field, alpha_ape);
#endif
    }
  }
  // read timeslice of the 12 up-type propagators and smear them
  //
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // !! implement twisting for _TM_FERMION
  // !!
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for(is=0;is<n_s*n_c;is++) {
    if(fermion_type != _TM_FERMION) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix, Nconf, sx0, sx1, sx2, sx3, is);
    } else {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix2, Nconf, sx0, sx1, sx2, sx3, is);
    }
    status = read_lime_spinor_timeslice(g_spinor_field[is], source_timeslice, filename, 0, spinor_field_checksum+is);
    if(status != 0) {
      fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
      exit(102);
    }
    if(N_Jacobi > 0) {
      fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
          is, N_Jacobi, kappa_Jacobi);
      for(c=0; c<N_Jacobi; c++) {
#ifdef OPENMP
        Jacobi_Smearing_Step_one_Timeslice_threads(g_gauge_field, g_spinor_field[is], work, kappa_Jacobi);
#else
        Jacobi_Smearing_Step_one_Timeslice(g_gauge_field, g_spinor_field[is], work, kappa_Jacobi);
#endif
      }
    }
  }
  for(is=0;is<g_fv_dim;is++) {
    for(ix=0;ix<VOL3;ix++) {
      iix = source_timeslice * VOL3 + ix;
      _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[is]+_GSI(iix));
      _co_eq_fv_dagger_ti_fv(&w, stochastic_source+_GSI(ix), spinor1);
      stochastic_fv[_GSI(ix)+2*is  ] = w.re;
      stochastic_fv[_GSI(ix)+2*is+1] = w.im;
    }
  }
  // Fourier transform
  items = g_fv_dim * (size_t)VOL3;
  bytes = sizeof(double);
  memcpy(in, stochastic_fv, items*bytes );
#ifdef OPENMP
  fftwnd_threads(num_threads, plan_p, g_fv_dim, in, g_fv_dim, 1, (fftw_complex*)(stochastic_fv), g_fv_dim, 1);
#else
  fftwnd(plan_p, g_fv_dim, in, g_fv_dim, 1, (fftw_complex*)(stochastic_fv), g_fv_dim, 1);
#endif


  /******************************************************
   * loop on sink momenta (most likely only one: Q=(0,0,0))
   ******************************************************/
  for(imom_snk=0;imom_snk<snk_momentum_no; imom_snk++) {

    // create Phi_tilde
    _fv_eq_zero( spinor2 );
    for(ix=0;ix<LX;ix++) {
    for(iy=0;iy<LY;iy++) {
    for(iz=0;iz<LZ;iz++) {
      iix = timeslice * VOL3 + ix;
      phase = -2.*M_PI*( (ix-sx1) * snk_momentum_list[imom_snk][0] / (double)LX 
                       + (iy-sx2) * snk_momentum_list[imom_snk][1] / (double)LY 
                       + (iz-sx3) * snk_momentum_list[imom_snk][2] / (double)LZ);
      w.re = cos(phase);
      w.im = sin(phase);
      _fv_eq_fv_ti_co(spinor1, stochastic_prop + _GSI(iix), &w);
      _fv_pl_eq_fv(spinor2, spinor);
    }}}
    // create Theta
    for(ir=0;ir<g_fv_dim;ir++) {
    for(is=0;is<g_fv_dim;is++) {
      _co_eq_co_ti_co( &(stochastic_fp[ix][ir][2*is]), &(spinor2[2*ir]), &(stochastic_fv[_GSI(ix)+2*is]) );
    }}

    /******************************************************
     * loop on timeslices
     ******************************************************/
    for(timeslice=0; timeslice<T; timeslice++) {
      append = (int)( timeslice != 0 );

      // read timeslice of the gauge field
      if( N_Jacobi>0) {
        switch(g_gauge_file_format) {
          case 0:
            status = read_lime_gauge_field_doubleprec_timeslice(g_gauge_field, gauge_field_filename, timeslice, &ildg_gauge_field_checksum);
            break;
          case 1:
            status = read_nersc_gauge_field_timeslice(g_gauge_field, gauge_field_filename, timeslice, &nersc_gauge_field_checksum);
            break;
        }
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read gauge field\n");
          exit(21);
        }

        for(i=0; i<N_ape; i++) {
#ifdef OPENMP
          status = APE_Smearing_Step_Timeslice_threads(g_gauge_field, alpha_ape);
#else
          status = APE_Smearing_Step_Timeslice(g_gauge_field, alpha_ape);
#endif
        }

      }

      // read timeslice of the 12 up-type propagators and smear them
      for(is=0;is<n_s*n_c;is++) {
          sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix, Nconf, sx0, sx1, sx2, sx3, is);
          status = read_lime_spinor_timeslice(g_spinor_field[is], timeslice, filename, 0, spinor_field_checksum+is);
          if(status != 0) {
            fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
            exit(102);
          }
          if(N_Jacobi > 0) {
            fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
                is, N_Jacobi, kappa_Jacobi);
            for(c=0; c<N_Jacobi; c++) {
#ifdef OPENMP
              Jacobi_Smearing_Step_one_Timeslice_threads(g_gauge_field, g_spinor_field[is], work, kappa_Jacobi);
#else
              Jacobi_Smearing_Step_one_Timeslice(g_gauge_field, g_spinor_field[is], work, kappa_Jacobi);
#endif
            }
          }
      }

      if(fermion_type == _TM_FERMION) {
        // read timeslice of the 12 down-type propagators, smear them
        for(is=0;is<n_s*n_c;is++) {
          if(do_gt == 0) {
            sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix2, Nconf, sx0, sx1, sx2, sx3, is);
            status = read_lime_spinor_timeslice(g_spinor_field[n_s*n_c+is], timeslice, filename, 0, spinor_field_checksum+n_s*n_c+is);
            if(status != 0) {
              fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
              exit(102);
            }
            if(N_Jacobi > 0) {
              fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
                   is, N_Jacobi, kappa_Jacobi);
              for(c=0; c<N_Jacobi; c++) {
#ifdef OPENMP
                Jacobi_Smearing_Step_one_Timeslice_threads(g_gauge_field, g_spinor_field[n_s*n_c+is], work, kappa_Jacobi);
#else
                Jacobi_Smearing_Step_one_Timeslice(g_gauge_field, g_spinor_field[n_s*n_c+is], work, kappa_Jacobi);
#endif
              }
            }
        }
      }

  
      /******************************************************
       * contractions
       ******************************************************/
      for(ix=0;ix<VOL3;ix++) 
      //for(ix=0;ix<1;ix++) 
      {
  
        // assign the propagators
        _assign_fp_point_from_field(uprop, g_spinor_field, ix);
        if(fermion_type==_TM_FERMION) {
          _assign_fp_point_from_field(dprop, g_spinor_field+n_s*n_c, ix);
        } else {
          _fp_eq_fp(dprop, uprop);
        }
        flavor rotation for twisted mass fermions
        if(fermion_type == _TM_FERMION) {
          _fp_eq_rot_ti_fp(fp1, uprop, +1, fermion_type, fp2);
          _fp_eq_fp_ti_rot(uprop, fp1, +1, fermion_type, fp2);
  //        _fp_eq_rot_ti_fp(fp1, dprop, -1, fermion_type, fp2);
  //        _fp_eq_fp_ti_rot(dprop, fp1, -1, fermion_type, fp2);
        }
  
        // test: print fermion propagator point
        //printf_fp(uprop, stdout);
  
  
        for(icomp=0; icomp<num_component; icomp++) {
  
          _sp_eq_zero( connq[ix*num_component+icomp]);
  
          /******************************************************
           * first contribution
           ******************************************************/
          _fp_eq_zero(fp1);
          _fp_eq_zero(fp2);
          _fp_eq_zero(fp3);
          // C Gamma_1 x S_u = g0 g2 Gamma_1 S_u
          _fp_eq_gamma_ti_fp(fp1, gamma_component[0][icomp], uprop);
          _fp_eq_gamma_ti_fp(fp3, 2, fp1);
          _fp_eq_gamma_ti_fp(fp1, 0, fp3);
  
          // S_u x C Gamma_2 = S_u x g0 g2 Gamma_2
          _fp_eq_fp_ti_gamma(fp2, 0, uprop);
          _fp_eq_fp_ti_gamma(fp3, 2, fp2);
          _fp_eq_fp_ti_gamma(fp2, gamma_component[1][icomp], fp3);
    
          // first part
          // reduce
          _fp_eq_zero(fp3);
          _fp_eq_fp_eps_contract13_fp(fp3, fp1, uprop);
          // reduce to spin propagator
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract23_fp(sp1, fp2, fp3);
          // second part
          // reduce to spin propagator
          _sp_eq_zero( sp2 );
          _sp_eq_fp_del_contract24_fp(sp2, fp2, fp3);
          // add and assign
          _sp_pl_eq_sp(sp1, sp2);
          _sp_eq_sp_ti_re(sp2, sp1, -gamma_component_sign[icomp]);
          _sp_eq_sp( connq[ix*num_component+icomp], sp2);
  
          /******************************************************
           * second contribution
           ******************************************************/
          _fp_eq_zero(fp1);
          _fp_eq_zero(fp2);
          _fp_eq_zero(fp3);
          // first part
          // C Gamma_1 x S_u = g0 g2 Gamma_1 S_u 
          _fp_eq_gamma_ti_fp(fp1, gamma_component[0][icomp], uprop);
          _fp_eq_gamma_ti_fp(fp3, 2, fp1);
          _fp_eq_gamma_ti_fp(fp1, 0, fp3);
          // S_u x C Gamma_2 = S_u g0 g2 Gamma_2 (same S_u as above)
          _fp_eq_fp_ti_gamma(fp2, 0, fp1);
          _fp_eq_fp_ti_gamma(fp3, 2, fp2);
          _fp_eq_fp_ti_gamma(fp1, gamma_component[1][icomp], fp3);
          // reduce
          _fp_eq_zero(fp3);
          _fp_eq_fp_eps_contract13_fp(fp3, fp1, uprop);
          // reduce to spin propagator
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract23_fp(sp1, uprop, fp3);
          // second part
          // C Gamma_1 x S_u = g0 g2 Gamma_1 S_u
          _fp_eq_gamma_ti_fp(fp1, gamma_component[0][icomp], uprop);
          _fp_eq_gamma_ti_fp(fp3, 2, fp1);
          _fp_eq_gamma_ti_fp(fp1, 0, fp3);
          // S_u x C Gamma_2 = S_u g0 g2 Gamma_2
          _fp_eq_fp_ti_gamma(fp2, 0, uprop);
          _fp_eq_fp_ti_gamma(fp3, 2, fp2);
          _fp_eq_fp_ti_gamma(fp2, gamma_component[1][icomp], fp3);
          // reduce
          _fp_eq_zero(fp3);
          _fp_eq_fp_eps_contract13_fp(fp3, fp1, fp2);
          // reduce to spin propagator
          _sp_eq_zero( sp2 );
          _sp_eq_fp_del_contract24_fp(sp2, uprop, fp3);
          // add and assign
          _sp_pl_eq_sp(sp1, sp2);
          _sp_eq_sp_ti_re(sp2, sp1, -gamma_component_sign[icomp]);
          _sp_pl_eq_sp( connq[ix*num_component+icomp], sp2);
  
          /******************************************************
           * third contribution
           ******************************************************/
          _fp_eq_zero(fp1);
          _fp_eq_zero(fp2);
          _fp_eq_zero(fp3);
          // first part
          // C Gamma_1 x S_u = g0 g2 Gamma_1 S_u
          _fp_eq_gamma_ti_fp(fp1, gamma_component[0][icomp], uprop);
          _fp_eq_gamma_ti_fp(fp3, 2, fp1);
          _fp_eq_gamma_ti_fp(fp1, 0, fp3);
          // S_u x C Gamma_2 = S_u g0 g2 Gamma_2
          _fp_eq_fp_ti_gamma(fp2, 0, fp1);
          _fp_eq_fp_ti_gamma(fp3, 2, fp2);
          _fp_eq_fp_ti_gamma(fp1, gamma_component[1][icomp], fp3);
          // reduce
          _fp_eq_zero(fp3);
          _fp_eq_fp_eps_contract13_fp(fp3, fp1, uprop);
          // reduce to spin propagator
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fp3);
          // second part
          // C Gamma_1 x S_u = g0 g2 Gamma_1 S_u
          _fp_eq_gamma_ti_fp(fp1, gamma_component[0][icomp], uprop);
          _fp_eq_gamma_ti_fp(fp3, 2, fp1);
          _fp_eq_gamma_ti_fp(fp1, 0, fp3);
          // S_u x C Gamma_2 = S_u g0 g2 Gamma_2
          _fp_eq_fp_ti_gamma(fp2, 0, uprop);
          _fp_eq_fp_ti_gamma(fp3, 2, fp2);
          _fp_eq_fp_ti_gamma(fp2, gamma_component[1][icomp], fp3);
          // reduce
          _fp_eq_zero(fp3);
          _fp_eq_fp_eps_contract13_fp(fp3, fp1, fp2);
          // reduce to spin propagator
          _sp_eq_zero( sp2 );
          _sp_eq_fp_del_contract34_fp(sp2, uprop, fp3);
          // add and assign
          _sp_pl_eq_sp(sp1, sp2);
          _sp_eq_sp_ti_re(sp2, sp1, -gamma_component_sign[icomp]);
          _sp_pl_eq_sp( connq[ix*num_component+icomp], sp2);
  
        }  // of icomp
  
      }    // of ix
  
      /***********************************************
       * finish calculation of connq
       ***********************************************/
      if(g_propagator_bc_type == 0) {
        // multiply with phase factor
        fprintf(stdout, "# [] multiplying timeslice %d with boundary phase factor\n", timeslice);
        ir = (timeslice - sx0 + T_global) % T_global;
        w1.re = cos( 3. * M_PI*(double)ir / (double)T_global );
        w1.im = sin( 3. * M_PI*(double)ir / (double)T_global );
        for(ix=0;ix<num_component*VOL3;ix++) {
          _sp_eq_sp(sp1, connq[ix] );
          _sp_eq_sp_ti_co( connq[ix], sp1, w1);
        }
      } else if (g_propagator_bc_type == 1) {
        // multiply with step function
        if(timeslice < sx0) {
          fprintf(stdout, "# [] multiplying timeslice %d with boundary step function\n", timeslice);
          for(ix=0;ix<num_component*VOL3;ix++) {
            _sp_eq_sp(sp1, connq[ix] );
            _sp_eq_sp_ti_re( connq[ix], sp1, -1.);
          }
        }
      }
    
      if(write_ascii) {
        sprintf(filename, "%s_x.%.4d.t%.2dx%.2dy%.2dz%.2d.ascii", outfile_prefix, Nconf, sx0, sx1, sx2, sx3);
        write_contraction2( connq[0][0], filename, num_component*g_sv_dim*g_sv_dim, VOL3, 1, append);
      }
  
      /******************************************************************
       * Fourier transform
       ******************************************************************/
      items =  2 * num_component * g_sv_dim * g_sv_dim * VOL3;
      bytes = sizeof(double);
  
      memcpy(in, connq[0][0], items * bytes);
      ir = num_component * g_sv_dim * g_sv_dim;
  #ifdef OPENMP
      fftwnd_threads(num_threads, plan_p, ir, in, ir, 1, (fftw_complex*)(connq[0][0]), ir, 1);
  #else
      fftwnd(plan_p, ir, in, ir, 1, (fftw_complex*)(connq[0][0]), ir, 1);
  #endif
  
      // add phase factor from the source location
      iix = 0;
      for(x1=0;x1<LX;x1++) {
        q[0] = (double)x1 / (double)LX;
      for(x2=0;x2<LY;x2++) {
        q[1] = (double)x2 / (double)LY;
      for(x3=0;x3<LZ;x3++) {
        q[2] = (double)x3 / (double)LZ;
        phase = 2. * M_PI * ( q[0]*sx1 + q[1]*sx2 + q[2]*sx3 );
        w1.re = cos(phase);
        w1.im = sin(phase);
  
        for(icomp=0; icomp<num_component; icomp++) {
          _sp_eq_sp(sp1, connq[iix] );
          _sp_eq_sp_ti_co( connq[iix], sp1, w1) ;
          iix++; 
        }
      }}}  // of x3, x2, x1
  
      // write to file
      sprintf(filename, "%s_q.%.4d.t%.2dx%.2dy%.2dz%.2d.Qx%.2dQy%.2dQz%.2d.%.5d", outfile_prefix, Nconf, sx0, sx1, sx2, sx3,
         qlatt_rep[snk_momentum_list[imom_snk]][1],qlatt_rep[snk_momentum_list[imom_snk]][2],qlatt_rep[snk_momentum_list[imom_snk]][3],
         g_sourceid2-g_sourceid+1);
      sprintf(contype, "2-pt. function, (t,q_1,q_2,q_3)-dependent, source_timeslice = %d", sx0);
      write_lime_contraction_timeslice(connq[0][0], filename, 64, num_component*g_sv_dim*g_sv_dim, contype, Nconf, 0, &connq_checksum, timeslice);
  
      if(write_ascii) {
        strcat(filename, ".ascii");
        write_contraction2(connq[0][0],filename, num_component*g_sv_dim*g_sv_dim, VOL3, 1, append);
      }
  
  
      /***********************************************
       * calculate connt
       ***********************************************/
      for(icomp=0;icomp<num_component; icomp++) {
        // fwd
        _sp_eq_sp(sp1, connq[icomp]);
        _sp_eq_gamma_ti_sp(sp2, 0, sp1);
        _sp_pl_eq_sp(sp1, sp2);
        _co_eq_tr_sp(&w, sp1);
        connt[2*(icomp*T + timeslice)  ] = w.re * 0.25;
        connt[2*(icomp*T + timeslice)+1] = w.im * 0.25;
        // bwd
        _sp_eq_sp(sp1, connq[icomp]);
        _sp_eq_gamma_ti_sp(sp2, 0, sp1);
        _sp_mi_eq_sp(sp1, sp2);
        _co_eq_tr_sp(&w, sp1);
        connt[2*(icomp*T+timeslice + num_component*T)  ] = w.re * 0.25;
        connt[2*(icomp*T+timeslice + num_component*T)+1] = w.im * 0.25;
      }
  
    }  // of loop on timeslice

    // write connt
    sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.fw", outfile_prefix, Nconf, sx0, sx1, sx2, sx3);
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
      exit(3);
    }
    fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d\n", g_kappa, T_global, LX, LY, LZ, g_mu, Nconf);
  
    for(icomp=0; icomp<num_component; icomp++) {
      ir = sx0;
      fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", gamma_component[0][icomp], gamma_component[1][icomp], 0, connt[2*(icomp*T+ir)], 0., Nconf);
      for(it=1;it<T/2;it++) {
        ir  = ( it + sx0 ) % T_global;
        ir2 = ( (T_global - it) + sx0 ) % T_global;
        fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", gamma_component[0][icomp], gamma_component[1][icomp], it, connt[2*(icomp*T+ir)], connt[2*(icomp*T+ir2)], Nconf);
      }
      ir = ( it + sx0 ) % T_global;
      fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", gamma_component[0][icomp], gamma_component[1][icomp], it, connt[2*(icomp*T+ir)], 0., Nconf);
    }
    fclose(ofs);
  
    sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.bw", outfile_prefix, Nconf, sx0, sx1, sx2, sx3);
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
      exit(3);
    }
    fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d\n", g_kappa, T_global, LX, LY, LZ, g_mu, Nconf);
  
    for(icomp=0; icomp<num_component; icomp++) {
      ir = sx0;
      fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", gamma_component[0][icomp], gamma_component[1][icomp], 0, connt[2*(num_component*T+icomp*T+ir)], 0., Nconf);
      for(it=1;it<T/2;it++) {
        ir  = ( it + sx0 ) % T_global;
        ir2 = ( (T_global - it) + sx0 ) % T_global;
        fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", gamma_component[0][icomp], gamma_component[1][icomp], it, connt[2*(num_component*T+icomp*T+ir)], connt[2*(num_component*T+icomp*T+ir2)], Nconf);
      }
      ir = ( it + sx0 ) % T_global;
      fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", gamma_component[0][icomp], gamma_component[1][icomp], it, connt[2*(num_component*T+icomp*T+ir)], 0., Nconf);
    }
    fclose(ofs);

  }  // of loop on sink momentum ( = Delta^++ momentum, Qvec)

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();
  if(connt!= NULL) free(connt);
  if(connq!= NULL) free(connq);
  if(gauge_trafo != NULL) free(gauge_trafo);

  if(g_spinor_field!=NULL) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field); g_spinor_field=(double**)NULL;
  }
  if(spinor_field_checksum !=NULL) free(spinor_field_checksum);
  if(g_gauge_field != NULL) free(g_gauge_field);

  if(snk_momemtum_list != NULL) {
    if(snk_momentum_list[0] != NULL) free(snk_momentum_list[0]);
    free(snk_momentum_list);
  }
  if(rel_momemtum_list != NULL) {
    if(rel_momentum_list[0] != NULL) free(rel_momentum_list[0]);
    free(rel_momentum_list);
  }

  // free the fermion propagator points
  free_fp( &uprop );
  free_fp( &dprop );
  free_fp( &fp1 );
  free_fp( &fp2 );
  free_fp( &fp3 );
  free_sp( &sp1 );
  free_sp( &sp2 );

  free(in);
  fftwnd_destroy_plan(plan_p);

  g_the_time = time(NULL);
  fprintf(stdout, "# [] %s# [] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [] %s# [] end fo run\n", ctime(&g_the_time));
  fflush(stderr);

#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}
