/****************************************************
 * delta_pp_2_pi_N_sequential_v4.c
 *
 * Fri Dec 16 12:16:45 EET 2011
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
  const char outfile_prefix[] = "deltapp2piN";

  int c, i, icomp, imom, count;
  int filename_set = 0;
  int append, status;
  int l_LX_at, l_LXstart_at;
  int ix, it, iix, x1,x2,x3;
  int ir, ir2, is;
  int VOL3;
  int do_gt=0;
  int dims[3];
  double *connt=NULL;
  spinor_propagator_type *connq=NULL, *connq_out=NULL;
  int verbose = 0;
  int sx0, sx1, sx2, sx3;
  int write_ascii=0;
  int fermion_type = _WILSON_FERMION;  // Wilson fermion type
  int smear_seq_source = 0;
  int threadid;
  char filename[200], contype[200], gauge_field_filename[200], line[200];
  double ratime, retime;
  //double plaq_m, plaq_r;
  int mode = -1;
  double *work=NULL;
  fermion_propagator_type *fp1=NULL, *fp2=NULL, *fp3=NULL, *fp4=NULL, *fpaux=NULL, *uprop=NULL, *dprop=NULL;
  spinor_propagator_type *sp1=NULL, *sp2=NULL;
  double q[3], phase, *gauge_trafo=NULL, spinor1[24];
  complex w, w1;
  size_t items, bytes;
  FILE *ofs;
  int timeslice;
  DML_Checksum ildg_gauge_field_checksum, *spinor_field_checksum=NULL, connq_checksum, *seq_spinor_field_checksum=NULL;
  uint32_t nersc_gauge_field_checksum;
  int gamma_proj_sign[] = {1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

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
  int snk_momentum_no = 0, isnk;
  int **snk_momentum_list = NULL;
  int snk_momentum_filename_set = 0;
  char snk_momentum_filename[200];
/***********************************************************/


/*******************************************************************
 * Gamma components for the Delta:
 */
  const int num_component = 4;
  int gamma_component[2][4] = { {0, 1, 2, 3},
                                {5, 5, 5, 5} };
  double gamma_component_sign[4] = {+1., +1., +1., +1.};
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

  while ((c = getopt(argc, argv, "Sah?vgf:F:p:P:s:m:")) != -1) {
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
    case 'g':
      do_gt = 1;
      fprintf(stdout, "# [] will perform gauge transform\n");
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
    case 'm':
      if(strcmp(optarg, "sequential")==0) {
        mode = 1;
      } else if(strcmp(optarg, "contract")==0) {
        mode = 2;
      }
      fprintf(stdout, "# [] will use mode %d\n", mode);
      break;
    case 'S':
      smear_seq_source = 1;
      fprintf(stdout, "# [] will smear sequential soucre\n");
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

#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[delta_pp_2_pi_N_sequential_v4] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

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

if(mode == 1 || mode == 2) {
  /***************************************************************************
   * read the relative momenta q to be used
   ***************************************************************************/
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
    if(rel_momentum_list[i][0] < 0 ) rel_momentum_list[i][0] += LX;
    if(rel_momentum_list[i][1] < 0 ) rel_momentum_list[i][1] += LY;
    if(rel_momentum_list[i][2] < 0 ) rel_momentum_list[i][2] += LZ;
    fprintf(stdout, "\t%3d%3d%3d%3d\n", i, rel_momentum_list[i][0], rel_momentum_list[i][1], rel_momentum_list[i][2]);
  }

}  // of if mode == 1

if(mode == 2) {
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
    if(snk_momentum_list[i][0]<0) snk_momentum_list[i][0] += LX;
    if(snk_momentum_list[i][1]<0) snk_momentum_list[i][1] += LY;
    if(snk_momentum_list[i][2]<0) snk_momentum_list[i][2] += LZ;
    fprintf(stdout, "\t%3d%3d%3d%3d\n", i, snk_momentum_list[i][0], snk_momentum_list[i][1], snk_momentum_list[i][2]);
  }
}  // of if mode == 2

  // allocate memory for the spinor fields
  g_spinor_field = NULL;
  if(mode == 1) {
    no_fields = 3;
    g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
    for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOL3);
    alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUME);
    if(N_Jacobi>0) work = g_spinor_field[1];
  } else if(mode == 2) {
    no_fields = 2*n_s*n_c;
    if(N_Jacobi>0) no_fields++;
    g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
    for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOL3);
    if(N_Jacobi>0) work = g_spinor_field[no_fields-1];
  }
  
  spinor_field_checksum = (DML_Checksum*)malloc(n_s*n_c * sizeof(DML_Checksum) );
  if(spinor_field_checksum == NULL ) {
    fprintf(stderr, "[] Error, could not alloc checksums for spinor fields\n");
    exit(73);
  }

  seq_spinor_field_checksum = (DML_Checksum*)malloc(rel_momentum_no*n_s*n_c * sizeof(DML_Checksum) );
  if(seq_spinor_field_checksum == NULL ) {
    fprintf(stderr, "[] Error, could not alloc checksums for seq. spinor fields\n");
    exit(73);
  }



if(mode == 1) {
  
  /*************************************************************************
   * sequential source
   *************************************************************************/

    // (1) read the prop., smear, multiply with gamma_5, save as source 

    // read timeslice of the gauge field
    if( N_Jacobi>0 && smear_seq_source) {
      switch(g_gauge_file_format) {
        case 0:
          status = read_lime_gauge_field_doubleprec_timeslice(g_gauge_field, gauge_field_filename, sx0, &ildg_gauge_field_checksum);
          break;
        case 1:
          status = read_nersc_gauge_field_timeslice(g_gauge_field, gauge_field_filename, sx0, &nersc_gauge_field_checksum);
          break;
      }
      if(status != 0) {
        if(status != 8) { // exit status 8 refers to mismatch in checksums
          fprintf(stderr, "[] Error, could not read gauge field\n");
          exit(21);
        } else {
          fprintf(stdout, "# [] Warning, mismatch in checksums\n");
        }
      }
      if(N_ape>0) {
        fprintf(stdout, "# [] APE smearing gauge field timeslice no %d with parameters N_ape=%d and alpha_ape=%e\n", sx0, N_ape, alpha_ape);
#ifdef OPENMP
        status = APE_Smearing_Step_Timeslice_threads(g_gauge_field, N_ape, alpha_ape);
#else
        for(i=0; i<N_ape; i++) {
          status = APE_Smearing_Step_Timeslice(g_gauge_field, alpha_ape);
        }
#endif
      }
    }
    // read timeslice of the 12 down-type propagators and smear them
    for(is=0;is<n_s*n_c;is++) {
      if(fermion_type != _TM_FERMION) {
        sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix, Nconf, sx0, sx1, sx2, sx3, is);
      } else {
        sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix2, Nconf, sx0, sx1, sx2, sx3, is);
      }
      status = read_lime_spinor_timeslice(g_spinor_field[0], sx0, filename, 0, spinor_field_checksum+is);
      if(status != 0) {
        fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
        exit(102);
      }
      
      if(N_Jacobi > 0 && smear_seq_source) {
        fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
            is, N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
        Jacobi_Smearing_Step_one_Timeslice_threads(g_gauge_field, g_spinor_field[0], work, N_Jacobi, kappa_Jacobi);
#else
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one_Timeslice(g_gauge_field, g_spinor_field[0], work, kappa_Jacobi);
        }
#endif
      }

      for(imom=0;imom<rel_momentum_no;imom++) {
        for(ix=0;ix<VOLUME;ix++) { _fv_eq_zero(g_spinor_field[2]+_GSI(ix)); }
        ix = 0;
        iix = sx0 * VOL3;
        for(x1=0;x1<LX;x1++) {
        for(x2=0;x2<LY;x2++) {
        for(x3=0;x3<LZ;x3++) {
          phase = 2. * M_PI * ( (x1-sx1) * rel_momentum_list[imom][0] / (double)LX
                              + (x2-sx2) * rel_momentum_list[imom][1] / (double)LY
                              + (x3-sx3) * rel_momentum_list[imom][2] / (double)LZ );
          w.re =  cos(phase);
          w.im = -sin(phase);
          _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[0] + _GSI(ix));
          _fv_eq_fv_ti_co(g_spinor_field[2]+_GSI(iix), spinor1, &w);
          ix++;
          iix++;
        }}}

        // save the sourceg_spinor_field[2]
        sprintf(filename, "seq_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d", filename_prefix, Nconf, sx0, sx1, sx2, sx3, is,
           rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
        status = write_lime_spinor(g_spinor_field[2], filename, 0, g_propagator_precision);
/*
        // TEST
        {
          sprintf(filename,"seq_source.ascii.%.2d.%.2d.%.2d", is,g_nproc, g_cart_id);
          ofs = fopen(filename,"w");
          fprintf(ofs, "# [] the sequential source:\n");
          for(ix=0;ix<VOLUME;ix++) {
            for(i=0;i<12;i++) {
              fprintf(ofs, "\t%6d%3d%25.16e%25.16e\n", ix, i, g_spinor_field[2][_GSI(ix)+2*i], g_spinor_field[2][_GSI(ix)+2*i+1]);
            }
          }
          fclose(ofs);
        }
*/
      }  // of imom
    }    // of is
}  // of if mode == 1

if(mode == 2) {

  /*************************************************************************
   * contractions
   *************************************************************************/

  // allocate memory for the contractions
  items = 4 * rel_momentum_no * num_component * T;
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

  items = (size_t)rel_momentum_no * (size_t)num_component * (size_t)T * (size_t)snk_momentum_no;
  connq_out = create_sp_field( items );
  if(connq_out == NULL) {
    fprintf(stderr, "\n[] Error, could not alloc connq_out\n");
    exit(22);
  }

  // initialize FFTW
  items = 2 * num_component * g_sv_dim * g_sv_dim * VOL3;
  bytes = sizeof(double);
  in  = (fftw_complex*)malloc(num_component*g_sv_dim*g_sv_dim*VOL3*sizeof(fftw_complex));
  if(in == NULL) {
    fprintf(stderr, "[] Error, could not malloc in for FFTW\n");
    exit(155);
  }
  dims[0]=LX; dims[1]=LY; dims[2]=LZ;
  //plan_p = fftwnd_create_plan(3, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_p = fftwnd_create_plan_specific(3, dims, FFTW_FORWARD, FFTW_MEASURE, in, num_component*g_sv_dim*g_sv_dim, (fftw_complex*)( connq[0][0] ), num_component*g_sv_dim*g_sv_dim);

  // create the fermion propagator points
  uprop = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
  if(uprop== NULL) {
    fprintf(stdout, "[] Error, could not alloc uprop\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_fp(uprop+i);
  }
  dprop = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
  if(dprop== NULL) {
    fprintf(stdout, "[] Error, could not alloc dprop\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_fp(dprop+i);
  }
  fp1 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
  if(fp1== NULL) {
    fprintf(stdout, "[] Error, could not alloc fp1\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_fp(fp1+i);
  }
  fp2 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
  if(fp2== NULL) {
    fprintf(stdout, "[] Error, could not alloc fp2\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_fp(fp2+i);
  }
  fp3 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
  if(fp3== NULL) {
    fprintf(stdout, "[] Error, could not alloc fp3\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_fp(fp3+i);
  }
  fp4 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
  if(fp4== NULL) {
    fprintf(stdout, "[] Error, could not alloc fp4\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_fp(fp4+i);
  }
  fpaux = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
  if(fpaux== NULL) {
    fprintf(stdout, "[] Error, could not alloc fpaux\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_fp(fpaux+i);
  }
  sp1 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
  if(sp1== NULL) {
    fprintf(stdout, "[] Error, could not alloc sp1\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_sp(sp1+i);
  }
  sp2 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
  if(sp2== NULL) {
    fprintf(stdout, "[] Error, could not alloc sp2\n");
    exit(172);
  } else {
    for(i=0;i<g_num_threads;i++) create_sp(sp2+i);
  }


  /******************************************************
   * loop on timeslices
   ******************************************************/
  for(timeslice=0; timeslice<T; timeslice++)
  // for(timeslice=1; timeslice<2; timeslice++)
  {
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
#ifdef OPENMP
      status = APE_Smearing_Step_Timeslice_threads(g_gauge_field, N_ape, alpha_ape);
#else
      for(i=0; i<N_ape; i++) {
        status = APE_Smearing_Step_Timeslice(g_gauge_field, alpha_ape);
      }
#endif
    }

    // read timeslice of the 12 up-type propagators and smear them
    for(is=0;is<n_s*n_c;is++) {
//      if(do_gt == 0) {
        sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix, Nconf, sx0, sx1, sx2, sx3, is);
        status = read_lime_spinor_timeslice(g_spinor_field[is], timeslice, filename, 0, spinor_field_checksum+is);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          exit(102);
        }
        if(N_Jacobi > 0) {
          fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
              is, N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
          Jacobi_Smearing_Step_one_Timeslice_threads(g_gauge_field, g_spinor_field[is], work, N_Jacobi, kappa_Jacobi);
#else
          for(c=0; c<N_Jacobi; c++) {
            Jacobi_Smearing_Step_one_Timeslice(g_gauge_field, g_spinor_field[is], work, kappa_Jacobi);
          }
#endif
        }
//      } else {  // of if do_gt == 0
//        // apply gt
//        apply_gt_prop(gauge_trafo, g_spinor_field[is], is/n_c, is%n_c, 4, filename_prefix, g_source_location);
//      } // of if do_gt == 0
    }


    /******************************************************
     * loop on relative momenta
     ******************************************************/
    for(imom=0;imom<rel_momentum_no; imom++) {

      // read 12 sequential propagators
      for(is=0;is<n_s*n_c;is++) {
//        if(do_gt == 0) {
          sprintf(filename, "seq_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.inverted", filename_prefix, Nconf, sx0, sx1, sx2, sx3, is,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
          status = read_lime_spinor_timeslice(g_spinor_field[n_s*n_c+is], timeslice, filename, 0, seq_spinor_field_checksum+imom*n_s*n_c+is);
          if(status != 0) {
            fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
            exit(102);
          }
          if(N_Jacobi > 0) {
            fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
                 is, N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
            Jacobi_Smearing_Step_one_Timeslice_threads(g_gauge_field, g_spinor_field[n_s*n_c+is], work, N_Jacobi, kappa_Jacobi);
#else
            for(c=0; c<N_Jacobi; c++) {
              Jacobi_Smearing_Step_one_Timeslice(g_gauge_field, g_spinor_field[n_s*n_c+is], work, kappa_Jacobi);
            }
#endif
          }
//        } else {  // of if do_gt == 0
//          // apply gt
//          apply_gt_prop(gauge_trafo, g_spinor_field[n_s*n_c+is], is/n_c, is%n_c, 4, filename_prefix, g_source_location);
//        } // of if do_gt == 0
      }
  
  
      /******************************************************
       * contractions
       *
       * REMEMBER:
       *
       *   uprop = S_u
       *   dprop = S_seq
       *   fp1   = C Gamma_1 S_u
       *   fp2   = C Gamma_1 S_u C Gamma_2
       *   fp3   =           S_u C Gamma_2
       *   fp4   = C Gamma_1 S_seq
       *   Gamma_1 = gamma_mu (always multiplied from the left)
       *   Gamma_2 = gamma-5  (always multiplied from the right)
       ******************************************************/
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private (ix,icomp,threadid) \
      firstprivate (fermion_type,gamma_component,connq,\
          gamma_component_sign,VOL3,g_spinor_field,fp1,fp2,fp3,fpaux,fp4,uprop,dprop,sp1,sp2,timeslice)
//      shared (num_component)
{
      threadid = omp_get_thread_num();
#else
      threadid = 0;
#endif
      for(ix=threadid; ix<VOL3; ix+=g_num_threads)
      {
        // assign the propagators
        _assign_fp_point_from_field(uprop[threadid], g_spinor_field, ix);
        _assign_fp_point_from_field(dprop[threadid], g_spinor_field+n_s*n_c, ix);
        // flavor rotation for twisted mass fermions
        if(fermion_type == _TM_FERMION) {
          _fp_eq_rot_ti_fp(fp1[threadid], uprop[threadid], +1, fermion_type, fp2[threadid]);
          _fp_eq_fp_ti_rot(uprop[threadid], fp1[threadid], +1, fermion_type, fp2[threadid]);
          _fp_eq_rot_ti_fp(fp1[threadid], dprop[threadid], +1, fermion_type, fp2[threadid]);
          _fp_eq_fp_ti_rot(dprop[threadid], fp1[threadid], -1, fermion_type, fp2[threadid]);
        }

        if(do_gt) {
          // up propagator
          _fp_eq_cm_ti_fp(fp1[threadid], gauge_trafo+18*(timeslice*VOL3+ix), uprop[threadid]);
          _fp_eq_fp_ti_cm_dagger(uprop[threadid], gauge_trafo+18*(timeslice*VOL3+ix), fp1[threadid]);
          // sequential propagator
          _fp_eq_cm_ti_fp(fp1[threadid], gauge_trafo+18*(timeslice*VOL3+ix), dprop[threadid]);
          _fp_eq_fp_ti_cm_dagger(dprop[threadid], gauge_trafo+18*(timeslice*VOL3+ix), fp1[threadid]);
        }
  
        // test: print fermion propagator point
/*
        fprintf(stdout, "# uprop[threadid]:\n");
        printf_fp(uprop[threadid], "uprop[threadid]", stdout);
        fprintf(stdout, "# dprop[threadid]:\n");
        printf_fp(dprop[threadid], "dprop[threadid]", stdout);
*/
/*
        double fp_in_base[32];
        int mu;
//        _project_fp_to_basis(fp_in_base, uprop[threadid], 0);
        _project_fp_to_basis(fp_in_base, dprop[threadid], 0);
        fprintf(stdout, "# [] t=%3d; ix=%6d\n", timeslice, ix);
        for(mu=0;mu<16;mu++) {
          fprintf(stdout, "\t%3d%16.7e%16.7e\n", mu, fp_in_base[2*mu], fp_in_base[2*mu+1]);
        }
*/
  
        for(icomp=0; icomp<num_component; icomp++) {
  
          _sp_eq_zero( connq[ix*num_component+icomp]);
  
          /******************************************************
           * prepare fermion propagators
           ******************************************************/
          _fp_eq_zero(fp1[threadid]);
          _fp_eq_zero(fp2[threadid]);
          _fp_eq_zero(fp3[threadid]);
          _fp_eq_zero(fp4[threadid]);
          _fp_eq_zero(fpaux[threadid]);
          // fp1[threadid] = C Gamma_1 x S_u = g0 g2 Gamma_1 S_u
          _fp_eq_gamma_ti_fp(fp1[threadid], gamma_component[0][icomp], uprop[threadid]);
          _fp_eq_gamma_ti_fp(fpaux[threadid], 2, fp1[threadid]);
          _fp_eq_gamma_ti_fp(fp1[threadid],   0, fpaux[threadid]);
  
          // fp2[threadid] = C Gamma_1 x S_u x C Gamma_2 = fp1[threadid] x g0 g2 Gamma_2
          _fp_eq_fp_ti_gamma(fp2[threadid], 0, fp1[threadid]);
          _fp_eq_fp_ti_gamma(fpaux[threadid], 2, fp2[threadid]);
          _fp_eq_fp_ti_gamma(fp2[threadid], gamma_component[1][icomp], fpaux[threadid]);
   
          // fp3[threadid] = S_u x C Gamma_2 = uprop[threadid] x g0 g2 Gamma_2
          _fp_eq_fp_ti_gamma(fp3[threadid],   0, uprop[threadid]);
          _fp_eq_fp_ti_gamma(fpaux[threadid], 2, fp3[threadid]);
          _fp_eq_fp_ti_gamma(fp3[threadid], gamma_component[1][icomp], fpaux[threadid]);
   
          // fp4[threadid] = C Gamma_1 x S_seq = g0 g2 Gamma_1 dprop[threadid] 
          _fp_eq_gamma_ti_fp(fp4[threadid], gamma_component[0][icomp], dprop[threadid]);
          _fp_eq_gamma_ti_fp(fpaux[threadid], 2, fp4[threadid]);
          _fp_eq_gamma_ti_fp(fp4[threadid],   0, fpaux[threadid]);

/*
        char name[20];
        sprintf(name, "fp1[%d,%d,%d,%d]", timeslice, ix, icomp, threadid);
        printf_fp(fp1[threadid], name, stdout);
        sprintf(name, "fp2[%d,%d,%d,%d]", timeslice, ix, icomp, threadid);
        printf_fp(fp2[threadid], name, stdout);
        sprintf(name, "fp3[%d,%d,%d,%d]", timeslice, ix, icomp, threadid);
        printf_fp(fp3[threadid], name, stdout);
        sprintf(name, "fp4[%d,%d,%d,%d]", timeslice, ix, icomp, threadid);
        printf_fp(fp4[threadid], name, stdout);
*/
/*
        sprintf(name, "uprop[%d,%d,%d,%d]", timeslice, ix, icomp, threadid);
        printf_fp(uprop[threadid], name, stdout);
        sprintf(name, "dprop[%d,%d,%d,%d]", timeslice, ix, icomp, threadid);
        printf_fp(dprop[threadid], name, stdout);
*/
/*
        double fp_in_base[4][32];
        int mu;
        _project_fp_to_basis(fp_in_base[0], fp1[threadid], 0);
        _project_fp_to_basis(fp_in_base[1], fp2[threadid], 0);
        _project_fp_to_basis(fp_in_base[2], fp3[threadid], 0);
        _project_fp_to_basis(fp_in_base[3], fp4[threadid], 0);
        fprintf(stdout, "# [] t=%3d; ix=%6d\n", timeslice, ix);
        for(mu=0;mu<16;mu++) {
          fprintf(stdout, "\t%3d%16.7e%16.7e%16.7e%16.7e%16.7e%16.7e%16.7e%16.7e\n", mu,
              fp_in_base[0][2*mu], fp_in_base[0][2*mu+1],
              fp_in_base[1][2*mu], fp_in_base[1][2*mu+1],
              fp_in_base[2][2*mu], fp_in_base[2][2*mu+1],
              fp_in_base[3][2*mu], fp_in_base[3][2*mu+1]);
        }
*/

          // (1)
          // reduce
          _fp_eq_zero(fpaux[threadid]);
          _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp2[threadid], uprop[threadid]);
          // reduce to spin propagator
          _sp_eq_zero( sp1[threadid] );
          _sp_eq_fp_del_contract23_fp(sp1[threadid], dprop[threadid], fpaux[threadid]);
          // (2)
          // reduce
          _fp_eq_zero(fpaux[threadid]);
          _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp1[threadid], fp3[threadid]);
          // reduce to spin propagator
          _sp_eq_zero( sp2[threadid] );
          _sp_eq_fp_del_contract24_fp(sp2[threadid], dprop[threadid], fpaux[threadid]);
          // add and assign
          _sp_pl_eq_sp(sp1[threadid], sp2[threadid]);
          _sp_eq_sp_ti_re(sp2[threadid], sp1[threadid], -gamma_component_sign[icomp]);
          _sp_pl_eq_sp( connq[ix*num_component+icomp], sp2[threadid]);

          // (3)
          // reduce
          _fp_eq_zero(fpaux[threadid]);
          _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp4[threadid], uprop[threadid]);
          // reduce to spin propagator
          _sp_eq_zero( sp1[threadid] );
          _sp_eq_fp_del_contract23_fp(sp1[threadid], fp3[threadid], fpaux[threadid]);
          // (4)
          // reduce
          _fp_eq_zero(fpaux[threadid]);
          _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp1[threadid], dprop[threadid]);
          // reduce to spin propagator
          _sp_eq_zero( sp2[threadid] );
          _sp_eq_fp_del_contract24_fp(sp2[threadid], fp3[threadid], fpaux[threadid]);
          // add and assign
          _sp_pl_eq_sp(sp1[threadid], sp2[threadid]);
          _sp_eq_sp_ti_re(sp2[threadid], sp1[threadid], -gamma_component_sign[icomp]);
          _sp_pl_eq_sp( connq[ix*num_component+icomp], sp2[threadid]);

          // (5)
          // reduce
          _fp_eq_zero(fpaux[threadid]);
          _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp4[threadid], fp3[threadid]);
          // reduce to spin propagator
          _sp_eq_zero( sp1[threadid] );
          _sp_eq_fp_del_contract34_fp(sp1[threadid], uprop[threadid], fpaux[threadid]);
          //fprintf(stdout, "# sp1[threadid]:\n");
          //printf_sp(sp1[threadid], "sp1[threadid]",stdout);
          // (6)
          // reduce
          _fp_eq_zero(fpaux[threadid]);
          _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp2[threadid], dprop[threadid]);
          // reduce to spin propagator
          _sp_eq_zero( sp2[threadid] );
          _sp_eq_fp_del_contract34_fp(sp2[threadid], uprop[threadid], fpaux[threadid]);
          //fprintf(stdout, "# sp2[threadid]:\n");
          //printf_sp(sp2[threadid], "sp2[threadid]",stdout);
          // add and assign
          _sp_pl_eq_sp(sp1[threadid], sp2[threadid]);
          _sp_eq_sp_ti_re(sp2[threadid], sp1[threadid], -gamma_component_sign[icomp]);
          _sp_pl_eq_sp( connq[ix*num_component+icomp], sp2[threadid]);

  
        }  // of icomp
  
      }    // of ix
#ifdef OPENMP
}
#endif
  
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
          _sp_eq_sp(sp1[0], connq[ix] );
          _sp_eq_sp_ti_co( connq[ix], sp1[0], w1);
        }
      } else if (g_propagator_bc_type == 1) {
        // multiply with step function
        if(timeslice < sx0) {
          fprintf(stdout, "# [] multiplying timeslice %d with boundary step function\n", timeslice);
          for(ix=0;ix<num_component*VOL3;ix++) {
            _sp_eq_sp(sp1[0], connq[ix] );
            _sp_eq_sp_ti_re( connq[ix], sp1[0], -1.);
          }
        }
      }
    
      if(write_ascii) {
        sprintf(filename, "%s_x.%.4d.t%.2dx%.2dy%.2dz%.2d.qx%.2dqy%.2dqz%.2d.ascii", outfile_prefix, Nconf, sx0, sx1, sx2, sx3,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
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
      fftwnd_threads(g_num_threads, plan_p, ir, in, ir, 1, (fftw_complex*)(connq[0][0]), ir, 1);
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
          _sp_eq_sp(sp1[0], connq[iix] );
          _sp_eq_sp_ti_co( connq[iix], sp1[0], w1) ;
          iix++; 
        }
      }}}  // of x3, x2, x1
  
      // write to file
/*
      sprintf(filename, "%s_q.%.4d.t%.2dx%.2dy%.2dz%.2d.qx%.2dqy%.2dqz%.2d", outfile_prefix, Nconf, sx0, sx1, sx2, sx3,
          rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
      sprintf(contype, "2-pt. function, (t,Q_1,Q_2,Q_3)-dependent, source_timeslice = %d, rel. momentum = (%d, %d. %d)", sx0,
          rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
      write_lime_contraction_timeslice(connq[0][0], filename, 64, num_component*g_sv_dim*g_sv_dim, contype, Nconf, 0, &connq_checksum, timeslice);
*/ 
      if(write_ascii) {
        sprintf(filename, "%s_q.%.4d.t%.2dx%.2dy%.2dz%.2d.qx%.2dqy%.2dqz%.2d.ascii", outfile_prefix, Nconf, sx0, sx1, sx2, sx3,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
        write_contraction2(connq[0][0],filename, num_component*g_sv_dim*g_sv_dim, VOL3, 1, append);
      }
  
      /***********************************************
       * save output data in connq_out
       ***********************************************/
      for(isnk=0;isnk<snk_momentum_no;isnk++) {
        ix = g_ipt[0][snk_momentum_list[isnk][0]][snk_momentum_list[isnk][1]][snk_momentum_list[isnk][2]];
        fprintf(stdout, "# [] sink momentum (%d, %d, %d) -> index %d\n", snk_momentum_list[isnk][0], snk_momentum_list[isnk][1], snk_momentum_list[isnk][2], ix);
        for(icomp=0;icomp<num_component; icomp++) {
          x1 = ( (imom * snk_momentum_no + isnk ) * num_component + icomp) * T + timeslice;
          _sp_eq_sp(connq_out[ x1 ], connq[ix*num_component+icomp]);
        }
      }

      /***********************************************
       * calculate connt
       ***********************************************/
      for(icomp=0;icomp<num_component; icomp++) {
        // fwd
        _sp_eq_sp(sp1[0], connq[icomp]);
        _sp_eq_gamma_ti_sp(sp2[0], 0, sp1[0]);
        _sp_pl_eq_sp(sp1[0], sp2[0]);
        _co_eq_tr_sp(&w, sp1[0]);
        connt[2*( (imom*2*num_component + icomp) * T + timeslice)  ] = w.re * 0.25;
        connt[2*( (imom*2*num_component + icomp) * T + timeslice)+1] = w.im * 0.25;
        // bwd
        _sp_eq_sp(sp1[0], connq[icomp]);
        _sp_eq_gamma_ti_sp(sp2[0], 0, sp1[0]);
        _sp_mi_eq_sp(sp1[0], sp2[0]);
        _co_eq_tr_sp(&w, sp1[0]);
        connt[2*( (imom*2*num_component + icomp + num_component ) * T + timeslice)  ] = w.re * 0.25;
        connt[2*( (imom*2*num_component + icomp + num_component ) * T + timeslice)+1] = w.im * 0.25;
      }

    }  // of loop on relative momenta

  }  // of loop on timeslice

  // write conq_out
  count=0;
  for(imom=0;imom<rel_momentum_no;imom++) {
    sprintf(filename, "%s_snk.%.4d.t%.2dx%.2dy%.2dz%.2d.qx%.2dqy%.2dqz%.2d", outfile_prefix, Nconf, sx0, sx1, sx2, sx3,
        rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
    ofs = fopen(filename, "w");
    fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d%3d%3d%3d\n", g_kappa, T_global, LX, LY, LZ, g_mu, Nconf,
        rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
    if(ofs == NULL) {
      fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
      exit(32);
    }
    for(isnk=0;isnk<snk_momentum_no;isnk++) {
      for(icomp=0;icomp<num_component;icomp++) {
        for(timeslice=0;timeslice<T;timeslice++) {
          for(ir=0;ir<g_sv_dim*g_sv_dim;ir++) {
            fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp],timeslice,
                connq_out[count][0][2*ir], connq_out[count][0][2*ir+1],
                snk_momentum_list[isnk][0],snk_momentum_list[isnk][1],snk_momentum_list[isnk][2]);
          }  // of ir
          count++;
        }    // of timeslice
      }      // of icomp
    }        // of isnk
    fclose(ofs); ofs = NULL;
  }
  

  // write connt
  sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.fw", outfile_prefix, Nconf, sx0, sx1, sx2, sx3);
  ofs = fopen(filename, "w");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
    exit(3);
  }
 
  for(imom=0;imom<rel_momentum_no;imom++) {
    fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d%3d%3d%3d\n", g_kappa, T_global, LX, LY, LZ, g_mu, Nconf,
        rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);

    for(icomp=0; icomp<num_component; icomp++) {
//      ir = sx0;
//      fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp], 0, connt[2*((imom*2*num_component+icomp)*T+ir)], 0., Nconf,
//          rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
//      for(it=1;it<T/2;it++) {
//        ir  = ( it + sx0 ) % T_global;
//       ir2 = ( (T_global - it) + sx0 ) % T_global;
//        fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp], it,
//            connt[2*((imom*2*num_component+icomp)*T+ir)], connt[2*((imom*2*num_component+icomp)*T+ir2)], Nconf,
//            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
//      }
//      ir = ( it + sx0 ) % T_global;
//      fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp], it, connt[2*((imom*2*num_component+icomp)*T+ir)], 0., Nconf,
//          rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
      for(it=0;it<T;it++) {
        ir  = ( it + sx0 ) % T_global;
        fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp], it,
            connt[2*((imom*2*num_component+icomp)*T+ir)], connt[2*((imom*2*num_component+icomp)*T+ir)+1], Nconf,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
      }
    }
  }
  fclose(ofs);
  
  sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.bw", outfile_prefix, Nconf, sx0, sx1, sx2, sx3);
  ofs = fopen(filename, "w");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
    exit(3);
  }

  for(imom=0;imom<rel_momentum_no;imom++) {
    fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d%3d%3d%3d\n", g_kappa, T_global, LX, LY, LZ, g_mu, Nconf,
        rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
  
    for(icomp=0; icomp<num_component; icomp++) {
/*
      ir = sx0;
      fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp], 0,
          connt[2*((imom*2*num_component+num_component+icomp)*T+ir)], 0., Nconf,
          rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
      for(it=1;it<T/2;it++) {
        ir  = ( it + sx0 ) % T_global;
        ir2 = ( (T_global - it) + sx0 ) % T_global;
        fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp], it,
            connt[2*((imom*2*num_component+num_component+icomp)*T+ir)], connt[2*((imom*2*num_component+num_component+icomp)*T+ir2)], Nconf,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
      }
      ir = ( it + sx0 ) % T_global;
      fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp], it,
          connt[2*((imom*2*num_component+num_component+icomp)*T+ir)], 0., Nconf,
          rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
*/
      for(it=0;it<T;it++) {
        ir  = ( it + sx0 ) % T_global;
        fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n", gamma_component[0][icomp], gamma_component[1][icomp], it,
            connt[2*((imom*2*num_component+num_component+icomp)*T+ir)  ],
            connt[2*((imom*2*num_component+num_component+icomp)*T+ir)+1], Nconf,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
      }
    }
  }
  fclose(ofs);

  if(in!=NULL) free(in);
  fftwnd_destroy_plan(plan_p);

  // create the fermion propagator points
  for(i=0;i<g_num_threads;i++) {
    free_fp( uprop+i );
    free_fp( dprop+i );
    free_fp( fp1+i );
    free_fp( fp2+i );
    free_fp( fp3+i );
    free_fp( fp4+i );
    free_fp( fpaux+i );
    free_sp( sp1+i );
    free_sp( sp2+i );
  }
  free(uprop);
  free(dprop);
  free(fp1);
  free(fp2);
  free(fp3);
  free(fp4);
  free(fpaux);
  free(sp1);
  free(sp2);
}  // of if mode == 2


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
  if(seq_spinor_field_checksum !=NULL) free(seq_spinor_field_checksum);
  if(g_gauge_field != NULL) free(g_gauge_field);

  if(rel_momentum_list!=NULL) {
    if(rel_momentum_list[0]!=NULL) free(rel_momentum_list[0]);
    free(rel_momentum_list);
  }
  if(snk_momentum_list!=NULL) {
    if(snk_momentum_list[0]!=NULL) free(snk_momentum_list[0]);
    free(snk_momentum_list);
  }

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
