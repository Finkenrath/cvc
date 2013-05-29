/****************************************************
 * sigma_ps_2_pi_lambda_sequential_v4_mpi.c
 *
 * Di 7. Mai 11:08:07 EEST 2013
 *
 * PURPOSE
 * - originally copied from delta_pp_2_pi_N_sequential_v4_mpi.c
 * - contractions for 3-point function <Sigma^{*+} pi^+^\dagger Lambda^0^\dagger>
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
  EXIT(0);
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
  void *buffer=NULL;
  int verbose = 0;
  int sx0, sx1, sx2, sx3;
  int write_ascii=0;
  int fermion_type = _WILSON_FERMION;  // Wilson fermion type
  int smear_seq_source = 0;
  int threadid;
  char filename[200], contype[200], gauge_field_filename[200], line[200];
  double ratime, retime;
  double plaq_m, plaq_r;
  int mode = -1;
  double *work=NULL;
  fermion_propagator_type *fp1=NULL, *fp2=NULL, *fp3=NULL, *fp4=NULL, *fp5=NULL, *fpaux=NULL, *uprop=NULL, *dprop=NULL, *sprop=NULL;
  spinor_propagator_type *sp1=NULL, *sp2=NULL, *sp3=NULL, *sp4=NULL, *sp5=NULL, *sp6=NULL, *sp7=NULL, *sp8=NULL, *sp9=NULL, *sp_aux=NULL,;
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
//#ifdef MPI
//   fftwnd_mpi_plan plan_p;
//#else
   fftwnd_plan plan_p;
//#endif 

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "Sah?vgf:F:P:s:m:")) != -1) {
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
        EXIT(145);
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



#if (defined PARALLELTX) || (defined PARALLELTXY)
  fprintf(stderr, "[] Error, 2-,3-dim. parallel version not yet implemented; exit\n");
  EXIT(1);
#endif

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
#else
  fprintf(stdout, "[delta_pp_2_pi_N_sequential_v4_mpi] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  // initialize MPI parameters
  mpi_init(argc, argv);

#ifdef OPENMP
  status = fftw_threads_init();
  if(status != 0) {
    fprintf(stderr, "\n[] Error from fftw_init_threads; status was %d\n", status);
    EXIT(120);
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
    EXIT(1);
  }

  geometry();

  if(N_Jacobi>0) {

    // alloc the gauge field
    alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
    switch(g_gauge_file_format) {
      case 0:
        sprintf(gauge_field_filename, "%s.%.4d", gaugefilename_prefix, Nconf);
        status = read_lime_gauge_field_doubleprec(gauge_field_filename);
        break;
      case 1:
        sprintf(gauge_field_filename, "%s.%.5d", gaugefilename_prefix, Nconf);
        status = read_nersc_gauge_field(g_gauge_field, gauge_field_filename, &plaq_r);
        break;
    }

    xchange_gauge();
    plaquette(&plaq_m);
    if(g_cart_id==0) {
      fprintf(stdout, "# [] read plaquette value = %25.16e\n", plaq_r);
      fprintf(stdout, "# [] measured plaquette value = %25.16e\n", plaq_m);
    }

    if(N_ape > 0) {
      if(g_cart_id==0) fprintf(stdout, "# [delta_pp_2_pi_N_sequential_v4_mpi] APE smearing gauge field with paramters N_APE=%d, alpha_APE=%e\n", N_ape, alpha_ape);
#ifdef OPENMP
      APE_Smearing_Step_threads(g_gauge_field, N_ape, alpha_ape);
#else
      for(i=0; i<N_ape; i++) {
        APE_Smearing_Step(g_gauge_field, alpha_ape);
      }
#endif
      xchange_gauge_field(g_gauge_field);
    }
    plaquette(&plaq_m);
    if(g_cart_id==0) {
      fprintf(stdout, "# [] measured plaquette value after smearing = %25.16e\n", plaq_m);
    }
  } else {
    g_gauge_field = NULL;
  }

  // determine the source location
  sx0 = g_source_location / ( LX_global*LY_global*LZ_global );
  sx1 = (g_source_location % ( LX_global*LY_global*LZ_global ) ) / (LY_global*LZ_global);
  sx2 = (g_source_location % ( LY_global*LZ_global ) ) / LZ_global;
  sx3 = (g_source_location % LZ_global);

  if(g_cart_id==0) fprintf(stdout, "# [] global source location %d = (%d,%d,%d,%d)\n",
      g_source_location, sx0, sx1, sx2, sx3);

  if(g_source_momentum_set) {
    if(g_source_momentum[0]<0) g_source_momentum[0] += LX_global;
    if(g_source_momentum[1]<0) g_source_momentum[1] += LY_global;
    if(g_source_momentum[2]<0) g_source_momentum[2] += LZ_global;
    if(g_cart_id==0) fprintf(stdout, "# [] using final source momentum ( %d, %d, %d )\n", 
        g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
  }


  /***************************************************************************
   * set the relative momentum data by hand
   ***************************************************************************/
  rel_momentum_no = 1;
  rel_momentum_list = (int**)malloc(sizeof(int*));
  *rel_momentum_list = (int*)malloc(3*sizeof(int));
  (*rel_momentum_list)[0] = g_source_momentum[0];
  (*rel_momentum_list)[1] = g_source_momentum[1];
  (*rel_momentum_list)[2] = g_source_momentum[2];
  

  if(mode == 2) {
    /***************************************************************************
     * read the nucleon final momenta to be used
     ***************************************************************************/
    ofs = fopen(snk_momentum_filename, "r");
    if(ofs == NULL) {
      fprintf(stderr, "[] Error, could not open file %s for reading\n", snk_momentum_filename);
      EXIT(6);
    }
    snk_momentum_no = 0;
    while( fgets(line, 199, ofs) != NULL) {
      if(line[0] != '#') {
        snk_momentum_no++;
      }
    }
    if(snk_momentum_no == 0) {
      if(g_cart_id==0) fprintf(stderr, "[] Error, number of momenta is zero\n");
      EXIT(7);
    } else {
      if(g_cart_id==0) fprintf(stdout, "# [] number of nucleon final momenta = %d\n", snk_momentum_no);
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
    if(g_cart_id==0) fprintf(stdout, "# [] the nucleon final momentum list:\n");
    for(i=0;i<snk_momentum_no;i++) {
      if(snk_momentum_list[i][0]<0) snk_momentum_list[i][0] += LX_global;
      if(snk_momentum_list[i][1]<0) snk_momentum_list[i][1] += LY_global;
      if(snk_momentum_list[i][2]<0) snk_momentum_list[i][2] += LZ_global;
      if(g_cart_id==0) fprintf(stdout, "\t%3d%3d%3d%3d\n", i, snk_momentum_list[i][0], snk_momentum_list[i][1], snk_momentum_list[i][2]);
    }
  }  // of if mode == 2

  // allocate memory for the spinor fields
  g_spinor_field = NULL;
  if(mode == 2) {
    no_fields = 3*n_s*n_c;
    if(N_Jacobi>0) no_fields++;
    g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
    for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);
    if(N_Jacobi>0) work = g_spinor_field[no_fields-1];
  }
  
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
      EXIT(2);
    }
    for(ix=0; ix<items; ix++) connt[ix] = 0.;
  
    items = num_component * (size_t)VOL3;
    connq = create_sp_field( items );
    if(connq == NULL) {
      fprintf(stderr, "\n[] Error, could not alloc connq\n");
      EXIT(2);
    }
  
    items = (size_t)rel_momentum_no * (size_t)num_component * (size_t)T * (size_t)snk_momentum_no;
    connq_out = create_sp_field( items );
    if(connq_out == NULL) {
      fprintf(stderr, "\n[] Error, could not alloc connq_out\n");
      EXIT(22);
    }
  
    // initialize FFTW
    items = 2 * num_component * g_sv_dim * g_sv_dim * VOL3;
    bytes = sizeof(double);
    in  = (fftw_complex*)malloc(num_component*g_sv_dim*g_sv_dim*VOL3*sizeof(fftw_complex));
    if(in == NULL) {
      fprintf(stderr, "[] Error, could not malloc in for FFTW\n");
      EXIT(155);
    }
    dims[0]=LX; dims[1]=LY; dims[2]=LZ;
    //plan_p = fftwnd_create_plan(3, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
    plan_p = fftwnd_create_plan_specific(3, dims, FFTW_FORWARD, FFTW_MEASURE, in, num_component*g_sv_dim*g_sv_dim, (fftw_complex*)( connq[0][0] ), num_component*g_sv_dim*g_sv_dim);
  
    // create the fermion propagator points
    uprop = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(uprop== NULL) {
      fprintf(stdout, "[] Error, could not alloc uprop\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(uprop+i);
    }

    dprop = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(dprop== NULL) {
      fprintf(stdout, "[] Error, could not alloc dprop\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(dprop+i);
    }

    sprop = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(sprop== NULL) {
      fprintf(stdout, "[] Error, could not alloc sprop\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(sprop+i);
    }

    fp1 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(fp1== NULL) {
      fprintf(stdout, "[] Error, could not alloc fp1\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(fp1+i);
    }
    fp2 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(fp2== NULL) {
      fprintf(stdout, "[] Error, could not alloc fp2\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(fp2+i);
    }
    fp3 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(fp3== NULL) {
      fprintf(stdout, "[] Error, could not alloc fp3\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(fp3+i);
    }
    fp4 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(fp4== NULL) {
      fprintf(stdout, "[] Error, could not alloc fp4\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(fp4+i);
    }

    fp5 = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(fp5== NULL) {
      fprintf(stdout, "[] Error, could not alloc fp5\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(fp5+i);
    }

    fpaux = (fermion_propagator_type*)malloc(g_num_threads*sizeof(fermion_propagator_type));
    if(fpaux== NULL) {
      fprintf(stdout, "[] Error, could not alloc fpaux\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_fp(fpaux+i);
    }
    sp1 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp1== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp1\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp1+i);
    }

    sp2 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp2== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp2\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp2+i);
    }
  
    sp3 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp3== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp3\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp3+i);
    }
  
    sp4 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp4== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp4\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp4+i);
    }
  
    sp5 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp5== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp5\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp5+i);
    }
  
    sp6 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp6== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp6\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp6+i);
    }
  
    sp7 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp7== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp7\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp7+i);
    }
  
    sp8 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp8== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp8\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp8+i);
    }
  
    sp9 = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp9== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp9\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp9+i);
    }
  
    sp_aux = (spinor_propagator_type*)malloc(g_num_threads*sizeof(spinor_propagator_type));
    if(sp_aux== NULL) {
      fprintf(stdout, "[] Error, could not alloc sp_aux\n");
      EXIT(172);
    } else {
      for(i=0;i<g_num_threads;i++) create_sp(sp_aux+i);
    }
  
    // read the 12 up-type propagators and smear them
    for(is=0;is<n_s*n_c;is++) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix, Nconf, sx0, sx1, sx2, sx3, is);
      status = read_lime_spinor(g_spinor_field[is], filename, 0);
      if(status != 0) {
        fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
        EXIT(102);
      }
      if(N_Jacobi > 0) {
        if(g_cart_id==0) fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
            is, N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
        Jacobi_Smearing_Step_one_threads(g_gauge_field, g_spinor_field[is], work, N_Jacobi, kappa_Jacobi);
#else
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(g_gauge_field, g_spinor_field[is], work, kappa_Jacobi);
        }
#endif
      }
    }
  
    // read the 12 s-type propagators and smear them
    for(is=0;is<n_s*n_c;is++) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix3, Nconf, sx0, sx1, sx2, sx3, is);
      status = read_lime_spinor(g_spinor_field[is+2*n_s*n_c], filename, 0);
      if(status != 0) {
        fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
        EXIT(102);
      }
      if(N_Jacobi > 0) {
        if(g_cart_id==0) fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
            is, N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
        Jacobi_Smearing_Step_one_threads(g_gauge_field, g_spinor_field[is+2*n_s*n_c], work, N_Jacobi, kappa_Jacobi);
#else
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(g_gauge_field, g_spinor_field[is+2*n_s*n_c], work, kappa_Jacobi);
        }
#endif
      }
    }
  
    /******************************************************
     * loop on relative momenta
     ******************************************************/
    for(imom=0;imom<rel_momentum_no; imom++) {
  
      // read 12 sequential propagators
      for(is=0;is<n_s*n_c;is++) {
        sprintf(filename, "seq_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.inverted",
            filename_prefix, Nconf, sx0, sx1, sx2, sx3, is,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
        status = read_lime_spinor(g_spinor_field[n_s*n_c+is], filename, 0);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          EXIT(102);
        }
        if(N_Jacobi > 0) {
          if(g_cart_id==0) fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
              is, N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
          Jacobi_Smearing_Step_one_threads(g_gauge_field, g_spinor_field[n_s*n_c+is], work, N_Jacobi, kappa_Jacobi);
#else
          for(c=0; c<N_Jacobi; c++) {
            Jacobi_Smearing_Step_one(g_gauge_field, g_spinor_field[n_s*n_c+is], work, kappa_Jacobi);
          }
#endif
        }
      }
  
      /******************************************************
       * loop on timeslices
       ******************************************************/
      for(timeslice=0; timeslice<T; timeslice++)
      {
        append = (int)( timeslice != 0 );
   
        if(g_cart_id==0) fprintf(stdout, "# [] processing timeslice no. %d\n", timeslice);


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
#pragma omp parallel private (ix,iix,icomp,threadid) \
        firstprivate (fermion_type,gamma_component,connq,\
            gamma_component_sign,VOL3,g_spinor_field,fp1,fp2,fp3,fpaux,fp4,fp5,uprop,dprop,sprop,\
            sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp_aux,timeslice)
  //      shared (num_component)
  {
        threadid = omp_get_thread_num();
#else
        threadid = 0;
#endif
        for(ix=threadid; ix<VOL3; ix+=g_num_threads)
        {
          iix = timeslice * VOL3 + ix;
          // assign the propagators
          _assign_fp_point_from_field(uprop[threadid], g_spinor_field, iix);

          _assign_fp_point_from_field(dprop[threadid], g_spinor_field+n_s*n_c, iix);

          _assign_fp_point_from_field(sprop[threadid], g_spinor_field+2*n_s*n_c, iix);

          // flavor rotation for twisted mass fermions
          if(fermion_type == _TM_FERMION) {
            _fp_eq_rot_ti_fp(fp1[threadid], uprop[threadid], +1, fermion_type, fp2[threadid]);
            _fp_eq_fp_ti_rot(uprop[threadid], fp1[threadid], +1, fermion_type, fp2[threadid]);

            _fp_eq_rot_ti_fp(fp1[threadid], dprop[threadid], +1, fermion_type, fp2[threadid]);
            _fp_eq_fp_ti_rot(dprop[threadid], fp1[threadid], -1, fermion_type, fp2[threadid]);

            _fp_eq_rot_ti_fp(fp1[threadid], sprop[threadid], +1, fermion_type, fp2[threadid]);
            _fp_eq_fp_ti_rot(sprop[threadid], fp1[threadid], +1, fermion_type, fp2[threadid]);

          }
    
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
  
            // fp5[threadid] = S_s x C Gamma_2 = sprop[threadid] x g0 g2 Gamma_2
            _fp_eq_fp_ti_gamma(fp5[threadid],   0, sprop[threadid]);
            _fp_eq_fp_ti_gamma(fpaux[threadid], 2, fp5[threadid]);
            _fp_eq_fp_ti_gamma(fp5[threadid], gamma_component[1][icomp], fpaux[threadid]);
  
            // (1)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract24_fp(fpaux[threadid], dprop[threadid], fp2[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract34_fp(sp_aux[threadid], sprop[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp1[threadid], sp_aux[threadid], -4.);

            // (2)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract24_fp(fpaux[threadid], sprop[threadid], fp2[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract34_fp(sp_aux[threadid], dprop[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp2[threadid], sp_aux[threadid], +2.);

            // (3)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract24_fp(fpaux[threadid], fp4[threadid], fp5[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract34_fp(sp_aux[threadid], uprop[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp3[threadid], sp_aux[threadid], +2.);


            // (4)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp2[threadid], dprop[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract23_fp(sp_aux[threadid], sprop[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp4[threadid], sp_aux[threadid], +2.);

            // (5)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract13_fp(fpaux[threadid], dprop[threadid], fp1[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract23_fp(sp_aux[threadid], fp5[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp5[threadid], sp_aux[threadid], +2.);

            // (6)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp2[threadid], sprop[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract23_fp(sp_aux[threadid], dprop[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp6[threadid], sp_aux[threadid], +4.);

            // (7)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp4[threadid], fp1[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract23_fp(sp_aux[threadid], fp3[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp7[threadid], sp_aux[threadid], +4.);

            // (8)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract13_fp(fpaux[threadid], sprop[threadid], fp4[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract23_fp(sp_aux[threadid], fp3[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp8[threadid], sp_aux[threadid], +2.);

            // (9)
            // reduce
            _fp_eq_zero(fpaux[threadid]);
            _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp5[threadid], fp1[threadid]);
            // reduce to spin propagator
            _sp_eq_zero( sp_aux[threadid] );
            _sp_eq_fp_del_contract23_fp(sp_aux[threadid], dprop[threadid], fpaux[threadid]);

            _sp_eq_sp_ti_re(sp9[threadid], sp_aux[threadid], +2.);

            // add and assign
            _sp_pl_eq_sp(sp1[threadid], sp2[threadid]);
            _sp_pl_eq_sp(sp1[threadid], sp3[threadid]);
            _sp_pl_eq_sp(sp1[threadid], sp4[threadid]);
            _sp_pl_eq_sp(sp1[threadid], sp5[threadid]);
            _sp_pl_eq_sp(sp1[threadid], sp6[threadid]);
            _sp_pl_eq_sp(sp1[threadid], sp7[threadid]);
            _sp_pl_eq_sp(sp1[threadid], sp8[threadid]);
            _sp_pl_eq_sp(sp1[threadid], sp9[threadid]);

            _sp_eq_sp_ti_re(sp2[threadid], sp1[threadid], gamma_component_sign[icomp] * _ONE_OVER_THREE_TI_SQRT2);
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
          if(g_cart_id==0) fprintf(stdout, "# [] multiplying timeslice %d with boundary phase factor\n", timeslice);
          ir = (timeslice + g_proc_coords[0]*T - sx0 + T_global) % T_global;
          w1.re = cos( 3. * M_PI*(double)ir / (double)T_global );
          w1.im = sin( 3. * M_PI*(double)ir / (double)T_global );
          for(ix=0;ix<num_component*VOL3;ix++) {
            _sp_eq_sp(sp1[0], connq[ix] );
            _sp_eq_sp_ti_co( connq[ix], sp1[0], w1);
          }
        } else if (g_propagator_bc_type == 1) {
          // multiply with step function
          if(timeslice+g_proc_coords[0]*T < sx0) {
            if(g_cart_id==0) fprintf(stdout, "# [] multiplying timeslice %d with boundary step function\n", timeslice);
            for(ix=0;ix<num_component*VOL3;ix++) {
              _sp_eq_sp(sp1[0], connq[ix] );
              _sp_eq_sp_ti_re( connq[ix], sp1[0], -1.);
            }
          }
        }
#ifndef MPI
        if(write_ascii) {
          sprintf(filename, "%s_x.%.4d.t%.2dx%.2dy%.2dz%.2d.qx%.2dqy%.2dqz%.2d.ascii",
              outfile_prefix, Nconf, sx0, sx1, sx2, sx3,
              rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
          write_contraction2( connq[0][0], filename, num_component*g_sv_dim*g_sv_dim, VOL3, 1, append);
        }
#endif
  
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
          q[0] = (double)(x1+g_proc_coords[1]*LX) / (double)LX_global;
        for(x2=0;x2<LY;x2++) {
          q[1] = (double)(x2+g_proc_coords[2]*LY) / (double)LY_global;
        for(x3=0;x3<LZ;x3++) {
          q[2] = (double)(x3+g_proc_coords[3]*LZ) / (double)LZ_global;
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
          if(g_cart_id==0 && timeslice==0) fprintf(stdout, "# [] sink momentum (%d, %d, %d) -> index %d\n", snk_momentum_list[isnk][0], snk_momentum_list[isnk][1], snk_momentum_list[isnk][2], ix);
          for(icomp=0;icomp<num_component; icomp++) {
            x1 = timeslice*rel_momentum_no*snk_momentum_no*num_component + (imom * snk_momentum_no + isnk ) * num_component + icomp;
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
          x1 = timeslice*rel_momentum_no*num_component + imom*num_component + icomp;
          connt[2*x1  ] = w.re * 0.25;
          connt[2*x1+1] = w.im * 0.25;
          // bwd
          _sp_eq_sp(sp1[0], connq[icomp]);
          _sp_eq_gamma_ti_sp(sp2[0], 0, sp1[0]);
          _sp_mi_eq_sp(sp1[0], sp2[0]);
          _co_eq_tr_sp(&w, sp1[0]);
          x1 = (timeslice+T)*rel_momentum_no*num_component + imom*num_component + icomp;
          connt[2*x1  ] = w.re * 0.25;
          connt[2*x1+1] = w.im * 0.25;
        }
  
      }  // of loop on timeslice
  
    }    // of loop on relative momenta

    // free the fermion propagator points
    for(i=0;i<g_num_threads;i++) {
      free_fp( uprop+i );
      free_fp( dprop+i );
      free_fp( fp1+i );
      free_fp( fp2+i );
      free_fp( fp3+i );
      free_fp( fp4+i );
      free_fp( fp5+i );
      free_fp( fpaux+i );
      free_sp( sp1+i ); free_sp( sp2+i ); free_sp( sp3+i );
      free_sp( sp4+i ); free_sp( sp5+i ); free_sp( sp6+i );
      free_sp( sp7+i ); free_sp( sp8+i ); free_sp( sp9+i );
      free_sp( sp_aux+i );
    }
    free(uprop);
    free(dprop);
    free(sprop);
    free(fp1); free(fp2); free(fp3); free(fp4); free(fp5); free(fpaux);
    free(sp1); free(sp2); free(sp3); free(sp4);
    free(sp5); free(sp6); free(sp7); free(sp8);
    free(sp9); free(sp_aux);

    // write connq_out
    items = (size_t)rel_momentum_no * (size_t)num_component * (size_t)T_global * (size_t)snk_momentum_no;
    if( (buffer = (void*)create_sp_field( items ) ) == NULL ) {
      fprintf(stderr, "[] Error, could not allocate buffer; exit\n");
      EXIT(152);
    }
    sp1 = buffer;
    count = rel_momentum_no * num_component * T * snk_momentum_no * 2*g_sv_dim*g_sv_dim;
#ifdef MPI
    status = MPI_Gather(connq_out[0][0], count, MPI_DOUBLE, sp1[0][0], count, MPI_DOUBLE, 0, g_cart_grid);
    if( status != MPI_SUCCESS) {
      fprintf(stderr, "[] Error from MPI_Gather; exit\n");
      EXIT(153);
    }
#else
    memcpy(sp1[0][0], connq_out[0][0], count*sizeof(double));
#endif

    if (g_cart_id == 0) {
      count=0;
      for(imom=0;imom<rel_momentum_no;imom++) {
        sprintf(filename, "%s_snk.%.4d.t%.2dx%.2dy%.2dz%.2d.qx%.2dqy%.2dqz%.2d", outfile_prefix, Nconf,
            sx0, sx1, sx2, sx3,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
        ofs = fopen(filename, "w");
        fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d%3d%3d%3d\n", g_kappa, T_global, LX_global, LY_global,
            LZ_global, g_mu, Nconf,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
        if(ofs == NULL) {
          fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
          EXIT(32);
        }
        for(isnk=0;isnk<snk_momentum_no;isnk++) {
          for(icomp=0;icomp<num_component;icomp++) {
            for(timeslice=0;timeslice<T_global;timeslice++) {
              x1 = ((timeslice*rel_momentum_no + imom)*snk_momentum_no+isnk)*num_component+icomp;
              for(ir=0;ir<g_sv_dim*g_sv_dim;ir++) {
                fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%3d%3d%3d\n",
                    gamma_component[0][icomp], gamma_component[1][icomp],timeslice,
                    sp1[x1][0][2*ir], sp1[x1][0][2*ir+1],
                    snk_momentum_list[isnk][0],snk_momentum_list[isnk][1],snk_momentum_list[isnk][2]);
              }  // of ir
              count++;
            }    // of timeslice
          }      // of icomp
        }        // of isnk
        fclose(ofs); ofs = NULL;
      }
    }
    free_sp_field( &sp1 ); buffer = NULL;

    // write connt
    items = 2 * rel_momentum_no * num_component * T_global;
    if( (buffer = malloc(items*sizeof(double))) == NULL ) {
      fprintf(stderr, "[] Error, could not allocate buffer; exit\n");
      EXIT(153);
    }
  
    // forward
    count = 2 * rel_momentum_no * num_component * T;
#ifdef MPI
    status = MPI_Gather(connt, count, MPI_DOUBLE, buffer, count, MPI_DOUBLE, 0, g_cart_grid);
    if( status != MPI_SUCCESS ) {
      fprintf(stderr, "[] Error from MPI_Gather; exit\n");
      EXIT(154);
    }
#else
    memcpy(buffer, connt, count*sizeof(double));
#endif
    if (g_cart_id == 0) {  
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.fw", outfile_prefix, Nconf, sx0, sx1, sx2, sx3);
      ofs = fopen(filename, "w");
      if(ofs == NULL) {
        fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
        EXIT(3);
      }
     
      for(imom=0;imom<rel_momentum_no;imom++) {
        fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d%3d%3d%3d\n", g_kappa, T_global, LX_global, LY_global, LZ_global, g_mu, Nconf,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
    
        for(icomp=0; icomp<num_component; icomp++) {
          for(it=0;it<T_global;it++) {
            ir  = ( it + sx0 ) % T_global;
            x1 = (ir*rel_momentum_no + imom)*num_component + icomp;
            fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n",
                gamma_component[0][icomp], gamma_component[1][icomp], it,
                ((double*)buffer)[2*x1  ], ((double*)buffer)[2*x1+1], Nconf,
                rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
          }
        }
      }
      fclose(ofs);
    }  // of if g_cart_id == 0 
  
    // backward
    count = 2 * rel_momentum_no * num_component * T;
#ifdef MPI
    if( MPI_Gather(connt+count, count, MPI_DOUBLE, buffer, count, MPI_DOUBLE, 0, g_cart_grid) != MPI_SUCCESS) {
      fprintf(stderr, "[] Error from MPI_Gather; exit\n");
      EXIT(155);
    }
#else
    memcpy(buffer, connt+count, count*sizeof(double));
#endif

    if(g_cart_id == 0) {
  
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.bw", outfile_prefix, Nconf, sx0, sx1, sx2, sx3);
      ofs = fopen(filename, "w");
      if(ofs == NULL) {
        fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
        EXIT(3);
      }
  
      for(imom=0;imom<rel_momentum_no;imom++) {
        fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d%3d%3d%3d\n", g_kappa, T_global, LX_global, LY_global, LZ_global,
            g_mu, Nconf,
            rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
    
        for(icomp=0; icomp<num_component; icomp++) {
          for(it=0;it<T_global;it++) {
            ir  = ( it + sx0 ) % T_global;
            x1 = (ir*rel_momentum_no + imom)*num_component + icomp;
            fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d%3d%3d%3d\n",
                gamma_component[0][icomp], gamma_component[1][icomp], it,
                ((double*)buffer)[2*x1  ], ((double*)buffer)[2*x1+1], Nconf,
                rel_momentum_list[imom][0],rel_momentum_list[imom][1],rel_momentum_list[imom][2]);
          }
        }
      }
      fclose(ofs);
    }
    free(buffer); buffer = NULL;

    if(in!=NULL) free(in);
    fftwnd_destroy_plan(plan_p);
  
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

  if(connq_out != NULL) free_sp_field(&connq_out);

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [] %s# [] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [] %s# [] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}
