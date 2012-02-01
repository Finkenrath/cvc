/****************************************************
 * delta_pp_2pt_v4.c
 *
 * Fri Dec 30 20:55:22 EET 2011
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
  const char outfile_prefix[] = "delta_pp_2pt_v4";

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
  int pos;
  char filename[200], contype[200], gauge_field_filename[200];
  double ratime, retime;
  //double plaq_m, plaq_r;
  double *work=NULL;
  fermion_propagator_type *fp1=NULL, *fp2=NULL, *fp3=NULL, *uprop=NULL, *dprop=NULL, *fpaux=NULL;
  spinor_propagator_type *sp1=NULL, *sp2=NULL;
  double q[3], phase, *gauge_trafo=NULL;
  complex w, w1;
  size_t items, bytes;
  FILE *ofs;
  int timeslice;
  DML_Checksum ildg_gauge_field_checksum, *spinor_field_checksum=NULL, connq_checksum;
  uint32_t nersc_gauge_field_checksum;
  int threadid, nthreads;

/*******************************************************************
 * Gamma components for the Delta:
 *                                                                 */
  const int num_component = 4;
  int gamma_component[2][4] = { {0, 1, 2, 3},
                                {0, 1, 2, 3} };
  double gamma_component_sign[4] = {+1.,+1.,-1.,+1.};
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

  while ((c = getopt(argc, argv, "ah?vgf:F:")) != -1) {
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
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
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

  // allocate memory for the spinor fields
  g_spinor_field = NULL;
  no_fields = n_s*n_c;
//  if(fermion_type == _TM_FERMION) {
//    no_fields *= 2;
//  }
  if(N_Jacobi>0) no_fields++;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOL3);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOL3);
  work = g_spinor_field[no_fields-1];

  spinor_field_checksum = (DML_Checksum*)malloc(no_fields * sizeof(DML_Checksum) );
  if(spinor_field_checksum == NULL ) {
    fprintf(stderr, "[] Error, could not alloc checksums for spinor fields\n");
    exit(73);
  }

  // allocate memory for the contractions
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


  /******************************************************
   * initialize FFTW
   ******************************************************/
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

  uprop = (fermion_propagator_type*)malloc(g_num_threads * sizeof(fermion_propagator_type) );
  fp1   = (fermion_propagator_type*)malloc(g_num_threads * sizeof(fermion_propagator_type) );
  fp2   = (fermion_propagator_type*)malloc(g_num_threads * sizeof(fermion_propagator_type) );
  fp3   = (fermion_propagator_type*)malloc(g_num_threads * sizeof(fermion_propagator_type) );
  fpaux = (fermion_propagator_type*)malloc(g_num_threads * sizeof(fermion_propagator_type) );
  if(uprop==NULL || fp1==NULL || fp2==NULL || fp3==NULL || fpaux==NULL ) {
    fprintf(stderr, "[] Error, could not alloc fermion propagator points\n");
    exit(57);
  }
  sp1 = (spinor_propagator_type*)malloc(g_num_threads * sizeof(spinor_propagator_type) ); 
  sp2 = (spinor_propagator_type*)malloc(g_num_threads * sizeof(spinor_propagator_type) ); 
  if(sp1==NULL || sp2==NULL) {
    fprintf(stderr, "[] Error, could not alloc spinor propagator points\n");
    exit(59);
  }
  for(i=0;i<g_num_threads;i++) { create_fp(uprop+i); }
  for(i=0;i<g_num_threads;i++) { create_fp(fp1+i); }
  for(i=0;i<g_num_threads;i++) { create_fp(fp2+i); }
  for(i=0;i<g_num_threads;i++) { create_fp(fp3+i); }
  for(i=0;i<g_num_threads;i++) { create_fp(fpaux+i); }
  for(i=0;i<g_num_threads;i++) { create_sp(sp1+i); }
  for(i=0;i<g_num_threads;i++) { create_sp(sp2+i); }

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

#ifdef OPENMP
      status = APE_Smearing_Step_Timeslice_threads(g_gauge_field, N_ape, alpha_ape);
#else
      for(i=0; i<N_ape; i++) { status = APE_Smearing_Step_Timeslice(g_gauge_field, alpha_ape); }
#endif

    }

    // read timeslice of the 12 up-type propagators and smear them
    for(is=0;is<n_s*n_c;is++) {
      if(do_gt == 0) {
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
      } else {  // of if do_gt == 0
        // apply gt
        apply_gt_prop(gauge_trafo, g_spinor_field[is], is/n_c, is%n_c, 4, filename_prefix, g_source_location);
      } // of if do_gt == 0
    }

    /******************************************************
     * contractions
     ******************************************************/
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private (ix,icomp,threadid) \
    firstprivate (fermion_type,gamma_component,num_component,connq,\
        gamma_component_sign,VOL3,g_spinor_field,fp1,fp2,fp3,fpaux,uprop,sp1,sp2)
{
    threadid = omp_get_thread_num();
#else
    threadid = 0;
#endif
    for(ix=threadid; ix<VOL3; ix+=g_num_threads)
    {
      // assign the propagators
      _assign_fp_point_from_field(uprop[threadid], g_spinor_field, ix);
      if(fermion_type == _TM_FERMION) {
        _fp_eq_rot_ti_fp(fp1[threadid], uprop[threadid], +1, fermion_type, fp2[threadid]);
        _fp_eq_fp_ti_rot(uprop[threadid], fp1[threadid], +1, fermion_type, fp2[threadid]);
      }

      for(icomp=0; icomp<num_component; icomp++) {

        _sp_eq_zero( connq[ix*num_component+icomp]);

        /******************************************************
         * prepare propagators
         ******************************************************/
        // fp1[threadid] = C Gamma_1 x S_u = g0 g2 Gamma_1 S_u
        _fp_eq_zero(fp1[threadid]);
        _fp_eq_zero(fpaux[threadid]);
        _fp_eq_gamma_ti_fp(fp1[threadid], gamma_component[0][icomp], uprop[threadid]);
        _fp_eq_gamma_ti_fp(fpaux[threadid], 2, fp1[threadid]);
        _fp_eq_gamma_ti_fp(fp1[threadid], 0, fpaux[threadid]);
        // fp2[threadid] = C Gamma_1 x S_u x C Gamma_2
        _fp_eq_zero(fp2[threadid]);
        _fp_eq_zero(fpaux[threadid]);
        _fp_eq_fp_ti_gamma(fp2[threadid], 0, fp1[threadid]);
        _fp_eq_fp_ti_gamma(fpaux[threadid], 2, fp2[threadid]);
        _fp_eq_fp_ti_gamma(fp2[threadid], gamma_component[1][icomp], fpaux[threadid]);
        // fp3[threadid] = S_u x C Gamma_2 = S_u g0 g2 Gamma_2
        _fp_eq_zero(fp3[threadid]);
        _fp_eq_zero(fpaux[threadid]);
        _fp_eq_fp_ti_gamma(fp3[threadid], 0, uprop[threadid]);
        _fp_eq_fp_ti_gamma(fpaux[threadid], 2, fp3[threadid]);
        _fp_eq_fp_ti_gamma(fp3[threadid], gamma_component[1][icomp], fpaux[threadid]);


        /******************************************************
         * contractions
         ******************************************************/
        // (1)
        // reduce
        _fp_eq_zero(fpaux[threadid]);
        _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp1[threadid], uprop[threadid]);
        // reduce to spin propagator
        _sp_eq_zero( sp1[threadid] );
        _sp_eq_fp_del_contract23_fp(sp1[threadid], fp3[threadid], fpaux[threadid]);
        // (2)
        // reduce to spin propagator
        _sp_eq_zero( sp2[threadid] );
        _sp_eq_fp_del_contract24_fp(sp2[threadid], fp3[threadid], fpaux[threadid]);
        // add and assign
        _sp_pl_eq_sp(sp1[threadid], sp2[threadid]);
        _sp_eq_sp_ti_re(sp2[threadid], sp1[threadid], -gamma_component_sign[icomp]);
        _sp_eq_sp( connq[ix*num_component+icomp], sp2[threadid]);

        // (3)
        // reduce
        _fp_eq_zero(fpaux[threadid]);
        _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp2[threadid], uprop[threadid]);
        // reduce to spin propagator
        _sp_eq_zero( sp1[threadid] );
        _sp_eq_fp_del_contract23_fp(sp1[threadid], uprop[threadid], fpaux[threadid]);
        // (4)
        // reduce
        _fp_eq_zero(fpaux[threadid]);
        _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp1[threadid], fp3[threadid]);
        // reduce to spin propagator
        _sp_eq_zero( sp2[threadid] );
        _sp_eq_fp_del_contract24_fp(sp2[threadid], uprop[threadid], fpaux[threadid]);
        // add and assign
        _sp_pl_eq_sp(sp1[threadid], sp2[threadid]);
        _sp_eq_sp_ti_re(sp2[threadid], sp1[threadid], -gamma_component_sign[icomp]);
        _sp_pl_eq_sp( connq[ix*num_component+icomp], sp2[threadid]);

        // (5)
        // reduce
        _fp_eq_zero(fpaux[threadid]);
        _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp2[threadid], uprop[threadid]);
        // reduce to spin propagator
        _sp_eq_zero( sp1[threadid] );
        _sp_eq_fp_del_contract34_fp(sp1[threadid], uprop[threadid], fpaux[threadid]);
        // (6)
        // reduce
        _fp_eq_zero(fpaux[threadid]);
        _fp_eq_fp_eps_contract13_fp(fpaux[threadid], fp1[threadid], fp3[threadid]);
        // reduce to spin propagator
        _sp_eq_zero( sp2[threadid] );
        _sp_eq_fp_del_contract34_fp(sp2[threadid], uprop[threadid], fpaux[threadid]);
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
    sprintf(filename, "%s_q.%.4d.t%.2dx%.2dy%.2dz%.2d", outfile_prefix, Nconf, sx0, sx1, sx2, sx3);
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
      _sp_eq_sp(sp1[0], connq[icomp]);
      _sp_eq_gamma_ti_sp(sp2[0], 0, sp1[0]);
      _sp_pl_eq_sp(sp1[0], sp2[0]);
      _co_eq_tr_sp(&w, sp1[0]);
      connt[2*(icomp*T + timeslice)  ] = w.re * 0.25;
      connt[2*(icomp*T + timeslice)+1] = w.im * 0.25;
      // bwd
      _sp_eq_sp(sp1[0], connq[icomp]);
      _sp_eq_gamma_ti_sp(sp2[0], 0, sp1[0]);
      _sp_mi_eq_sp(sp1[0], sp2[0]);
      _co_eq_tr_sp(&w, sp1[0]);
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

  for(i=0;i<g_num_threads;i++) { free_fp(uprop+i); }
  for(i=0;i<g_num_threads;i++) { free_fp(fp1+i); }
  for(i=0;i<g_num_threads;i++) { free_fp(fp2+i); }
  for(i=0;i<g_num_threads;i++) { free_fp(fp3+i); }
  for(i=0;i<g_num_threads;i++) { free_fp(fpaux+i); }
  for(i=0;i<g_num_threads;i++) { free_sp(sp1+i); }
  for(i=0;i<g_num_threads;i++) { free_sp(sp2+i); }
  if(uprop!=NULL) free(uprop);
  if(fp1!=NULL) free(fp1);
  if(fp2!=NULL) free(fp2);
  if(fp3!=NULL) free(fp3);
  if(fpaux!=NULL) free(fpaux);
  if(sp1!=NULL) free(sp1);
  if(sp2!=NULL) free(sp2);

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
