/****************************************************
 * proton_2pt_v3.c
 *
 * Sun Nov 20 12:44:25 EET 2011
 *
 * PURPOSE:
 * - calculate the proton 2-point function from point sources
 * - like proton_2pt, proton_2pt_v2;
 * - do not use specific projector; save all 4x4 spinor index combinations
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

  int c, i, j, ll, sl;
  int filename_set = 0;
  int mms1=0, status;
  int l_LX_at, l_LXstart_at;
  int ix, idx, it, iix, x1,x2,x3;
  int ir, ir1, ir2, ir3, iperm, is;
  int VOL3, ia0, ia1, ia2, ib;
  int do_gt=0;
  int dims[3];
  double *connt=NULL;
  spinor_propagator_type *connq=NULL;
  int verbose = 0;
  int sx0, sx1, sx2, sx3;
  int write_ascii=0;
  int fermion_type = 1;  // Wilson fermion type
  int pos;
  char filename[200], contype[200];
  double ratime, retime;
  double plaq_m, plaq_r, dsign, dtmp, dtmp2;
  double *gauge_field_timeslice=NULL, *gauge_field_f=NULL;
  double **chi=NULL, **chi2=NULL, **psi=NULL, **psi2=NULL, *work=NULL;
  double scs[18];
  fermion_propagator_type fp1=NULL, fp2=NULL, fp3=NULL, uprop=NULL, dprop=NULL;
  spinor_propagator_type sp1, sp2;
  double q[3], phase, *gauge_trafo=NULL;
  complex w, w1;
  size_t items, bytes;
  FILE *ofs;

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
  fprintf(stdout, "[proton_2pt_v3] Warning, resetting global thread number to 1\n");
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
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  if(N_Jacobi>0) {

    /* read the gauge field */
    alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
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
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 21);
      MPI_Finalize();
#endif
      exit(21);
    }
#ifdef MPI
    xchange_gauge();
#endif

    /* measure the plaquette */
    plaquette(&plaq_m);
    if(g_cart_id==0) fprintf(stdout, "# read plaquette value    : %25.16e\n", plaq_r);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq_m);

    if(N_ape>0) {
      if(g_cart_id==0) fprintf(stdout, "# apply APE smearing with parameters N_ape = %d, alpha_ape = %f\n", N_ape, alpha_ape);
      fprintf(stdout, "# [] APE smearing gauge field with paramters N_APE=%d, alpha_APE=%e\n", N_ape, alpha_ape);
#ifdef OPENMP
      APE_Smearing_Step_threads(g_gauge_field, N_ape, alpha_ape);
#else
      for(i=0; i<N_ape; i++) {
        APE_Smearing_Step(g_gauge_field, alpha_ape);
      }
#endif
    }
  } else {
    g_gauge_field = NULL;
  }  // of if N_Jacobi>0

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


  // initialize permutation tables
  init_perm_tabs();


  // allocate memory for the spinor fields
  g_spinor_field = NULL;
  no_fields = n_s*n_c;
  if(fermion_type == _TM_FERMION) {
    no_fields *= 2;
  }
  if(N_Jacobi>0) no_fields++;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUMEPLUSRAND);
  work = g_spinor_field[no_fields-1];

  // allocate memory for the contractions
  items = 2*2*T;
  bytes = sizeof(double);
  connt = (double*)malloc(items*bytes);
  if(connt == NULL) {
    fprintf(stderr, "\n[] Error, could not alloc connt\n");
    exit(2);
  }
  for(ix=0; ix<items; ix++) connt[ix] = 0.;

  items = (size_t)VOLUME;
  connq = create_sp_field( items );
  if(connq == NULL) {
    fprintf(stderr, "\n[] Error, could not alloc connq\n");
    exit(2);
  }


  /******************************************************
   * initialize FFTW
   ******************************************************/
  items = 2 * g_sv_dim * g_sv_dim * VOL3;
  bytes = sizeof(double);
  in  = (fftw_complex*)malloc(g_sv_dim*g_sv_dim*VOL3*sizeof(fftw_complex));
  if(in == NULL) {
    fprintf(stderr, "[] Error, could not malloc in for FFTW\n");
    exit(155);
  }
  dims[0]=LX; dims[1]=LY; dims[2]=LZ;
  //plan_p = fftwnd_create_plan(3, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_p = fftwnd_create_plan_specific(3, dims, FFTW_FORWARD, FFTW_MEASURE, in, g_sv_dim*g_sv_dim, (fftw_complex*)( connq[0][0] ), g_sv_dim*g_sv_dim);


  // read the 12 up-type propagators and smear them
  for(is=0;is<n_s*n_c;is++) {
    if(do_gt == 0) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix, Nconf, sx0, sx1, sx2, sx3, is);
      status = read_lime_spinor(g_spinor_field[is], filename, 0);
      if(status != 0) {
        fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
        exit(102);
      }
      if(N_Jacobi > 0) {
        fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
             is, N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
        Jacobi_Smearing_Step_one_threads(g_gauge_field, g_spinor_field[is], work, N_Jacobi, kappa_Jacobi);
#else
        for(c=0; c<N_Jacobi; c++) {
          Jacobi_Smearing_Step_one(g_gauge_field, g_spinor_field[is], work, kappa_Jacobi);
        }
#endif
      }
    } else {  // of if do_gt == 0
      // apply gt
      apply_gt_prop(gauge_trafo, g_spinor_field[is], is/n_c, is%n_c, 4, filename_prefix, g_source_location);
    } // of if do_gt == 0
  }

  if(fermion_type == _TM_FERMION) {
    // read 12 down-type propagators, smear them
    for(is=0;is<n_s*n_c;is++) {
      if(do_gt == 0) {
        sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", filename_prefix2, Nconf, sx0, sx1, sx2, sx3, is);
        status = read_lime_spinor(g_spinor_field[n_s*n_c+is], filename, 0);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          exit(102);
        }
        if(N_Jacobi > 0) {
          fprintf(stdout, "# [] Jacobi smearing propagator no. %d with paramters N_Jacobi=%d, kappa_Jacobi=%f\n",
               is, N_Jacobi, kappa_Jacobi);
#ifdef OPENMP
          Jacobi_Smearing_Step_one_threads(g_gauge_field, g_spinor_field[n_s*n_c+is], work, N_Jacobi, kappa_Jacobi);
#else
          for(c=0; c<N_Jacobi; c++) {
            Jacobi_Smearing_Step_one(g_gauge_field, g_spinor_field[n_s*n_c+is], work, kappa_Jacobi);
          }
#endif
        }
      } else {  // of if do_gt == 0
        // apply gt
        apply_gt_prop(gauge_trafo, g_spinor_field[n_s*n_c+is], is/n_c, is%n_c, 4, filename_prefix, g_source_location);
      } // of if do_gt == 0
    }
  }


  // create the fermion propagator points
  create_fp(&uprop);
  create_fp(&dprop);
  create_fp(&fp1);
  create_fp(&fp2);
  create_fp(&fp3);
  create_sp(&sp1);
  create_sp(&sp2);


  /******************************************************
   * contractions
   ******************************************************/
  for(ix=0;ix<VOLUME;ix++) 
  // for(ix=0;ix<2;ix++) 
  {

    // assign the propagators
    _assign_fp_point_from_field(uprop, g_spinor_field, ix);
    if(fermion_type==_TM_FERMION) {
      _assign_fp_point_from_field(dprop, g_spinor_field+n_s*n_c, ix);
    } else {
      _fp_eq_fp(dprop, uprop);
    }

    // flavor rotation for twisted mass fermions
    if(fermion_type == _TM_FERMION) {
      _fp_eq_rot_ti_fp(fp1, uprop, +1, fermion_type, fp2);
      _fp_eq_fp_ti_rot(uprop, fp1, +1, fermion_type, fp2);
      _fp_eq_rot_ti_fp(fp1, dprop, -1, fermion_type, fp2);
      _fp_eq_fp_ti_rot(dprop, fp1, -1, fermion_type, fp2);
    }

    // S_u x Cg5
    _fp_eq_fp_ti_Cg5(fp1, uprop, fp3);

    // Cg5 x S_d
    _fp_eq_Cg5_ti_fp(fp2, dprop, fp3);
    
    /******************************************************
     * first contribution
     ******************************************************/

    // reduce
    _fp_eq_zero(fp3);
    _fp_eq_fp_eps_contract13_fp(fp3, fp1, fp2);

    // reduce to spin propagator
    _sp_eq_zero( sp1 );
    _sp_eq_fp_del_contract34_fp(sp1, uprop, fp3);

    /******************************************************
     * second contribution
     ******************************************************/

    // reduce
    _fp_eq_zero(fp3);
    _fp_eq_fp_eps_contract24_fp(fp3, fp1, fp2);

    // reduce to spin propagator
    _sp_eq_zero( sp2 );
    _sp_eq_fp_del_contract23_fp(sp2, fp3, uprop);

    /******************************************************
     * add both contribution, write to field
     * - the second contribution receives additional minus
     *   from rearranging the epsilon tensor indices
     ******************************************************/

    _sp_pl_eq_sp(sp1, sp2);

    _sp_eq_sp( connq[ix], sp1);
  }

  /***********************************************
   * free gauge fields and spinor fields
   ***********************************************/
  if(g_gauge_field != NULL) {
    free(g_gauge_field);
    g_gauge_field=(double*)NULL;
  }
  if(g_spinor_field!=NULL) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field); g_spinor_field=(double**)NULL;
  }


  /***********************************************
   * finish calculation of connq
   ***********************************************/
  if(g_propagator_bc_type == 0) {
    // multiply with phase factor
    fprintf(stdout, "# [] multiplying with boundary phase factor\n");
    iix = 0;
    for(it=0;it<T_global;it++) {
      ir = (it - sx0 + T_global) % T_global;
      w1.re = cos( 3. * M_PI*(double)ir / (double)T_global );
      w1.im = sin( 3. * M_PI*(double)ir / (double)T_global );
      for(ix=0;ix<VOL3;ix++) {
        _sp_eq_sp(sp1, connq[iix] );
        _sp_eq_sp_ti_co( connq[iix], sp1, w1);
        iix++;
      }
    }
  } else if (g_propagator_bc_type == 1) {
    // multiply with step function
    fprintf(stdout, "# [] multiplying with boundary step function\n");
    for(it=0;it<sx0;it++) {
      iix = it * VOL3;
       for(ix=0;ix<VOL3;ix++) {
        _sp_eq_sp(sp1, connq[iix] );
        _sp_eq_sp_ti_re( connq[iix], sp1, -1.);
        iix++;
      }
    }
  }

  if(write_ascii) {
    sprintf(filename, "proton_2pt_v3_x.%.4d.t%.2dx%.2dy%.2dz%.2d.ascii", Nconf, sx0, sx1, sx2, sx3);
    write_contraction(connq[0][0], (int*)NULL, filename, g_sv_dim*g_sv_dim, 1, 0);
  }

  /******************************************************************
   * Fourier transform
   ******************************************************************/
  items = 2 * g_sv_dim * g_sv_dim * VOL3;
  bytes = sizeof(double);

  for(it=0;it<T;it++) {
    memcpy(in, connq[0][0] + it * items, items * bytes);
#ifdef OPENMP
    // fftwnd_threads_one(g_num_threads, plan_p, in, NULL);
    fftwnd_threads(g_num_threads, plan_p, g_sv_dim*g_sv_dim, in, g_sv_dim*g_sv_dim, 1, (fftw_complex*)(connq[0][0]+it*items), g_sv_dim*g_sv_dim, 1);
#else
    // fftwnd_one(plan_p, in, NULL);
    fftwnd(plan_p, g_sv_dim*g_sv_dim, in, g_sv_dim*g_sv_dim, 1, (fftw_complex*)(connq[0][0]+it*items), g_sv_dim*g_sv_dim, 1);
#endif
  }
  free(in);
  fftwnd_destroy_plan(plan_p);

  // add phase factor from the source location

  iix = 0;
  for(it=0;it<T;it++) {
    for(x1=0;x1<LX;x1++) {
      q[0] = (double)x1 / (double)LX;
    for(x2=0;x2<LY;x2++) {
      q[1] = (double)x2 / (double)LY;
    for(x3=0;x3<LZ;x3++) {
      q[2] = (double)x3 / (double)LZ;
      phase = 2. * M_PI * ( q[0]*sx1 + q[1]*sx2 + q[2]*sx3 );
      w1.re = cos(phase);
      w1.im = sin(phase);

      _sp_eq_sp(sp1, connq[iix] );
      _sp_eq_sp_ti_co( connq[iix], sp1, w1) ;

      iix++; 
    }}}  // of x3, x2, x1
  }  // of it

  // write to file
  sprintf(filename, "proton_2pt_v3_q.%.4d.t%.2dx%.2dy%.2dz%.2d", Nconf, sx0, sx1, sx2, sx3);
  sprintf(contype, "proton 2-pt. function, (t,q_1,q_2,q_3)-dependent, source_timeslice = %d", sx0);
  write_lime_contraction_v2(connq[0][0], filename, 64, g_sv_dim*g_sv_dim, contype, Nconf, 0);

  if(write_ascii) {
    strcat(filename, ".ascii");
    write_contraction(connq[0][0], (int*)NULL, filename, g_sv_dim*g_sv_dim, 1, 0);
  }


  /***********************************************
   * calculate connt
   ***********************************************/
  for(it=0;it<T;it++) {
    _sp_eq_sp(sp1, connq[it*VOL3]);
    _sp_eq_gamma_ti_sp(sp2, 0, sp1);
    _sp_pl_eq_sp(sp1, sp2);
    _co_eq_tr_sp(&w, sp1);
    connt[2*it  ] = w.re * 0.25;
    connt[2*it+1] = w.im * 0.25;
    _sp_eq_sp(sp1, connq[it*VOL3]);
    _sp_eq_gamma_ti_sp(sp2, 0, sp1);
    _sp_mi_eq_sp(sp1, sp2);
    _co_eq_tr_sp(&w, sp1);
    connt[2*(T+it)  ] = w.re * 0.25;
    connt[2*(T+it)+1] = w.im * 0.25;
  }

  sprintf(filename, "proton_2pt_v3.%.4d.t%.2dx%.2dy%.2dz%.2d.fw", Nconf, sx0, sx1, sx2, sx3);
  ofs = fopen(filename, "w");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
    exit(3);
  }
  fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d\n", g_kappa, T_global, LX, LY, LZ, g_mu, Nconf);

  ir = sx0;
  fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", 0, 0, 0, connt[2*ir], 0., Nconf);
  for(it=1;it<T/2;it++) {
    ir  = ( it + sx0 ) % T_global;
    ir2 = ( (T_global - it) + sx0 ) % T_global;
    fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", 0, 0, it, connt[2*ir], connt[2*ir2], Nconf);
  }
  ir = ( it + sx0 ) % T_global;
  fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", 0, 0, it, connt[2*ir], 0., Nconf);
  fclose(ofs);

  sprintf(filename, "proton_2pt_v3.%.4d.t%.2dx%.2dy%.2dz%.2d.bw", Nconf, sx0, sx1, sx2, sx3);
  ofs = fopen(filename, "w");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
    exit(3);
  }
  fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d\n", g_kappa, T_global, LX, LY, LZ, g_mu, Nconf);

  ir = sx0;
  fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", 0, 0, 0, connt[2*(T+ir)], 0., Nconf);
  for(it=1;it<T/2;it++) {
    ir  = ( it + sx0 ) % T_global;
    ir2 = ( (T_global - it) + sx0 ) % T_global;
    fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", 0, 0, it, connt[2*(T+ir)], connt[2*(T+ir2)], Nconf);
  }
  ir = ( it + sx0 ) % T_global;
  fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", 0, 0, it, connt[2*(T+ir)], 0., Nconf);
  fclose(ofs);

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();
  if(connt!= NULL) free(connt);
  if(connq!= NULL) free(connq);
  if(gauge_trafo != NULL) free(gauge_trafo);

  // create the fermion propagator points
  free_fp( &uprop );
  free_fp( &dprop );
  free_fp( &fp1 );
  free_fp( &fp2 );
  free_fp( &fp3 );
  free_sp( &sp1 );
  free_sp( &sp2 );


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
