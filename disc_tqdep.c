/****************************************************
 * disc_tqdep.c
 *
 * Thu Nov 10 10:33:03 EET 2011
 *
 * PURPOSE
 * - calculate loops psibar Gamma psi from timeslice ( / volume ?)
 *   sources; save the (t,q1,q2,q3)- dependend fields
 * TODO:
 * - think of solution for tm-case and V^C_0 (t,t+0^) of up and down
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
#include "Q_phi.h"
#include "fuzz.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "gauge_io.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for disconnected contributions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose [no effect, lots of stdout output anyway]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
 
  const int K = 20; 
  int c, i, mu;
  int filename_set = 0;
  int status;
  int l_LX_at, l_LXstart_at;
  int it, x0, x1, ix, idx, x2, x3, iy;
  int VOL3;
  int dims[3];
  double *disc = (double*)NULL;
  int verbose = 0;
  int write_ascii=0;
  int fermion_type = 1;
  char filename[100], contype[500];
  double ratime, retime;
  double plaq_r=0., plaq_m=0.;
  double spinor1[24], spinor2[24];
  double _2kappamu;
  double *gauge_field_smeared=NULL;
  double v4norm = 0., vvnorm = 0.;
  complex w, w1, w2;
  double q[3], phase, U_[18];
  FILE *ofs;
  int isample;
  double *work=NULL;
/*  double sign_adj5[] = {-1., -1., -1., -1., +1., +1., +1., +1., +1., +1., -1., -1., -1., 1., -1., -1.}; */
  double hopexp_coeff[8], addreal, addimag, fnorm;
  int num_threads=1, no_fields, num_timeslices=1;
  size_t bytes, items;

/***********************************************************************************************/            
/*                    g5  gi           g0g5 g0gi         id  gig5        g0  g[igj]            */
 // int gindex[]    = { 5 , 1 , 2 , 3 ,  6 ,  10 ,11 ,12 , 4 , 7 , 8 , 9 , 0 , 15 , 14 ,13 };
 // int isimag[]    = { 0 , 0 , 0 , 0 ,  1 ,   1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 ,  1 ,  1 , 1 };
 // double gsign[]  = {-1., 1., 1., 1., -1.,   1., 1., 1., 1., 1., 1., 1., 1.,  1., -1., 1.};
  int  gindex[]  = { 0, 1, 2, 3, 4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  int isimag[]   = { 0, 0, 0, 0, 0,  0,  1, 1, 1, 1,  1,  1,  1,  1,  1,  1 };
  double gsign[] = { 1, 1, 1, 1, 1,  1,  1, 1, 1, 1,  1,  1,  1,  1, -1,  1 };
/***********************************************************************************************/            

  fftwnd_plan plan_p;
  fftw_complex *in = NULL;

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ah?vgf:t:F:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 't':
      num_threads = atoi(optarg);
      fprintf(stdout, "# [] will set number of threads to %d\n", num_threads);
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [] will write output in ascii format\n");
      break;
    case 'F':
      if(strcmp(optarg, "Wilson")==0) {
        fermion_type = _WILSON_FERMION;
      } else if(strcmp(optarg, "tm")==0) {
        fermion_type = _TM_FERMION;
      } else {
        fprintf(stderr, "[] Error, unrecognized fermion type\n");
        exit(145);
      }
      fprintf(stdout, "# [] use fermion type %s - id %d\n", optarg, fermion_type);
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

  VOL3    = LX*LY*LZ;

  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [] parameters for process %d: "\
		  "l_LX_at      = %3d; l_LXstart_at = %3d; FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, l_LX_at, l_LXstart_at, FFTW_LOC_VOLUME);
  fflush(stdout);

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
    if(g_cart_id==0) fprintf(stdout, "# [invert_quda] Setting up unit gauge field\n");
    unit_gauge_field(g_gauge_field, VOLUME);
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
    
  // measure the plaquette
  plaquette(&plaq_m);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq_m);
  if(g_cart_id==0) fprintf(stdout, "# read plaquette value    : %25.16e\n", plaq_r);

  if(N_ape > 0) {
    if(g_cart_id==0) fprintf(stdout, "# [] APE smearing of gauge field with parameters N_ape = %d\n# alpha_ape = %f\n", N_ape, alpha_ape);
    alloc_gauge_field(&gauge_field_smeared, VOLUMEPLUSRAND);
#ifdef OPENMP
    APE_Smearing_Step_threads(gauge_field_smeared, N_ape, alpha_ape);
#else
    for(i=0; i<N_ape; i++) {
      APE_Smearing_Step(gauge_field_smeared, alpha_ape);
    }
#endif
  }

  /* allocate memory for the spinor fields */
  no_fields = 2;
  if(N_Jacobi > 0) { no_fields++; }
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);
  if(N_Jacobi>0) {
    work = g_spinor_field[no_fields-1];
  }

  // allocate memory for the contractions
  if(g_coherent_source == 1) num_timeslices = T / g_coherent_source_delta;
  if(g_source_type == 1 ) num_timeslices = T;
  fprintf(stdout, "# [] number of timeslices = %d\n", num_timeslices);


  items =   2*K*num_timeslices*VOL3;  // 2[complex] x (16+1)[Gamma_mu +  cvc] x VOLUME
  bytes = sizeof(double);
  disc = (double*)calloc(items, bytes);
  if( disc==NULL ) {
    fprintf(stderr, "could not allocate memory for disc\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 3);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0;ix<items;ix++) disc[ix] = 0.;


  /************************************
   * initialize FFTW
   ************************************/
  dims[0] = LX;
  dims[1] = LY;
  dims[2] = LZ;
  in = (fftw_complex*)malloc(K * VOL3 * sizeof(fftw_complex));
  if(in == NULL) {
    fprintf(stderr, "[] Error, could not alloc in for FFTW\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 5);
    MPI_Finalize();
#endif
    exit(5);
  }
  plan_p = fftwnd_create_plan_specific(3, dims, FFTW_FORWARD, FFTW_MEASURE, in, K, (fftw_complex*)( disc ), K);

  // set the global source timeslice if it is a coherent source
  if(g_coherent_source == 1) {
    g_source_timeslice = g_coherent_source_base;
    fprintf(stdout, "# [] Warning: reset source timeslice to %d\n", g_source_timeslice);
  }

  /************************************
   * loop on samples
   ************************************/
  for(isample=0;isample<g_nsample;isample++) {

    /* read the new propagator */
    switch(g_source_type) {
      case 2:
        // timeslice source
        sprintf(filename, "%s.%.4d.%.2d.%.5d.inverted", filename_prefix, Nconf, g_source_timeslice, isample);
        break;
      case 1:
        // volume source
        sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, isample);
        break;
      default:
        fprintf(stderr, "[] source format not yet implemented\n");
        exit(7);
       break;
    }
    if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) {
      fprintf(stderr, "[%2d] Error, could not read from file %s\n", g_cart_id, filename);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(4);
    }
    xchange_field(g_spinor_field[1]);

    /* calculate the source: apply Q_phi_tbc */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(fermion_type && g_propagator_bc_type == 1) {
      Q_Wilson_phi(g_spinor_field[0], g_spinor_field[1]);
    } else {
      Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);
    }
    xchange_field(g_spinor_field[0]); 
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to apply Dirac operator: %e seconds\n", retime-ratime);

    // sink smearing
    if(N_Jacobi > 0) {
#ifdef OPENMP
      Jacobi_Smearing_Step_one_threads(gauge_field_smeared, g_spinor_field[1], work, N_Jacobi, kappa_Jacobi);
#else
      for(c=0; c<N_Jacobi; c++) {
        Jacobi_Smearing_Step_one(gauge_field_smeared, g_spinor_field[1], work, kappa_Jacobi);
      }
#endif
    }
  
    // printf_spinor_field(g_spinor_field[0], stdout);

    // add new contractions to disc
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(mu=0; mu<16; mu++) {  // loop on index of gamma matrix
#ifdef OPENMP
#pragma omp parallel for private(it, x0, x1, ix, spinor1, spinor2, U_, w) shared(mu)
#endif
      for(it=0; it<num_timeslices; it++) {       // loop on timeslices
        if(g_coherent_source==1) { x0 = (g_coherent_source_base + it*g_coherent_source_delta ) % T_global; }
        else                     { x0 = ( g_source_timeslice + it ) % T_global; }
        // fprintf(stdout, "# [] using x0 = %d\n", x0);
        for(x1=0; x1<VOL3; x1++) {  // loop on sites in timeslice
          ix = x0*VOL3 + x1;
          iy = it*VOL3 + x1;
          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[1][_GSI(ix)]);
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1);
	  disc[2 * ( (it*K + mu)*VOL3 + x1 )  ] += w.re;
	  disc[2 * ( (it*K + mu)*VOL3 + x1 )+1] += w.im;
        }  // of x1 = 0, ..., VOL3
      }  // of x0
    }    // of mu 

    // conserved vector current, component 0
    if(fermion_type == _WILSON_FERMION) {
      mu = 0;
#ifdef OPENMP
#pragma omp parallel for private(it, ix, iy, x0, x1, spinor1, spinor2, U_, w) shared(mu)
#endif
      for(it=0; it<num_timeslices; it++) {       // loop on time
        if(g_coherent_source==1) { x0 = (g_coherent_source_base + it*g_coherent_source_delta ) % T_global; }
        else                     { x0 = ( g_source_timeslice + it ) % T_global; }
        for(x1=0; x1<VOL3; x1++) {  // loop on sites in timeslice
          ix = x0*VOL3 + x1;
          iy = it*VOL3 + x1;

          if(g_propagator_bc_type==0) {
             _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &co_phase_up[mu]);
          } else if(g_propagator_bc_type == 1 && x0==T_global-1) {
            _cm_eq_cm_ti_re(U_, g_gauge_field+_GGI(ix,mu), -1.);
          } else {
            _cm_eq_cm(U_, g_gauge_field+_GGI(ix,mu));
          }
          _fv_eq_cm_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(g_iup[ix][mu]));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_mi_eq_fv(spinor2, spinor1);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor2);

          // disc[2 * ( (it*K + 16+mu)*VOL3 + x1 )  ] += -0.5 * w.re;
          disc[2 * ( (it*K + 16+mu)*VOL3 + x1 )+1] += -w.im;
        }  // of x1 = 0, ..., VOL3
      }

    // conserved vector current, components 1,2,3
    for(mu=1; mu<4; mu++) {  // loop on index of gamma matrix
#ifdef OPENMP
#pragma omp parallel for private(it, ix, iy, x0, x1, spinor1, spinor2, U_, w) shared(mu)
#endif
        for(it=0; it<num_timeslices; it++) {       // loop on time
          if(g_coherent_source==1) { x0 = (g_coherent_source_base + it*g_coherent_source_delta ) % T_global; }
          else                     { x0 = ( g_source_timeslice + it ) % T_global; }
          for(x1=0; x1<VOL3; x1++) {  // loop on sites in timeslice
            ix = x0*VOL3 + x1;
            iy = it*VOL3 + x1;

            if(g_propagator_bc_type==0) {
             _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &co_phase_up[mu]);
            } else {
             _cm_eq_cm(U_, g_gauge_field+_GGI(ix,mu));
            }
            _fv_eq_cm_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(g_iup[ix][mu]));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
            _fv_mi_eq_fv(spinor2, spinor1);
            _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor2);

            // disc[2 * ( (it*K + 16+mu)*VOL3 + x1 )  ] += -w.re;
            disc[2 * ( (it*K + 16+mu)*VOL3 + x1 ) + 1] += -w.im;
        
            //_fv_eq_cm_dag_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(ix));
            //_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
            //_fv_pl_eq_fv(spinor2, spinor1);
            //_co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_iup[ix][mu]), spinor2);
            //
            //disc[2 * ( (it*K + 16+mu)*VOL3 + x1 )  ] += -0.5 * w.re;
            //disc[2 * ( (it*K + 16+mu)*VOL3 + x1 )+1] += -0.5 * w.im;
          }  // of x1 = 0, ..., VOL3
        }
      }
    } // of if fermion_type == 
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# contractions in %e seconds\n", retime-ratime);

  }  // of isample

  /**************************************************************
   * normalization
   **************************************************************/
  fnorm = 1. / ( (double)g_nsample * g_prop_normsqr );
  fprintf(stdout, "# [] using normalization with fnorm = %e\n", fnorm);
  items = num_timeslices * 2*K*VOL3;
  for(ix=0;ix<items;ix++) disc[ix] *= fnorm;


  // write t-x-dep data to ASCII file
  if(write_ascii) {
    for(it=0;it<num_timeslices;it++) {
      if(g_coherent_source==1) { x0 = (g_coherent_source_base + it*g_coherent_source_delta ) % T_global; }
      else                     { x0 = ( g_source_timeslice + it ) % T_global; }
      switch(g_source_type) {
        case 1:
          // volume
          sprintf(filename, "%s_tx.%.4d.%.2d.%.5d.ascii", filename_prefix2, Nconf, x0, g_nsample);
          break;
        case 2:
          // timeslice
          sprintf(filename, "%s_tx.%.4d.%.2d.%.5d.ascii", filename_prefix2, Nconf, x0, g_nsample);
          break;
      }
      write_contraction2(disc+it*2*K*VOL3, filename, K, VOL3, 2, 0);


      // test: write out by "hand"
      //fprintf(stdout, "# [] contractions for sample %d, timeslice no. %d:\n", isample, it);
      //for(mu=0;mu<K;mu++) {
      //  for(ix=0; ix<VOL3;ix++) {
      //    fprintf(stdout, "\t%3d%6d%25.16e%25.16e\n", mu, ix, disc[2*((it*K+mu)*VOL3+ix)], disc[2*((it*K+mu)*VOL3+ix)+1]);
      //  }
      //}
      //fprintf(stdout, "# []\n");
      //fflush(stdout);
    }
  }

  /***********************************************
   * Fourier transform
   ***********************************************/
  items = 2*K*VOL3;
  bytes = sizeof(double);
  for(it=0;it<num_timeslices;it++) {
    memcpy(in, disc + it*items, items*bytes);
#ifdef OPENMP
    // fftwnd_threads_one(num_threads, plan_p, in, NULL);
    fftwnd_threads(num_threads, plan_p, K, in, 1, VOL3, (fftw_complex*)(disc + it*items), 1, VOL3);
#else
    // fftwnd_one(plan_p, in, NULL);
    fftwnd(plan_p, K, in, 1, VOL3, (fftw_complex*)(disc + it*items), 1, VOL3);
#endif
  }
  free(in);

  // add phase factors for conserved vector current in spatial directions
  for(it=0;it<num_timeslices;it++) {
    for(mu=1;mu<4;mu++) {
      ix = 0;
      for(x1=0;x1<LX;x1++) {
        q[0] = (double)x1 / (double)LX;
      for(x2=0;x2<LY;x2++) {
        q[1] = (double)x2 / (double)LY;
      for(x3=0;x3<LZ;x3++) {
        q[2] = (double)x3 / (double)LZ;
        phase = -M_PI * q[mu-1];
        w.re  = cos(phase);
        w.im  = sin(phase);
        w1.re = disc[2*( (it*K + 16+mu)*VOL3 + ix)  ];
        w1.im = disc[2*( (it*K + 16+mu)*VOL3 + ix)+1];
        _co_eq_co_ti_co(&w2, &w1, &w);
        disc[2*( (it*K + 16+mu)*VOL3 + ix)  ] = w2.re;
        disc[2*( (it*K + 16+mu)*VOL3 + ix)+1] = w2.im;
        ix++;
      }}}
    }
  }

  /***********************************************
   * write current disc to file
   ***********************************************/

  for(it=0;it<num_timeslices; it++) {
    if(g_coherent_source==1) { x0 = (g_coherent_source_base + it*g_coherent_source_delta ) % T_global; }
    else                     { x0 = ( g_source_timeslice + it ) % T_global; }
    switch(g_source_type) {
      case 1:
        // volume
        sprintf(filename, "%s_tq.%.4d.%.2d.%.5d", filename_prefix2, Nconf, x0, g_nsample);
        break;
      case 2:
        // timeslice
        sprintf(filename, "%s_tq.%.4d.%.2d.%.5d", filename_prefix2, Nconf, x0, g_nsample);
        break;
    }
    sprintf(contype, "quark-disconnected contractions; 16 Gamma structures, conserved vector current (20 types in total); timeslice no. %d;", x0);
    write_lime_contraction_3d(disc+it*2*K*VOL3, filename, 64, K, contype, Nconf, g_nsample);
    
    if(write_ascii) {
      strcat(filename,".ascii");
      write_contraction2(disc+it*2*K*VOL3, filename, K, VOL3, 2, 0);
    }
  }


  /********************************************************
   * free the allocated memory, finalize
   ********************************************************/
  if(g_gauge_field != NULL) { free(g_gauge_field); g_gauge_field=(double*)NULL; }
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); g_spinor_field=(double**)NULL;
  free_geometry();
  free(disc);
  if(gauge_field_smeared!=NULL) free(gauge_field_smeared);

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [] %s# [] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [] %s# [] end of run\n", ctime(&g_the_time));
  }

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
