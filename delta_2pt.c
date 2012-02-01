/****************************************************
 * delta_2pt.c
 *
 * Di 1. Nov 09:40:13 EET 2011
 *
 * PURPOSE:
 * - calculate the delta 2-point function from point sources
 * - write the delta as
 *     delta^{tb} = 2 epsilon_{abc} [ u^{T a} C ig- (-ig5)) d^b ] P5+ u^c + epsilon_{abc} [ u^{T a} C ig- u^b ] P5- d^c
 *     thes two term will be called (1) and (2)
 * - cf. proton_2pt.c, proton_2pt_v2.c
 * TODO:
 * - verify contractions
 * - version  for Wilson fermions and domain-wall fermions
 * DONE:
 *
#define _UNFINISHED
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
#include <getopt.h>
#ifdef OPENMP
#include <omp.h>
#endif

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
#include "make_H3orbits.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for connected contributions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose [no effect, lots of stdout output it]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, j, ll, sl;
  int filename_set = 0;
  int mms1=0;
  int l_LX_at, l_LXstart_at;
  int ix, idx, it, iix;
  int ir, ir1, ir2, ir3, iperm, is;
  int VOL3, ia0, ia1, ia2, ib;
  int n_c=1, n_s=4;
  int K=20, nK=20, itype;
  int use_mms=0;
  int dims[3];
  double *connt=NULL, *connq=NULL;
  int verbose = 0;
  int sx0, sx1, sx2, sx3;
  int write_ascii=0;
  int status;
  char filename[200], contype[200];
  double ratime, retime;
  double plaq_r=0., plaq_m=0., dsign, dtmp, dtmp2;
  double *gauge_field_timeslice=NULL;
  double **chi=NULL, **chi2=NULL, **psi=NULL, **psi2=NULL;
  double spinor1[24], spinor2[24], spinor3[24], *work=NULL;
  double cigmmig5_spinor[4][24], p5g0p5_spinor[4][24], cigm_spinor[4][24], cigm_spinor2[4][24], id_spinor[4][24], cigmmig5_spinor2[4][24];
  double spinor_field[12];
  double scs[18];
  complex w, w1, w2, w3;
  FILE *ofs;
  int num_threads = 1, Wilson_case = 0;

  int    Cg5igm_perm[] = { 2 , -1 ,  0 , -1 };
  double Cg5igm_sign[] = {-2.,  0., +2.,  0.};
  int ifactor1 =   1;

  int    Cigm_perm[]   = { 2 , -1 ,  0 , -1 };
  double Cigm_sign[]   = {-2.,  0., -2.,  0.};
  int ifactor2 =  0;

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

#ifdef _UNFINISHED
  exit(255);
#endif

  while ((c = getopt(argc, argv, "Wah?vgf:p:t:")) != -1) {
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
      fprintf(stdout, "# [delta_2pt] number of colors set to %d\n", n_c);
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [delta_2pt] will write in ascii format\n");
      break;
    case 't':
      num_threads = atoi(optarg);
      fprintf(stdout, "# [delta_2pt] number of threads set to %d\n", num_threads);
      break;
    case 'W':
      Wilson_case = 1;
      fprintf(stdout, "# [] will calculate the Wilson case\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // get the time
  g_the_time = time(NULL);

#ifdef OPENMP
  omp_set_num_threads(num_threads);
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
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  // initialize MPI parameters
  mpi_init(argc, argv);

#ifdef OPENMP
  status = fftw_threads_init();
  if(status != 0) {
    fprintf(stderr, "\n[] Error from fftw_init_threads; status was %d\n", status);
    exit(120);
  }
#endif


  dims[0]=LX; dims[1]=LY; dims[2]=LZ;
#ifndef MPI
  plan_p = fftwnd_create_plan(3, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
#endif
  FFTW_LOC_VOLUME = LX*LY*LZ;
  VOL3 = LX*LY*LZ;
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
      case 1:
        sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
        if(g_cart_id==0) fprintf(stdout, "\n# [] reading gauge field from file %s\n", filename);
        status = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
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
      for(i=0; i<N_ape; i++) {
#ifdef OPENMP
        APE_Smearing_Step_threads(g_gauge_field, alpha_ape);
#else
        APE_Smearing_Step(g_gauge_field, alpha_ape);
#endif
      }
    }
  } else {
    g_gauge_field = NULL;
  }  // of if N_Jacobi>0

  // determine the source location
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
//  g_source_time_slice = sx0;
  fprintf(stdout, "# [delta_2pt] source location %d = (%d,%d,%d,%d)\n", g_source_location, sx0, sx1, sx2, sx3);


  // initialize permutation tables
  init_perm_tabs();


  /* allocate memory for the spinor fields */
  no_fields = 12;
  if(N_Jacobi>0) no_fields++;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUMEPLUSRAND);
  work = g_spinor_field[no_fields-1];

  /* allocate memory for the contractions */
  connt = (double*)malloc(2*T*sizeof(double));
  if(connt == NULL) {
    fprintf(stderr, "\n[delta_2pt] Error, could not alloc connt\n");
    exit(2);
  }
  for(ix=0; ix<2*T; ix++) connt[ix] = 0.;

  connq = (double*)malloc(2*VOLUME*sizeof(double));
  if(connq == NULL) {
    fprintf(stderr, "\n[delta_2pt] Error, could not alloc connq\n");
    exit(2);
  }
  for(ix=0; ix<2*VOLUME; ix++) connq[ix] = 0.;

  if(Wilson_case) {
    for(is=0;is<n_s*n_c;is++) {
      sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia0);
      status = read_lime_spinor(g_spinor_field[is], filename, 0);
      if(status != 0) {
        fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
        exit(6);
      }
    }
  }

  /******************************************************
   * contractions
   ******************************************************/
  for(iperm=0;iperm<6;iperm++) {
    ia0 = perm_tab_3[iperm][0];
    ia1 = perm_tab_3[iperm][1];
    ia2 = perm_tab_3[iperm][2];
    dsign = perm_tab_3_sign[iperm];

    fprintf(stdout, "# [delta_2pt] working on combination (%d, %d, %d) with sign %3.0f\n", dsign, ia0, ia1,ia2);
    if(!Wilson_case) {
      for(is=0;is<4;is++) {
        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia0);
        status = read_lime_spinor(g_spinor_field[is  ], filename, 0);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          exit(6);
        }
        spinor_field[is] = g_spinor_field[is];
        
        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia1);
        status = read_lime_spinor(g_spinor_field[4+is], filename, 1);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          exit(6);
        }
        spinor_field[is+4] = g_spinor_field[is+4];

        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia2);
        status = read_lime_spinor(g_spinor_field[8+is], filename, 0);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          exit(6);
        }
        spinor_field[is+8] = g_spinor_field[is+8];
      }
    } else {
      for(is=0;is<4;is++) {
        spinor_field[is  ] = g_spinor_field[3*is+ia0];
        spinor_field[is+4] = g_spinor_field[3*is+ia1];
        spinor_field[is+8] = g_spinor_field[3*is+ia2];
      }
    }

    iix = 0;
    for(it=0;it<T;it++) {
    for(ix=0;ix<VOL3;ix++) {
      /****************************************************************
       * contribution (1) - (1)
       ****************************************************************/
      // P5+ g0 P5+ spinor_field[0,1,2,3];
      for(is=0;is<4;is++) {
        // (1) P5+
        _fv_eq_gamma_ti_fv (spinor1, 5, spinor_field[is]+_GSI(iix));
        _fv_eq_fv_ti_im    (spinor2, spinor1, 1.);
        _fv_pl_eq_fv       (spinor2, spinor_field[is]+_GSI(iix) );
        _fv_eq_fv_ti_re    (spinor1, spinor2, _ONE_OVER_SQRT2);
        // (2) g0
        _fv_eq_gamma_ti_fv (spinor2, 0, spinor1);
        _fv_pl_eq_fv       (spinor2, spinor1 );
        _fv_ti_eq_re       (spinor2, 0.25);
        // (3) P5+
        _fv_eq_gamma_ti_fv (spinor1, 5, spinor2);
        _fv_eq_fv_ti_im    (spinor3, spinor1, 1.);
        _fv_pl_eq_fv       (spinor3, spinor2 );
        _fv_eq_fv_ti_re    (p5g0p5_spinor[is], spinor3, _ONE_OVER_SQRT2);
      }
    

      // C ig- (-ig5) spinor_field[4,5,6,7] = g0 g2 (ig1 + g2) (-ig5) spinor_field[4,5,6,7]
      for(is=0;is<4;is++) {
        // -ig5
        _fv_eq_gamma_ti_fv (spinor1, 5, spinor_field[4+is]+_GSI(iix));
        _fv_eq_fv_ti_im    (spinor2, spinor1, -1.);
        // i ( g1 - i g2 )
        _fv_eq_gamma_ti_fv (spinor1, 1, spinor2);
        _fv_eq_gamma_ti_fv (spinor3, 2, spinor2);
        _fv_eq_fv_ti_im    (spinor2, spinor3, -1.);
        _fv_pl_eq_fv       (spinor1, spinor2);
        _fv_eq_fv_ti_im    (spinor2, spinor1,  1.);
        // C = g0 g2
        _fv_eq_gamma_ti_fv (spinor1, 2, spinor2);
        _fv_eq_gamma_ti_fv (cigmmig5_spinor[is], 0, spinor1);
      }

      // id_spinor = Identity spinor_field[8,9,10,11]
      for(is=0;is<4;is++) {
        _fv_eq_fv(id_spinor[is], spinor_field[8+is]+_GSI(iix));
      }

      // first contribution
      for(ir1=0;ir1<4;ir1++) {
        for(ir2=0; ir2<4;ir2++) {

          ir3 = Cg5igm_perm[ir2];
          if( ir3 == -1 ) continue;

          _cm_eq_fv_trans_ti_fv(scs, cigmmig5_spinor[ir3], id_spinor[ir2]);
          

          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
            w2.re = p5g0p5_spinor[ir1][2*(3*ir1+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir1][2*(3*ir1+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);

            // extra minus sign because of pemutation a'b'c' ---> c'b'a', ifactor is 1 for Cg5igm
            dtmp  = -w.re * Cg5igm_sign[ir2] * dsign * perm_tab_3_sign[ib] * 4.;
            dtmp2 = -w.im * Cg5igm_sign[ir2] * dsign * perm_tab_3_sign[ib] * 4.;
            connt[2*it  ]  -= dtmp2;
            connt[2*it+1]  += dtmp;
            connq[2*iix  ] -= dtmp2;
            connq[2*iix+1] += dtmp;

          }  // of ib
        }  // of ir2
      }  // of ir1


      // second contribution

      for(ir1=0;ir1<4;ir1++) {
        for(ir2=0; ir2<4;ir2++) {

          ir3 = Cg5igm_perm[ir2];
          if( ir3 == -1 ) continue;

          _cm_eq_fv_trans_ti_fv(scs, cigmmig5_spinor[ir3], id_spinor[ir1]);
              
          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
            w2.re = p5g0p5_spinor[ir2][2*(3*ir1+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir2][2*(3*ir1+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);

            // ifactor is 1 for Cg5igm
            dtmp  = -w.re * Cg5igm_sign[ir2] * dsign * perm_tab_3_sign[ib] * 4.;
            dtmp2 = -w.im * Cg5igm_sign[ir2] * dsign * perm_tab_3_sign[ib] * 4.;
            connt[2*it  ]  -= dtmp2;
            connt[2*it+1]  += dtmp;
            connq[2*iix  ] -= dtmp2;
            connq[2*iix+1] += dtmp;

          }  // of ib
        }  // of ir2
      }  // of ir1



      /****************************************************************
       * contribution (2) - (2)
       ****************************************************************/
      // cigm_spinor = C igm spinor_field[0,1,2,3] = g0 g2 i(g1 - ig2) spinor_field[0,1,2,3]
      for(is=0;is<4;is++) {
        // igm
        _fv_eq_gamma_ti_fv(spinor1, 1, spinor_field[is]+_GSI(iix));
        _fv_eq_gamma_ti_fv(spinor3, 2, spinor_field[is]+_GSI(iix));
        _fv_eq_fv_ti_im(spinor2, spinor3, -1.);
        _fv_pl_eq_fv(spinor1, spinor2);
        _fv_eq_fv_ti_im(spinor2, spinor1, 1.);
        // C = g0 g2
        _fv_eq_gamma_ti_fv(spinor1, 2, spinor2);
        _fv_eq_gamma_ti_fv(cigm_spinor[is], 0, spinor1);
      }

      // cigm_spinor2 = C igm spinor_field[8,9,10,11] = g0 g2 i(g1 - ig2) spinor_field[8,9,10,11]
      for(is=0;is<4;is++) {
        // igm
        _fv_eq_gamma_ti_fv(spinor1, 1, spinor_field[8+is]+_GSI(iix));
        _fv_eq_gamma_ti_fv(spinor3, 2, spinor_field[8+is]+_GSI(iix));
        _fv_eq_fv_ti_im(spinor2, spinor3, -1.);
        _fv_pl_eq_fv(spinor1, spinor2);
        _fv_eq_fv_ti_im(spinor2, spinor1, 1.);
        // C = g0 g2
        _fv_eq_gamma_ti_fv(spinor1, 2, spinor2);
        _fv_eq_gamma_ti_fv(cigm_spinor2[is], 0, spinor1);
      }

      // p5g0p5 = P5- Gamma^0 P5- spinor_field[4,5,6,7];
      for(is=0;is<4;is++) {
        // (1) P5-
        _fv_eq_gamma_ti_fv(spinor1, 5, spinor_field[4+is]+_GSI(iix));
        _fv_eq_fv_ti_im(spinor2, spinor1, -1.);
        _fv_pl_eq_fv(spinor2, spinor_field[4+is]+_GSI(iix) );
        _fv_eq_fv_ti_re(spinor1, spinor2, _ONE_OVER_SQRT2);
        // (2) Gamma
        _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1 );
        _fv_ti_eq_re(spinor2, 0.25);
        // (3) P5-
        _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        _fv_eq_fv_ti_im(spinor3, spinor1, -1.);
        _fv_pl_eq_fv(spinor3, spinor2 );
        _fv_eq_fv_ti_re( p5g0p5_spinor[is], spinor3, _ONE_OVER_SQRT2);
      }
    
      // id_spinor = Identity spinor_field[8,9,10,11]
      for(is=0;is<4;is++) {
        _fv_eq_fv(id_spinor[is], spinor_field[8+is]+_GSI(iix));
      }


      // first contribution
      for(ir1=0;ir1<4;ir1++) {
        ir2 = Cigm_perm[ir1];
        if( ir2 == -1 ) continue;

        _cm_eq_fv_trans_ti_fv(scs, cigm_spinor[ir2], id_spinor[ir1]);

        for(ir3=0; ir3<4;ir3++) { 

          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
            w2.re = p5g0p5_spinor[ir3][2*(3*ir3+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir3][2*(3*ir3+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);

            dtmp  = w.re * Cigm_sign[ir1] * dsign * perm_tab_3_sign[ib]; 
            dtmp2 = w.im * Cigm_sign[ir1] * dsign * perm_tab_3_sign[ib];
            connt[2*it  ]  += dtmp;
            connt[2*it+1]  += dtmp2;
            connq[2*iix  ] += dtmp;
            connq[2*iix+1] += dtmp2;

          }  // of ib
        }  // of ir2
      }  // of ir1


      // second contribution
      for(ir1=0;ir1<4;ir1++) {
      for(ir2=0;ir2<4;ir2++) {

        for(ir3=0; ir3<4;ir3++) {

          for(ib=0;ib<6;ib++) {

            w1.re = cigm_spinor[ir1][2*(3*ir2+perm_tab_3[ib][1])  ];
            w1.im = cigm_spinor[ir1][2*(3*ir2+perm_tab_3[ib][1])+1];
  
            w2.re = cigm_spinor2[ir2][2*(3*ir1+perm_tab_3[ib][0])  ];
            w2.im = cigm_spinor2[ir2][2*(3*ir1+perm_tab_3[ib][0])+1];
  

            w3.re = p5g0p5_spinor[ir3][2*(3*ir3+perm_tab_3[ib][2])  ];
            w3.im = p5g0p5_spinor[ir3][2*(3*ir3+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);
            _co_eq_co_ti_co(&w1, &w, &w3);

            // extra minus sign from epsilon_{abc} ---> epsilon_{bac}
            dtmp  = w1.re * dsign * perm_tab_3_sign[ib];
            dtmp2 = w1.im * dsign * perm_tab_3_sign[ib];
            connt[2*it  ]  += dtmp;
            connt[2*it+1]  += dtmp2;
            connq[2*iix  ] += dtmp;
            connq[2*iix+1] += dtmp2;

          }  // of ib
        }  // of ir3
      }}  // of ir1, ir2

       
      /****************************************************************
       * contribution (1) - (2)
       ****************************************************************/
      // P5+ Gamma P5- spinor_field[4+is]
      for(is=0;is<4;is++) {
        // P5-
        _fv_eq_gamma_ti_fv(spinor1, 5, spinor_field[4+is]+_GSI(iix));
        _fv_eq_fv_ti_im(spinor2, spinor1, -1.);
        _fv_pl_eq_fv(spinor2, spinor_field[4+is]+_GSI(iix));
        _fv_eq_fv_ti_re(spinor1, spinor2, _ONE_OVER_SQRT2);
        // Gamma
        _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1);
        _fv_eq_fv_ti_re(spinor1, spinor2, 0.25);
        // P5+
        _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
        _fv_eq_fv_ti_im(spinor3, spinor2, 1.);
        _fv_pl_eq_fv(spinor3, spinor1);
        _fv_eq_fv_ti_re(p5g0p5_spinor[is], spinor3, _ONE_OVER_SQRT2);
      }

      // cigm_spinor = C igm spinor_fields[0,1,2,3]
      for(is=0;is<4;is++) {
        // igm
        _fv_eq_gamma_ti_fv(spinor1, 1, spinor_field[is]+_GSI(iix));
        _fv_eq_gamma_ti_fv(spinor2, 2, spinor_field[is]+_GSI(iix));
        _fv_eq_fv_ti_im(spinor3, spinor2, -1.);
        _fv_pl_eq_fv(spinor1, spinor3);
        _fv_eq_fv_ti_im(spinor2, spinor1, 1.);
        // C = g0 g2
        _fv_eq_gamma_ti_fv(spinor1, 2, spinor2);
        _fv_eq_gamma_ti_fv(cigm_spinor[is], 0, spinor1);
      }
 
      // id_spinor = Identity spinor_field[8,9,10,11]
      for(is=0;is<4;is++) {
        _fv_eq_fv(id_spinor[is], spinor_field[8+is]+_GSI(iix));
      }

      // first contribution
      for(ir1=0;ir1<4;ir1++) {
        ir2 = Cg5igm_perm[ir1];
        if( ir2 == -1 ) continue;

        for(ir3=0; ir3<4;ir3++) { 

          _cm_eq_fv_trans_ti_fv(scs, cigm_spinor[ir3], id_spinor[ir1]);

          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
            w2.re = p5g0p5_spinor[ir2][2*(3*ir3+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir2][2*(3*ir3+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);

            // ifactor is one for Cg5igm
            dtmp  = -w.re * Cg5igm_sign[ir1] * dsign * perm_tab_3_sign[ib] * 2.; 
            dtmp2 = -w.im * Cg5igm_sign[ir1] * dsign * perm_tab_3_sign[ib] * 2.;
            connt[2*it  ]  -= dtmp2;
            connt[2*it+1]  += dtmp;
            connq[2*iix  ] -= dtmp2;
            connq[2*iix+1] += dtmp;

          }  // of ib
        }  // of ir2
      }  // of ir1

      // second contribution
      for(ir1=0;ir1<4;ir1++) {
        ir2 = Cg5igm_perm[ir1];
        if( ir2 == -1 ) continue;

        for(ir3=0; ir3<4;ir3++) { 

          _cm_eq_fv_trans_ti_fv(scs, cigm_spinor[ir1], id_spinor[ir3]);

          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
            w2.re = p5g0p5_spinor[ir2][2*(3*ir3+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir2][2*(3*ir3+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);

            // ifactor is one for Cg5igm, additional minus sign
            dtmp  = -w.re * Cg5igm_sign[ir1] * dsign * perm_tab_3_sign[ib] * 2.; 
            dtmp2 = -w.im * Cg5igm_sign[ir1] * dsign * perm_tab_3_sign[ib] * 2.;
            connt[2*it  ]  -= dtmp2;
            connt[2*it+1]  += dtmp;
            connq[2*iix  ] -= dtmp2;
            connq[2*iix+1] += dtmp;

          }  // of ib
        }  // of ir2
      }  // of ir1

      /****************************************************************
       * contribution (2) - (1)
       ****************************************************************/
      // p5g0p5_spinor = P5- Gamma P5+ spinor_field[0,1,2,3]
      for(is=0;is<4;is++) {
        // P5+
        _fv_eq_gamma_ti_fv(spinor1, 5, spinor_field[is]+_GSI(iix));
        _fv_eq_fv_ti_im(spinor2, spinor1, 1.);
        _fv_pl_eq_fv(spinor2, spinor_field[is]+_GSI(iix));
        _fv_eq_fv_ti_re(spinor1, spinor2, _ONE_OVER_SQRT2);
        // Gamma
        _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1);
        _fv_eq_fv_ti_re(spinor1, spinor2, 0.25);
        // P5-
        _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
        _fv_eq_fv_ti_im(spinor3, spinor2, -1.);
        _fv_pl_eq_fv(spinor3, spinor1);
        _fv_eq_fv_ti_re(p5g0p5_spinor[is], spinor3, _ONE_OVER_SQRT2);
      }

      // cigmmig5_spinor = C igm (-ig5) spinor_fields[4,5,6,7]
      for(is=0;is<4;is++) {
        // -ig5
        _fv_eq_gamma_ti_fv(spinor1, 5, spinor_field[4+is]+_GSI(iix));
        _fv_eq_fv_ti_im(spinor3, spinor1, -1.);
        // igm
        _fv_eq_gamma_ti_fv(spinor1, 1, spinor3);
        _fv_eq_gamma_ti_fv(spinor2, 2, spinor3);
        _fv_eq_fv_ti_im(spinor3, spinor2, -1.);
        _fv_pl_eq_fv(spinor1, spinor3);
        _fv_eq_fv_ti_im(spinor2, spinor1, 1.);
        // C = g0 g2
        _fv_eq_gamma_ti_fv(spinor1, 2, spinor2);
        _fv_eq_gamma_ti_fv(cigmmig5_spinor[is], 0, spinor1);
      }

      // id_spinor = Identity spinor_field[8,9,10,11]
      for(is=0;is<4;is++) {
        _fv_eq_fv(id_spinor[is], spinor_field[8+is]+_GSI(iix));
      } 

      // first contribution
      for(ir1=0;ir1<4;ir1++) {
        ir2 = Cigm_perm[ir1];
        if( ir2 == -1 ) continue;

        for(ir3=0; ir3<4;ir3++) { 

          _cm_eq_fv_trans_ti_fv(scs, cigmmig5_spinor[ir3], id_spinor[ir1]);

          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
            w2.re = p5g0p5_spinor[ir2][2*(3*ir3+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir2][2*(3*ir3+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);

            // ifactor is zero for Cigm
            dtmp  = -w.re * Cigm_sign[ir1] * dsign * perm_tab_3_sign[ib] * 2.; 
            dtmp2 = -w.im * Cigm_sign[ir1] * dsign * perm_tab_3_sign[ib] * 2.;
            connt[2*it  ]  += dtmp;
            connt[2*it+1]  += dtmp2;
            connq[2*iix  ] += dtmp;
            connq[2*iix+1] += dtmp2;

          }  // of ib
        }  // of ir2
      }  // of ir1

      // second contribution
      for(ir1=0;ir1<4;ir1++) {
        ir2 = Cigm_perm[ir1];
        if( ir2 == -1 ) continue;

        for(ir3=0; ir3<4;ir3++) { 

          _cm_eq_fv_trans_ti_fv(scs, cigmmig5_spinor[ir3], id_spinor[ir2]);

          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
            w2.re = p5g0p5_spinor[ir1][2*(3*ir3+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir1][2*(3*ir3+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);

            // ifactor is one for Cigm, additional minus sign
            dtmp  = -w.re * Cigm_sign[ir1] * dsign * perm_tab_3_sign[ib]; 
            dtmp2 = -w.im * Cigm_sign[ir1] * dsign * perm_tab_3_sign[ib];
            connt[2*it  ]  += dtmp;
            connt[2*it+1]  += dtmp2;
            connq[2*iix  ] += dtmp;
            connq[2*iix+1] += dtmp2;

          }  // of ib
        }  // of ir2
      }  // of ir1

      iix ++;
    }} // of ix, it

  }  // of loop on iperm


  /***********************************************
   * free gauge fields and spinor fields
   ***********************************************/
  if(g_gauge_field != NULL) free(g_gauge_field); g_gauge_field=(double*)NULL;
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); g_spinor_field=(double**)NULL;

  /***********************************************
   * finish calculation of connt
   ***********************************************/
  if(g_propagator_bc_type == 0) {
    // multiply with phase factor
    fprintf(stdout, "\n# [] multiply with boundary phase factor\n");
    for(it=0;it<T_global;it++) {
      ir = (it - sx0 + T_global) % T_global;
      w1.re = cos( 3. * M_PI*(double)ir / (double)T_global );
      w1.im = sin( 3. * M_PI*(double)ir / (double)T_global );
      w2.re = connt[2*it  ];
      w2.im = connt[2*it+1];
      _co_eq_co_ti_co( (complex*)(connt+2*it), &w1, &w2) ;
    }
  } else if(g_propagator_bc_type == 1) { 
    // multiply with step function
    fprintf(stdout, "\n# [] multiply with boundary step function\n");
    for(it=0;it<sx0;it++) {
      connt[2*it  ] *= -1.;
      connt[2*it+1] *= -1.;
    }
  }

  // write to file
  sprintf(filename, "delta_2pt.%.4d", Nconf);
  ofs = fopen(filename, "w");
  if(ofs == NULL) {
    fprintf(stderr, "[delta_2pt] Error, could not open file %s for writing\n", filename);
    exit(3);
  }
  fprintf(ofs, "#%12.8f%3d%3d%3d%3d%8.4f%6d\n", g_kappa, T_global, LX, LY, LZ, g_mu, Nconf);
  for(it=0;it<T;it++) {
    ir = ( it+sx0 ) % T_global;
    fprintf(ofs, "%3d%3d%3d%16.7e%16.7e%6d\n", 0, 0, it, connt[2*ir], connt[2*ir+1], Nconf);
  }
  fclose(ofs);

  /***********************************************
   * finish calculation of connq
   ***********************************************/
  if(g_propagator_bc_type == 0) {
    // multiply with phase factor
    fprintf(stdout, "\n# [] multiply with boundary phase factor\n");
    iix = 0;
    for(it=0;it<T_global;it++) {
      ir = (it - sx0 + T_global) % T_global;
      w1.re = cos( 3. * M_PI*(double)ir / (double)T_global );
      w1.im = sin( 3. * M_PI*(double)ir / (double)T_global );
      for(ix=0;ix<VOL3;ix++) {
        w2.re = connq[iix  ];
        w2.im = connq[iix+1];
        _co_eq_co_ti_co( (complex*)(connq+iix), &w1, &w2) ;
        iix += 2;
      }
    }
  } else if(g_propagator_bc_type == 1) {
    // multiply with boundary step function
    fprintf(stdout, "\n# [] multiply with boundary step function\n");
    iix = 0;
    for(it=0;it<sx0;it++) {
      for(ix=0;ix<VOL3;ix++) {
        connq[iix  ] *= -1.;
        connq[iix+1] *= -1.;
        iix += 2;
      }
    }
  }

  // Fourier transform
  in  = (fftw_complex*)malloc(VOL3*sizeof(fftw_complex));
  for(it=0;it<T;it++) {
    memcpy(in, connq+2*it*VOL3, 2*VOL3*sizeof(double));
#ifdef OPENMP
    fftwnd_threads_one(num_threads, plan_p, in, NULL);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy(connq+2*it*VOL3, in, 2*VOL3*sizeof(double));
  }
  // write to file
  sprintf(filename, "delta_2pt_q.%.4d", Nconf);
  sprintf(contype, "delta 2-pt. function, (t,q_1,q_2,q_3)-dependent");
  write_lime_contraction(connq, filename, 64, 1, contype, Nconf, 0);
  if(write_ascii) {
    sprintf(filename, "delta_2pt_q.%.4d.ascii", Nconf);
    write_contraction(connq, (int*)NULL, filename, 1, 2, 0);
  }

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();
  if(connt!= NULL) free(connt);
  if(connq!= NULL) free(connq);

  free(in);
  fftwnd_destroy_plan(plan_p);

  g_the_time = time(NULL);
  fprintf(stdout, "# [delta_2pt] %s# [delta_2pt] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [delta_2pt] %s# [delta_2pt] end fo run\n", ctime(&g_the_time));
  fflush(stderr);

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
