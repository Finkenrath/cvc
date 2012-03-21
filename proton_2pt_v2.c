/****************************************************
 * proton_2pt_v2.c
 *
 * Di 1. Nov 09:22:15 EET 2011
 *
 * PURPOSE:
 * - calculate the proton 2-point function from point sources
 * - like proton_2pt, but 
 * TODO:
 * - contractions for sequential source method
 * - ask Giannis about smearing ---> Gaussian smearing ? == ? Jacobi smearing?
 * - 3-dimensional Fourier transform in the spatial coordinates
 * - NOTE ON FFTW EXPONENT SIGN: FFTW_FORWARD  = -1
 * -                             FFTW_BACKWARD = +1
 * DONE:
 * - checked against Jen's momentum summation method in free case
 *   in Wilson and tm case 
 * - checked gauge invariance in Wilson case
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
#include "make_H3orbits.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for connected contributions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose [no effect, lots of stdout output it]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -l Nlong for fuzzing [default -1, no fuzzing]\n");
  fprintf(stdout, "         -a no of steps for APE smearing [default -1, no smearing]\n");
  fprintf(stdout, "         -k alpha for APE smearing [default 0.]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, j, ll, sl;
  int filename_set = 0;
  int mms1=0, status;
  int l_LX_at, l_LXstart_at;
  int ix, idx, it, iix, x1,x2,x3;
  int ir, ir1, ir2, ir3, iperm, is;
  int VOL3, ia0, ia1, ia2, ib;
  int n_c=1, n_s=4;
  int K=20, nK=20, itype;
  int do_gt=0;
  int dims[3];
  double *connt=NULL, *connq=NULL;
  int verbose = 0;
  int sx0, sx1, sx2, sx3;
  int write_ascii=0;
  int Wilson_case = 0;
  int num_threads=1;
  int pos;
  char filename[200], contype[200];
  double ratime, retime;
  double plaq_m, plaq_r, dsign, dtmp, dtmp2;
  double *gauge_field_timeslice=NULL, *gauge_field_f=NULL;
  double **chi=NULL, **chi2=NULL, **psi=NULL, **psi2=NULL, *work=NULL;
  double spinor1[24], spinor2[24], spinor3[24];
  double scs[18];
  double p5g0p5_spinor[4][24], cg5_spinor[4][24], id_spinor[4][24], *spinor_field[12];
  double q[3], phase, *gauge_trafo=NULL;
  complex w, w1, w2;
  FILE *ofs;

  int Cg5_perm[] = {1,0,3,2};
  double Cg5_sign[] = {-1., 1., -1., 1.};

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
      fprintf(stdout, "# [] number of colors set to %d\n", n_c);
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [] will write in ascii format\n");
      break;
    case 'W':
      Wilson_case = 1;
      fprintf(stdout, "# [] will calculate for the Wilson case\n");
      break;
    case 't':
      num_threads = atoi(optarg);
      fprintf(stdout, "# [] number of threads set to %d\n", num_threads);
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
  omp_set_num_threads(num_threads);
#else
  fprintf(stdout, "[proton_2pt_v2] Warning, resetting global thread number to 1\n");
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

  dims[0]=LX; dims[1]=LY; dims[2]=LZ;
  plan_p = fftwnd_create_plan(3, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
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


  /* allocate memory for the spinor fields */
  g_spinor_field = NULL;
  no_fields = 12;
  if(N_Jacobi>0) no_fields++;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUMEPLUSRAND);
  work = g_spinor_field[no_fields-1];

  /* allocate memory for the contractions */
  connt = (double*)malloc(2*T*sizeof(double));
  if(connt == NULL) {
    fprintf(stderr, "\n[] Error, could not alloc connt\n");
    exit(2);
  }
  for(ix=0; ix<2*T; ix++) connt[ix] = 0.;

  connq = (double*)malloc(2*VOLUME*sizeof(double));
  if(connq == NULL) {
    fprintf(stderr, "\n[] Error, could not alloc connq\n");
    exit(2);
  }
  for(ix=0; ix<2*VOLUME; ix++) connq[ix] = 0.;

  if(Wilson_case) {
    for(is=0;is<12;is++) {
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
        apply_gt_prop(gauge_trafo, g_spinor_field[is], is/3, is%3, 4, filename_prefix, g_source_location);
      } // of if do_gt == 0
    }
  }  // of if Wilson_case

  /******************************************************
   * contractions
   ******************************************************/
  for(iperm=0;iperm<6;iperm++) {
    ia0 = perm_tab_3[iperm][0];
    ia1 = perm_tab_3[iperm][1];
    ia2 = perm_tab_3[iperm][2];
    dsign = perm_tab_3_sign[iperm];

    fprintf(stdout, "# [] working on combination (%d, %d, %d) with sign %3.0f\n", dsign, ia0, ia1,ia2);
  
    if(!Wilson_case) {
      for(is=0;is<4;is++) {
        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia0);
        status = read_lime_spinor(g_spinor_field[is  ], filename, 0);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          exit(102);
        }
        spinor_field[is] = g_spinor_field[is];

        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia1);
        pos = Wilson_case ? 0 : 1;
        status = read_lime_spinor(g_spinor_field[4+is], filename, pos);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          exit(102);
        }
        spinor_field[is+4] = g_spinor_field[is+4];

        sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia2);
        status = read_lime_spinor(g_spinor_field[8+is], filename, 0);
        if(status != 0) {
          fprintf(stderr, "[] Error, could not read propagator from file %s\n", filename);
          exit(102);
        }
        spinor_field[is+8] = g_spinor_field[is+8];

        if(N_Jacobi > 0) {
#ifdef OPENMP
          Jacobi_Smearing_Step_one_threads(g_gauge_field, g_spinor_field[is  ], work, N_Jacobi, kappa_Jacobi);
          Jacobi_Smearing_Step_one_threads(g_gauge_field, g_spinor_field[is+4], work, N_Jacobi, kappa_Jacobi);
          Jacobi_Smearing_Step_one_threads(g_gauge_field, g_spinor_field[is+8], work, N_Jacobi, kappa_Jacobi);
#else
          for(c=0; c<N_Jacobi; c++) {
            Jacobi_Smearing_Step_one(g_gauge_field, g_spinor_field[is  ], work, kappa_Jacobi);
            Jacobi_Smearing_Step_one(g_gauge_field, g_spinor_field[is+4], work, kappa_Jacobi);
            Jacobi_Smearing_Step_one(g_gauge_field, g_spinor_field[is+8], work, kappa_Jacobi);
          }
#endif
        }
      }
    } else {
      for(is=0;is<4;is++) {
        spinor_field[is  ] = g_spinor_field[3*is+ia0];
        spinor_field[is+4] = g_spinor_field[3*is+ia1];
        spinor_field[is+8] = g_spinor_field[3*is+ia2];
      }
    }

    iix=0;
    for(it=0; it<T;   it++) {
    for(ix=0; ix<VOL3;ix++) {
      if(!Wilson_case) {
        // P_5 Gamma^0 P_5 spinor_field[0,1,2,3];
        for(is=0;is<4;is++) {
          // (1) P_5
          _fv_eq_gamma_ti_fv(spinor1, 5, spinor_field[is]+_GSI(iix));
          _fv_eq_fv_ti_im(spinor2, spinor1, 1.);
          _fv_pl_eq_fv(spinor2, spinor_field[is]+_GSI(iix) );
          _fv_eq_fv_ti_re(spinor1, spinor2, _ONE_OVER_SQRT2);
          // (2) Gamma^0
          _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
          _fv_pl_eq_fv(spinor2, spinor1 );
          _fv_ti_eq_re(spinor2, 0.25);
          // (3) P_5
          _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
          _fv_eq_fv_ti_im(spinor3, spinor1, 1.);
          _fv_pl_eq_fv(spinor3, spinor2 );
          _fv_eq_fv_ti_re( p5g0p5_spinor[is], spinor3, _ONE_OVER_SQRT2);
        }
      } else {
        // Gamma^0 spinor_field[0,1,2,3];
        for(is=0;is<4;is++) {
          // (1) Gamma^0
          _fv_eq_gamma_ti_fv(spinor2, 0, spinor_field[is]+_GSI(iix));
          _fv_pl_eq_fv(spinor2, spinor_field[is]+_GSI(iix) );
          _fv_eq_fv_ti_re(p5g0p5_spinor[is], spinor2, 0.25);
        }
      }

      // C g5 spinor_field[4,5,6,7] = g0 g2 g5 spinor_field[4,5,6,7]
      for(is=0;is<4;is++) {
        _fv_eq_gamma_ti_fv(spinor1, 5, spinor_field[4+is]+_GSI(iix));
        _fv_eq_gamma_ti_fv(spinor2, 2, spinor1);
        _fv_eq_gamma_ti_fv(cg5_spinor[is], 0, spinor2);
      }
 
      // id_spinor = Identity x spinor_field[8,9,10,11]
      for(is=0;is<4;is++) {
        _fv_eq_fv(id_spinor[is], spinor_field[8+is]+_GSI(iix));
      }


      // first contribution

      for(ir1=0;ir1<4;ir1++) {
        for(ir2=0; ir2<4;ir2++) {

          ir3 = Cg5_perm[ir2];

          _cm_eq_fv_trans_ti_fv(scs, cg5_spinor[ir3], id_spinor[ir2]);
          
          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];

            w2.re = p5g0p5_spinor[ir1][2*(3*ir1+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir1][2*(3*ir1+perm_tab_3[ib][2])+1];
           
            _co_eq_co_ti_co(&w, &w1, &w2);

            // extra minus sign because of pemutation a'b'c' ---> c'b'a'
            dtmp  = -w.re * Cg5_sign[ir2] * dsign * perm_tab_3_sign[ib];
            dtmp2 = -w.im * Cg5_sign[ir2] * dsign * perm_tab_3_sign[ib];
            connt[2*it  ]  += dtmp;
            connt[2*it+1]  += dtmp2;
            connq[2*iix  ] += dtmp;
            connq[2*iix+1] += dtmp2;

          }  // of ib
        }  // of ir2
      }  // of ir1


      // second contribution

      for(ir1=0;ir1<4;ir1++) {
        for(ir2=0; ir2<4;ir2++) {

          ir3 = Cg5_perm[ir2];

          _cm_eq_fv_trans_ti_fv(scs, cg5_spinor[ir3], id_spinor[ir1]);
              
          for(ib=0;ib<6;ib++) {

            w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
            w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
            w2.re = p5g0p5_spinor[ir2][2*(3*ir1+perm_tab_3[ib][2])  ];
            w2.im = p5g0p5_spinor[ir2][2*(3*ir1+perm_tab_3[ib][2])+1];
            
            _co_eq_co_ti_co(&w, &w1, &w2);

            dtmp  = -w.re * Cg5_sign[ir2] * dsign * perm_tab_3_sign[ib];
            dtmp2 = -w.im * Cg5_sign[ir2] * dsign * perm_tab_3_sign[ib];
            connt[2*it  ]  += dtmp;
            connt[2*it+1]  += dtmp2;
            connq[2*iix  ] += dtmp;
            connq[2*iix+1] += dtmp2;

          }  // of ib
        }  // of ir2
      }  // of ir1

      iix++;
    }} // of ix, it

  }  // of loop on iperm


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
   * finish calculation of connt
   ***********************************************/
  if(g_propagator_bc_type == 0 ) {
    // multiply with phase factor
    fprintf(stdout, "# [] multiplying with boundary phase factor\n");
    for(it=0;it<T_global;it++) {
      ir = (it - sx0 + T_global) % T_global;
      w1.re = cos( 3. * M_PI*(double)ir / (double)T_global );
      w1.im = sin( 3. * M_PI*(double)ir / (double)T_global );
      w2.re = connt[2*it  ];
      w2.im = connt[2*it+1];
      _co_eq_co_ti_co( (complex*)(connt+2*it), &w1, &w2) ;
    }

  } else if(g_propagator_bc_type == 1 ) {
    // multiply with step function
    fprintf(stdout, "# [] multiplying with boundary step function\n");
    for(it=0;it<T_global;it++) {
      if( it < sx0 ) {
        connt[2*it  ] *= -1.;
        connt[2*it+1] *= -1.;
      }
    }

  }

  sprintf(filename, "proton_2pt_v2.%.4d.t%.2dx%.2dy%.2dz%.2d", Nconf, sx0, sx1, sx2, sx3);
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
        w2.re = connq[iix  ];
        w2.im = connq[iix+1];
        _co_eq_co_ti_co( (complex*)(connq+iix), &w1, &w2) ;
        iix += 2;
      }
    }
  } else if (g_propagator_bc_type == 1) {
    // multiply with step function
    fprintf(stdout, "# [] multiplying with boundary step function\n");
    for(it=0;it<T_global;it++) {
      if( it < sx0 ) {
        iix = 2 * it * VOL3;
        for(ix=0;ix<VOL3;ix++) {
          connq[iix  ] *= -1.;
          connq[iix+1] *= -1.;
          iix += 2;
        }
      }
    }
  }

  if(write_ascii) {
    sprintf(filename, "proton_2pt_v2_x.%.4d.t%.2dx%.2dy%.2dz%.2d.ascii", Nconf, sx0, sx1, sx2, sx3);
    write_contraction(connq, (int*)NULL, filename, 1, 2, 0);
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

      w2.re = connq[iix  ];
      w2.im = connq[iix+1];
      _co_eq_co_ti_co( (complex*)(connq+iix), &w1, &w2) ;

      iix += 2; 
    }}}  // of x3, x2, x1
  }  // of it


  // write to file
  sprintf(filename, "proton_2pt_v2_q.%.4d.t%.2dx%.2dy%.2dz%.2d", Nconf, sx0, sx1, sx2, sx3);
  sprintf(contype, "proton 2-pt. function, (t,q_1,q_2,q_3)-dependent, source_timeslice = %d", sx0);
  write_lime_contraction(connq, filename, 64, 1, contype, Nconf, 0);
  if(write_ascii) {
    sprintf(filename, "proton_2pt_v2_q.%.4d.%.2d.ascii", Nconf, sx0);
    write_contraction(connq, (int*)NULL, filename, 1, 2, 0);
  }

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();
  if(connt!= NULL) free(connt);
  if(connq!= NULL) free(connq);
  if(gauge_trafo != NULL) free(gauge_trafo);

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
