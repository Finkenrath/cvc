/****************************************************
 * proton_2pt.c
 *
 * Fr 21. Okt 14:00:25 EEST 2011
 *
 * PURPOSE:
 * - calculate the proton 2-point function from point sources
 * TODO:
 * - contractions for sequential source method
 * - ask Giannis about smearing ---> Gaussian smearing ? == ? Jacobi smearing?
 * - 3-dimensional Fourier transform in the spatial coordinates
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
  char filename[200], contype[200];
  double ratime, retime;
  double plaq, dsign, dtmp, dtmp2;
  double *gauge_field_timeslice=NULL, *gauge_field_f=NULL;
  double **chi=NULL, **chi2=NULL, **psi=NULL, **psi2=NULL;
  double spinor1[24], spinor2[24], spinor3[24];
  double scs[18];
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

  while ((c = getopt(argc, argv, "ah?vguf:p:m:")) != -1) {
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
    case 'm':
      mms1 = atoi(optarg);
      fprintf(stdout, "# [] mms1 set to %d\n", mms1);
      break;
    case 'u':
      use_mms = 1;
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [] will write in ascii format\n");
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

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  dims[0]=LX; dims[1]=LY; dims[2]=LZ;
  plan_p = fftwnd_create_plan(3, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
#ifdef MPI
  T = T_global / g_nproc;
  Tstart = g_cart_id * T;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = LX*LY*LZ;
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
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  if(N_ape>0 || N_Jacobi>0) {

    /* read the gauge field */
    alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
    xchange_gauge();

    /* measure the plaquette */
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);


/*    N_ape     = 5; */
/*    alpha_ape = 0.4; */
    if(g_cart_id==0) fprintf(stdout, "# apply fuzzing of gauge field and propagators with parameters:\n"\
                                     "# Nlong = %d\n# N_ape = %d\n# alpha_ape = %f\n", Nlong, N_ape, alpha_ape);
    alloc_gauge_field(&gauge_field_f, VOLUME);
    if( (gauge_field_timeslice = (double*)malloc(72*VOL3*sizeof(double))) == (double*)NULL  ) {
      fprintf(stderr, "Error, could not allocate mem for gauge_field_timeslice\n");
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(2);
    }
    if(g_cart_id==0) fprintf(stdout, "# APE-smearing / fuzzing gauge field with Nlong=%d, N_APE=%d, alpha_APE=%f\n",
          Nlong, N_ape, alpha_ape);
    for(it=0; it<T; it++) {
      memcpy((void*)gauge_field_timeslice, (void*)(g_gauge_field+_GGI(g_ipt[it][0][0][0],0)), 72*VOL3*sizeof(double));
      for(i=0; i<N_ape; i++) {
        APE_Smearing_Step_Timeslice(gauge_field_timeslice, alpha_ape);
      }
      if(Nlong > -1) {
        fuzzed_links_Timeslice(gauge_field_f, gauge_field_timeslice, Nlong, it);
      } else {
        memcpy((void*)(gauge_field_f+_GGI(g_ipt[it][0][0][0],0)), (void*)gauge_field_timeslice, 72*VOL3*sizeof(double));
      }
    }
    free(gauge_field_timeslice);

  /* test: print the fuzzed APE smeared gauge field to stdout */
/*
    for(ix=0; ix<36*VOLUME; ix++) {
      fprintf(stdout, "%6d%25.16e%25.16e\n", ix, g_gauge_field[2*ix], g_gauge_field[2*ix+1]);
    }
*/

  }

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
  no_fields = 12;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUMEPLUSRAND);

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


  /******************************************************
   * contractions
   ******************************************************/
  for(iperm=0;iperm<6;iperm++) {
    ia0 = perm_tab_3[iperm][0];
    ia1 = perm_tab_3[iperm][1];
    ia2 = perm_tab_3[iperm][2];
    dsign = perm_tab_3_sign[iperm];

    fprintf(stdout, "# [] working on combination (%d, %d, %d) with sign %3.0f\n", dsign, ia0, ia1,ia2);

    for(is=0;is<4;is++) {
      sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia0);
      read_lime_spinor(g_spinor_field[is  ], filename, 0);

      sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia1);
      read_lime_spinor(g_spinor_field[4+is], filename, 1);

      sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, g_source_timeslice, 3*is+ia2);
      read_lime_spinor(g_spinor_field[8+is], filename, 0);
    }

    // P_5 Gamma^0 P_5 g_spinor_field[0,1,2,3];
#if !( defined _WILSON_CASE )
    for(is=0;is<4;is++) {
      for(ix=0;ix<VOLUME;ix++) {
        // (1) P_5
        _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[is]+_GSI(ix));
        _fv_eq_fv_ti_im(spinor2, spinor1, 1.);
        _fv_pl_eq_fv(spinor2, g_spinor_field[is]+_GSI(ix) );
        _fv_eq_fv_ti_re(spinor1, spinor2, _ONE_OVER_SQRT2);
        // (2) Gamma^0
        _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1 );
        _fv_ti_eq_re(spinor2, 0.25);
        // (3) P_5
        _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        _fv_eq_fv_ti_im(spinor3, spinor1, 1.);
        _fv_pl_eq_fv(spinor3, spinor2 );
        _fv_eq_fv_ti_re(g_spinor_field[is]+_GSI(ix), spinor3, _ONE_OVER_SQRT2);
      }
    }
#else  // this is the Wilson case; P_5 = identity, use only 1/4(1 + gamma_0)
    for(is=0;is<4;is++) {
      for(ix=0;ix<VOLUME;ix++) {
        // (1) P_5
        _fv_eq_fv(spinor1, g_spinor_field[is]+_GSI(ix));
        // (2) Gamma^0
        _fv_eq_gamma_ti_fv(spinor2, 0, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1 );
        _fv_ti_eq_re(spinor2, 0.25);
        // (3) P_5
        _fv_eq_fv(g_spinor_field[is]+_GSI(ix), spinor2);
      }
    }
#endif


    // C g5 g_spinor_field[4,5,6,7] = g0 g2 g5 g_spinor_field[4,5,6,7]
    for(is=0;is<4;is++) {
      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[4+is]+_GSI(ix));
        _fv_eq_gamma_ti_fv(spinor2, 2, spinor1);
        _fv_eq_gamma_ti_fv(g_spinor_field[4+is]+_GSI(ix), 0, spinor2);
      }
    }

//#ifdef _UNDEF
//#endif
    // first contribution

    for(ir1=0;ir1<4;ir1++) {
      for(ir2=0; ir2<4;ir2++) {

        ir3 = Cg5_perm[ir2];

        for(it=0;it<T;it++) {
          for(ix=0;ix<VOL3;ix++) {
            iix = it * VOL3 + ix;

            _cm_eq_fv_trans_ti_fv(scs, g_spinor_field[4+ir3]+_GSI(iix), g_spinor_field[8+ir2]+_GSI(iix));
          
/*
            fprintf(stdout, "# [] the spinors:\n");
            for(i=0;i<4;i++) {
              for(j=0;j<3;j++) {
                fprintf(stdout, "\t%3d%3d%16.7e%16.7e%16.7e%16.7e\n", i, j,
                   g_spinor_field[4+ir3][_GSI(iix)+2*(3*i+j)], g_spinor_field[4+ir3][_GSI(iix)+2*(3*i+j)+1],
                   g_spinor_field[8+ir2][_GSI(iix)+2*(3*i+j)], g_spinor_field[8+ir2][_GSI(iix)+2*(3*i+j)+1]);
              }
            }
            fprintf(stdout, "# [] scs:\n");
            for(i=0;i<3;i++) {
              for(j=0;j<3;j++) {
                fprintf(stdout, "\t%3d%3d%16.7e%16.7e\n", i,j, scs[2*(3*i+j)], scs[2*(3*i+j)+1]);
              }
            }
*/

            for(ib=0;ib<6;ib++) {

              w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
              w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
              w2.re = g_spinor_field[ir1][_GSI(iix)+ 2*(3*ir1+perm_tab_3[ib][2])  ];
              w2.im = g_spinor_field[ir1][_GSI(iix)+ 2*(3*ir1+perm_tab_3[ib][2])+1];
            
              _co_eq_co_ti_co(&w, &w1, &w2);

              // extra minus sign because of pemutation a'b'c' ---> c'b'a'
              dtmp  = -w.re * Cg5_sign[ir2] * dsign * perm_tab_3_sign[ib];
              dtmp2 = -w.im * Cg5_sign[ir2] * dsign * perm_tab_3_sign[ib];
              connt[2*it  ]  += dtmp;
              connt[2*it+1]  += dtmp2;
              connq[2*iix  ] += dtmp;
              connq[2*iix+1] += dtmp2;

            }  // of ib
          }  // of ix
        }  // of it
      }  // of ir2
    }  // of ir1


    // second contribution

    for(ir1=0;ir1<4;ir1++) {
      for(ir2=0; ir2<4;ir2++) {

        ir3 = Cg5_perm[ir2];

        for(it=0;it<T;it++) {
          for(ix=0;ix<VOL3;ix++) {
            iix = it * VOL3 + ix;

            _cm_eq_fv_trans_ti_fv(scs, g_spinor_field[4+ir3]+_GSI(iix), g_spinor_field[8+ir1]+_GSI(iix));
              
            for(ib=0;ib<6;ib++) {

              w1.re = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])  ];
              w1.im = scs[2*(3*perm_tab_3[ib][0] + perm_tab_3[ib][1])+1];
  
              w2.re = g_spinor_field[ir2][_GSI(iix)+ 2*(3*ir1+perm_tab_3[ib][2])  ];
              w2.im = g_spinor_field[ir2][_GSI(iix)+ 2*(3*ir1+perm_tab_3[ib][2])+1];
            
              _co_eq_co_ti_co(&w, &w1, &w2);

              dtmp  = -w.re * Cg5_sign[ir2] * dsign * perm_tab_3_sign[ib];
              dtmp2 = -w.im * Cg5_sign[ir2] * dsign * perm_tab_3_sign[ib];
              connt[2*it  ]  += dtmp;
              connt[2*it+1]  += dtmp2;
              connq[2*iix  ] += dtmp;
              connq[2*iix+1] += dtmp2;

            }  // of ib
          }  // of ix
        }  // of it
      }  // of ir2
    }  // of ir1


  }  // of loop on iperm


  /***********************************************
   * free gauge fields and spinor fields
   ***********************************************/
  free(g_gauge_field); g_gauge_field=(double*)NULL;
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); g_spinor_field=(double**)NULL;
  if(gauge_field_f != NULL) free(gauge_field_f);

  /***********************************************
   * finish calculation of connt
   ***********************************************/
  // multiply with phase factor
  for(it=0;it<T_global;it++) {
    ir = (it - sx0 + T_global) % T_global;
    w1.re = cos( 3. * M_PI*(double)ir / (double)T_global );
    w1.im = sin( 3. * M_PI*(double)ir / (double)T_global );
//    w1.re = 1.;
//    w1.im = 0.;
    w2.re = connt[2*it  ];
    w2.im = connt[2*it+1];
    _co_eq_co_ti_co( (complex*)(connt+2*it), &w1, &w2) ;
  }

  sprintf(filename, "proton_2pt.%.4d", Nconf);
  ofs = fopen(filename, "w");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
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
  // multiply with phase factor
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
  // Fourier transform
  in  = (fftw_complex*)malloc(VOL3*sizeof(fftw_complex));
  for(it=0;it<T;it++) {
    memcpy(in, connq+2*it*VOL3, 2*VOL3*sizeof(double));
    fftwnd_one(plan_p, in, NULL);
    memcpy(connq+2*it*VOL3, in, 2*VOL3*sizeof(double));
  }
  // write to file
  sprintf(filename, "proton_2pt_q.%.4d", Nconf);
  sprintf(contype, "proton 2-pt. function, (t,q_1,q_2,q_3)-dependent");
  write_lime_contraction(connq, filename, 64, 1, contype, Nconf, 0);
  if(write_ascii) {
    sprintf(filename, "proton_2pt_q.%.4d.ascii", Nconf);
    write_contraction(connq, (int*)NULL, filename, 1, 2, 0);
  }

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  // free(g_gauge_field); g_gauge_field=(double*)NULL;
  // for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  // free(g_spinor_field); g_spinor_field=(double**)NULL;
  // if(gauge_field_f != NULL) free(gauge_field_f);
  free_geometry();
  if(connt!= NULL) free(connt);
  if(connq!= NULL) free(connq);

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
