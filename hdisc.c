/****************************************************
 * hdisc.c
 *
 * So 9. Okt 16:27:16 EEST 2011
 *
 * PURPOSE:
 * - psi0, psi2 - propgators (original, fuzzed)
 * - psi1, psi3 - sources (original, noise-reduced)
 * - _NOTE_ that (mubar, epsbar) <---> (musigma, -mudelta)
 * TODO:
 * DONE:
 * CHANGES:
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

#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "Q_h_phi.h"
#include "fuzz.h"
#include "fuzz2.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for disconnected contributions\n");
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
  
  int c, i, mu, K=16;
  int count        = 0;
  int filename_set = 0;
  int x0;
  int estat; // exit status
  unsigned long int ix, idx;
  unsigned long int x1, VOL3, index_min;
  int sid;
  double *disc = (double*)NULL, *buffer=NULL, *buffer2=NULL;
  int verbose = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24];
  double *gauge_field_f=NULL, *gauge_field_timeslice=NULL;
  double v4norm = 0., vvnorm = 0.;
  double *psi0 = NULL, *psi1 = NULL, *psi2 = NULL, *psi3 = NULL;
  complex w;
  FILE *ofs[4];
  double addreal, addimag;

/* Initialise all the gamma matrix combinations 
   g5, g1, g2, g3, ig0g5, ig0gi, -1, -g5gi, g0 -g5g0gi */
  int gindex[] = {5, 1, 2, 3, 6, 7, 8, 9, 4, 10, 11, 12, 0, 13, 14, 15};

  double gsign[]  = {-1., 1., 1., 1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1.};

#ifdef MPI
  MPI_Status status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vgf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // local time stamp
  g_the_time = time;
  if(g_cart_id == 0) {
    fprintf(stdout, "\n# [disc] using global time stamp %s", ctime(&g_the_time));
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

#ifdef MPI
#  if ! ( (defined PARALLELTX) || (defined PARALLELTXY) )
  T = T_global / g_nproc;
  Tstart = g_cart_id * T;
#  endif
#else
  T            = T_global;
  Tstart       = 0;
#endif
  VOL3 = LX*LY*LZ;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n",
		  g_cart_id, g_cart_id, T_global, g_cart_id, T, g_cart_id, Tstart, g_cart_id, LX_global, g_cart_id, LX,
		  g_cart_id, LXstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  /**********************************************
   * read the gauge field 
   **********************************************/
//  if(N_ape>0 || Nlong>0) {
    alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
    xchange_gauge();
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);
//  } else {
//    g_gauge_field = (double*)NULL;
//  } 
 
  if(Nlong > 0) {
//    N_ape     = 1; 
//    alpha_ape = 0.4;
    if(g_cart_id==0) fprintf(stdout, "# apply fuzzing of gauge field and propagators with parameters:\n"\
                                     "# Nlong = %d\n# N_ape = %d\n# alpha_ape = %f\n", Nlong, N_ape, alpha_ape);
    alloc_gauge_field(&gauge_field_f, VOLUMEPLUSRAND);
#if !( (defined PARALLELTX) || (defined PARALLELTXY) )
    gauge_field_timeslice = (double*)calloc(72*VOL3, sizeof(double));
    if( gauge_field_timeslice == (double*)NULL  ) {
      fprintf(stderr, "Error, could not allocate mem for gauge_field_timeslice\n");
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(2);
    }

    for(x0=0; x0<T; x0++) {
      memcpy((void*)gauge_field_timeslice, (void*)(g_gauge_field+_GGI(g_ipt[x0][0][0][0],0)), 72*VOL3*sizeof(double));
      for(i=0; i<N_ape; i++) {
        fprintf(stdout, "# [] APE smearing time slice %d step %d\n", x0, i);
        APE_Smearing_Step_Timeslice(gauge_field_timeslice, alpha_ape);
      }
      if(Nlong > 0) {
        fuzzed_links_Timeslice(gauge_field_f, gauge_field_timeslice, Nlong, x0);
      } else {
        memcpy(gauge_field_f+_GGI(g_ipt[x0][0][0][0], 0), gauge_field_timeslice, 72*VOL3*sizeof(double));
      }
    }
    free(gauge_field_timeslice);
#else 
    for(i=0; i<N_ape; i++) {
      APE_Smearing_Step(g_gauge_field, alpha_ape);
      xchange_gauge_field_timeslice(g_gauge_field);
    }

    if ( Nlong > 0 ) {
      if(g_cart_id==0) fprintf(stdout, "\n# [hdisc] fuzzing gauge field ...\n");
      fuzzed_links2(gauge_field_f, g_gauge_field, Nlong);
    } else {
      memcpy(gauge_field_f, g_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));
    }
    xchange_gauge_field(gauge_field_f);
    read_lime_gauge_field_doubleprec(filename);
    xchange_gauge();
#endif
/*
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<4; mu++) {
      for(i=0; i<9; i++) {
        fprintf(stdout, "%6d%3d%3d%25.16e%25.16e%25.16e%25.16e\n", ix, mu, i,
          gauge_field_f[_GGI(ix,mu)+2*i], gauge_field_f[_GGI(ix,mu)+2*i+1],
          g_gauge_field[_GGI(ix,mu)+2*i], g_gauge_field[_GGI(ix,mu)+2*i+1]);
      }}
    }
*/
  }  // of if Nlong > 0

  /* allocate memory for the spinor fields */
  no_fields = 8;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  /* allocate memory for the contractions */
/*
#ifdef PARALLELTX
  if(g_xs_id==0) {idx = 4 * 4 * K * T_global * 2;}
  else           {idx = 4 * 4 * K * T        * 2;}
#else
  if(g_cart_id==0) {idx = 4 * 4 * K * T_global * 2;}
  else             {idx = 4 * 4 * K*  T        * 2;}
#endif
*/
  disc = (double*)calloc(32*K*T, sizeof(double));
  if( disc==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for disc\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  buffer = (double*)calloc(32*K*T, sizeof(double));
  if( buffer==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for buffer\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }
  buffer2 = (double*)calloc(32*K*T_global, sizeof(double));
  if( buffer2==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for buffer2\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(5);
  }

  if(g_cart_id==0) {
    sprintf(filename, "hdisc-ss.k0v4.%.4d", Nconf);
    ofs[0] = fopen(filename, "w");
    sprintf(filename, "hdisc-sc.k0v4.%.4d", Nconf);
    ofs[1] = fopen(filename, "w");
    sprintf(filename, "hdisc-cs.k0v4.%.4d", Nconf);
    ofs[2] = fopen(filename, "w");
    sprintf(filename, "hdisc-cc.k0v4.%.4d", Nconf);
    ofs[3] = fopen(filename, "w");
    if(ofs[0]==(FILE*)NULL || ofs[1]==(FILE*)NULL || ofs[2]==(FILE*)NULL || ofs[3]==(FILE*)NULL) {
      fprintf(stderr, "Error, could not open files for writing.\n");
#ifdef MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
#endif
        exit(6);
    }
  }

  /*****************************************
   *  HPE coefficients
   *****************************************/
/*
  if(format==1) {
*/
    addimag = 2*g_kappa*g_musigma/sqrt(1 + 4*g_kappa*g_kappa*(g_musigma*g_musigma-g_mudelta*g_mudelta)) * 
      LX*LY*LZ*3*4*2.*g_kappa*g_kappa*4;
//    addreal = (1.+2*g_kappa*g_mudelta)/sqrt(1 + 4*g_kappa*g_kappa*(g_musigma*g_musigma-g_mudelta*g_mudelta)) * 
//      LX*LY*LZ*3*4*2.*g_kappa*g_kappa*4;
    addreal = (1.- 2*g_kappa*g_mudelta)/sqrt(1 + 4*g_kappa*g_kappa*(g_musigma*g_musigma-g_mudelta*g_mudelta)) * 
      LX*LY*LZ*3*4*2.*g_kappa*g_kappa*4;
    v4norm = 1. / ( 8. * g_kappa * g_kappa );
    vvnorm = 1. / ( 8. * g_kappa * g_kappa );
/*
  } else {
    addimag = 2*g_kappa*g_musigma/sqrt(1 + 4*g_kappa*g_kappa*(g_musigma*g_musigma-g_mudelta*g_mudelta)) * 
      LX*LY*LZ*3*4*2.*g_kappa*2;
    addreal = (1.+2*g_kappa*g_mudelta)/sqrt(1 + 4*g_kappa*g_kappa*(g_musigma*g_musigma-g_mudelta*g_mudelta)) * 
      LX*LY*LZ*3*4*2.*g_kappa*2;
    v4norm = 1. / ( 4. * g_kappa  );
    vvnorm = 1. / ( 4. * g_kappa  );
  }
*/
  if(g_cart_id==0) fprintf(stdout, "# addimag = %25.16e;\t addreal = %25.16e\n"\
                                   "# v4norm  = %25.16e;\t vvnorm  = %25.16e\n", addimag, addreal, v4norm, vvnorm);

  /******************************************
   * start loop on source id.s
   ******************************************/
  count = -1;
  for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
    for(ix=0; ix<32*K*T; ix++) disc[ix]   = 0.;
    for(ix=0; ix<32*K*T; ix++) buffer[ix]   = 0.;
    for(ix=0; ix<32*K*T_global; ix++) buffer2[ix]   = 0.;

    /* read the new propagator */
    sprintf(filename, "%s.%.4d.%.5d.hinverted", filename_prefix, Nconf, sid); 
/*    sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid); */
    estat = read_lime_spinor(g_spinor_field[2], filename, 0);
    if( estat != 0 ) {
      fprintf(stderr, "[%2d] Error, could not read from file %s at position 0\n", g_cart_id, filename);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(7);
    }
    estat =  read_lime_spinor(g_spinor_field[3], filename, 1);
    if( estat != 0 ) {
      fprintf(stderr, "[%2d] Error, could not read from file %s at position 1\n", g_cart_id, filename);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(7);
    }

    count++;
    xchange_field(g_spinor_field[2]);
    xchange_field(g_spinor_field[3]);

    /* calculate the source: apply Q_phi_tbc */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    Q_h_phi(g_spinor_field[0], g_spinor_field[1], g_spinor_field[2], g_spinor_field[3]);
    xchange_field(g_spinor_field[0]); 
    xchange_field(g_spinor_field[1]); 

    // print the sources
/*
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<12; mu++) {
        fprintf(stdout, "%6d%3d%25.16e%25.16e%25.16e%25.16e\n", ix, mu,
          g_spinor_field[0][_GSI(ix)+2*mu], g_spinor_field[0][_GSI(ix)+2*mu+1],
          g_spinor_field[1][_GSI(ix)+2*mu], g_spinor_field[1][_GSI(ix)+2*mu+1]);
      }
    }
*/


#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "\n# [hdisc] time for applying Q_tm_h: %e seconds\n", retime-ratime);


    /* apply gamma5_BdagH4_gamma5 */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    gamma5_B_h_dagH4_gamma5(g_spinor_field[4], g_spinor_field[5], g_spinor_field[0], g_spinor_field[1], g_spinor_field[6], g_spinor_field[7]);

/*
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<12; mu++) {
        fprintf(stdout, "%6d%3d%25.16e%25.16e%25.16e%25.16e\n", ix, mu,
          g_spinor_field[4][_GSI(ix)+2*mu], g_spinor_field[4][_GSI(ix)+2*mu+1],
          g_spinor_field[5][_GSI(ix)+2*mu], g_spinor_field[5][_GSI(ix)+2*mu+1]);
      }
    }
*/

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for applying noise reduction: %e seconds\n", retime-ratime);


#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(Nlong>0) {
      if(g_cart_id==0) fprintf(stdout, "# fuzzing propagator with Nlong = %d\n", Nlong);
      memcpy((void*)g_spinor_field[6], (void*)g_spinor_field[2], 24*(VOLUME+RAND)*sizeof(double));
/*      xchange_field_timeslice(g_spinor_field[6]); */
      Fuzz_prop3(gauge_field_f, g_spinor_field[6], g_spinor_field[0], Nlong);
      xchange_field_timeslice(g_spinor_field[6]);

      memcpy((void*)g_spinor_field[7], (void*)g_spinor_field[3], 24*(VOLUME+RAND)*sizeof(double));
/*      xchange_field_timeslice(g_spinor_field[7]); */
      Fuzz_prop3(gauge_field_f, g_spinor_field[7], g_spinor_field[1], Nlong);
      xchange_field_timeslice(g_spinor_field[7]);
    } else {
      for(ix=0;ix<VOLUME;ix++) { _fv_eq_zero(g_spinor_field[6]+_GSI(ix)); }
      for(ix=0;ix<VOLUME;ix++) { _fv_eq_zero(g_spinor_field[7]+_GSI(ix)); }
    }

/*
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<12; mu++) {
        fprintf(stdout, "%6d%3d%25.16e%25.16e%25.16e%25.16e\n", ix, mu,
          g_spinor_field[6][_GSI(ix)+2*mu], g_spinor_field[6][_GSI(ix)+2*mu+1],
          g_spinor_field[7][_GSI(ix)+2*mu], g_spinor_field[7][_GSI(ix)+2*mu+1]);
      }
    }
*/
    // recalculate the sources --- they are changed in Fuzz_prop3
    Q_h_phi(g_spinor_field[0], g_spinor_field[1], g_spinor_field[2], g_spinor_field[3]);
    xchange_field(g_spinor_field[0]); 
    xchange_field(g_spinor_field[1]); 

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for fuzzing: %e seconds\n", retime-ratime);

    /********************************
     * add new contractions to disc
     ********************************/
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(c=0; c<4; c++)
    {
      if(c==0) { psi0 = g_spinor_field[2]; psi1 = g_spinor_field[4]; psi2 = g_spinor_field[6]; psi3 = g_spinor_field[0]; }
      if(c==1) { psi0 = g_spinor_field[2]; psi1 = g_spinor_field[5]; psi2 = g_spinor_field[6]; psi3 = g_spinor_field[1]; }
      if(c==2) { psi0 = g_spinor_field[3]; psi1 = g_spinor_field[4]; psi2 = g_spinor_field[7]; psi3 = g_spinor_field[0]; }
      if(c==3) { psi0 = g_spinor_field[3]; psi1 = g_spinor_field[5]; psi2 = g_spinor_field[7]; psi3 = g_spinor_field[1]; }

/*
      for(ix=0; ix<VOLUME; ix++) {
        for(mu=0; mu<12; mu++) {
          fprintf(stdout, "%6d%3d%16.7e%16.7e%16.7e%16.7e%16.7e%16.7e%16.7e%16.7e\n", ix, mu,
            psi0[_GSI(ix)+2*mu], psi0[_GSI(ix)+2*mu+1], psi1[_GSI(ix)+2*mu], psi1[_GSI(ix)+2*mu+1],
            psi2[_GSI(ix)+2*mu], psi2[_GSI(ix)+2*mu+1], psi3[_GSI(ix)+2*mu], psi3[_GSI(ix)+2*mu+1]);
        }
      }
*/

      for(x0=0; x0<T; x0++) {
        for(mu=0; mu<16; mu++) {
          index_min =  x0 * K + mu + c * 4 * K * T;
          for(x1=0; x1<VOL3; x1++) {
            ix  = x0*VOL3 + x1;
            idx = _GSI( ix );

            _fv_eq_gamma_ti_fv(spinor1, mu, psi0+idx);
            _co_eq_fv_dag_ti_fv(&w, psi1+idx, spinor1);
	    disc[2*(         index_min)  ] += w.re;
	    disc[2*(         index_min)+1] += w.im;

            if(Nlong>0) {
              _fv_eq_gamma_ti_fv(spinor1, mu, psi2+idx);
    	      _co_eq_fv_dag_ti_fv(&w, psi1+idx, spinor1);
  	      disc[2*(  K*T + index_min)  ] += w.re;
	      disc[2*(  K*T + index_min)+1] += w.im;
            }

            _fv_eq_gamma_ti_fv(spinor1, mu, psi0+idx);
            _co_eq_fv_dag_ti_fv(&w, psi3+idx, spinor1);
	    disc[2*(2*K*T + index_min)  ] += w.re;
	    disc[2*(2*K*T + index_min)+1] += w.im;

            if(Nlong>0) {
              _fv_eq_gamma_ti_fv(spinor1, mu, psi2+idx);
    	      _co_eq_fv_dag_ti_fv(&w, psi3+idx, spinor1);
  	      disc[2*(3*K*T + index_min)  ] += w.re;
	      disc[2*(3*K*T + index_min)+1] += w.im;
            }
          }
        }
      }  // of loop on x0

      for(x0=0; x0<T; x0++) {   
          disc[2*(      x0*K+4 + 4*c*K*T)  ] += addreal;
          disc[2*(      x0*K+5 + 4*c*K*T)+1] -= addimag;

        if(Nlong>0) {
          disc[2*(K*T + x0*K+4 + 4*c*K*T)  ] += addreal;
          disc[2*(K*T + x0*K+5 + 4*c*K*T)+1] -= addimag;
        }

      }
    }  // of c=0,...,4 
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time for contracting: %e seconds\n", retime-ratime);


#ifdef MPI
    /* collect results to disc */
#if (defined PARALLELTX) || (defined PARALLELTXY)
    MPI_Allreduce(disc, buffer, 32*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
    MPI_Allgather(buffer, 32*K*T, MPI_DOUBLE, buffer2, 32*K*T, MPI_DOUBLE, g_xs_comm);
#  else
    MPI_Gather(disc, 32*K*T, MPI_DOUBLE, buffer2, 32*K*T, MPI_DOUBLE, 0, g_cart_grid);
#  endif
#else 
    memcpy((void*)buffer2, (void*)disc, 32*K*T_global*sizeof(double));
#endif

    /* write current disc to file */

    if(g_cart_id==0) {
      for(c=0; c<4; c++) {
        if(sid==g_sourceid) fprintf(ofs[c], "#%6d%3d%3d%3d%3d\t%f\t%f\t%f\t%f\n", Nconf, T_global, LX_global, LY_global, LZ, 
          g_kappa, g_mu, g_musigma, g_mudelta);
        for(x0=0; x0<T_global; x0++) {
          for(mu=0; mu<16; mu++) {
            idx = gindex[mu];
            ix = K*(x0%T) + idx + 16*K*T*(x0/T) + c*4*K*T;
            fprintf(ofs[c], "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e\n",
              Nconf, mu, x0, sid,
              gsign[mu]*buffer2[2*(      ix)]*v4norm, gsign[mu]*buffer2[2*(      ix)+1]*v4norm,
              gsign[mu]*buffer2[2*(  K*T+ix)]*v4norm, gsign[mu]*buffer2[2*(  K*T+ix)+1]*v4norm,
              gsign[mu]*buffer2[2*(2*K*T+ix)]*vvnorm, gsign[mu]*buffer2[2*(2*K*T+ix)+1]*vvnorm,
              gsign[mu]*buffer2[2*(3*K*T+ix)]*vvnorm, gsign[mu]*buffer2[2*(3*K*T+ix)+1]*vvnorm);
          }
        }
      }
    }
    if(g_cart_id==0) fprintf(stdout, "# finished all sid %d\n", sid);


  }  /* of loop on sid */

  if(g_cart_id==0) { fclose(ofs[0]); fclose(ofs[1]); fclose(ofs[2]); fclose(ofs[3]); }



  /* free the allocated memory, finalize */
  free(g_gauge_field); 
  if(no_fields>0) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field);
  }
  free_geometry();
  free(disc);
  free(buffer);
  free(buffer2);
  if(Nlong>0) free(gauge_field_f);

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [disc] %s# [disc] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [disc] %s# [disc] end of run\n", ctime(&g_the_time));
  }

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
