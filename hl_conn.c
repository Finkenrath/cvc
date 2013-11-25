/****************************************************
 * hl_conn.c
 *
 * Thu Mar  4 09:29:56 CET 2010
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

#ifdef _MMS_ALL_MASS_COMB
#  define _MMS_MASS_DIAGONAL
#  define _MMS_LIGHT_ALL_HEAVY
#endif

int main(int argc, char **argv) {
  
  int c, i, j, k, ll, sl;
  int count;
  int filename_set = 0;
  int timeslice, mms1=0, mms2=0;
  int x0, x1, x2, ix, idx;
  int VOL3;
  int n_c=1, n_s=4;
  int K=20, itype;
  double *cconn  = (double*)NULL;
#ifdef MPI
  double *buffer = (double*)NULL;
#endif
  double *work=NULL;
  int sigmalight=0, sigmaheavy=0;
  double *mms_masses=NULL, mulight=0., muheavy=0.;
  char *mms_extra_masses_file="cvc.extra_masses.input";
  int verbose = 0;
  char filename[200];
  double ratime, retime;
  double plaq;
  double *gauge_field_timeslice=NULL, *gauge_field_f=NULL;
  double **chi=NULL, **psi=NULL;
  double *Ctmp=NULL;
  FILE *ofs=NULL;
/*  double sign_adj5[] = {-1., -1., -1., -1., +1., +1., +1., +1., +1., +1., -1., -1., -1., 1., -1., -1.}; */
  double conf_gamma_sign[] = {1., 1., 1., 1., 1., -1., -1., -1., -1.};

  /**************************************************************************************************
   * charged stuff
   * here we loop over ll, ls, sl, ss (order source-sink)
   * pion:
   * g5-g5, g5-g0g5, g0g5-g5, g0g5-g0g5, g0-g0, g5-g0, g0-g5, g0g5-g0, g0-g0g5
   * rho:
   * gig0-gig0, gi-gi, gig5-gig5, gig0-gi, gi-gig0, gig0-gig5, gig5-gig0, gi-gig5, gig5-gi
   * a0, b1:
   * 1-1, gig0g5-gig0g5
   **************************************************************************************************/
  int gindex1[] = {5, 5, 6, 6, 0, 5, 0, 6, 0,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 10, 11, 12, 1, 2, 3, 10, 11, 12, 7, 8, 9, 1, 2, 3, 7, 8, 9,
                   4, 13, 14, 15};

  int gindex2[] = {5, 6, 5, 6, 0, 0, 5, 0, 6,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 1, 2, 3, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 1, 2, 3,
                   4, 13, 14, 15};

  /* due to twisting we have several correlators that are purely imaginary */
  int isimag[]  = {0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0};

  double isneg_std[]=    {+1., -1., +1., -1., +1., +1., +1., +1., -1.,
                          -1., +1., -1., -1., +1., +1., +1., -1., +1.,
                          +1., -1.};
  double isneg[20];

  /* every correlator for the rho part including gig0 either at source
   * or at sink has a different relative sign between the 3 contributions */
  double vsign[]= {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1.,
                   1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};

#ifdef MPI
  MPI_Status status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vgf:p:")) != -1) {
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

  /* initialize MPI parameters */
  mpi_init(argc, argv);

#ifdef MPI
#  ifndef PARALLELTX
  T = T_global / g_nproc_t;
  Tstart = g_proc_coords[0] * T;
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
		  g_cart_id, g_cart_id, T_global,  g_cart_id, T,  g_cart_id, Tstart, 
                             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  for(i = 0; i < 20; i++) isneg[i] = isneg_std[i];

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  read_lime_gauge_field_doubleprec(filename);
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
 if(g_cart_id==0) fprintf(stdout, "# time to read gauge field: %e seconds\n", retime-ratime);



#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  xchange_gauge();
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
 if(g_cart_id==0) fprintf(stdout, "# time to exchange gauge field: %e seconds\n", retime-ratime);

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);


  if(g_cart_id==0) {
    fprintf(stdout, "# apply fuzzing of gauge field and propagators with parameters:\n"\
                    "# Nlong = %d\n# N_ape = %d\n# alpha_ape = %f\n", Nlong, N_ape, alpha_ape);
  }
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
#ifndef PARALLELTX
  alloc_gauge_field(&gauge_field_f, VOLUME);
  if( (gauge_field_timeslice = (double*)malloc(72*VOL3*sizeof(double))) == (double*)NULL  ) {
    fprintf(stderr, "Error, could not allocate mem for gauge_field_timeslice\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 3);
    MPI_Finalize();
#endif
    exit(2);
  }
  for(x0=0; x0<T; x0++) {
    memcpy((void*)gauge_field_timeslice, (void*)(g_gauge_field+_GGI(g_ipt[x0][0][0][0],0)), 72*VOL3*sizeof(double));
    for(i=0; i<N_ape; i++) {
      APE_Smearing_Step_Timeslice(gauge_field_timeslice, alpha_ape);
    }
    if(Nlong > -1) {
      fuzzed_links_Timeslice(gauge_field_f, gauge_field_timeslice, Nlong, x0);
    } else {
      memcpy((void*)(gauge_field_f+_GGI(g_ipt[x0][0][0][0],0)), (void*)gauge_field_timeslice, 72*VOL3*sizeof(double));
    }
  }
  free(gauge_field_timeslice);
#else
  for(i=0; i<N_ape; i++) {
    APE_Smearing_Step(g_gauge_field, alpha_ape);
    xchange_gauge_field_timeslice(g_gauge_field);
  }

  alloc_gauge_field(&gauge_field_f, VOLUMEPLUSRAND);

  if(Nlong > 0) {
    fuzzed_links(gauge_field_f, g_gauge_field, Nlong);
  } else {
    memcpy((void*)gauge_field_f, (void*)g_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));
  }
  xchange_gauge_field(gauge_field_f);

  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();
#endif
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
 if(g_cart_id==0) fprintf(stdout, "# time for smearing/fuzzing gauge field: %e seconds\n", retime-ratime);

  /* test: print the fuzzed APE smeared gauge field to stdout */
/*
  for(ix=0; ix<36*VOLUME; ix++) {
    fprintf(stdout, "%6d%25.16e%25.16e\n", ix, g_gauge_field[2*ix], g_gauge_field[2*ix+1]);
  }
*/

  /* allocate memory for the spinor fields */
  no_fields = 8;
  no_fields *= n_c;
  no_fields++;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
#ifndef PARALLELTX
  for(i=0; i<no_fields-1; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUMEPLUSRAND);
#else
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);
#endif

  /* allocate memory for the contractions */
#ifdef PARALLELTX
  if(g_xs_id==0) { idx = 8*K*T_global; } 
  else           { idx = 8*K*T; }
#else
  if(g_cart_id==0) { idx = 8*K*T_global; } 
  else             { idx = 8*K*T; }
#endif
  cconn = (double*)calloc(idx, sizeof(double));
  if( cconn==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for cconn\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 4);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<idx; ix++) cconn[ix] = 0.;

#ifdef MPI
  buffer = (double*)calloc(idx, sizeof(double));
  if( buffer==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for buffer\n");
    MPI_Abort(MPI_COMM_WORLD, 5);
    MPI_Finalize();
    exit(4);
  }
#endif

  if( (Ctmp = (double*)calloc(2*T, sizeof(double))) == NULL ) {
    fprintf(stderr, "Error, could not allocate mem for Ctmp\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 6);
    MPI_Finalize();
#endif
    exit(5);
  }

  /*********************************************
   * read the extra masses for mms
   * - if mms is not used, use g_mu as the mass 
   *********************************************/
  if( (mms_masses = (double*)calloc(g_no_extra_masses+1, sizeof(double))) == NULL ) {
    fprintf(stderr, "Error, could allocate mms_masses\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 7);
    MPI_Finalize();
#endif
    exit(6);
  }
  mms_masses[0] = g_mu;
  if(g_no_extra_masses>0) {
    if( (ofs=fopen(mms_extra_masses_file, "r"))==NULL ) {
      fprintf(stderr, "Error, could not open file %s for reading\n", mms_extra_masses_file);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 8);
      MPI_Finalize();
#endif
      exit(7);
    }
    for(i=0; i<g_no_extra_masses; i++) fscanf(ofs, "%lf", mms_masses+i+1);
    fclose(ofs);
  }
  if(g_cart_id==0) {
    fprintf(stdout, "# mms masses:\n");
    for(i=0; i<=g_no_extra_masses; i++) fprintf(stdout, "# mass[%2d] = %e\n",
      i, mms_masses[i]);
  }
  timeslice = g_source_timeslice;

#ifdef _MMS_MASS_DIAGONAL
  /****************************************************************
   * (1) loop on the mass-diagonal correlators mms1 = mms2 = k
   *     for 0 \le k \le g_no_extra_masses
   ****************************************************************/
  for(k=0; k<=g_no_extra_masses; k++) {  
    mms1 = k; 
    mms2 = k;
    count = -1;
/*    for(sigmalight=-1; sigmalight<=1; sigmalight+=2) { */
    for(sigmalight=-1; sigmalight<=-1; sigmalight+=2) {
    for(sigmaheavy=-1; sigmaheavy<=1; sigmaheavy+=2) {
      count++;
      for(j=0; j<8*K*T; j++) cconn[j] = 0.;
      mulight = (double)sigmalight * mms_masses[k];
      muheavy = (double)sigmaheavy * mms_masses[k];
      /*************************************
       * begin loop on LL, LS, SL, SS
       *************************************/
      ll = 0;
      for(j=0; j<4; j++) {
        work = g_spinor_field[no_fields-1];
        if(j==0) {
          /* local-local (source-sink) -> phi[0-3]^dagger.p[0-3] -> p.p */
          ll = 0;
          for(i=0; i<n_s*n_c; i++) {
            sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
              filename_prefix, Nconf, timeslice, i, mms1);
#ifdef MPI
            ratime = MPI_Wtime();
#else
            ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
            read_lime_spinor(work, filename, 0);
#ifdef MPI
            retime = MPI_Wtime();
#else
            retime = (double)clock() / CLOCKS_PER_SEC;
#endif
            if(g_cart_id==0) fprintf(stdout, "# time to read spinor field: %e seconds\n", retime-ratime);

#ifdef MPI
            ratime = MPI_Wtime();
#else
            ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
            xchange_field(work);
#ifdef MPI
            retime = MPI_Wtime();
#else
            retime = (double)clock() / CLOCKS_PER_SEC;
#endif
            if(g_cart_id==0) fprintf(stdout, "# time to exchange spinor field: %e seconds\n", retime-ratime);
            Qf5(g_spinor_field[i], work, mulight);
            Qf5(g_spinor_field[i+n_s*n_c], work, muheavy);
          }
          chi  = &g_spinor_field[0];
          psi  = &g_spinor_field[n_s*n_c];
        } else if(j==1) {
          if(Nlong>-1) {
            /* fuzzed-local -> phi[0-3]^dagger.phi[4-7] -> p.f */
            ll = 2;
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i=n_s*n_c; i<2*n_s*n_c; i++) {
              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i, mms2);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i], work, muheavy);
            }
          } else {
            /* local-smeared */
            ll = 1; 
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i = 0; i < 2*n_s*n_c; i++) {
              xchange_field_timeslice(g_spinor_field[i]);
              for(c=0; c<N_Jacobi; c++) {
                Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i], work, kappa_Jacobi);
                xchange_field_timeslice(g_spinor_field[i]);
              }
            }
          }
        } else if(j==2) {
          if(Nlong>-1) {
            /* local-fuzzed -> phi[0-3]^dagger.phi[4-7] -> p.pf */
            ll = 1;
            chi  = &g_spinor_field[0];
            psi  = &g_spinor_field[n_s*n_c];
            for(i=0; i<n_s*n_c; i++) {
              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i, mms2);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i+n_s*n_c], work, muheavy);
              if(g_cart_id==0) fprintf(stdout, "# fuzzing prop. with Nlong=%d, N_APE=%d, alpha_APE=%f\n",
                                 Nlong, N_ape, alpha_ape);
              xchange_field_timeslice(g_spinor_field[i+n_s*n_c]);
#ifdef MPI
              ratime = MPI_Wtime();
#else
              ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
              Fuzz_prop2(gauge_field_f, g_spinor_field[i+n_s*n_c], work, Nlong);
#ifdef MPI
              retime = MPI_Wtime();
#else
              retime = (double)clock() / CLOCKS_PER_SEC;
#endif
              if(g_cart_id==0) fprintf(stdout, "# time to fuzz spinor field no. %d: %e seconds\n", i+n_s*n_c, retime-ratime);
            }
          } else {
            /* smeared-local */
            ll = 2;
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i=0; i<n_s*n_c; i++) {
              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i+n_s*n_c, mms1);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i], work, mulight);

              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i+n_s*n_c, mms2);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i+n_s*n_c], work, muheavy);
            }
          }
        } else if(j==3) {
          ll = 3;
          if(Nlong>-1) {
            /* fuzzed-fuzzed -> phi[0-3]^dagger.phi[4-7] -> f.pf */
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i=0; i<n_s*n_c; i++) {
              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i+n_s*n_c, mms1);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i], work, mulight);
            }
          } else {
            /* smeared-smeared -> phi[0-3]^dagger.phi[4-7] -> f.pf */
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i = 0; i < 2*n_s*n_c; i++) {
              xchange_field_timeslice(g_spinor_field[i]);
              for(c=0; c<N_Jacobi; c++) {
                Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i], work, kappa_Jacobi);
                xchange_field_timeslice(g_spinor_field[i]);
              }
            }
          }
        }

        /************************************************************
         * the charged contractions
         ************************************************************/
#ifdef MPI
        ratime = MPI_Wtime();
#else
        ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
        sl = 2*ll*T*K;
        itype = 1; 
        /* pion sector */
        for(idx=0; idx<9; idx++) {
          contract_twopoint(&cconn[sl], gindex1[idx], gindex2[idx], chi, psi, n_c);
/*          for(x0=0; x0<T; x0++) fprintf(stdout, "pion: %3d%25.16e%25.16e\n", x0, 
            cconn[sl+2*x0]/(double)VOL3/2./g_kappa/g_kappa, cconn[sl+2*x0+1]/(double)VOL3/2./g_kappa/g_kappa); */
          sl += (2*T);        
          itype++; 
        }

        /* rho sector */
        for(idx = 9; idx < 36; idx+=3) {
          for(i = 0; i < 3; i++) {
            for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
            contract_twopoint(Ctmp, gindex1[idx+i], gindex2[idx+i], chi, psi, n_c);
            for(x0=0; x0<T; x0++) {
              cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-9)/3]*vsign[idx-9+i]*Ctmp[2*x0  ]);
              cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-9)/3]*vsign[idx-9+i]*Ctmp[2*x0+1]);
            }
/*
            for(x0=0; x0<T; x0++) {
              x1 = (x0+timeslice)%T_global;
              fprintf(stdout, "rho: %3d%25.16e%25.16e\n", x0, 
                vsign[idx-9+i]*Ctmp[2*x1  ]/(double)VOL3/2./g_kappa/g_kappa, 
                vsign[idx-9+i]*Ctmp[2*x1+1]/(double)VOL3/2./g_kappa/g_kappa);
            }
*/
          }
          sl += (2*T); 
          itype++;
        }
    
        /* the a0 */
        contract_twopoint(&cconn[sl], gindex1[36], gindex2[36], chi, psi, n_c);
        sl += (2*T);
        itype++;

        /* the b1 */
        for(i=0; i<3; i++) {
          for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
          idx = 37;
          contract_twopoint(Ctmp, gindex1[idx+i], gindex2[idx+i], chi, psi, n_c);
          for(x0=0; x0<T; x0++) { 
            cconn[sl+2*x0  ] += (vsign[idx-9+i]*Ctmp[2*x0  ]);
            cconn[sl+2*x0+1] += (vsign[idx-9+i]*Ctmp[2*x0+1]);
          }
        }
#ifdef MPI
        retime = MPI_Wtime();
#else
        retime = (double)clock() / CLOCKS_PER_SEC;
#endif
        if(g_cart_id==0) fprintf(stdout, "# time for contraction j=%d: %e seconds\n", j, retime-ratime);
      }  /* of j=0,...,3 */

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
#ifdef MPI
      /* collect the results */
#ifdef PARALLELTX
      MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
      MPI_Gather(buffer, 8*K*T, MPI_DOUBLE, cconn, 8*K*T, MPI_DOUBLE, 0, g_xs_comm);
#else
      MPI_Gather(cconn, 8*K*T, MPI_DOUBLE, buffer, 8*K*T, MPI_DOUBLE, 0, g_cart_grid);
      if(g_cart_id==0) memcpy((void*)cconn, (void*)buffer, 8*K*T_global*sizeof(double));
#endif
#endif
      /* write/add to file */
      if(g_cart_id==0) {
        sprintf(filename, "charged.%.4d.%.2d.%.2d.%.2d", Nconf, timeslice, mms1, mms2);
        if(count==0) {
          ofs=fopen(filename, "w");
        } else {
          ofs=fopen(filename, "a");
        }
        if( ofs == (FILE*)NULL ) {
          fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
          MPI_Abort(MPI_COMM_WORLD, 9);
          MPI_Finalize();
#endif
          exit(8);
        }
        fprintf(stdout, "# writing correlators to file %s\n", filename);
        fprintf(ofs, "# %5d%3d%3d%3d%3d%15.8e%15.8e%15.8e%3d%3d\n", 
          Nconf, T_global, LX, LY, LZ, g_kappa, mms_masses[mms1], mms_masses[mms2], -sigmalight, -sigmaheavy);
        for(idx=0; idx<K; idx++) {
          for(ll=0; ll<4; ll++) {
            x1 = (0+timeslice) % T_global;
            i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0,
              isneg[idx]*cconn[i]/(VOL3*g_nproc_x)/g_kappa/g_kappa/2., 0.);
            for(x0=1; x0<T_global/2; x0++) {
              x1 = ( x0+timeslice) % T_global;
              x2 = (-x0+timeslice+T_global) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
              j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + isimag[idx];
/*              fprintf(stdout, "idx=%d; x0=%d, x1=%d, x2=%d, i=%d, j=%d\n", idx, x0, x1, x2, i, j); */
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0,
                isneg[idx]*cconn[i]/(VOL3*g_nproc_x)/g_kappa/g_kappa/2., isneg[idx]*cconn[j]/(VOL3*g_nproc_x)/g_kappa/g_kappa/2.); 
            }
            x0 = T_global/2;
            x1 = (x0+timeslice) % T_global;
            i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0,
              isneg[idx]*cconn[i]/(VOL3*g_nproc_x)/g_kappa/g_kappa/2., 0.);
          }
        }    
        fclose(ofs);
      }  /* of if g_cart_id == 0 */
#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
      if(g_cart_id==0) fprintf(stdout, "# time for collecting/writing contractions: %e seconds\n", retime-ratime);

    }}  /* of sigmalight/sigmaheavy */
  }  /* of i=0,...,g_no_extra_masses */
#endif

#ifdef _MMS_LIGHT_ALL_HEAVY
  /**************************************************************************
   * (2) loop on the mms extra masses
   * - mms1 is fixed to the light quark mass, i.e. mms1 = 0
   * - mms2 runs through the extra masses, 1 \le mms2 \le g_no_extra_masses
   **************************************************************************/
  mms1 = 0;
  for(k=0; k<g_no_extra_masses; k++) {
    mms2    = k+1;
    count = -1; 
/*    for(sigmalight=-1; sigmalight<=1; sigmalight+=2) { */
    for(sigmalight=-1; sigmalight<=-1; sigmalight+=2) {
    for(sigmaheavy=-1; sigmaheavy<=1; sigmaheavy+=2) {
      count++;
      for(j=0; j<8*K*T; j++) cconn[j] = 0.;
      mulight = (double)sigmalight * mms_masses[mms1];
      muheavy = (double)sigmaheavy * mms_masses[mms2];

      /*************************************
       * begin loop on LL, LS, SL, SS
       *************************************/
      ll = 0;
      for(j=0; j<4; j++) {
        work = g_spinor_field[no_fields-1];
        if(j==0) {
          /* local-local (source-sink) -> phi[0-3]^dagger.p[0-3] -> p.p */
          ll = 0;
          for(i=0; i<n_s*n_c; i++) {
            sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
              filename_prefix, Nconf, timeslice, i, mms1);
            read_lime_spinor(work, filename, 0);
            xchange_field(work);
            Qf5(g_spinor_field[i], work, mulight);

            sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
              filename_prefix, Nconf, timeslice, i, mms2);
            read_lime_spinor(work, filename, 0);
            xchange_field(work);
            Qf5(g_spinor_field[i+n_s*n_c], work, muheavy);
          }
          chi  = &g_spinor_field[0];
          psi  = &g_spinor_field[n_s*n_c];
        } else if(j==1) {
          if(Nlong>-1) {
            /* fuzzed-local -> phi[0-3]^dagger.phi[4-7] -> p.f */
            ll = 2;
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i=n_s*n_c; i<2*n_s*n_c; i++) {
              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted",
                filename_prefix, Nconf, timeslice, i, mms2);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i], work, muheavy);
            }
          } else {
            /* local-smeared */
            ll = 1; 
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i = 0; i < 2*n_s*n_c; i++) {
              xchange_field_timeslice(g_spinor_field[i]);
              for(c=0; c<N_Jacobi; c++) {
                Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i], work, kappa_Jacobi);
                xchange_field_timeslice(g_spinor_field[i]);
              }
            }
          }
        } else if(j==2) {
          if(Nlong>-1) {
            /* local-fuzzed -> phi[0-3]^dagger.phi[4-7] -> p.pf */
            ll = 1;
            chi  = &g_spinor_field[0];
            psi  = &g_spinor_field[n_s*n_c];
            for(i=0; i<n_s*n_c; i++) {
              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i, mms2);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i+n_s*n_c], work, muheavy);
              if(g_cart_id==0) fprintf(stdout, "# fuzzing prop. with Nlong=%d, N_APE=%d, alpha_APE=%f\n",
                                 Nlong, N_ape, alpha_ape);
              xchange_field_timeslice(g_spinor_field[i+n_s*n_c]);
              Fuzz_prop2(gauge_field_f, g_spinor_field[i+n_s*n_c], work, Nlong);
            }
          } else {
            /* smeared-local */
            ll = 2;
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i=0; i<n_s*n_c; i++) {
              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i+n_s*n_c, mms1);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i], work, mulight);

              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i+n_s*n_c, mms2);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i+n_s*n_c], work, muheavy);
            }
          }
        } else if(j==3) {
          ll = 3;
          if(Nlong>-1) {
            /* fuzzed-fuzzed -> phi[0-3]^dagger.phi[4-7] -> f.pf */
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i=0; i<n_s*n_c; i++) {
              sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", 
                filename_prefix, Nconf, timeslice, i+n_s*n_c, mms1);
              read_lime_spinor(work, filename, 0);
              xchange_field(work);
              Qf5(g_spinor_field[i], work, mulight);
            }
          } else {
            /* smeared-smeared -> phi[0-3]^dagger.phi[4-7] -> f.pf */
            chi = &g_spinor_field[0];
            psi = &g_spinor_field[n_s*n_c];
            for(i = 0; i < 2*n_s*n_c; i++) {
              xchange_field_timeslice(g_spinor_field[i]);
              for(c=0; c<N_Jacobi; c++) {
                Jacobi_Smearing_Step_one(gauge_field_f, g_spinor_field[i], work, kappa_Jacobi);
                xchange_field_timeslice(g_spinor_field[i]);
              }
            }
          }
        }

        /************************************************************
         * the charged contractions
         ************************************************************/
        sl = 2*ll*T*K;
        itype = 1; 
        /* pion sector */
        for(idx=0; idx<9; idx++) {
          contract_twopoint(&cconn[sl], gindex1[idx], gindex2[idx], chi, psi, n_c);
/*          for(x0=0; x0<T; x0++) fprintf(stdout, "pion: %3d%25.16e%25.16e\n", x0, 
            cconn[sl+2*x0]/(double)VOL3/2./g_kappa/g_kappa, cconn[sl+2*x0+1]/(double)VOL3/2./g_kappa/g_kappa); */
          sl += (2*T);        
          itype++; 
        }

        /* rho sector */
        for(idx = 9; idx < 36; idx+=3) {
          for(i = 0; i < 3; i++) {
            for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
            contract_twopoint(Ctmp, gindex1[idx+i], gindex2[idx+i], chi, psi, n_c);
            for(x0=0; x0<T; x0++) {
              cconn[sl+2*x0  ] += (conf_gamma_sign[(idx-9)/3]*vsign[idx-9+i]*Ctmp[2*x0  ]);
              cconn[sl+2*x0+1] += (conf_gamma_sign[(idx-9)/3]*vsign[idx-9+i]*Ctmp[2*x0+1]);
            }
/*
            for(x0=0; x0<T; x0++) {
              x1 = (x0+timeslice)%T_global;
              fprintf(stdout, "rho: %3d%25.16e%25.16e\n", x0, 
                vsign[idx-9+i]*Ctmp[2*x1  ]/(double)VOL3/2./g_kappa/g_kappa, 
                vsign[idx-9+i]*Ctmp[2*x1+1]/(double)VOL3/2./g_kappa/g_kappa);
            }
*/
          }
          sl += (2*T); 
          itype++;
        }
    
        /* the a0 */
        contract_twopoint(&cconn[sl], gindex1[36], gindex2[36], chi, psi, n_c);
        sl += (2*T);
        itype++;

        /* the b1 */
        for(i=0; i<3; i++) {
          for(x0=0; x0<2*T; x0++) Ctmp[x0] = 0.;
          idx = 37;
          contract_twopoint(Ctmp, gindex1[idx+i], gindex2[idx+i], chi, psi, n_c);
          for(x0=0; x0<T; x0++) { 
            cconn[sl+2*x0  ] += (vsign[idx-9+i]*Ctmp[2*x0  ]);
            cconn[sl+2*x0+1] += (vsign[idx-9+i]*Ctmp[2*x0+1]);
          }
        }
      }  /* of j=0,...,3 */

#ifdef MPI
      /* collect the results */
#ifdef PARALLELTX
      MPI_Allreduce(cconn, buffer, 8*K*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
      MPI_Gather(buffer, 8*K*T, MPI_DOUBLE, cconn, 8*K*T, MPI_DOUBLE, 0, g_xs_comm);
#else
      MPI_Gather(cconn, 8*K*T, MPI_DOUBLE, buffer, 8*K*T, MPI_DOUBLE, 0, g_cart_grid);
      if(g_cart_id==0) memcpy((void*)cconn, (void*)buffer, 8*K*T_global*sizeof(double));
#endif
#endif
      /* write/add to file */
      if(g_cart_id==0) {
        sprintf(filename, "charged.%.4d.%.2d.%.2d.%.2d", Nconf, timeslice, mms1, mms2);
        if(count==0) {
          ofs=fopen(filename, "w");
        } else {
          ofs=fopen(filename, "a");
        }
        if( ofs == (FILE*)NULL ) {
          fprintf(stderr, "Error, could not open file %s for writing\n", filename);
#ifdef MPI
          MPI_Abort(MPI_COMM_WORLD, 10);
          MPI_Finalize();
#endif
          exit(8);
        }
        fprintf(stdout, "# writing correlators to file %s\n", filename);
        fprintf(ofs, "# %5d%3d%3d%3d%3d%15.8e%15.8e%15.8e%3d%3d\n", 
          Nconf, T_global, LX, LY, LZ, g_kappa, mms_masses[mms1], mms_masses[mms2], -sigmalight, -sigmaheavy);
        for(idx=0; idx<K; idx++) {
          for(ll=0; ll<4; ll++) {
            x1 = (0+timeslice) % T_global;
            i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, 0,
              isneg[idx]*cconn[i]/(VOL3*g_nproc_x)/g_kappa/g_kappa/2., 0.);
            for(x0=1; x0<T_global/2; x0++) {
              x1 = ( x0+timeslice) % T_global;
              x2 = (-x0+timeslice+T_global) % T_global;
              i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
              j = 2* ( (x2/T)*4*K*T + ll*K*T + idx*T + x2%T ) + isimag[idx];
/*              fprintf(stdout, "idx=%d; x0=%d, x1=%d, x2=%d, i=%d, j=%d\n", idx, x0, x1, x2, i, j); */
              fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0,
                isneg[idx]*cconn[i]/(VOL3*g_nproc_x)/g_kappa/g_kappa/2., isneg[idx]*cconn[j]/(VOL3*g_nproc_x)/g_kappa/g_kappa/2.); 
            }
            x0 = T_global/2;
            x1 = (x0+timeslice) % T_global;
            i = 2* ( (x1/T)*4*K*T + ll*K*T + idx*T + x1%T ) + isimag[idx];
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", idx+1, 2*ll+1, x0,
              isneg[idx]*cconn[i]/(VOL3*g_nproc_x)/g_kappa/g_kappa/2., 0.);
          }
        }    
        fclose(ofs);
      }  /* of if g_cart_id == 0 */

    }}  /* of sigmalight/sigmaheavy */
  }  /* of i=0,...,g_no_extra_masses */
#endif

  /**************************************************
   * free the allocated memory, finalize 
   **************************************************/
  free(g_gauge_field); 
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); 
  free_geometry(); 
  free(cconn);
  free(Ctmp);
  free(gauge_field_f);
  if(mms_masses!=NULL) free(mms_masses);
#ifdef MPI
  free(buffer); 
  MPI_Finalize();
#endif
  return(0);

}
