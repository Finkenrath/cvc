/****************************************************
 * cvc_2pt_disc.c
 *
 * Mo 10. Dez 09:48:15 CET 2012
 *
 * PURPOSE
 * - originally copied from cvc_2pt.c
 * - local loops g0, g1, g2, g3 and
 *   1-point-split conserved current loop
 * - without fuzzing
 * - without hopping expansion
 * DONE:
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
#include "read_input_parser.h"

void usage() {
  fprintf(stdout, "Code to perform contractions for disconnected contributions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose [no effect, lots of stdout output it]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -l Nlong for fuzzing [default -1, no fuzzing]\n");
  fprintf(stdout, "         -a no of steps for APE smearing [default -1, no smearing]\n");
  fprintf(stdout, "         -k alpha for APE smearing [default 0.]\n");
  EXIT(0);
}

#ifdef MPI
#define CLOCK MPI_Wtime()
#else
#define CLOCK ((double)clock() / CLOCKS_PER_SEC )
#endif

int main(int argc, char **argv) {
  
  const int K = 20;
  int c, i, mu;
  int count        = 0;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, ix, idx;
  int VOL3;
  int sid;
  double *disc = (double*)NULL;
  int verbose = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_field_f=NULL;
  complex w;
  FILE *ofs1=NULL;

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

#ifdef MPI
  T = T_global / g_nproc;
  Tstart = g_cart_id * T;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
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
    fprintf(stderr, "[cvc_2pt_disc] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  // read the gauge field
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_disc] reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();

  // measure the plaquette
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_disc] measured plaquette value: %25.16e\n", plaq);

  // allocate memory for the spinor fields
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  // allocate memory for the contractions
  disc = (double*)calloc(K*T*2, sizeof(double));
  if( disc==(double*)NULL ) {
    fprintf(stderr, "[cvc_2pt_disc] Error, could not allocate memory for disc\n");
    EXIT(3);
  }
  memset(disc, 0, K*T*2*sizeof(double));

  if(g_cart_id==0) {
    sprintf(filename, "cvc_2pt_disc.%.4d", Nconf);
    ofs1 = fopen(filename, "w");
    if(ofs1==NULL) {
      EXIT(5);
    }
  }

  // start loop on source id.s
  for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
    memset(disc, 0, 2*K*T*sizeof(double));

    // read the new propagator
    // sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid); 
    // sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
    sprintf(filename, "%s.%.4d.0000.%.2d.inverted", filename_prefix, Nconf, sid);
    fprintf(stdout, "# [cvc_2pt_disc] reading spinor field from file %s\n", filename);
    if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) {
      fprintf(stderr, "[cvc_2pt_disc] proc%.2d Error, could not read from file %s\n", g_cart_id, filename);
      EXIT(4);
    }
    count++;
    xchange_field(g_spinor_field[1]);

    // calculate the source: apply Q_phi_tbc
    ratime = CLOCK;

    Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);
    xchange_field(g_spinor_field[0]); 

    retime = CLOCK;
    if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_disc] time to apply Q_tm %e seconds\n", retime-ratime);

    // add new contractions to disc
    ratime = CLOCK;

    for(x0=0; x0<T; x0++) {       // loop on time
      for(x1=0; x1<VOL3; x1++) {  // loop on sites in timeslice
        ix = x0*VOL3 + x1;
        // (1) local currents
        for(mu=0; mu<16; mu++) {  // loop on index of gamma matrix
          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[1][_GSI(ix)]);
  	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1);
          disc[2*(mu*T + x0)  ] += w.re;
	  disc[2*(mu*T + x0)+1] += w.im;
        }  // of loop on gamma matrices

        // (2) point-split currents
        for(mu=0; mu<4; mu++) {
          _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

          _fv_eq_cm_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(g_iup[ix][mu]));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_mi_eq_fv(spinor2, spinor1);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor2);
          disc[2*( (16+mu)*T + x0)  ] += 0.5 * w.re;
          disc[2*( (16+mu)*T + x0)+1] += 0.5 * w.im;

          _fv_eq_cm_dag_ti_fv(spinor1, U_, g_spinor_field[1]+_GSI(ix));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_pl_eq_fv(spinor2, spinor1);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(g_iup[ix][mu]), spinor2);
          disc[2*( (16+mu)*T + x0)  ] += 0.5 * w.re;
          disc[2*( (16+mu)*T + x0)+1] += 0.5 * w.im;

        }  // of loop on currents
      }    // of loop on spacial volume
    }      // of loop on timeslices

    retime = CLOCK;
    if(g_cart_id==0) fprintf(stdout, "# [cvc_2pt_disc] contractions in %e seconds\n", retime-ratime);

    // write current disc to file

    if(g_cart_id==0) {
      if(sid==g_sourceid) fprintf(ofs1, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, T, LX, LY, LZ, g_kappa, g_mu);

      for(mu=0; mu<K; mu++) {
        for(x0=0; x0<T; x0++) {
          ix = mu*T + x0;
          fprintf(ofs1, "%6d%4d%4d%3d%25.16e%25.16e\n",
              Nconf, sid, mu, x0, disc[2*ix  ], disc[2*ix+1]);
        }
      }
#ifdef MPI
      for(c=1; c<g_nproc; c++) {
        MPI_Recv(disc, K*2*T, MPI_DOUBLE, c, 100+c, g_cart_grid, &status);
        for(mu=0; mu<K; mu++) {
          for(x0=0; x0<T; x0++) {
            ix = mu*T + x0;

            fprintf(ofs1, "%6d%4d%4d%3d%25.16e%25.16e\n",
                Nconf, sid, mu, c*T+x0, disc[2*ix  ], disc[2*ix+1]);
          }
        }
      }
#endif
    }
#ifdef MPI
    else {
      for(c=1; c<g_nproc; c++) {
        if(g_cart_id==c) {
          MPI_Send(disc, 2*K*T, MPI_DOUBLE, 0, 100+c, g_cart_grid);
        }
      }
    }
#endif
  }  // of loop on sid

  if(g_cart_id==0) { fclose(ofs1); }

  // free the allocated memory, finalize
  free(g_gauge_field); g_gauge_field=NULL;
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); g_spinor_field=NULL;
  free_geometry();
  free(disc);

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [cvc_2pt_disc] %s# [cvc_2pt_disc] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [cvc_2pt_disc] %s# [cvc_2pt_disc] end of run\n", ctime(&g_the_time));
  }

#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}
