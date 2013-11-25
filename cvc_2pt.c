/****************************************************
 * cvc_2pt.c
 *
 * Sat Feb 06 17:52:00 MEST 2010
 *
 * PURPOSE
 * -
 * - the fixed values for Nlong, N_ape and alpha_ape used
 *   in case of fuzzing are taken from Carsten's disc programme
 * TODO:
 * - include disc. contractions of conserved vector current
 * - take out the boundary of gauge_field_f to reduce memory
 *   consumption
 * DONE:
 * - included vv, v4, adapted output format to disc
 * - included fuzzing, smearing
 * - tested tested fuzzing, vv/v4, serial/t-parallel
 *   against disc
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
  double spinor1[24], spinor2[24];
  double _2kappamu;
  double *gauge_field_f=NULL, *gauge_field_timeslice=NULL;
  double v4norm = 0., vvnorm = 0.;
  complex w;
  FILE *ofs1, *ofs2;
/*  double sign_adj5[] = {-1., -1., -1., -1., +1., +1., +1., +1., +1., +1., -1., -1., -1., 1., -1., -1.}; */
  double hopexp_coeff[8], addreal, addimag;
  int gindex[]    = { 5 , 1 , 2 , 3 ,  6 ,10 ,11 ,12 , 4 , 7 , 8 , 9 , 0 ,15 , 14 ,13 };
  int isimag[]    = { 0 , 0 , 0 , 0 ,  1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 ,  1 , 1 };
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
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

  if(Nlong > -1) {
/*    N_ape     = 5; */
    alpha_ape = 0.4;
    if(g_cart_id==0) fprintf(stdout, "# apply fuzzing of gauge field and propagators with parameters:\n"\
                                     "# Nlong = %d\n# N_ape = %d\n# alpha_ape = %f\n", Nlong, N_ape, alpha_ape);
    alloc_gauge_field(&gauge_field_f, VOLUMEPLUSRAND);
    if( (gauge_field_timeslice = (double*)malloc(72*VOL3*sizeof(double))) == (double*)NULL  ) {
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
        APE_Smearing_Step_Timeslice(gauge_field_timeslice, alpha_ape);
      }
      fuzzed_links_Timeslice(gauge_field_f, gauge_field_timeslice, Nlong, x0);
    }
    free(gauge_field_timeslice);
  }

  /* test: print the fuzzed APE smeared gauge field to stdout */
/*
  for(ix=0; ix<36*VOLUME; ix++) {
    fprintf(stdout, "%6d%25.16e%25.16e%25.16e%25.16e\n", ix, gauge_field_f[2*ix], gauge_field_f[2*ix+1], g_gauge_field[2*ix], g_gauge_field[2*ix+1]);
  }
*/

  /* allocate memory for the spinor fields */
  no_fields = 4;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  disc = (double*)calloc(4*16*T*2, sizeof(double));
  if( disc==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for disc\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<4*32*T; ix++) disc[ix] = 0.;

  if(g_cart_id==0) {
    sprintf(filename, "cvc_2pt_disc_vv.%.4d", Nconf);
    ofs1 = fopen(filename, "w");
    sprintf(filename, "cvc_2pt_disc_v4.%.4d", Nconf);
    ofs2 = fopen(filename, "w");
    if(ofs1==(FILE*)NULL || ofs2==(FILE*)NULL) {
#ifdef MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
#endif
        exit(5);
    }
  }

  /* add the HPE coefficients */
  if(format==1) {
    addimag = 2*g_kappa*g_mu/sqrt(1 + 4*g_kappa*g_kappa*g_mu*g_mu)* LX*LY*LZ*3*4*2.*g_kappa*g_kappa*4;
    addreal = 1./sqrt(1 + 4*g_kappa*g_kappa*g_mu*g_mu)*LX*LY*LZ*3*4*2.*g_kappa*g_kappa*4;
    v4norm = 1. / ( 8. * g_kappa * g_kappa );
    vvnorm = g_mu / ( 4. * g_kappa );
  } else {
    addimag = 2*g_kappa*g_mu/sqrt(1 + 4*g_kappa*g_kappa*g_mu*g_mu)* LX*LY*LZ*3*4*2.*g_kappa*2;
    addreal = 1./sqrt(1 + 4*g_kappa*g_kappa*g_mu*g_mu)*LX*LY*LZ*3*4*2.*g_kappa*2;
    v4norm = 1. / ( 4. * g_kappa  );
    vvnorm = g_mu / ( 4. * g_kappa );
  }

  /* calculate additional contributions for 1 and gamma_5 */
  _2kappamu = 2.*g_kappa*g_mu;
  hopexp_coeff[0] = 24. * g_kappa * LX*LY*LZ / (1. + _2kappamu*_2kappamu);
  hopexp_coeff[1] = 0.;
  
  hopexp_coeff[2] = -768. * g_kappa*g_kappa*g_kappa * LX*LY*LZ * _2kappamu*_2kappamu /
   ( (1.+_2kappamu*_2kappamu)*(1.+_2kappamu*_2kappamu)*(1.+_2kappamu*_2kappamu) );
  hopexp_coeff[3] = 0.;

  hopexp_coeff[4] = 0.;
  hopexp_coeff[5] = -24.*g_kappa * LX*LY*LZ * _2kappamu / (1. + _2kappamu*_2kappamu);

  hopexp_coeff[6] = 0.;
  hopexp_coeff[7] = -384. * g_kappa*g_kappa*g_kappa * LX*LY*LZ * 
    (1.-_2kappamu*_2kappamu)*_2kappamu /
   ( (1.+_2kappamu*_2kappamu)*(1.+_2kappamu*_2kappamu)*(1.+_2kappamu*_2kappamu) );

  /* start loop on source id.s */
  for(sid=g_sourceid; sid<=g_sourceid2; sid+=g_sourceid_step) {
    for(ix=0; ix<4*32*T; ix++) disc[ix] = 0.;

    /* read the new propagator */
    sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid); 
/*    sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid); */
    if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) {
      fprintf(stderr, "[%2d] Error, could not read from file %s\n", g_cart_id, filename);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(4);
    }
    count++;
    xchange_field(g_spinor_field[1]);

    /* calculate the source: apply Q_phi_tbc */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);
    xchange_field(g_spinor_field[0]); 
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# time to apply Q_tm %e seconds\n", retime-ratime);


    /* apply gamma5_BdagH4_gamma5 */
    gamma5_BdagH4_gamma5(g_spinor_field[2], g_spinor_field[0], g_spinor_field[3]);

    /* attention: additional factor 2kappa because of CMI format */
/*
    if(format==1) {
      for(ix=0; ix<VOLUME; ix++) {
        _fv_ti_eq_re(&g_spinor_field[2][_GSI(ix)], 2.*g_kappa);
      }
    }
*/

    if(Nlong>-1) {
      if(g_cart_id==0) fprintf(stdout, "# fuzzing propagator with Nlong = %d\n", Nlong);
      memcpy((void*)g_spinor_field[3], (void*)g_spinor_field[1], 24*VOLUMEPLUSRAND*sizeof(double));
      Fuzz_prop(gauge_field_f, g_spinor_field[3], Nlong);
    }

    /* add new contractions to disc */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    for(x0=0; x0<T; x0++) {             /* loop on time */
      for(x1=0; x1<VOL3; x1++) {    /* loop on sites in timeslice */
        ix = x0*VOL3 + x1;
        for(mu=0; mu<16; mu++) { /* loop on index of gamma matrix */

          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[1][_GSI(ix)]);
  	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[2][_GSI(ix)], spinor1);
	  disc[2*(       x0*16+mu)  ] += w.re;
	  disc[2*(       x0*16+mu)+1] += w.im;
     
          _fv_eq_gamma_ti_fv(spinor1, 5, &g_spinor_field[1][_GSI(ix)]);
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[1][_GSI(ix)], spinor2);
	  disc[2*(16*T + x0*16+mu)  ] += w.re;
	  disc[2*(16*T + x0*16+mu)+1] += w.im;
        
          if(Nlong>-1) {
            _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[3][_GSI(ix)]);
    	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[2][_GSI(ix)], spinor1);
	    disc[2*(32*T + x0*16+mu)  ] += w.re;
	    disc[2*(32*T + x0*16+mu)+1] += w.im;
          
            _fv_eq_gamma_ti_fv(spinor1, 5, &g_spinor_field[3][_GSI(ix)]);
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[1][_GSI(ix)], spinor2);
	    disc[2*(48*T + x0*16+mu)  ] += w.re;
	    disc[2*(48*T + x0*16+mu)+1] += w.im;
          }
        }
      }
    }

    if(g_cart_id==0) fprintf(stdout, "# addimag = %25.16e\n", addimag);
    if(g_cart_id==0) fprintf(stdout, "# addreal = %25.16e\n", addreal);
    for(x0=0; x0<T; x0++) {   
      disc[2*(       x0*16+4)  ] += addreal;
      disc[2*(       x0*16+5)+1] -= addimag;
/* 
      if(Nlong>-1) {
        disc[2*(32*T + x0*16+4)  ] += addreal;
        disc[2*(32*T + x0*16+5)+1] -= addimag; 
      }
*/
    }
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# contractions in %e seconds\n", retime-ratime);

    /* write current disc to file */

    if(g_cart_id==0) {
      if(sid==g_sourceid) fprintf(ofs1, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, T, LX, LY, LZ, g_kappa, g_mu);
      if(sid==g_sourceid) fprintf(ofs2, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, T, LX, LY, LZ, g_kappa, g_mu);
      for(x0=0; x0<T; x0++) {
        for(mu=0; mu<16; mu++) {
          idx = gindex[mu];
          ix = 16*x0 + idx;
          if(isimag[mu]==0) {
            fprintf(ofs2, "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e\n",
              Nconf, mu, x0, sid,
              gsign[mu]*disc[2*      ix ]*v4norm, gsign[mu]*disc[2*      ix +1]*v4norm,
              gsign[mu]*disc[2*(32*T+ix)]*v4norm, gsign[mu]*disc[2*(32*T+ix)+1]*v4norm);
          } else {
            fprintf(ofs2, "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e\n",
              Nconf, mu, x0, sid,
              gsign[mu]*disc[2*(     ix)+1]*v4norm, -gsign[mu]*disc[2*      ix ]*v4norm,
              gsign[mu]*disc[2*(32*T+ix)+1]*v4norm, -gsign[mu]*disc[2*(32*T+ix)]*v4norm);
          }
        }
      }
      for(x0=0; x0<T; x0++) {
        for(mu=0; mu<16; mu++) {
          idx = gindex[mu];
          ix = 16*x0 + idx;
          if(isimag[mu]==0) {
            fprintf(ofs1, "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e\n",
              Nconf, mu, x0, sid,
              gsign[mu]*disc[2*(16*T+ix)+1]*vvnorm, -gsign[mu]*disc[2*(16*T+ix)]*vvnorm,
              gsign[mu]*disc[2*(48*T+ix)+1]*vvnorm, -gsign[mu]*disc[2*(48*T+ix)]*vvnorm);
          } else {
            fprintf(ofs1, "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e\n",
              Nconf, mu, x0, sid,
              -gsign[mu]*disc[2*(16*T+ix)]*vvnorm, -gsign[mu]*disc[2*(16*T+ix)+1]*vvnorm,
              -gsign[mu]*disc[2*(48*T+ix)]*vvnorm, -gsign[mu]*disc[2*(48*T+ix)+1]*vvnorm);
          }
        }
      }
#ifdef MPI
      for(c=1; c<g_nproc; c++) {
        MPI_Recv(disc, 128*T, MPI_DOUBLE, c, 100+c, g_cart_grid, &status);
        for(x0=0; x0<T; x0++) {
          for(mu=0; mu<16; mu++) {
            idx=gindex[mu];
            ix = 16*x0 + idx;
            if(isimag[mu]==0) {
              fprintf(ofs2, "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e\n",
                Nconf, mu, c*T+x0, sid,
                gsign[mu]*disc[2*      ix ]*v4norm, gsign[mu]*disc[2*      ix +1]*v4norm,
                gsign[mu]*disc[2*(32*T+ix)]*v4norm, gsign[mu]*disc[2*(32*T+ix)+1]*v4norm);
            } else {
              fprintf(ofs2, "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e\n",
                Nconf, mu, c*T+x0, sid,
                gsign[mu]*disc[2*(     ix)+1]*v4norm, -gsign[mu]*disc[2*      ix ]*v4norm,
                gsign[mu]*disc[2*(32*T+ix)+1]*v4norm, -gsign[mu]*disc[2*(32*T+ix)]*v4norm);
            }
          }
        }
        for(x0=0; x0<T; x0++) {
          for(mu=0; mu<16; mu++) {
            idx = gindex[mu];
            ix = 16*x0 + idx;
            if(isimag[mu]==0) {
              fprintf(ofs1, "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e\n",
                Nconf, mu, c*T+x0, sid,
                gsign[mu]*disc[2*(16*T+ix)+1]*vvnorm, -gsign[mu]*disc[2*(16*T+ix)]*vvnorm,
                gsign[mu]*disc[2*(48*T+ix)+1]*vvnorm, -gsign[mu]*disc[2*(48*T+ix)]*vvnorm);
            } else {
              fprintf(ofs1, "%6d%3d%4d%4d%25.16e%25.16e%25.16e%25.16e\n",
                Nconf, mu, c*T+x0, sid,
                -gsign[mu]*disc[2*(16*T+ix)]*vvnorm, -gsign[mu]*disc[2*(16*T+ix)+1]*vvnorm,
                -gsign[mu]*disc[2*(48*T+ix)]*vvnorm, -gsign[mu]*disc[2*(48*T+ix)+1]*vvnorm);
            }
          }
        }
      }
#endif
    }
#ifdef MPI
    else {
      for(c=1; c<g_nproc; c++) {
        if(g_cart_id==c) {
          MPI_Send(disc, 128*T, MPI_DOUBLE, 0, 100+c, g_cart_grid);
        }
      }
    }
#endif
  }  /* of loop on sid */

  if(g_cart_id==0) { fclose(ofs1); fclose(ofs2); }

  if(g_cart_id==0) {
    fprintf(stdout, "# contributions from HPE:\n");
    fprintf(stdout, "(1) X = id\t%25.16e%25.16e\n"\
                    "          \t%25.16e%25.16e\n"\
    		    "(2) X =  5\t%25.16e%25.16e\n"\
                    "          \t%25.16e%25.16e\n",
		    hopexp_coeff[0], hopexp_coeff[1], hopexp_coeff[2], hopexp_coeff[3],
		    hopexp_coeff[4], hopexp_coeff[5], hopexp_coeff[6], hopexp_coeff[7]);
  }

  /* free the allocated memory, finalize */
  free(g_gauge_field); g_gauge_field=(double*)NULL;
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); g_spinor_field=(double**)NULL;
  free_geometry();
  free(disc);
  if(Nlong>-1) free(gauge_field_f);
#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
