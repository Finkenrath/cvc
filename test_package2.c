/****************************************************
 * test_package2.c
 *
 * Wed Dec  9 21:06:00 CET 2009
 *
 * PURPOSE:
 * - compare the results of the different contraction
 *   methods
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
#include "invert_Qtm.h"
#include "gauge_io.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to test the cvc package\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, mu;
  int filename_set = 0;
  int sl0, sl1, sl2, sl3;
  double *disc;
  double vp1[8], vp2[8], vp3[8], vp4[8], vp5[8];
  char filename[200];

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
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

  /**************************************
   * set the default values, read input
   **************************************/
  if(filename_set==0) strcpy(filename, "cvc.input.test");
  fprintf(stdout, "# Reading test input from file %s\n", filename);
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

  T = T_global;

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /********************************
   * the source locaton 
   ********************************/
  sl0 = g_source_location/(LX*LY*LZ);
  sl1 = ( g_source_location%(LX*LY*LZ) ) / (LY*LZ);
  sl2 = ( g_source_location%(LY*LZ) ) / (LZ);
  sl3 = g_source_location%LZ;
  fprintf(stdout, "# global sl = (%d, %d, %d, %d)\n", sl0, sl1, sl2, sl3);
  
  if( (disc = (double*)malloc(32*VOLUME*sizeof(double))) == (double*)NULL) {
    exit(102);
  }
   
  /*******************************************************************
   * (1) comparison of results from
   *   - avc_disc_stochastic
   *   - avc_disc_hpe and avc_disc_hpe5
   *   - vp_disc_hpe_loops_red/vp_disc_hpe_stoch to 3rd and 5th order
   *******************************************************************/

  sprintf(filename, "outcvc_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading avc_disc_stochastic data from file %s\n", filename);
  read_contraction(disc, NULL, filename, 4);
  for(mu=0; mu<4; mu++) {
    vp1[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ] / 60.;
    vp1[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1] / 60.;
  }


  sprintf(filename, "cvc_hpe_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading avc_disc_hpe data from file %s\n", filename);
  read_contraction(disc, NULL, filename, 4);
  for(mu=0; mu<4; mu++) {
    vp2[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ] / 60.;
    vp2[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1] / 60.;
  }

  sprintf(filename, "cvc_hpe5_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading avc_disc_hpe5 data from file %s\n", filename);
  read_lime_contraction(disc, filename, 4, 0);
  for(mu=0; mu<4; mu++) {
    vp3[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ];
    vp3[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1];
  }

  sprintf(filename, "vp_disc_hpe03_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading vp_disc_hpe03 data from file %s\n", filename);
  read_lime_contraction(disc, filename, 4, 0);
  for(mu=0; mu<4; mu++) {
    vp4[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ];
    vp4[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1];
  }

  sprintf(filename, "vp_disc_hpe05_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading vp_disc_hpe05 data from file %s\n", filename);
  read_lime_contraction(disc, filename, 4, 0);
  for(mu=0; mu<4; mu++) {
    vp5[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ];
    vp5[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1];
  }

  for(mu=0; mu<4; mu++) {
    
    fprintf(stdout, "\n#--------------------------------------------\n"\
      "# mu = %d\n", mu);
    fprintf(stdout, "%30s%30s%30s\n", "method", "real part", "imaginary part");
    fprintf(stdout, "%30s%30.16e%30.16e\n", "avc_disc_stochastic", vp1[2*mu], vp1[2*mu+1]);
    fprintf(stdout, "%30s%30.16e%30.16e\n", "avc_disc_hpe", vp2[2*mu], vp2[2*mu+1]);
    fprintf(stdout, "%30s%30.16e%30.16e\n", "avc_disc_hpe5", vp3[2*mu], vp3[2*mu+1]);
    fprintf(stdout, "%30s%30.16e%30.16e\n", "vp_disc_hpe03", vp4[2*mu], vp4[2*mu+1]);
    fprintf(stdout, "%30s%30.16e%30.16e\n", "vp_disc_hpe05", vp5[2*mu], vp5[2*mu+1]);
  }

  fprintf(stdout, "\n#=======================================================\n");

  /*******************************************************************
   * (2) comparison of results from
   *   - lvc_disc_stochastic 
   *   - lvc_disc_hpe for 4th and 6th order
   *******************************************************************/
  
  sprintf(filename, "outlvc_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading lvc_disc_stochastic data from file %s\n", filename);
  read_contraction(disc, NULL, filename, 4);
  for(mu=0; mu<4; mu++) {
    vp1[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ] / 60.;
    vp1[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1] / 60.;
  }

  sprintf(filename, "lvc_disc_hpe04_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading lvc_disc_hpe04 data from file %s\n", filename);
  read_lime_contraction(disc, filename, 4, 0);
  for(mu=0; mu<4; mu++) {
    vp2[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ];
    vp2[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1];
  }

  sprintf(filename, "lvc_disc_hpe06_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading lvc_disc_hpe06 data from file %s\n", filename);
  read_lime_contraction(disc, filename, 4, 0);
  for(mu=0; mu<4; mu++) {
    vp3[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ];
    vp3[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1];
  }

  for(mu=0; mu<4; mu++) {
    
    fprintf(stdout, "\n#--------------------------------------------\n"\
      "# mu = %d\n", mu);
    fprintf(stdout, "%30s%30s%30s\n", "method", "real part", "imaginary part");
    fprintf(stdout, "%30s%30.16e%30.16e\n", "lvc_disc_stochastic", vp1[2*mu], vp1[2*mu+1]);
    fprintf(stdout, "%30s%30.16e%30.16e\n", "lvc_disc_hpe04", vp2[2*mu], vp2[2*mu+1]);
    fprintf(stdout, "%30s%30.16e%30.16e\n", "lvc_disc_hpe06", vp3[2*mu], vp3[2*mu+1]);
  }

  fprintf(stdout, "\n#=======================================================\n");


  /*******************************************************************
   * (3) comparison of results from
   *   - vp_disc_hpe_mc1/2 for 3rd and 5th order
   *******************************************************************/
  sprintf(filename, "vp_disc_hpe-01_mc2_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading vp_disc_hpe-01_mc2 data from file %s\n", filename);
  read_lime_contraction(disc, filename, 4, 0);
  for(mu=0; mu<4; mu++) {
    vp1[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ];
    vp1[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1];
  }

  sprintf(filename, "vp_disc_hpe03_mc2_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading vp_disc_hpe03_mc2 data from file %s\n", filename);
  read_lime_contraction(disc, filename, 4, 0);
  for(mu=0; mu<4; mu++) {
    vp2[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ];
    vp2[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1];
  }

  sprintf(filename, "vp_disc_hpe05_mc2_X.%.4d.%.4d", Nconf, Nsave);
  fprintf(stdout, "\n# Reading vp_disc_hpe05_mc2 data from file %s\n", filename);
  read_lime_contraction(disc, filename, 4, 0);
  for(mu=0; mu<4; mu++) {
    vp3[2*mu  ] = disc[_GWI(mu,g_source_location,VOLUME)  ];
    vp3[2*mu+1] = disc[_GWI(mu,g_source_location,VOLUME)+1];
  }


  for(mu=0; mu<4; mu++) {
    
    fprintf(stdout, "\n#--------------------------------------------\n"\
      "# mu = %d\n", mu);
    fprintf(stdout, "%30s%30s%30s\n", "method", "real part", "imaginary part");
    fprintf(stdout, "%30s%30.16e%30.16e\n", "vp_disc_hpe00_mc", vp1[2*mu], vp1[2*mu+1]);
    fprintf(stdout, "%30s%30.16e%30.16e\n", "vp_disc_hpe03_mc", vp2[2*mu], vp2[2*mu+1]);
    fprintf(stdout, "%30s%30.16e%30.16e\n", "vp_disc_hpe05_mc", vp3[2*mu], vp3[2*mu+1]);
  }

  fprintf(stdout, "\n#=======================================================\n");

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  free(disc);

#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
