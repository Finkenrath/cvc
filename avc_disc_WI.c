/****************************************************
 * avc_disc_WI.c
 *
 * Mon Aug 17 13:20:45 MEST 2009
 *
 * TODO: 
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

void usage() {
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, mu, nu;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix;
  double *disc  = (double*)NULL;
  double *pseu=(double*)NULL, *scal=(double*)NULL, *xavc=(double*)NULL;
  double *work = (double*)NULL;
  double q[4];
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  FILE *ofs;

#ifdef MPI
  int *status;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vgf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'g':
      do_gt = 1;
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
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");

  /* read the input file */
  read_input(filename);

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
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

  /* initialize */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifndef MPI
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
#endif
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n",
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at);

#ifdef MPI
  if(T==0) {
    fprintf(stderr, "[%2d] local T is zero; exit\n", g_cart_id);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(2);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  /* allocate memory for the contractions */
  disc  = (double*)calloc(32*VOLUME, sizeof(double));
  work  = (double*)calloc(4*VOLUME, sizeof(double));
  pseu  = (double*)calloc(2*VOLUME, sizeof(double));
  scal  = (double*)calloc(2*VOLUME, sizeof(double));
  xavc  = (double*)calloc(2*VOLUME, sizeof(double));
  if( (disc==(double*)NULL) || (work==(double*)NULL) ||
    (pseu==(double*)NULL) || (scal==(double*)NULL) || (xavc==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) disc[ix] = 0.;
  for(ix=0; ix<4*VOLUME; ix++) work[ix] = 0.;

  sprintf(filename, "%s", filename_prefix);
  ofs = fopen(filename, "r");
  for(ix=0; ix<VOLUME; ix++) 
    for(mu=0; mu<16; mu++) { 
      fscanf(ofs, "%lf%lf", disc+_GWI(mu,ix,VOLUME), disc+_GWI(mu,ix,VOLUME)+1);
/*      fprintf(stdout, "%6d%6d%25.16e%25.16e\n", ix, mu, disc[_GWI(mu,ix,VOLUME)], disc[_GWI(mu,ix,VOLUME)+1]); */
    }
  fclose(ofs);

  for(x0=0; x0<T;  x0++) {
    q[0] = 2. * sin( M_PI * (double)(Tstart+x0) / (double)T_global );
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin( M_PI * (double)(x1) / (double)LX );
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * sin( M_PI * (double)(x2) / (double)LY );
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * sin( M_PI * (double)(x3) / (double)LZ );
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      work[2*ix  ] += q[mu]*q[nu] * disc[_GWI(4*mu+nu,ix,VOLUME)  ];
      work[2*ix+1] += q[mu]*q[nu] * disc[_GWI(4*mu+nu,ix,VOLUME)+1];
    }
    }
  }
  }
  }
  }

  fopen("avc_WI_P", "w");
  for(ix=0; ix<VOLUME; ix++) 
    fprintf(ofs, "%6d%25.16e%25.16e\n", ix, work[2*ix], work[2*ix+1]);
  fclose(ofs);

  ofs = fopen(filename_prefix2, "r");
  for(ix=0; ix<VOLUME; ix++) 
    for(mu=0; mu<4; mu++) { 
      fscanf(ofs, "%lf%lf", disc+_GWI(mu,ix,VOLUME), disc+_GWI(mu,ix,VOLUME)+1);
    }
  fclose(ofs);
  
  for(ix=0; ix<VOLUME; ix++) {
    work[2*ix] = 0.;
    work[2*ix+1] = 0.;
    for(mu=0; mu<4; mu++) {
      work[2*ix] += disc[_GWI(mu,ix,VOLUME)] - disc[_GWI(mu,g_idn[ix][mu],VOLUME)];
      work[2*ix+1] += disc[_GWI(mu,ix,VOLUME)+1] - disc[_GWI(mu,g_idn[ix][mu],VOLUME)+1];
    }
  }

  sprintf(filename, "%s.%.4d.%.2d", gaugefilename_prefix, Nconf, g_sourceid2+1);
  fprintf(stdout, "reading from file %s\n", filename);
  if((ofs = fopen(filename, "r"))==(FILE*)NULL) {return(-6);}
  for(ix=0; ix<VOLUME; ix++) {
    fscanf(ofs, "%lf%lf%lf%lf%lf%lf", 
      pseu+2*ix, pseu+2*ix+1, scal+2*ix, scal+2*ix+1, xavc+2*ix, xavc+2*ix+1);
  }
  fclose(ofs);

  for(ix=0; ix<VOLUME; ix++) {
    work[2*(VOLUME+ix)  ] = 2.*(1./(2.*g_kappa)-4.)*pseu[2*ix  ] - 
      2.*g_mu*scal[2*ix+1] + xavc[2*ix  ];
    work[2*(VOLUME+ix)+1] = 2.*(1./(2.*g_kappa)-4.)*pseu[2*ix+1] + 
      2.*g_mu*scal[2*ix  ] + xavc[2*ix+1];
  }

  fopen("avc_WI_X", "w");
  for(ix=0; ix<VOLUME; ix++) 
    fprintf(ofs, "%6d%25.16e%25.16e%25.16e%25.16e\n", ix, 
      work[2*ix], work[2*ix+1], work[2*(VOLUME+ix)], work[2*(VOLUME+ix)+1]);
  fclose(ofs);
  


  free_geometry();
  free(disc);
  free(work);
  free(pseu);
  free(scal);
  free(xavc);
#ifdef MPI
  free(status);
  MPI_Finalize();
#endif

  return(0);

}
