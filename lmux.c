/****************************************************
  
 * lmux.c
 *
 * Sun Oct  4 15:56:30 CEST 2009
 *
 * PURPOSE:
 * - calculate the Tr[J_\mu(x)] for one lattice site x and all mu
 * - needs (4x3)(spin-colour) x 2 (u/d) pnt source propagators 
 * DONE:
 * TODO:
 * CHANGES: 
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
  fprintf(stdout, "Code to perform contraction of one current operator\n");
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
  
  int c, i, j, mu, nu, ir, is, ia, ib;
  int filename_set = 0;
  int x0, x1, x2, x3, ix;
  int sx0, sx1, sx2, sx3;
  double lmux[8], lmux2[8], up[4][12][24], dn[4][12][24];
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_trafo=(double*)NULL;
  double Usourcebuff[72], *Usource[4];
  FILE *ofs;


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

  T            = T_global;
  Tstart       = 0;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n",\
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /* keep only the four links at source location */ 
  Usource[0] = Usourcebuff;
  Usource[1] = Usourcebuff+18;
  Usource[2] = Usourcebuff+36;
  Usource[3] = Usourcebuff+54;
  _cm_eq_cm(Usource[0], g_gauge_field+_GGI(g_source_location, 0));
  _cm_eq_cm(Usource[1], g_gauge_field+_GGI(g_source_location, 1));
  _cm_eq_cm(Usource[2], g_gauge_field+_GGI(g_source_location, 2));
  _cm_eq_cm(Usource[3], g_gauge_field+_GGI(g_source_location, 3));
  free(g_gauge_field);

  /* allocate memory for the spinor fields */
  no_fields = 1;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* read the propagators, keep only S_u/d (x+mu,x) */
  for(ia=0; ia<12; ia++) {
    get_filename(filename, 4, ia, 1);
    read_lime_spinor(g_spinor_field[0], filename, 0);
    for(mu=0; mu<4; mu++) {
      _fv_eq_fv(up[mu][ia], g_spinor_field[0]+_GSI(g_iup[g_source_location][mu]));
    }
  }

  for(ia=0; ia<12; ia++) {
    get_filename(filename, 4, ia, -1);
    read_lime_spinor(g_spinor_field[0], filename, 0);
    for(mu=0; mu<4; mu++) {
      _fv_eq_fv(dn[mu][ia], g_spinor_field[0]+_GSI(g_iup[g_source_location][mu]));
    }
  }
  free(g_spinor_field[0]);
  free(g_spinor_field);

  /* set lmux to zero */
  for(ix=0; ix< 8; ix++) lmux[ix] = 0.;
  for(ix=0; ix< 8; ix++) lmux2[ix] = 0.;

  /* loop on right Lorentz index nu */
  for(mu=0; mu<4; mu++) {

    _cm_eq_cm_ti_co(U_, Usource[mu], co_phase_up+mu);

    for(ia=0; ia<12; ia++) {
      _fv_eq_cm_ti_fv(spinor1, U_, up[mu][ia]);
      _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
      _fv_mi_eq_fv(spinor2, spinor1);
      lmux[2*mu  ] += -0.5 * spinor2[2*ia  ];
      lmux[2*mu+1] += -0.5 * spinor2[2*ia+1];
    }

    for(ia=0; ia<12; ia++) {
      _fv_eq_cm_ti_fv(spinor1, U_, dn[mu][ia]);
      _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
      _fv_mi_eq_fv(spinor2, spinor1);
      lmux2[2*mu  ] -= -0.5 * spinor2[2*ia  ];
      lmux2[2*mu+1] += -0.5 * spinor2[2*ia+1];
    }

  }

  /* write results */
  sprintf(filename, "cvc_lnuy_X.%.4d", Nconf);
  ofs = fopen(filename, "w");
  for(mu=0; mu<4; mu++)
    fprintf(ofs, "%25.16e%25.16e\n", lmux[2*mu]+lmux2[2*mu], lmux[2*mu+1]+lmux2[2*mu+1]);
/*    fprintf(ofs, "%25.16e%25.16e%25.16e%25.16e\n", lmux[2*mu], lmux[2*mu+1], lmux2[2*mu], lmux2[2*mu+1]); */

  /* test: print U and Su and Sd */
/*
  for(mu=0; mu<4; mu++) {
    sprintf(filename, "Usource.%1d", mu);
    ofs=fopen(filename, "w");
    for(ix=0; ix<9; ix++) fprintf(ofs, "%25.16e%25.16e\n", Usource[mu][2*ix], Usource[mu][2*ix+1]);
    fclose(ofs);
  }

  for(mu=0; mu<4; mu++) {
    sprintf(filename, "up.%1d", mu);
    ofs=fopen(filename, "w");
    for(ia=0; ia<12; ia++) {
      for(ix=0; ix<12; ix++) fprintf(ofs, "%25.16e%25.16e\n", up[mu][ia][2*ix], up[mu][ia][2*ix+1]);
    }
    fclose(ofs);
  }

  for(mu=0; mu<4; mu++) {
    sprintf(filename, "dn.%1d", mu);
    ofs=fopen(filename, "w");
    for(ia=0; ia<12; ia++) {
      for(ix=0; ix<12; ix++) fprintf(ofs, "%25.16e%25.16e\n", dn[mu][ia][2*ix], dn[mu][ia][2*ix+1]);
    }
    fclose(ofs);
  }
*/

  return(0);

}
