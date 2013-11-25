/****************************************************
 * comp_spinors.c
 *
 * Mon Feb  1 09:52:25 CET 2010
 *
 * PURPOSE:
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
#include "dml.h"
#include "invert_Qtm.h"

void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu;
  int i, j, k, ncon=-1;
  int filename_set = 0;
  int x0, x1, x2, x3, ix;
  double adiffre, adiffim, mdiffre, mdiffim, Mdiffre, Mdiffim, hre, him;
  double *chi=NULL, *psi=NULL, *eta=NULL, *lambda=NULL;
  double plaq, spinor1[24], *gfield=NULL;
  char filename[200];
  char file1[200];
  char file2[200];
  char file3[200];
  char file4[200];
  DML_Checksum checksum;
  FILE *ofs = NULL;
  double norm, norm2, norm3;
  int status;

  while ((c = getopt(argc, argv, "h?vf:N:c:C:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      ncon = atoi(optarg);
      break;
    case 'c':
      strcpy(file1, optarg);
      break;
    case 'C':
      strcpy(file2, optarg);
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
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
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

  /* initialize */
  T      = T_global;
  Tstart = 0;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n",\
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(101);
  }

  geometry();
  init_geometry_5d();
  geometry_5d();

  /* read the gauge field */
/*
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  // alloc_gauge_field(&gfield, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);

  //for(x0=0; x0<T_global; x0++) {
  //  fprintf(stdout, "#\t timeslice %3d\n", x0);
  //  read_lime_gauge_field_doubleprec_timeslice(gfield+_GGI(g_ipt[x0][0][0][0],0), filename, x0, &checksum);
  //}

  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();

  // measure the plaquette
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

*/
/*
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(stdout, "# x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", x0, x1, x2, x3);
    for(i=0; i<4; i++) {
      ix = _GGI( g_ipt[x0][x1][x2][x3], i );
      fprintf(stdout, "#\t direction i=%3d\n", i);
      for(j=0; j<9; j++) {
      fprintf(stdout, "%3d%25.16e%25.16e\n", j, 
        g_gauge_field[ix+2*j], g_gauge_field[ix+2*j+1]);
    }}
  }}}}
*/

  no_fields=5;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /****************************************
   * read read the spinor fields
   ****************************************/

  sprintf(filename, "sc/%s", filename_prefix);
  check_error( read_lime_spinor(g_spinor_field[0], filename, 0), "read_lime_spinor", NULL, 1);

  sprintf(filename, "sg/%s", filename_prefix);
  check_error( read_lime_spinor(g_spinor_field[1], filename, 0), "read_lime_spinor", NULL, 2);

  sprintf(filename, "mc/%s", filename_prefix);
  check_error( read_lime_spinor(g_spinor_field[2], filename, 0), "read_lime_spinor", NULL, 3);

  sprintf(filename, "mg/%s", filename_prefix);
  check_error( read_lime_spinor(g_spinor_field[3], filename, 0), "read_lime_spinor", NULL, 4);


  for(i=0;i<3;i++) {
    for(j=i+1;j<4;j++) {
      for(ix=0;ix<VOLUME;ix++) {
        _fv_eq_fv_mi_fv(g_spinor_field[4]+_GSI(ix), g_spinor_field[i]+_GSI(ix), g_spinor_field[j]+_GSI(ix));
      }
//      for(ix=0;ix<LX*LY*LZ;ix++) {
//        for(k=0;k<12;k++) {
//          fprintf(stdout, "%8d%3d%25.16e%25.16e\n", ix, k, g_spinor_field[4][_GSI(ix)+2*k],  g_spinor_field[4][_GSI(ix)+2*k+1]);
//        }
//      }
      spinor_scalar_product_re(&norm, g_spinor_field[4], g_spinor_field[4], VOLUME);
      spinor_scalar_product_re(&norm2, g_spinor_field[i], g_spinor_field[i], VOLUME);
      spinor_scalar_product_re(&norm3, g_spinor_field[j], g_spinor_field[j], VOLUME);
      fprintf(stdout, "# [] difference for pair (%d, %d) is %e (%e, %e)\n", i, j, sqrt(norm/norm2), norm2, norm3);
    }
  }
/*
  for(ix=0;ix<LX*LY*LZ;ix++) {
    for(i=0;i<12;i++) {
      fprintf(stdout, "%8d%3d%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e\n", ix, i,
          g_spinor_field[0][_GSI(ix)+2*i],  g_spinor_field[0][_GSI(ix)+2*i+1],
          g_spinor_field[1][_GSI(ix)+2*i],  g_spinor_field[1][_GSI(ix)+2*i+1],
          g_spinor_field[2][_GSI(ix)+2*i],  g_spinor_field[2][_GSI(ix)+2*i+1],
          g_spinor_field[3][_GSI(ix)+2*i],  g_spinor_field[3][_GSI(ix)+2*i+1]);
    }
  }
*/
/*
  spinor_scalar_product_re(&norm2, g_spinor_field[0], g_spinor_field[0], VOLUME); 

  status = read_lime_spinor(g_spinor_field[1], filename_prefix2, 0);
  if(status != 0) {
    fprintf(stderr, "[] Error, could not read from file %s\n", filename_prefix2);
    exit(1);
  }
  spinor_scalar_product_re(&norm3, g_spinor_field[1], g_spinor_field[1], VOLUME); 

  for(ix=0;ix<VOLUME;ix++) {
    _fv_mi_eq_fv(g_spinor_field[1]+_GSI(ix), g_spinor_field[0]+_GSI(ix));
  }
  spinor_scalar_product_re(&norm, g_spinor_field[1], g_spinor_field[1], VOLUME); 
  fprintf(stdout, "# [] norm = %e\n", norm);
  fprintf(stdout, "# [] norm2 = %e\n", norm2);
  fprintf(stdout, "# [] norm3 = %e\n", norm3);
*/
/*
  for(i=0; i<1; i++) { 

    sprintf(filename, "comp_spinors.%.2d", i);
    ofs = fopen(filename, "w");

    sprintf(file1, "%s.%.2d.cgmms.00.inverted", filename_prefix, i);
    read_lime_spinor(g_spinor_field[0], file1, 0);

    Qf5(g_spinor_field[1], g_spinor_field[0], -g_mu); 
    //Q_phi_tbc(g_spinor_field[2], g_spinor_field[1]);

    sprintf(file2, "%s.%.2d.inverted", filename_prefix2, i);
    read_lime_spinor(g_spinor_field[3], file2, 0);

    chi    = g_spinor_field[1];
    psi    = g_spinor_field[3];
 
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      fprintf(ofs, "# x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", x0, x1, x2, x3);
      ix = _GSI( g_ipt[x0][x1][x2][x3] );
      for(mu=0; mu<12; mu++) {
        fprintf(ofs, "%3d%25.16e%25.16e%25.16e%25.16e\n", mu, chi[ix+2*mu], chi[ix+2*mu+1], psi[ix+2*mu], psi[ix+2*mu+1]);
      }
    }}}}

    fclose(ofs);

  }
*/
  /****************************************
   * calculate difference
   ****************************************/
/*
  chi = g_spinor_field[1];
  psi = g_spinor_field[3];
 
  ncon = 12;
  mdiffre = fabs(chi[0] - psi[0]);
  mdiffim = fabs(chi[1] - psi[1]);
  Mdiffre = 0.;
  Mdiffim = 0.;
  adiffre = 0.;
  adiffim = 0.;
  for(ix=0; ix<ncon*VOLUME; ix++) {
    adiffre += chi[2*ix  ] - psi[2*ix  ];
    adiffim += chi[2*ix+1] - psi[2*ix+1];
    hre = fabs(chi[2*ix  ] - psi[2*ix  ]);
    him = fabs(chi[2*ix+1] - psi[2*ix+1]);
    if(hre<mdiffre) mdiffre = hre;
    if(hre>Mdiffre) Mdiffre = hre;
    if(him<mdiffim) mdiffim = him;
    if(him>Mdiffim) Mdiffim = him;
  }
  adiffre /= (double)VOLUME * (double)ncon;
  adiffim /= (double)VOLUME * (double)ncon;

  fprintf(stdout, "# Results for files Qf5 %s and %s:\n", file1, file2);
  fprintf(stdout, "average difference\t%25.16e\t%25.16e\n", adiffre, adiffim);
  fprintf(stdout, "minimal abs. difference\t%25.16e\t%25.16e\n", mdiffre, mdiffim);
  fprintf(stdout, "maximal abs. difference\t%25.16e\t%25.16e\n", Mdiffre, Mdiffim);
*/
  /****************************************
   * calculate difference
   ****************************************/
/*
  chi = g_spinor_field[2];
  psi = g_spinor_field[3];
 
  ncon = 12;
  mdiffre = fabs(chi[0] - psi[0]);
  mdiffim = fabs(chi[1] - psi[1]);
  Mdiffre = 0.;
  Mdiffim = 0.;
  adiffre = 0.;
  adiffim = 0.;
  for(ix=0; ix<ncon*VOLUME; ix++) {
    adiffre += chi[2*ix  ] - psi[2*ix  ];
    adiffim += chi[2*ix+1] - psi[2*ix+1];
    hre = fabs(chi[2*ix  ] - psi[2*ix  ]);
    him = fabs(chi[2*ix+1] - psi[2*ix+1]);
    if(hre<mdiffre) mdiffre = hre;
    if(hre>Mdiffre) Mdiffre = hre;
    if(him<mdiffim) mdiffim = him;
    if(him>Mdiffim) Mdiffim = him;
  }
  adiffre /= (double)VOLUME * (double)ncon;
  adiffim /= (double)VOLUME * (double)ncon;

  fprintf(stdout, "# Results for files 1/2/kappa Qf5 %s and %s:\n", file1, file3);
  fprintf(stdout, "average difference\t%25.16e\t%25.16e\n", adiffre, adiffim);
  fprintf(stdout, "minimal abs. difference\t%25.16e\t%25.16e\n", mdiffre, mdiffim);
  fprintf(stdout, "maximal abs. difference\t%25.16e\t%25.16e\n", Mdiffre, Mdiffim);
*/
  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  g_the_time = time(NULL);
  fprintf(stdout, "# [comp_spinor] %s# [comp_spinor] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [comp_spinor] %s# [comp_spinor] end fo run\n", ctime(&g_the_time));
  fflush(stderr);

  return(0);

}
