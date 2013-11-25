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

int main(int argc, char **argv) {
  
  int c, i, mu;
  int count        = 0;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, ix;
  int sid;
  double *disc = (double*)NULL;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24];
  double _2kappamu;
  complex w;
  FILE *ofs;
  double hopexp_coeff[8];

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
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
  }

  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n",
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at);

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

  /* allocate memory for the spinor fields */
  no_fields = 3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);


for(sid=0; sid<12; sid++) {

  /* read the new propagator */
  if(format==0) {
    sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
    if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) return(-4);
  }
  else if(format==1) {
    sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
    if(read_cmi(g_spinor_field[1], filename) != 0) return(-4);
  }
  count++;

  /* calculate the source: apply Q_phi_tbc */
  Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);


  /* apply the Hopping matrix */
  Hopping(g_spinor_field[2], g_spinor_field[1]);


  /* add the missing part for having applied full Q */

  _2kappamu = 2. * g_kappa * g_mu;
  mul_one_pm_imu_inv(g_spinor_field[1], -1., VOLUME);
  for(ix=0; ix<VOLUME; ix++) {
    _fv_ti_eq_re(&g_spinor_field[1][_GSI(ix)], (1.+_2kappamu*_2kappamu));
    _fv_pl_eq_fv(&g_spinor_field[2][_GSI(ix)], &g_spinor_field[1][_GSI(ix)]);
    _fv_ti_eq_re(&g_spinor_field[2][_GSI(ix)], 1./(2.*g_kappa));
  }
/*
  for(ix=0; ix<24; ix++) spinor1[ix] = 0.;
  spinor1[2*sid] = 1.;
  mul_one_pm_imu_inv(spinor1, -1., 1);
  fprintf(stdout, "sid = %d\n", sid);
  for(ix=0; ix<12; ix++) fprintf(stdout, "%3d%25.16e%25.16e\n", ix, spinor1[2*ix], spinor1[2*ix+1]);
*/
  /* write current disc to file */

  sprintf(filename, "check_hop.%.2d", sid);
  ofs = fopen(filename, "w");
  if(ofs==(FILE*)NULL) return(-5);
  fprintf(ofs, "#%6d%3d%3d%3d%3d\t%f\t%f\n", Nconf, 
    T, LX, LY, LZ, g_kappa, g_mu);
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<12; mu++) {
      fprintf(ofs, "%6d%4d%25.16e%25.16e%25.16e%25.16e\n", ix, mu,
        g_spinor_field[2][_GSI(ix)+2*mu], g_spinor_field[2][_GSI(ix)+2*mu+1],
	g_spinor_field[0][_GSI(ix)+2*mu], g_spinor_field[0][_GSI(ix)+2*mu+1]);
    }
  }
  fclose(ofs);

}
  /* free the allocated memory, finalize */
  free(g_gauge_field); g_gauge_field=(double*)NULL;
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field); g_spinor_field=(double**)NULL;
  free_geometry();
#ifdef MPI
  MPI_Finalize();
#endif

  return(0);

}
