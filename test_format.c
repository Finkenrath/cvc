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

  sid = g_sourceid;
  if(format==0) {
    sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
    fprintf(stdout, "Reading spinor field in lime format from %s\n", filename);
    if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) return(-4);
  }
  else if(format==1) {
    sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
    fprintf(stdout, "Reading spinor field in CMI format from %s\n", filename);
    if(read_cmi(g_spinor_field[1], filename) != 0) return(-4);
  }


  for(ix=0; ix<12*VOLUME; ix++) fprintf(stdout, "%8d%25.16e%25.16e\n", ix,
    g_spinor_field[1][2*ix], g_spinor_field[1][2*ix+1]);
  
  return(0);

}
