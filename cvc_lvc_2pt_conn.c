/****************************************************
  
 * cvc_lvc_2pt_conn.c
 *
 * Do 2. Aug 18:13:59 CEST 2012
 *
 * PURPOSE:
 * - contraction for cvc - lvc : conserved / local vector current
 *   at sink / source
 * - t-dependent correlator at zero momentum
 *
 * DONE:
 * TODO:
 * - check, whether only the real part can be used
 * - check, whether second contribution from cvc has to be included 
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include "ifftw.h"
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

#ifdef MPI
#define CLOCK MPI_Wtime()
#else
#define CLOCK ((double)clock() / CLOCKS_PER_SEC)
#endif

void usage() {
  fprintf(stdout, "Code to perform C/L vector current correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
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
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int source_location, have_source_flag = 0;
  int x0, x1, x2, x3, ix, iix, it;
  int sx0, sx1, sx2, sx3;
  int gx0, gx1, gx2, gx3;
  int isimag, source_timeslice;
  double *conn = NULL;
  double phase[4];
  int verbose = 0;
  char filename[100];
  double ratime, retime;
  int psource[4], source_proc_coords[4];
  double plaq, ssource[4];
  double spinor1[24], spinor2[24], U_[18], q[4];
  complex w, w1;
  FILE *ofs;
  unsigned int VOL3;
  int status;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p;
#else
  fftwnd_plan plan_p;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

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

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [cvc_lvc_2pt_conn] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[cvc_lvc_2pt_conn] T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stderr, "[cvc_lvc_2pt_conn] Error, kappa should be > 0.n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize fftw */
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;

  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

  if(T==0) {
    fprintf(stderr, "[%2d] local T is zero; exit\n", g_cart_id);
    EXIT(2);
  }

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    EXIT(1);
  }

  geometry();

  VOL3 = LX * LY * LZ;

  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(strcmp(gaugefilename_prefix, "identity") == 0) {
    unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [cvc_lvc_2pt_conn] reading gauge field from file %s\n", filename);
    status = read_lime_gauge_field_doubleprec(filename);
    if(status != 0) {
      fprintf(stderr, "[] Error, could not read gauge field from file %s\n", filename);
      EXIT(125);
    }
  }
#ifdef MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [cvc_lvc_2pt_conn] measured plaquette value: %25.16e\n", plaq);

  /* allocate memory for the spinor fields */
  no_fields = 8;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  conn = (double*)calloc(2 * T, sizeof(double));
  if( conn == NULL ) {
    fprintf(stderr, "[cvc_lvc_2pt_conn] could not allocate memory for contr. fields\n");
    EXIT(3);
  }
  memset(conn, 0, 2*T*sizeof(double));

  /* determine source coordinates, find out, if source_location is in this process */
  gx0 =   g_source_location / (LX_global * LY_global * LZ_global);
  gx1 = ( g_source_location % (LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gx2 = ( g_source_location % (LY_global * LZ_global ) ) / LZ_global;
  gx3 = ( g_source_location % LZ_global);
  if(g_cart_id == 0) fprintf(stdout, "# [cvc_lvc_2pt_conn] global source coordinates: (%3d,%3d,%3d,%3d)\n", gx0, gx1, gx2, gx3);
#ifdef MPI
  source_proc_coords[0] = gx0 / T;
  source_proc_coords[1] = gx1 / LX;
  source_proc_coords[2] = gx2 / LY;
  source_proc_coords[3] = gx3 / LZ;
  MPI_Cart_rank(g_cart_grid, source_proc_coords, &have_source_flag);
  have_source_flag == have_source_flag == g_cart_id;
#else
  have_source_flag = 1;
#endif
  sx0 = gx0 % T;
  sx1 = gx1 % LX;
  sx2 = gx2 % LY;
  sx3 = gx3 % LZ;
  if(have_source_flag==1) { 
    fprintf(stdout, "# [cvc_lvc_2pt_conn] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
  }
  source_timeslice = gx0;

  /**********************************************
   * loop on colour index
   **********************************************/
  ratime = CLOCK;
  for(ia=0; ia<3; ia++) {
  
    // read the 4 spinor components 
    for(ib=0; ib<4; ib++) {
      get_filename(filename, 4, ib*3+ia, 1);
      status = read_lime_spinor(g_spinor_field[ib], filename, 0);
      if(status != 0) {
        fprintf(stderr, "[] Error , could not read spinor field from file %s\n", filename);
        EXIT(126);
      }
      xchange_field(g_spinor_field[ib]);

      get_filename(filename, 4, ib*3+ia, -1);
      status = read_lime_spinor(g_spinor_field[4+ib], filename, 0);
      if(status != 0) {
        fprintf(stderr, "[] Error , could not read spinor field from file %s\n", filename);
        EXIT(127);
      }
      xchange_field(g_spinor_field[4+ib]);
    }

    // loop on right Lorentz index nu 
    for(nu=1; nu<4; nu++) {
      psource[0] = gamma_permutation[nu][ 0] / 6;
      psource[1] = gamma_permutation[nu][ 6] / 6;
      psource[2] = gamma_permutation[nu][12] / 6;
      psource[3] = gamma_permutation[nu][18] / 6;
//      fprintf(stdout, "# [cvc_lvc_2pt_conn] psource = (%d, %d, %d, %d)\n", psource[0], 
//        psource[1], psource[2], psource[3]);
      isimag = gamma_permutation[nu][ 0] % 2;
      // sign from the source gamma matrix; the minus sign
      //   in the lower two lines is the action of gamma_5 
      ssource[0] =  gamma_sign[nu][ 0] * gamma_sign[5][gamma_permutation[nu][ 0]];
      ssource[1] =  gamma_sign[nu][ 6] * gamma_sign[5][gamma_permutation[nu][ 6]];
      ssource[2] =  gamma_sign[nu][12] * gamma_sign[5][gamma_permutation[nu][12]];
      ssource[3] =  gamma_sign[nu][18] * gamma_sign[5][gamma_permutation[nu][18]];

      mu = nu;

      for(it=0; it<T; it++) {
      for(iix=0; iix<VOL3; iix++) {
        ix = it * VOL3 + iix;
        for(ir=0; ir<4; ir++) {
          _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &co_phase_up[mu]);
	  _fv_eq_cm_ti_fv(spinor1, U_, g_spinor_field[ir]+_GSI(g_iup[ix][mu]));
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_mi_eq_fv(spinor2, spinor1);
          _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
          _co_eq_fv_dag_ti_fv(&w, g_spinor_field[4+psource[ir]]+_GSI(ix), spinor1);

          if(!isimag) {
            conn[2*it  ] +=  ssource[ir] * w.re;
            conn[2*it+1] +=  ssource[ir] * w.im;
          } else {
            conn[2*it  ] +=  ssource[ir] * w.im;
            conn[2*it+1] += -ssource[ir] * w.re;
          }

        }
      }}
    }  // of nu
  }    // of loop on ia (colour) 

  retime = CLOCK;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_lvc_2pt_conn] contractions in %e seconds\n", retime-ratime);

  /* save results */
  ratime = CLOCK;
  sprintf(filename, "cvc_lvc.t%.2dx%.2dy%.2dz%.2d.%.4d", gx0, gx1, gx2, gx3, Nconf);
  ofs = fopen(filename, "w");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for writing\n", filename);
    EXIT(12);
  }
  it = 0;
  ir = gx0;
  fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 11, 1, it, conn[2*ir], 0., Nconf);
  for(it=1; it<T_global/2; it++) {
    ir  = ( it + gx0            ) % T_global;
    is = (-it + gx0 + T_global ) % T_global;
    fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 11, 1, it, conn[2*ir], conn[2*is], Nconf);
  }
  it = T_global / 2;
  ir = (it + gx0 ) % T_global;
  fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 11, 1, it, conn[2*ir], 0., Nconf);

  fclose(ofs);
  retime = CLOCK;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_lvc_2pt_conn]saved momentum space results in %e seconds\n", retime-ratime);

  /* free the allocated memory, finalize */
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  if(conn != NULL) free(conn);

  fprintf(stdout, "# [cvc_lvc_2pt_conn] %s# [cvc_lvc_2pt_conn] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "[cvc_lvc_2pt_conn] %s[cvc_lvc_2pt_conn] end of run\n", ctime(&g_the_time));
  fflush(stderr);

#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}
