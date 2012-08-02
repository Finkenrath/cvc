/****************************************************
  
 * cvc_lvc_conn.c
 *
 * Do 2. Aug 18:13:59 CEST 2012
 *
 * PURPOSE:
 * - contraction for cvc - lvc : conserved / local vector current
 *   at sink / source
 * DONE:
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
  int x0, x1, x2, x3, ix;
  int sx0, sx1, sx2, sx3;
  int isimag;
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

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p;
  int *status;
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
  fprintf(stdout, "# Reading input from file %s\n", filename);
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
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

  /* initialize fftw */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
#endif
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

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
    EXIT(1);
  }

  geometry();

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);
#ifdef MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /* allocate memory for the spinor fields */
  no_fields = 8;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  conn = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  if( conn == NULL ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    EXIT(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] = 0.;

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
    EXIT(4);
  }

  /* determine source coordinates, find out, if source_location is in this process */
  sx0 =   g_source_location / (LX_global * LY_global * LZ_global);
  sx1 = ( g_source_location % (LX_global * LY_global * LZ_gloval)) / (LY_global * LZ_global);
  sx2 = ( g_source_location % (LY_global * LZ_global ) ) / LZ_global;
  sx3 = ( g_source_location % LZ_global);
  if(g_cart_id == 0) fprintf(stdout, "# [cvc_lvc_conn] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
#ifdef MPI
  source_proc_coords[0] = sx0 / T;
  source_proc_coords[1] = sx1 / LX;
  source_proc_coords[2] = sx2 / LY;
  source_proc_coords[3] = sx3 / LZ;
  MPI_Cart_rank(g_cart_grid, source_proc_coords, &have_source_flag);
  have_source_flag == have_source_flag == g_cart_id;
  sx0 = sx0 % T;
  sx1 = sx1 % LX;
  sx2 = sx2 % LY;
  sx3 = sx3 % LZ;
#else
  have_source_flag = 1;
#endif
  if(have_source_flag==1) { 
    fprintf(stdout, "# [cvc_lvc_conn] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
  }

  /**********************************************
   * loop on colour index
   **********************************************/
  ratime = CLOCK;
  for(ia=0; ia<3; ia++) {
  
    // read the 4 spinor components 
    for(ib=0; ib<4; ib++) {
      get_filename(filename, 4, ib*3+ia, 1);
      read_lime_spinor(g_spinor_field[ib], filename, 0);
      xchange_field(g_spinor_field[ib]);

      get_filename(filename, 4, ib*3+ia, -1);
      read_lime_spinor(g_spinor_field[4+ib], filename, 0);
      xchange_field(g_spinor_field[4+ib]);
    }

    // loop on right Lorentz index nu 
    for(nu=0; nu<4; nu++) {
      psource[0] = gamma_permutation[nu][ 0] / 6;
      psource[1] = gamma_permutation[nu][ 6] / 6;
      psource[2] = gamma_permutation[nu][12] / 6;
      psource[3] = gamma_permutation[nu][18] / 6;
//      fprintf(stdout, "# [cvc_lvc_conn] psource = (%d, %d, %d, %d)\n", psource[0], 
//        psource[1], psource[2], psource[3]);
      isimag = gamma_permutation[nu][ 0] % 2;
      // sign from the source gamma matrix; the minus sign
       * in the lower two lines is the action of gamma_5 
      ssource[0] =  gamma_sign[nu][ 0] * gamma_sign[5][gamma_permutation[nu][ 0]];
      ssource[1] =  gamma_sign[nu][ 6] * gamma_sign[5][gamma_permutation[nu][ 6]];
      ssource[2] =  gamma_sign[nu][12] * gamma_sign[5][gamma_permutation[nu][12]];
      ssource[3] =  gamma_sign[nu][18] * gamma_sign[5][gamma_permutation[nu][18]];

      for(mu=0; mu<4; mu++) {

        for(ix=0; ix<VOLUME; ix++) {
          for(ir=0; ir<4; ir++) {
            _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &co_phase_up[mu]);
	    _fv_eq_cm_ti_fv(spinor1, U_, g_spinor_field[ir]+_GSI(g_iup[ix][mu]));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
            _fv_mi_eq_fv(spinor2, spinor1);
            _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
            _co_eq_fv_dag_ti_fv(&w, g_spinor_field[4+psource[ir]]+_GSI(ix), spinor1);

            if(!isimag) {
              conn[_GWI(4*mu+nu, ix, VOLUME)  ] +=  ssource[ir] * w.re;
            } else {
              conn[_GWI(4*mu+nu, ix, VOLUME)+1] += -ssource[ir] * w.re;
            }
          }
        }

      }
    }
  }  // of loop on ia (colour) 

  retime = CLOCK;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_lvc_conn] contractions in %e seconds\n", retime-ratime);

  /* save results */
  ratime = CLOCK;
  sprintf(filename, "cvc_lvc_x.%.4d", Nconf);
  write_contraction(conn, (int*)NULL, filename, 16, 0, 0);
  retime = CLOCK;
  if(g_cart_id==0) fprintf(stdout, "saved position space results in %e seconds\n", retime-ratime);

#ifndef MPI
  /* check the Ward identity in position space */
  w.re = 0.; w.im = 0.;
  sprintf(filename, "cvc_lvc_WI_x.%.4d", Nconf);
  ofs = fopen(filename, "w");
  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(ofs, "# %3d%3d%3d%3d\n", x0, x1, x2, x3);
    ix=g_ipt[x0][x1][x2][x3];
    for(nu=0; nu<4; nu++) {
      w.re = conn[_GWI(4*0+nu,ix,VOLUME)] + conn[_GWI(4*1+nu,ix,VOLUME)]
           + conn[_GWI(4*2+nu,ix,VOLUME)] + conn[_GWI(4*3+nu,ix,VOLUME)]
	   - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)]
	   - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)];
      fprintf(ofs, "%3d%25.16e%25.16e\n", nu, w.re, w.im);
    }
  }}}}
  fclose(ofs);
#endif

  /* Fourier transformation */
  ratime = CLOCK;
  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)&conn[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&conn[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

  /* add phase factors */
  for(x0=0; x0<T; x0++) {
    phase[0] = (double)(Tstart+x0) * M_PI / (double)T_global;
  for(x1=0; x1<LX; x1++) {
    phase[1] = (double)(x1) * M_PI / (double)LX;
  for(x2=0; x2<LY; x2++) {
    phase[2] = (double)(x2) * M_PI / (double)LY;
  for(x3=0; x3<LZ; x3++) {
    phase[3] = (double)(x3) * M_PI / (double)LZ;
    ix = g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      w.re = cos( phase[mu]-phase[nu]-2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      w.im = sin( phase[mu]-phase[nu]-2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );

      _co_eq_co_ti_co(&w1,(complex*)&conn[_GWI(4*mu+nu,ix,VOLUME)],&w);
      conn[_GWI(4*mu+nu,ix,VOLUME)  ] = w1.re;
      conn[_GWI(4*mu+nu,ix,VOLUME)+1] = w1.im;
    }  /* of nu */
    }  /* of mu */
  }}}} 
  retime = CLOCK;
  if(g_cart_id==0) fprintf(stdout, "Fourier transform in %e seconds\n", retime-ratime);

#ifndef MPI
  sprintf(filename, "cvc_lvc_WI_p.%.4d", Nconf);
  ofs = fopen(filename, "w");
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * sin( M_PI * (double)x0 / (double)T );
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin( M_PI * (double)x1 / (double)LX );
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * sin( M_PI * (double)x2 / (double)LY );
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * sin( M_PI * (double)x3 / (double)LZ );
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "# qt=%.2d, qx=%.2d, qy=%.2d, qz=%.2d\n", x0, x1, x2, x3);
    for(mu=0; mu<4; mu++) {
      w.re = q[0] * conn[_GWI(0*4+mu,ix,VOLUME)  ] + q[1] * conn[_GWI(1*4+mu,ix,VOLUME)  ]
           + q[2] * conn[_GWI(2*4+mu,ix,VOLUME)  ] + q[3] * conn[_GWI(3*4+mu,ix,VOLUME)  ];
      w.im = q[0] * conn[_GWI(0*4+mu,ix,VOLUME)+1] + q[1] * conn[_GWI(1*4+mu,ix,VOLUME)+1]
           + q[2] * conn[_GWI(2*4+mu,ix,VOLUME)+1] + q[3] * conn[_GWI(3*4+mu,ix,VOLUME)+1];
      fprintf(ofs, "%3d%25.16e%25.16e\n", mu, w.re, w.im);
    }
  }}}}
  fclose(ofs);
#endif

  /* save momentum space results */
  ratime = CLOCK;
  sprintf(filename, "cvc_lvc_p.%.4d", Nconf);
  write_contraction(conn, (int*)NULL, filename, 16, 0, 0);
  sprintf(filename, "cvc_lvc_p.%.4d.ascii", Nconf);
  write_contraction(conn, (int*)NULL, filename, 16, 2, 0);
  retime = CLOCK;
  if(g_cart_id==0) fprintf(stdout, "saved momentum space results in %e seconds\n", retime-ratime);

  /* free the allocated memory, finalize */
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(conn);

#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  free(status);
#else
  fftwnd_destroy_plan(plan_p);
#endif

  fprintf(stdout, "# [cvc_lvc_conn] %s# [cvc_lvc_conn] end of run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "[cvc_lvc_conn] %s[cvc_lvc_conn] end of run\n", ctime(&g_the_time));
  fflush(stderr);

#ifdef MPI
  MPI_Finalize();
#else
  return(0);
}
