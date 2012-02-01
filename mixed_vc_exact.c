/****************************************************
 * mixed_vc_exact.c
 *
 * Tue Mar 22 19:14:08 CET 2011
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 * - try using only 2x4 fermion fields at once
 * - at sink one only needs to resolve spinor indices
 *   so one can sum over the color index in 3 steps
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
  fprintf(stdout, "Code to perform local vector current correlator conn. contractions\n");
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
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int source_location, have_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int sx0, sx1, sx2, sx3;
  int n_c=1;
  double *conn = (double*)NULL;
  double *conn2 = (double*)NULL;
  double *conn3 = (double*)NULL;
  double A[4][12][24];
  double V[4][12][24];
  double B[4][12][24];
  double phase[4], q[4];
  int verbose = 0;
  int do_gt   = 0;
  char filename[800], contype[800];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_trafo=(double*)NULL;
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

  while ((c = getopt(argc, argv, "h?vgf:p:")) != -1) {
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
    case 'p':
      n_c = atoi(optarg);
      fprintf(stdout, "\n# [] will use number of colors n_c = %d\n", n_c);
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /**************************
   * set the default values *
   **************************/
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");

  /***********************
   * read the input file *
   ***********************/
  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * some checks on the input data *
   *********************************/
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  /*****************************
   * initialize MPI parameters *
   *****************************/
  mpi_init(argc, argv);
#ifdef MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

  /*******************
   * initialize fftw *
   *******************/
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
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  /************************
   * read the gauge field *
   ************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if( !(strcmp(gaugefilename_prefix,"identity")==0) ) {
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [mixed_vc_exact] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
  /* xchange_gauge(); */

  /*************************
   * measure the plaquette *
   *************************/
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

  /*****************************************
   * allocate memory for the spinor fields *
   *****************************************/
  no_fields = 8;  // 2(up, dn) x 4 (source spinor index)
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  conn  = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  conn2 = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  conn3 = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  if( (conn==(double*)NULL) || (conn2==(double*)NULL) || (conn3==NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] = 0.;
  for(ix=0; ix<32*VOLUME; ix++) conn2[ix] = 0.;
  for(ix=0; ix<32*VOLUME; ix++) conn3[ix] = 0.;

  /*****************************************
   * prepare Fourier transformation arrays *
   *****************************************/
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  /*********************************************************************************
   * determine source coordinates, find out, if source_location is in this process *
   *********************************************************************************/
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  if(have_source_flag==1) { 
    fprintf(stdout, "local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
  }
#ifdef MPI
  MPI_Gather(&have_source_flag, 1, MPI_INT, status, 1, MPI_INT, 0, g_cart_grid);
  if(g_cart_id==0) {
    for(mu=0; mu<g_nproc; mu++) fprintf(stdout, "status[%1d]=%d\n", mu,status[mu]);
  }
  if(g_cart_id==0) {
    for(have_source_flag=0; status[have_source_flag]!=1; have_source_flag++);
    fprintf(stdout, "have_source_flag= %d\n", have_source_flag);
  }
  MPI_Bcast(&have_source_flag, 1, MPI_INT, 0, g_cart_grid);
  fprintf(stdout, "[%2d] have_source_flag = %d\n", g_cart_id, have_source_flag);
#else
  have_source_flag = 0;
#endif

  /********************************
   * contractions
   ********************************/

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  for(ia=0; ia<n_c; ia++) {

    /* read 4 up-type propagators */
    for(i=0; i<4; i++) {
      get_filename(filename, 4, 3*i+ia, 1);
      read_lime_spinor(g_spinor_field[i], filename, 0);
      xchange_field(g_spinor_field[i]); 
    }

    /* read 4 dn-type propagators */
    for(i=0; i<4; i++) {
      get_filename(filename, 4, 3*i+ia, -1);
      read_lime_spinor(g_spinor_field[4+i], filename, 0);
      xchange_field(g_spinor_field[4+i]); 
    }

    for(ix=0; ix<VOLUME; ix++) {

      for(mu=0; mu<4; mu++) {
        for(ir=0; ir<12; ir++) {
        for(is=0; is<24; is++) {
          V[mu][ir][is] = 0.;
          A[mu][ir][is] = 0.;
          B[mu][ir][is] = 0.;
        }}
      }
      

      for(ir=0; ir<4; ir++) {
      for(is=0; is<4; is++) {
     
        for(mu=0; mu<4; mu++) {
          // contrib to lvc
          _fv_eq_gamma_ti_fv(spinor1, mu, &g_spinor_field[ir][_GSI(ix)]);
	  _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[4+is][_GSI(ix)], spinor2);
	  V[mu][3*ir+ia][2*(3*is+ia)  ] = -w.re;
	  V[mu][3*ir+ia][2*(3*is+ia)+1] = -w.im;

          // contrib to mixed vc
          _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

          _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[ir][_GSI(g_iup[ix][mu])]);
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_mi_eq_fv(spinor2, spinor1);
          _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[4+is][_GSI(ix)], spinor1);
          A[mu][3*ir+ia][2*(3*is+ia)  ] = -0.5 * w.re;
          A[mu][3*ir+ia][2*(3*is+ia)+1] = -0.5 * w.im;

          _fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[ir][_GSI(ix)]);
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _fv_pl_eq_fv(spinor2, spinor1);
          _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[4+is][_GSI(g_iup[ix][mu])], spinor1);
          A[mu][3*ir+ia][2*(3*is+ia)  ] -= 0.5 * w.re;
          A[mu][3*ir+ia][2*(3*is+ia)+1] -= 0.5 * w.im;

          // contrib to mixed avc
          _fv_eq_cm_ti_fv(spinor1, U_, &g_spinor_field[ir][_GSI(g_iup[ix][mu])]);
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[4+is][_GSI(ix)], spinor2);
          B[mu][3*ir+ia][2*(3*is+ia)  ] = -0.5 * w.re;
          B[mu][3*ir+ia][2*(3*is+ia)+1] = -0.5 * w.im;

          _fv_eq_cm_dag_ti_fv(spinor1, U_, &g_spinor_field[ir][_GSI(ix)]);
          _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[4+is][_GSI(g_iup[ix][mu])], spinor2);
          B[mu][3*ir+ia][2*(3*is+ia)  ] -= 0.5 * w.re;
          B[mu][3*ir+ia][2*(3*is+ia)+1] -= 0.5 * w.im;
        }/* of mu */

      } /* of is */
      } /* of ir */

      for(nu=0; nu<4; nu++) {
     
        /* take the trace of product gamma_nu V */
        for(mu=0; mu<4; mu++) {
          _co_eq_tr_gamma_sm(&w,6+nu,V[mu]);
          conn2[_GWI(4*mu+nu, ix, VOLUME)  ] += w.re;
          conn2[_GWI(4*mu+nu, ix, VOLUME)+1] += w.im;
        }
        for(mu=0; mu<4; mu++) {
          _co_eq_tr_gamma_sm(&w,6+nu,A[mu]);
          conn[_GWI(4*mu+nu, ix, VOLUME)  ] += w.re;
          conn[_GWI(4*mu+nu, ix, VOLUME)+1] += w.im;
        }
        for(mu=0; mu<4; mu++) {
          _co_eq_tr_gamma_sm(&w,nu,B[mu]);
          conn3[_GWI(4*mu+nu, ix, VOLUME)  ] += w.re;
          conn3[_GWI(4*mu+nu, ix, VOLUME)+1] += w.im;
        }
      }

    }  // of ix (lattice site index)

  }  // of ia = 0,...,n_c-1 (source color index)

#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "contractions in %e seconds\n", retime-ratime);

  if(do_gt==0) free(gauge_trafo);

  /****************
   * save results *
   ****************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  sprintf(filename, "ll_v_x.%.4d", Nconf);
  sprintf(contype, "lvc-lvc in position space");
  write_lime_contraction(conn2, filename, 64, 16, contype, Nconf, 0);

  sprintf(filename, "lc_v_x.%.4d", Nconf);
  sprintf(contype, "lvc-cvc in position space");
  write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0);

  sprintf(filename, "lc_v_x.%.4d.ascii", Nconf);
  write_contraction(conn, NULL, filename, 16, 2, 0);

  sprintf(filename, "aa_v_x.%.4d", Nconf);
  sprintf(contype, "avc-avc in position space");
  write_lime_contraction(conn3, filename, 64, 16, contype, Nconf, 0);
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved position space results in %e seconds\n", retime-ratime);

  /**********************************************
   * Check the Ward identity in position space
   **********************************************/

  fprintf(stdout, "\n# [] Check the Ward identity in position space ...\n");
  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(stdout, "# t=%2d x=%2d y=%2d z=%2d\n", x0, x1, x2, x3);
    ix=g_ipt[x0][x1][x2][x3];
    for(nu=0; nu<4; nu++) {
      w.re = conn[_GWI(4*0+nu,ix,VOLUME)] + conn[_GWI(4*1+nu,ix,VOLUME)]
           + conn[_GWI(4*2+nu,ix,VOLUME)] + conn[_GWI(4*3+nu,ix,VOLUME)]
	   - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)]
	   - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)];

      w.im = conn[_GWI(4*0+nu,ix,VOLUME)+1] + conn[_GWI(4*1+nu,ix,VOLUME)+1]
           + conn[_GWI(4*2+nu,ix,VOLUME)+1] + conn[_GWI(4*3+nu,ix,VOLUME)+1]
	   - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)+1] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)+1]
	   - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)+1] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)+1];
      
      fprintf(stdout, "%3d%25.16e%25.16e\n", nu, w.re, w.im);

    }
  }}}}

  /**************************
   * Fourier transformation *
   **************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)&conn2[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&conn2[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)&conn[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&conn[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)&conn3[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
    fftwnd_one(plan_p, in, NULL);
#endif
    memcpy((void*)&conn3[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
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

      _co_eq_co_ti_co(&w1,(complex*)&conn2[_GWI(4*mu+nu,ix,VOLUME)],&w);
      conn2[_GWI(4*mu+nu,ix,VOLUME)  ] = w1.re;
      conn2[_GWI(4*mu+nu,ix,VOLUME)+1] = w1.im;

      _co_eq_co_ti_co(&w1,(complex*)&conn2[_GWI(4*mu+nu,ix,VOLUME)],&w);
      conn3[_GWI(4*mu+nu,ix,VOLUME)  ] = w1.re;
      conn3[_GWI(4*mu+nu,ix,VOLUME)+1] = w1.im;
    }}  /* of mu and nu */
  }}}}

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "Fourier transform in %e seconds\n", retime-ratime);


  /****************************************
   * Check Ward identity in momentum space
   ****************************************/

  fprintf(stdout, "\n# [] Check Ward identity in momentum space ...\n");
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * sin( (double)(Tstart+x0) * M_PI / (double)T_global );
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin( (double)(x1) * M_PI / (double)LX );
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * sin( (double)(x2) * M_PI / (double)LY );
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * sin( (double)(x3) * M_PI / (double)LZ );
    fprintf(stdout, "# t=%2d x=%2d y=%2d z=%2d\n", x0, x1, x2, x3);
    ix = g_ipt[x0][x1][x2][x3];
    for(nu=0; nu<4; nu++) {
      w.re = q[0] * conn[_GWI(4*0+nu,ix,VOLUME)  ] \
           + q[1] * conn[_GWI(4*1+nu,ix,VOLUME)  ] \
           + q[2] * conn[_GWI(4*2+nu,ix,VOLUME)  ] \
           + q[3] * conn[_GWI(4*3+nu,ix,VOLUME)  ];

      w.im = q[0] * conn[_GWI(4*0+nu,ix,VOLUME)+1] \
           + q[1] * conn[_GWI(4*1+nu,ix,VOLUME)+1] \
           + q[2] * conn[_GWI(4*2+nu,ix,VOLUME)+1] \
           + q[3] * conn[_GWI(4*3+nu,ix,VOLUME)+1];
      fprintf(stdout, "%3d%25.16e%25.16e\n", nu, w.re, w.im);
    }
  }}}}

  /*******************************
   * save momentum space results *
   *******************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  sprintf(filename, "ll_p.%.4d", Nconf);
  sprintf(contype, "lvc-lvc in momentum space");
  write_lime_contraction(conn2, filename, 64, 16, contype, Nconf, 0);

  sprintf(filename, "lc_p.%.4d", Nconf);
  sprintf(contype, "lvc-cvc in momentum space");
  write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0);

  sprintf(filename, "lc_v_p.%.4d.ascii", Nconf);
  write_contraction(conn, NULL, filename, 16, 2, 0);

  sprintf(filename, "aa_p.%.4d", Nconf);
  sprintf(contype, "avc-avc in momentum space");
  write_lime_contraction(conn3, filename, 64, 16, contype, Nconf, 0);

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved momentum space results in %e seconds\n", retime-ratime);

  /* free the allocated memory, finalize */
  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(conn);
  free(conn2);
  free(conn3);
#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
#endif

  return(0);

}
