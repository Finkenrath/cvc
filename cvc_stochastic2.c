/****************************************************
 * cvc_stochastic2.c
 *
 * Tue Jul 21 17:17:55 MEST 2009
 *
 * TODO: 
 * - include the calculation of the connected part
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

void usage() {
  fprintf(stdout, "Code to perform light neutral contractions\n");
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
  
  int c, i, mu, nu;
  int count=0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix;
  int sid, sid2;
  double *disc      = (double*)NULL;
  double *work      = (double*)NULL;
  double *conn      = (double*)NULL;
  double contact_term[8], buffer[8];
  double q[4];
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
  int *status;
#else
  fftwnd_plan plan_p, plan_m;
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

  /* initialize fftw */
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  plan_m = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_FORWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD,  FFTW_MEASURE | FFTW_IN_PLACE);
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

  /* allocate geometry fields */
  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  /* initialize geometry fields */
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
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  disc = (double*)calloc(20*VOLUME, sizeof(double));
  conn = (double*)calloc(20*VOLUME, sizeof(double));
  work = (double*)calloc(32*VOLUME, sizeof(double));
  if( (disc==(double*)NULL) || conn==(double*)NULL || (work==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for disc/conn/work\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<20*VOLUME; ix++) disc[ix] = 0.;
  for(ix=0; ix<20*VOLUME; ix++) conn[ix] = 0.;
  for(ix=0; ix<8; ix++) contact_term[ix] = 0.;

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

/*
  if(g_resume==1) { 
    sprintf(filename, "outcvc_conn.%.4d", Nconf);
    c = read_contraction(conn, &count, filename, 10);
    if(c==0) {
      sprintf(filename, ".outcvc_disc_current.%.4d", Nconf);
      c = read_contraction(disc, (int*)NULL, filename, 10);
    }
#ifdef MPI
    MPI_Gather(&c, 1, MPI_INT, status, 1, MPI_INT, 0, g_cart_grid);
    if(g_cart_id==0) {
      for(i=0; i<g_nproc; i++) 
        if(status[i]!=0) { status[0] = 1; break; }
    }
    MPI_Bcast(status, 1, MPI_INT, 0, g_cart_grid);
    if(status[0]==1) {
      for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
      count = 0;
    }
#else
    if(c != 0) {
      fprintf(stdout, "could not read current disc; start new\n");
      for(ix=0; ix<20*VOLUME; ix++) disc[ix] = 0.;
      for(ix=0; ix<20*VOLUME; ix++) conn[ix] = 0.;
      count = 0;
    }
#endif
    if(g_cart_id==0) fprintf(stdout, "starting with count = %d\n", count);
  }  
*/

  /**********************************************
   * Phase I:
   *
   * - calculate phi_mu, xi, zeta_mu, mu=0,1,2,3
   **********************************************/

#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  /* start loop on source id.s */
  for(sid=g_sourceid; sid<=g_sourceid2; sid++) {

    /* read the new propagator */
    sprintf(filename, "%s.%.4d.%.2d", filename_prefix, Nconf, sid);
/*    sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid); */
    if(format==0) {
      if(read_lime_spinor(g_spinor_field[1], filename, 0) != 0) break;
    }
    else if(format==1) {
      if(read_cmi(g_spinor_field[1], filename) != 0) break;
    }
    
    xchange_field(g_spinor_field[1]); 

    /* calculate phi_mu */
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
	_fv_eq_cm_ti_fv(spinor1,U_,&g_spinor_field[1][_GSI(g_iup[ix][mu])]);
	_fv_eq_gamma_ti_fv(spinor2,mu,spinor1);
	_fv_eq_fv_mi_fv(&g_spinor_field[0][_GSI(ix)],spinor2,spinor1);
      }
      sprintf(filename, "phi.%1d.%.5d.%.2d", mu, sid, g_cart_id);
      if((ofs=fopen(filename, "w"))==(FILE*)NULL) return(1);
      fwrite(g_spinor_field[0], sizeof(double), 24*VOLUME, ofs);
      fclose(ofs);
    }
    sprintf(filename, "Phi.%.5d.%.2d", sid, g_cart_id);
    if((ofs=fopen(filename, "w"))==(FILE*)NULL) return(2);
    fwrite(g_spinor_field[1], sizeof(double), 24*VOLUME, ofs);
    fclose(ofs);

    /* calculate the source: apply Q_phi_tbc */
    Q_phi_tbc(g_spinor_field[0], g_spinor_field[1]);
    xchange_field(g_spinor_field[0]); 

    sprintf(filename, "xi.%.5d.%.2d", sid, g_cart_id);
    if((ofs=fopen(filename, "w"))==(FILE*)NULL) return(3);
    fwrite(g_spinor_field[0], sizeof(double), 24*VOLUME, ofs);
    fclose(ofs);

    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        _fv_eq_cm_ti_fv(spinor1,U_,&g_spinor_field[0][_GSI(g_iup[ix][mu])]);
        _fv_eq_gamma_ti_fv(spinor2,mu,spinor1);
        _fv_eq_fv_pl_fv(&g_spinor_field[1][_GSI(ix)],spinor2,spinor1);
      }
      sprintf(filename, "zeta.%1d.%.5d.%.2d", mu, sid, g_cart_id);
      if((ofs=fopen(filename, "w"))==(FILE*)NULL) return(4);
      fwrite(g_spinor_field[1], sizeof(double), 24*VOLUME, ofs);
      fclose(ofs);
    }
  } /* of sid */
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  fprintf(stdout, "[%2d] calculation of phi/xi/zeta in %e seconds\n", 
    g_cart_id, retime-ratime);

  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free(g_gauge_field);

  /**********************************************
   * Phase II:
   *
   * - build the correlations in Fourier space
   * - PROBLEM: N^2 Fourier transformations
   *            required
   **********************************************/

  no_fields=20;
  g_spinor_field = (double**)calloc(20, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);

#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  for(sid=g_sourceid; sid<g_sourceid2; sid++) {

    /* read the Phi/phi/xi/zeta for sid */
    sprintf(filename, "Phi.%.5d.%.2d", sid, g_cart_id);
    if((ofs=fopen(filename, "r"))==(FILE*)NULL) return(5);
    fread(g_spinor_field[4], sizeof(double), 24*VOLUME, ofs);
    fclose(ofs);
    for(mu=0; mu<4; mu++) {
      sprintf(filename, "phi.%1d.%.5d.%.2d", mu, sid, g_cart_id);
      if((ofs=fopen(filename, "r"))==(FILE*)NULL) return(6);
      fread(g_spinor_field[mu], sizeof(double), 24*VOLUME, ofs);
      fclose(ofs);
    }
    sprintf(filename, "xi.%.5d.%.2d", sid, g_cart_id);
    if((ofs=fopen(filename, "r"))==(FILE*)NULL) return(7);
    fread(g_spinor_field[9], sizeof(double), 24*VOLUME, ofs);
    fclose(ofs);
    for(mu=0; mu<4; mu++) {
      sprintf(filename, "zeta.%1d.%.5d.%.2d", mu, sid, g_cart_id);
      if((ofs=fopen(filename, "r"))==(FILE*)NULL) return(8);
      fread(g_spinor_field[5+mu], sizeof(double), 24*VOLUME, ofs);
      fclose(ofs);
    }

    for(sid2=sid+1; sid2<=g_sourceid2; sid2++) {
   
      /* read the Phi/phi/xi/zeta for sid2 */
      sprintf(filename, "Phi.%.5d.%.2d", sid2, g_cart_id);
      if((ofs=fopen(filename, "r"))==(FILE*)NULL) return(9);
      fread(g_spinor_field[14], sizeof(double), 24*VOLUME, ofs);
      fclose(ofs);
      for(mu=0; mu<4; mu++) {
        sprintf(filename, "phi.%1d.%.5d.%.2d", mu, sid2, g_cart_id);
        if((ofs=fopen(filename, "r"))==(FILE*)NULL) return(10);
        fread(g_spinor_field[10+mu], sizeof(double), 24*VOLUME, ofs);
        fclose(ofs);
      }
      sprintf(filename, "xi.%.5d.%.2d", sid2, g_cart_id);
      if((ofs=fopen(filename, "r"))==(FILE*)NULL) return(11);
      fread(g_spinor_field[19], sizeof(double), 24*VOLUME, ofs);
      fclose(ofs);
      for(mu=0; mu<4; mu++) {
        sprintf(filename, "zeta.%1d.%.5d.%.2d", mu, sid2, g_cart_id);
        if((ofs=fopen(filename, "r"))==(FILE*)NULL) return(12);
        fread(g_spinor_field[15+mu], sizeof(double), 24*VOLUME, ofs);
        fclose(ofs);
      }

      /* calculate the correlations */

      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
	  _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[9][_GSI(ix)], &g_spinor_field[10+mu][_GSI(ix)]);
	  _co_eq_fv_dag_ti_fv(&w1,&g_spinor_field[5+mu][_GSI(ix)],&g_spinor_field[14][_GSI(ix)]);
	  work[_GWI(mu,ix,VOLUME)  ] = w.re + w1.re;
	  work[_GWI(mu,ix,VOLUME)+1] = w.im + w1.im;
	}
      }

      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
	  _co_eq_fv_dag_ti_fv(&w,&g_spinor_field[19][_GSI(ix)],&g_spinor_field[mu][_GSI(ix)]);
	  _co_eq_fv_dag_ti_fv(&w1,&g_spinor_field[15+mu][_GSI(ix)],&g_spinor_field[4][_GSI(ix)]);
	  work[_GWI(8+mu,ix,VOLUME)  ] = w.re + w1.re;
	  work[_GWI(8+mu,ix,VOLUME)+1] = w.im + w1.im;
	}
      }

      /* do the Fourier transformations */
      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)&work[_GWI(mu,0,VOLUME)], VOLUME*2*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)&work[_GWI(4+mu,0,VOLUME)], (void*)in, VOLUME*2*sizeof(double));

        memcpy((void*)in, (void*)&work[_GWI(mu,0,VOLUME)], VOLUME*2*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)&work[_GWI(mu,0,VOLUME)], (void*)in, VOLUME*2*sizeof(double));
      }

      for(mu=0; mu<4; mu++) {
        memcpy((void*)in, (void*)&work[_GWI(8+mu,0,VOLUME)], VOLUME*2*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_m, in, NULL);
#endif
        memcpy((void*)&work[_GWI(12+mu,0,VOLUME)], (void*)in, VOLUME*2*sizeof(double));

        memcpy((void*)in, (void*)&work[_GWI(8+mu,0,VOLUME)], VOLUME*2*sizeof(double));
#ifdef MPI
        fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
        fftwnd_one(plan_p, in, NULL);
#endif
        memcpy((void*)&work[_GWI(8+mu,0,VOLUME)], (void*)in, VOLUME*2*sizeof(double));
      }

      /* add the contribution to the connected part */
      for(x0=0; x0<T;  x0++) {
        q[0] = M_PI * (double)(Tstart+x0) / (double)T_global;
      for(x1=0; x1<LX; x1++) {
        q[1] = M_PI * (double)(x1) / (double)LX;
      for(x2=0; x2<LY; x2++) {
        q[2] = M_PI * (double)(x2) / (double)LY;
      for(x3=0; x3<LZ; x3++) {
        q[3] = M_PI * (double)(x3) / (double)LZ;
        ix = g_ipt[x0][x1][x2][x3];
	i = -1;
	for(mu=0; mu<4; mu++) {
	for(nu=mu; nu<4; nu++) {
	  w.re = cos(q[mu]-q[nu]);
	  w.im = sin(q[mu]-q[nu]);
	  i++;
	  _co_eq_co_ti_co(&w1,(complex*)&work[_GWI(mu,ix,VOLUME)],(complex*)&work[_GWI(12+nu,ix,VOLUME)]);
	  _co_pl_eq_co_ti_co((complex*)&conn[_GWI(ix,i,10)],&w1,&w);

	  _co_eq_co_ti_co(&w1,(complex*)&work[_GWI(8+mu,ix,VOLUME)],(complex*)&work[_GWI(4+nu,ix,VOLUME)]);
	  _co_pl_eq_co_ti_co((complex*)&conn[_GWI(ix,i,10)],&w1,&w);
	}
	}
      }
      }
      }
      }

      count += 2;

    } /* of sid2 */

    /* add contribution from sid to counter term */
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[9][_GSI(ix)], &g_spinor_field[mu][_GSI(ix)]);
	contact_term[2*mu  ] += w.re;
	contact_term[2*mu+1] += w.im;
        _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[5+mu][_GSI(ix)], &g_spinor_field[4][_GSI(ix)]);
	contact_term[2*mu  ] -= w.re;
	contact_term[2*mu+1] -= w.im;
      }
    }
    if(sid==g_sourceid2-1) {
      if(g_cart_id==0) fprintf(stdout, "adding contrib. for (sid,sid2) = (%2d.%2d)\n", sid, sid2);
      for(mu=0; mu<4; mu++) {
        for(ix=0; ix<VOLUME; ix++) {
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[19][_GSI(ix)], &g_spinor_field[10+mu][_GSI(ix)]);
	  contact_term[2*mu  ] += w.re;
	  contact_term[2*mu+1] += w.im;
          _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[15+mu][_GSI(ix)], &g_spinor_field[14][_GSI(ix)]);
 	  contact_term[2*mu  ] -= w.re;
	  contact_term[2*mu+1] -= w.im;
        }
      }
    }

  } /* of sid */
#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  fprintf(stdout, "[%2d] time to calculate correlations: %e\n", g_cart_id, retime-ratime);

#ifdef MPI
  MPI_Allreduce(contact_term, buffer, 8, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  memcpy(contact_term, buffer, 8*sizeof(double));
#endif

  for(i=0; i<8; i++) contact_term[i] /= 4. * (double)(g_sourceid2 - g_sourceid + 1) * (double)(T_global*LX*LY*LZ);

  for(ix=0; ix<20*VOLUME; ix++) conn[ix] /= -16. * (double)(g_nproc*VOLUME) * (double)count;

  for(ix=0; ix<VOLUME; ix++) {
    conn[_GWI(ix,0,10)  ] += contact_term[0];
    conn[_GWI(ix,0,10)+1] += contact_term[1];
    conn[_GWI(ix,4,10)  ] += contact_term[2];
    conn[_GWI(ix,4,10)+1] += contact_term[3];
    conn[_GWI(ix,7,10)  ] += contact_term[4];
    conn[_GWI(ix,7,10)+1] += contact_term[5];
    conn[_GWI(ix,9,10)  ] += contact_term[6];
    conn[_GWI(ix,9,10)+1] += contact_term[7];
  }

  /* write current result to file */
  sprintf(filename, "outcvc_conn.%.4d", Nconf);
  write_contraction(conn, &count, filename, 10, 1, 0);

  if(g_cart_id==0) fprintf(stdout, "[%2d] counter term:\n"\
                  "[%2d] 0\t%25.16e\t%25.16e\n"\
                  "[%2d] 1\t%25.16e\t%25.16e\n"\
                  "[%2d] 2\t%25.16e\t%25.16e\n"\
		  "[%2d] 3\t%25.16e\t%25.16e\n",
		  g_cart_id,
		  g_cart_id, contact_term[0], contact_term[1],
		  g_cart_id, contact_term[2], contact_term[3],
		  g_cart_id, contact_term[4], contact_term[5], 
		  g_cart_id, contact_term[6], contact_term[7]);

  /* free the allocated memory, finalize */
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  fftw_free(in);
  free(disc);
  free(work);
  free(conn);
#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  fftwnd_mpi_destroy_plan(plan_m);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);
#endif

  /* remove the auxilliary files */
  for(sid=g_sourceid; sid<=g_sourceid2; sid++) {
    sprintf(filename, "Phi.%.5d.%.2d", sid, g_cart_id);
    remove(filename);
    sprintf(filename, "xi.%.5d.%.2d", sid, g_cart_id);
    remove(filename);
    for(mu=0; mu<4; mu++) {
      sprintf(filename, "phi.%1d.%.5d.%.2d", mu, sid, g_cart_id);
      remove(filename);
      sprintf(filename, "zeta.%1d.%.5d.%.2d", mu, sid, g_cart_id);
      remove(filename);
    }
  }

  return(0);

}
