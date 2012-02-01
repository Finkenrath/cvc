/****************************************************
 * check_avc_disc.c
 *
 * Fri Aug 14 10:07:01 MEST 2009 
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

void usage(void) {}

int main(int argc, char **argv) {
  
  int c, i, mu, nu;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int sid;
  double *disc  = (double*)NULL;
  double *disc2 = (double*)NULL;
  double *work = (double*)NULL;
  double q[4], fnorm, d[2];
  int verbose = 0;
  int do_gt   = 0;
  char filename[100];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1, *cp1, *cp2, *cp3;
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

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }

  geometry();

  disc  = (double*)calloc( 8*VOLUME, sizeof(double));
  disc2 = (double*)calloc( 8*VOLUME, sizeof(double));
  work  = (double*)calloc(48*VOLUME, sizeof(double));
  if( (disc==(double*)NULL) || (disc2==(double*)NULL) || (work==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for disc/work\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
  for(ix=0; ix<8*VOLUME; ix++) disc[ix] = 0.;
  for(ix=0; ix<8*VOLUME; ix++) disc2[ix] = 0.;

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }


  for(ix=0; ix<VOLUME; ix++) {
    sprintf(filename, "%.4d/outcvc_X.%.4d.0012", ix, Nconf);
    ofs = fopen(filename, "r");
    fprintf(stdout, "reading from file %s\n", filename);
    for(iix=0; iix<VOLUME; iix++) {
      for(mu=0; mu<4; mu++) {
/*        fscanf(ofs, "%lf%lf", work+_GWI(mu,iix,VOLUME), work+_GWI(mu,iix,VOLUME)+1); */
        fscanf(ofs, "%lf%lf", d, d+1);
	disc[_GWI(mu,iix,VOLUME)  ] += d[0];
	disc[_GWI(mu,iix,VOLUME)+1] += d[1];
      }
    }
    fclose(ofs);
/*
    for(mu=0; mu<4; mu++) {
      _co_pl_eq_co((complex*)(disc+_GWI(mu,ix,VOLUME)), (complex*)(work+_GWI(mu,ix,VOLUME)));
      _co_pl_eq_co((complex*)(disc+_GWI(mu,g_idn[ix][mu],VOLUME)), (complex*)(work+_GWI(mu,g_idn[ix][mu],VOLUME)));

    }
*/
  }

  for(ix=0; ix<VOLUME; ix++) {
    sprintf(filename, "%.4d/outavc_X.%.4d.0012", ix, Nconf);
    ofs = fopen(filename, "r");
    fprintf(stdout, "reading from file %s\n", filename);
    for(iix=0; iix<VOLUME; iix++) {
      for(mu=0; mu<4; mu++) {
        fscanf(ofs, "%lf%lf", d, d+1);
	disc2[_GWI(mu,iix,VOLUME)  ] += d[0];
	disc2[_GWI(mu,iix,VOLUME)+1] += d[1];
      }
    }
    fclose(ofs);
  }

  sprintf(filename, "outcvc_X_exact.%.4d", Nconf);
  write_contraction(disc, NULL, filename, 4, 2, 0);

  for(mu=0; mu<4; mu++) {
    memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    fftwnd_one(plan_p, in, NULL);
    memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

    memcpy((void*)in, (void*)(disc+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
  }

  for(mu=0; mu<4; mu++) {
  for(nu=0; nu<4; nu++) {
    cp1 = (complex*)(work+_GWI(mu,0,VOLUME));
    cp2 = (complex*)(work+_GWI(4+nu,0,VOLUME));
    cp3 = (complex*)(work+_GWI(8+4*mu+nu,0,VOLUME));

    for(x0=0; x0<T; x0++) {
      q[0] = (double)(x0+Tstart) / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      q[1] = (double)(x1) / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = (double)(x2) / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = (double)(x3) / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re = cos( M_PI * (q[mu]-q[nu]) );
      w.im = sin( M_PI * (q[mu]-q[nu]) );
      _co_eq_co_ti_co(&w1, cp1, cp2)
      _co_eq_co_ti_co(cp3, &w1, &w);
      cp1++; cp2++; cp3++;
    }
    }
    }
    }
  }
  }
  sprintf(filename, "outcvc_P_exact.%.4d", Nconf);
  write_contraction(work+_GWI(8,0,VOLUME), NULL, filename, 16, 2, 0);

  for(ix=0; ix<16*VOLUME; ix++) work[ix] = 0.;

  /* check the Ward Identity */
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * sin(M_PI * (double)(x0+Tstart) / (double)T_global);
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin(M_PI * (double)(x1) / (double)LX);
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * sin(M_PI * (double)(x2) / (double)LY);
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * sin(M_PI * (double)(x3) / (double)LZ);
    ix = g_ipt[x0][x1][x2][x3];
    for(nu=0; nu<4; nu++) {
      work[_GWI(nu,ix,VOLUME)] += q[0] * work[_GWI(8+4*0+nu,ix,VOLUME)] + q[1] * work[_GWI(8+4*1+nu,ix,VOLUME)] 
                                + q[2] * work[_GWI(8+4*2+nu,ix,VOLUME)] + q[3] * work[_GWI(8+4*3+nu,ix,VOLUME)];
  
      work[_GWI(nu,ix,VOLUME)+1] += q[0] * work[_GWI(8+4*0+nu,ix,VOLUME)+1] + q[1] * work[_GWI(8+4*1+nu,ix,VOLUME)+1] 
                                  + q[2] * work[_GWI(8+4*2+nu,ix,VOLUME)+1] + q[3] * work[_GWI(8+4*3+nu,ix,VOLUME)+1];
  
      work[_GWI(4+nu,ix,VOLUME)] += q[0] * work[_GWI(8+4*nu+0,ix,VOLUME)] + q[1] * work[_GWI(8+4*nu+1,ix,VOLUME)] 
                                  + q[2] * work[_GWI(8+4*nu+2,ix,VOLUME)] + q[3] * work[_GWI(8+4*nu+3,ix,VOLUME)];
  
      work[_GWI(4+nu,ix,VOLUME)+1] += q[0] * work[_GWI(8+4*nu+0,ix,VOLUME)+1] + q[1] * work[_GWI(8+4*nu+1,ix,VOLUME)+1] 
                                    + q[2] * work[_GWI(8+4*nu+2,ix,VOLUME)+1] + q[3] * work[_GWI(8+4*nu+3,ix,VOLUME)+1];
    }
  }
  }
  }
  }
  sprintf(filename, "WI_P_exact.%.4d", Nconf);
  write_contraction(work, NULL, filename, 8, 2, 0);

  for(mu=0; mu<4; mu++) {
    memcpy((void*)in, (void*)(disc2+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    fftwnd_one(plan_p, in, NULL);
    memcpy((void*)(work+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

    memcpy((void*)in, (void*)(disc2+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    fftwnd_one(plan_m, in, NULL);
    memcpy((void*)(work+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
  }

  for(mu=0; mu<4; mu++) {
  for(nu=0; nu<4; nu++) {
    cp1 = (complex*)(work+_GWI(mu,0,VOLUME));
    cp2 = (complex*)(work+_GWI(4+nu,0,VOLUME));
    cp3 = (complex*)(work+_GWI(8+4*mu+nu,0,VOLUME));

    for(x0=0; x0<T; x0++) {
      q[0] = (double)(x0+Tstart) / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      q[1] = (double)(x1) / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = (double)(x2) / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = (double)(x3) / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re = cos( M_PI * (q[mu]-q[nu]) );
      w.im = sin( M_PI * (q[mu]-q[nu]) );
      _co_eq_co_ti_co(&w1, cp1, cp2)
      _co_eq_co_ti_co(cp3, &w1, &w);
      cp1++; cp2++; cp3++;
    }
    }
    }
    }
  }
  }
  sprintf(filename, "outavc_P_exact.%.4d", Nconf);
  write_contraction(work+_GWI(8,0,VOLUME), NULL, filename, 16, 2, 0);

  for(ix=0; ix<16*VOLUME; ix++) work[ix] = 0.;

  /* check the Ward Identity */
  for(x0=0; x0<T; x0++) {
    q[0] = 2. * sin(M_PI * (double)(x0+Tstart) / (double)T_global);
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin(M_PI * (double)(x1) / (double)LX);
  for(x2=0; x2<LY; x2++) {
    q[2] = 2. * sin(M_PI * (double)(x2) / (double)LY);
  for(x3=0; x3<LZ; x3++) {
    q[3] = 2. * sin(M_PI * (double)(x3) / (double)LZ);
    ix = g_ipt[x0][x1][x2][x3];
    for(nu=0; nu<4; nu++) {
      work[_GWI(nu,ix,VOLUME)] += q[0] * work[_GWI(8+4*0+nu,ix,VOLUME)] + q[1] * work[_GWI(8+4*1+nu,ix,VOLUME)] 
                                + q[2] * work[_GWI(8+4*2+nu,ix,VOLUME)] + q[3] * work[_GWI(8+4*3+nu,ix,VOLUME)];
  
      work[_GWI(nu,ix,VOLUME)+1] += q[0] * work[_GWI(8+4*0+nu,ix,VOLUME)+1] + q[1] * work[_GWI(8+4*1+nu,ix,VOLUME)+1] 
                                  + q[2] * work[_GWI(8+4*2+nu,ix,VOLUME)+1] + q[3] * work[_GWI(8+4*3+nu,ix,VOLUME)+1];
  
      work[_GWI(4+nu,ix,VOLUME)] += q[0] * work[_GWI(8+4*nu+0,ix,VOLUME)] + q[1] * work[_GWI(8+4*nu+1,ix,VOLUME)] 
                                  + q[2] * work[_GWI(8+4*nu+2,ix,VOLUME)] + q[3] * work[_GWI(8+4*nu+3,ix,VOLUME)];
  
      work[_GWI(4+nu,ix,VOLUME)+1] += q[0] * work[_GWI(8+4*nu+0,ix,VOLUME)+1] + q[1] * work[_GWI(8+4*nu+1,ix,VOLUME)+1] 
                                    + q[2] * work[_GWI(8+4*nu+2,ix,VOLUME)+1] + q[3] * work[_GWI(8+4*nu+3,ix,VOLUME)+1];
    }
  }
  }
  }
  }
  sprintf(filename, "avc_WI_P_exact.%.4d", Nconf);
  write_contraction(work, NULL, filename, 8, 2, 0);

  /* free the allocated memory, finalize */
  free_geometry();
  fftw_free(in);
  free(disc);
  free(disc2);
  free(work);
#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  fftwnd_mpi_destroy_plan(plan_m);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);
#endif

  return(0);

}
