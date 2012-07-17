/****************************************************
 * vivi_v3.c
 *
 * Mon Jul 16 11:18:14 CEST 2012
 *
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
#include "get_index.h"
#include "make_H3orbits.h"
#include "make_x_orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to calculate the vacuum polarization function from momentum/position space data.\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu, nu, status, dims[4], count, i;
  int filename_set = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  int iq;
  int tsrc, xsrc, ysrc, zsrc;
  int tsrc2, xsrc2, ysrc2, zsrc2;
  int do_write_x = 0, do_read = 0;
  int Thm1 = 0;
  double *conn=NULL, *vivi=NULL;
  double **mom2=NULL;
  char filename[800], qval_list_filename[400];
  double ratime, retime;
  double q[4], phase, pimm, pimm2;
  double qval = 0.;
  double y[4], z[4], zsqr;
  FILE *ofs=NULL;
  complex w, w1;

  fftw_complex *in=NULL;

  fftwnd_plan plan_m;

  /***************************************
   * remember: qval = k means
   *   q = 2 pi k /L
   ***************************************/
  
  const int qval_num = 2000;
  double qval_list[qval_num];
  const double qval_min = 0.;
  const double qval_max = 12.;
  for(i=0;i<qval_num;i++) { qval_list[i] = qval_min + (qval_max - qval_min)*(double)i / (double)qval_num; }
  //double qval_list[] = {0., 0.5, 1., 2., 3.};
  //int qval_num = 5;

  while ((c = getopt(argc, argv, "wh?f:q:r:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      do_write_x = 1;
      fprintf(stdout, "\n# [vivi] will write x-data to file\n");
      break;
    case 'r':
      do_read = atoi(optarg);
      fprintf(stdout, "\n# [vivi] will use reading option %d\n", do_read);
      break;
    case 'q':
      strcpy(qval_list_filename, optarg);
      fprintf(stdout, "\n# [vivi] will read external momenta from file%s\n", qval_list_filename);
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
  fprintf(stdout, "\n# [vivi] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "\n[vivi] Error, T and L's must be set\n");
    usage();
  }

  if( LX != LY || LX != LZ ) {
    if(g_proc_id==0) fprintf(stdout, "\n[vivi] Error, need LX = LY = LZ\n");
    usage();
  }

  /******************************************
   * the global time stamp
   ******************************************/
  g_the_time = time(NULL);
  fprintf(stdout, "\n# [vivi] using global time stamp %s", ctime(&g_the_time));


  /******************************************************
   * initialize fftw 
   * create plan with FFTW_FORWARD ---  in contrast to
   *   FFTW_BACKWARD in e.g. avc_exact 
   ******************************************************/
  T = T_global;
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "#       T            = %3d\n"\
                  "#       Tstart       = %3d\n"\
                  "#       l_LX_at      = %3d\n"\
	          "#       l_LXstart_at = %3d\n"\
	          "#       FFTW_LOC_VOLUME = %3d\n", 
      g_cart_id, T, Tstart, l_LX_at, l_LXstart_at, FFTW_LOC_VOLUME);
  Thm1 = T_global / 2 - 1;

  if(init_geometry() != 0) {
    fprintf(stderr, "\n [vivi] ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  vivi = (double*)calloc(2*VOLUME, sizeof(double));
  if( (vivi == NULL) ) {
    fprintf(stderr, "\n[vivi] could not allocate memory for vivi\n");
    EXIT(5);
  }

  conn = (double**)calloc(3*sizeof(double*));
  conn[0] = (double*)calloc(Thm1*sizeof(double));  // e-Kt -1
  conn[1] = (double*)calloc(Thm1*sizeof(double));  // (Kt)^2
  conn[2] = (double*)calloc(Thm1*sizeof(double));  // e-Kt-1 - (Kt)^2  
  if( (conn == NULL) ) {
    fprintf(stderr, "\n[vivi] Error, could not allocate memory for contr. fields\n");
    EXIT(3);
  }

  tsrc = g_source_location/(LX*LY*LZ);
  xsrc = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  ysrc = (g_source_location%(LY*LZ)) / LZ;
  zsrc = (g_source_location%LZ);
  fprintf(stdout, "\n# [vivi] source location: (%d, %d, %d, %d)\n", \
      tsrc, xsrc, ysrc, zsrc);

  tsrc2 = tsrc > T_global/2 ? -T_global + tsrc : tsrc;
  xsrc2 = xsrc > LX/2       ? -LX       + xsrc : xsrc; 
  ysrc2 = ysrc > LY/2       ? -LY       + ysrc : ysrc; 
  zsrc2 = zsrc > LZ/2       ? -LZ       + zsrc : zsrc; 
  fprintf(stdout, "\n# [vivi] modified source location: (%d, %d, %d, %d)\n", \
      tsrc2, xsrc2, ysrc2, zsrc2);

  /***********************
   * read contractions   *
   ***********************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "\n# [vivi] Reading data from file %s\n", filename_prefix);
  //status = read_contraction(conn, NULL, filename_prefix, 16);
  status = read_lime_contraction(conn, filename_prefix, 16, 0);
  if(status == 106) {
    fprintf(stderr, "\n[vivi] Error: could not read from file %s; status was %d\n", 
        filename_prefix, status);
    exit(6);
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "\n# [vivi] time to read contractions: %e seconds\n", retime-ratime);

    /******************************
     * WI check in momentum space
     ******************************/

    ratime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "\n# [vivi] check WI in momentum space\n");
    for(x0=0; x0<T; x0++) {
      q[0] = 2. * sin( M_PI * (double)(x0+Tstart) / (double)T_global );
    for(x1=0; x1<LX; x1++) {
      q[1] = 2. * sin( M_PI * (double)(x1) / (double)LX );
    for(x2=0; x2<LY; x2++) {
      q[2] = 2. * sin( M_PI * (double)(x2) / (double)LY );
    for(x3=0; x3<LZ; x3++) {
      q[3] = 2. * sin( M_PI * (double)(x3) / (double)LZ );
      ix = g_ipt[x0][x1][x2][x3];
      fprintf(stdout, "# WICheck t=%2d, x=%2d, y=%2d, z=%2d\n", x0, x1, x2, x3);
      for(nu=0; nu<4; nu++) {
        w.re = q[0] * conn[_GWI(4*0+nu, ix, VOLUME)  ] +
               q[1] * conn[_GWI(4*1+nu, ix, VOLUME)  ] +
               q[2] * conn[_GWI(4*2+nu, ix, VOLUME)  ] +
               q[3] * conn[_GWI(4*3+nu, ix, VOLUME)  ];

        w.im = q[0] * conn[_GWI(4*0+nu, ix, VOLUME)+1] +
               q[1] * conn[_GWI(4*1+nu, ix, VOLUME)+1] +
               q[2] * conn[_GWI(4*2+nu, ix, VOLUME)+1] +
               q[3] * conn[_GWI(4*3+nu, ix, VOLUME)+1];

        fprintf(stdout, "# WICheck\t %3d%25.16e%25.16e\n", nu, w.re, w.im);
      }
    }}}}
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "\n# [vivi] time to check WI in momentum space: %e seconds\n", retime-ratime);

    /******************************
     * backward Fourier transform
     ******************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    for(x0=0; x0<T; x0++) {
      q[0] = M_PI * (double)(x0+Tstart) / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      q[1] = M_PI * (double)(x1) / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = M_PI * (double)(x2) / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = M_PI * (double)(x3) / (double)LZ;
      phase = 2. * ( q[0]*tsrc + q[1]*xsrc + q[2]*ysrc + q[3]*zsrc );
      ix = g_ipt[x0][x1][x2][x3];
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        w.re =  cos( q[mu] - q[nu] - phase );
        w.im = -sin( q[mu] - q[nu] - phase );
        _co_eq_co_ti_co(&w1, (complex*)(conn+_GWI(4*mu+nu, ix, VOLUME)), &w);
        conn[_GWI(4*mu+nu, ix, VOLUME)  ] = w1.re;
        conn[_GWI(4*mu+nu, ix, VOLUME)+1] = w1.im;
      }}
    }}}}
/*
    for(mu=0; mu<4; mu++) {
      memcpy((void*)in, (void*)(conn+_GWI(5*mu,0,VOLUME)), 2*VOLUME*sizeof(double));
      fftwnd_one(plan_m, in, NULL);
      memcpy((void*)(vivi + _GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
    }
*/
    for(mu=0; mu<16; mu++) {
      memcpy((void*)in, (void*)(conn+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
      fftwnd_one(plan_m, in, NULL);
      memcpy((void*)(conn + _GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
    }
    for(mu=0; mu<4; mu++) {
      memcpy((void*)(vivi + _GWI(mu,0,VOLUME)), (void*)(conn+_GWI(5*mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "\n# [vivi] time for backward Fourier transform: %e seconds\n", retime-ratime);


    if(do_write_x) {
      size_t items = 2*VOLUME;
      sprintf(filename, "vivi.%.4d", Nconf);
      ofs = fopen(filename, "w");
      if(ofs==NULL) {
        fprintf(stderr, "\n# [vivi] Error, could not open file %s for writing\n", filename);
        exit(120);
      }
      for(mu=0;mu<4;mu++) {
        if( fwrite(conn+_GWI(5*mu,0,VOLUME), sizeof(double), items, ofs) != items ) {
          fprintf(stderr, "\n# [vivi] Error, could not write proper amount of data to file %s\n", filename);
          exit(121);
        }
      }
      fclose(ofs);
    }
  } else if(do_read==1){  // of do_read
    size_t items = 8*VOLUME;
    sprintf(filename, "vivi.%.4d", Nconf);
    ofs = fopen(filename, "r");
    if(ofs==NULL) {
      fprintf(stderr, "\n# [vivi] Error, could not open file %s for reading\n", filename);
      exit(120);
    }
    fprintf(stdout, "\n# [vivi] reading vivi data from file %s\n", filename);
    if( fread(vivi, sizeof(double), items, ofs) != items ) {
      fprintf(stderr, "\n# [vivi] Error, could not read proper amount of data to file %s\n", filename);
      exit(121);
    }
    fclose(ofs);
  } else if (do_read==2) {
    strcpy(filename, filename_prefix);
    fprintf(stdout, "\n# [vivi] reading vivi data from file %s\n", filename);
    status = read_lime_contraction(conn, filename, 16, 0);
    if( status != 0) {
      fprintf(stderr, "\n# [vivi] Error, could not read proper amount of data to file %s\n", filename);
      exit(122);
    }
    for(mu=0; mu<4; mu++) {
      memcpy((void*)(vivi + _GWI(mu,0,VOLUME)), (void*)(conn+_GWI(5*mu,0,VOLUME)), 2*VOLUME*sizeof(double));
    }
  }  // end of if do_read

  /****************************************
   * check Ward identity in position space
   ****************************************/
/*
  ratime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "\n# [vivi] check WI in position space\n");
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(stdout, "# WICheck t=%2d, x=%2d, y=%2d, z=%2d\n", x0, x1, x2, x3);
    for(nu=0; nu<4; nu++) {
      w.re =  conn[_GWI(0*4+nu, ix, VOLUME)  ]           
            + conn[_GWI(1*4+nu, ix, VOLUME)  ] 
            + conn[_GWI(2*4+nu, ix, VOLUME)  ] 
            + conn[_GWI(3*4+nu, ix, VOLUME)  ] 
            - conn[_GWI(0*4+nu, g_idn[ix][0], VOLUME)  ] 
            - conn[_GWI(1*4+nu, g_idn[ix][1], VOLUME)  ] 
            - conn[_GWI(2*4+nu, g_idn[ix][2], VOLUME)  ] 
            - conn[_GWI(3*4+nu, g_idn[ix][3], VOLUME)  ];

      w.im =  conn[_GWI(0*4+nu, ix, VOLUME)+1]           
            + conn[_GWI(1*4+nu, ix, VOLUME)+1] 
            + conn[_GWI(2*4+nu, ix, VOLUME)+1]
            + conn[_GWI(3*4+nu, ix, VOLUME)+1] 
            - conn[_GWI(0*4+nu, g_idn[ix][0], VOLUME)+1] 
            - conn[_GWI(1*4+nu, g_idn[ix][1], VOLUME)+1]
            - conn[_GWI(2*4+nu, g_idn[ix][2], VOLUME)+1] 
            - conn[_GWI(3*4+nu, g_idn[ix][3], VOLUME)+1];

      fprintf(stdout, "# WICheck\t %3d%25.16e%25.16e\n", nu, w.re, w.im);
    }
  }}}}
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "\n# [vivi] time for checking Ward identity: %e seconds\n", retime-ratime);
*/
/*
  fprintf(stdout, "\n# [vivi] write position space correlator\n");
  for(x0=0; x0<T; x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(stdout, "# t=%2d, x=%2d, y=%2d, z=%2d\n", x0, x1, x2, x3);
    for(mu=0; mu<16; mu++) {
      fprintf(stdout, "%3d%25.16e%25.16e\n", mu, conn[_GWI(mu, ix, VOLUME)]/(double)VOLUME, conn[_GWI(mu, ix, VOLUME)+1]/(double)VOLUME);
    }
  }}}}
*/

  /*****************************************
   * q-dep. 2nd temporal moment
   *****************************************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  int Thm1=T/2-1;
  double dtmp;
  mom2 = (double**)malloc(qval_num*sizeof(double*));
  if(mom2 == NULL) {
    fprintf(stderr, "\n[vivi] Error, could not alloc mom2\n");
    exit(15);
  }
  mom2[0] = (double*)malloc(Thm1*qval_num*sizeof(double));
  if(mom2[0] == NULL) {
    fprintf(stderr, "\n[vivi] Error, could not alloc mom2[0]\n");
    exit(16);
  }
  for(iq=1;iq<qval_num; iq++) mom2[iq] = mom2[iq-1] + Thm1;
  for(x0=0; x0<Thm1*qval_num; x0++) mom2[0][x0] = 0.;

  for(iq=0;iq<qval_num;iq++) {
    //fprintf(stdout, "# x0\ty0\ty1\tpi+\tpi-\n");
    qval = qval_list[iq];
    phase = qval * 2. * M_PI / (double)LX;
    fprintf(stdout, "\n# [vivi] qval%.2d using external momentum = 2 pi %e / L\n", iq, qval);
    for(x0=1; x0<T/2; x0++) {
      y0 = (tsrc + x0    ) % T;
      y1 = (tsrc - x0 + T) % T;
      pimm  = 0.;
      pimm2 = 0.;
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix  = g_ipt[y0][x1][x2][x3];
        iix = g_ipt[y1][x1][x2][x3];

        pimm  += vivi[_GWI(1, ix, VOLUME)  ]*cos( phase * (double)(x1 - xsrc) ) \
               - vivi[_GWI(1, ix, VOLUME)+1]*sin( phase * (double)(x1 - xsrc) ) \
               + vivi[_GWI(2, ix, VOLUME)  ]*cos( phase * (double)(x2 - ysrc) ) \
               - vivi[_GWI(2, ix, VOLUME)+1]*sin( phase * (double)(x2 - ysrc) ) \
               + vivi[_GWI(3, ix, VOLUME)  ]*cos( phase * (double)(x3 - zsrc) ) \
               - vivi[_GWI(3, ix, VOLUME)+1]*sin( phase * (double)(x3 - zsrc) );

/*  
        pimm  += vivi[_GWI(1, ix, VOLUME)  ]*cos( phase * (double)(x1 - xsrc) ) \
               + vivi[_GWI(2, ix, VOLUME)  ]*cos( phase * (double)(x2 - ysrc) ) \
               + vivi[_GWI(3, ix, VOLUME)  ]*cos( phase * (double)(x3 - zsrc) );
*/

        pimm2 += vivi[_GWI(1, iix, VOLUME)  ]*cos( phase * (double)(x1 - xsrc) ) \
               - vivi[_GWI(1, iix, VOLUME)+1]*sin( phase * (double)(x1 - xsrc) ) \
               + vivi[_GWI(2, iix, VOLUME)  ]*cos( phase * (double)(x2 - ysrc) ) \
               - vivi[_GWI(2, iix, VOLUME)+1]*sin( phase * (double)(x2 - ysrc) ) \
               + vivi[_GWI(3, iix, VOLUME)  ]*cos( phase * (double)(x3 - zsrc) ) \
               - vivi[_GWI(3, iix, VOLUME)+1]*sin( phase * (double)(x3 - zsrc) );

/*        
        pimm2 += vivi[_GWI(1, iix, VOLUME)  ]*cos( phase * (double)(x1 - xsrc) ) \
               + vivi[_GWI(2, iix, VOLUME)  ]*cos( phase * (double)(x2 - ysrc) ) \
               + vivi[_GWI(3, iix, VOLUME)  ]*cos( phase * (double)(x3 - zsrc) );
*/
      }}}
      fprintf(stdout, "%d\t%d\t%d\t%e\t%e\n", x0, y0, y1, pimm, pimm2);
      dtmp = (pimm + pimm2 ) * (double)(x0 * x0);
      if(x0==1) {
        mom2[iq][0] = dtmp;
      } else {
        mom2[iq][x0-1] = mom2[iq][x0-2] + dtmp;
      }
    }  // of x0 = 1,...,T/2-1
  }  // of iq = 0,...,qval_num

  /* normalization */
  for(i=0;i<Thm1*qval_num;i++) mom2[0][i] /= 6.*(double)(LX*LY*LZ*T_global);

  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "\n# [vivi] time to fill correlator: %e seconds\n", retime-ratime);

  /*****************************************
   * write
   *****************************************/
  sprintf(filename, "vivi.%.4d.log", Nconf);
  ofs = fopen(filename, "w");
  if(ofs == NULL) {
    fprintf(stderr, "\n[vivi] Error, could not open file %s for writing\n", filename);
    exit(117);
  }
  fprintf(ofs, "# results for q-dep. 2nd temporal moment:\n");
  fprintf(ofs, "# %s", ctime(&g_the_time));
  for(iq=0;iq<qval_num;iq++) {
    for(i=0;i<Thm1;i++) { fprintf(ofs, "%6d%16.7e%3d%25.16e\n", iq, qval_list[iq], i+1, mom2[iq][i]); }
  }
  fclose(ofs);

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  //finalize_x_orbits(&h4_id, &h4_count, &h4_val, &h4_rep);
  free_geometry();
  if(vivi !=NULL) free(vivi);
  if(conn !=NULL) free(conn);
  if(in   !=NULL) free(in);
  if(mom2 !=NULL) {
    if(mom2[0] != NULL) free(mom2[0]);
    free(mom2);
  }
  if(do_read==0) {
    fftwnd_destroy_plan(plan_m);
  }

  fflush(stdout);
  fprintf(stdout, "\n# [vivi] %s# end of run", ctime(&g_the_time));
  fflush(stderr);
  fprintf(stderr, "\n[vivi] %send of run", ctime(&g_the_time));


  return(0);
}
