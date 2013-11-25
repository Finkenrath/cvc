/****************************************************
 * contract_disc_tqdep_v2.c
 *
 * Mon May  7 08:35:59 EEST 2012
 *
 * PURPOSE
 * - calculate loops psibar Gamma psi from timeslice ( / volume ?)
 *   sources; save the (t,q1,q2,q3)- dependend fields
 * - like disc_tqdep, but with optional momentum list and lhpc-aff library
 *
 * - REMEMBER: FFTW_FORWARD  ---> exp( - i phase)
 * - REMEMBER: FFTW_BACKWARD ---> exp( + i phase)
 *
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#ifdef OPENMP
#include <omp.h>
#endif
#include "ifftw.h"


#include "../cvc_complex.h"
#include "../cvc_linalg.h"
#include "../global.h"
#include "../cvc_geometry.h"
#include "../cvc_utils.h"
#include "../mpi_init.h"
#include "../io.h"
#include "../Q_phi.h"
#include "../smearing_techniques.h"
#include "../contractions_io.h"
#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif
#include "contract_disc_tqdep_v2.h"


int contract_disc_tqdep_v2(double *src, double *prop, momentum_info_type *momentum_info, int isample) {

  const char *outfile_prefix = "disc";
  const int K = 20; 
  int c, i, mu, count,imom, threadid;
  int filename_set = 0;
  int status;
  int l_LX_at, l_LXstart_at;
  int it, x0, x1, ix, idx, x2, x3, iy;
  int VOL3;
  int dims[3];
  double *disc = (double*)NULL;
  int verbose = 0;
  int fermion_type = 1;
  char filename[100], contype[500];
  double ratime, retime;
  double **spinor1=NULL, **spinor2=NULL;
  double _2kappamu;
  double *gauge_field_smeared=NULL;
  complex w, w1, w2;
  double q[3], phase, U_[18];
  FILE *ofs;
  int *timeslice_list=NULL, *have_timeslice_list=NULL;
/*  double sign_adj5[] = {-1., -1., -1., -1., +1., +1., +1., +1., +1., +1., -1., -1., -1., 1., -1., -1.}; */
  double fnorm;
  int num_timeslices=0 ;
  size_t bytes, items;
  int read_source = 0;
  char line[200];
  double *send_buffer = NULL;
#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * status_str;
  double *buffer = NULL;
  char buffer_path[200];
  uint32_t buffer_size;
#endif

/***********************************************************/
  int *qlatt_id=NULL, *qlatt_count=NULL, **qlatt_rep=NULL, **qlatt_map=NULL, qlatt_nclass=0;
  double **qlatt_list=NULL;
/***********************************************************/

/***********************************************************************************************/            
/*                    g5  gi           g0g5 g0gi         id  gig5        g0  g[igj]            */
 // int gindex[]    = { 5 , 1 , 2 , 3 ,  6 ,  10 ,11 ,12 , 4 , 7 , 8 , 9 , 0 , 15 , 14 ,13 };
 // int isimag[]    = { 0 , 0 , 0 , 0 ,  1 ,   1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 ,  1 ,  1 , 1 };
 // double gsign[]  = {-1., 1., 1., 1., -1.,   1., 1., 1., 1., 1., 1., 1., 1.,  1., -1., 1.};
  int  gindex[]  = { 0, 1, 2, 3, 4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  int isimag[]   = { 0, 0, 0, 0, 0,  0,  1, 1, 1, 1,  1,  1,  1,  1,  1,  1 };
  double gsign[] = { 1, 1, 1, 1, 1,  1,  1, 1, 1, 1,  1,  1,  1,  1, -1,  1 };
/***********************************************************************************************/            

  fftwnd_plan plan_p;
  fftw_complex *in = NULL;

  VOL3    = LX*LY*LZ;
  FFTW_LOC_VOLUME = LX*LY*LZ;

  // allocate memory for the contractions
  if(g_source_type == 2) {
    // timeslice source
    if(g_coherent_source == 0) {
      have_timeslice_list    = (int*)malloc(sizeof(int));
      have_timeslice_list[0] = g_source_timeslice / T == g_proc_coords[0] ? 1 : 0;
      timeslice_list         = (int*)malloc(sizeof(int));
      timeslice_list[0]      = g_source_timeslice % T;
    } else if(g_coherent_source == 1) {
      num_timeslices      =  T_global / g_coherent_source_delta;
      timeslice_list      = (int*)malloc(num_timeslices*sizeof(int));
      have_timeslice_list = (int*)malloc(num_timeslices*sizeof(int));
      for(it=0;it<num_timeslices; it++) {
        x0 = (g_coherent_source_base+it*g_coherent_source_delta) % T_global;
        timeslice_list[it] = x0 % T;
        have_timeslice_list[it] = ( x0 / T == g_proc_coords[0]);
      }
    }
  } else if(g_source_type == 1 ) {
    // volume source
    num_timeslices = T;
    timeslice_list = (int*)malloc(T*sizeof(int));
    for(it=0;it<T;it++) {
      timeslice_list[it] = it;
      have_timeslice_list[it] = 1;
    }
  }
  if(g_cart_id==0) {
    fprintf(stdout, "# [contract_disc_tqdep_v2] number of timeslices = %d\n", num_timeslices);
  }
  for(it=0;it<num_timeslices;it++) {
    fprintf(stdout, "\t %3d%3d%3d%3d\n", g_cart_id, it, timeslice_list[it], have_timeslice_list[it]);
  }

  items =   2*K*num_timeslices*VOL3;  // 2[complex] x (16+1)[Gamma_mu +  cvc] x num_timeslices x VOL3
  bytes = sizeof(double);
  disc = (double*)calloc(items, bytes);
  if( disc==NULL ) {
    EXIT_WITH_MSG(3, "[contract_disc_tqdep_v2] Error, could not allocate memory for disc\n");
  }
  memset(disc, 0, items*sizeof(double));

  /************************************
   * initialize FFTW
   ************************************/
  dims[0] = LX;
  dims[1] = LY;
  dims[2] = LZ;
  in = (fftw_complex*)malloc(K * VOL3 * sizeof(fftw_complex));
  if(in == NULL) {
    EXIT_WITH_MSG(5, "[] Error, could not alloc in for FFTW\n");
  }
  // plan_p = fftwnd_create_plan_specific(3, dims, FFTW_FORWARD, FFTW_MEASURE, in, K, (fftw_complex*)( disc ), K);
  plan_p = fftwnd_create_plan_specific(3, dims, FFTW_BACKWARD, FFTW_MEASURE, in, K, (fftw_complex*)( disc ), K);

  // set the global source timeslice if it is a coherent source
  if(g_coherent_source == 1) {
    g_source_timeslice = g_coherent_source_base;
    if(g_cart_id==0) fprintf(stdout, "# [contract_disc_tqdep_v2] Warning: reset source timeslice to %d\n", g_source_timeslice);
  }

  /********************************************
   * create spinor1, spinor2
   ********************************************/
  spinor1 = (double**)malloc(g_num_threads*sizeof(double*));
  spinor1[0] = (double*)malloc(g_num_threads*24*sizeof(double));
  for(mu=1;mu<g_num_threads;mu++) spinor1[mu] = spinor1[mu-1] + 24;

  spinor2 = (double**)malloc(g_num_threads*sizeof(double*));
  spinor2[0] = (double*)malloc(g_num_threads*24*sizeof(double));
  for(mu=1;mu<g_num_threads;mu++) spinor2[mu] = spinor2[mu-1] + 24;

  /**************************************************************
   * normalization
   **************************************************************/
  fnorm = 1. / g_prop_normsqr;
  if(g_cart_id==0) fprintf(stdout, "# [contract_disc_tqdep_v2] using normalization with fnorm = %e\n", fnorm);

  // fill disc with new contractions
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  for(mu=0; mu<16; mu++) {  // loop on index of gamma matrix
    for(it=0; it<num_timeslices; it++) {       // loop on timeslices
      x0 = timeslice_list[it];
      if(have_timeslice_list[it]) fprintf(stdout, "# [contract_disc_tqdep_v2] proc%.2d processing mu-timeslice pair%3d%3d/%3d\n", g_cart_id, mu, x0,
          x0+g_proc_coords[0]*T);
      if(!have_timeslice_list[it]) continue;
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(threadid, x1, ix, iy, w) firstprivate(fnorm) shared(mu, it, x0, spinor1, spinor2)
{
  threadid = omp_get_thread_num();
#else
  threadid=0;
#endif
      for(x1=threadid; x1<VOL3; x1+=g_num_threads) {  // loop on sites in timeslice
        ix = x0*VOL3 + x1;
        iy = it*VOL3 + x1;
        _fv_eq_gamma_ti_fv(spinor1[threadid], mu, &g_spinor_field[1][_GSI(ix)]);
        _co_eq_fv_dag_ti_fv(&w, &g_spinor_field[0][_GSI(ix)], spinor1[threadid]);
        disc[2 * ( (it*K + mu)*VOL3 + x1 )  ] = w.re * fnorm;
        disc[2 * ( (it*K + mu)*VOL3 + x1 )+1] = w.im * fnorm;
      }  // of x1 = 0, ..., VOL3
#ifdef OPENMP
}
#endif
    }  // of x0
  }    // of mu 
#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [contract_disc_tqdep_v2] contractions in %e seconds\n", retime-ratime);

  
    /***********************************************
     * Fourier transform
     ***********************************************/
    items = 2*K*VOL3;
    bytes = sizeof(double);
    for(it=0;it<num_timeslices;it++) {
      memcpy(in, disc + it*items, items*bytes);
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);

      fftwnd_threads(g_num_threads, plan_p, K, in, 1, VOL3, (fftw_complex*)(disc + it*items), 1, VOL3);
#else
      fftwnd(plan_p, K, in, 1, VOL3, (fftw_complex*)(disc + it*items), 1, VOL3);
#endif
    }
    free(in);
  
    /***********************************************
     * write current disc to file
     ***********************************************/
  if(g_source_type==2) {
    // timeslice source
    items = 2*num_timeslices * K * momentum_info->number;
    send_buffer = (double*)malloc(items*sizeof(double));
    if(momentum_info->number == VOL3) {
      memcpy(send_buffer, disc, items*sizeof(double));
    } else {
      for(it=0;it<num_timeslices;it++) {
        for(mu=0;mu<K;mu++) {
          for(imom=0;imom<momentum_info->number;imom++) {
            ix = g_ipt[0][momentum_info->list[imom][0]][momentum_info->list[imom][1]][momentum_info->list[imom][2]];
            send_buffer[2*( (it*K + mu) * momentum_info->number+imom)  ] = disc[2*( (it*K + mu) * VOL3+ix)  ];
            send_buffer[2*( (it*K + mu) * momentum_info->number+imom)+1] = disc[2*( (it*K + mu) * VOL3+ix)+1];
          }
        }
      }
    }
    memset(disc, 0, 2*num_timeslices * K * VOL3 * sizeof(double));
#ifdef MPI
    MPI_Reduce(send_buffer, disc, items, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
    memcpy(disc, send_buffer, items*sizeof(double));
#endif
    if(g_cart_id==0) {
#ifndef HAVE_LHPC_AFF
      for(it=0;it<num_timeslices; it++) {
        if(g_coherent_source==1) { x0 = (g_coherent_source_base + it*g_coherent_source_delta ) % T_global; }
        else                     { x0 = ( g_source_timeslice + it ) % T_global; }
        sprintf(filename, "%s_tq.%.4d.%.2d.%.5d", outfile_prefix, Nconf, x0, isample);
        if(momentum_info->number == VOL3) {
          sprintf(contype, "quark-disconnected contractions; 16 Gamma structures, conserved vector current (20 types in total); timeslice no. %d;", x0);
          write_lime_contraction_3d(disc+it*2*K*momentum_info->number, filename, 64, K, contype, Nconf, g_nsample);
        } else {
          if( (ofs = fopen(filename, "w")) == NULL ) {
            fprintf(stderr, "[contract_disc_tqdep_v2] Error, could not open file %s for writing; exit\n", filename);
            return(102);
          }
          for(imom=0;imom<momentum_info->number;imom++) {
            for(mu=0;mu<K;mu++) {
              fprintf(ofs, "%3d%25.16e%25.16e%3d%3d%3d\n", mu,
                  disc[2*( (it*K + mu) * momentum_info->number + imom)  ], disc[2*( (it*K + mu) * momentum_info->number + imom)+1],
                  momentum_info->list[imom][0], momentum_info->list[imom][1], momentum_info->list[imom][2]);
            }
          }
          fclose(ofs);
        }
      }
#else
      buffer = (double*)malloc(2*num_timeslices*sizeof(double));
      if(buffer== NULL) return(101);
  
      status_str = aff_version();
      fprintf(stdout, "# [contract_disc_tqdep_v2] using aff version %s\n", status_str);
  
      sprintf(filename, "%s_tq_v2.%.4d.%.5d", outfile_prefix, Nconf, isample);
      affw = aff_writer(filename);
      status_str = aff_writer_errstr(affw);
      if( status_str != NULL ) {
        fprintf(stderr, "[contract_disc_tqdep_v2] Error from aff_writer, status was %s\n", status_str);
        return(102);
      }
  
      if( (affn = aff_writer_root(affw)) == NULL ) {
        fprintf(stderr, "[contract_disc_tqdep_v2] Error, aff writer is not initialized\n");
        return(103);
      }
      strcpy(buffer_path, "loops");
      if( (affdir = aff_writer_mkdir (affw, affn, buffer_path)) == NULL ) {
        fprintf(stderr, "[contract_disc_tqdep_v2] Error, could not make /loops key\n");
        return(104);
      }
  
      if(g_coherent_source==1) { x0 = g_coherent_source_base; }
      else                     { x0 = g_source_timeslice; }
      buffer_size = (uint32_t)num_timeslices;
  
      for(mu=0;mu<K;mu++) {
        for(imom=0;imom<momentum_info->number;imom++) {
          for(it=0;it<num_timeslices;it++) {
            buffer[2*it  ] = disc[2*( (it*K + mu) * momentum_info->number + imom)  ];
            buffer[2*it+1] = disc[2*( (it*K + mu) * momentum_info->number + imom)+1];
          }
          sprintf(buffer_path, "gamma%.2d/px%.2dpy%.2dpz%.2d/base%.2d", mu,
              momentum_info->list[imom][0], momentum_info->list[imom][1], momentum_info->list[imom][2], x0);
  
          if( ( affn = aff_writer_mkpath (affw, affdir, buffer_path)) == NULL ) {
            fprintf(stderr, "[contract_disc_tqdep_v2] Error, could not make buffer path %s\n", buffer_path);
            return(105);
          }
          status = aff_node_put_complex (affw, affn, buffer, buffer_size);
          if(status != 0) {
            fprintf(stderr, "[contract_disc_tqdep_v2] Error, could not write buffer, status was %d\n", status);
            return(106);
          }
        }
      }
  
      status_str = aff_writer_close(affw);
      if( status_str != NULL ) {
        fprintf(stderr, "[contract_disc_tqdep_v2] Error from aff_writer_close, status was %s\n", status_str);
        return(103);
      }
#endif
    }  // of if g_cart_id == 0
  } else if(g_source_type==1) {
    // volume source
    EXIT_WITH_MSG(107, "[contract_disc_tqdep_v2] Warning, writing for volume sources not yet implemented\n");
  }

  /********************************************************
   * free the allocated memory, finalize
   ********************************************************/
  if(disc!=NULL) free(disc);
  if(timeslice_list!=NULL) free(timeslice_list);
  if(have_timeslice_list!=NULL) free(have_timeslice_list);
  if(send_buffer != NULL) free(send_buffer);
  if(spinor1!=NULL) {
    if(spinor1[0]!=NULL) free(spinor1[0]);
    free(spinor1);
  }
  if(spinor2!=NULL) {
    if(spinor2[0]!=NULL) free(spinor2[0]);
    free(spinor2);
  }

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [contract_disc_tqdep_v2] %s# [contract_disc_tqdep_v2] end of contract_disc_tqdep_v2\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [contract_disc_tqdep_v2] %s# [contract_disc_tqdep_v2] end of contract_disc_tqdep_v2\n", ctime(&g_the_time));
  }

  return(0);

}
