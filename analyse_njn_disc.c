/****************************************************
 * analyse_njn_disc.c
 *
 * Mon Nov 14 15:14:07 EET 2011
 *
 * PURPOSE
 * DONE:
 * - checked disc_class; checked nucleon_2pt_class
 * TODO:
 * CHANGES:
 * - try version where only real parts are kept
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ifftw.h"
#include <getopt.h>

#define MAIN_PROGRAM

#include "dml.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "get_index.h"
#include "contractions_io.h"
#include "make_q_orbits.h"

void usage() {
  fprintf(stdout, "# [analyse_njn_disc] code to build the gauge configuration-wise quark-disconnected correlators for the investigaton of  < N | j | N > \n");
  fprintf(stdout, "# [analyse_njn_disc] Usage:    [options]\n");
  fprintf(stdout, "# [analyse_njn_disc] Options: -v verbose\n");
  fprintf(stdout, "# [analyse_njn_disc]          -f input filename [default cvc.input]\n");
  fprintf(stdout, "# [analyse_njn_disc]          -j <i> current id i [default 4]\n");
  fprintf(stdout, "# [analyse_njn_disc]          -s use lattice momente [default no]\n");
  fprintf(stdout, "# [analyse_njn_disc]          -p <name> use current momentum list in file name [no default]\n");
  fprintf(stdout, "# [analyse_njn_disc]          -P <name> use nucleon momentum list in file name [no default]\n");
  exit(0);
}


int main(int argc, char **argv) {
 
  const int K = 20;

  int c, mu, status, i, j, it, count;
  int filename_set = 0;
  // int mode = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, iiy, gid, iclass, iclass2, imom;
  int sx0, sx1, sx2, sx3;
  int Thp1, qlatt_nclass, tauf, deltat;
  int k, iy, iz, ir;
  spinor_propagator_type *nucleon_2pt_tq = NULL, sp1=NULL, sp2=NULL;
  double *nucleon_2pt_class=NULL;
  double *disc_tq = NULL, *disc_class=NULL, *disc_class_nophase=NULL;
  double *nucleon_2pt_disc_class = NULL;
  double q[3], qsqr, pinitial[3];
  int verbose = 0;
  char filename[800] , line[200]; 
  double ratime, retime;
  // double p2final, p2initial;
  int current_id=4;
  int isample;
  size_t items, bytes;
  int *qlatt_id=NULL, *qlatt_count=NULL, **qlatt_rep=NULL, **qlatt_map=NULL;
  double **qlatt_list=NULL, q2max=0., phase, fnorm; 
  unsigned int VOL3;
  complex w, w2, w1;
  int use_lattice_momenta = 0;
  DML_Checksum nucleon_checksum;

/***********************************************************/
  int current_momentum_filename_set = 0, current_momentum_no=0;
  int *current_momentum_list=NULL;
  char current_momentum_filename[200];
/***********************************************************/

/***********************************************************/
  int nucleon_momentum_no = 1;
  int *nucleon_momentum_list = NULL;
  int nucleon_momentum_filename_set = 0;
  char nucleon_momentum_filename[200];
/***********************************************************/

/***********************************************************

  const int gamma_proj_no = 1;

  int gamma_proj1[] = {4};
  int gamma_proj2[] = {0};

  int gamma_proj_isimag1[] = {0};
  int gamma_proj_isimag2[] = {0};
  double gamma_proj_sign1[] = {0.25};
  double gamma_proj_sign2[] = {0.25};

  int gamma_proj_fw_bw[] = {3};
  char gamma_proj_string[6];
  int gamma_proj_id, gamma_proj_isimag_id;
 ***********************************************************/

  FILE *ofs;

  while ((c = getopt(argc, argv, "sh?vf:j:p:P:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'j':
      current_id = atoi(optarg);
      fprintf(stdout, "\n# [] gamma id set to %d\n", current_id);
      break;
    case 's':
      use_lattice_momenta = 1;
      fprintf(stdout, "# [] will use lattice momenta\n");
      break;
    case 'p':
      current_momentum_filename_set = 1;
      strcpy(current_momentum_filename, optarg);
      fprintf(stdout, "# [] will use current momentum file %s\n", current_momentum_filename);
      break;
    case 'P':
      nucleon_momentum_filename_set = 1;
      strcpy(nucleon_momentum_filename, optarg);
      fprintf(stdout, "# [] will use nucleon momentum file %s\n", nucleon_momentum_filename);
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // set the default values
  if(filename_set==0) strcpy(filename, "analyse.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  VOL3 = LX*LY*LZ;
  Thp1 = T/2 + 1;

  if(!use_lattice_momenta) {
    status = make_qcont_orbits_3d_parity_avg(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
  } else {
    status = make_qlatt_orbits_3d_parity_avg(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
  } 
  if(status != 0) {
    fprintf(stderr, "\n[] Error while creating h4-lists\n");
    exit(4);
  }
  fprintf(stdout, "# [] number of classes = %d\n", qlatt_nclass);


  /***************************************************************************
   * read the current insertion momenta q to be used
   ***************************************************************************/
  ofs = fopen(current_momentum_filename, "r");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for reading\n", current_momentum_filename);
    exit(6);
  }
  current_momentum_no = 0;
  while( fgets(line, 199, ofs) != NULL) {
    if(line[0] != '#') {
      current_momentum_no++;
    }
  }
  if(current_momentum_no == 0) {
    fprintf(stderr, "[] Error, number of momenta is zero\n");
    exit(7);
  } else {
    fprintf(stdout, "# [] number of current momenta = %d\n", current_momentum_no);
  }
  rewind(ofs);
  current_momentum_list = (int*)malloc(current_momentum_no * sizeof(int));
  count=0;
  while( fgets(line, 199, ofs) != NULL) {
    if(line[0] != '#') {
      sscanf(line, "%d", current_momentum_list+count);
      count++;
    }
  }
  fclose(ofs);
  fprintf(stdout, "# [] current momentum list:\n");
  for(i=0;i<current_momentum_no;i++) {
    fprintf(stdout, "\t%3d%6d\t%3d%3d%3d\n", i, current_momentum_list[i],
        qlatt_rep[current_momentum_list[i]][1], qlatt_rep[current_momentum_list[i]][2],qlatt_rep[current_momentum_list[i]][3]);
  }


  /***************************************************************************
   * read the nucleon final momenta to be used
   ***************************************************************************/
  ofs = fopen(nucleon_momentum_filename, "r");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for reading\n", nucleon_momentum_filename);
    exit(6);
  }
  nucleon_momentum_no = 0;
  while( fgets(line, 199, ofs) != NULL) {
    if(line[0] != '#') {
      nucleon_momentum_no++;
    }
  }
  if(nucleon_momentum_no == 0) {
    fprintf(stderr, "[] Error, number of momenta is zero\n");
    exit(7);
  } else {
    fprintf(stdout, "# [] number of nucleon final momenta = %d\n", nucleon_momentum_no);
  }
  rewind(ofs);
  nucleon_momentum_list = (int*)malloc(nucleon_momentum_no * sizeof(int));
  count=0;
  while( fgets(line, 199, ofs) != NULL) {
    if(line[0] != '#') {
      sscanf(line, "%d", nucleon_momentum_list+count);
      count++;
    }
  }
  fclose(ofs);
  fprintf(stdout, "# [] the nucleon final momentum list:\n");
  for(i=0;i<nucleon_momentum_no;i++) {
    fprintf(stdout, "\t%3d%6d\t%3d%3d%3d\n", i, nucleon_momentum_list[i],
        qlatt_rep[nucleon_momentum_list[i]][1], qlatt_rep[nucleon_momentum_list[i]][2],qlatt_rep[nucleon_momentum_list[i]][3]);
  }

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  // disconnected part, D
  items = VOL3 * K*2;
  bytes = sizeof(double);
  disc_tq = (double*)calloc(items, bytes);
  if( (disc_tq==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    exit(3);
  }

  bytes = sizeof(double);
  items = 2 * current_momentum_no;
  disc_class = (double*) malloc(items*bytes);
  if(disc_class == NULL) {
    fprintf(stdout, "[] Error, could not alloc class fields\n");
    exit(113);
  }
  disc_class_nophase = (double*) malloc(items*bytes);
  if(disc_class_nophase == NULL) {
    fprintf(stdout, "[] Error, could not alloc class fields\n");
    exit(116);
  }

  // nucleon part
  items = (size_t)VOL3;
  nucleon_2pt_tq =  create_sp_field( items );
  if(nucleon_2pt_tq == NULL) {
    fprintf(stderr, "\nError, could not alloc nucleon_2pt_tq\n");
    exit(3);
  }

  items = 2 * T * nucleon_momentum_no;
  bytes = sizeof(double);
  nucleon_2pt_class = (double*) malloc(items*bytes);
  if(nucleon_2pt_class == NULL) {
    fprintf(stderr, "[] Error, could not alloc nucleon class field\n");
    exit(115);
  }

  // product part
  items = 2 * T * current_momentum_no * nucleon_momentum_no;
  bytes = sizeof(double);
  nucleon_2pt_disc_class = (double*)malloc(items * bytes);
  if(nucleon_2pt_disc_class == NULL) {
    fprintf(stdout, "[] Error, could not alloc corr\n");
    exit(114);
  }

  sp1 = create_sp();
  sp2 = create_sp();

  // determine the source location
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  fprintf(stdout, "# [] point source location %d = (%d,%d,%d,%d)\n", g_source_location, sx0, sx1, sx2, sx3);
  fprintf(stdout, "# [] stochastic source timnslice %d\n", g_source_timeslice);
    
  ratime = (double)clock() / CLOCKS_PER_SEC;

  // read the disc data
  sprintf(filename, "%s_tq.%.4d.%.2d.%.5d", filename_prefix2, Nconf, g_source_timeslice, g_nsample);
  fprintf(stdout, "# [] reading disc tq data from file %s\n", filename);
  status = read_lime_contraction_3d(disc_tq, filename, K, 0);
  if(status != 0) {
    fprintf(stderr, "[] Error, could not read data from file %s\n", filename);
    exit(22);
  }

  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to read contractions %e seconds\n", retime-ratime);

  // test: write disc
  //fprintf(stdout, "# [] disc data:\n");
  //for(ix=0;ix<VOL3;ix++) {
  //  fprintf(stdout, "\t%8d%16.7e%16.7e\n", ix, disc_tq[2*(current_id*VOL3+ix)  ], disc_tq[2*(current_id*VOL3+ix)+1]);
  //}

  // sort into 3-momentum classes
  for(i=0;i<2*current_momentum_no;i++) disc_class[i] = 0.;
  for(i=0;i<2*current_momentum_no;i++) disc_class_nophase[i] = 0.;

  // add q-dep. phase factor from source location, average over classes, use only current_id
  for(iclass=0;iclass<current_momentum_no; iclass++) {
    imom = current_momentum_list[iclass];
    for(i=0;i<qlatt_count[imom]; i++) {
      ix = qlatt_map[imom][i] / (LY*LZ);
      iy = ( iz = qlatt_map[imom][i] - ix * LY*LZ ) / LZ;
      iz -= iy * LZ;
      q[0] = (double)(ix) / (double)LX;
      q[1] = (double)(iy) / (double)LY;
      q[2] = (double)(iz) / (double)LZ;

      phase = 2.*M_PI * ( q[0]*sx1 + q[1]*sx2 + q[2]*sx3 );
      // fprintf(stdout, "# [] p=(%d,%d,%d), phase=%f +I %f\n", ix, iy, iz, cos( phase), sin(phase));
      w.re = cos ( phase );
      w.im = sin ( phase );
      w2.re = disc_tq[_GWI(current_id, qlatt_map[imom][i], VOL3)  ];
      w2.im = disc_tq[_GWI(current_id, qlatt_map[imom][i], VOL3)+1];
      _co_eq_co_ti_co(&w1, &w, &w2);
      // try using only real parts
/*
      disc_class[2*(iclass)  ] += w1.re;
      disc_class[2*(iclass)+1] += w1.im;
      disc_class_nophase[2*(iclass)  ] += w2.re;
      disc_class_nophase[2*(iclass)+1] += w2.im;
*/
      disc_class[2*(iclass)  ]         += w1.re;
      disc_class_nophase[2*(iclass)  ] += w2.re;
    }
    // normalization
    disc_class[2*iclass  ] /= qlatt_count[imom];
    disc_class[2*iclass+1] /= qlatt_count[imom];
    disc_class_nophase[2*iclass  ] /= qlatt_count[imom];
    disc_class_nophase[2*iclass+1] /= qlatt_count[imom];
  }

  // test: write the momentum-orbit averages disc data
  //fprintf(stdout, "# [] momentum-orbit averaged current data:\n");
  //for(iclass=0;iclass<current_momentum_no; iclass++) {
  //  fprintf(stdout, "\t%3d%25.16e%25.16e\n", current_momentum_list[iclass], disc_class_nophase[2*iclass], disc_class_nophase[2*iclass+1]);
  //} 

  // initialize the nucleon class array
  items = 2 * T * nucleon_momentum_no;
  for(it=0;it<items;it++) nucleon_2pt_class[it] = 0.;

  /***********************************************************************
   * read the nucleon data timeslice wise, project, add to correlators
   ***********************************************************************/
  for(it=0; it<T; it++) {
    tauf   = ( it - sx0 + T_global ) % T_global;
    deltat = ( it - g_source_timeslice + T_global ) % T_global;
    sprintf(filename, "%s_q.%.4d.t%.2dx%.2dy%.2dz%.2d", filename_prefix, Nconf, sx0, sx1, sx2, sx3);
    //if(it == 0) DML_checksum_init(&nucleon_checksum);
    status = read_lime_contraction_timeslice(nucleon_2pt_tq[0][0], filename, g_sv_dim*g_sv_dim, 0, &nucleon_checksum, it);
    if(status != 0) {
      fprintf(stderr, "[] Error, could not read from file %si for timeslice %d\n", filename, it);
      exit(79);
    }
    //if(it==T-1) {
    //  fprintf(stdout, "# [] final checksum for contractions in file %s at position %d is %#x %#x\n", filename, 0, nucleon_checksum.suma, nucleon_checksum.sumb);
    //}


    // build the correlators
    for(iclass=0;iclass<nucleon_momentum_no; iclass++) {
      imom = nucleon_momentum_list[iclass];

      for(i=0;i<qlatt_count[imom]; i++) {
        // project to nucleon
        _sp_eq_sp(sp1, nucleon_2pt_tq[qlatt_map[imom][i]]);
        _sp_eq_gamma_ti_sp(sp2, 0, sp1);
        if(tauf<Thp1) {
          //fwd
          _sp_pl_eq_sp(sp1, sp2);
          _co_eq_tr_sp(&w, sp1);
          w.re *= +0.25;
          w.im *= +0.25;
        } else {
          //bwd
          _sp_mi_eq_sp(sp1, sp2);
          _co_eq_tr_sp(&w, sp1);
          w.re *= -0.25;
          w.im *= -0.25;
        }

        // try using only real parts
        w.im = 0.;
        nucleon_2pt_class[2*(deltat * nucleon_momentum_no + iclass)  ] += w.re;
        nucleon_2pt_class[2*(deltat * nucleon_momentum_no + iclass)+1] += w.im;

       
        // add to the correlator C_NN D:
        for(iclass2=0; iclass2<current_momentum_no; iclass2++) {
          w1.re = disc_class[2*iclass2  ];
          w1.im = disc_class[2*iclass2+1];
          _co_eq_co_ti_co(&w2, &w, &w1);

          nucleon_2pt_disc_class[2*( (iclass*current_momentum_no + iclass2)*T + deltat)  ] += w2.re;
          nucleon_2pt_disc_class[2*( (iclass*current_momentum_no + iclass2)*T + deltat)+1] += w2.im;

        }


      }  // of loop on orbit members

    }    // of loop on classes


    // normalization of nucleon_2pt_class and nucleon_2pt_disc_class
    for(iclass=0;iclass<nucleon_momentum_no; iclass++) {
      imom = nucleon_momentum_list[iclass];
      fnorm = 1. / (double)qlatt_count[imom];
      nucleon_2pt_class[2*(deltat * nucleon_momentum_no + iclass)  ] *= fnorm;
      nucleon_2pt_class[2*(deltat * nucleon_momentum_no + iclass)+1] *= fnorm;
     
      for(iclass2=0; iclass2<current_momentum_no; iclass2++) {
        // this was wrong: disc_class has been normalized already above
        // fnorm = 1. / ( (double)qlatt_count[imom] * (double)qlatt_count[current_momentum_list[iclass2]] );
        fnorm = 1. / ( (double)qlatt_count[imom] );
        nucleon_2pt_disc_class[2*( (iclass*current_momentum_no + iclass2)*T+deltat)  ] *= fnorm;
        nucleon_2pt_disc_class[2*( (iclass*current_momentum_no + iclass2)*T+deltat)+1] *= fnorm;
      }
    }

  }  // of loop on timeslices


  // write to file
  sprintf(filename, "%s.%.4d.%.2d.t%.2dx%.2dy%.2dz%.2d.%.2d.%.5d", g_outfile_prefix, Nconf, current_id, sx0, sx1, sx2, sx3, g_source_timeslice, g_nsample);
  ofs = fopen(filename, "w");
  if( ofs == NULL ) {
    fprintf(stderr, "Error, could not open file %s for writing\n", filename);
    exit(23);
  }
  fprintf(ofs, "# %5d%3d%3d%3d%3d%10.6f%8.4f\n", Nconf, T, LX, LY, LZ, g_kappa, g_mu);
  for(i=0; i<nucleon_momentum_no; i++) {
  for(j=0; j<current_momentum_no; j++) {
    for(it=0;it<T;it++) {
      ix = (i*current_momentum_no + j)*T + it;

      fprintf(ofs, "%3d%3d%3d%16.7f%3d%3d%3d%16.7f%3d%25.16e%25.16e%25.16e%25.16e%25.15e%25.16e\n",  
          qlatt_rep[nucleon_momentum_list[i]][1], qlatt_rep[nucleon_momentum_list[i]][2], qlatt_rep[nucleon_momentum_list[i]][3],
          qlatt_list[nucleon_momentum_list[i]][0],
          qlatt_rep[current_momentum_list[j]][1], qlatt_rep[current_momentum_list[j]][2], qlatt_rep[current_momentum_list[j]][3],
          qlatt_list[current_momentum_list[j]][0],
          it, nucleon_2pt_disc_class[2*ix], nucleon_2pt_disc_class[2*ix+1],
          nucleon_2pt_class[2*(it*nucleon_momentum_no + i)], nucleon_2pt_class[2*(it*nucleon_momentum_no + i)+1],
          disc_class_nophase[2*j], disc_class_nophase[2*j+1]);
    }
  }}
  fclose(ofs);


  /***************************************
   * free the allocated memory, finalize
   ***************************************/
  free_geometry();

  if(nucleon_2pt_tq != NULL) free_sp_field(&nucleon_2pt_tq);
  if(nucleon_2pt_disc_class != NULL) free(nucleon_2pt_disc_class);
  if(nucleon_2pt_class != NULL) free(nucleon_2pt_class);
  if(disc_tq != NULL)           free(disc_tq);
  if(disc_class != NULL)        free(disc_class);
  if(disc_class_nophase != NULL)   free(disc_class_nophase);

  finalize_q_orbits(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_rep);
  if(qlatt_map != NULL) {
    free(qlatt_map[0]);
    free(qlatt_map);
  }

  if(sp1 != NULL) free_sp(&sp1);
  if(sp2 != NULL) free_sp(&sp2);

  if(nucleon_momentum_list != NULL) free(nucleon_momentum_list);
  if(current_momentum_list != NULL) free(current_momentum_list);

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [] %s# [] end of run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "\n# [] %s# [] end of run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
