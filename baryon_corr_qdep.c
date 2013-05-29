/****************************************************
 * baryon_corr_qdep.c
 *
 * Tue Nov 29 17:01:28 EET 2011
 *
 * PURPOSE
 * - produce the (q_x, q_y, q_z)- and t-dependent baryon correlator by
 *   projection; e.g. from output of proton_2pt_v3
 * - uses continuum momentum form (q_x = 2 pi nx / LX )
 * - needs momentum list as input
 * DONE:
 * TODO:
 * CHANGES:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#define MAIN_PROGRAM

#include "ifftw.h"
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
#include "icontract.h"
#include "spin_projection.h"

void usage() {
  fprintf(stdout, "Code to produce the (q_x,q_y,q_z)-dependent baryon correlator from the output of e.g. proton_2pt_v3\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, mu, status, count, i, j, ir, ir2;
  int filename_set = 0;
  int mode = 0;
  int l_LX_at, l_LXstart_at;
  int x0, it, ix, iclass, igamma, imom;
  int qlatt_nclass;
  spinor_propagator_type *connq=NULL, sp1, sp2, sp3, connq_proj[16];
  double q[4], qsqr, fnorm;
  int verbose = 0;
  int orbit_average=0;
  int read_ascii = 0;
  char filename[800], line[200], momentum_string[15];
  double ratime, retime;
  int sx0, sx1, sx2, sx3;
  int *qlatt_id=NULL, *qlatt_count=NULL, **qlatt_rep=NULL, **qlatt_map=NULL;
  double **qlatt_list=NULL;
  unsigned int VOL3;
  int use_lattice_momenta = 0;
  size_t items, bytes;
  FILE *ofs=NULL;
  double ***corrt=NULL;
  complex w;
  int num_component=1, icg, icomp;
  int do_spin_projection=-1, write_spin_projection=0;
/**************************************************************************/
  int momentum_filename_set = 0, momentum_no=0;
  char momentum_filename[200];
  int **momentum_list=NULL, *momentum_id=NULL;
/**************************************************************************/

 /*****************************************************************************************************************/
 /*
  * list of gamma projections to used:
  */
  const int gamma_proj_no = 1;
  int gamma_proj1[] = {4};
  int gamma_proj2[] = {0};
  int gamma_proj_isimag1[] = {0};
  int gamma_proj_isimag2[] = {0};
  double gamma_proj_sign1[] = {0.25};
  double gamma_proj_sign2[] = {0.25};
  int gamma_proj_fw_bw[] = {3};

//  const int gamma_proj_no = 16;
//  int gamma_proj1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15};
//  int gamma_proj2[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
//  int gamma_proj_isimag1[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//  int gamma_proj_isimag2[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; 
//  double gamma_proj_sign1[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};   
//  double gamma_proj_sign2[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};    
//  int gamma_proj_fw_bw[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};


  char gamma_proj_string[10];
  int gamma_proj_id, gamma_proj_isimag_id;
/*****************************************************************************************************************/

/***********************************************************/
  int snk_momentum_no = 0, isnk;
  int **snk_momentum_list = NULL;
  int snk_momentum_filename_set = 0;
  char snk_momentum_filename[200];
/***********************************************************/


  while ((c = getopt(argc, argv, "arsh?vf:p:n:P:S:W:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 's':
      use_lattice_momenta = 1;
      fprintf(stdout, "# [baryon_corr_qdep] will use lattice momenta\n");
      break;
    case 'p':
      momentum_filename_set = 1;
      strcpy(momentum_filename, optarg);
      fprintf(stdout, "# [baryon_corr_qdep] will use momentum file %s\n", momentum_filename);
      break;
    case 'n':
      num_component = atoi(optarg);
      fprintf(stdout, "# [baryon_corr_qdep] using number of components = %d\n", num_component);
      break;
    case 'P':
      snk_momentum_filename_set = 1;
      strcpy(snk_momentum_filename, optarg);
      fprintf(stdout, "# [baryon_corr_qdep] will read input data in ASCII format\n");
      break;
    case 'a':
      orbit_average = 1;
      fprintf(stdout, "# [baryon_corr_qdep] will average over momentum orbits\n");
      break;
    case 'r':
      read_ascii = 1;
      fprintf(stdout, "# [baryon_corr_qdep] will data in ascii format\n");
      break;
    case 'S':
      do_spin_projection = atoi(optarg);
      fprintf(stdout, "# [baryon_corr_qdep] will do spin projection of type %d\n", do_spin_projection);
      break;
    case 'W':
      write_spin_projection = atoi(optarg);
      fprintf(stdout, "# [baryon_corr_qdep] will write spin projection in mode (1 binary / 2 ascii) %d\n", write_spin_projection);
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  // set the default values
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [baryon_corr_qdep] Reading input from file %s\n", filename);
  read_input_parser(filename);

  // some checks on the input data
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[baryon_corr_qdep] T and L's must be set\n");
    usage();
  }

  if(do_spin_projection > -1 && num_component != 16) {
    fprintf(stderr, "[baryon_corr_qdep] Error, need 16 components to do spin projection\n");
    exit(4);
  }

  // initialize MPI parameters
  mpi_init(argc, argv);

  if(init_geometry() != 0) {
    fprintf(stderr, "[baryon_corr_qdep] ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  // the 3-dim. volume
  VOL3 = LX*LY*LZ;

  if( !use_lattice_momenta ) {
    status = make_qcont_orbits_3d_parity_avg( &qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
    if(status != 0) {
      fprintf(stderr, "\n[baryon_corr_qdep] Error while creating O_3-lists\n");
      exit(4);
    }
    fprintf(stdout, "# [baryon_corr_qdep] number of classes = %d\n", qlatt_nclass);
  } else {
    status = make_qlatt_orbits_3d_parity_avg( &qlatt_id, &qlatt_count, &qlatt_list, &qlatt_nclass, &qlatt_rep, &qlatt_map);
    if(status != 0) {
      fprintf(stderr, "\n[baryon_corr_qdep] Error while creating O_3-lists\n");
      exit(5);
    }
    fprintf(stdout, "# [baryon_corr_qdep] number of classes = %d\n", qlatt_nclass);
  }


  // determine the source location
  sx0 = g_source_location/(LX_global*LY_global*LZ);
  sx1 = (g_source_location%(LX_global*LY_global*LZ)) / (LY_global*LZ);
  sx2 = (g_source_location%(LY_global*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  fprintf(stdout, "# [baryon_corr_qdep] source location %d = (%d,%d,%d,%d)\n", g_source_location, sx0, sx1, sx2, sx3);



  if(snk_momentum_filename_set) {
    /***************************************************************************
     * read the nucleon final momenta to be used
     ***************************************************************************/
    ofs = fopen(snk_momentum_filename, "r");
    if(ofs == NULL) {
      fprintf(stderr, "[baryon_corr_qdep] Error, could not open file %s for reading\n", snk_momentum_filename);
      exit(6);
    }
    snk_momentum_no = 0;
    while( fgets(line, 199, ofs) != NULL) {
      if(line[0] != '#') {
        snk_momentum_no++;
      }
    }
    if(snk_momentum_no == 0) {
      fprintf(stderr, "[baryon_corr_qdep] Error, number of momenta is zero\n");
      exit(7);
    } else {
      fprintf(stdout, "# [baryon_corr_qdep] number of nucleon final momenta = %d\n", snk_momentum_no);
      fflush(stdout);
    }
    rewind(ofs);
    snk_momentum_list = (int**)malloc(snk_momentum_no * sizeof(int*));
    snk_momentum_list[0] = (int*)malloc(3*snk_momentum_no * sizeof(int));
    for(i=1;i<snk_momentum_no;i++) { snk_momentum_list[i] = snk_momentum_list[i-1] + 3; }
    count=0;
    while( fgets(line, 199, ofs) != NULL) {
      if(line[0] != '#') {
        sscanf(line, "%d%d%d", snk_momentum_list[count], snk_momentum_list[count]+1, snk_momentum_list[count]+2);
        count++;
      }
    }
    fclose(ofs);
    fprintf(stdout, "# [baryon_corr_qdep] the nucleon final momentum list:\n");
    for(i=0;i<snk_momentum_no;i++) {
      if(snk_momentum_list[i][0]<0) snk_momentum_list[i][0] += LX;
      if(snk_momentum_list[i][1]<0) snk_momentum_list[i][1] += LY;
      if(snk_momentum_list[i][2]<0) snk_momentum_list[i][2] += LZ;
      fprintf(stdout, "\t%3d%3d%3d%3d\n", i, snk_momentum_list[i][0], snk_momentum_list[i][1], snk_momentum_list[i][2]);
    }
  }  // of if snk_momentum_filename_set

  if(momentum_filename_set) {
    /***************************************************************************
     * read the momentum list to be used
     ***************************************************************************/
    ofs = fopen(momentum_filename, "r");
    if(ofs == NULL) {
      fprintf(stderr, "[baryon_corr_qdep] Error, could not open file %s for reading\n", momentum_filename);
      exit(6);
    }
    momentum_no = 0;
    while( fgets(line, 199, ofs) != NULL) {
      if(line[0] != '#') {
        momentum_no++;
      }
    }
    if(momentum_no == 0) {
      fprintf(stderr, "[baryon_corr_qdep] Error, number of momenta is zero\n");
      exit(7);
    } else {
      fprintf(stdout, "# [baryon_corr_qdep] number of momenta = %d\n", momentum_no);
      fflush(stdout);
    }
    rewind(ofs);
    momentum_list = (int**)malloc(momentum_no * sizeof(int*));
    momentum_list[0] = (int*)malloc(3*momentum_no * sizeof(int));
    for(i=1;i<momentum_no;i++) momentum_list[i] = momentum_list[i-1] + 3;
    count=0;
    while( fgets(line, 199, ofs) != NULL) {
      if(line[0] != '#') {
        sscanf(line, "%d%d%d", momentum_list[count], momentum_list[count]+1, momentum_list[count]+2);
        count++;
      }
    }
    fclose(ofs);
    for(i=0;i<momentum_no;i++) {
      if(momentum_list[i][0]<0) momentum_list[i][0] += LX;
      if(momentum_list[i][1]<0) momentum_list[i][1] += LY;
      if(momentum_list[i][2]<0) momentum_list[i][2] += LZ;
    }

    momentum_id = (int*)malloc(momentum_no * sizeof(int));
    if(orbit_average) {
      for(i=0;i<momentum_no;i++) {
        momentum_id[i] = qlatt_id[g_ipt[0][momentum_list[i][0]][momentum_list[i][1]][momentum_list[i][2]]];
      }
    } else {
      for(i=0;i<momentum_no;i++) {
        momentum_id[i] = g_ipt[0][momentum_list[i][0]][momentum_list[i][1]][momentum_list[i][2]];
      }
    }
    fprintf(stdout, "# [baryon_corr_qdep] the momentum id list:\n");
    for(i=0;i<momentum_no;i++) {
      fprintf(stdout, "\t%3d%6d\t%3d%3d%3d\n", i, momentum_id[i], momentum_list[i][0],  momentum_list[i][1], momentum_list[i][2]);
    }
    // check for multiple occurences
    for(i=0;i<momentum_no-1;i++) {
      for(j=i+1;j<momentum_no;j++) {
        if(momentum_id[i] == momentum_id[j]) {
          fprintf(stderr, "[baryon_corr_qdep] Error, multiple occurence of momentum no. %d: %6d = (%d, %d, %d)\n", i, momentum_id[i],
              momentum_list[i][0], momentum_list[i][1], momentum_list[i][2]);
          exit(127);
        }
      }
    }
  }  // of if momentum_filename_set

  // check source momentum
  if(g_source_momentum_set) {
    if(g_source_momentum[0] < 0) g_source_momentum[0] += LX_global;
    if(g_source_momentum[1] < 0) g_source_momentum[1] += LY_global;
    if(g_source_momentum[2] < 0) g_source_momentum[2] += LZ_global;
    fprintf(stdout, "# [] final source momentum = (%d, %d, %d)\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
  }


  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  if(!snk_momentum_filename_set) snk_momentum_no = LX*LY*LZ;
  fprintf(stdout, "# [baryon_corr_qdep] using snk_momentum_no = %d\n", snk_momentum_no);

  items = (size_t)T * snk_momentum_no * (size_t)num_component;
  connq = create_sp_field( items );
  if(connq == NULL) {
    fprintf(stderr, "\n[baryon_corr_qdep] Error, could not alloc connq\n");
    exit(2);
  }

  items = 4 * T * momentum_no * gamma_proj_no * num_component;
  bytes = sizeof(double);
  corrt = (double***)calloc(gamma_proj_no*num_component, sizeof(double**));
  if(corrt == NULL) {
    fprintf(stderr, "\nError, could not alloc corrt\n");
    exit(32);
  }
  corrt[0] = (double**)calloc(gamma_proj_no*momentum_no*num_component, sizeof(double*));
  if(corrt[0] == NULL) {
    fprintf(stderr, "\nError, could not alloc corrt\n");
    exit(33);
  }
  for(i=1;i<gamma_proj_no*num_component;i++) corrt[i] = corrt[i-1] + momentum_no;
  corrt[0][0] = (double*)calloc(gamma_proj_no*num_component*momentum_no*4*T, sizeof(double));
  if(corrt[0][0] == NULL) {
    fprintf(stderr, "\nError, could not alloc corrt\n");
    exit(34);
  }
  count=0;
  for(i=0;i<gamma_proj_no*num_component;i++) {
    for(j=0;j<momentum_no;j++) {
      if(count>0) {
        corrt[i][j] = corrt[0][0] + count * 4*T;
      }
      count++;
    }
  }

  for(ix=0;ix<items;ix++)  corrt[0][0][ix] = 0.;


  create_sp(&sp1);
  create_sp(&sp2);
  create_sp(&sp3);

  /***********************
   * read contractions   *
   ***********************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  if(!read_ascii) {  // read binary input data from xxx_q-file
    if(!g_source_momentum_set) {
      sprintf(filename, "%s_q.%.4d.t%.2dx%.2dy%.2dz%.2d", filename_prefix, Nconf, sx0, sx1, sx2, sx3);
    } else {
      sprintf(filename, "%s_q.%.4d.t%.2dx%.2dy%.2dz%.2d.qx%.2dqy%.2dqz%.2d", filename_prefix, Nconf, sx0, sx1, sx2, sx3,
          g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    }
    fprintf(stdout, "# [baryon_corr_qdep] Reading data from file %s\n", filename);
    status = read_lime_contraction_v2(connq[0][0], filename, g_sv_dim*g_sv_dim*num_component, 0);
    if(status != 0) {
      fprintf(stderr, "Error: could not read from file %s; status was %d\n", filename, status);
      exit(12);
    }
  } else {  // read ascii input data from xxx_snk-file
    int itmp[6];
    if(!g_source_momentum_set) {
      sprintf(filename, "%s_snk.%.4d.t%.2dx%.2dy%.2dz%.2d", filename_prefix, Nconf, sx0, sx1, sx2, sx3);
    } else {
      sprintf(filename, "%s_snk.%.4d.t%.2dx%.2dy%.2dz%.2d.qx%.2dqy%.2dqz%.2d", filename_prefix, Nconf, sx0, sx1, sx2, sx3,
          g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    }
    fprintf(stdout, "# [baryon_corr_qdep] Reading ascii data from file %s\n", filename);
    if( (ofs = fopen(filename, "r")) == NULL) {
      return(102);
    }

    // check for # at beginning of first line
    c = fgetc(ofs);
    rewind(ofs);
    if(c == '#') {
      fprintf(stdout, "# [baryon_corr_qdep] header line: %s", fgets(filename, 199, ofs) );
    }

    for(isnk=0;isnk<snk_momentum_no;isnk++) {
      for(icomp=0; icomp<num_component; icomp++) {
        for(it=0;it<T;it++) {
          for(i=0;i<g_sv_dim*g_sv_dim;i++) {
            fscanf(ofs, "%d%d%d%lf%lf%d%d%d", itmp, itmp+1,itmp+2, connq[(it*snk_momentum_no+isnk)*num_component+icomp][i/g_sv_dim]+2*(i%g_sv_dim), connq[(it*snk_momentum_no+isnk)*num_component+icomp][i/g_sv_dim]+2*(i%g_sv_dim)+1, itmp+3,itmp+4,itmp+5);
            fprintf(stdout, "\t%3d%3d%3d%25.16e%25.16e%3d%3d%3d\n", itmp[0], itmp[1], itmp[2], connq[(it*snk_momentum_no+isnk)*num_component+icomp][i/g_sv_dim][2*(i%g_sv_dim)], connq[(it*snk_momentum_no+isnk)*num_component+icomp][i/g_sv_dim][2*(i%g_sv_dim)+1], itmp[3], itmp[4], itmp[5]);
            fflush(stdout);
          }
        }
      }
    }
    fclose(ofs);
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to read contractions %e seconds\n", retime-ratime);

  /***********************
   * spin projection
   ***********************/
  if(do_spin_projection > -1) {
    // spin projection using zero momentum formula
    for(i=0; i<16; i++) { create_sp(connq_proj+i); }

    if(do_spin_projection == 30) {
      // zero momentum, spin 3/2
      for(isnk=0; isnk<snk_momentum_no; isnk++)
      {
        for(it=0;it<T;it++)
        {
          spin_projection_3_2_zero_momentum (connq_proj, connq+(it*snk_momentum_no+isnk)*num_component);
          for(icomp=0; icomp<num_component; icomp++) {
            _sp_eq_sp(connq[(it*snk_momentum_no+isnk)*num_component+icomp], connq_proj[icomp]);
          }
        }
      }
    } else if (do_spin_projection == 10) {
      // zero momentum, spin 1/2
      for(isnk=0; isnk<snk_momentum_no; isnk++)
      {
        for(it=0;it<T;it++)
        {
          spin_projection_1_2_zero_momentum (connq_proj, connq+(it*snk_momentum_no+isnk)*num_component);
          for(icomp=0; icomp<num_component; icomp++) {
            _sp_eq_sp(connq[(it*snk_momentum_no+isnk)*num_component+icomp], connq_proj[icomp]);
          }
        }
      }
    } else {
      // spin projection for arbitrary momentum
      fprintf(stderr, "[baryon_corr_qdep] not yet implemented\n");
    }

    for(i=0; i<16; i++) { free_sp(connq_proj+i); }

    if(write_spin_projection==2) {
      sprintf(filename, "%s_proj%d.%.4d.t%.2dx%.2dy%.2dz%.2d.ascii",
          filename_prefix, do_spin_projection, Nconf, sx0, sx1, sx2, sx3);
      fprintf(stdout, "# [baryon_corr_qdep] writing spin projection in ascii format to file %s\n", filename);
      write_contraction2( connq[0][0], filename, num_component*g_sv_dim*g_sv_dim, T*snk_momentum_no, 1, 0);
    } else if (write_spin_projection == 1) {
      fprintf(stderr, "[baryon_corr_qdep] not yet implemented\n");
    }

  } else {
    fprintf(stdout, "[baryon_corr_qdep] proceed without spin projection\n");
  }

  /***********************
   * fill the correlator *
   ***********************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  /*********************************************************************************************
   * with orbit averaging
   *********************************************************************************************/
  if(orbit_average) {

    for(iclass=0;iclass<momentum_no; iclass++) {

      for(it=0;it<T;it++) {

        count=0;
        for(ix=0;ix<qlatt_count[momentum_id[iclass]]; ix++) {
    
          // check if momentum is availabley
          if(snk_momentum_no<VOL3 || snk_momentum_filename_set) {
            imom = -1;
            for(i=0;i<snk_momentum_no; i++) {
              if(  qlatt_map[momentum_id[iclass]][ix] == g_ipt[0][snk_momentum_list[i][0]][snk_momentum_list[i][1]][snk_momentum_list[i][2]] ) {
                imom = i;
                break;
              }
            }
            if(imom == -1) {
              fprintf(stdout, "[baryon_corr_qdep] Warning, could not find representative no. %d (%d) for momentum class no. %d in list of available sink momenta; skip\n",
                  ix, qlatt_map[iclass][ix], momentum_id[iclass]);
              continue;
            } else {
              fprintf(stdout, "momentum no. %d/%d = momentum no. %d/(%d,%d,%d) in list of available sink momenta\n", ix, momentum_id[iclass], imom,
                snk_momentum_list[imom][0], snk_momentum_list[imom][1], snk_momentum_list[imom][2]);
              count++;
           }
          } else {
            imom = qlatt_map[momentum_id[iclass]][ix];
            count++;
          }
          // fprintf(stdout, "momentum no. %d/%d = momentum no. %d in list of available sink momenta\n", ix, momentum_id[iclass], imom);
          
        
          icg = 0; 
          for(icomp=0; icomp<num_component; icomp++) {

            for(igamma=0;igamma<gamma_proj_no; igamma++) { 
  
              // forward part
              if(gamma_proj_fw_bw[igamma] == 1 || gamma_proj_fw_bw[igamma] == 3) {
                if(gamma_proj1[igamma]>=0 ) {
                  _sp_eq_gamma_ti_sp(sp3, gamma_proj1[igamma], connq[ (it*snk_momentum_no + imom)*num_component+icomp ]);
                  if(gamma_proj_isimag1[igamma]) {
                    _sp_eq_sp_ti_im( sp1, sp3, gamma_proj_sign1[igamma] );
                  } else {
                    _sp_eq_sp_ti_re( sp1, sp3, gamma_proj_sign1[igamma] );
                  }
                } else {
                  _sp_eq_zero( sp1 );
                }
  
                if(gamma_proj2[igamma]>=0 ) {
                  _sp_eq_gamma_ti_sp(sp3, gamma_proj2[igamma], connq[ (it*snk_momentum_no + imom)*num_component+icomp]);
                  if(gamma_proj_isimag2[igamma]) {
                    _sp_eq_sp_ti_im( sp2, sp3, gamma_proj_sign2[igamma] );
                  } else {
                    _sp_eq_sp_ti_re( sp2, sp3, gamma_proj_sign2[igamma] );
                  }
                } else {
                  _sp_eq_zero( sp2 );
                }
   
                _sp_pl_eq_sp(sp1, sp2);
                _co_eq_tr_sp(&w, sp1);
                corrt[icg][iclass][2*it  ] += w.re;
                corrt[icg][iclass][2*it+1] += w.im;
              }
  
              // backward part
              if(gamma_proj_fw_bw[igamma] == 2 || gamma_proj_fw_bw[igamma] == 3) {
                if(gamma_proj1[igamma]>=0 ) {
                  _sp_eq_gamma_ti_sp(sp3, gamma_proj1[igamma], connq[ (it*snk_momentum_no + imom)*num_component+icomp]);
                  if(gamma_proj_isimag1[igamma]) {
                    _sp_eq_sp_ti_im( sp1, sp3, gamma_proj_sign1[igamma] );
                  } else {
                    _sp_eq_sp_ti_re( sp1, sp3, gamma_proj_sign1[igamma] );
                  }
                } else {
                  _sp_eq_zero( sp1 );
                }
  
                if(gamma_proj2[igamma]>=0 ) {
                  _sp_eq_gamma_ti_sp(sp3, gamma_proj2[igamma], connq[ (it*snk_momentum_no + imom)*num_component+icomp]);
                  if(gamma_proj_isimag2[igamma]) {
                    _sp_eq_sp_ti_im( sp2, sp3, gamma_proj_sign2[igamma] );
                  } else {
                    _sp_eq_sp_ti_re( sp2, sp3, gamma_proj_sign2[igamma] );
                  }
                } else {
                  _sp_eq_zero( sp2 );
                }
                _sp_mi_eq_sp(sp1, sp2);
                _co_eq_tr_sp(&w, sp1);
                corrt[icg][iclass][2*(T+it)  ] += w.re;
                corrt[icg][iclass][2*(T+it)+1] += w.im;
              }
  
              icg++;
            }  // of igamma
          }    // of icomp
        
        }  // of loop on momentum representations

        // normalization
        if(count==0) {
          fprintf(stderr, "[baryon_corr_qdep] Error, no momenta from class %d available\n", momentum_id[iclass]);
        } else {
          fnorm = 1. / (double)count;
          //fprintf(stdout, "# [baryon_corr_qdep] norm for momentum class no. %d (%d) = %e\n", iclass, momentum_id[iclass], fnorm);
          for(icg=0;icg<gamma_proj_no*num_component; icg++) { 
            corrt[icg][iclass][2*(  it)  ] *= fnorm;
            corrt[icg][iclass][2*(  it)+1] *= fnorm;
            corrt[icg][iclass][2*(T+it)  ] *= fnorm;
            corrt[icg][iclass][2*(T+it)+1] *= fnorm;
          }
        }

      }    // of loop on times (it)
    }      // of loops on mometum classes (iclass)
  /*********************************************************************************************
   * without orbit averaging
   *********************************************************************************************/
  } else {  // do not average over momentum orbits
    for(iclass=0;iclass<momentum_no; iclass++) {

      if(snk_momentum_no<VOL3 || snk_momentum_filename_set) {
        ix = -1;
        for(i=0;i<snk_momentum_no; i++) {
          if(  momentum_list[iclass][0]==snk_momentum_list[i][0] && momentum_list[iclass][1]==snk_momentum_list[i][1]
            && momentum_list[iclass][2]==snk_momentum_list[i][2]) {
            ix=i;
            break;
          }
        }
        if(ix== -1) {
          fprintf(stdout, "[baryon_corr_qdep] Warning, could not find momentum no. %d in list of available sink momenta; skip\n", iclass);
          continue;
        }
      } else {
        ix = g_ipt[0][momentum_list[iclass][0]][momentum_list[iclass][1]][momentum_list[iclass][2]];
      }
      fprintf(stdout, "momentum no. %d = momentum no. %d in list of available sink momenta\n", iclass, ix);
      for(it=0;it<T;it++) {
          icg = 0; 
          for(icomp=0; icomp<num_component; icomp++) {

            for(igamma=0;igamma<gamma_proj_no; igamma++) { 
  
              // forward part
              if(gamma_proj_fw_bw[igamma] == 1 || gamma_proj_fw_bw[igamma] == 3) {
                if(gamma_proj1[igamma]>=0 ) {
                  _sp_eq_gamma_ti_sp(sp3, gamma_proj1[igamma], connq[ (it*snk_momentum_no+ix)*num_component+icomp ]);
                  if(gamma_proj_isimag1[igamma]) {
                    _sp_eq_sp_ti_im( sp1, sp3, gamma_proj_sign1[igamma] );
                  } else {
                    _sp_eq_sp_ti_re( sp1, sp3, gamma_proj_sign1[igamma] );
                  }
                } else {
                  _sp_eq_zero( sp1 );
                }
  
                if(gamma_proj2[igamma]>=0 ) {
                  _sp_eq_gamma_ti_sp(sp3, gamma_proj2[igamma], connq[ (it*snk_momentum_no+ix)*num_component+icomp]);
                  if(gamma_proj_isimag2[igamma]) {
                    _sp_eq_sp_ti_im( sp2, sp3, gamma_proj_sign2[igamma] );
                  } else {
                    _sp_eq_sp_ti_re( sp2, sp3, gamma_proj_sign2[igamma] );
                  }
                } else {
                  _sp_eq_zero( sp2 );
                }
   
                _sp_pl_eq_sp(sp1, sp2);
                _co_eq_tr_sp(&w, sp1);
                corrt[icg][iclass][2*it  ] += w.re;
                corrt[icg][iclass][2*it+1] += w.im;
              }
  
              // backward part
              if(gamma_proj_fw_bw[igamma] == 2 || gamma_proj_fw_bw[igamma] == 3) {
                if(gamma_proj1[igamma]>=0 ) {
                  _sp_eq_gamma_ti_sp(sp3, gamma_proj1[igamma], connq[ (it*snk_momentum_no+ix)*num_component+icomp]);
                  if(gamma_proj_isimag1[igamma]) {
                    _sp_eq_sp_ti_im( sp1, sp3, gamma_proj_sign1[igamma] );
                  } else {
                    _sp_eq_sp_ti_re( sp1, sp3, gamma_proj_sign1[igamma] );
                  }
                } else {
                  _sp_eq_zero( sp1 );
                }
  
                if(gamma_proj2[igamma]>=0 ) {
                  _sp_eq_gamma_ti_sp(sp3, gamma_proj2[igamma], connq[ (it*snk_momentum_no+ix)*num_component+icomp]);
                  if(gamma_proj_isimag2[igamma]) {
                    _sp_eq_sp_ti_im( sp2, sp3, gamma_proj_sign2[igamma] );
                  } else {
                    _sp_eq_sp_ti_re( sp2, sp3, gamma_proj_sign2[igamma] );
                  }
                } else {
                  _sp_eq_zero( sp2 );
                }
                _sp_mi_eq_sp(sp1, sp2);
                _co_eq_tr_sp(&w, sp1);
                corrt[icg][iclass][2*(T+it)  ] += w.re;
                corrt[icg][iclass][2*(T+it)+1] += w.im;
              }
  
              icg++;
            }  // of igamma
          }    // of icomp
        
      }    // of loop on times
    }      // of loops on mometum classes
  }        // else of if orbit_average


  /*************************************************************************** 
   * write to file   
   ***************************************************************************/
  icg = 0;
  for(icomp=0;icomp<num_component;icomp++) {
    for(igamma=0;igamma<gamma_proj_no; igamma++) {
      // set gamma string
      if(gamma_proj1[igamma] >= 0) {
        if(gamma_proj_isimag1[igamma]) {
          sprintf(gamma_proj_string, "ig%.2d", gamma_proj1[igamma]);
        } else {
          sprintf(gamma_proj_string, "g%.2d", gamma_proj1[igamma]);
        }
        if(gamma_proj2[igamma] >= 0) {
          if(gamma_proj_isimag2[igamma]) {
            sprintf(line, "%s_ig%.2d", gamma_proj_string, gamma_proj2[igamma]);
          } else {
            sprintf(line, "%s_g%.2d", gamma_proj_string, gamma_proj2[igamma]);
          }
          strcpy(gamma_proj_string, line);
        }
      } else {
        if(gamma_proj2[igamma] >=0 ) {
          if(gamma_proj_isimag2[igamma]) {
            sprintf(gamma_proj_string, "ig%.2d", gamma_proj2[igamma]);
          } else {
            sprintf(gamma_proj_string, "g%.2d", gamma_proj2[igamma]);
          }
        } else {
          fprintf(stderr, "[baryon_corr_qdep] Error, cannot use gamma combination\n");
          continue;
        }
      }  // of if gamma_proj1 >= 0

      // set unique gamma id and gamma isimag
      gamma_proj_isimag_id = 0;
      gamma_proj_id        = 0;
      if(gamma_proj1[igamma]>= 0) {
        if(gamma_proj_isimag1[igamma]) gamma_proj_isimag_id += 1;
        gamma_proj_id = gamma_proj1[igamma];
      }
      if(gamma_proj2[igamma]>= 0) {
        if(gamma_proj_isimag2[igamma]) gamma_proj_isimag_id += 2;
        gamma_proj_id = 16 * gamma_proj_id + gamma_proj2[igamma];
      }
      if(gamma_proj1[igamma] >= 0 && gamma_proj2[igamma] >= 0) {
        gamma_proj_id += 16;
      }
      if(gamma_proj1[igamma] < 0 && gamma_proj2[igamma] < 0) {
        gamma_proj_id = -1;
      }

      for(iclass=0;iclass<momentum_no; iclass++) {

        sprintf(momentum_string, "x%.2dy%.2dz%.2d",
//              qlatt_rep[momentum_list[iclass]][1], qlatt_rep[momentum_list[iclass]][2], qlatt_rep[momentum_list[iclass]][3]);
            momentum_list[iclass][0], momentum_list[iclass][1], momentum_list[iclass][2]);

        // forward part
        if(gamma_proj_fw_bw[igamma] == 1 || gamma_proj_fw_bw[igamma] == 3) {

          sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%s.%s.fw", filename_prefix2, Nconf, sx0, sx1, sx2, sx3, gamma_proj_string, momentum_string);
          ofs = icomp==0 ? fopen(filename, "w") : fopen(filename, "a");
          if( ofs == (FILE*)NULL ) {
            fprintf(stderr, "Error: could not open file %s for writing\n", filename);
            exit(5);
          }
//            fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f%16.7e%16.7e%16.7e%16.7e\n", Nconf, T_global, LX_global, LY_global, LZ, g_kappa, g_mu,
//                qlatt_list[momentum_list[iclass]][0], qlatt_list[momentum_list[iclass]][1],
//                qlatt_list[momentum_list[iclass]][2], qlatt_list[momentum_list[iclass]][3]);
          fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f%3d%3d%3d\n", Nconf, T_global, LX_global, LY_global, LZ, g_kappa, g_mu,
              momentum_list[iclass][0], momentum_list[iclass][1], momentum_list[iclass][2]);
    
//            ir  = sx0;
//            fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d\n", gamma_proj_id, gamma_proj_isimag_id, 0, corrt[icg][iclass][2*ir], 0., Nconf, icomp);
//            for(it=1; it<(T_global/2); it++) {
//              ir  = (it + sx0) % T_global;
//              ir2 = ( (T_global - it) + sx0) % T_global;
//              fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d\n", gamma_proj_id, gamma_proj_isimag_id, it, corrt[icg][iclass][2*ir],
//                 corrt[icg][iclass][2*ir2], Nconf, icomp);
//            }
//            ir = ( (T_global/2) + sx0 ) % T_global;
//            fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d\n", gamma_proj_id, gamma_proj_isimag_id, (T_global/2), corrt[icg][iclass][2*ir], 0., Nconf, icomp);
          for(it=0; it<T_global; it++) {
            fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d\n", gamma_proj_id, gamma_proj_isimag_id, it, corrt[icg][iclass][2*it],
               corrt[icg][iclass][2*it+1], Nconf, icomp);
          }
          fflush(ofs);
          fclose(ofs);
        }

        // backward part
        if(gamma_proj_fw_bw[igamma] == 2 || gamma_proj_fw_bw[igamma] == 3) {
          sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%s.%s.bw", filename_prefix2, Nconf, sx0, sx1, sx2, sx3, gamma_proj_string, momentum_string);
          ofs = icomp==0 ? fopen(filename, "w") : fopen(filename, "a");
          if( ofs == (FILE*)NULL ) {
            fprintf(stderr, "Error: could not open file %s for writing\n", filename);
            exit(5);
          }
//            fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f%16.7e%16.7e%16.7e%16.7e\n", Nconf, T_global, LX_global, LY_global, LZ, g_kappa, g_mu,
//                qlatt_list[momentum_list[iclass]][0], qlatt_list[momentum_list[iclass]][1],
//                qlatt_list[momentum_list[iclass]][2], qlatt_list[momentum_list[iclass]][3]);
    
          fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f%3d%3d%3d\n", Nconf, T_global, LX_global, LY_global, LZ, g_kappa, g_mu,
              momentum_list[iclass][0], momentum_list[iclass][1], momentum_list[iclass][2]);
//            ir  = sx0;
//            fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d\n", gamma_proj_id, gamma_proj_isimag_id, 0, corrt[icg][iclass][2*(T_global+ir)], 0., Nconf, icomp);
//            for(it=1; it<(T_global/2); it++) {
//              ir  = (it + sx0) % T_global               + T_global;
//              ir2 = ( (T_global - it) + sx0) % T_global + T_global;
//              fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d\n", gamma_proj_id, gamma_proj_isimag_id, it, corrt[icg][iclass][2*ir], corrt[icg][iclass][2*ir2], Nconf, icomp);
//            }
//            ir = ( (T_global/2) + sx0 ) % T_global      + T_global;
//            fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d\n", gamma_proj_id, gamma_proj_isimag_id, (T_global/2), corrt[icg][iclass][2*ir], 0., Nconf, icomp);
          for(it=0; it<T_global; it++) {
            ir  = it + T_global;
            fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d%3d\n", gamma_proj_id, gamma_proj_isimag_id, it, corrt[icg][iclass][2*ir], corrt[icg][iclass][2*ir+1], Nconf, icomp);
          }
          fflush(ofs);
          fclose(ofs);
        }

      }  // of loop on classes


      icg++;
    }    // of loop on gamma_proj
  }      // of loop on icomp

  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# [baryon_corr_qdep] time to fill and write correlator: %e seconds\n", retime-ratime);


  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  if(corrt != NULL) {
    if(corrt[0] != NULL) {
      if(corrt[0][0] != NULL) {
        free(corrt[0][0]);
      }
      free(corrt[0]);
    }
    free(corrt);
  }
  free_sp_field(&connq);
  if(momentum_list != NULL) {
    if(momentum_list[0]!=NULL) free(momentum_list[0]);
    free(momentum_list);
  }
  if(momentum_id!=NULL) free(momentum_id);
  if(snk_momentum_list != NULL) {
    if(snk_momentum_list[0]!=NULL) free(snk_momentum_list[0]);
    free(snk_momentum_list);
  }

  free_geometry();

  free_sp(&sp1);
  free_sp(&sp2);
  free_sp(&sp3);

  finalize_q_orbits(&qlatt_id, &qlatt_count, &qlatt_list, &qlatt_rep);
  if(qlatt_map != NULL) {
    free(qlatt_map[0]);
    free(qlatt_map);
  }

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [baryon_corr_qdep] %s# [baryon_corr_qdep] end of run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "\n# [baryon_corr_qdep] %s# [baryon_corr_qdep] end of run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);

}
