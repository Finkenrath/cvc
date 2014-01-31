/***********************************************************************
 *
 * Copyright (C) 2013 Bartosz Kostrzewa
 *
 * This file is part of CVC.
 *
 * CVC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * CVC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with CVC.  If not, see <http://www.gnu.org/licenses/>.
 *
 ************************************************************************/

#define MAIN_PROGRAM

#include <iostream>
#include <algorithm>

#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <getopt.h>

#include "global.h"
#include "cvc_complex.h"
#include "read_input_parser.h"
#include "cvc_linalg.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "init_gauge_field.h"
#include "fatal_error.h"

#include "flavour.hpp"
#include "flavour_pairing.hpp"
#include "flavour_params.hpp"
#include "correlator.hpp"

using namespace std;

void free_global_data_structures();
void init_global_data_structures();
void collect_props(double** prop_a, double** prop_b, vector<propagator>& props_a, vector<propagator>& props_b, unsigned int no_spin_colour_indices);
void zero_correls(vector< vector<correlator*> >& correls);

string construct_correlator_filename_create_subdirectory(const string flavour_pairing_name, const flavour* const fl_a, const flavour* const fl_b, const unsigned int mass_index_a, const unsigned int mass_index_b);
void output_correlators(const vector< vector<correlator*> >& correls, const string flavour_pairing_name, const flavour* const fl_a, const flavour* const fl_b, const unsigned int mass_index_a, const unsigned int mass_index_b);

void process_args(int argc, char **argv);
void usage();

/*********************************************************************************************************
 * Some global definitions for convenience
 */

  double c_conf_gamma_sign[]  = {1., 1., 1., -1., -1., -1., -1., 1., 1., 1., -1., -1.,  1.,  1., 1., 1.};
  double n_conf_gamma_sign[]  = {1., 1., 1., -1., -1., -1., -1., 1., 1., 1.,  1.,  1., -1., -1., 1., 1.};

  // 16 (pseudo-)scalar, 16 (pseudo-)vector [ averaged over 3 spatial directions -> 48 gamma combinations,
  // see gindex[1,2] and ngindex[1,2] below ]
  const unsigned int N_observables = 32;

  /**************************************************************************************************
   * charged stuff
   *
   * (pseudo-)scalar: observables 1-16
   * 
   * g5 - g5,	g5   - g0g5,	g0g5 - g5,	g0g5 - g0g5,
   * g0 - g0,	g5   - g0,	g0   - g5,	g0g5 - g0,
   * g0 - g0g5,	1    - 1,	1    - g5,	g5   - 1,
   * 1  - g0g5,	g0g5 - 1,	1    - g0,	g0   - 1
   *
   * (pseudo-)vector: observables 17-32
   * 
   * gig0 - gig0,	gi     - gi,		gig5 - gig5,	gig0   - gi,
   * gi   - gig0,	gig0   - gig5,		gig5 - gig0,	gi     - gig5,
   * gig5 - gi,		gig0g5 - gig0g5,	gig0 - gig0g5,	gig0g5 - gig0,
   * gi   - gig0g5,	gig0g5 - gi,		gig5 - gig0g5,	gig0g5 - gig5
   **************************************************************************************************/
  int gindex1[] = {5, 5, 6, 6, 0, 5, 0, 6, 0, 4, 4, 5, 4, 6, 4, 0,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 10, 11, 12, 1, 2, 3, 10, 11, 12, 7, 8, 9, 1, 2, 3, 7, 8, 9,
                   13, 14, 15, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13, 7, 8, 9, 15, 14, 13};

  int gindex2[] = {5, 6, 5, 6, 0, 0, 5, 0, 6, 4, 5, 4, 6, 4, 0, 4,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 1, 2, 3, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 1, 2, 3,
                   13, 14, 15, 15, 14, 13, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13, 7, 8, 9};

  /* due to twisting we have several correlators that are purely imaginary */
  int isimag[]  = {0, 0, 0, 0, 
                   0, 1, 1, 1, 
                   1, 0, 0, 0, 
                   1, 1, 0, 0,

                   0, 0, 0, 0, 
                   0, 1, 1, 1, 
                   1, 0, 1, 1, 
                   1, 1, 0, 0};

  double vsign[]  = {1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1., -1., 1.,   1., -1., 1., 
                     1., -1., 1.,   1., -1., 1.,   1., -1., 1.,   1., -1., 1.};


  /**************************************************************************************************
   * neutral stuff 
   *
   * (pseudo-)scalar:
   * g5 - g5,	g5   - g0g5,	g0g5 - g5,	g0g5 - g0g5,
   * 1  - 1,	g5   - 1,	1    - g5,	g0g5 - 1,
   * 1  - g0g5,	g0   - g0,	g0   - g5,	g5   - g0,
   * g0 - g0g5,	g0g5 - g0,	g0   - 1,	1    - g0
   *
   * (pseudo-)vector:
   * gig0   - gig0,		gi   - gi,	gig0g5 - gig0g5,	gig0   - gi, 
   * gi     - gig0,		gig0 - gig0g5,	gig0g5 - gig0,		gi     - gig0g5,
   * gig0g5 - gi		gig5 - gig5,	gig5   - gi,		gi     - gig5,
   * gig5   - gig0,		gig0 - gig5,	gig5   - gig0g5,	gig0g5 - gig5
   **************************************************************************************************/
  int ngindex1[] = {5, 5, 6, 6, 4, 5, 4, 6, 4, 0, 0, 5, 0, 6, 0, 4,
                    10, 11, 12, 1, 2, 3, 13, 14, 15, 10, 11, 12,  1,  2,  3, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13,
                     7,  8,  9, 7, 8, 9,  1,  2,  3,  7,  8,  9, 10, 11, 12,  7,  8,  9, 15, 14, 13};

  int ngindex2[] = {5, 6, 5, 6, 4, 4, 5, 4, 6, 0, 5, 0, 6, 0, 4, 0,
                    10, 11, 12, 1, 2, 3, 13, 14, 15,  1,  2,  3, 10, 11, 12, 15, 14, 13, 10, 11, 12, 15, 14, 13, 1, 2, 3,
                     7,  8,  9, 1, 2, 3,  7,  8,  9, 10, 11, 12,  7,  8,  9, 15, 14, 13,  7,  8, 9};

  int nisimag[]  = {0, 0, 0, 0,
                    0, 1, 1, 1,
                    1, 0, 1, 1,
                    1, 1, 0, 0,

                    0, 0, 0, 0,
                    0, 1, 1, 1, 
                    1, 0, 1, 1,
                    1, 1, 0, 0};

  double nvsign[] = {1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1., 
                     1.,  1., 1.,   1., -1., 1.,   1., -1., 1.,   1., -1., 1.,
                     1., -1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                     1.,  1., 1.,   1.,  1., 1.,   1., -1., 1.,   1., -1., 1. };

/*
  double isneg_std[]=    {+1., -1., +1., -1., +1., +1., +1., +1., -1., +1., +1., +1., +1., +1., +1., +1.,    
                          -1., +1., -1., -1., +1., +1., +1., -1., +1., -1., +1., +1., +1., +1., +1., +1.}; 
*/
  double isneg_std[]=    {+1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1.,    
                          +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1.};

  double* current_isneg = isneg_std;
  
  // these two will be set depending on whether we are computing neutral or charged meson
  // correlators
  int *current_gindex1, *current_gindex2, *current_isimag;
  double *current_vsign, *current_conf_gamma_sign;
  
  double correlator_norm = 1;

/**************************************************************************************/

int verbose = 0;
string input_filename = "cvc.input";

int main(int argc, char **argv) {  
#ifdef MPI
  MPI_Status status;
  MPI_Init(&argc, &argv);
#endif  

  process_args(argc,argv);
  // the input file defines flavours and flavour combinations
  // and the relevant data structures are initialized below
  read_input_parser(input_filename.c_str()); 

  mpi_init(argc,argv);
  if(init_geometry() != 0) {
    fatal_error(1, "ERROR: init_geometry failed!\n");
  }
  
  geometry();
  
  // this seems to be the default correlator normalisation used by the ETMC toolset
  correlator_norm = 1/(2*g_kappa*g_kappa*VOL3*g_nproc_x*g_nproc_y*g_nproc_z);  
  
  // the initialization functions for flavour and flavour_pairing
  // take care of memory management, reading propagators and smearing
  init_global_data_structures();
  
  // pointers to the 1, 4 or 12 propagators for each flavour
  // depending on n_s*n_c
  double* prop_a[12];
  double* prop_b[12];
  
  for(unsigned int i = 0; i < 12; ++i){
    prop_a[i] = NULL;
    prop_b[i] = NULL;
  }

  // buffer memory for allreduce in correlator computation
#ifdef MPI
  double allreduce_buffer[2*T];
#else
  double* allreduce_buffer = (double*)NULL;
#endif

  // storage for correlators for N_observables obsevables and (up to) 4 smearing combinations each (L/L,L/S,S/L,S/S - source/sink)
  vector< vector<correlator*> > correls;
  correls.resize(N_observables);
  for(vector< vector<correlator*> >::iterator iter = correls.begin(); iter != correls.end(); ++iter){
    for(unsigned int smearing_combinations = 0; smearing_combinations < 4; ++smearing_combinations){
      iter->push_back( new correlator(allreduce_buffer) );
    }
  }
  // explicitly zero out all correlators
  zero_correls(correls);

  /* temporary storage for correlator
   *   2 - complex
   * T/2 - forward correlator
   * T/2 - backward correlator
   * ---------------------------> 2*T */
  double corr_temp[2*T];
  double corr_temp_vector[2*T];
  
  for( vector<flavour_pairing*>::iterator fp_it = g_flavour_pairings.begin(); fp_it != g_flavour_pairings.end(); ++fp_it ) {
    // convenience variables for the flavour objects
    flavour* fl_a = (*fp_it)->a;
    flavour* fl_b = (*fp_it)->b;
    
    for( unsigned int mass_index_a = 0; mass_index_a < fl_a->params.no_masses ; ++mass_index_a ) {
      for( unsigned int mass_index_b = 0; mass_index_b < fl_b->params.no_masses ; ++mass_index_b ) {
        
        // if the two flavours are the same, we only do the mass diagonal case!
        if ( fl_a == fl_b ) {
          // since the flavours are the same, the mass indices will have the same ranges
          // and we can use this as a criterion for staying on the mass diagonal
          if( mass_index_a != mass_index_b ) {
            continue;
          }
          
        // select arrays of gamma permutations depending on whether we are
        // contracting a neutral (same flavours) or a charged meson (different flavours)
        // FIXME: this actually won't work because in twisted mass the propagators
        // in the pseudoscalar correlator can be related and are effectively of the same
        // flavor!
        // need some control code in flavour combination
        //  current_gindex1 = ngindex1;
        //  current_gindex2 = ngindex2;
        //  current_isimag = nisimag;
        //  current_vsign = nvsign;
        //  current_conf_gamma_sign = n_conf_gamma_sign;
        } else {
        //  current_gindex1 = gindex1;
        //  current_gindex2 = gindex2;
        //  current_isimag = isimag;
        //  current_vsign = vsign;
        //  current_conf_gamma_sign = c_conf_gamma_sign;
        }
        
        // FIXME: for now hard-code for charged gamma combinations
        
        current_gindex1 = gindex1;
        current_gindex2 = gindex2;
        current_isimag = isimag;
        current_vsign = vsign;
        current_conf_gamma_sign = c_conf_gamma_sign;
        
        // zero out memory for correlator storage
        zero_correls(correls);

        // loop over the smearing combinations (LL, LS, SL, SS)
        for( t_smear_index smear_index = 0; smear_index < fl_a->params.no_smearing_combinations; ++smear_index ) {
          
          // collect the propagators from the two flavours for this smearing combination
          collect_props(prop_a, prop_b, 
                          fl_a->propagators[mass_index_a][smear_index], fl_b->propagators[mass_index_b][smear_index],
                          fl_a->params.n_c*fl_a->params.n_s);
          
          // carry out the contractions, looping over gamma combinations
          for( unsigned int observable = 0; observable < N_observables; ++observable) {
            // pseudo-scalar
            memset(corr_temp, 0, sizeof(double)*2*T);
            if(observable < 16){
              contract_twopoint(corr_temp, current_gindex1[observable], current_gindex2[observable], prop_a, prop_b, fl_a->params.n_c);
            // (pseudo-)vector
            } else {
              for(unsigned int gamma_vector_index = 0; gamma_vector_index < 3; ++gamma_vector_index) {
                memset(corr_temp_vector, 0, sizeof(double)*2*T);
                contract_twopoint(corr_temp_vector, current_gindex1[observable+gamma_vector_index], 
                                    current_gindex2[observable+gamma_vector_index], prop_a, prop_b, fl_a->params.n_c);
                
                for(unsigned int x0=0; x0<T; x0++) {                  
                  corr_temp[2*x0  ] += (current_conf_gamma_sign[(observable-16)/3]*
                                          current_vsign[observable-16+gamma_vector_index]*corr_temp_vector[2*x0  ]);
                  corr_temp[2*x0+1] += (current_conf_gamma_sign[(observable-16)/3]*
                                          current_vsign[observable-16+gamma_vector_index]*corr_temp_vector[2*x0+1]);
                }
              }
            }
            // assign correlator to holding data structure
            correls[observable][smear_index]->set_correlator_array(corr_temp);
          } /* observable */
        } /* smear_index */
        
        // output correlators uses construct_correlator_filename to build a filename
        output_correlators(correls, (*fp_it)->get_name(), fl_a, fl_b, mass_index_a, mass_index_b );
        
      } /* mass_index_b */
    } /* mass_index_a */
  } /* flavour pairings */
  
  deb_printf(0,"# [test_hl_helpers] All contractions complete!\n");
  
  // free correlator storage
  for(vector< vector<correlator*> >::iterator iter = correls.begin(); iter != correls.end(); ++iter){
    while(iter->size() > 0){
      delete iter->back();
      iter->pop_back();
    }
  }
  
  free_global_data_structures();
  free_geometry();
  #ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  #endif
}



/***************************************************************************************/

void init_global_data_structures() {
  alloc_spinor_field(&g_work_spinor_field, VOLUMEPLUSRAND);
    
  /* all memory management is done at the level of the init methods */
  init_gauge_field();
  
  for( vector<flavour*>::iterator it = g_flavours.begin(); it != g_flavours.end(); ++it ) {
    (*it)->init();
  }
  for( vector<flavour_pairing*>::iterator it = g_flavour_pairings.begin(); it != g_flavour_pairings.end(); ++it ) {
    (*it)->init();
  }
}

void free_global_data_structures() {
  if( g_work_spinor_field != NULL ) {
    free(g_work_spinor_field);
  }
  
  /* the destructors take care of any internal memory management */
  free_gauge_field();
  
  while( g_flavour_pairings.size() > 0 ) {
    delete g_flavour_pairings.back();
    g_flavour_pairings.pop_back(); 
  }
  while( g_flavours.size() > 0 ) {
    delete g_flavours.back();
    g_flavours.pop_back();
  }
}

void usage() {
  deb_printf(0, "# [test_libcvcpp] Testing code for the CVC++ addon library for CVC\n");
  deb_printf(0, "# [test_libcvcpp] Usage:   test_libcvcpp [options]\n");
  deb_printf(0, "# [test_libcvcpp] Options: -h, -? this help and exit\n");
  deb_printf(0, "# [test_libcvcpp]          -v verbose [no effect, lots of stdout output anyway]\n");
  deb_printf(0, "# [test_libcvcpp]          -f input filename [default cvc.input]\n");
#ifdef MPI
  MPI_Finalize();
#endif
  exit(0);
}

void process_args( int argc, char **argv ) {
  char c;
  while ((c = getopt(argc, argv, "h?vf")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      input_filename = optarg;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  } 
}

string construct_correlator_filename_create_subdirectory(const string flavour_pairing_name, const flavour* const fl_a, const flavour* const fl_b, const unsigned int mass_index_a, const unsigned int mass_index_b) {
  stringstream rval;
  stringstream dirname;
  
  dirname << flavour_pairing_name << "_";
  dirname << fl_a->params.name << "_";
  dirname << fl_a->params.masses[mass_index_a] << "-";
  dirname << fl_b->params.name << "_";
  dirname << fl_b->params.masses[mass_index_b];
  
  if( g_proc_id == 0 ){
    if( access(dirname.str().c_str(),F_OK) != 0 ) {
      if( mkdir( dirname.str().c_str(), 0700 ) != 0 ) {
        fatal_error(22,"Creation of subdirectory %s failed!\n",dirname.str().c_str() );
      }
    }
  }
  
  // puttogether.sh requires the filename to be $basename.$timeslice(2).$confnum(4)
  // silly, really...
  rval << dirname.str() << "/" << "outprcv.";
  rval << setw(2) << setfill('0') << fl_a->params.source_timeslice << '.';
  rval << setw(4) << setfill('0') << Nconf;
  return rval.str();
}

void collect_props(double** prop_a, double** prop_b, vector<propagator>& props_a, vector<propagator>& props_b, unsigned int no_spin_colour_indices) {
  for( unsigned int spin_colour_index = 0; spin_colour_index < no_spin_colour_indices; ++spin_colour_index ) {
    prop_a[spin_colour_index] = props_a[spin_colour_index].field.mem;
    prop_b[spin_colour_index] = props_b[spin_colour_index].field.mem;
  } /* spin_colour_index */
}

void output_correlators(const vector< vector<correlator*> >& correls, const string flavour_pairing_name, const flavour* const fl_a, const flavour* const fl_b, const unsigned int mass_index_a, const unsigned int mass_index_b){
  unsigned int fwd_corr_ts, bwd_corr_ts;
  unsigned int corr_fwd_array_index;
  unsigned int corr_bwd_array_index;
  double fwd_val, bwd_val;
    
  unsigned int source_timeslice = fl_a->params.source_timeslice;
  
  string correlator_filename( construct_correlator_filename_create_subdirectory(flavour_pairing_name, fl_a, fl_b, mass_index_a, mass_index_b ) );
  
  if(g_proc_id==0){
    FILE* ofs;
    ofs=fopen(correlator_filename.c_str(), "w");
    if( ofs == (FILE*)NULL ) {
      fatal_error(9,"Could not open file %s for writing!\n",correlator_filename.c_str());
    }
  
    // Marcus's format
    //fprintf(ofs, "# %5d%4d%4d%4d%4d%15.8e %s %s_%15.8e %s_%15.8e\tLL,LS,SL,SS\n",
    //  Nconf, T_global, LX_global, LY_global, LZ, g_kappa, flavour_pairing_name.c_str(), fl_a->params.name.c_str(), fl_a->params.masses[mass_index_a], fl_b->params.name.c_str(), fl_b->params.masses[mass_index_b]);
    
    // CMI Format
    fprintf(ofs, "%5d%4d%4d%4d%4d%4d%15.8e\n",
      Nconf, source_timeslice, LX_global, LY_global, LZ_global, T_global, g_kappa);
  
    
    /* the output format is as follows:
     * observable smearing_combination timeslice forward_prop backward_prop
     */
    
    for(unsigned int observable = 0; observable < N_observables; ++observable ) {
      for(unsigned int smear_index = 0; smear_index < fl_a->params.no_smearing_combinations; ++smear_index) {
      
        for(unsigned int x0=0; x0<=T_global/2; x0++) {
          fwd_corr_ts = ( x0+source_timeslice) % T_global;
          bwd_corr_ts = (-x0+source_timeslice+T_global) % T_global;
          corr_fwd_array_index = 2* ( (fwd_corr_ts/T)*T + fwd_corr_ts%T ) + current_isimag[observable%16];
          corr_bwd_array_index = 2* ( (bwd_corr_ts/T)*T + bwd_corr_ts%T ) + current_isimag[observable%16];
          
          fwd_val = correlator_norm*current_isneg[observable]*correls[observable][smear_index]->correlator_array_global[corr_fwd_array_index];
          // for x0 == 0 and x0 == T_global/2, the backward correlator is 0
          bwd_val = (x0 > 0 && x0 < T_global/2) ? 
              correlator_norm*current_isneg[observable]*correls[observable][smear_index]->correlator_array_global[corr_bwd_array_index] 
            : 0;
          
          fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", observable+1, smear_index_to_cmi_int(smear_index), x0, fwd_val, bwd_val);
        } // x0
      } // smear_index
    } // observable
    fclose(ofs);
  } // if(g_proc_id == 0)
}

void zero_correls(vector< vector<correlator*> >& correls) {
  for(vector< vector<correlator*> >::iterator obs_iter = correls.begin(); obs_iter != correls.end(); ++obs_iter) {
    for(vector<correlator*>::iterator smearing_iter = (obs_iter)->begin(); smearing_iter != (obs_iter)->end(); ++smearing_iter) {
      (*smearing_iter)->zero_out();
    }
  }
}
