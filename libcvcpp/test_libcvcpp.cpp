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

#include "quark_line.hpp"
#include "quark_line_pair.hpp"
#include "quark_line_params.hpp"
#include "correlator.hpp"
#include "charged_conn_meson_20.hpp"
#include "meson.hpp"

using namespace std;

void free_global_data_structures();
void init_global_data_structures();

void process_args(int argc, char **argv);
void usage();

int verbose = 0;
string input_filename = "cvc.input";

int main(int argc, char **argv) {  
#ifdef MPI
  MPI_Status status;
  MPI_Init(&argc, &argv);
#endif  

  process_args(argc,argv);
  // the input file defines quark_lines and quark_line combinations
  // and the relevant data structures are initialized below
  read_input_parser(input_filename.c_str()); 

  mpi_init(argc,argv);
  if(init_geometry() != 0) {
    fatal_error(1, "ERROR: init_geometry failed!\n");
  }
  
  geometry();
  
  // the initialization functions for quark_line and quark_line_pair
  // take care of memory management, reading propagators and smearing
  init_global_data_structures();
  
  for( vector<quark_line_pair*>::iterator fp_it = g_quark_line_pairs.begin(); fp_it != g_quark_line_pairs.end(); ++fp_it ) {
    // convenience variables for the quark_line objects
    quark_line* fl_a = (*fp_it)->a;
    quark_line* fl_b = (*fp_it)->b;
    
    // loop over the observables defined for this quark_line pairing
    
    for( vector<meson*>::const_iterator obs_it = (*fp_it)->observables.begin(); obs_it != (*fp_it)->observables.end(); ++obs_it ) {
      
      for( unsigned int mass_index_a = 0; mass_index_a < fl_a->params.no_masses ; ++mass_index_a ) {
        for( unsigned int mass_index_b = 0; mass_index_b < fl_b->params.no_masses ; ++mass_index_b ) {
          
          /* if the two quark_lines are the same, we only do the mass diagonal case
           * we do the same if pairing has been defined to be mass-diagonal by
           * the user */
          if ( fl_a == fl_b || (*fp_it)->is_mass_diagonal() == true ) {
            /* since the quark_lines are the same, the mass indices will have the same ranges
             * and we can use this as a criterion for staying on the mass diagonal
             * the same is hopefully true if the user has set the relevant flag */
            if( mass_index_a != mass_index_b ) {
              continue;
            }
            if( fl_a->params.masses[mass_index_a] != fl_b->params.masses[mass_index_b] ) {
              deb_printf(0,"WARNING: mass-diagonal contraction for '%s', but the masses differ!\n \
quark_line '%s' mass: %f, quark_line '%s' mass: %f\n", (*fp_it)->get_name().c_str(), 
                    fl_a->params.name.c_str(), fl_a->params.masses[mass_index_a], 
                    fl_b->params.name.c_str(), fl_b->params.masses[mass_index_b]); 
            }
          }
        
        // carry out the contractions for the given set of observables
        (*obs_it)->do_contractions( *fp_it, mass_index_a, mass_index_b );
        
        } /* mass_index_b */
      } /* mass_index_a */
    } /* for(obs_iter) */
  } /* quark_line pairings */
  
  deb_printf(0,"# [test_hl_helpers] All contractions complete!\n");
    
  free_global_data_structures();
  free_geometry();
  #ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  #endif
}

void init_global_data_structures() {
  alloc_spinor_field(&g_work_spinor_field, VOLUMEPLUSRAND);
    
  /* all memory management is done at the level of the init methods */
  init_gauge_field();
  
  for( vector<quark_line*>::iterator it = g_quark_lines.begin(); it != g_quark_lines.end(); ++it ) {
    (*it)->init();
  }
  for( vector<quark_line_pair*>::iterator it = g_quark_line_pairs.begin(); it != g_quark_line_pairs.end(); ++it ) {
    (*it)->init();
  }
}

void free_global_data_structures() {
  if( g_work_spinor_field != NULL ) {
    free(g_work_spinor_field);
  }
  
  /* the destructors take care of any internal memory management */
  free_gauge_field();
  
  while( g_quark_line_pairs.size() > 0 ) {
    delete g_quark_line_pairs.back();
    g_quark_line_pairs.pop_back(); 
  }
  while( g_quark_lines.size() > 0 ) {
    delete g_quark_lines.back();
    g_quark_lines.pop_back();
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
  int c;
  while ( (c = getopt (argc, argv, "vhf:")) != -1 ) {
    switch (c) {
      case 'v':
        verbose = 1;
        break;
      case 'f':
        input_filename = optarg;
        break;
      case 'h':
      default:
        usage();
        break;
    }
  }
}
