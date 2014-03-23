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

#include <string.h>

#include "quark_line.hpp"
#include "global.h"
#include "Q_phi.h"
#include "cvc_utils.h"

quark_line::quark_line() {
  initialized = false;
}

void quark_line::init() {
  if( initialized ) {
    propagators.clear();
    initialized = false;
    init();
  } else {
    deb_printf(1,"# [quark_line::init] Initialising quark line %s.\n",name.c_str());
    
    // enforce some consistency conditions
    // force smearing_combinations to 1 if set to local only
    if( delocalization_type == DELOCAL_NONE && no_smearing_combinations > 1 ) {
      deb_printf(0,"# OVERRIDE: [quark_line::init] Delocalization 'none' selected for quark line %s, forcing no_smearing_combinations=1!\n",name.c_str());
      no_smearing_combinations = 1;
    }
    // force in_mms_file to false if we are dealing with a doublet
    if( type == QL_TYPE_DOUBLET && in_mms_file ) {
      deb_printf(0,"# OVERRIDE: [quark_line::init] MMS does not exist for the doublet quark, forcing in_mss_file=no!\n");
      in_mms_file=false;
    }      
    // we currently only support flavour non-diagonal doublets
    if( type == QL_TYPE_DOUBLET && no_flavour_combinations != 4 ) {
      deb_printf(0,"# OVERRIDE: [quark_line::init] Currently only support for non-flavour-diagonal doublet quark line, forcing no_flavour_combinations=4!\n");
      no_flavour_combinations = 4;
    } else if ( (type == QL_TYPE_DOWN || type == QL_TYPE_UP) && no_flavour_combinations != 1 ) {
      deb_printf(0,"# OVERRIDE: [quark_line::init] Up/down type quark line, forcing no_flavour_combinations=1!\n");
      no_flavour_combinations = 1;
    }
        
    propagators.resize(no_masses);
    for( unsigned int mass_ctr = 0; mass_ctr < no_masses; ++mass_ctr ) {
      propagators[mass_ctr].resize(no_flavour_combinations);
      for( unsigned int flavour_ctr = 0; flavour_ctr < no_flavour_combinations; ++flavour_ctr ) {
        propagators[mass_ctr][flavour_ctr].resize(no_smearing_combinations);
        for( unsigned int smear_ctr = 0; smear_ctr < no_smearing_combinations; ++smear_ctr ) {
          propagators[mass_ctr][flavour_ctr][smear_ctr].resize(n_c*n_s);
          for( unsigned int index = 0; index < n_c*n_s; ++index ) {
            deb_printf(1,"# [quark_line::init] Initialising propagator for quark line %s, mass %e (mudelta %e), flavour component %u, delocalalization combination %s, index %u\n", 
              name.c_str(),
              masses[mass_ctr].mu, masses[mass_ctr].mudelta,
              flavour_ctr,
              smear_index_to_string( smear_ctr, (bool)(delocalization_type == DELOCAL_FUZZING) ).c_str(),
              index);
          
            unsigned int mass_index = first_mass_index+mass_ctr;
          
            /* propagator files can contain multiple propagators. either just two
             * when up and down are stored in the same file or 2*2*n_c*n_s
             * when all indices are stored in the same file 
             * (one factor of two comes from considering local and smeared sources
             * which are stored as consecutive indices)
             * in the case of MMS files, there are no explicit down propagators
             * scidac_offset keeps track of which propagator to read out 
             * in the case of doublet propagators there will be two propagators 
             * per index if the Dirac matrix is flavour-diagonal (currently unsupported)
             * or four propagators if it isn't (the latter is the case for
             * non-degenerate twisted mass). Of course in that case there are no explicit
             * "down" propagators as these are included anyway */
            
            // example: propagators from local sources, indices:   00, 01, 02, 03 (spin dilution)
            //          propagators from smeared (fuzzed) sources, indices: 04, 05, 06, 07
            unsigned int delocal_index = ( smear_ctr < 2 ? 0 : 1 );
             
            unsigned int scidac_offset = 0;
            if(splitted_propagator == false) {
              if( type != QL_TYPE_DOUBLET ) {
                if(in_mms_file == true) {
                  scidac_offset = delocal_index*n_c*n_s + index;
                } else {
                  scidac_offset = 2*delocal_index*n_c*n_s + 2*index;
                }
              } else {
                scidac_offset = 4*delocal_index*n_c*n_s + 4*index;
              }
            }
            
            /* up and down propagators are stored in the same file, unless
             * we are dealing with MMS files, in which case down propagators
             * are not computed explicitly */
            if(type == QL_TYPE_DOWN && in_mms_file != true){
              scidac_offset += 1;
            }
            
            /* for the doublet we need to take into account which component
             * of the flavour matrix we are dealing with */
            if( type == QL_TYPE_DOUBLET ) {
              scidac_offset += flavour_ctr;
            }
            
            /* the complication of scidac_offset is reflected also in filename_index */
            unsigned int filename_index = delocal_index*n_c*n_s + index;
            
            string filename = construct_propagator_filename( mass_index, filename_index );
            
            propagators[mass_ctr][flavour_ctr][smear_ctr][index].init( filename, smear_ctr, scidac_offset, in_mms_file, delocalization_type );
            
            /* propagator::read_from_x() applies Qf5 if we're dealing with an MMS propagator
             * this currently uses g_kappa and g_mu, so we have to set g_mu and g_kappa accordingly */
            double mu_save = g_mu;
            double kappa_save = g_kappa;
            if ( in_mms_file ) {
              g_mu = (type == QL_TYPE_DOWN ? -1 : 1) * masses[mass_ctr].mu;
              g_kappa = kappa;
            }
            
            /* for smearing indices 1 and 3 we can reuse data and just copy the previous propagator */
            if( smear_ctr % 2 == 0 ) {
              propagators[mass_ctr][flavour_ctr][smear_ctr][index].read_from_file();
            } else {
              propagators[mass_ctr][flavour_ctr][smear_ctr][index].read_from_memory( propagators[mass_ctr][flavour_ctr][smear_ctr-1][index] );
            }
            
            // reset g_kappa and g_mu
            if( in_mms_file ) {
              g_kappa = kappa_save;
              g_mu = mu_save;
            }
            
          } /* for(index) */
        } /* for(smear_ctr) */
      } /* for(flavour_ctr) */
    } /* for(mass_ctr) */
    initialized = true;
  } /* if(initialized) */
}

/* these are the types of filenames that exist with the zero-padded width of the numbers given by (#)
 * 
 * splitted:
 * a) basename.confnum(4).t_source(2).ix(2).[h]inverted
 * b) basename.operator_id(2).confnum(4).t_source(2).ix(2).cgmms.mass(2).inverted
 * 
 * combined:
 * c) basename.confnum(4).t_source(2).[h]inverted
 * d) basename.operator_id(2).confnum(4).t_source(2).cgmms.mass(2).inverted
 * 
 * volume source:
 * e) basename.confnum(4).sample(5).[h]inverted
 * f) basename.operator_id(2).confnum(4).t_source(2).mass(2).inverted
 * 
 * these are further specified by a subdirectory in which propagators may be stored
 * as well as a further subdirectory in case multiple masses are provided 
 * as non-MMS propagators
 * 
 * so a full filename could be, for example:
 * 
 * light/light_02/source.0235.23.04.inverted (a)
 * 
 * or equivalently in an mms file
 * 
 * light/source.00.0235.23.04.cgmms.02.inverted (b)
 * 
 * or for doublets:
 * 
 * doublet/doublet_03/source.0235.23.04.hinverted (a)
 * 
 * It must be noted that in the case of doublet inversions, the propagator file
 * contains either 1 or 4 propagators depending on whether volume sources have
 * been used or not.
 *
 */

string quark_line::construct_propagator_filename( const unsigned int i_mass_ctr, const unsigned int i_index ) {
  stringstream filename;
  
  // TODO: add support for volume sources?
  //          -> need two introduce two categories of filenames
  
  filename << setfill('0');
  
  // propagators stored in subdirectory
  if(propagator_dirname != string("") ) {
    filename << propagator_dirname << "/";
  }
  
  // multiple masses not in mms file but in subdirectories
  if(no_masses > 1 && in_mms_file == false) {
    if( propagator_dirname != string("") ) {
      filename << propagator_dirname << '_' << setw(2) << i_mass_ctr << "/";
    } else {
      filename << setw(2) << i_mass_ctr << "/";
    }
  }
  filename << propagator_basename << ".";
  
  if(in_mms_file){
    // TODO: remove from tmLQCD, generalize in tmLQCD or generalize here
    filename << setw(2) << 0 << "."; // operator_id, usually 0
  } 
  
  filename << setw(4) << Nconf << ".";
  
  filename << setw(2) << source_timeslice << ".";
  
  if(splitted_propagator){
    filename << setw(2) << i_index << ".";
  }
  
  if( in_mms_file ){
    filename << "cgmms." << setw(2) << i_mass_ctr << ".";
  }
  
  if( type == QL_TYPE_DOUBLET ) {
    filename << "hinverted";
  } else {
    filename << "inverted";
  }
  
  return filename.str();
}
