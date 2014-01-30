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
 
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "flavour_pairing.hpp"
 
#include "global.h"
#include "fatal_error.h"
 
using namespace std;
 
flavour_pairing::flavour_pairing() {
  constructor_common();
}

flavour_pairing::flavour_pairing( vector<string>& i_flavour_names, vector<flavour*>* i_flavours_collection ) {
  constructor_common();
  flavour_names = i_flavour_names;
  flavours_collection = i_flavours_collection;
}

flavour_pairing::flavour_pairing( vector<flavour*>* i_flavours_collection ) {
  constructor_common();
  flavours_collection = i_flavours_collection;
}

void flavour_pairing::constructor_common() {
  initialized = false;
  a = NULL;
  b = NULL;
  flavours_collection = NULL;
}

void flavour_pairing::init() {
  if( !initialized ){
    
    // check if names, flavour collection and smearing combinations have been properly set
    // after instantiation
    if( preinit_check() ) {
      /* traverse the flavour collection and assign the correct ones to the pairing
      * by matching names */ 
      for( vector<flavour*>::iterator it = (*flavours_collection).begin(); it != (*flavours_collection).end(); ++it ){
        if( (*it)->params.name == flavour_names[0] ) {
          a = (*it);
        }
        if( (*it)->params.name == flavour_names[1] ) {
          b = (*it);
        }
      }
      /* check_consistency will fatal_error the program if anything fails */
      initialized = check_consistency();
      
    } else { // preinit_check
      fatal_error(1, "ERROR: [flavour_pairing::init] For flavour pairing %s, preinit_check failed!\n",name.c_str() );
    }

  } else { // initialized
    deb_printf(0, "WARNING: [flavour_pairing::init] Init called for flavour pairing %s, but it is already initialized! Was that intended?\n", name.c_str() );
  }
}

bool flavour_pairing::preinit_check() {
  return( !flavour_names.empty() && !smearing_combinations.empty() && !name.empty() );
}

bool flavour_pairing::check_consistency() {
  if( a == NULL ) {
    fatal_error(1,"ERROR: [flavour_pairing::check_consistency] Flavour a in flavour pairing %s could not be found!\n", name.c_str() );
  }
  if( b == NULL ) {
    fatal_error(1,"ERROR: [flavour_pairing::check_consistency] Flavour b in flavour pairing %s could not be found!\n", name.c_str() );
  }
  if( a->params.no_smearing_combinations != b->params.no_smearing_combinations ) {
    fatal_error(1,"ERROR: [flavour_pairing::check_consistency] For flavour pairing %s, smearing_combinations do not match for flavours %s and %s!\n", name.c_str(), a->params.name.c_str(), b->params.name.c_str() );
  }
  if( a->params.n_s != b->params.n_s ) {
    fatal_error(1,"ERROR: [flavour_pairing::check_consistency] For flavour pairing %s, n_s do not match for flavours %s and %s!\n", name.c_str(), a->params.name.c_str(), b->params.name.c_str() );    
  }
  if( a->params.n_c != b->params.n_c ) {
    fatal_error(1,"ERROR: [flavour_pairing::check_consistency] For flavour pairing %s, n_c do not match for flavours %s and %s!\n", name.c_str(), a->params.name.c_str(), b->params.name.c_str() );        
  }
  if( a->params.source_timeslice != b->params.source_timeslice ) {
    fatal_error(1, "ERROR: [flavour_pairing::check_consistency] For flavour pairing %s, the source timeslice differs between flavours %s and %s!\n", name.c_str(), a->params.name.c_str(), b->params.name.c_str() );
  }
  
  // any failures will result in program termination so we simply return true if we reach this point!
  return true;
}
  

void flavour_pairing::set_smearing_combinations( vector<string> i_smearing_combinations_descriptors ){
  if( !smearing_combinations.empty() ){ 
    smearing_combinations.clear();
  }
  for( vector<string>::iterator it = i_smearing_combinations_descriptors.begin(); it != i_smearing_combinations_descriptors.end(); ++it ) {
    smearing_combinations.push_back( smear_string_to_bitmask( *it ) );
  }
}

void flavour_pairing::set_flavour_names( vector<string> i_flavour_names ){
  flavour_names = i_flavour_names;
}

string flavour_pairing::get_flavour_names_string() {
  stringstream flavour_names_string;
  for( vector<string>::iterator it = flavour_names.begin(); it != flavour_names.end(); ++it ) {
    flavour_names_string << *it << " ";
  }
  
  return flavour_names_string.str();
}

string flavour_pairing::get_smearing_combinations_string() {
  stringstream smearing_descriptor_string;
  for( vector<t_smear_bitmask>::iterator it = smearing_combinations.begin(); it != smearing_combinations.end(); ++it ) {
    smearing_descriptor_string << smear_bitmask_to_string( *it ) << " ";
  }
  
  return smearing_descriptor_string.str();
}

void flavour_pairing::set_name( string i_name ) {
  name = i_name;
}

string flavour_pairing::get_name() {
  return name;
}