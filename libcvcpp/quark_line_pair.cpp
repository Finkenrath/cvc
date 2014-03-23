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
#include <sstream>
 
#include "global.h"
#include "fatal_error.h"

#include "quark_line_pair.hpp"
#include "quark_line.hpp"
#include "meson.hpp"
 
using namespace std;
 
quark_line_pair::quark_line_pair() {
  constructor_common();
}

quark_line_pair::quark_line_pair( vector<string>& i_quark_line_names, vector<quark_line*>* i_quark_lines_collection ) {
  constructor_common();
  quark_line_names = i_quark_line_names;
  quark_lines_collection = i_quark_lines_collection;
}

quark_line_pair::quark_line_pair( vector<quark_line*>* i_quark_lines_collection ) {
  constructor_common();
  quark_lines_collection = i_quark_lines_collection;
}

quark_line_pair::~quark_line_pair() {
  // free all allocated "meson" objects, freeing correlator memory with the last one
  while( observables.size() > 0 ) {
    delete observables.back();
    observables.pop_back();
  }
}

void quark_line_pair::constructor_common() {
  initialized = false;
  mass_diagonal = false;
  a = NULL;
  b = NULL;
  quark_lines_collection = NULL;
}

void quark_line_pair::init() {
  if( !initialized ){
    
    // check if names, observables and quark_line collection have been properly set
    // after instantiation
    if( preinit_check() ) {
      /* traverse the quark_line collection and assign the correct ones to the pairing
      * by matching names */ 
      for( vector<quark_line*>::iterator it = (*quark_lines_collection).begin(); it != (*quark_lines_collection).end(); ++it ){
        if( (*it)->name == quark_line_names[0] ) {
          a = (*it);
        }
        if( (*it)->name == quark_line_names[1] ) {
          b = (*it);
        }
      }
      
      // initialize observables for this quark_line pairing
      for( vector<string>::const_iterator obs_name_iter = observable_names.begin(); obs_name_iter != observable_names.end(); ++obs_name_iter) {
        observables.push_back( get_meson_pointer_from_name( *obs_name_iter ) );
      }
      
      /* check_consistency will fatal_error the program if anything fails */
      initialized = check_consistency();
      
    } else { // preinit_check
      fatal_error(1, "ERROR: [quark_line_pair::init] For quark line pair %s, preinit_check failed!\n",name.c_str() );
    }

  } else { // initialized
    deb_printf(0, "WARNING: [quark_line_pair::init] Init called for quark line pair %s, but it is already initialized! Was that intended?\n", name.c_str() );
  }
}

bool quark_line_pair::preinit_check() {
  return( !quark_line_names.empty() && !observable_names.empty() && !name.empty() );
}

bool quark_line_pair::check_consistency() {
  if( a == NULL ) {
    fatal_error(50,"ERROR: [quark_line_pair::check_consistency] Quark line a (%s) in pair %s could not be found! Maybe you misspelled the name in the definition of the pair?\n", quark_line_names[0].c_str() , name.c_str() );
  }
  if( b == NULL ) {
    fatal_error(51,"ERROR: [quark_line_pair::check_consistency] Quark line b (%s) in pair %s could not be found! Maybe you misspelled the name in the definition of the pair?\n", quark_line_names[1].c_str(), name.c_str() );
  }
  if( a->no_smearing_combinations != b->no_smearing_combinations ) {
    fatal_error(52,"ERROR: [quark_line_pair::check_consistency] For pair %s, smearing_combinations do not match for quark lines %s and %s!\n", name.c_str(), a->name.c_str(), b->name.c_str() );
  }
  if( a->n_s != b->n_s ) {
    fatal_error(53,"ERROR: [quark_line_pair::check_consistency] For pair %s, n_s do not match for quark lines %s and %s!\n", name.c_str(), a->name.c_str(), b->name.c_str() );    
  }
  if( a->n_c != b->n_c ) {
    fatal_error(54,"ERROR: [quark_line_pair::check_consistency] For pair %s, n_c do not match for quark lines %s and %s!\n", name.c_str(), a->name.c_str(), b->name.c_str() );        
  }
  if( a->source_timeslice != b->source_timeslice ) {
    fatal_error(55, "ERROR: [quark_line_pair::check_consistency] For pair %s, the source timeslice differs between quark lines %s and %s!\n", name.c_str(), a->name.c_str(), b->name.c_str() );
  }
  
  // any failures will result in program termination so we simply return true if we reach this point!
  return true;
}

void quark_line_pair::set_observable_names( const vector<string>& i_observable_names ) {
  observable_names = i_observable_names;
}

string quark_line_pair::get_observable_names_string() {
  stringstream rval;
  for( vector<string>::iterator obs_name_iter = observable_names.begin(); obs_name_iter != observable_names.end(); ++obs_name_iter ) {
    rval << *obs_name_iter << " ";
  }
  return rval.str();
}

void quark_line_pair::set_quark_line_names( const vector<string>& i_quark_line_names ){
  quark_line_names = i_quark_line_names;
}

string quark_line_pair::get_quark_line_names_string() {
  stringstream quark_line_names_string;
  for( vector<string>::iterator it = quark_line_names.begin(); it != quark_line_names.end(); ++it ) {
    quark_line_names_string << *it << " ";
  }
  
  return quark_line_names_string.str();
}

void quark_line_pair::set_name( string i_name ) {
  name = i_name;
}

string quark_line_pair::get_name() {
  return name;
}

string quark_line_pair::get_name() const {
  return name;
}

bool quark_line_pair::is_mass_diagonal() {
  return mass_diagonal;
}

void quark_line_pair::set_mass_diagonal( const bool& i_mass_diagonal ) {
  mass_diagonal = i_mass_diagonal;
}
  
