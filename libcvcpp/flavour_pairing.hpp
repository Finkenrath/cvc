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

/* The flavour pairing class is a slightly inefficient but necessary way of
 * pairing up flavours for contractions during program initialization. 
 * Initialization: It is passed a vector of two strings via 'set_flavour_names' 
 * which could be, e.g. "up" and "strange".
 * In addition, it must be passed a pointer to a vector of 'flavour'. The 'init'
 * method must be called when these two conditions are met and it will traverse
 * the 'flavour_collection' looking for names which match the pairing descriptor.
 * When a name matches, the pointers a and b are set to point to the respective flavours.
 *  */

#include <vector>
#include <string>

#include "flavour.hpp"
#include "smearing_bits.hpp"
#include "meson.hpp"

using namespace std;

#ifndef _FLAVOUR_PAIRING_HPP
#define _FLAVOUR_PAIRING_HPP

class flavour_pairing {
public:
  
  flavour_pairing();
  flavour_pairing( vector<flavour*>* i_flavours_collection );
  flavour_pairing( vector<string>& i_flavour_names, vector<flavour*>* i_flavours_collection );
  
  void init();
  
  //void set_smearing_combinations( vector<string> i_smearing_combinations_descriptors );
  //string get_smearing_combinations_string();
  
  void set_observable_names( const vector<string>& i_observables_names );
  string get_observable_names_string();
  
  void set_flavour_names(vector<string> i_flavour_names);
  void set_flavours_collection(vector<flavour*>* i_flavours_collection);
  
  string get_flavour_names_string();
  
  void set_name( string i_name );
  string get_name();
  
  // when init is called, parts of the data structure must already be set
  bool preinit_check();
  // during initialization, the two flavours are checked to be consistent
  bool check_consistency();
  
  flavour* a;
  flavour* b;
  
  // collection of observable types for this flavour pairing
  vector<meson*> observables;

private:
  bool initialized;
  string name;
  vector<flavour*>* flavours_collection;
  vector<string> flavour_names;
  vector<string> observable_names;
  
  void constructor_common();
};

#endif /* _FLAVOUR_PAIRING_HPP */
