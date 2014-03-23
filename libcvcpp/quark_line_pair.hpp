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

/* The quark line pair is a pairing of two quark lines (i.e. propagators)
 * which pairs up quark lines defined during program initialization.
 * The pair can then be passed to different "observables" which may, for instance,
 * contract the quark lines.
 * Initialization: It is passed a vector of two strings via 'set_quark_line_names' 
 * which could be, e.g. "up" and "strange".
 * In addition, it must be passed a pointer to a vector of 'quark_line'. The 'init'
 * method must be called when these two conditions are met and it will traverse
 * the 'quark_line_collection' looking for names which match the pairing descriptor.
 * When a name matches, the pointers a and b are set to point to the respective quark lines.
 *  */

#ifndef QUARK_LINE_PAIR_HPP_
#define QUARK_LINE_PAIR_HPP_

#include <string>
#include <vector>

// forward declarations
class meson;
class quark_line;

using namespace std;

class quark_line_pair {
public:
  
  quark_line_pair();
  quark_line_pair( vector<quark_line*>* i_quark_lines_collection );
  quark_line_pair( vector<string>& i_quark_line_names, vector<quark_line*>* i_quark_lines_collection );
  ~quark_line_pair();
  
  void init();
  
  void set_observable_names( const vector<string>& i_observables_names );
  string get_observable_names_string();
  
  void set_quark_line_names( const vector<string>& i_quark_line_names);
  string get_quark_line_names_string();
  
  void set_quark_lines_collection(vector<quark_line*>* i_quark_lines_collection);

  void set_name( string i_name );
  
  // i messed up a little with constness in the "meson" class I think, 
  // so this is a quick fix for a silly problem -Bartek
  string get_name();
  string get_name() const;

  void set_mass_diagonal( const bool& i_mass_diagonal );
  bool is_mass_diagonal();
  
  // when init is called, parts of the data structure must already be set
  // and this method checks whether it's safe to call init
  bool preinit_check();
  // during initialization, the two quark_lines are checked to be consistent
  bool check_consistency();
  
  quark_line* a;
  quark_line* b;
  
  // collection of observable types for this quark_line pairing
  vector<meson*> observables;

private:
  bool initialized;
  bool mass_diagonal;
  string name;
  vector<quark_line*>* quark_lines_collection;
  vector<string> quark_line_names;
  vector<string> observable_names;
  
  void constructor_common();
};

#endif /* QUARK_LINE_PAIR_HPP_ */
