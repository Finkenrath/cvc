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

#include <string>

#include "propagator_io.h"

#include "spinor_field.hpp"

#include "smearing_bits.hpp"

using namespace std;

#ifndef _PROPAGATOR_HPP
#define _PROPAGATOR_HPP

class propagator{
public:
  propagator();
  propagator(string i_filename, t_smear_index i_smear_index, unsigned int i_scidac_pos, bool i_in_mms_file );
  // the copy constructor is a dummy function which DOES NOT actually copy anything
  // it exists solely for the purpose of being able to resize() vectors of propagators
  propagator(const propagator& i_propagator);
  ~propagator();
  
  void init( string i_filename, t_smear_index i_smearing_type, unsigned int scidac_pos, bool i_in_mms_file );
  void read_from_memory( const propagator & i_prop );
  void read_from_file();
  void de_init();
  
  void set_smearing_type(t_smear_index i_smear_index);
  void set_smearing_type(t_smear_bitmask i_smear_bitmask);

  t_smear_bitmask get_smear_bitmask() const;
  bool is_initialized() const;
  string get_filename() const;
  
  spinor_field field;
  
private:
  void Jacobi_smear();
  void post_read_common();

  bool in_mms_file;
  t_smear_bitmask smear_bitmask;
  bool initialized;
  unsigned int scidac_pos;
  string filename;
};

#endif /* _PROPAGATOR_HPP */
