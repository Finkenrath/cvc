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

#ifndef QUARK_LINE_PARAMS_HPP_
#define QUARK_LINE_PARAMS_HPP_

#include <vector>
#include <string>

#include "smearing_bits.hpp"

using namespace std;

typedef enum quark_line_type{
  up_type,
  down_type,
  doublet,
  indeterminate_type } quark_line_type;

typedef struct quark_line_params{
  unsigned int no_masses;
  vector<double> masses;
  double kappa;
  
  bool in_mms_file;
  
  /* using the word 'splitted' for historical reasons as this is what is used in tmLQCD */
  bool splitted_propagator;
  string propagator_dirname;
  string propagator_basename;
  string name;
  quark_line_type type;
  unsigned int n_c;
  unsigned int n_s;
  unsigned int source_timeslice;
  unsigned int no_smearing_combinations;
  t_delocalization_type delocalization_type;
  
  vector<t_smear_bitmask> smearing_combinations;
  
  /* When using the multiple mass solver one could have all light, strange
   * and charm propagators in one file. first_mass_index defines
   * to the first SciDAC record to be loaded from this file.
   * It is also possible that inversions for more masses were done than we
   * would like to contract. In this case only the required masses
   * can be provided and first_mass_index can be used to provide the offset.
   * A more general behaviour supporting a non-consecutive set of 
   * propagators in the file is currently not planned.
   */
  unsigned int first_mass_index;
    
} quark_line_params;

#endif
