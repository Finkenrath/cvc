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

#ifndef FLAVOUR_PARAMS_HPP_
#define FLAVOUR_PARAMS_HPP_

#include <vector>
#include <string>

#include "smearing_bits.hpp"

using namespace std;

typedef enum flavour_type{
  up_type,
  down_type,
  indeterminate_type } flavour_type;

typedef struct flavour_params{
  unsigned int no_masses;
  vector<double> masses;
  double kappa;
  
  bool in_mms_file;
  
  /* using 'splitted' for historical reasons as this is what is used in tmLQCD */
  bool splitted_propagator;
  string propagator_dirname;
  string propagator_basename;
  string name;
  flavour_type type;
  unsigned int n_c;
  unsigned int n_s;
  unsigned int source_timeslice;
  unsigned int no_smearing_combinations;
  
  vector<t_smear_bitmask> smearing_combinations;
  
  /* When using the multiple mass solver one could have all light, strange
   * and charm propagators in one collection. first_mass_index is then set
   * to the first mass index to be loaded.
   * It is also possible that inversions for more masses were done than we
   * would like to contract. In this case only the required masses
   * can be provided and first_mass_index can be used to provide the offset.
   * A more general behaviour where only some mass indices in a consecutive
   * range are used is currently not planned.
   */
  unsigned int first_mass_index;
    
} flavour_params;

#endif
