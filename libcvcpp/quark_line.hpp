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

/* a slightly abstract representation of a 'quark line' for the purpose of
 * contractions of propagators for the computation of various observables.
 * It provides a mechanism for automatically loading propagators in ILDG
 * format as produced by tmLQCD
 * there exist a number of conditions which affect the filenames and content
 * of propagator files produced by tmLQCD:
 * 
 * 1) are up and down in the same file? (AddDownPropagator option in tmLQCD)
 * 2) are propagators for all 'indices' in the same file? (SplittedPropagator option in tmLQCD)
 * 3) are we dealing with propagators derived from volume sources?
 * 4) was the CGMMS solver used? (excludes 1, may affect 5)
 * 5) if multiple masses are present, do these reside in subdirectories or are
 *    they referenced as part of the filename? (may depend on 5)
 * 6) how many "smearing types" are in use? (none, source smearing only, propagator smearing only, both)
 * 7) how many 'indices' are there? (in the case of spin dilution for instance this is 4 while without dilution we have 12)
 * 8) are these in the unitary setup or OS?
 * 
 * These conditions are taken into account in the "construct_propagator_filename" method
 * and during the "init" stage.
 * Some of the complexity of this is shoved onto the user in the input file, but
 * it allows for maximum flexibility with regards to combinations of the above conditions
 * without too much "centralized" code to actually manage that flexibility. 
 * 
 * All the parameters controlling this behaviour are kept in the "params" struct which
 * is a bit obfuscating but allows the whole parameter set to be passed from one
 * object to another. */

#ifndef QUARK_LINE_HPP_
#define QUARK_LINE_HPP_

#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

#include "deb_printf.h"

#include "propagator.hpp"

using namespace std;

typedef enum t_quark_line_type{
  QL_TYPE_UP,
  QL_TYPE_DOWN,
  QL_TYPE_DOUBLET } t_quark_line_type;

typedef enum t_source_type {
  SOURCE_TYPE_POINT,
  SOURCE_TYPE_TIMESLICE,
  SOURCE_TYPE_VOLUME } t_source_type;
  
typedef struct {
  double mu;
  double mudelta;
} t_quark_mass;

class quark_line{
public:
  quark_line();
  
  void init();
  
   /* using the word 'splitted' for historical reasons as this is what is used in tmLQCD 
    * it means that different 'indices' are in different files, rather than one large file */
  bool splitted_propagator;
  string propagator_dirname;
  string propagator_basename;
  string name;
  
  t_quark_line_type type;
  t_source_type source_type;
  
  unsigned int n_c;
  unsigned int n_s;
  unsigned int source_timeslice;
  unsigned int source_location;
  unsigned int no_masses;
  unsigned int no_smearing_combinations;
  unsigned int no_flavour_combinations;
  t_delocalization_type delocalization_type;
  vector<t_smear_bitmask> delocal_combinations;
  
  double kappa;
  
  bool in_mms_file;
  vector<t_quark_mass> masses;
  
    /* When using the multiple mass solver one could have all light, strange
   * and charm propagators in one file. first_mass_index defines
   * to the first SciDAC record to be loaded from this file for this quark line.
   * It is also possible that inversions for more masses were done than we
   * would like to contract. In this case only the required masses
   * can be provided and first_mass_index can be used to provide the offset.
   * Unfortunately this does not include the more general behaviour of selecting
   * a non-consecutive set of propagators in the file.
   */
  unsigned int first_mass_index;  
  
  /* vector order: mass, flavour matrix, smearing, spin_colour_index
   * the flavour matrix indices are in the following order (source->sink):
   *      down->down, down->up, up->down, up->up
   * for unfathomable historical reasons, probably because the kaon
   * is the ground state when contracting a light quark line with a
   * non-degenerate non-flavour-diagonal strange-charm quark line */
  vector< vector< vector < vector<propagator> > > > propagators;

private:
  bool initialized;
  string construct_propagator_filename(const unsigned int i_mass_ctr, const unsigned i_index );
};

#endif /* QUARK_LINE_HPP_ */
  
