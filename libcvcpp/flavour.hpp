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

/* a slightly abstract representation of a quark 'flavour' for the purpose of
 * contractions of propagators for extraction of meson masses, for example
 * it provides a mechanism for automatically loading propagators in ILDG
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

#ifndef _FLAVOUR_HPP
#define _FLAVOUR_HPP

#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

#include "deb_printf.h"

#include "flavour_params.hpp"
#include "propagator.hpp"

class flavour{
public:
  flavour();
  flavour( flavour_params i_params );
  void init();
  void init( flavour_params i_params );
  string construct_propagator_filename(const unsigned int i_mass_ctr, const unsigned i_index );
    
  flavour_params params;  
  
  // index order: mass, smearing, spin_colour_index
  vector< vector < vector<propagator> > > propagators;

private:
  bool initialized;
};

#endif /*_FLAVOUR_HPP*/
  
