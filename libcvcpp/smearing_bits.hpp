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

/* bits for creating bitmasks describing a smearing combination compactly
 * and in a way that is immediately understandable:
 * ( SOURCE_LOCAL | PROP_SMEARED ), for instance, corresponds to a proagator
 * generated from a local source which has had smearing applied to it
 * 
 * Unfortunately it will not always be possible to use only this notation 
 * for historical reasons. Also, data storage in vectors is a bit tricky 
 * in C++ with arbitrary integers as indices. */

#ifndef SMEARING_BITS_HPP_
#define SMEARING_BITS_HPP_

#include <string>

using namespace std;

typedef enum t_smearing_bits { 
  SOURCE_LOCAL=1,
  SOURCE_SMEARED=2,
  PROP_LOCAL=4,
  PROP_SMEARED=8,
  SOURCE_FUZZED=16,
  PROP_FUZZED=32
} t_smearing_bits;

typedef enum t_delocalization_type {
  DELOCAL_SMEARING,
  DELOCAL_FUZZING
} t_delocalization_type;

typedef unsigned int t_smear_index;
typedef unsigned int t_smear_cmi_int;
typedef unsigned char t_smear_bitmask;
typedef string t_smear_string;

/* to simplify conversions we use the "index" as common ground
 * so we convert everything to index and then simply return the corresponding
 * element of these arrays */
 
extern t_smear_index smear_index_array[4];
extern t_smear_cmi_int smear_cmi_int_array[4];
extern t_smear_bitmask smear_bitmask_array[7];
extern t_smear_string smear_string_array[7];

/* a selection of conversion functions between the different ways of specifying
 * which kind of smearing a given propagator has had applied to it
 * before / after inversion 
 * 
 * There are four types of descriptions:
 * 1) text in the input file: LL, LS, SL, SS, LF, FL and FF (source-sink)
 * 2) the smearing bits above which can be used to form bitmasks
 * 3) the integers 1, 3, 5 and 7 which are used in Chris Micheal's correlator format
 *    and correspond to (1) in this order
 * 4) the loop index in a loop which goes from 0 to 3 inclusive 
 * 
 * There is a complication because the integer format and the loop index in the
 * loops over "smearing combinations" are the same whether fuzzing or smearing
 * is used. So LS and LF both correspond to 3 in a CMI correlator...
 * 
 * Note: After some thought it was realized that doing this with templates would
 * be complicated, repetitive and annoying, so it was decided to do it as implemented.
 * */

t_smear_index smear_bitmask_to_index( t_smear_bitmask i_smear_bitmask );
t_smear_index smear_cmi_int_to_index( t_smear_cmi_int i_smear_cmi_int );
t_smear_index smear_string_to_index( t_smear_string i_smear_string );

t_smear_bitmask smear_index_to_bitmask( t_smear_index i_smear_index, bool fuzz );
t_smear_cmi_int smear_index_to_cmi_int( t_smear_index i_smear_index );
t_smear_string smear_index_to_string( t_smear_index i_smear_index, bool fuzz );

t_smear_cmi_int smear_bitmask_to_cmi_int( t_smear_bitmask i_smear_bitmask );
t_smear_string smear_bitmask_to_string( t_smear_bitmask i_smear_bitmask );

t_smear_bitmask smear_cmi_int_to_bitmask( t_smear_cmi_int i_smear_cmi_int, bool fuzz );
t_smear_string smear_cmi_int_to_string( t_smear_cmi_int i_smear_cmi_int, bool fuzz );

t_smear_bitmask smear_string_to_bitmask( t_smear_string i_smear_string );
t_smear_cmi_int smear_string_to_cmi_int( t_smear_string i_smear_string );

#endif /*SMEARING_BITS_HPP_*/
