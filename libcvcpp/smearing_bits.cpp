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
 
#include "smearing_bits.hpp"
#include "deb_printf.h"

t_smear_index smear_index_array[4] = { 0, 1, 2, 3 };
t_smear_cmi_int smear_cmi_int_array[4] = { 1, 3, 5, 7 };
t_smear_bitmask smear_bitmask_array[7] = { ( PROP_LOCAL | SOURCE_LOCAL ),
                                  ( PROP_SMEARED | SOURCE_LOCAL ),
                                  ( PROP_LOCAL | SOURCE_SMEARED ),
                                  ( PROP_SMEARED | SOURCE_SMEARED ),
                                  ( PROP_FUZZED | SOURCE_LOCAL ),
                                  ( PROP_LOCAL | SOURCE_FUZZED ),
                                  ( PROP_FUZZED | SOURCE_FUZZED ) };

t_smear_string smear_string_array[7] = { "LL", "LS", "SL", "SS", "LF", "FL", "FF" };

unsigned int smear_bitmask_to_index( unsigned char i_smear_bitmask ) {
  switch(i_smear_bitmask){
    case ( SOURCE_LOCAL | PROP_LOCAL ):
      return 0;
      break;
    case ( SOURCE_LOCAL | PROP_SMEARED ):
    case ( SOURCE_LOCAL | PROP_FUZZED ):
      return 1;
      break;
    case ( SOURCE_SMEARED | PROP_LOCAL ):
    case ( SOURCE_FUZZED | PROP_LOCAL ):
      return 2;
      break;
    case ( SOURCE_SMEARED | PROP_SMEARED ):
    case ( SOURCE_FUZZED | PROP_FUZZED ):
      return 3;
      break;
    default:
      deb_printf(0," # [smear_bitmask_to_index] called with invalid bitmask %#x \n", i_smear_bitmask);
      return 254;
      break;
  }  
}

unsigned int smear_cmi_int_to_index( unsigned int i_smear_cmi_int ) {
  switch(i_smear_cmi_int){
    case 1:
      return 0;
      break;
    case 3:
      return 1;
      break;
    case 5:
      return 2;
      break;
    case 7:
      return 3;
      break;
    default:
      deb_printf(0," # [smear_cmi_int_to_index] called with invalid cmi integer %u \n", i_smear_cmi_int);
      return 254;
      break;
  }  
}

unsigned int  smear_string_to_index( string i_smear_string ) {
  if( i_smear_string == "LL" ) {
    return 0;
  } else if ( i_smear_string == "LS" || i_smear_string == "LF" ) {
    return 1;
  } else if ( i_smear_string == "SL" || i_smear_string == "FL" ) {
    return 2;
  } else if ( i_smear_string == "SS" || i_smear_string == "FF" ) {
    return 3;
  } else {
    deb_printf(0, " # [smear_string_to_index] called with invalid string %s \n", i_smear_string.c_str() );
    return 254;
  }
}

unsigned int smear_bitmask_to_cmi_int( unsigned char i_smear_bitmask ) {
  unsigned int temp_idx = smear_bitmask_to_index( i_smear_bitmask );
  if( temp_idx <= 3 ) {
    return( smear_cmi_int_array[temp_idx] ); 
  } else {
    // panic
    return 0;
  }
}

string smear_bitmask_to_string( unsigned char i_smear_bitmask ) {
  unsigned int temp_idx = smear_bitmask_to_index( i_smear_bitmask );
  unsigned int fuzz_offset = 0;
  // for the LL case, we do not need an offset!
  if( i_smear_bitmask != ( PROP_LOCAL | SOURCE_LOCAL ) &&
      ( (i_smear_bitmask & PROP_FUZZED) == PROP_FUZZED || (i_smear_bitmask & SOURCE_FUZZED) == SOURCE_FUZZED) ) {
    fuzz_offset = 3;
  }
  if( temp_idx <= 3 ) {
    return( smear_string_array[temp_idx+fuzz_offset] );
  } else {
    //panic
    string empty;
    return empty;
  }
}

unsigned char smear_cmi_int_to_bitmask( unsigned int i_smear_cmi_int, bool fuzz ) {
  unsigned int temp_idx = smear_cmi_int_to_index( i_smear_cmi_int );
  unsigned int fuzz_offset = 0;
  // for the local-local case, no offset!
  if( i_smear_cmi_int != 1 ) {
    fuzz_offset = fuzz ? 3 : 0;
  } 
  if( temp_idx <= 3 ) {
    return( smear_bitmask_array[temp_idx+fuzz_offset] );
  } else {
    // panic
    return 0;
  }
}

string smear_cmi_int_to_string( unsigned int i_smear_cmi_int, bool fuzz ) {
  unsigned int temp_idx = smear_cmi_int_to_index( i_smear_cmi_int );
  unsigned int fuzz_offset = 0;
  // for the local-local case, no offset!
  if( i_smear_cmi_int != 1 ) {
    fuzz_offset = fuzz ? 3 : 0;
  } 
  if( temp_idx <= 3 ) {
    return( smear_string_array[temp_idx+fuzz_offset] );
  } else {
    //panic 
    string empty;
    return empty;
  }
}

unsigned char smear_string_to_bitmask( string i_smear_string ) {
  unsigned int temp_idx = smear_string_to_index( i_smear_string );
  unsigned int fuzz_offset = 0;
  if( i_smear_string.find("F") != string::npos ) {
    fuzz_offset = 3;
  }
  if( temp_idx <= 3 ) {
    return( smear_bitmask_array[temp_idx+fuzz_offset] );
  } else {
    // panic
    return 0;
  }
}

unsigned int  smear_string_to_cmi_int( string i_smear_string ) {
  unsigned int temp_idx = smear_string_to_index( i_smear_string );
  if( temp_idx <= 3 ) {
    return( smear_cmi_int_array[temp_idx] );
  } else {
    // panic
    return 0;
  }
}

unsigned char smear_index_to_bitmask( unsigned int i_smear_index, bool fuzz ) {
  unsigned int fuzz_offset = 0;
  // for the local-local case, no offset!
  if( i_smear_index != 0 ) {
    fuzz_offset = fuzz ? 3 : 0;
  }  
  if( i_smear_index <= 3 ) {
    return( smear_bitmask_array[i_smear_index+fuzz_offset] );
  } else {
    // panic
    return 0;
  }
}

unsigned int smear_index_to_cmi_int( unsigned int i_smear_index ) {
  if( i_smear_index <= 3 ) {
    return( smear_cmi_int_array[i_smear_index] );
  } else {
    // panic
    return 0;
  }
}

string smear_index_to_string( unsigned int i_smear_index, bool fuzz ) {
  unsigned int fuzz_offset = 0;
  // for the local-local case, no offset!
  if( i_smear_index != 0 ) {
    fuzz_offset = fuzz ? 3 : 0;
  }
  if( i_smear_index <= 3 ) {
    return( smear_string_array[i_smear_index+fuzz_offset] );
  } else {
    // panic
    string empty;
    return empty;
  }
}
