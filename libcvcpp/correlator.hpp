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

#ifndef _CORRELATOR_HPP
#define _CORRELATOR_HPP

#include <string>
#include "global.h"

using namespace std;

class correlator{
  public:
    correlator();
    correlator( double* i_allreduce_buffer );
    
    ~correlator();
    
    void set_correlator_array( const double* const i_correlator_array );
    void set_allreduce_buffer( double* i_allreduce_buffer );
    void zero_out();
    
    double* correlator_array_global;
    double* correlator_array;
    
  private:
    void exchange();
    void constructor_common();
    
    void deallocate();
    void allocate();
    
    double* allreduce_buffer;
    double* buffer_array;
    
    bool initialized;
    bool allocated;
};

#endif /* _CORRELATOR_HPP */
