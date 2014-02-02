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

/* Attempt at an intelligent objectification of the concept of a correlator
 * with semi-automatic MPI parallelization.
 * 
 * WARNINGS: 
 * 1) this object is not thread-safe by default because "allreduce_buffer"
 *    could well be shared between many instances
 * 
 * PLANNED FEATURES:
 * 1) at some point in the future one could imagine this class to posess
 *    an 'output' method which would be resposible for writing out to disk,
 *    supplied with a few function arguments to avoid unnecessary
 *    fopen/fclose iterations
 */ 

#include <string>
#include "global.h"

using namespace std;

#ifndef _CORRELATOR_HPP
#define _CORRELATOR_HPP

class correlator{
  public:
    correlator();
    correlator( double* i_allreduce_buffer );
    // the copy constructor is a dummy, it does NOT COPY anything
    // it is only available to allow the resizing of vectors of correlators
    correlator(const correlator& i_correlator);
    
    ~correlator();
    
    void set_correlator_array( const double* const i_correlator_array );
    void set_allreduce_buffer( double* i_allreduce_buffer );
    void zero_out();
    
    /* correlator_array_global will hold the complete correlators
     * from 0 to T_global/2; this is what g_cart_id == 0 will
     * write to disk! */
    double* correlator_array_global;
    double* correlator_array;
    
  private:
    void exchange();
    void constructor_common();
    
    void deallocate();
    void allocate();
    
    double* allreduce_buffer;
    
    bool initialized;
    bool allocated;
};

#endif /* _CORRELATOR_HPP */
