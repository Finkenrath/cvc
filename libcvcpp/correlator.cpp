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

#include <string.h>

#include "fatal_error.h"
#include "correlator.hpp"

correlator::correlator(){
  constructor_common();
  allocate();
}

correlator::correlator(const correlator& i_correlator){
  constructor_common();
  allocate();
}

correlator::correlator(double* i_allreduce_buffer ){
  constructor_common();
  allocate();
  set_allreduce_buffer( i_allreduce_buffer );
}

correlator::~correlator() {
  deallocate();
  constructor_common();
}

void correlator::constructor_common() {
  initialized = false;
  allocated = false;
  allreduce_buffer = NULL;
  correlator_array = NULL;
  correlator_array_global = NULL;
}

void correlator::set_allreduce_buffer( double* i_allreduce_buffer ) {
  allreduce_buffer = i_allreduce_buffer;
  initialized = true;
}

void correlator::set_correlator_array( const double* const i_correlator_array ) {
  if(allocated) {
    memcpy(correlator_array,i_correlator_array,2*T*sizeof(double));
    exchange();
  } else {
    deb_printf(0,"WARNING: [correlator::set_correlator_array] called but correlator memory has not been allocated!\n");
  }
}

void correlator::allocate(){
  if(!allocated){
    if( T == 0 ) {
      fatal_error(11, "ERROR: [correlator::allocate] called but T has not been set!\n");
    }
    correlator_array = (double*)malloc(2*T*sizeof(double));
    if(correlator_array == NULL)
      fatal_error(12,"ERROR: [correlator::allocate] Could not allocate rank-local correlator memory!\n");

#ifdef MPI
    if( T_global == 0 ) {
      fatal_error(13, "ERROR: [correlator::allocate] called but T_global has not been set!\n");
    }
    correlator_array_global = (double*)malloc(2*T_global*sizeof(double));
    if(correlator_array_global == NULL)
      fatal_error(14,"ERROR: [correlator::allocate] Could not allocate MPI-global correlator memory!\n");
#else
    correlator_array_global = correlator_array;
#endif

    allocated = true;
  } else {
    deb_printf(0,"WARNING: [correlator::allocate] called while already allocated, was that intended?\n");
  }
}

void correlator::zero_out(){
  if(allocated && correlator_array != NULL ){
    memset(correlator_array, 0, sizeof(double)*2*T);
#ifdef MPI
    memset(correlator_array_global, 0, sizeof(double)*2*T_global);
#endif
  } else {
    deb_printf(0, "WARNING: [correlator::zero_out] called while not allocated, was that intended?\n");
  }
}

void correlator::deallocate(){
  if(correlator_array != NULL)
    free(correlator_array);
  
#ifdef MPI
  if(correlator_array_global != NULL)
    free(correlator_array_global);
#else
  correlator_array_global = NULL;
#endif
  allocated = false;
}

void correlator::exchange(){
  if(initialized) {
#ifdef MPI
#if (defined PARALLELTX) || (defined PARALLELTXY)
  for(unsigned int ix=0; ix<2*T; ix++) allreduce_buffer[ix] = 0.;
  MPI_Allreduce(correlator_array, allreduce_buffer, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  MPI_Allgather(allreduce_buffer, 2*T, MPI_DOUBLE, correlator_array_global, 2*T, MPI_DOUBLE, g_xs_comm);
#else
  MPI_Gather(correlator_array, 2*T, MPI_DOUBLE, correlator_array_global, 2*T, MPI_DOUBLE, 0, g_cart_grid);
#endif
#else
  // without MPI, correlator_array_global points to correlator_array and no copying is necessary!
#endif
  }
}

