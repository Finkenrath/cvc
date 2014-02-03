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

#include "correlator_memory.hpp"

#include "correlator.hpp"

#include "global.h"
#include "fatal_error.h"


// define static members
bool correlator_memory::initialized = false;
unsigned int correlator_memory::ref_count = 0;
double* correlator_memory::temp_correl=(double*)NULL;
double* correlator_memory::temp_vector_correl=(double*)NULL;
double* correlator_memory::allreduce_buffer=(double*)NULL;
vector< vector<correlator*> > correlator_memory::correls;

correlator_memory::correlator_memory() {
  ++ref_count;
  if(!initialized) {
    init();
  }
}

correlator_memory::correlator_memory(const correlator_memory& i_correlator_memory) {
  ++ref_count;
  if(!initialized){
    init();
  }
}

correlator_memory::~correlator_memory() {
  // <= 1 to catch the odd case where memory was initialized but counter was not
  // incremented despite there being an instance to destroy
  if( initialized && ref_count <= 1 ) {
    de_init();
  }
  if(ref_count != 0 ) {
    --ref_count;
  }
}

void correlator_memory::init() {
  if( T <= 0 ) {
    fatal_error(1,"ERROR: [correlator_memory::init] Time extent T = %d!\n",T);
  }
  
  // we only allow initialization when an instance of the class exists because
  if(ref_count > 0) {
    temp_correl = (double*)malloc(sizeof(double)*2*T);
    temp_vector_correl = (double*)malloc(sizeof(double)*2*T);
#ifdef MPI
    allreduce_buffer = (double*)malloc(sizeof(double)*2*T);
    if( allreduce_buffer == NULL ) {
      fatal_error(1,"ERROR: [correlator_memory::ctor] Failed to allocate allreduce_buffer!\n");
    }
#endif
    if( temp_correl == NULL || temp_vector_correl == NULL ) {
      fatal_error(1,"ERROR: [correlator_memory::ctor] Failed to allocated temp_correl or temp_vector_correl!\n");
    }
    initialized = true;
  } else {
    deb_printf(0,"WARNING: [correlator_memory::init] called but there are no class instances! No memory allocated!\n");
  }
}

void correlator_memory::de_init() {
  // free auxiliary memory
  if(temp_correl != NULL)
    free(temp_correl);
  if(temp_vector_correl != NULL)
    free(temp_vector_correl);
  if(allreduce_buffer != NULL)
    free(allreduce_buffer);
    
  // free actual correlator storage because this is the last reference and we can assume 
  // that we won't have another opportunity to clean up later!
  if( !correls.empty() ) {
    for(vector< vector<correlator*> >::iterator iter = correls.begin(); iter != correls.end(); ++iter){
      while(iter->size() > 0){
        delete iter->back();
        iter->pop_back();
      }
    }
  }
    
  initialized = false;  
}

// we return the address of the private "correls" member, but first we check
// if it fulfills the callers requirements

vector< vector<correlator*> >* 
correlator_memory::get_correls_pointer(const unsigned int N_correlators, const unsigned int no_smearing_combinations) {
  if(!initialized){
    init();
  }
  
  if( initialized ) {
    if( correls.size() < N_correlators ) {
      correls.resize(N_correlators);
    }
    // check if all elements can hold all smearing combinations
    for(vector<vector <correlator*> >::iterator iter = correls.begin(); iter != correls.end(); ++iter) {
      while( iter->size() < no_smearing_combinations ) {
        iter->push_back( new correlator(allreduce_buffer) );
      }
    }
    return &correls;
  } else {
    fatal_error(0,"ERROR: [correlator_memory::get_correls_pointer] called but initialization failed. Something must have gone wrong!\n");
    // no return because we just terminated the program!
  }
}

double* correlator_memory::get_temp_correl_pointer() {
  if(!initialized){
    init();
  }
  return temp_correl;
}

double* correlator_memory::get_temp_vector_correl_pointer() {
  if(!initialized){
    init();
  }
  return temp_vector_correl;  
}

void correlator_memory::zero_out() {
  for(vector< vector<correlator*> >::iterator obs_iter = correls.begin(); obs_iter != correls.end(); ++obs_iter) {
    for(vector<correlator*>::iterator smearing_iter = obs_iter->begin(); smearing_iter != obs_iter->end(); ++smearing_iter) {
      (*smearing_iter)->zero_out();
    }
  }  
}

void correlator_memory::print_info() {
  deb_printf(0,"temp_correl %lu \n temp_vector_correl %lu \n allreduce_buffer %lu \n correls %lu \n ref_count %d \n",
              temp_correl, temp_vector_correl, allreduce_buffer, &correls, ref_count );
}
