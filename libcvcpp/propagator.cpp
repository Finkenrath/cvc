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

#include <iostream>

#include <string.h>

#include "global.h"
#include "cvc_utils.h"
#include "smearing_techniques.h"
#include "deb_printf.h"
#include "Q_phi.h"
#include "propagator_io.h"
#include "fatal_error.h"
#include "fuzz2.h"

#include "propagator.hpp"

using namespace std;

// initialize static member variables
unsigned int propagator::ref_count = 0;
double* propagator::spinor_mpi_buffer = (double*)NULL;
bool propagator::spinor_mpi_buffer_allocated = false;

propagator::propagator() {
  ++ref_count;
  initialized = false;
}

propagator::propagator(string i_filename, t_smear_index i_smear_index, unsigned int i_scidac_pos, bool i_in_mms_file, t_delocalization_type i_delocalization_type ){
  initialized = false;
  ++ref_count;
  init( i_filename, i_smear_index, i_scidac_pos, i_in_mms_file, i_delocalization_type );
}

propagator::propagator(const propagator& i_propagator){
  ++ref_count;
  initialized = false;    
}

propagator::~propagator(){
  de_init();
}

void propagator::init( string i_filename, t_smear_index i_smear_index, unsigned int i_scidac_pos, bool i_in_mms_file, t_delocalization_type i_delocalization_type ){
  if( !initialized ) {
#ifdef MPI
    if( spinor_mpi_buffer == NULL ) {
      if( VOLUMEPLUSRAND == 0 ) {
        fatal_error(87, "ERROR: [propagator::init] Attempted to allocate memory for spinor_mpi_buffer but VOLUMEPLUSRAND has not been defined yet!\n");
      }
      spinor_mpi_buffer = (double*)malloc( sizeof(double)*24*VOLUMEPLUSRAND );
      if( spinor_mpi_buffer == NULL ) {
        fatal_error(88, "ERROR: [propagator::init] Failed to allocate memory for spinor_mpi_buffer!\n");
      }
      spinor_mpi_buffer_allocated = true;
    }
#endif

    in_mms_file = i_in_mms_file;
    filename = i_filename;
    scidac_pos = i_scidac_pos;
    set_smearing_type(i_smear_index, i_delocalization_type);
    field.allocate();
    initialized = true;
  }
}

void propagator::read_from_file() {
  if( initialized ){
    deb_printf( 1, "[propagator] Attempting to read propagator from SciDAC position %d in file %s!\n",scidac_pos,filename.c_str() );
    /* the cast to (char*) is technically invalid because it should be
     * a pointer to const. However, when using LEMON the filename
     * is passed through to MPI_File_open which, at least for OpenMPI,
     * stupidly calls for a (char*) argument */
    read_lime_spinor(field.mem, (char*)filename.c_str(), scidac_pos);
    post_read_common();
  } else {
    deb_printf( 0, "WARNING: [propagator] read_from_file called but data structure is not initialised!\n" );
    return;
  }
}

void propagator::read_from_memory( const propagator & i_prop ) {
  if( initialized && i_prop.is_initialized() ){
    field.copy( i_prop.field );
    post_read_common();
  } else {
    if( !initialized ) { deb_printf(0,"# [propagator] Destination propagator is not initialized!\n"); }
    if( !i_prop.is_initialized() ) { deb_printf(0," # [propagator] Source propagator is not initialized!\n"); } 
    return;
  }
}

void propagator::post_read_common() {
  if( in_mms_file ) {
    deb_printf(3, "# [propagator::post_read_common] Applying Qf5 with mass %e\n",g_mu);
    memcpy( g_work_spinor_field, field.mem, 24*VOLUME*sizeof(double) );
    xchange_field( g_work_spinor_field );
    Qf5( field.mem, g_work_spinor_field, g_mu );
  }
  if( (smear_bitmask & PROP_SMEARED) == PROP_SMEARED ){
    Jacobi_smear();
  } else if( (smear_bitmask & PROP_FUZZED) == PROP_FUZZED ){
    Fuzz();
  }
}

void propagator::Jacobi_smear() {
  deb_printf(1,"# [propagator::Jacobi_smear] Smearing propagator!\n");
#ifdef MPI
  memcpy( spinor_mpi_buffer, field.mem, 24*VOLUME*sizeof(double) );
  xchange_field_timeslice(spinor_mpi_buffer);
  for(unsigned int smearing_step=0; smearing_step<N_Jacobi; ++smearing_step) {
    Jacobi_Smearing_Step_one(g_gauge_field_f, spinor_mpi_buffer, g_work_spinor_field, kappa_Jacobi);
    xchange_field_timeslice(spinor_mpi_buffer);
  }
  memcpy( field.mem, spinor_mpi_buffer, 24*VOLUME*sizeof(double) );
#else
  for(unsigned int smearing_step=0; smearing_step<N_Jacobi; ++smearing_step) {
    Jacobi_Smearing_Step_one(g_gauge_field_f, field.mem, g_work_spinor_field, kappa_Jacobi);
  }
#endif
}

void propagator::Fuzz() {
  deb_printf(1,"# [propagator::Fuzz] Fuzzing propagator!\n");
#ifdef MPI
  memcpy( spinor_mpi_buffer , field.mem, 24*VOLUME*sizeof(double) );
  xchange_field_timeslice( spinor_mpi_buffer );
  Fuzz_prop3(g_gauge_field_f, spinor_mpi_buffer, g_work_spinor_field, Nlong);
  memcpy( field.mem, spinor_mpi_buffer, 24*VOLUME*sizeof(double) );
#else
  Fuzz_prop3(g_gauge_field_f, field.mem, g_work_spinor_field, Nlong);
#endif
}

void propagator::set_smearing_type(t_smear_index i_smear_index, t_delocalization_type i_delocalization_type){
  smear_bitmask = smear_index_to_bitmask(i_smear_index, i_delocalization_type == DELOCAL_FUZZING ? true : false );
}

void propagator::set_smearing_type(t_smear_bitmask i_smear_bitmask){
  smear_bitmask = i_smear_bitmask;
}

t_smear_bitmask propagator::get_smear_bitmask() const {
  return smear_bitmask;
}

bool propagator::is_initialized() const {
  return initialized;
}

string propagator::get_filename() const {
  return filename;
}

void propagator::de_init() {
  if( ref_count <= 1 ) {
    if( spinor_mpi_buffer != NULL ) {
      free(spinor_mpi_buffer);
      spinor_mpi_buffer_allocated = false;
    }
  }
  if(ref_count != 0) {
    --ref_count;
  }
    
  if( initialized ) {
    field.deallocate();
    initialized = false;
  } else {
    return;
  }
}
