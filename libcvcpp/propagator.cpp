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

#include "propagator.hpp"

using namespace std;

propagator::propagator() {
  initialized = false;
}

//propagator::propagator(string i_filename, t_smear_index i_smear_index ){
//  initialized = false;
//  init( i_filename, i_smear_index );
//}

propagator::propagator(string i_filename, t_smear_index i_smear_index, unsigned int i_scidac_pos, bool i_in_mms_file ){
  initialized = false;
  init( i_filename, i_smear_index, i_scidac_pos, i_in_mms_file );
}

propagator::propagator(const propagator& i_propagator){
  initialized = false;    
}

propagator::~propagator(){
  de_init();
}

void propagator::init( string i_filename, t_smear_index i_smear_index, unsigned int i_scidac_pos, bool i_in_mms_file ){
  if( !initialized ) {
    in_mms_file = i_in_mms_file;
    filename = i_filename;
    scidac_pos = i_scidac_pos;
    set_smearing_type(i_smear_index);
    field.allocate();
    initialized = true;
  }
}

//void propagator::init( string i_filename, t_smear_index i_smear_index ){
//  init( i_filename, i_smear_index, 0 , false );
//}

void propagator::read_from_file() {
  if( initialized ){
    /* the cast to (char*) is technically invalid but when using LEMON, the filename
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
  xchange_field(field.mem);
  if( in_mms_file ) {
    deb_printf(3, "# [propagator::post_read_common] Applying Qf5 with mass %e\n",g_mu);
    memcpy( g_work_spinor_field, field.mem, 24*VOLUMEPLUSRAND*sizeof(double) );
    Qf5( field.mem, g_work_spinor_field, g_mu );
  }
  if( (smear_bitmask & PROP_SMEARED) == PROP_SMEARED ){
    Jacobi_smear();
  }    
}

void propagator::Jacobi_smear() {
  deb_printf(1,"# [propagator::Jacobi_smear] Smearing propagator!\n");
  xchange_field_timeslice(field.mem);
  for(unsigned int smearing_step=0; smearing_step<N_Jacobi; ++smearing_step) {
    Jacobi_Smearing_Step_one(g_gauge_field_f, field.mem, g_work_spinor_field, kappa_Jacobi);
    xchange_field_timeslice(field.mem);
  }
}

void propagator::set_smearing_type(t_smear_index i_smear_index){
  smear_bitmask = smear_index_to_bitmask(i_smear_index);
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
  if( initialized ) {
    field.deallocate();
    initialized = false;
  } else {
    return;
  }
}
