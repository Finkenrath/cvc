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

#include "global.h"
#include "cvc_utils.h"
#include "deb_printf.h"

spinor_field::spinor_field() {
  allocated = false;
}

spinor_field::spinor_field(unsigned int i_size) {
  allocated = false;
  allocate(i_size);
}

spinor_field::~spinor_field() {
  deallocate();
}

void spinor_field::deallocate() {
  if(allocated){
    free(mem);
    allocated = false;
  }
}

void spinor_field::allocate() {
  if(!allocated && VOLUMEPLUSRAND != 0){
    allocate(VOLUMEPLUSRAND);
  } else {
    if(VOLUMEPLUSRAND == 0){
      deb_printf(0,"# [spinor_field] VOLUMEPLUSRAND has not been initialized! Cannot allocate.\n");
    }
    if(allocated) {
      deb_printf(0,"# [spinor_field] allocate() called despite being already allocated, doing nothing!\n");
    }
  }
}

void spinor_field::allocate(unsigned int i_size) {
  if(!allocated){
    deb_printf(5,"# [spinor_field::allocate] Not yet allocated, allocating %d lattice sites\n",i_size);
    size = i_size;
    alloc_spinor_field(&mem,size);
    allocated = true;
  } else {
    if(size == i_size){
      deb_printf(5,"# [spinor field::allocate] Already allocated and size matches, doing nothing.\n");
      return;
    } else {
      deb_printf(5,"# [spinor field::allocate] Already allocated, but size doesn't match, reallocating!\n");
      size = i_size;
      deallocate();
      allocate(size);
    }
  }
}

unsigned int spinor_field::get_size() const {
  return size;
}

bool spinor_field::is_allocated() const{
  return allocated;
}

void spinor_field::copy( const spinor_field & i_spinor_field ){
  size_t bytes = sizeof(double)*24*size;
  if( allocated && i_spinor_field.is_allocated() ){
    if( size == i_spinor_field.get_size() ) {
      memcpy( (void*) mem, (void*) i_spinor_field.mem, bytes );
      deb_printf(5,"# [spinor_field::copy] Copying %d lattice sites!\n", size);
    } else {
      deb_printf(0,"# [spinor_field::copy] Source and destination have differing sizes, cannot copy!\n");
    }
  } else {
    if(!allocated){ 
      deb_printf(0,"# [spinor_field::copy] Destination is not allocated, cannot copy!\n"); 
    }
    if(!i_spinor_field.is_allocated()) {
      deb_printf(0,"# [spinor_field::copy] Source is not allocated, cannot copy!\n"); 
    }
  }
}
