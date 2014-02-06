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

/* a spinor field object providing a layer of abstraction to memory
 * handling for spinor fields
 * it also provides a wrapper for MPI exchange */

#ifndef SPINOR_FIELD_HPP_
#define SPINOR_FIELD_HPP_

class spinor_field{
public:
  spinor_field();
  spinor_field(unsigned int i_size);
  // the copy constructor is a dummy which DOES NOT copy anything
  // it exists solely for the purpose of allowing resize() in vectors of spinor_field type
  spinor_field(const spinor_field& i_spinor_field);
  ~spinor_field();
  
  void allocate(unsigned int i_size);
  void allocate();
  void deallocate();
  
  bool is_allocated() const; 
  unsigned int get_size() const;
  
  void copy( const spinor_field & i_spinor_field );
  
  double* mem;
  
private:
  bool allocated;
  unsigned int size;
};

#endif /* SPINOR_FIELD_HPP_ */
