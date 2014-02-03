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
 
/* The "meson" class is a semi-abstract base class for collecting the arrays
 * of gamma combinations that were historically found in the 
 * 
 * Further, it provides virtual functions for contracting propagators
 * with the configured gamma combinations and finally an output
 * function which deals with creating necessary directories
 * and writing the correlators to disk in the correct format.
 * 
 * The intended usecase for this class is to create a derived class
 * which contains the "actual" arrays defining the correlators as
 * static const class members.
 * 
 * When this derived class is instantiated, the pointers in this base
 * class are set to point to the static class members of the derived class.
 * The contraction and output functions can compute and output
 * the desired set of correlators.
 * 
 * Because the functions are virtual, a derived class may contain
 * specialized implementations of these functions if that proves 
 * necessary.
 * */ 
 
#ifndef _MESON_HPP
#define _MESON_HPP

#include <vector>
#include <string>

using namespace std;

#include "correlator_memory.hpp"

class flavour;
class flavour_pairing;
class correlator;
class propagator;

class meson {
  public:
    meson();
    meson(const meson& i_meson);
    meson(const string& i_name,
          const unsigned int i_N_correlators,
          const unsigned int* i_is_vector_correl,
          const int* i_isimag,
          const double* i_isneg,
          const int* i_gindex1,
          const int* i_gindex2,
          const double* i_vsign,
          const double* i_conf_gamma_sign);
  
    virtual void output_correlators(const vector< vector<correlator*> >* const correls, 
                                    const string& flavour_pairing_name, 
                                    const flavour* const fl_a, const flavour* const fl_b, 
                                    const unsigned int mass_index_a, const unsigned int mass_index_b ); 
                                     
    virtual void collect_props(double** prop_a, double** prop_b, const vector<propagator>& props_a, 
                               const vector<propagator>& props_b, const unsigned int no_spin_colour_indices);
                               
    virtual void do_contractions(const flavour_pairing* const fp,
                                 const unsigned int mass_index_a, const unsigned int mass_index_b );
    
    virtual string construct_correlator_filename_create_subdirectory(
                                  const string& flavour_pairing_name, 
                                  const flavour* const fl_a, const flavour* const fl_b, 
                                  const unsigned int mass_index_a, const unsigned int mass_index_b );
 
    virtual void print_info();
    
    string get_name();
 
  protected:
    bool initialized;
    correlator_memory correl_mem;
    
    unsigned int N_correlators;
    string name;
    const unsigned int* is_vector_correl;
    const int* isimag;
    const double* isneg;
    const int* gindex1;
    const int* gindex2;
    const double* vsign;
    const double* conf_gamma_sign;
};

meson* get_meson_pointer_from_name( const string i_observables_name );
 
#endif
