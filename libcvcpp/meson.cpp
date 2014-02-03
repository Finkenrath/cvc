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

#include <cstring>
#include <iostream> 

#include <sys/stat.h>

#include "global.h"
#include "cvc_utils.h"
#include "deb_printf.h"
#include "fatal_error.h"

#include "flavour.hpp"
#include "flavour_pairing.hpp"
#include "correlator.hpp"
#include "correlator_memory.hpp"
#include "propagator.hpp"

using namespace std;

// derived class headers should be listed here
#include "charged_conn_meson_20.hpp"
#include "neutral_conn_meson_20.hpp"
#include "charged_conn_meson_32.hpp"
#include "neutral_conn_meson_32.hpp"

// derived class constructors should be called here
meson* get_meson_pointer_from_name( const string i_observables_name ) {
  meson* rval;
  if( i_observables_name == "charged_conn_meson_20" ) {
    rval = new charged_conn_meson_20();
  } else if (i_observables_name == "neutral_conn_meson_20" ) {
    rval = new neutral_conn_meson_20();
  } else if ( i_observables_name == "charged_conn_meson_32" ) {
    rval = new charged_conn_meson_32();
  } else if ( i_observables_name == "neutral_conn_meson_32" ) {
    rval = new neutral_conn_meson_32();
  } else {
    deb_printf(0, "WARNING: [meson::get_meson_pointer_from_name] Uknown observable type '%s', returning pointer to uninitialized meson base class!\n",i_observables_name.c_str());
    rval = new meson();
  }
  return rval;
}

meson::meson() {
  initialized = false;
}

meson::meson(const meson& i_meson) {
  initialized = false;
}

meson::meson( const string& i_name,
              const unsigned int i_N_correlators,
              const unsigned int* i_is_vector_correl, 
              const int* i_isimag, 
              const double* i_isneg,
              const int* i_gindex1, 
              const int* i_gindex2, 
              const double* i_vsign, 
              const double* i_conf_gamma_sign)
{
  
  name = i_name;
  N_correlators = i_N_correlators;
  is_vector_correl = i_is_vector_correl;
  isimag = i_isimag;
  isneg = i_isneg;
  gindex1 = i_gindex1;
  gindex2 = i_gindex2;
  vsign = i_vsign;
  conf_gamma_sign = i_conf_gamma_sign;
  initialized = true;
  
}

string meson::get_name() {
  return name;
}

void meson::do_contractions(const flavour_pairing* fp, const unsigned int mass_index_a, const unsigned int mass_index_b) {  
  if(initialized){
    const flavour* const fl_a = fp->a;
    const flavour* const fl_b = fp->b;
    
    deb_printf(1,"# [meson::do_contractions] Doing '%s' contractions for flavours '%s' and '%s', masses %f and %f!\n",
                  name.c_str(), fl_a->params.name.c_str(), fl_b->params.name.c_str(), 
                  fl_a->params.masses[mass_index_a], fl_b->params.masses[mass_index_b]);
    
    // convenience arrays to assemble propagators for contract_twopoint function
    double** prop_a = (double**) malloc( fl_a->params.n_c*fl_a->params.n_s*sizeof(double*) );
    double** prop_b = (double**) malloc( fl_b->params.n_c*fl_b->params.n_s*sizeof(double*) );
    
    if( prop_a == NULL || prop_b == NULL ){
      fatal_error(3,"ERROR: [meson::do_contractions] memory allocation for propagator convenience arrays failed!\n");
    }
    
    // get a pointer to some memory to store our correlators!
    vector< vector<correlator*> >* correls = correl_mem.get_correls_pointer(N_correlators,fl_a->params.no_smearing_combinations);
    double* temp_correl = correl_mem.get_temp_correl_pointer();
    double* temp_vector_correl = correl_mem.get_temp_vector_correl_pointer();
    
    correl_mem.zero_out();
    
    // the slightly inverted looping over smearing combinations first is a result of the way
    // the propagators are stored
    for( t_smear_index smear_index = 0; smear_index < fl_a->params.no_smearing_combinations; ++smear_index) {
      // collect the propagators from the two flavours for this smearing combination
      collect_props(prop_a, prop_b, 
                    fl_a->propagators[mass_index_a][smear_index], fl_b->propagators[mass_index_b][smear_index],
                    fl_a->params.n_c*fl_a->params.n_s);
      
      
      // offset for the gindex arrays
      unsigned int gamma_offset = 0;
      // offset for the vector arrays
      unsigned int vector_offset = 0;
          
      // carry out the contractions, looping over correlator types
      for( unsigned int corr_idx = 0; corr_idx < N_correlators; ++corr_idx) {
        // zero out temporary storage for correlator
        memset(temp_correl, 0, sizeof(double)*2*T);
        
        // (pseudo-)scalar
        if( is_vector_correl[corr_idx] == 0 ){
          contract_twopoint(temp_correl, gindex1[gamma_offset], gindex2[gamma_offset], prop_a, prop_b, fl_a->params.n_c);
          gamma_offset += 1;
        // (pseudo-)vector
        } else {
          for(unsigned int gamma_vector_index = 0; gamma_vector_index < 3; ++gamma_vector_index) {
            // zero out temporary storage for vector correlator component
            memset(temp_vector_correl, 0, sizeof(double)*2*T);
            contract_twopoint(temp_vector_correl, gindex1[gamma_offset], 
                              gindex2[gamma_offset], prop_a, prop_b, fl_a->params.n_c);
            
            gamma_offset += 1;
                
            for(unsigned int x0=0; x0<T; x0++) {                  
              temp_correl[2*x0  ] += (conf_gamma_sign[vector_offset/3]*
                                     vsign[vector_offset]*temp_vector_correl[2*x0  ]);
              temp_correl[2*x0+1] += (conf_gamma_sign[vector_offset/3]*
                                          vsign[vector_offset]*temp_vector_correl[2*x0+1]);
            }
            vector_offset += 1;
          }
        }
      // assign correlator to holding data structure, this also does any MPI exchange if necessary
      // and collects everything at g_proc_id==0
      (*correls)[corr_idx][smear_index]->set_correlator_array(temp_correl);
      } /* for(corr_idx) */
    } /* for(smear_index) */
    
    output_correlators(correls, fp->get_name(), fl_a, fl_b, mass_index_a, mass_index_b );
    
    // free convenience arrays
    free(prop_a);
    free(prop_b);
    
  } /* if(initialized) */
}  

void meson::collect_props(double** prop_a, double** prop_b, const vector<propagator>& props_a, const vector<propagator>& props_b, const unsigned int no_spin_colour_indices) {
  for( unsigned int spin_colour_index = 0; spin_colour_index < no_spin_colour_indices; ++spin_colour_index ) {
    prop_a[spin_colour_index] = props_a[spin_colour_index].field.mem;
    prop_b[spin_colour_index] = props_b[spin_colour_index].field.mem;
  } /* spin_colour_index */
}

void meson::output_correlators(const vector< vector<correlator*> >* const correls, const string& flavour_pairing_name, const flavour* const fl_a, const flavour* const fl_b, const unsigned int mass_index_a, const unsigned int mass_index_b ) {
  if(initialized){
    // this seems to be the default correlator norm for the ETMC toolset (at least hadron)
    double correlator_norm = 1/(2*g_kappa*g_kappa*VOL3*g_nproc_x*g_nproc_y*g_nproc_z);
    
    unsigned int fwd_corr_ts, bwd_corr_ts;
    unsigned int corr_fwd_array_idx;
    unsigned int corr_bwd_array_idx;
    double fwd_val, bwd_val;
    
    unsigned int source_timeslice = fl_a->params.source_timeslice;
  
    string correlator_filename( construct_correlator_filename_create_subdirectory(flavour_pairing_name, fl_a, fl_b, mass_index_a, mass_index_b ) );
  
    if(g_proc_id==0){
      FILE* ofs;
      ofs=fopen(correlator_filename.c_str(), "w");
      if( ofs == (FILE*)NULL ) {
        fatal_error(9,"ERROR: [meson::output_correlators] Could not open file %s for writing!\n",correlator_filename.c_str());
      }
  
      if( name == "charged_conn_meson_32" || name == "neutral_conn_meson_32" ) {
        // Marcus's format header with additions
        fprintf(ofs, "# %5d%4d%4d%4d%4d%15.8e %s %s %s_%15.8e %s_%15.8e\tLL,LS,SL,SS\n",
          Nconf, T_global, LX_global, LY_global, LZ, g_kappa, name.c_str() ,flavour_pairing_name.c_str(), fl_a->params.name.c_str(), fl_a->params.masses[mass_index_a], fl_b->params.name.c_str(), fl_b->params.masses[mass_index_b]);
      } else {
        // CMI Format header
        fprintf(ofs, "%5d%4d%4d%4d%4d%4d%15.8e\n",
          Nconf, source_timeslice, LX_global, LY_global, LZ_global, T_global, g_kappa);
      }
  
    
      /* the output format is as follows:
      * observable smearing_combination timeslice forward_prop backward_prop
      */
    
      for(unsigned int corr_idx = 0; corr_idx < N_correlators; ++corr_idx ) {
        for(unsigned int smear_index = 0; smear_index < fl_a->params.no_smearing_combinations; ++smear_index) {
      
          for(unsigned int x0=0; x0<=T_global/2; x0++) {
            fwd_corr_ts = ( x0+source_timeslice) % T_global;
            bwd_corr_ts = (-x0+source_timeslice+T_global) % T_global;
            corr_fwd_array_idx = 2* ( (fwd_corr_ts/T)*T + fwd_corr_ts%T ) + isimag[corr_idx];
            corr_bwd_array_idx = 2* ( (bwd_corr_ts/T)*T + bwd_corr_ts%T ) + isimag[corr_idx];
          
            fwd_val = correlator_norm*isneg[corr_idx]*(*correls)[corr_idx][smear_index]->correlator_array_global[corr_fwd_array_idx];
            // for x0 == 0 and x0 == T_global/2, the backward correlator is 0
            bwd_val = (x0 > 0 && x0 < T_global/2) ? 
                correlator_norm*isneg[corr_idx]*(*correls)[corr_idx][smear_index]->correlator_array_global[corr_bwd_array_idx] 
              : 0;
          
            fprintf(ofs, "%3d%3d%4d%25.16e%25.16e\n", corr_idx+1, smear_index_to_cmi_int(smear_index), x0, fwd_val, bwd_val);
          } // for(x0)
        } // for(smear_index)
      } // for(corr_idx)
      fclose(ofs);
    } // if(g_proc_id == 0)
  } // if(initialized)
}

string meson::construct_correlator_filename_create_subdirectory(const string& flavour_pairing_name, const flavour* const fl_a, const flavour* const fl_b, const unsigned int mass_index_a, const unsigned int mass_index_b) {
  stringstream rval;
  stringstream dirname;
  
  dirname << flavour_pairing_name << "_";
  dirname << fl_a->params.name << "_";
  dirname << fl_a->params.masses[mass_index_a] << "-";
  dirname << fl_b->params.name << "_";
  dirname << fl_b->params.masses[mass_index_b];
  
  if( g_proc_id == 0 ){
    if( access(dirname.str().c_str(),F_OK) != 0 ) {
      if( mkdir( dirname.str().c_str(), 0700 ) != 0 ) {
        fatal_error(22,"ERROR: [meson::construct_correlator_filename_create_subdirectory]\n Creation of subdirectory %s failed!\n",dirname.str().c_str() );
      }
    }
  }
  
  /* strictly speaking one would have to implement do_contractions and
   * construct_correlator_filename_create_subdirectory methods in the 
   * derived classes, but because do_contractions for these basic types
   * is definitely the same, there is no need to do extra work */
  string basename;
  if( name == "neutral_conn_meson_20" ) {
    basename = "outprcvn.";
  } else if ( name == "charged_conn_meson_32" ) {
    basename = "correl.";
  } else if( name == "neutral_conn_meson_32" ) {
    basename = "ncorrel.";
  } else {
    basename = "outprcv.";
  } 
  
  // puttogether.sh requires the filename to be $basename.$timeslice(2).$confnum(4)
  // silly, really...
  rval << dirname.str() << "/" << basename;
  rval << setw(2) << setfill('0') << fl_a->params.source_timeslice << '.';
  rval << setw(4) << setfill('0') << Nconf;
  return rval.str();
}

void meson::print_info() {
  deb_printf(0,"Observables: %s\n", name.c_str() );
  deb_printf(0,"N_correlators: %u\n", N_correlators);
  deb_printf(0,"is_vector_correl[3]: %d\n",is_vector_correl[3]);  
  deb_printf(0,"isimag[3]: %d\n",isimag[3]);  
  deb_printf(0,"gindex1[3]: %d\n",gindex1[3]);  
  deb_printf(0,"gindex2[3]: %d\n",gindex2[3]);  
  deb_printf(0,"vsign[3]: %f\n",vsign[3]);  
  deb_printf(0,"isneg[3]: %f\n",isneg[3]);  
  deb_printf(0,"conf_gamma_sign[3]: %f\n",conf_gamma_sign[3]);    
}
