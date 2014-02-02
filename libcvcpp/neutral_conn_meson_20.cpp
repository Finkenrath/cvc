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

/* Gamma combinations for 20 neutral connected meson correlators from hl_conn.cc in the "Contractions"
 * code by Marc Wagner and Carsten Urbach.
 *  
 * pion:
 * g5-g5, g5-g0g5, g0g5-g5, g0g5-g0g5, 1-1, g5-1, 1-g5, g0g5-1, 1-g0g5
 * rho:
 * gig0-gig0, gi-gi, gig0g5-gig0g5, gig0-gi, gi-gig0, gig0-gig0g5, gig0g5-gig0, gi-gig0g5, gig0g5-gi
 * a0, b1:
 * g0-g0, gig5-gig5 
 * */

#include "neutral_conn_meson_20.hpp"

#include "global.h"

// initialize the static const class members

const string neutral_conn_meson_20::neutral_conn_meson_20_name = "neutral_conn_meson_20";

const unsigned int neutral_conn_meson_20::neutral_conn_meson_20_N_correlators = 20;

const bool neutral_conn_meson_20::neutral_conn_meson_20_is_mass_diagonal = true;

const int neutral_conn_meson_20::neutral_conn_meson_20_isimag[20] = {
                                            0, 0, 0, 0, 0, 1, 1, 1, 1,
                                            0, 0, 0, 0, 0, 1, 1, 1, 1,
                                            0, 0};
                                            
const unsigned int neutral_conn_meson_20::neutral_conn_meson_20_is_vector_correl[20] = {
                                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            0, 1};                                             
                                        
const double neutral_conn_meson_20::neutral_conn_meson_20_isneg[20] = {
                          +1., -1., +1., -1., +1., +1., +1., +1., -1.,
                          -1., +1., -1., -1., +1., +1., +1., -1., +1., 
                          +1., -1.};
    
const int neutral_conn_meson_20::neutral_conn_meson_20_gindex1[40] = {
                  5, 5, 6, 6, 4, 5, 4, 6, 4,
                  10, 11, 12, 1, 2, 3, 13, 14, 15, 10, 11, 12, 1, 2, 3, 10, 11, 12, 13, 14, 15, 1, 2, 3, 13, 14, 15, 
                  0, 7, 8, 9};
                   
const int neutral_conn_meson_20::neutral_conn_meson_20_gindex2[40] = {
                  5, 6, 5, 6, 4, 4, 5, 4, 6, 
                  10, 11, 12, 1, 2, 3, 13, 14, 15, 1, 2, 3, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 1, 2, 3, 
                  0, 7, 8, 9};
    
const double neutral_conn_meson_20::neutral_conn_meson_20_vsign[31] = {
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1.,                                                  
                  1., 1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1.};
    
const double neutral_conn_meson_20::neutral_conn_meson_20_conf_gamma_sign[9] = {1., 1., 1., 1., 1., -1., -1., -1., -1.};

// construct underlying meson base class
neutral_conn_meson_20::neutral_conn_meson_20() : 
  meson(neutral_conn_meson_20_name,
        neutral_conn_meson_20_N_correlators, neutral_conn_meson_20_is_mass_diagonal, 
        neutral_conn_meson_20_is_vector_correl, 
        neutral_conn_meson_20_isimag, neutral_conn_meson_20_isneg, neutral_conn_meson_20_gindex1, 
        neutral_conn_meson_20_gindex2, neutral_conn_meson_20_vsign, neutral_conn_meson_20_conf_gamma_sign)
{
}
