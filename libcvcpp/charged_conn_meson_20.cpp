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

/*********************************************************
 * Gamma combinations for 20 charged meson correlators from hl_conn.cc in the "Contractions"
 * code by Marc Wagner and Carsten Urbach.
 *  
 * pion:
 * g5-g5, g5-g0g5, g0g5-g5, g0g5-g0g5, g0-g0, g5-g0, g0-g5, g0g5-g0, g0-g0g5
 * rho:
 * gig0-gig0, gi-gi, gig5-gig5, gig0-gi, gi-gig0, gig0-gig5, gig5-gig0, gi-gig5, gig5-gi
 * a0, b1:
 * 1-1, gig0g5-gig0g5 
 *
 *********************************************************/

#include "charged_conn_meson_20.hpp"

#include "global.h"

// initialize the static const class members

const string charged_conn_meson_20::charged_conn_meson_20_name = "charged_conn_meson_20";

const unsigned int charged_conn_meson_20::charged_conn_meson_20_N_correlators = 20;

const int charged_conn_meson_20::charged_conn_meson_20_isimag[20] = {
                                            0, 0, 0, 0, 0, 1, 1, 1, 1,
                                            0, 0, 0, 0, 0, 1, 1, 1, 1,
                                            0, 0};
                                            
const unsigned int charged_conn_meson_20::charged_conn_meson_20_is_vector_correl[20] = {
                                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            0, 1};                                             
                                        
const double charged_conn_meson_20::charged_conn_meson_20_isneg[20] = {
                          +1., -1., +1., -1., +1., +1., +1., +1., -1.,
                          -1., +1., -1., -1., +1., +1., +1., -1., +1., 
                          +1., -1.};
    
const int charged_conn_meson_20::charged_conn_meson_20_gindex1[40] = {
                   5, 5, 6, 6, 0, 5, 0, 6, 0, 
                   10, 11, 12,   1, 2, 3,   7, 8, 9,   10, 11, 12,   1, 2, 3,
                   10, 11, 12,   7, 8, 9,   1, 2, 3,    7,  8, 9, 
                   4, 
                   13, 14, 15};
                   
const int charged_conn_meson_20::charged_conn_meson_20_gindex2[40] = {
                   5, 6, 5, 6, 0, 0, 5, 0, 6, 
                   10, 11, 12,   1, 2, 3,   7, 8, 9,   1, 2, 3,   10, 11, 12,
                    7,  8,  9,  10, 11, 12,   7, 8, 9,   1, 2, 3, 
                   4, 
                   13, 14, 15};
    
const double charged_conn_meson_20::charged_conn_meson_20_vsign[30] = {
                   1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 
                   1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    
const double charged_conn_meson_20::charged_conn_meson_20_conf_gamma_sign[10] = {1., 1., 1., 1., 1., -1., -1., -1., -1., 1.};

// construct underlying meson base class
charged_conn_meson_20::charged_conn_meson_20() : 
  meson(charged_conn_meson_20_name,
        charged_conn_meson_20_N_correlators, charged_conn_meson_20_is_vector_correl, 
        charged_conn_meson_20_isimag, charged_conn_meson_20_isneg, charged_conn_meson_20_gindex1, 
        charged_conn_meson_20_gindex2, charged_conn_meson_20_vsign, charged_conn_meson_20_conf_gamma_sign)
{
}
