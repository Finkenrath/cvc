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

/* Gamma combinations for 32 charged connected meson correlators from hl_conn_5.c in the CVC
 * code by Marcus Petschlies.
 * 
 * (pseudo-)scalar: observables 1-16
 * 
 * g5 - g5,	g5   - g0g5,	g0g5 - g5,	g0g5 - g0g5,
 * g0 - g0,	g5   - g0,	g0   - g5,	g0g5 - g0,
 * g0 - g0g5,	1    - 1,	1    - g5,	g5   - 1,
 * 1  - g0g5,	g0g5 - 1,	1    - g0,	g0   - 1
 *
 * (pseudo-)vector: observables 17-32
 * 
 * gig0 - gig0,	gi     - gi,		gig5 - gig5,	gig0   - gi,
 * gi   - gig0,	gig0   - gig5,		gig5 - gig0,	gi     - gig5,
 * gig5 - gi,		gig0g5 - gig0g5,	gig0 - gig0g5,	gig0g5 - gig0,
 * gi   - gig0g5,	gig0g5 - gi,		gig5 - gig0g5,	gig0g5 - gig5
 **************************************************************************************************/

#include "charged_conn_meson_32.hpp"

#include "global.h"

// initialize the static const class members

const string charged_conn_meson_32::charged_conn_meson_32_name = "charged_conn_meson_32";

const unsigned int charged_conn_meson_32::charged_conn_meson_32_N_correlators = 32;

const int charged_conn_meson_32::charged_conn_meson_32_isimag[32] = {
                                                                      0, 0, 0, 0, 
                                                                      0, 1, 1, 1, 
                                                                      1, 0, 1, 1, 
                                                                      1, 1, 0, 0,
                                                   
                                                                      0, 0, 0, 0, 
                                                                      0, 1, 1, 1, 
                                                                      1, 0, 1, 1, 
                                                                      1, 1, 0, 0};
                                            
const unsigned int charged_conn_meson_32::charged_conn_meson_32_is_vector_correl[32] = {
                                                                      0, 0, 0, 0, 
                                                                      0, 0, 0, 0, 
                                                                      0, 0, 0, 0, 
                                                                      0, 0, 0, 0,
                                                      
                                                                      1, 1, 1, 1, 
                                                                      1, 1, 1, 1, 
                                                                      1, 1, 1, 1, 
                                                                      1, 1, 1, 1};                               
                                        
const double charged_conn_meson_32::charged_conn_meson_32_isneg[32] = {
                          +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1.,    
                          +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1.};
    
const int charged_conn_meson_32::charged_conn_meson_32_gindex1[64] = {
                   5, 5, 6, 6, 0, 5, 0, 6, 0, 4, 4, 5, 4, 6, 4, 0,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 10, 11, 12, 1, 2, 3, 10, 11, 12, 7, 8, 9, 1, 2, 3, 7, 8, 9,
                   13, 14, 15, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13, 7, 8, 9, 15, 14, 13};
                   
const int charged_conn_meson_32::charged_conn_meson_32_gindex2[64] = {
                   5, 6, 5, 6, 0, 0, 5, 0, 6, 4, 5, 4, 6, 4, 0, 4,
                   10, 11, 12, 1, 2, 3, 7, 8, 9, 1, 2, 3, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 1, 2, 3,
                   13, 14, 15, 15, 14, 13, 10, 11, 12, 15, 14, 13, 1, 2, 3, 15, 14, 13, 7, 8, 9};
    
const double charged_conn_meson_32::charged_conn_meson_32_vsign[48] = {
                    1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                    1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,   1.,  1., 1.,
                    1.,  1., 1.,   1.,  1., 1.,   1., -1., 1.,   1., -1., 1., 
                    1., -1., 1.,   1., -1., 1.,   1., -1., 1.,   1., -1., 1.};
    
const double charged_conn_meson_32::charged_conn_meson_32_conf_gamma_sign[16] = {
                    1., 1., 1., -1., -1., -1., -1., 1., 1., 1., -1., -1.,  1.,  1., 1., 1.};

// construct underlying meson base class
charged_conn_meson_32::charged_conn_meson_32() : 
  meson(charged_conn_meson_32_name,
        charged_conn_meson_32_N_correlators, charged_conn_meson_32_is_vector_correl, 
        charged_conn_meson_32_isimag, charged_conn_meson_32_isneg, charged_conn_meson_32_gindex1, 
        charged_conn_meson_32_gindex2, charged_conn_meson_32_vsign, charged_conn_meson_32_conf_gamma_sign)
{
}
