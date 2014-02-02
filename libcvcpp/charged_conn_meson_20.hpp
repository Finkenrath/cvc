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
 
/* See meson.hpp for a description of what this class and its static class members
* represent */

#include "meson.hpp"
 
#ifndef _CHARGED_CONN_MESON_20_HPP
#define _CHARGED_CONN_MESON_20_HPP

/* Gamma combinations for 20 charged connected meson correlators from hl_conn.cc in the "Contractions"
 * code by Marc Wagner and Carsten Urbach.
 *  
 * pion:
 * g5-g5, g5-g0g5, g0g5-g5, g0g5-g0g5, g0-g0, g5-g0, g0-g5, g0g5-g0, g0-g0g5
 * rho:
 * gig0-gig0, gi-gi, gig5-gig5, gig0-gi, gi-gig0, gig0-gig5, gig5-gig0, gi-gig5, gig5-gi
 * a0, b1:
 * 1-1, gig0g5-gig0g5 */

class charged_conn_meson_20 : public meson 
{
  public: 
    charged_conn_meson_20();
                          
  protected:
    static const string charged_conn_meson_20_name;
    static const unsigned int charged_conn_meson_20_N_correlators;
    static const bool charged_conn_meson_20_is_mass_diagonal;
    static const unsigned int charged_conn_meson_20_is_vector_correl[20];
    static const int charged_conn_meson_20_isimag[20];
    static const int charged_conn_meson_20_gindex1[40];
    static const int charged_conn_meson_20_gindex2[40];
    static const double charged_conn_meson_20_isneg[20];
    static const double charged_conn_meson_20_vsign[31];
    static const double charged_conn_meson_20_conf_gamma_sign[9];
};

#endif // _CHARGED_CONN_MESON_20_HPP
