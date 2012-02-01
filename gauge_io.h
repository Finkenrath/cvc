/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/
/* $Id: gauge_io.h,v 1.2 2009/02/16 16:52:01 urbach Exp $ */

#ifndef _GAUGE_IO_H
#define _GAUGE_IO_H

#include"dml.h"

int write_lime_gauge_field(char * filename, const double plaq, const int counter, const int prec);
int write_xlf_info(const double plaq, const int counter, char * filename, const int append, char * data_buf);
int write_ildg_format_xml(char *filename, LimeWriter * limewriter, const int prec);
n_uint64_t file_size(FILE *fp);
int read_nersc_gauge_field(double*s, char*filename, double *plaq_val);
int read_nersc_gauge_field_timeslice(double*s, char*filename, int timeslice, uint32_t *checksum);
int read_nersc_gauge_binary_data_3col(FILE*ifs, double*s, DML_Checksum*ans);
int read_nersc_gauge_field_3x3(double*s, char*filename, double *plaq);
#endif
