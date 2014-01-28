/***********************************************************************
 *
 * Copyright (C) 2012, 2013 Marcus Petschlies 
 * Copyright (C)       2013 Bartosz Kostrzewa
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
 
 /* initialization function for global gauge fields, including fuzzing */

#include <stdio.h>
#include <string.h>

#include "global.h"
#include "cvc_utils.h"
#include "smearing_techniques.h"
#include "fuzz.h"
#include "fuzz2.h"
#include "gettime.h"
#include "io.h"

void init_gauge_field() {
  double ratime, retime, plaq;
  char filename[300];
  
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [init_gauge_field] reading gauge field from file %s\n", filename);
  ratime = gettime();
  if(strcmp(gaugefilename_prefix,"identity")==0) {
    if(g_cart_id==0) fprintf(stdout, "\n# [init_gauge_field] initializing unit matrices\n");
    for(unsigned int ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  } else {
    read_lime_gauge_field_doubleprec(filename);
  }
  retime = gettime();
  if(g_cart_id==0) fprintf(stdout, "# [init_gauge_field] time for reading gauge field: %e seconds\n", retime-ratime);

  ratime=gettime();
  xchange_gauge();
  retime=gettime();
  if(g_cart_id==0) fprintf(stdout, "# [init_gauge_field] time for exchanging gauge field: %e seconds\n", retime-ratime);

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [init_gauge_field] measured plaquette value: %25.16e\n", plaq);


  /* prepare smeared / fuzzed gauge field g_gauge_field_f */
  if( (N_Jacobi>0) || (Nlong>0) ) {
    /* temporary memory for one gauge field timeslice (allocated below) */
    double* gauge_field_timeslice = (double*)NULL;
    double* gauge_field_timeslice_old = (double*)NULL;

    if(g_cart_id==0) fprintf(stdout, "# [init_gauge_field] apply APE smearing of gauge field with parameters:\n"\
                                     "# N_ape = %d\n# alpha_ape = %f\n", N_ape, alpha_ape);
                                     
    ratime=gettime();

#if !( (defined PARALLELTX) || (defined PARALLELTXY) )
    alloc_gauge_field(&g_gauge_field_f, VOLUME);
    if( (gauge_field_timeslice = (double*)malloc(72*VOL3*sizeof(double))) == (double*)NULL || 
        (gauge_field_timeslice_old = (double*)malloc(72*VOL3*sizeof(double))) == (double*)NULL ) {
      fprintf(stderr, "Error, could not allocate mem for gauge_field_timeslice/_old\n");
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 3);
      MPI_Finalize();
#endif
      exit(2);
    }
    for(unsigned int x0=0; x0<T; x0++) {
      deb_printf(3,"# [init_gauge_field] APE smearing gauge field timeslice %d\n",x0);
      memcpy((void*)gauge_field_timeslice, (void*)(g_gauge_field+_GGI(g_ipt[x0][0][0][0],0)), 72*VOL3*sizeof(double));
      for(unsigned int smearing_step=0; smearing_step<N_ape; smearing_step++) {
        APE_Smearing_Step_Timeslice_noalloc(gauge_field_timeslice, gauge_field_timeslice_old, alpha_ape);
      }
      if(Nlong > -1) {
        fuzzed_links_Timeslice(g_gauge_field_f, gauge_field_timeslice, Nlong, x0);
      } else {
        memcpy((void*)(g_gauge_field_f+_GGI(g_ipt[x0][0][0][0],0)), (void*)gauge_field_timeslice, 72*VOL3*sizeof(double));
      }
    }
    free(gauge_field_timeslice);
    free(gauge_field_timeslice_old);
#else
    alloc_gauge_field(&g_gauge_field_f, VOLUMEPLUSRAND);
    for(unsigned int smearing_step=0; smearing_step<N_ape; smearing_step++) {
      APE_Smearing_Step_noalloc(g_gauge_field, g_gauge_field_f, alpha_ape);
      xchange_gauge_field_timeslice(g_gauge_field);
    }

    alloc_gauge_field(&g_gauge_field_f, VOLUMEPLUSRAND);

    if(Nlong > 0) {
      fuzzed_links2(g_gauge_field_f, g_gauge_field, Nlong);
    } else {
      memcpy((void*)g_gauge_field_f, (void*)g_gauge_field, 72*VOLUMEPLUSRAND*sizeof(double));
    }
    xchange_gauge_field(g_gauge_field_f);

    if(strcmp(gaugefilename_prefix,"identity")==0) {
      if(g_cart_id==0) fprintf(stdout, "\n# [init_gauge_field] re-initializing unit matrices\n");
      for(unsigned int ix=0;ix<VOLUME;ix++) {
        _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
      }
    } else {
      read_lime_gauge_field_doubleprec(filename);
    }
    xchange_gauge();
#endif

    retime = gettime();
    if(g_cart_id==0) fprintf(stdout, "# [init_gauge_field] time for APE smearing gauge field: %e seconds\n", retime-ratime);
  }
}

void free_gauge_field() {
  if( (N_Jacobi > 0) || (Nlong > 0) ) {
    if(g_gauge_field_f != NULL) {
      free(g_gauge_field_f);
    }
  }
  if(g_gauge_field != NULL) {
    free(g_gauge_field);
  }
}
