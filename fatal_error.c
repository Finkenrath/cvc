/***********************************************************************
 *
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
 
 #ifdef MPI
 #include <mpi.h>
 #endif
 
 #include <stdio.h>
 #include <stdarg.h>
 
 #include "global.h"
 #include "fatal_error.h"
 
 void fatal_error(const unsigned int signal, const char * format, ...) {
  if(g_proc_id == 0) {
    va_list va;
    va_start(va,format);
    vprintf(format,va);
    va_end(va);
  }
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, signal);
  MPI_Finalize();
#endif
  exit(signal);
}
 
