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

#include <stdio.h>
#include <spi/include/kernel/memory.h>

#include "global.h"
#include "printmem_bgq.h"
#include "deb_printf.h"

void printmem_bgq()
{
  uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;

  Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);

  deb_printf(0,"Allocated heap: %.2f MB, avail. heap: %.2f MB\n", (double)heap/(1024*1024),(double)heapavail/(1024*1024));
  deb_printf(0,"Allocated stack: %.2f MB, avail. stack: %.2f MB\n", (double)stack/(1024*1024), (double)stackavail/(1024*1024));
  deb_printf(0,"Memory: shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n", (double)shared/(1024*1024), (double)persist/(1024*1024), (double)guard/(1024*1024), (double)mmap/(1024*1024));
}

