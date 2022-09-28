#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 2.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date
#//////////////////////////////////////////////////////////////////////////////

# --------------------
# programs

CC        = gcc
CXX       = g++
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib


# --------------------
# flags

# Use -fPIC to make shared (.so) and static (.a) library;
# can be commented out if making only static library.
FPIC      = -fPIC

CFLAGS    = -O3 $(FPIC) -DADD_ -Wall -fopenmp -DCL_TARGET_OPENCL_VERSION=300
FFLAGS    = -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument -DCL_TARGET_OPENCL_VERSION=300
F90FLAGS  = -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input -DCL_TARGET_OPENCL_VERSION=300
LDFLAGS   =     $(FPIC)              -fopenmp -DCL_TARGET_OPENCL_VERSION=300

# C++11 (gcc >= 4.7) is not required, but has benefits like atomic operations
CXXFLAGS := $(CFLAGS) -std=c++11 -DCL_TARGET_OPENCL_VERSION=300
CFLAGS   += -std=c99


# --------------------
# libraries

# gcc with OpenBLAS (includes LAPACK)
LIB       = -lopenblas

LIB      += -lclblast -lOpenCL


# --------------------
# directories

# define library directories preferably in your environment, or here.
# OPENBLASDIR ?= /usr/lib/x86_64-linux-gnu/openblas/
OPENBLASDIR ?= /opt/OpenBLAS
clBLAS      ?= /usr/local

-include make.check-openblas
-include make.check-clblas

LIBDIR    = -L$(clBLAS)/lib \
            -L$(OPENBLASDIR)/lib \
            -L/usr/lib/x86_64-linux-gnu/mwv207

INC       = -I/usr/include/mwv207 -I$(clBLAS)/include 
