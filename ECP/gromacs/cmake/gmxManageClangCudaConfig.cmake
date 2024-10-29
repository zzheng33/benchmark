#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2017- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

function (gmx_test_clang_cuda_support)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "Clang is required with GMX_CLANG_CUDA=ON!")
    endif()

    # NOTE: we'd ideally like to use a compile check here, but the link-stage
    # fails as the clang invocation generated seems to not handle well some
    # (GPU code) in the object file generated during compilation.
    # SET(CMAKE_REQUIRED_FLAGS ${FLAGS})
    # SET(CMAKE_REQUIRED_LIBRARIES ${LIBS})
    # CHECK_CXX_SOURCE_COMPILES("int main() { int c; cudaGetDeviceCount(&c); return 0; }" _CLANG_CUDA_COMPILES)
endfunction ()

if (GMX_CUDA_TARGET_COMPUTE)
    message(WARNING "Values passed in GMX_CUDA_TARGET_COMPUTE will be ignored; clang will by default include PTX in the binary.")
endif()

# Clang 17 supports CUDA SDK 12.1, which GROMACS requires,
# so no need to check for earlier versions. Clang 18 supports
# up to 12.3 and clang 19 supports up to 12.5
set(_cuda_version_warning "")
if ((CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.1) AND (CUDAToolkit_VERSION VERSION_LESS_EQUAL 12.5))
    # supported
elseif ((CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 18.1) AND (CUDAToolkit_VERSION VERSION_LESS_EQUAL 12.3))
    # supported
elseif ((CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17.0) AND (CUDAToolkit_VERSION VERSION_LESS_EQUAL 12.1))
    # supported
elseif (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
    set(_cuda_version_warning "officially incompatible")
else (CUDAToolkit_VERSION VERSION_GREATER 12.5)
    # We don't know the future; so far Clang 19 states that CUDA 7.0-12.5 are supported.
    set(_cuda_version_warning "possibly incompatible")
endif()
if(NOT CUDA_CLANG_WARNING_DISPLAYED STREQUAL _cuda_version_warning)
    message(NOTICE "Using ${_cuda_version_warning} version of CUDA ${CUDAToolkit_VERSION} "
      "with Clang ${CMAKE_CXX_COMPILER_VERSION}.")
    message(NOTICE "If Clang fails to recognize CUDA version, consider creating doing "
      "`echo \"CUDA Version ${CUDAToolkit_VERSION}\" | sudo tee \"${CUDAToolkit_TARGET_DIR}/version.txt\"`")
endif()
set(CUDA_CLANG_WARNING_DISPLAYED "${_cuda_version_warning}" CACHE INTERNAL
    "Don't warn about this Clang CUDA compatibility issue again" FORCE)
if(CUDA_CLANG_WARNING_DISPLAYED)
    list(APPEND _CUDA_CLANG_FLAGS "-Wno-unknown-cuda-version")
endif()

if (GMX_CUDA_TARGET_SM)
    set(_CUDA_CLANG_GENCODE_FLAGS)
    set(_target_sm_list ${GMX_CUDA_TARGET_SM})
    foreach(_target ${_target_sm_list})
        list(APPEND _CUDA_CLANG_GENCODE_FLAGS "${_target};")
    endforeach()
else()
    list(APPEND _CUDA_CLANG_GENCODE_FLAGS "50;")
    list(APPEND _CUDA_CLANG_GENCODE_FLAGS "52;")
    list(APPEND _CUDA_CLANG_GENCODE_FLAGS "60;")
    list(APPEND _CUDA_CLANG_GENCODE_FLAGS "61;")
    list(APPEND _CUDA_CLANG_GENCODE_FLAGS "70;")
    list(APPEND _CUDA_CLANG_GENCODE_FLAGS "75;")
    list(APPEND _CUDA_CLANG_GENCODE_FLAGS "80;")
    list(APPEND _CUDA_CLANG_GENCODE_FLAGS "86;")
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16.0) # Clang 15 and earlier fail to recognize the flags below
        list(APPEND _CUDA_CLANG_GENCODE_FLAGS "87;")
        list(APPEND _CUDA_CLANG_GENCODE_FLAGS "89;")
        list(APPEND _CUDA_CLANG_GENCODE_FLAGS "90;")
    endif()
endif()
if (GMX_CUDA_TARGET_SM)
    set_property(CACHE GMX_CUDA_TARGET_SM PROPERTY HELPSTRING "List of CUDA GPU architecture codes to compile for (without the sm_ prefix)")
    set_property(CACHE GMX_CUDA_TARGET_SM PROPERTY TYPE STRING)
endif()

# default flags
list(APPEND _CUDA_CLANG_FLAGS "-ffast-math" "-fcuda-flush-denormals-to-zero")
# CUDA toolkit
list(APPEND _CUDA_CLANG_FLAGS "--cuda-path=${CUDAToolkit_TARGET_DIR}")

set(GMX_CUDA_CLANG_FLAGS ${_CUDA_CLANG_FLAGS})


if (CUDA_USE_STATIC_CUDA_RUNTIME)
    set(GMX_CUDA_CLANG_LINK_LIBS "cudart_static")
else()
    set(GMX_CUDA_CLANG_LINK_LIBS "cudart")
endif()
set(GMX_CUDA_CLANG_LINK_LIBS "${GMX_CUDA_CLANG_LINK_LIBS}" "dl" "rt")

set(GMX_CUDA_CLANG_LINK_DIRS "${CUDAToolkit_LIBRARY_DIR}")

set(CMAKE_CUDA_COMPILER ${CMAKE_CXX_COMPILER})

gmx_test_clang_cuda_support()
