# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

list(APPEND ALL_EXE_SRCS
  ex0.cpp
  ex1.cpp
  ex2.cpp
  ex3.cpp
  ex4.cpp
  ex5.cpp
  ex6.cpp
  ex7.cpp
  ex8.cpp
  ex9.cpp
  ex10.cpp
  ex14.cpp
  ex15.cpp
  ex16.cpp
  ex17.cpp
  ex18.cpp
  ex19.cpp
  ex20.cpp
  ex21.cpp
  ex22.cpp
  ex23.cpp
  ex24.cpp
  ex25.cpp
  ex26.cpp
  ex27.cpp
  ex28.cpp
  ex29.cpp
  ex30.cpp
  ex31.cpp
  ex33.cpp
  ex34.cpp
  ex36.cpp
  ex37.cpp
  )

if(MFEM_USE_LAPACK)
list(APPEND ALL_EXE_SRCS
  ex38.cpp
  )
endif()

if (MFEM_USE_MPI)
  list(APPEND ALL_EXE_SRCS
    ex0p.cpp
    ex1p.cpp
    ex2p.cpp
    ex3p.cpp
    ex4p.cpp
    ex5p.cpp
    ex6p.cpp
    ex7p.cpp
    ex8p.cpp
    ex9p.cpp
    ex10p.cpp
    ex11p.cpp
    ex12p.cpp
    ex13p.cpp
    ex14p.cpp
    ex15p.cpp
    ex16p.cpp
    ex17p.cpp
    ex18p.cpp
    ex19p.cpp
    ex20p.cpp
    ex21p.cpp
    ex22p.cpp
    ex24p.cpp
    ex25p.cpp
    ex26p.cpp
    ex27p.cpp
    ex28p.cpp
    ex29p.cpp
    ex30p.cpp
    ex31p.cpp
    ex32p.cpp
    ex33p.cpp
    ex34p.cpp
    ex35p.cpp
    ex36p.cpp
    ex37p.cpp
    )
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add one executable per cpp file
add_mfem_examples(ALL_EXE_SRCS)

# Add a test for each example
if (MFEM_ENABLE_TESTING)
  foreach(SRC_FILE ${ALL_EXE_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
    string(REPLACE ".cpp" "" TEST_NAME ${SRC_FILENAME})

    set(THIS_TEST_OPTIONS "-no-vis")
    if (${TEST_NAME} MATCHES "ex0p?")
      set(THIS_TEST_OPTIONS)
    endif()
    if (${TEST_NAME} MATCHES "ex10p*")
      list(APPEND THIS_TEST_OPTIONS "-tf" "5")
    elseif(${TEST_NAME} MATCHES "ex15p*")
      list(APPEND THIS_TEST_OPTIONS "-e" "1")
    elseif(${TEST_NAME} MATCHES "ex27p*")
      list(APPEND THIS_TEST_OPTIONS "-dg")
    elseif(${TEST_NAME} MATCHES "ex37p*")
      list(APPEND THIS_TEST_OPTIONS "-mi" "3")
    endif()

    if (NOT (${TEST_NAME} MATCHES ".*p$"))
      add_test(NAME ${TEST_NAME}_ser
        COMMAND ${TEST_NAME} ${THIS_TEST_OPTIONS})
    else()
      add_test(NAME ${TEST_NAME}_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
        ${MPIEXEC_POSTFLAGS})
    endif()
  endforeach()

  # Add CUDA/HIP tests.
  set(DEVICE_EXAMPLES
      # serial examples with device support:
      ex1 ex3 ex4 ex5 ex6 ex9 ex22 ex24 ex25 ex26 ex34
      # parallel examples with device support:
      ex1p ex2p ex3p ex4p ex5p ex6p ex7p ex9p ex13p ex22p ex24p ex25p ex26p
      ex34p ex35p)
  set(MFEM_TEST_DEVICE)
  if (MFEM_USE_CUDA)
    set(MFEM_TEST_DEVICE "cuda")
  elseif (MFEM_USE_HIP)
    set(MFEM_TEST_DEVICE "hip")
  endif()
  if (MFEM_TEST_DEVICE)
    foreach(TEST_NAME ${DEVICE_EXAMPLES})
      set(THIS_TEST_OPTIONS "-no-vis" "-d" "${MFEM_TEST_DEVICE}")
      if (NOT (${TEST_NAME} MATCHES ".*p$"))
        add_test(NAME ${TEST_NAME}_${MFEM_TEST_DEVICE}_ser
          COMMAND ${TEST_NAME} ${THIS_TEST_OPTIONS})
      elseif (MFEM_USE_MPI)
        add_test(NAME ${TEST_NAME}_${MFEM_TEST_DEVICE}_np=${MFEM_MPI_NP}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
          ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
          ${MPIEXEC_POSTFLAGS})
      endif()
    endforeach()
  endif()

  # If STRUMPACK is enabled, add a test run that uses it.
  if (MFEM_USE_STRUMPACK)
    add_test(NAME ex11p_strumpack_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ex11p> "-no-vis" "--strumpack"
      ${MPIEXEC_POSTFLAGS})
  endif()

  # If SuperLU_DIST is enabled, add a test run that uses it.
  if (MFEM_USE_SUPERLU)
    add_test(NAME ex11p_superlu_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ex11p> "-no-vis" "--superlu"
      ${MPIEXEC_POSTFLAGS})
  endif()

  # If MUMPS is enabled, add a test run that uses it.
  if (MFEM_USE_MUMPS)
    add_test(NAME ex25p_mumps_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ex25p> "-no-vis" "--mumps-solver"
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()

# Include the examples/amgx directory if AmgX is enabled
if (MFEM_USE_AMGX)
  add_subdirectory(amgx)
endif()

# Include the examples/ginkgo directory if GINKGO is enabled.
if (MFEM_USE_GINKGO)
  add_subdirectory(ginkgo)
endif()

# Include the examples/hiop directory if HiOp is enabled
if (MFEM_USE_HIOP)
  add_subdirectory(hiop)
endif()

# Include the examples/petsc directory if PETSc is enabled.
if (MFEM_USE_PETSC)
  add_subdirectory(petsc)
endif()

# Include the examples/pumi directory if PUMI is enabled
if (MFEM_USE_PUMI)
  add_subdirectory(pumi)
endif()

# Include the examples/sundials directory if SUNDIALS is enabled.
if (MFEM_USE_SUNDIALS)
  add_subdirectory(sundials)
endif()

if(MFEM_USE_CALIPER)
  add_subdirectory(caliper)
endif()

# Include the examples/superlu directory if SUPERLU is enabled.
if (MFEM_USE_SUPERLU)
  add_subdirectory(superlu)
endif()

if(MFEM_USE_MOONOLITH)
  add_subdirectory(moonolith)
endif()
