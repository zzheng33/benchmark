/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS
// clang-format off
BondStyle(gaussian,BondGaussian);
// clang-format on
#else

#ifndef LMP_BOND_GAUSSIAN_H
#define LMP_BOND_GAUSSIAN_H

#include "bond.h"

namespace LAMMPS_NS {

class BondGaussian : public Bond {
 public:
  BondGaussian(class LAMMPS *);
  ~BondGaussian() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  double equilibrium_distance(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, double, int, int, double &) override;
  void born_matrix(int, double, int, int, double &, double &) override;
  void *extract(const char *, int &) override;

 protected:
  int *nterms;
  double *bond_temperature;
  double **alpha, **width, **r0;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
