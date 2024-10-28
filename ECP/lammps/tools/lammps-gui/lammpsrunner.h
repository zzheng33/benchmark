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

#ifndef LAMMPSRUNNER_H
#define LAMMPSRUNNER_H

#include <QThread>

class LammpsRunner : public QThread {
    Q_OBJECT

public:
    LammpsRunner(QObject *parent = nullptr) : QThread(parent), lammps(nullptr), input(nullptr) {}
    ~LammpsRunner() = default;

public:
    // execute LAMMPS in runner thread
    void run() override
    {
        if (input) {
            lammps->commands_string(input);
            delete[] input;
        } else if (file) {
            lammps->file(file);
            delete[] file;
        }
        emit resultReady();
    }

    // transfer info to worker thread and reset LAMMPS instance
    void setup_run(LammpsWrapper *_lammps, const char *_input, const char *_file = nullptr)
    {
        lammps = _lammps;
        input  = _input;
        file   = _file;
        lammps->command("clear");
    }

signals:
    void resultReady();

private:
    LammpsWrapper *lammps;
    const char *input;
    const char *file;
};

#endif
// Local Variables:
// c-basic-offset: 4
// End:
