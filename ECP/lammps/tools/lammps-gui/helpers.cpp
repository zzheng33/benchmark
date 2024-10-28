/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "helpers.h"

#include <QBrush>
#include <QColor>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QPalette>
#include <QProcess>
#include <QStringList>

// duplicate string, STL version
char *mystrdup(const std::string &text)
{
    auto *tmp = new char[text.size() + 1];
    memcpy(tmp, text.c_str(), text.size() + 1);
    return tmp;
}

// duplicate string, pointer version
char *mystrdup(const char *text)
{
    return mystrdup(std::string(text));
}

// duplicate string, Qt version
char *mystrdup(const QString &text)
{
    return mystrdup(text.toStdString());
}

// find if executable is in path
// https://stackoverflow.com/a/51041497

bool has_exe(const QString &exe)
{
    QProcess findProcess;
    QStringList arguments;
    arguments << exe;
#if defined(_WIN32)
    findProcess.start("where", arguments);
#else
    findProcess.start("which", arguments);
#endif
    findProcess.setReadChannel(QProcess::ProcessChannel::StandardOutput);

    if (!findProcess.waitForFinished()) return false; // Not found or which does not work

    QString retStr(findProcess.readAll());
    retStr = retStr.trimmed();

    QFile file(retStr);
    QFileInfo check_file(file);
    if (check_file.exists() && check_file.isFile())
        return true; // Found!
    else
        return false; // Not found!
}

// recursively remove all contents from a directory

void purge_directory(const QString &dir)
{
    QDir directory(dir);

    directory.setFilter(QDir::AllEntries | QDir::NoDotAndDotDot);
    const auto &entries = directory.entryList();
    for (auto &entry : entries) {
        if (!directory.remove(entry)) {
            directory.cd(entry);
            directory.removeRecursively();
            directory.cdUp();
        }
    }
}

// compare black level of foreground and background color
bool is_light_theme()
{
    QPalette p;
    int fg = p.brush(QPalette::Active, QPalette::WindowText).color().black();
    int bg = p.brush(QPalette::Active, QPalette::Window).color().black();

    return (fg > bg);
}

// Local Variables:
// c-basic-offset: 4
// End:
