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

#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QComboBox>
#include <QDialog>
#include <QImage>
#include <QString>

class QAction;
class QMenuBar;
class QDialogButtonBox;
class QLabel;
class QObject;
class QScrollArea;
class QScrollBar;
class QStatusBar;
class LammpsWrapper;
class QComboBox;

class ImageViewer : public QDialog {
    Q_OBJECT

public:
    explicit ImageViewer(const QString &fileName, LammpsWrapper *_lammps,
                         QWidget *parent = nullptr);

private slots:
    void saveAs();
    void copy();
    void quit();

    void set_atom_size();
    void edit_size();
    void reset_view();
    void toggle_ssao();
    void toggle_anti();
    void toggle_shiny();
    void toggle_vdw();
    void toggle_box();
    void toggle_axes();
    void do_zoom_in();
    void do_zoom_out();
    void do_rot_left();
    void do_rot_right();
    void do_rot_up();
    void do_rot_down();
    void do_recenter();
    void cmd_to_clipboard();
    void change_group(int);

public:
    void createImage();

private:
    void createActions();
    void updateActions();
    void saveFile(const QString &fileName);
    void scaleImage(double factor);
    void adjustScrollBar(QScrollBar *scrollBar, double factor);

private:
    QImage image;
    QMenuBar *menuBar;
    QLabel *imageLabel;
    QScrollArea *scrollArea;
    QDialogButtonBox *buttonBox;
    double scaleFactor;
    double atomSize;

    QAction *saveAsAct;
    QAction *copyAct;
    QAction *cmdAct;
    QAction *zoomInAct;
    QAction *zoomOutAct;
    QAction *normalSizeAct;

    LammpsWrapper *lammps;
    QString group;
    QString filename;
    QString last_dump_cmd;
    int xsize, ysize;
    int hrot, vrot;
    double zoom, vdwfactor, shinyfactor;
    double xcenter, ycenter, zcenter;
    bool showbox, showaxes, antialias, usessao, useelements, usediameter, usesigma;
};
#endif

// Local Variables:
// c-basic-offset: 4
// End:
