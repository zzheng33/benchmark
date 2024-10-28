########################################
LAMMPS Documentation (|version| version)
########################################

.. toctree::
   :caption: About LAMMPS

****************************
About LAMMPS and this manual
****************************

LAMMPS stands for **L**\ arge-scale **A**\ tomic/**M**\ olecular
**M**\ assively **P**\ arallel **S**\ imulator.

LAMMPS is a classical molecular dynamics simulation code focusing on
materials modeling.  It was designed to run efficiently on parallel
computers and to be easy to extend and modify.  Originally developed at
Sandia National Laboratories, a US Department of Energy facility, LAMMPS
now includes contributions from many research groups and individuals
from many institutions.  Most of the funding for LAMMPS has come from
the US Department of Energy (DOE).  LAMMPS is open-source software
distributed under the terms of the GNU Public License Version 2 (GPLv2).

The `LAMMPS website <lws_>`_ has a variety of information about the
code.  It includes links to an online version of this manual, an
`online forum <https://www.lammps.org/forum.html>`_ where users can post
questions and discuss LAMMPS, and a `GitHub site
<https://github.com/lammps/lammps>`_ where all LAMMPS development is
coordinated.

----------

The content for this manual is part of the LAMMPS distribution in its
doc directory.

* The version of the manual on the LAMMPS website corresponds to the
  latest LAMMPS feature release.  It is available at:
  `https://docs.lammps.org/ <https://docs.lammps.org/>`_.
* A version of the manual corresponding to the latest LAMMPS stable
  release (state of the *stable* branch on GitHub) is available online
  at: `https://docs.lammps.org/stable/
  <https://docs.lammps.org/stable/>`_
* A version of the manual with the features most recently added to
  LAMMPS (state of the *develop* branch on GitHub) is available at:
  `https://docs.lammps.org/latest/ <https://docs.lammps.org/latest/>`_

If needed, you can build a copy on your local machine of the manual
(HTML pages or PDF file) for the version of LAMMPS you have
downloaded.  Follow the steps on the :doc:`Build_manual` page.

.. only:: html

   If you have difficulties viewing the HTML pages, please :ref:`see this note
   <webbrowser>` about compatibility with web browsers.

-----------

The manual is organized into three parts:

1. The :ref:`User Guide <user_documentation>` with information about how
   to obtain, configure, compile, install, and use LAMMPS,
2. the :ref:`Programmer Guide <programmer_documentation>` with
   information about how to use the LAMMPS library interface from
   different programming languages, how to modify and extend LAMMPS,
   the program design, internal programming interfaces, and code
   design conventions,
3. the :ref:`Command Reference <command_reference>` with detailed
   descriptions of all input script commands available in LAMMPS.

----------

.. only:: html

   After becoming familiar with LAMMPS, consider bookmarking
   :doc:`this page <Commands_all>`, since it gives quick access to
   tables with links to the documentation for all LAMMPS commands.

.. _lws: https://www.lammps.org

----------

.. _user_documentation:

************
User Guide
************

.. toctree::
   :maxdepth: 2
   :numbered: 3
   :caption: User Guide
   :name: userdoc
   :includehidden:

   Intro
   Install
   Build
   Run_head
   Commands
   Packages
   Speed
   Howto
   Examples
   Tools
   Errors


.. _programmer_documentation:

******************
Programmer Guide
******************

.. toctree::
   :maxdepth: 2
   :numbered: 3
   :caption: Programmer Guide
   :name: progdoc
   :includehidden:

   Library
   Python_head
   Modify
   Developer

*****************
Command Reference
*****************

.. _command_reference:
.. toctree::
   :name: reference
   :maxdepth: 1
   :caption: Command Reference

   commands_list
   fixes
   computes
   pairs
   bonds
   angles
   dihedrals
   impropers
   dumps
   fix_modify_atc_commands
   Bibliography

******************
Indices and tables
******************

.. only:: html

   * :ref:`genindex`
   * :ref:`search`

.. only:: html

  .. _webbrowser:
  .. admonition:: Web Browser Compatibility
     :class: note

     The HTML version of the manual makes use of advanced features present
     in "modern" web browsers.  This leads to incompatibilities with older
     web browsers and specific vendor browsers (e.g. Internet Explorer on Windows)
     where parts of the pages are not rendered as expected (e.g. the layout is
     broken or mathematical expressions not typeset).  In that case we
     recommend to install/use a different/newer web browser or use
     the `PDF version of the manual <https://docs.lammps.org/Manual.pdf>`_.

     The following web browser versions have been verified to work as
     expected on Linux, macOS, and Windows where available:

     - Safari version 11.1 and later
     - Firefox version 54 and later
     - Chrome version 54 and later
     - Opera version 41 and later
     - Edge version 80 and later

     Also Android version 7.1 and later and iOS version 11 and later have
     been verified to render this website as expected.
