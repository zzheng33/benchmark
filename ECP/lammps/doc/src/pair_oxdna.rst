.. index:: pair_style oxdna/excv
.. index:: pair_style oxdna/stk
.. index:: pair_style oxdna/hbond
.. index:: pair_style oxdna/xstk
.. index:: pair_style oxdna/coaxstk

pair_style oxdna/excv command
=============================

pair_style oxdna/stk command
============================

pair_style oxdna/hbond command
==============================

pair_style oxdna/xstk command
=============================

pair_style oxdna/coaxstk command
================================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style style1

   pair_coeff * * style2 args

* style1 = *hybrid/overlay oxdna/excv oxdna/stk oxdna/hbond oxdna/xstk oxdna/coaxstk*

* style2 = *oxdna/excv* or *oxdna/stk* or *oxdna/hbond* or *oxdna/xstk* or *oxdna/coaxstk*
* args = list of arguments for these particular styles

.. parsed-literal::

     *oxdna/stk* args = seq T xi kappa 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65
       seq = seqav (for average sequence stacking strength) or seqdep (for sequence-dependent stacking strength)
       T = temperature (LJ units: 0.1 = 300 K, real units: 300 = 300 K)
       xi = 1.3448 (LJ units) or 8.01727944817084 (real units), temperature-independent coefficient in stacking strength
       kappa = 2.6568 (LJ units) or 0.005279604 (real units), coefficient of linear temperature dependence in stacking strength
     *oxdna/hbond* args = seq eps 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
       seq = seqav (for average sequence base-pairing strength) or seqdep (for sequence-dependent base-pairing strength)
       eps = 1.077 (LJ units) or 6.42073911784652 (real units), average hydrogen bonding strength between A-T and C-G Watson-Crick base pairs, 0 between all other pairs

Examples
""""""""

.. code-block:: LAMMPS

   # LJ units
   pair_style hybrid/overlay oxdna/excv oxdna/stk oxdna/hbond oxdna/xstk oxdna/coaxstk
   pair_coeff * * oxdna/excv    2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32
   pair_coeff * * oxdna/stk     seqdep 0.1 1.3448 2.6568 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65
   pair_coeff * * oxdna/hbond   seqdep 0.0 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
   pair_coeff 1 4 oxdna/hbond   seqdep 1.077 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
   pair_coeff 2 3 oxdna/hbond   seqdep 1.077 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
   pair_coeff * * oxdna/xstk    47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793 0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0 0.65 1.7 0.875 0.68 1.7 0.875 0.68
   pair_coeff * * oxdna/coaxstk 46.0 0.4 0.6 0.22 0.58 2.0 2.541592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 -0.65 2.0 -0.65

   pair_style hybrid/overlay oxdna/excv oxdna/stk oxdna/hbond oxdna/xstk oxdna/coaxstk
   pair_coeff * * oxdna/excv    oxdna_lj.cgdna
   pair_coeff * * oxdna/stk     seqav 0.1 1.3448 2.6568 oxdna_lj.cgdna
   pair_coeff * * oxdna/hbond   seqav oxdna_lj.cgdna
   pair_coeff 1 4 oxdna/hbond   seqav oxdna_lj.cgdna
   pair_coeff 2 3 oxdna/hbond   seqav oxdna_lj.cgdna
   pair_coeff * * oxdna/xstk    oxdna_lj.cgdna
   pair_coeff * * oxdna/coaxstk oxdna_lj.cgdna

   # Real units
   pair_style hybrid/overlay oxdna/excv oxdna/stk oxdna/hbond oxdna/xstk oxdna/coaxstk
   pair_coeff * * oxdna/excv    11.92337812042065 5.9626 5.74965 11.92337812042065 4.38677 4.259 11.92337812042065 2.81094 2.72576
   pair_coeff * * oxdna/stk     seqdep 300.0 8.01727944817084 0.005279604 0.70439070204273 3.4072 7.6662 2.72576 6.3885 1.3 0.0 0.8 0.9 0.0 0.95 0.9 0.0 0.95 2.0 0.65 2.0 0.65
   pair_coeff * * oxdna/hbond   seqdep 0.0 0.93918760272364 3.4072 6.3885 2.89612 5.9626 1.5 0.0 0.7 1.5 0.0 0.7 1.5 0.0 0.7 0.46 3.141592654 0.7 4.0 1.570796327 0.45 4.0 1.570796327 0.45
   pair_coeff 1 4 oxdna/hbond   seqdep 6.42073911784652 0.93918760272364 3.4072 6.3885 2.89612 5.9626 1.5 0.0 0.7 1.5 0.0 0.7 1.5 0.0 0.7 0.46 3.141592654 0.7 4.0 1.570796327 0.45 4.0 1.570796327 0.45
   pair_coeff 2 3 oxdna/hbond   seqdep 6.42073911784652 0.93918760272364 3.4072 6.3885 2.89612 5.9626 1.5 0.0 0.7 1.5 0.0 0.7 1.5 0.0 0.7 0.46 3.141592654 0.7 4.0 1.570796327 0.45 4.0 1.570796327 0.45
   pair_coeff * * oxdna/xstk    3.9029021145006 4.89785 5.74965 4.21641 5.57929 2.25 0.791592654 0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0.0 0.65 1.7 0.875 0.68 1.7 0.875 0.68
   pair_coeff * * oxdna/coaxstk 3.77965257404268 3.4072 5.1108 1.87396 4.94044 2.0 2.541592654 0.65 1.3 0.0 0.8 0.9 0.0 0.95 0.9 0.0 0.95 2.0 -0.65 2.0 -0.65

   pair_style hybrid/overlay oxdna/excv oxdna/stk oxdna/hbond oxdna/xstk oxdna/coaxstk
   pair_coeff * * oxdna/excv    oxdna_real.cgdna
   pair_coeff * * oxdna/stk     seqav 300.0 8.01727944817084 0.005279604 oxdna_real.cgdna
   pair_coeff * * oxdna/hbond   seqav oxdna_real.cgdna
   pair_coeff 1 4 oxdna/hbond   seqav oxdna_real.cgdna
   pair_coeff 2 3 oxdna/hbond   seqav oxdna_real.cgdna
   pair_coeff * * oxdna/xstk    oxdna_real.cgdna
   pair_coeff * * oxdna/coaxstk oxdna_real.cgdna

.. note::

   The coefficients in the above examples are provided in forms
   compatible with both *units lj* and *units real* (see documentation
   of :doc:`units <units>`).  These can also be read from a potential
   file with correct unit style by specifying the name of the
   file. Several potential files for each unit style are included in the
   ``potentials`` directory of the LAMMPS distribution.

Description
"""""""""""

The *oxdna* pair styles compute the pairwise-additive parts of the oxDNA
force field for coarse-grained modelling of DNA. The effective
interaction between the nucleotides consists of potentials for the
excluded volume interaction *oxdna/excv*, the stacking *oxdna/stk*,
cross-stacking *oxdna/xstk* and coaxial stacking interaction
*oxdna/coaxstk* as well as the hydrogen-bonding interaction
*oxdna/hbond* between complementary pairs of nucleotides on opposite
strands. Average sequence or sequence-dependent stacking and
base-pairing strengths are supported :ref:`(Sulc) <Sulc1>`. Quasi-unique
base-pairing between nucleotides can be achieved by using more
complementary pairs of atom types like 5-8 and 6-7, 9-12 and 10-11,
13-16 and 14-15, etc.  This prevents the hybridization of in principle
complementary bases within Ntypes/4 bases up and down along the
backbone.

The exact functional form of the pair styles is rather complex.  The
individual potentials consist of products of modulation factors, which
themselves are constructed from a number of more basic potentials
(Morse, Lennard-Jones, harmonic angle and distance) as well as quadratic
smoothing and modulation terms.  We refer to :ref:`(Ouldridge-DPhil)
<Ouldridge-DPhil1>` and :ref:`(Ouldridge) <Ouldridge1>` for a detailed
description of the oxDNA force field.

.. note::

   These pair styles have to be used together with the related oxDNA
   bond style *oxdna/fene* for the connectivity of the phosphate
   backbone (see also documentation of :doc:`bond_style oxdna/fene
   <bond_oxdna>`). Most of the coefficients in the above example have to
   be kept fixed and cannot be changed without reparameterizing the
   entire model.  Exceptions are the first four coefficients after
   *oxdna/stk* (seq=seqdep, T=0.1, xi=1.3448 and kappa=2.6568 and
   corresponding *real unit* equivalents in the above examples) and the
   first coefficient after *oxdna/hbond* (seq=seqdep in the above
   example).  When using a Langevin thermostat, e.g. through :doc:`fix
   langevin <fix_langevin>` or :doc:`fix nve/dotc/langevin
   <fix_nve_dotc_langevin>` the temperature coefficients have to be
   matched to the one used in the fix.

.. note::

   These pair styles have to be used with the *atom_style hybrid bond
   ellipsoid oxdna* (see documentation of :doc:`atom_style
   <atom_style>`). The *atom_style oxdna* stores the 3'-to-5' polarity
   of the nucleotide strand, which is set through the bond topology in
   the data file. The first (second) atom in a bond definition is
   understood to point towards the 3'-end (5'-end) of the strand.

Example input and data files for DNA duplexes can be found in
``examples/PACKAGES/cgdna/examples/oxDNA/`` and ``.../oxDNA2/``.  A
simple python setup tool which creates single straight or helical DNA
strands, DNA duplexes or arrays of DNA duplexes can be found in
``examples/PACKAGES/cgdna/util/``.

Please cite :ref:`(Henrich) <Henrich1>` in any publication that uses
this implementation. An updated documentation that contains general
information on the model, its implementation and performance as well as
the structure of the data and input file can be found `here
<PDF/CG-DNA.pdf>`_.

Please cite also the relevant oxDNA publications
:ref:`(Ouldridge) <Ouldridge1>`,
:ref:`(Ouldridge-DPhil) <Ouldridge-DPhil1>`
and :ref:`(Sulc) <Sulc1>`.

----------

Potential file reading
""""""""""""""""""""""

For each pair style above the first non-modifiable argument can be a
filename, and if it is, no further arguments should be
supplied. Therefore the following command:

.. code-block:: LAMMPS

   pair_coeff 1 4 oxdna/hbond   seqav oxdna_lj.cgdna

will be interpreted as a request to read the corresponding hydrogen
bonding potential parameters from the file with the given name. The file
can define multiple potential parameters for both bonded and pair
interactions, but for the example pair interaction above there must
exist in the file a line of the form:

.. code-block:: LAMMPS

   1 4 hbond     <coefficients>

If potential customization is required, the potential file reading can
be mixed with the manual specification of the potential parameters. For
example, the following command:

.. code-block:: LAMMPS

   pair_style hybrid/overlay oxdna/excv oxdna/stk oxdna/hbond oxdna/xstk oxdna/coaxstk
   pair_coeff * * oxdna/excv    oxdna_lj.cgdna
   pair_coeff * * oxdna/stk     seqav 0.1 1.3448 2.6568 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65
   pair_coeff * * oxdna/hbond   seqav oxdna_lj.cgdna
   pair_coeff 1 4 oxdna/hbond   seqav oxdna_lj.cgdna
   pair_coeff 2 3 oxdna/hbond   seqav oxdna_lj.cgdna
   pair_coeff * * oxdna/xstk    oxdna_lj.cgdna
   pair_coeff * * oxdna/coaxstk 46.0 0.4 0.6 0.22 0.58 2.0 2.541592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 -0.65 2.0 -0.65

will read the stacking and coaxial stacking potential parameters from
the manual specification and all others from the potential file
*oxdna_lj.cgdna*.

There are sample potential files for each unit style in the
``potentials`` directory of the LAMMPS distribution. The potential file
unit system must align with the units defined via the :doc:`units
<units>` command. For conversion between different *LJ* and *real* unit
systems for oxDNA, the python tool *lj2real.py* located in the
``examples/PACKAGES/cgdna/util/`` directory can be used. This tool
assumes similar file structure to the examples found in
``examples/PACKAGES/cgdna/examples/``.

----------

Restrictions
""""""""""""

These pair styles can only be used if LAMMPS was built with the
CG-DNA package and the MOLECULE and ASPHERE package.  See the
:doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`bond_style oxdna/fene <bond_oxdna>`, :doc:`pair_coeff <pair_coeff>`,
:doc:`bond_style oxdna2/fene <bond_oxdna>`, :doc:`pair_style oxdna2/excv <pair_oxdna2>`,
:doc:`bond_style oxrna2/fene <bond_oxdna>`, :doc:`pair_style oxrna2/excv <pair_oxrna2>`,
:doc:`atom_style oxdna <atom_style>`, :doc:`fix nve/dotc/langevin <fix_nve_dotc_langevin>`

Default
"""""""

none

----------

.. _Henrich1:

**(Henrich)** O. Henrich, Y. A. Gutierrez-Fosado, T. Curk, T. E. Ouldridge, Eur. Phys. J. E 41, 57 (2018).

.. _Ouldridge-DPhil1:

**(Ouldridge-DPhil)** T.E. Ouldridge, Coarse-grained modelling of DNA and DNA self-assembly, DPhil. University of Oxford (2011).

.. _Ouldridge1:

**(Ouldridge)** T.E. Ouldridge, A.A. Louis, J.P.K. Doye, J. Chem. Phys. 134, 085101 (2011).

.. _Sulc1:

**(Sulc)** P. Sulc, F. Romano, T.E. Ouldridge, L. Rovigatti, J.P.K. Doye, A.A. Louis, J. Chem. Phys. 137, 135101 (2012).
