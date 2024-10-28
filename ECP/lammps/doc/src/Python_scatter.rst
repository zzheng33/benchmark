Scatter/gather operations
=========================

.. code-block:: python

   data = lmp.gather_atoms(name,type,count)  # return per-atom property of all atoms gathered into data, ordered by atom ID
                                             # name = "x", "charge", "type", etc
   data = lmp.gather_atoms_concat(name,type,count)  # ditto, but concatenated atom values from each proc (unordered)
   data = lmp.gather_atoms_subset(name,type,count,ndata,ids)  # ditto, but for subset of Ndata atoms with IDs

   lmp.scatter_atoms(name,type,count,data)   # scatter per-atom property to all atoms from data, ordered by atom ID
                                             # name = "x", "charge", "type", etc
                                             # count = # of per-atom values, 1 or 3, etc

   lmp.scatter_atoms_subset(name,type,count,ndata,ids,data)  # ditto, but for subset of Ndata atoms with IDs


The gather methods collect peratom info of the requested type (atom
coords, atom types, forces, etc) from all processors, and returns the
same vector of values to each calling processor.  The scatter
functions do the inverse.  They distribute a vector of peratom values,
passed by all calling processors, to individual atoms, which may be
owned by different processors.

Note that the data returned by the gather methods,
e.g. :py:meth:`gather_atoms("x") <lammps.lammps.gather_atoms()>`, is
different from the data structure returned by
:py:meth:`extract_atom("x") <lammps.lammps.extract_atom()>` in four ways.
(1) :code:`gather_atoms()` returns a vector which you index as x[i];
:code:`extract_atom()` returns an array which you index as x[i][j].
(2) :code:`gather_atoms()` orders the atoms by atom ID while
:code:`extract_atom()` does not.  (3) :code:`gather_atoms()` returns
a list of all atoms in the simulation; :code:`extract_atoms()` returns just
the atoms local to each processor.  (4) Finally, the :code:`gather_atoms()`
data structure is a copy of the atom coords stored internally in
LAMMPS, whereas :code:`extract_atom()` returns an array that effectively
points directly to the internal data.  This means you can change
values inside LAMMPS from Python by assigning a new values to the
:code:`extract_atom()` array.  To do this with the :code:`gather_atoms()` vector, you
need to change values in the vector, then invoke the
:py:meth:`scatter_atoms("x") <lammps.lammps.scatter_atoms()>` method.

For the scatter methods, the array of coordinates passed to must be a
ctypes vector of ints or doubles, allocated and initialized something
like this:

.. code-block:: python

   from ctypes import c_double
   natoms = lmp.get_natoms()
   n3 = 3*natoms
   x = (n3*c_double)()
   x[0] = x coord of atom with ID 1
   x[1] = y coord of atom with ID 1
   x[2] = z coord of atom with ID 1
   x[3] = x coord of atom with ID 2
   ...
   x[n3-1] = z coord of atom with ID natoms
   lmp.scatter_atoms("x", 1, 3, x)

The coordinates can also be provided as arguments to the initializer of x:

.. code-block:: python

   from ctypes import c_double
   natoms = 2
   n3 = 3*natoms
   # init in constructor
   x = (n3*c_double)(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
   lmp.scatter_atoms("x", 1, 3, x)
   # or using a list
   coords = [1.0, 2.0, 3.0, -3.0, -2.0, -1.0]
   x = (c_double*len(coords))(*coords)

Alternatively, you can just change values in the vector returned by
the gather methods, since they are also ctypes vectors.
