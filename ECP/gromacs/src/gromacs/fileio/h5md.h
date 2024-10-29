/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

/*! \brief Declares the i/o interface to H5MD HDF5 files.
 *
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 */

#ifndef GMX_FILEIO_H5MD_H
#define GMX_FILEIO_H5MD_H

#include "config.h" // To define GMX_USE_HDF5

#include <filesystem>

enum class PbcType : int;

namespace gmx
{

typedef int64_t hid_t;

enum class H5mdFileMode : char
{
    Read  = 'r', //! Only read from the file.
    Write = 'w', //! Write to the file, replaces it if it exists. Also allows reading from the file.
    Append = 'a' //! Write to the file without replacing it if it exists. Also allows reading from the file.
};

/*! \brief Manager of an H5MD filehandle.
 * The class is designed to read/write data according to de Buyl et al., 2014
 * (https://doi.org/10.1016/j.cpc.2014.01.018) and https://www.nongnu.org/h5md/h5md.html
 */
class H5md
{
#if GMX_USE_HDF5
private:
    hid_t file_;            //!< The HDF5 identifier of the file. This is the H5MD root.
    H5mdFileMode filemode_; //!< Whether the file is open for reading ('r'), writing ('w') or appending ('a')
#endif

public:
    /*! \brief Open an H5MD file and manage its filehandle.
     *
     * \param[in] fileName    Name of the file to open. The same as the file path.
     * \param[in] mode        The mode to open the file.
     * \throws FileIOError if fileName is specified and the file cannot be opened.
     */
    H5md(const std::filesystem::path& fileName, const H5mdFileMode mode);

    ~H5md();

    H5md(const H5md&) = delete;
    H5md& operator=(const H5md&) = delete;
    H5md(H5md&&)                 = delete;
    H5md& operator=(H5md&&) = delete;
};

} // namespace gmx
#endif // GMX_FILEIO_H5MD_H
