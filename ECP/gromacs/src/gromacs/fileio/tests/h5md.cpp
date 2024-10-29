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
/*! \internal \file
 * \brief
 * Tests for H5MD file I/O routines
 *
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 * \ingroup module_fileio
 */
#include "gmxpre.h"

#include "config.h"

#if GMX_USE_HDF5
#    include <gtest/gtest.h>

#    include "gromacs/fileio/h5md.h"
#    include "gromacs/utility/exceptions.h"
#    include "gromacs/utility/smalloc.h"

#    include "testutils/testasserts.h"
#    include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

/*! \brief Tests that opening (creating a new), closing, re-opening and closing
 * an H5MD file works
 */
TEST(H5mdIoTest, CanCreateAndCloseH5mdFile)
{
    TestFileManager       fileManager;
    std::filesystem::path filename = fileManager.getTemporaryFilePath("ref.h5md");
    {
        EXPECT_THROW_GMX(H5md fileToRead(filename, H5mdFileMode::Read), FileIOError);
    }
    {
        gmx::H5md fileToWrite(filename, H5mdFileMode::Write);
    }
    {
        gmx::H5md fileToRead(filename, H5mdFileMode::Read);
    }
}

} // namespace
} // namespace test
} // namespace gmx

#endif // GMX_USE_HDF5
