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
 * Declares the options for NNPot MDModule class,
 * set during pre-processing in the .mdp-file.
 *
 * \author Lukas Müllender <lukas.muellender@gmail.com>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_NNPOTOPTIONS_H
#define GMX_APPLIED_FORCES_NNPOTOPTIONS_H

#include <string>

#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/topology/atoms.h"

// some forward declarations
struct gmx_mtop_t;
struct t_commrec;
class WarningHandler;
enum class PbcType;

namespace gmx
{

// more forward declarations
class MDLogger;
class IOptionsContainerWithSections;
class IKeyValueTreeTransformRules;
class KeyValueTreeObjectBuilder;
class KeyValueTreeObject;
class IndexGroupsAndNames;
class LocalAtomSet;

//! Tag with name of the NNPot MDModule
static const std::string c_nnpotModuleName = "nnpot";

//!\brief \internal Data structure to store NNPot input parameters
struct NNPotParameters
{
    //! indicates whether NN Potential is active (default false)
    bool active_ = false;

    //! stores file name of NNPot model
    std::string modelFileName_ = "model.pt";

    //! indicates whether NNPot provides forces
    bool providesForces_ = false;

    //! stores atom group name for neural network input (default whole System)
    std::string inputGroup_ = "System";
    //! Indices of the atoms that are part of the NN input region (default whole System)
    std::vector<Index> inpIndices_;
    //! Local set of atoms that are part of the NN input region
    std::unique_ptr<LocalAtomSet> inpAtoms_;

    //! Indices of the atoms that are part of the MM region (default no MM atoms)
    std::vector<Index> mmIndices_;
    //! Local set of atoms that are part of the MM region
    std::unique_ptr<LocalAtomSet> mmAtoms_;

    //! User defined input to NN model (4 options as of now)
    std::vector<std::string> modelInput_{ "", "", "", "" };

    //! stores pbc type used by the simulation
    std::unique_ptr<PbcType> pbcType_;

    //! stores all (global) atom info
    t_atoms atoms_;
    int     numAtoms_;

    //! stores communication record
    const t_commrec* cr_ = nullptr;

    bool modelNeedsInput(const std::string& input) const
    {
        return std::find(modelInput_.begin(), modelInput_.end(), input) != modelInput_.end();
    }
};

class NNPotOptions final : public IMdpOptionProvider
{
public:
    //! Implementation of IMdpOptionProvider method
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override;

    //! \brief Connects option names and data.
    void initMdpOptions(IOptionsContainerWithSections* options) override;

    /*! \brief Build mdp parameters for NNPot to be output after pre-processing.
     * \param[in, out] builder the builder for the mdp options output KVT.
     */
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override;

    //! return active state of NNPot module
    bool isActive() const;

    //! get model file name
    std::string getModelFileName() const;

    //! set atom group for neural network input
    void setInputGroupIndices(const IndexGroupsAndNames&);

    //! set local atom set for neural network input during simulation setup
    void setLocalInputAtomSet(const LocalAtomSet&);

    //! set local MM atom set during simulation setup
    void setLocalMMAtomSet(const LocalAtomSet&);

    //! modify topology of the system during preprocessing
    void modifyTopology(gmx_mtop_t*);

    //! set topology of the system during simulation setup
    void setTopology(const gmx_mtop_t&);

    //! set communication record during simulation setup
    void setCommRec(const t_commrec&);

    //! Store the paramers that are not mdp options in the tpr file
    // This is needed to retain data from preprocessing to simulation setup
    void writeParamsToKvt(KeyValueTreeObjectBuilder);

    //! Set the internal parameters that are stored in the tpr file
    void readParamsFromKvt(const KeyValueTreeObject&);

    //! get NNPot parameters
    const NNPotParameters& parameters();

    //! set PBC type during simulation setup
    void setPbcType(const PbcType&);

    void            setLogger(const MDLogger&);
    void            setWarninp(WarningHandler*);
    const MDLogger* logger();

private:
    //! Make sure that model and model inputs are compatible
    void checkNNPotModel();

    /*! \brief Following Tags denotes names of parameters from .mdp file
     * \note Changing this strings will break .tpr backwards compability
     */
    //! \{
    const std::string c_activeTag_         = "active";
    const std::string c_modelFileNameTag_  = "modelfile";
    const std::string c_providesForcesTag_ = "provides_forces";
    const std::string c_inputGroupTag_     = "input_group";
    //! complement to input_group, needed to write to tpr
    const std::string c_mmGroupTag_ = "mm_group";

    /*! \brief User defined input to NN model.
     *
     *  Possible values:
     * - "atom_positions" vector of atom positions
     * - "atom_numbers" vector of atom types
     * - "box" unit vectors of simulation box
     * - "pbc" boolean vector indicating periodic boundary conditions
     */
    const std::string c_modelInput1Tag_ = "model_input1";
    const std::string c_modelInput2Tag_ = "model_input2";
    const std::string c_modelInput3Tag_ = "model_input3";
    const std::string c_modelInput4Tag_ = "model_input4";
    //! \}

    //! NNPot parameters
    NNPotParameters params_;

    //! Logger instance
    const MDLogger* logger_ = nullptr;

    //! Instance of warning bookkeeper
    WarningHandler* wi_ = nullptr;
};

} // namespace gmx

#endif
