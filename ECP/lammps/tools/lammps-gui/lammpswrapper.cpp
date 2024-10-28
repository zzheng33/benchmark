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

#include "lammpswrapper.h"

#if defined(LAMMPS_GUI_USE_PLUGIN)
#include "liblammpsplugin.h"
#else
#include "library.h"
#endif

LammpsWrapper::LammpsWrapper() : lammps_handle(nullptr)
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    plugin_handle = nullptr;
#endif
}

void LammpsWrapper::open(int narg, char **args)
{
    // since there may only be one LAMMPS instance in LAMMPS-GUI we don't open a second one
    if (lammps_handle) return;
#if defined(LAMMPS_GUI_USE_PLUGIN)
    lammps_handle = ((liblammpsplugin_t *)plugin_handle)->open_no_mpi(narg, args, nullptr);
#else
    lammps_handle = lammps_open_no_mpi(narg, args, nullptr);
#endif
}

int LammpsWrapper::version()
{
    int val = 0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->version(lammps_handle);
#else
        val = lammps_version(lammps_handle);
#endif
    }
    return val;
}

int LammpsWrapper::extract_setting(const char *keyword)
{
    int val = 0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->extract_setting(lammps_handle, keyword);
#else
        val = lammps_extract_setting(lammps_handle, keyword);
#endif
    }
    return val;
}

void *LammpsWrapper::extract_global(const char *keyword)
{
    void *val = nullptr;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->extract_global(lammps_handle, keyword);
#else
        val = lammps_extract_global(lammps_handle, keyword);
#endif
    }
    return val;
}

void *LammpsWrapper::extract_pair(const char *keyword)
{
    void *val = nullptr;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->extract_pair(lammps_handle, keyword);
#else
        val = lammps_extract_pair(lammps_handle, keyword);
#endif
    }
    return val;
}

void *LammpsWrapper::extract_atom(const char *keyword)
{
    void *val = nullptr;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->extract_atom(lammps_handle, keyword);
#else
        val = lammps_extract_atom(lammps_handle, keyword);
#endif
    }
    return val;
}

// note: equal style and compatible variables only
double LammpsWrapper::extract_variable(const char *keyword)
{
    void *ptr = nullptr;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        ptr = ((liblammpsplugin_t *)plugin_handle)->extract_variable(lammps_handle, keyword, nullptr);
#else
        ptr = lammps_extract_variable(lammps_handle, keyword, nullptr);
#endif
    }
    double val = *((double *)ptr);
#if defined(LAMMPS_GUI_USE_PLUGIN)
    ((liblammpsplugin_t *)plugin_handle)->free(ptr);
#else
    lammps_free(ptr);
#endif
    return val;
}

int LammpsWrapper::id_count(const char *keyword)
{
    int val = 0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->id_count(lammps_handle, keyword);
#else
        val = lammps_id_count(lammps_handle, keyword);
#endif
    }
    return val;
}

int LammpsWrapper::id_name(const char *keyword, int idx, char *buf, int len)
{
    int val = 0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->id_name(lammps_handle, keyword, idx, buf, len);
#else
        val = lammps_id_name(lammps_handle, keyword, idx, buf, len);
#endif
    }
    return val;
}

int LammpsWrapper::style_count(const char *keyword)
{
    int val = 0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->style_count(lammps_handle, keyword);
#else
        val = lammps_style_count(lammps_handle, keyword);
#endif
    }
    return val;
}

int LammpsWrapper::style_name(const char *keyword, int idx, char *buf, int len)
{
    int val = 0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val =
            ((liblammpsplugin_t *)plugin_handle)->style_name(lammps_handle, keyword, idx, buf, len);
#else
        val = lammps_style_name(lammps_handle, keyword, idx, buf, len);
#endif
    }
    return val;
}

int LammpsWrapper::variable_info(int idx, char *buf, int len)
{
    int val = 0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->variable_info(lammps_handle, idx, buf, len);
#else
        val = lammps_variable_info(lammps_handle, idx, buf, len);
#endif
    }
    return val;
}

double LammpsWrapper::get_thermo(const char *keyword)
{
    double val = 0.0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->get_thermo(lammps_handle, keyword);
#else
        val = lammps_get_thermo(lammps_handle, keyword);
#endif
    }
    return val;
}

void *LammpsWrapper::last_thermo(const char *keyword, int index)
{
    void *ptr = nullptr;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        ptr = ((liblammpsplugin_t *)plugin_handle)->last_thermo(lammps_handle, keyword, index);
#else
        ptr = lammps_last_thermo(lammps_handle, keyword, index);
#endif
    }
    return ptr;
}

bool LammpsWrapper::is_running()
{
    int val = 0;
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        val = ((liblammpsplugin_t *)plugin_handle)->is_running(lammps_handle);
#else
        val = lammps_is_running(lammps_handle);
#endif
    }
    return val != 0;
}

void LammpsWrapper::command(const char *input)
{
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        ((liblammpsplugin_t *)plugin_handle)->command(lammps_handle, input);
#else
        lammps_command(lammps_handle, input);
#endif
    }
}

void LammpsWrapper::file(const char *filename)
{
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        ((liblammpsplugin_t *)plugin_handle)->file(lammps_handle, filename);
#else
        lammps_file(lammps_handle, filename);
#endif
    }
}

void LammpsWrapper::commands_string(const char *input)
{
    if (lammps_handle) {
#if defined(LAMMPS_GUI_USE_PLUGIN)
        ((liblammpsplugin_t *)plugin_handle)->commands_string(lammps_handle, input);
#else
        lammps_commands_string(lammps_handle, input);
#endif
    }
}

// may be called with null handle. returns global error then.
bool LammpsWrapper::has_error() const
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    return ((liblammpsplugin_t *)plugin_handle)->has_error(lammps_handle) != 0;
#else
    return lammps_has_error(lammps_handle) != 0;
#endif
}

// may be called with null handle. returns global error then.
int LammpsWrapper::get_last_error_message(char *buf, int buflen)
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    return ((liblammpsplugin_t *)plugin_handle)->get_last_error_message(lammps_handle, buf, buflen);
#else
    return lammps_get_last_error_message(lammps_handle, buf, buflen);
#endif
}

void LammpsWrapper::force_timeout()
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    if (lammps_handle) ((liblammpsplugin_t *)plugin_handle)->force_timeout(lammps_handle);
#else
    if (lammps_handle) lammps_force_timeout(lammps_handle);
#endif
}

void LammpsWrapper::close()
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    if (lammps_handle) ((liblammpsplugin_t *)plugin_handle)->close(lammps_handle);
#else
    if (lammps_handle) lammps_close(lammps_handle);
#endif
    lammps_handle = nullptr;
}

void LammpsWrapper::finalize()
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    if (lammps_handle) {
        liblammpsplugin_t *lammps = (liblammpsplugin_t *)plugin_handle;
        lammps->close(lammps_handle);
        lammps->mpi_finalize();
        lammps->kokkos_finalize();
        lammps->python_finalize();
    }
#else
    if (lammps_handle) {
        lammps_close(lammps_handle);
        lammps_mpi_finalize();
        lammps_kokkos_finalize();
        lammps_python_finalize();
    }
#endif
}

bool LammpsWrapper::config_has_package(const char *package) const
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    return ((liblammpsplugin_t *)plugin_handle)->config_has_package(package) != 0;
#else
    return lammps_config_has_package(package) != 0;
#endif
}

bool LammpsWrapper::config_accelerator(const char *package, const char *category,
                                       const char *setting) const
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    return ((liblammpsplugin_t *)plugin_handle)->config_accelerator(package, category, setting) !=
           0;
#else
    return lammps_config_accelerator(package, category, setting) != 0;
#endif
}

bool LammpsWrapper::has_gpu_device() const
{
#if defined(LAMMPS_GUI_USE_PLUGIN)
    return ((liblammpsplugin_t *)plugin_handle)->has_gpu_device() != 0;
#else
    return lammps_has_gpu_device() != 0;
#endif
}

#if defined(LAMMPS_GUI_USE_PLUGIN)
bool LammpsWrapper::has_plugin() const
{
    return true;
}

bool LammpsWrapper::load_lib(const char *libfile)
{
    if (plugin_handle) {
        close();
        liblammpsplugin_release((liblammpsplugin_t *)plugin_handle);
    }
    plugin_handle = liblammpsplugin_load(libfile);
    if (!plugin_handle) return false;
    if (((liblammpsplugin_t *)plugin_handle)->abiversion != LAMMPSPLUGIN_ABI_VERSION) {
        liblammpsplugin_release((liblammpsplugin_t *)plugin_handle);
        plugin_handle = nullptr;
        return false;
    }
    return true;
}
#else
bool LammpsWrapper::has_plugin() const
{
    return false;
}

bool LammpsWrapper::load_lib(const char *)
{
    return true;
}
#endif

// Local Variables:
// c-basic-offset: 4
// End:
