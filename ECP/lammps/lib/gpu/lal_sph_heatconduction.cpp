/***************************************************************************
                                sph_heatconduction.cpp
                             -------------------
                            Trung Nguyen (U Chicago)

  Class for acceleration of the sph_heatconduction pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                : September 2023
    email                : ndactrung@gmail.com
 ***************************************************************************/

#if defined(USE_OPENCL)
#include "sph_heatconduction_cl.h"
#elif defined(USE_CUDART)
const char *sph_heatconduction=0;
#else
#include "sph_heatconduction_cubin.h"
#endif

#include "lal_sph_heatconduction.h"
#include <cassert>
namespace LAMMPS_AL {
#define SPHHeatConductionT SPHHeatConduction<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
SPHHeatConductionT::SPHHeatConduction() : BaseSPH<numtyp,acctyp>(), _allocated(false) {
  _max_dE_size = 0;
}

template <class numtyp, class acctyp>
SPHHeatConductionT::~SPHHeatConduction() {
  clear();
}

template <class numtyp, class acctyp>
int SPHHeatConductionT::bytes_per_atom(const int max_nbors) const {
  return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int SPHHeatConductionT::init(const int ntypes,
                 double **host_cutsq, double **host_cut,
                 double **host_alpha, double* host_mass,
                 const int dimension, double *host_special_lj,
                 const int nlocal, const int nall,
                 const int max_nbors, const int maxspecial,
                 const double cell_size,
                 const double gpu_split, FILE *_screen) {
  const int max_shared_types=this->device->max_shared_types();

  int onetype=0;
  #ifdef USE_OPENCL
  if (maxspecial==0)
    for (int i=1; i<ntypes; i++)
      for (int j=i; j<ntypes; j++)
        if (host_cutsq[i][j]>0) {
          if (onetype>0)
            onetype=-1;
          else if (onetype==0)
            onetype=i*max_shared_types+j;
        }
  if (onetype<0) onetype=0;
  #endif

  int success;
  int extra_fields = 4; // round up to accomodate quadruples of numtyp values
                        // rho, esph
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,
                            gpu_split,_screen,sph_heatconduction,"k_sph_heatconduction",
                            onetype,extra_fields);
  if (success!=0)
    return success;

  // If atom type constants fit in shared memory use fast kernel
  int lj_types=ntypes;
  shared_types=false;
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    shared_types=true;
  }
  _lj_types=lj_types;

  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(lj_types*lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<lj_types*lj_types; i++)
    host_write[i]=0.0;

  coeff.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,coeff,host_write,host_alpha,
                         host_cut, host_cutsq);

  UCL_H_Vec<numtyp> dview_mass(ntypes, *(this->ucl_device), UCL_WRITE_ONLY);
  for (int i = 0; i < ntypes; i++)
    dview_mass[i] = host_mass[i];
  mass.alloc(ntypes,*(this->ucl_device), UCL_READ_ONLY);
  ucl_copy(mass,dview_mass,false);

  UCL_H_Vec<double> dview;
  sp_lj.alloc(4,*(this->ucl_device),UCL_READ_ONLY);
  dview.view(host_special_lj,4,*(this->ucl_device));
  ucl_copy(sp_lj,dview,false);

  // allocate per-atom array Q

  int ef_nall=nall;
  if (ef_nall==0)
    ef_nall=2000;

  _max_dE_size=static_cast<int>(static_cast<double>(ef_nall)*1.10);
  dE.alloc(_max_dE_size,*(this->ucl_device),UCL_READ_WRITE,UCL_READ_WRITE);

  _dimension = dimension;

  _allocated=true;
  this->_max_bytes=coeff.row_bytes()+dE.row_bytes()+sp_lj.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void SPHHeatConductionT::clear() {
  if (!_allocated)
    return;
  _allocated=false;

  coeff.clear();
  mass.clear();
  dE.clear();
  sp_lj.clear();
  this->clear_atomic();
}

template <class numtyp, class acctyp>
double SPHHeatConductionT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+sizeof(SPHHeatConduction<numtyp,acctyp>);
}

template <class numtyp, class acctyp>
void SPHHeatConductionT::update_dE(void **dE_ptr) {
  *dE_ptr=dE.host.begin();
  dE.update_host(_max_dE_size,false);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
int SPHHeatConductionT::loop(const int eflag, const int vflag) {

  int nall = this->atom->nall();

  // Resize dE array if necessary
  if (nall > _max_dE_size) {
    _max_dE_size=static_cast<int>(static_cast<double>(nall)*1.10);
    dE.resize(_max_dE_size);
  }

  // signal that we need to transfer extra data from the host

  this->atom->extra_data_unavail();

  numtyp4 *pextra=reinterpret_cast<numtyp4*>(&(this->atom->extra[0]));

  int n = 0;
  int nstride = 1;
  for (int i = 0; i < nall; i++) {
    int idx = n+i*nstride;
    numtyp4 v;
    v.x = rho[i];
    v.y = esph[i];
    v.z = 0;
    v.w = 0;
    pextra[idx] = v;
  }
  this->atom->add_extra_data();

  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));


  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  if (shared_types) {
    this->k_pair_sel->set_size(GX,BX);
    this->k_pair_sel->run(&this->atom->x, &this->atom->extra, &coeff, &mass, &sp_lj,
                          &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                          &this->ans->force, &this->ans->engv, &dE, &eflag, &vflag,
                          &ainum, &nbor_pitch, &this->atom->v, &_dimension, &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->x, &this->atom->extra, &coeff, &mass,
                     &_lj_types, &sp_lj, &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                     &this->ans->force, &this->ans->engv, &dE, &eflag, &vflag,
                     &ainum, &nbor_pitch, &this->atom->v, &_dimension, &this->_threads_per_atom);
  }

  this->time_pair.stop();
  return GX;
}

// ---------------------------------------------------------------------------
// Get the extra data pointers from host
// ---------------------------------------------------------------------------

template <class numtyp, class acctyp>
void SPHHeatConductionT::get_extra_data(double *host_rho, double *host_esph) {
  rho = host_rho;
  esph = host_esph;
}

template class SPHHeatConduction<PRECISION,ACC_PRECISION>;
}
