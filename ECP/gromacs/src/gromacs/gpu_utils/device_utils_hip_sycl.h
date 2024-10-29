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

#ifndef GMX_GPU_UTILS_DEVICE_UTILS_HIP_SYCL_H
#define GMX_GPU_UTILS_DEVICE_UTILS_HIP_SYCL_H

/*! \file
 *  \brief Shared device methods for SYCL and HIP targets.
 *
 *  This file contains the following shared methods: *
 *  * Cross lane move operations for AMD targets.
 *  * Packed float implementation
 *
 *  TODO add more shared methods to this file
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *  \author Paul Bauer <paul.bauer.q@gmail.com>
 */

// We only want to use the methods in this header when we are actually compiling device code
#if (defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)) || defined(__HIPCC__)

#    include "config.h"

#    include "gromacs/math/functions.h"

// As this header is used both in SYCL and HIP builds, we define the __host__ __device__ attributes
// based on the build type. We also can only use assertions here if they are actually usable
#    define GMX_ALWAYS_INLINE_ATTRIBUTE __attribute__((always_inline))

#    if GMX_GPU_SYCL
#        define GMX_HOST_ATTRIBUTE
#        define GMX_DEVICE_ATTRIBUTE
#        define GMX_HOSTDEVICE_ATTRIBUTE GMX_HOST_ATTRIBUTE GMX_DEVICE_ATTRIBUTE
#        if defined(SYCL_EXT_ONEAPI_ASSERT) && SYCL_EXT_ONEAPI_ASSERT
#            define PACKED_FLOAT_ASSERT(condition) assert(condition)
#        else
#            define PACKED_FLOAT_ASSERT(condition)
#        endif
#        include "gputraits_sycl.h"
#    elif GMX_GPU_HIP
#        define GMX_HOST_ATTRIBUTE __host__
#        define GMX_DEVICE_ATTRIBUTE __device__
#        define GMX_HOSTDEVICE_ATTRIBUTE GMX_HOST_ATTRIBUTE GMX_DEVICE_ATTRIBUTE
#        define PACKED_FLOAT_ASSERT(condition) assert(condition)
#        include "gputraits_hip.h"
#    else
#        error Including packed_float implementation header in unsupported build config
#    endif

#    define GMX_FUNC_ATTRIBUTE GMX_HOSTDEVICE_ATTRIBUTE GMX_ALWAYS_INLINE_ATTRIBUTE

/* !\brief Cross-lane move operation using AMD DPP (Data-Parallel Primitives).
 *
 * Uses the __builtin_amdgcn_update_dpp intrinsic which expressed the data
 * movement but the compiler will combine these with subsequent instructions
 * if possible.
 *
 * Note that this is a generic implementation for any type T (for current use
 * it could be more simple).
 *
 * Ref: https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations
 */
template<class T, int dppCtrl, int rowMask = 0xf, int bankMask = 0xf, bool boundCtrl = true>
GMX_DEVICE_ATTRIBUTE GMX_ALWAYS_INLINE_ATTRIBUTE T amdDppUpdateShfl(const T& input)
{
    static constexpr int c_wordCount = gmx::divideRoundUp(sizeof(T), sizeof(int));

    struct V
    {
        int words[c_wordCount];
    };
    V wordList = __builtin_bit_cast(V, input);

#    pragma unroll
    for (int i = 0; i < c_wordCount; i++)
    {
        wordList.words[i] = __builtin_amdgcn_update_dpp(
                0, wordList.words[i], dppCtrl, rowMask, bankMask, boundCtrl);
    }

    return __builtin_bit_cast(T, wordList);
}

/*! \brief Special packed Float3 flavor to help compiler optimizations on AMD CDNA2 devices.
 *
 * Full FP32 performance of AMD CDNA2 devices, like MI200-series, can only be achieved
 * when operating on float2, in a SIMD2-fashion. Compiler (at least up to ROCm 5.6)
 * can use packed math automatically for normal Float3, but generates a lot of
 * data movement between normal and packed registers. Using this class helps avoid
 * this problem.
 *
 * The approach is based on the similar solution used by AMD and StreamHPC in their port.
 *
 * Currently only used in NBNXM kernels if GMX_NBNXM_ENABLE_PACKED_FLOAT3 is enabled
 *
 * \todo This class shall be removed as soon as the compiler is improved.
 *
 * See issue #4854 for more details.
 */
struct AmdPackedFloat3
{
    typedef float __attribute__((ext_vector_type(2))) Native_float2_;

    /* According to C++ standard, we should give names to all
     * the types and fields declared below. This, however, makes
     * this code very verbose, and harms readability in a major
     * way while this code is aimed to be used in a pretty niche
     * case with relatively small selection of compilers
     * (flavors of Clang 14-18, maybe later). So, we prefer
     * to disable the warnings. */
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wnested-anon-types"
#    pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
    struct __attribute__((packed))
    {
        union
        {
            Native_float2_ xy_;
            struct
            {
                float x_, y_;
            };
        };
        float z_;
    };
#    pragma clang diagnostic pop
    template<typename Index>
    GMX_FUNC_ATTRIBUTE float operator[](Index i) const
    {
        switch (i)
        {
            case 0: return xy_.x;
            case 1: return xy_.y;
            default: PACKED_FLOAT_ASSERT(i == 2); return z_;
        }
    }
    template<typename Index>
    GMX_FUNC_ATTRIBUTE float& operator[](Index i)
    {
        switch (i)
        {
            case 0: return x_;
            case 1: return y_;
            default: PACKED_FLOAT_ASSERT(i == 2); return z_;
        }
    }

    GMX_FUNC_ATTRIBUTE float          x() const { return xy_.x; }
    GMX_FUNC_ATTRIBUTE float          y() const { return xy_.y; }
    GMX_FUNC_ATTRIBUTE Native_float2_ xy() const { return xy_; }
    GMX_FUNC_ATTRIBUTE float          z() const { return z_; }

    GMX_HOSTDEVICE_ATTRIBUTE AmdPackedFloat3() = default;

    GMX_HOSTDEVICE_ATTRIBUTE AmdPackedFloat3(float x, float y, float z) : xy_{ x, y }, z_{ z } {}

    GMX_HOSTDEVICE_ATTRIBUTE AmdPackedFloat3(Native_float2_ xy, float z) : xy_{ xy }, z_{ z } {}

    GMX_HOSTDEVICE_ATTRIBUTE AmdPackedFloat3(Float3 r) : xy_{ r[0], r[1] }, z_{ r[2] } {}

    explicit operator Float3() const { return Float3{ xy_.x, xy_.y, z_ }; }

    GMX_FUNC_ATTRIBUTE AmdPackedFloat3& operator=(const AmdPackedFloat3& x)
    {
        xy_ = x.xy_;
        z_  = x.z_;
        return *this;
    }

    GMX_HOSTDEVICE_ATTRIBUTE AmdPackedFloat3(const AmdPackedFloat3& x) : xy_(x.xy_), z_(x.z_) {}

    //! Allow inplace addition for AmdPackedFloat3
    GMX_FUNC_ATTRIBUTE AmdPackedFloat3& operator+=(const AmdPackedFloat3& right)
    {
        return *this = *this + right;
    }
    //! Allow inplace subtraction for AmdPackedFloat3
    GMX_FUNC_ATTRIBUTE AmdPackedFloat3& operator-=(const AmdPackedFloat3& right)
    {
        return *this = *this - right;
    }
    //! Allow vector addition
    GMX_FUNC_ATTRIBUTE AmdPackedFloat3 operator+(const AmdPackedFloat3& right) const
    {
        return { xy_ + right.xy(), z_ + right.z() };
    }
    //! Allow vector subtraction
    GMX_FUNC_ATTRIBUTE AmdPackedFloat3 operator-(const AmdPackedFloat3& right) const
    {
        return { xy_ - right.xy(), z_ - right.z() };
    }
    //! Scale vector by a scalar
    GMX_FUNC_ATTRIBUTE AmdPackedFloat3& operator*=(const float& right)
    {
        xy_ *= right;
        z_ *= right;
        return *this;
    }

    //! Length^2 of vector
    GMX_FUNC_ATTRIBUTE float norm2() const { return dot(*this); }

    //! Return dot product
    GMX_FUNC_ATTRIBUTE float dot(const AmdPackedFloat3& right) const
    {
        return x() * right.x() + y() * right.y() + z() * right.z();
    }
};
static_assert(sizeof(AmdPackedFloat3) == 12);

GMX_FUNC_ATTRIBUTE static AmdPackedFloat3 operator*(const AmdPackedFloat3& v, const float& s)
{
    return { v.xy() * s, v.z() * s };
}
GMX_FUNC_ATTRIBUTE static AmdPackedFloat3 operator*(const float& s, const AmdPackedFloat3& v)
{
    return { v.xy() * s, v.z() * s };
}

#    undef GMX_HOST_ATTRIBUTE
#    undef GMX_DEVICE_ATTRIBUTE
#    undef GMX_HOSTDEVICE_ATTRIBUTE
#    undef GMX_ALWAYS_INLINE_ATTRIBUTE
#    undef GMX_FUNC_ATTRIBUTE

#endif /* Device code only */

#endif /* GMX_GPU_UTILS_WAVE_MOVE_DPP_H */
