/*
 * Copyright [2020] [Technical University of Munich]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//////////////////////////////////////////////////////////////////////////////////////
/// \file softvector.cpp
/// \brief C++ Source for ETISS JIT libary or independent C application.
/// Implements a C interface when compiled to library beforehand
/// \date 06/23/2020
//////////////////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "softvector.h"
#include "arithmetic/softfloat-extension.hpp"
#include "operations.hpp"

#include "base/base.hpp"
#include "lsu/lsu.hpp"
#include "arithmetic/integer.hpp"
#include "arithmetic/floatingpoint.hpp"
#include "arithmetic/fixedpoint.hpp"
#include "misc/mask.hpp"
#include "misc/permutation.hpp"

#ifdef ETISS_SOFTFLOAT
extern "C"
{
#include "softfloat_orig.h"
}
#else
#include "softfloat.h"
#endif

#define GO_FAST __attribute__((always_inline))

// extern "C"
// {

/* --- Private enums --- */
enum class SignType
{
    Signed,
    Unsigned
};

enum class MaskType
{
    Masked,
    Unmasked
};

enum class ImmExtensionType
{
    SignExtend,
    ZeroExtend
};

template <typename BaseType>
struct TypeWidener;

template <>
struct TypeWidener<uint8_t>
{
    using wide_type = uint16_t;
};
template <>
struct TypeWidener<uint16_t>
{
    using wide_type = uint32_t;
};
template <>
struct TypeWidener<uint32_t>
{
    using wide_type = uint64_t;
};
template <>
struct TypeWidener<int8_t>
{
    using wide_type = int16_t;
};
template <>
struct TypeWidener<int16_t>
{
    using wide_type = int32_t;
};
template <>
struct TypeWidener<int32_t>
{
    using wide_type = int64_t;
};

template <uint8_t Sew, SignType Sign>
struct ElementTypeMap;

template <>
struct ElementTypeMap<8, SignType::Signed>
{
    using element_type = int8_t;
};
template <>
struct ElementTypeMap<16, SignType::Signed>
{
    using element_type = int16_t;
};
template <>
struct ElementTypeMap<32, SignType::Signed>
{
    using element_type = int32_t;
};
template <>
struct ElementTypeMap<64, SignType::Signed>
{
    using element_type = int64_t;
};
template <>
struct ElementTypeMap<8, SignType::Unsigned>
{
    using element_type = uint8_t;
};
template <>
struct ElementTypeMap<16, SignType::Unsigned>
{
    using element_type = uint16_t;
};
template <>
struct ElementTypeMap<32, SignType::Unsigned>
{
    using element_type = uint32_t;
};
template <>
struct ElementTypeMap<64, SignType::Unsigned>
{
    using element_type = uint64_t;
};

template <uint8_t Sew, SignType Sign, uint8_t Factor>
struct ExtTypes;

template <>
struct ExtTypes<16, SignType::Signed, 2>
{
    using source_type = int8_t;
    using dest_type = int16_t;
};

template <>
struct ExtTypes<16, SignType::Unsigned, 2>
{
    using source_type = uint8_t;
    using dest_type = uint16_t;
};

template <>
struct ExtTypes<32, SignType::Signed, 2>
{
    using source_type = int16_t;
    using dest_type = int32_t;
};

template <>
struct ExtTypes<32, SignType::Unsigned, 2>
{
    using source_type = uint16_t;
    using dest_type = uint32_t;
};

template <>
struct ExtTypes<32, SignType::Signed, 4>
{
    using source_type = int8_t;
    using dest_type = int32_t;
};

template <>
struct ExtTypes<32, SignType::Unsigned, 4>
{
    using source_type = uint8_t;
    using dest_type = uint32_t;
};

template <>
struct ExtTypes<64, SignType::Signed, 2>
{
    using source_type = int32_t;
    using dest_type = int64_t;
};

template <>
struct ExtTypes<64, SignType::Unsigned, 2>
{
    using source_type = uint32_t;
    using dest_type = uint64_t;
};

template <>
struct ExtTypes<64, SignType::Signed, 4>
{
    using source_type = int16_t;
    using dest_type = int64_t;
};

template <>
struct ExtTypes<64, SignType::Unsigned, 4>
{
    using source_type = uint16_t;
    using dest_type = uint64_t;
};

template <>
struct ExtTypes<64, SignType::Signed, 8>
{
    using source_type = int8_t;
    using dest_type = int64_t;
};

template <>
struct ExtTypes<64, SignType::Unsigned, 8>
{
    using source_type = uint8_t;
    using dest_type = uint64_t;
};

/* --- Private globals --- */

constexpr auto masked_instruction_value = false;
constexpr auto masked_element_value = false;
constexpr auto sew_8 = 8;
constexpr auto sew_16 = 16;
constexpr auto sew_32 = 32;
constexpr auto sew_64 = 64;

/* --- Private function declarations --- */

template <typename VectorElementType, bool IsSigned>
auto dump_v_register(unsigned v_register, unsigned vlen, void *const vector_field) -> void;

inline constexpr auto decode_sew(uint32_t vtype) -> unsigned;

inline constexpr auto is_masked_instruction(bool const instruction_mask_bit) -> bool;

inline constexpr auto is_masked_element(bool const element_mask_bit) -> bool;

template <SignType Sign>
inline constexpr uint64_t get_scalar(void *const scalar_field, unsigned const sew, unsigned const xlen,
                                     unsigned const rs1);

inline constexpr uint64_t get_float_scalar(void *const float_scalar_field, unsigned const sew, unsigned const flen,
                                           unsigned const rs1);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_vv(void *const vector_field, uint16_t const vtype,
                                                  uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                  uint8_t const vs2, uint16_t const vstart, uint16_t const vlen,
                                                  uint16_t const vl, OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_vv(void *const vector_field, uint16_t const vtype,
                                                           uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                           uint8_t const vs2, uint16_t const vstart,
                                                           uint16_t const vlen, uint16_t const vl, OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_wv(void *const vector_field, uint16_t const vtype,
                                                           uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                           uint8_t const vs2, uint16_t const vstart,
                                                           uint16_t const vlen, uint16_t const vl, OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_narrowing_wv(void *const vector_field, uint16_t const vtype,
                                                            uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                            uint8_t const vs2, uint16_t const vstart,
                                                            uint16_t const vlen, uint16_t const vl, OpType const op);

template <SignType Sign, ImmExtensionType ImmExtension, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_vi(void *const vector_field, uint16_t const vtype,
                                                  uint8_t const mask_bit, uint8_t const vd, uint8_t const vs2,
                                                  uint8_t immediate, uint16_t const vstart, uint16_t const vlen,
                                                  uint16_t const vl, OpType const op);

template <SignType Sign, ImmExtensionType ImmExtension, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_narrowing_wi(void *const vector_field, uint16_t const vtype,
                                                            uint8_t const mask_bit, uint8_t const vd, uint8_t const vs2,
                                                            uint8_t const immediate, uint16_t const vstart,
                                                            uint16_t const vlen, uint16_t const vl, OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_vx(void *const vector_field, void *const scalar_field,
                                                  uint16_t const vtype, uint8_t const mask_bit, uint8_t const vd,
                                                  uint8_t const vs2, uint8_t rs1, uint16_t const vstart,
                                                  uint16_t const vlen, uint16_t xlen, uint16_t const vl,
                                                  OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_vx(void *const vector_field, void *const scalar_field,
                                                           uint16_t const vtype, uint8_t const mask_bit,
                                                           uint8_t const vd, uint8_t const vs2, uint8_t const rs1,
                                                           uint16_t const vstart, uint16_t const vlen, uint16_t xlen,
                                                           uint16_t const vl, OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_wx(void *const vector_field, void *const scalar_field,
                                                           uint16_t const vtype, uint8_t const mask_bit,
                                                           uint8_t const vd, uint8_t const vs2, uint8_t const rs1,
                                                           uint16_t const vstart, uint16_t const vlen, uint16_t xlen,
                                                           uint16_t const vl, OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_narrowing_wx(void *const vector_field, void *const scalar_field,
                                                            uint16_t const vtype, uint8_t const mask_bit,
                                                            uint8_t const vd, uint8_t const vs2, uint8_t const rs1,
                                                            uint16_t const vstart, uint16_t const vlen, uint16_t xlen,
                                                            uint16_t const vl, OpType const op);

template <unsigned Sew, unsigned Factor>
inline constexpr GO_FAST void dispatch_iterate_vext(void *const vector_field, uint16_t const vtype,
                                                    bool const is_masked, uint8_t const vd, uint8_t const vs2,
                                                    bool const is_signed, uint16_t const vstart, uint16_t const vlen,
                                                    uint16_t const vl);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_reduce(void *const vector_field, uint16_t const vtype,
                                                      uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                      uint8_t const vs2, uint16_t const vstart, uint16_t const vlen,
                                                      uint16_t const vl, OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_reduce(void *const vector_field, uint16_t const vtype,
                                                               uint8_t const mask_bit, uint8_t const vd,
                                                               uint8_t const vs1, uint8_t const vs2,
                                                               uint16_t const vstart, uint16_t const vlen,
                                                               uint16_t const vl, OpType const op);

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_v_unary(void *const vector_field, uint16_t const vtype,
                                                       uint8_t const mask_bit, uint8_t const vd, uint8_t const vs2,
                                                       uint16_t const vstart, uint16_t const vlen, uint16_t const vl,
                                                       OpType const op);

template <typename OpType, SignType Sign = SignType::Unsigned>
inline constexpr GO_FAST void dispatch_iterate_vf(void *const vector_field, void *const scalar_field,
                                                  uint16_t const vtype, uint8_t const mask_bit, uint8_t const vd,
                                                  uint8_t const vs2, uint8_t const rs1, uint16_t const vstart,
                                                  uint16_t const vlen, uint16_t flen, uint16_t const vl,
                                                  OpType const op);

template <typename OpType, SignType Sign = SignType::Unsigned>
inline constexpr GO_FAST void dispatch_iterate_widening_vf(void *const vector_field, void *const scalar_field,
                                                           uint16_t const vtype, uint8_t const mask_bit,
                                                           uint8_t const vd, uint8_t const vs2, uint8_t const rs1,
                                                           uint16_t const vstart, uint16_t const vlen, uint16_t flen,
                                                           uint16_t const vl, OpType const op);

template <typename OpType, SignType Sign = SignType::Unsigned>
inline constexpr GO_FAST void dispatch_iterate_widening_wf(void *const vector_field, void *const scalar_field,
                                                           uint16_t const vtype, uint8_t const mask_bit,
                                                           uint8_t const vd, uint8_t const vs2, uint8_t const rs1,
                                                           uint16_t const vstart, uint16_t const vlen, uint16_t flen,
                                                           uint16_t const vl, OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void vv_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                 unsigned const vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                 OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_vv_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                          unsigned vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                          OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_wv_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                          unsigned vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                          OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, ValueResultOpSewData>
inline constexpr void narrowing_wv_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                           unsigned vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                           OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void vxi_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                  unsigned const vd_base, unsigned const vs2_base, uint64_t const scalar,
                                  OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_vx_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                          unsigned const vd_base, unsigned const vs2_base, uint64_t const scalar,
                                          OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_wx_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                          unsigned const vd_base, unsigned const vs2_base, uint64_t const scalar,
                                          OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, ValueResultOpSewData>
inline constexpr void narrowing_wxi_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                            unsigned const vd_base, unsigned const vs2_base, uint64_t const scalar,
                                            OpType const op);

template <typename SourceType, typename DestType, MaskType Masked>
    requires ValidVectorElementType<SourceType> and ValidVectorElementType<DestType>
inline constexpr void vext_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                   unsigned const vd_base, unsigned const vs2_base);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void reduce_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                     unsigned const vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                     OpType const op);

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_reduce_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                              unsigned const vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                              OpType const op);

/* --- Public function definitions --- */
#define VRED_OP(name, inner_op, sign)                                                                                 \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,    \
                 uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl) \
    {                                                                                                                 \
        dispatch_iterate_reduce<sign>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen, vl,    \
                                      inner_op);                                                                      \
        return 0;                                                                                                     \
    }

#define W_VRED_OP(name, inner_op, sign)                                                                               \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,    \
                 uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl) \
    {                                                                                                                 \
        dispatch_iterate_widening_reduce<sign>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart,     \
                                               vlen, vl, inner_op);                                                   \
        return 0;                                                                                                     \
    }

#define VFRED_OP(name, inner_op)                                                                                       \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,     \
                 uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl,  \
                 uint8_t const rounding_mode)                                                                          \
    {                                                                                                                  \
        softfloat_exceptionFlags = 0;                                                                                  \
        softfloat_roundingMode = rounding_mode;                                                                        \
        dispatch_iterate_reduce<SignType::Unsigned>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, \
                                                    vlen, vl, inner_op);                                               \
        return 0;                                                                                                      \
    }

#define W_VFRED_OP(name, inner_op)                                                                                    \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,    \
                 uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl, \
                 uint8_t const rounding_mode)                                                                         \
    {                                                                                                                 \
        softfloat_exceptionFlags = 0;                                                                                 \
        softfloat_roundingMode = rounding_mode;                                                                       \
        dispatch_iterate_widening_reduce<SignType::Unsigned>(vector_field, vtype, masked_instruction_bit, vd, vs1,    \
                                                             vs2, vstart, vlen, vl, inner_op);                        \
        return 0;                                                                                                     \
    }

#define VV_OP(name, inner_op, sign)                                                                                   \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,    \
                 uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl) \
    {                                                                                                                 \
        dispatch_iterate_vv<sign>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen, vl,        \
                                  inner_op);                                                                          \
        return 0;                                                                                                     \
    }

#define W_VV_OP(name, inner_op, sign)                                                                                 \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,    \
                 uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl) \
    {                                                                                                                 \
        dispatch_iterate_widening_vv<sign>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen,   \
                                           vl, inner_op);                                                             \
        return 0;                                                                                                     \
    }

#define W_WV_OP(name, inner_op, sign)                                                                                 \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,    \
                 uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl) \
    {                                                                                                                 \
        dispatch_iterate_widening_wv<sign>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen,   \
                                           vl, inner_op);                                                             \
        return 0;                                                                                                     \
    }

#define N_WV_OP(name, inner_op, sign)                                                                                 \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,    \
                 uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl) \
    {                                                                                                                 \
        dispatch_iterate_narrowing_wv<sign>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen,  \
                                            vl, inner_op);                                                            \
        return 0;                                                                                                     \
    }

#define VI_OP(name, inner_op, sign, imm_extension)                                                                  \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,  \
                 uint8_t const vs2, uint8_t imm, uint16_t const vstart, uint16_t const vlen, uint16_t const vl)     \
    {                                                                                                               \
        dispatch_iterate_vi<sign, imm_extension>(vector_field, vtype, masked_instruction_bit, vd, vs2, imm, vstart, \
                                                 vlen, vl, inner_op);                                               \
        return 0;                                                                                                   \
    }

#define N_WI_OP(name, inner_op, sign, imm_extension)                                                                  \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t masked_instruction_bit, uint8_t const vd,    \
                 uint8_t const vs2, uint8_t imm, uint16_t const vstart, uint16_t const vlen, uint16_t const vl)       \
    {                                                                                                                 \
        dispatch_iterate_narrowing_wi<sign, imm_extension>(vector_field, vtype, masked_instruction_bit, vd, vs2, imm, \
                                                           vstart, vlen, vl, inner_op);                               \
        return 0;                                                                                                     \
    }

#define VX_OP(name, inner_op, sign)                                                                                \
    uint8_t name(void *const vector_field, void *const scalar_field, uint16_t const vtype,                         \
                 uint8_t masked_instruction_bit, uint8_t const vd, uint8_t const vs2, uint8_t rs1,                 \
                 uint16_t const vstart, uint16_t const vlen, uint16_t const vl, uint8_t xlen)                      \
    {                                                                                                              \
        dispatch_iterate_vx<sign>(vector_field, scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1, vstart, \
                                  vlen, xlen, vl, inner_op);                                                       \
        return 0;                                                                                                  \
    }

#define W_VX_OP(name, inner_op, sign)                                                                               \
    uint8_t name(void *const vector_field, void *const scalar_field, uint16_t const vtype,                          \
                 uint8_t masked_instruction_bit, uint8_t const vd, uint8_t const vs2, uint8_t rs1,                  \
                 uint16_t const vstart, uint16_t const vlen, uint16_t const vl, uint8_t xlen)                       \
    {                                                                                                               \
        dispatch_iterate_widening_vx<sign>(vector_field, scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1, \
                                           vstart, vlen, xlen, vl, inner_op);                                       \
        return 0;                                                                                                   \
    }

#define W_WX_OP(name, inner_op, sign)                                                                               \
    uint8_t name(void *const vector_field, void *const scalar_field, uint16_t const vtype,                          \
                 uint8_t masked_instruction_bit, uint8_t const vd, uint8_t const vs2, uint8_t rs1,                  \
                 uint16_t const vstart, uint16_t const vlen, uint16_t const vl, uint8_t xlen)                       \
    {                                                                                                               \
        dispatch_iterate_widening_wx<sign>(vector_field, scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1, \
                                           vstart, vlen, xlen, vl, inner_op);                                       \
        return 0;                                                                                                   \
    }

#define N_WX_OP(name, inner_op, sign)                                                                                \
    uint8_t name(void *const vector_field, void *const scalar_field, uint16_t const vtype,                           \
                 uint8_t masked_instruction_bit, uint8_t const vd, uint8_t const vs2, uint8_t rs1,                   \
                 uint16_t const vstart, uint16_t const vlen, uint16_t const vl, uint8_t xlen)                        \
    {                                                                                                                \
        dispatch_iterate_narrowing_wx<sign>(vector_field, scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1, \
                                            vstart, vlen, xlen, vl, inner_op);                                       \
        return 0;                                                                                                    \
    }

#define F_VV_OP(name, inner_op)                                                                                      \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t const masked_instruction_bit,               \
                 uint8_t const vd, uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, \
                 uint16_t const vl, uint8_t const rounding_mode)                                                     \
    {                                                                                                                \
        softfloat_exceptionFlags = 0;                                                                                \
        softfloat_roundingMode = rounding_mode;                                                                      \
        dispatch_iterate_vv<SignType::Unsigned>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart,   \
                                                vlen, vl, inner_op);                                                 \
        return 0;                                                                                                    \
    }

#define W_F_VV_OP(name, inner_op)                                                                                    \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t const masked_instruction_bit,               \
                 uint8_t const vd, uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, \
                 uint16_t const vl, uint8_t const rounding_mode)                                                     \
    {                                                                                                                \
        softfloat_exceptionFlags = 0;                                                                                \
        softfloat_roundingMode = rounding_mode;                                                                      \
        dispatch_iterate_widening_vv<SignType::Unsigned>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2,  \
                                                         vstart, vlen, vl, inner_op);                                \
        return 0;                                                                                                    \
    }

#define W_F_WV_OP(name, inner_op)                                                                                    \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t const masked_instruction_bit,               \
                 uint8_t const vd, uint8_t const vs1, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, \
                 uint16_t const vl, uint8_t const rounding_mode)                                                     \
    {                                                                                                                \
        softfloat_exceptionFlags = 0;                                                                                \
        softfloat_roundingMode = rounding_mode;                                                                      \
        dispatch_iterate_widening_wv<SignType::Unsigned>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2,  \
                                                         vstart, vlen, vl, inner_op);                                \
        return 0;                                                                                                    \
    }

#define F_VF_OP(name, inner_op)                                                                                      \
    uint8_t name(void *const vector_field, void *const float_scalar_field, uint16_t const vtype,                     \
                 uint8_t masked_instruction_bit, uint8_t const vd, uint8_t const vs2, uint8_t rs1,                   \
                 uint16_t const vstart, uint16_t const vlen, uint16_t const vl, uint8_t flen, uint8_t rounding_mode) \
    {                                                                                                                \
        softfloat_exceptionFlags = 0;                                                                                \
        softfloat_roundingMode = rounding_mode;                                                                      \
        dispatch_iterate_vf(vector_field, float_scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1, vstart,   \
                            vlen, flen, vl, inner_op);                                                               \
        return 0;                                                                                                    \
    }

#define W_F_VF_OP(name, inner_op)                                                                                    \
    uint8_t name(void *const vector_field, void *const float_scalar_field, uint16_t const vtype,                     \
                 uint8_t masked_instruction_bit, uint8_t const vd, uint8_t const vs2, uint8_t rs1,                   \
                 uint16_t const vstart, uint16_t const vlen, uint16_t const vl, uint8_t flen, uint8_t rounding_mode) \
    {                                                                                                                \
        softfloat_exceptionFlags = 0;                                                                                \
        softfloat_roundingMode = rounding_mode;                                                                      \
        dispatch_iterate_widening_vf(vector_field, float_scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1,  \
                                     vstart, vlen, flen, vl, inner_op);                                              \
        return 0;                                                                                                    \
    }

#define W_F_WF_OP(name, inner_op)                                                                                    \
    uint8_t name(void *const vector_field, void *const float_scalar_field, uint16_t const vtype,                     \
                 uint8_t masked_instruction_bit, uint8_t const vd, uint8_t const vs2, uint8_t rs1,                   \
                 uint16_t const vstart, uint16_t const vlen, uint16_t const vl, uint8_t flen, uint8_t rounding_mode) \
    {                                                                                                                \
        softfloat_exceptionFlags = 0;                                                                                \
        softfloat_roundingMode = rounding_mode;                                                                      \
        dispatch_iterate_widening_wf(vector_field, float_scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1,  \
                                     vstart, vlen, flen, vl, inner_op);                                              \
        return 0;                                                                                                    \
    }

#define F_V_UNARY_OP(name, inner_op)                                                                                 \
    uint8_t name(void *const vector_field, uint16_t const vtype, uint8_t const masked_instruction_bit,               \
                 uint8_t const vd, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen, uint16_t const vl, \
                 uint8_t const rounding_mode)                                                                        \
    {                                                                                                                \
        softfloat_exceptionFlags = 0;                                                                                \
        softfloat_roundingMode = rounding_mode;                                                                      \
        dispatch_iterate_v_unary<SignType::Unsigned>(vector_field, vtype, masked_instruction_bit, vd, vs2, vstart,   \
                                                     vlen, vl, inner_op);                                            \
        return 0;                                                                                                    \
    }

// 11. Vector Integer Arithmetic Instructions
// 11.1. Vector Single-Width Integer Add and Subtrac
VV_OP(vadd_vv, add_int, SignType::Signed)
VI_OP(vadd_vi, add_int, SignType::Signed, ImmExtensionType::SignExtend)
VX_OP(vadd_vx, add_int, SignType::Signed)
VV_OP(vsub_vv, sub_int, SignType::Signed)
VX_OP(vsub_vx, sub_int, SignType::Signed)
VX_OP(vrsub_vx, rsub_int, SignType::Signed)
VI_OP(vrsub_vi, rsub_int, SignType::Signed, ImmExtensionType::SignExtend)

// 11.2. Vector Widening Integer Add/Subtract
W_VV_OP(vwaddu_vv, add_int, SignType::Unsigned)
W_VX_OP(vwaddu_vx, add_int, SignType::Unsigned)
W_VV_OP(vwsubu_vv, sub_int, SignType::Unsigned)
W_VX_OP(vwsubu_vx, sub_int, SignType::Unsigned)
W_VV_OP(vwadd_vv, add_int, SignType::Signed) W_VX_OP(vwadd_vx, add_int, SignType::Signed)
    W_VV_OP(vwsub_vv, sub_int, SignType::Signed) W_VX_OP(vwsub_vx, sub_int, SignType::Signed)
        W_WV_OP(vwaddu_w_vv, add_int, SignType::Unsigned) W_WX_OP(vwaddu_w_vx, add_int, SignType::Unsigned)
            W_WV_OP(vwsubu_w_vv, sub_int, SignType::Unsigned) W_WX_OP(vwsubu_w_vx, sub_int, SignType::Unsigned)
                W_WV_OP(vwadd_w_vv, add_int, SignType::Signed) W_WX_OP(vwadd_w_vx, add_int, SignType::Signed)
                    W_WV_OP(vwsub_w_vv, sub_int, SignType::Signed) W_WX_OP(vwsub_w_vx, sub_int, SignType::Signed)

    // 11.3. Vector Integer Extension
    uint8_t vext_vf(void *const vector_field, uint16_t const vtype, uint8_t const vm, uint8_t const vd,
                    uint8_t const vs2, uint8_t const vs1, uint16_t const vstart, uint16_t const vlen, uint16_t const vl)
{
    auto const sew = decode_sew(vtype);

    static constexpr auto f8 = 0b1;
    static constexpr auto f4 = 0b10;
    static constexpr auto f2 = 0b11;

    switch (vs1 >> 1)
    {
    case f8:
        switch (sew)
        {
        case 64:
            dispatch_iterate_vext<64, 8>(vector_field, vtype, is_masked_instruction(vm), vd, vs2,
                                         static_cast<bool>(vs1 & 1), vstart, vlen, vl);
            break;
        default:
            break;
        }
        break;
    case f4:
        switch (sew)
        {
        case 32:
            dispatch_iterate_vext<32, 4>(vector_field, vtype, is_masked_instruction(vm), vd, vs2,
                                         static_cast<bool>(vs1 & 1), vstart, vlen, vl);
            break;
        case 64:
            dispatch_iterate_vext<64, 4>(vector_field, vtype, is_masked_instruction(vm), vd, vs2,
                                         static_cast<bool>(vs1 & 1), vstart, vlen, vl);
            break;
        default:
            break;
        }
        break;
    case f2:
        switch (sew)
        {
        case 16:
            dispatch_iterate_vext<16, 2>(vector_field, vtype, is_masked_instruction(vm), vd, vs2,
                                         static_cast<bool>(vs1 & 1), vstart, vlen, vl);
            break;
        case 32:
            dispatch_iterate_vext<32, 2>(vector_field, vtype, is_masked_instruction(vm), vd, vs2,
                                         static_cast<bool>(vs1 & 1), vstart, vlen, vl);
            break;
        case 64:
            dispatch_iterate_vext<64, 2>(vector_field, vtype, is_masked_instruction(vm), vd, vs2,
                                         static_cast<bool>(vs1 & 1), vstart, vlen, vl);
            break;
        default:
            break;
        }
        break;
    }

    return 0;
}

// 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
// TODO

// 11.5. Vector Bitwise Logical Instructions
VV_OP(vand_vv, and_int, SignType::Signed)
VI_OP(vand_vi, and_int, SignType::Signed, ImmExtensionType::SignExtend)
VX_OP(vand_vx, and_int, SignType::Signed)

VV_OP(vor_vv, or_int, SignType::Signed)
VI_OP(vor_vi, or_int, SignType::Signed, ImmExtensionType::SignExtend)
VX_OP(vor_vx, or_int, SignType::Signed)

VV_OP(vxor_vv, xor_int, SignType::Signed)
VI_OP(vxor_vi, xor_int, SignType::Signed, ImmExtensionType::SignExtend)
VX_OP(vxor_vx, xor_int, SignType::Signed)

// 11.6. Vector Single-Width Shift Instructions
VV_OP(vsll_vv, sll_int, SignType::Unsigned);
VI_OP(vsll_vi, sll_int, SignType::Unsigned, ImmExtensionType::ZeroExtend);
VX_OP(vsll_vx, sll_int, SignType::Unsigned);

VV_OP(vsrl_vv, srl_int, SignType::Unsigned);
VI_OP(vsrl_vi, srl_int, SignType::Unsigned, ImmExtensionType::ZeroExtend);
VX_OP(vsrl_vx, srl_int, SignType::Unsigned);

VV_OP(vsra_vv, sra_int, SignType::Signed);
VI_OP(vsra_vi, sra_int, SignType::Signed, ImmExtensionType::ZeroExtend);
VX_OP(vsra_vx, sra_int, SignType::Signed);

// 11.7. Vector Narrowing Integer Right Shift Instructions
N_WV_OP(vnsrl_wv, srl_int, SignType::Unsigned)
N_WI_OP(vnsrl_wi, srl_int, SignType::Unsigned, ImmExtensionType::ZeroExtend)
N_WX_OP(vnsrl_wx, srl_int, SignType::Unsigned)

N_WV_OP(vnsra_wv, sra_int, SignType::Signed)
N_WI_OP(vnsra_wi, sra_int, SignType::Signed, ImmExtensionType::ZeroExtend)
N_WX_OP(vnsra_wx, sra_int, SignType::Signed)

// 11.8. Vector Integer Compare Instructions
VV_OP(vmseq_vv, eq_int, SignType::Signed)
VI_OP(vmseq_vi, eq_int, SignType::Signed, ImmExtensionType::SignExtend)
VX_OP(vmseq_vx, eq_int, SignType::Signed)

VV_OP(vmsne_vv, ne_int, SignType::Signed)
VI_OP(vmsne_vi, ne_int, SignType::Signed, ImmExtensionType::SignExtend)
VX_OP(vmsne_vx, ne_int, SignType::Signed)

VV_OP(vmsltu_vv, ltu_int, SignType::Unsigned)
VX_OP(vmsltu_vx, ltu_int, SignType::Unsigned)

VV_OP(vmslt_vv, lt_int, SignType::Signed)
VX_OP(vmslt_vx, lt_int, SignType::Signed)

VV_OP(vmsleu_vv, leu_int, SignType::Unsigned)
VI_OP(vmsleu_vi, leu_int, SignType::Unsigned, ImmExtensionType::SignExtend)
VX_OP(vmsleu_vx, leu_int, SignType::Unsigned)

VV_OP(vmsle_vv, le_int, SignType::Signed)
VI_OP(vmsle_vi, le_int, SignType::Signed, ImmExtensionType::SignExtend)
VX_OP(vmsle_vx, le_int, SignType::Signed)

VV_OP(vmsgtu_vv, gtu_int, SignType::Unsigned)
VX_OP(vmsgtu_vx, gtu_int, SignType::Unsigned)

VV_OP(vmsgt_vv, gt_int, SignType::Signed)
VX_OP(vmsgt_vx, gt_int, SignType::Signed)

// 11.9. Vector Integer Min/Max Instructions
VV_OP(vminu_vv, minu_int, SignType::Unsigned)
VX_OP(vminu_vx, minu_int, SignType::Unsigned)

VV_OP(vmin_vv, min_int, SignType::Signed)
VX_OP(vmin_vx, min_int, SignType::Signed)

VV_OP(vmaxu_vv, maxu_int, SignType::Unsigned)
VX_OP(vmaxu_vx, maxu_int, SignType::Unsigned)

VV_OP(vmax_vv, max_int, SignType::Signed)
VX_OP(vmax_vx, max_int, SignType::Signed)

// 11.10. Vector Single-Width Integer Multiply Instructions
VV_OP(vmul_vv, mul_int, SignType::Signed)
VX_OP(vmul_vx, mul_int, SignType::Signed)

VV_OP(vmulh_vv, mulh_int, SignType::Signed)
VX_OP(vmulh_vx, mulh_int, SignType::Signed)

VV_OP(vmulhu_vv, mulhu_int, SignType::Unsigned)
VX_OP(vmulhu_vx, mulhu_int, SignType::Unsigned)

// Use signed instruction, mask off unsigned operand later in mulhsu_int
VV_OP(vmulhsu_vv, mulhsu_int, SignType::Signed)
VX_OP(vmulhsu_vx, mulhsu_int, SignType::Signed)

// 11.11. Vector Integer Divide Instructions
VV_OP(vdivu_vv, divu_int, SignType::Unsigned)
VX_OP(vdivu_vx, divu_int, SignType::Unsigned)

VV_OP(vdiv_vv, div_int, SignType::Signed)
VX_OP(vdiv_vx, div_int, SignType::Signed)

VV_OP(vremu_vv, remu_int, SignType::Unsigned)
VX_OP(vremu_vx, remu_int, SignType::Unsigned)

VV_OP(vrem_vv, rem_int, SignType::Signed)
VX_OP(vrem_vx, rem_int, SignType::Signed)

// 11.12. Vector Widening Integer Multiply Instructions
W_VV_OP(vwmul_vv, mul_int, SignType::Signed)
W_VX_OP(vwmul_vx, mul_int, SignType::Signed)

W_VV_OP(vwmulu_vv, mulu_int, SignType::Unsigned)
W_VX_OP(vwmulu_vx, mulu_int, SignType::Unsigned)

W_VV_OP(vwmulsu_vv, mulsu_int, SignType::Signed)
W_VX_OP(vwmulsu_vx, mulsu_int, SignType::Signed)

// 11.13. Vector Single-Width Integer Multiply-Add Instructions
VV_OP(vmacc_vv, macc, SignType::Signed)
VX_OP(vmacc_vx, macc, SignType::Signed)

VV_OP(vnmsac_vv, nmsac, SignType::Signed)
VX_OP(vnmsac_vx, nmsac, SignType::Signed)

VV_OP(vmadd_vv, madd, SignType::Signed)
VX_OP(vmadd_vx, madd, SignType::Signed)

VV_OP(vnmsub_vv, nmsub, SignType::Signed)
VX_OP(vnmsub_vx, nmsub, SignType::Signed)

// 11.14. Vector Widening Integer Multiply-Add Instructions
W_VV_OP(vwmaccu_vv, macc, SignType::Unsigned)
W_VX_OP(vwmaccu_vx, macc, SignType::Unsigned)

W_VV_OP(vwmacc_vv, macc, SignType::Signed)
W_VX_OP(vwmacc_vx, macc, SignType::Signed)

W_VV_OP(vwmaccsu_vv, maccsu, SignType::Signed)
W_VX_OP(vwmaccsu_vx, maccsu, SignType::Signed)

W_VX_OP(vwmaccus_vx, maccus, SignType::Signed)

// 11.15. Vector Integer Merge Instructions
// TODO

// 13. Vector Floating-Point Instructions
// 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions
F_VV_OP(vfadd_vv, add_float)
F_VF_OP(vfadd_vf, add_float)

F_VV_OP(vfsub_vv, sub_float)
F_VF_OP(vfsub_vf, sub_float)

F_VF_OP(vfrsub_vf, rsub_float)

// 13.3. Vector Widening Floating-Point Add/Subtract Instructions
W_F_VV_OP(vfwadd_vv, wadd_float)
W_F_VF_OP(vfwadd_vf, wadd_float)

W_F_VV_OP(vfwsub_vv, wsub_float)
W_F_VF_OP(vfwsub_vf, wsub_float)

W_F_WV_OP(vfwadd_wv, wadd_w_float)
W_F_WF_OP(vfwadd_wf, wadd_w_float)

W_F_WV_OP(vfwsub_wv, wsub_w_float)
W_F_WF_OP(vfwsub_wf, wsub_w_float)

// 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
F_VV_OP(vfmul_vv, mul_float)
F_VF_OP(vfmul_vf, mul_float)

F_VV_OP(vfdiv_vv, div_float)
F_VF_OP(vfdiv_vf, div_float)

F_VF_OP(vfrdiv_vf, rdiv_float)

// 13.5. Vector Widening Floating-Point Multiply
W_F_VV_OP(vfwmul_vv, wmul_float)
W_F_VF_OP(vfwmul_vf, wmul_float)

// 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
F_VV_OP(vfmacc_vv, macc_float)
F_VF_OP(vfmacc_vf, macc_float)

F_VV_OP(vfnmacc_vv, nmacc_float)
F_VF_OP(vfnmacc_vf, nmacc_float)

F_VV_OP(vfmsac_vv, msac_float)
F_VF_OP(vfmsac_vf, msac_float)

F_VV_OP(vfnmsac_vv, nmsac_float)
F_VF_OP(vfnmsac_vf, nmsac_float)

F_VV_OP(vfmadd_vv, madd_float)
F_VF_OP(vfmadd_vf, madd_float)

F_VV_OP(vfnmadd_vv, nmadd_float)
F_VF_OP(vfnmadd_vf, nmadd_float)

F_VV_OP(vfmsub_vv, msub_float)
F_VF_OP(vfmsub_vf, msub_float)

F_VV_OP(vfnmsub_vv, nmsub_float)
F_VF_OP(vfnmsub_vf, nmsub_float)

// 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
W_F_VV_OP(vfwmacc_vv, wmacc_float)
W_F_VF_OP(vfwmacc_vf, wmacc_float)

W_F_VV_OP(vfwnmacc_vv, wnmacc_float)
W_F_VF_OP(vfwnmacc_vf, wnmacc_float)

W_F_VV_OP(vfwmsac_vv, wmsac_float)
W_F_VF_OP(vfwmsac_vf, wmsac_float)

W_F_VV_OP(vfwnmsac_vv, wnmsac_float)
W_F_VF_OP(vfwnmsac_vf, wnmsac_float)

// 13.8. Vector Floating-Point Square-Root Instruction
F_V_UNARY_OP(vfsqrt_v, sqrt_float)

// 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
F_V_UNARY_OP(vfrsqrt7_v, rsqrt7_float)

// 13.10. Vector Floating-Point Reciprocal Estimate Instruction
F_V_UNARY_OP(vfrec7_v, rec7_float)

// 13.11. Vector Floating-Point MIN/MAX Instructions
F_VV_OP(vfmin_vv, min_float)
F_VF_OP(vfmin_vf, min_float)

F_VV_OP(vfmax_vv, max_float)
F_VF_OP(vfmax_vf, max_float)

// 13.12. Vector Floating-Point Sign-Injection Instructions
F_VV_OP(vfsgnj_vv, sgnj_float)
F_VF_OP(vfsgnj_vf, sgnj_float)

F_VV_OP(vfsgnjn_vv, sgnjn_float)
F_VF_OP(vfsgnjn_vf, sgnjn_float)

F_VV_OP(vfsgnjx_vv, sgnjx_float)
F_VF_OP(vfsgnjx_vf, sgnjx_float)

// 13.13. Vector Floating-Point Compare Instructions
F_VV_OP(vmfeq_vv, eq_float)
F_VF_OP(vmfeq_vf, eq_float)

F_VV_OP(vmfne_vv, ne_float)
F_VF_OP(vmfne_vf, ne_float)

F_VV_OP(vmflt_vv, lt_float)
F_VF_OP(vmflt_vf, lt_float)

F_VV_OP(vmfle_vv, le_float)
F_VF_OP(vmfle_vf, le_float)

F_VF_OP(vmfgt_vf, gt_float)

F_VF_OP(vmfge_vf, ge_float)

// 13.14. Vector Floating-Point Classify Instruction
F_V_UNARY_OP(vfclass_v, classify_float)

// 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions
F_V_UNARY_OP(vfcvt_xu_f_v, convert_xu_f)
F_V_UNARY_OP(vfcvt_x_f_v, convert_x_f)

F_V_UNARY_OP(vfcvt_rtz_xu_f_v, convert_rtz_xu_f)
F_V_UNARY_OP(vfcvt_rtz_x_f_v, convert_rtz_x_f)

F_V_UNARY_OP(vfcvt_f_xu_v, convert_f_xu)
F_V_UNARY_OP(vfcvt_f_x_v, convert_f_x)

// 13.18. Widening Floating-Point/Integer Type-Convert Instructions
F_V_UNARY_OP(vfwcvt_xu_f_v, convert_widening_xu_f)
F_V_UNARY_OP(vfwcvt_x_f_v, convert_widening_x_f)

F_V_UNARY_OP(vfwcvt_rtz_xu_f_v, convert_widening_rtz_xu_f)
F_V_UNARY_OP(vfwcvt_rtz_x_f_v, convert_widening_rtz_x_f)

F_V_UNARY_OP(vfwcvt_f_xu_v, convert_widening_f_xu)
F_V_UNARY_OP(vfwcvt_f_x_v, convert_widening_f_x)

F_V_UNARY_OP(vfwcvt_f_f_v, convert_widening_f_f)

// 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions
F_V_UNARY_OP(vfncvt_xu_f_w, convert_narrowing_xu_f)
F_V_UNARY_OP(vfncvt_x_f_w, convert_narrowing_x_f)

F_V_UNARY_OP(vfncvt_rtz_xu_f_w, convert_narrowing_rtz_xu_f)
F_V_UNARY_OP(vfncvt_rtz_x_f_w, convert_narrowing_rtz_x_f)

F_V_UNARY_OP(vfncvt_f_xu_w, convert_narrowing_f_xu)
F_V_UNARY_OP(vfncvt_f_x_w, convert_narrowing_f_x)

F_V_UNARY_OP(vfncvt_f_f_w, convert_narrowing_f_f)

uint8_t vfncvt_rod_f_f_w(void *const vector_field, uint16_t const vtype, uint8_t const masked_instruction_bit,
                         uint8_t const vd, uint8_t const vs2, uint16_t const vstart, uint16_t const vlen,
                         uint16_t const vl, uint8_t const rounding_mode)
{
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = softfloat_round_odd;
    dispatch_iterate_v_unary<SignType::Unsigned>(vector_field, vtype, masked_instruction_bit, vd, vs2, vstart, vlen, vl,
                                                 convert_narrowing_f_f);
    return 0;
}

// 14. Vector Reduction Operations
// 14.1. Vector Single-Width Integer Reduction Instructions
VRED_OP(vredsum_vs, add_int, SignType::Signed)
VRED_OP(vredmaxu_vs, maxu_int, SignType::Unsigned)
VRED_OP(vredmax_vs, max_int, SignType::Signed)
VRED_OP(vredminu_vs, minu_int, SignType::Unsigned)
VRED_OP(vredmin_vs, min_int, SignType::Signed)
VRED_OP(vredand_vs, and_int, SignType::Signed)
VRED_OP(vredor_vs, or_int, SignType::Signed)
VRED_OP(vredxor_vs, xor_int, SignType::Signed)

// 14.2. Vector Widening Integer Reduction Instructions
W_VRED_OP(vwredsumu_vs, add_int, SignType::Unsigned)
W_VRED_OP(vwredsum_vs, add_int, SignType::Signed)

// 14.3. Vector Single-Width Floating-Point Reduction Instructions
VFRED_OP(vfredosum_vs, add_float)
VFRED_OP(vfredusum_vs, add_float)
VFRED_OP(vfredmax_vs, max_float)
VFRED_OP(vfredmin_vs, min_float)

// 14.4. Vector Widening Floating-Point Reduction Instructions
W_VFRED_OP(vfwredosum_vs, add_float)
W_VFRED_OP(vfwredusum_vs, add_float)

// 15. Vector Mask Instructions
// 15.1. Vector Mask-Register Logical Instructions
VV_OP(vmand_mm, and_mask, SignType::Unsigned)
VV_OP(vmnand_mm, nand_mask, SignType::Unsigned)
VV_OP(vmandn_mm, andn_mask, SignType::Unsigned)
VV_OP(vmxor_mm, xor_mask, SignType::Unsigned)
VV_OP(vmor_mm, or_mask, SignType::Unsigned)
VV_OP(vmnor_mm, nor_mask, SignType::Unsigned)
VV_OP(vmorn_mm, orn_mask, SignType::Unsigned)
VV_OP(vmxnor_mm, xnor_mask, SignType::Unsigned)

/* --- Private function definitions --- */

template <typename VectorElementType, bool IsSigned>
auto dump_v_register(unsigned v_register, unsigned vlen, void *const vector_field) -> void
{
    auto *const vector_elements = static_cast<uint8_t *>(vector_field);
    auto const sew = sizeof(uint8_t) * 8;
    auto const elements_per_register = vlen / sew;
    auto const v_base = v_register * elements_per_register;

    std::printf("v%u: ", v_register);
    if constexpr (IsSigned)
    {
        for (int i = 0; i < elements_per_register; ++i)
        {
            std::printf("%i | ", vector_elements[v_base + i]);
        }
        std::printf("\n");
    }
    else
    {
        for (int i = 0; i < elements_per_register; ++i)
        {
            std::printf("%x ", vector_elements[v_base + i]);
        }
        std::printf("\n");
    }
}

template <SignType Sign>
inline constexpr uint64_t get_scalar(void *const scalar_field, unsigned const sew, unsigned const xlen,
                                     unsigned const rs1)
{
    uint64_t scalar = 0;
    uint64_t const sew_mask = (1_u64 << sew) - 1;
    switch (xlen)
    {
    case 32:
        scalar = static_cast<uint64_t>((static_cast<uint32_t *>(scalar_field))[rs1]) & sew_mask;
        break;
    case 64:
        scalar = (static_cast<uint64_t *>(scalar_field))[rs1] & sew_mask;
        break;
    default:
        // Invalid XLEN!
        break;
    }

    if constexpr (Sign == SignType::Signed)
    {
        bool const msb_is_set = scalar & (1 << (sew - 1));
        scalar |= (msb_is_set * (~sew_mask));
    }

    return scalar;
}

inline constexpr uint64_t get_float_scalar(void *const float_scalar_field, unsigned const sew, unsigned const flen,
                                           unsigned const rs1)
{
    uint64_t scalar = 0;
    uint64_t const sew_mask = (1_u64 << sew) - 1;
    switch (flen)
    {
    case 32:
        scalar = static_cast<uint64_t>((static_cast<uint32_t *>(float_scalar_field))[rs1]) & sew_mask;
        break;
    case 64:
        scalar = (static_cast<uint64_t *>(float_scalar_field))[rs1] & sew_mask;
        break;
    default:
        // Invalid XLEN!
        break;
    }

    if (flen > sew)
    {
        switch (sew)
        {
        case 16:
            scalar = check_and_unbox_f16(f64(scalar)).v;
            break;
        case 32:
            scalar = check_and_unbox_f32(f64(scalar)).v;
            break;
        default:
            // Illegal
            break;
        }
    }

    return scalar;
}

inline constexpr auto decode_sew(uint32_t const vtype) -> unsigned
{
    constexpr auto sew_offset = 3;
    constexpr auto sew_bitmask = 0b11;

    auto const vsew = (vtype >> sew_offset) & sew_bitmask;
    return 8 << vsew;
}

inline constexpr auto is_masked_element(bool const element_mask_bit) -> bool
{
    return !element_mask_bit;
}

inline constexpr auto is_masked_instruction(bool const instruction_mask_bit) -> bool
{
    return !instruction_mask_bit;
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void reduce_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                     unsigned const vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                     OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);
    uint64_t reduction_accumulator = vector_elements[vs1_base];
    if (vl == 0)
    {
        return;
    }

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (mask_bit == masked_element_value)
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            reduction_accumulator = op(reduction_accumulator, vector_elements[vs2_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpSewData>)
        {
            reduction_accumulator = op(reduction_accumulator, vector_elements[vs2_base + i], static_cast<SewType>(sew));
        }
        else
        {
            static_assert(false, "Invalid operation for reduce");
        }
    }
    vector_elements[vd_base] = reduction_accumulator;
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_reduce_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                              unsigned const vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                              OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    using WideType = TypeWidener<VectorElementType>::wide_type;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);
    auto *const wide_elements = static_cast<WideType *>(vector_field);
    uint64_t reduction_accumulator = wide_elements[vs1_base];
    if (vl == 0)
    {
        return;
    }

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (mask_bit == masked_element_value)
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            reduction_accumulator = op(reduction_accumulator, vector_elements[vs2_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpSewData>)
        {
            reduction_accumulator = op(reduction_accumulator, vector_elements[vs2_base + i], static_cast<SewType>(sew));
        }
        else
        {
            static_assert(false, "Invalid operation for widening reduce");
        }
    }
    wide_elements[vd_base] = reduction_accumulator;
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void vv_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                 unsigned const vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                 OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            // Casting signed to larger unsigned will sign extend.
            // As vector elements can be interpreted as int or uint, this should already take care of signed/unsigned
            // instructions
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpSewData>)
        {
            vector_elements[vd_base + i] =
                op(vector_elements[vs2_base + i], vector_elements[vs1_base + i], static_cast<SewType>(sew));
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpMaskData>)
        {
            auto const mask_bit = static_cast<Bit>((vector_elements[i / sew] >> (i % sew)) & 1);
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i], mask_bit);
        }
        else if constexpr (std::is_same_v<OpType, BitResultOp>)
        {
            // Clear bit
            vector_elements[vd_base + (i / sew)] &= ~(1_u64 << (i % sew));
            // Conditionally set bit
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], vector_elements[vs1_base + i])
                                                    << (i % sew);
        }
        else if constexpr (std::is_same_v<OpType, BitResultOpSewData>)
        {
            // Clear bit
            vector_elements[vd_base + (i / sew)] &= ~(1_u64 << (i % sew));
            // Conditionally set bit
            vector_elements[vd_base + (i / sew)] |=
                op(vector_elements[vs2_base + i], vector_elements[vs1_base + i], static_cast<SewType>(sew))
                << (i % sew);
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOp>)
        {
            // Accumulator instructions have vs2 and vs1 elements switched compared to e.g. vadd.vv
            // I.e. here vs2 is rhs and vs1 lhs
            vector_elements[vd_base + i] =
                op(vector_elements[vs1_base + i], vector_elements[vs2_base + i], vector_elements[vd_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOpSewData>)
        {
            // Accumulator instructions have vs2 and vs1 elements switched compared to e.g. vadd.vv
            // I.e. here vs2 is rhs and vs1 lhs
            vector_elements[vd_base + i] = op(vector_elements[vs1_base + i], vector_elements[vs2_base + i],
                                              vector_elements[vd_base + i], static_cast<SewType>(sew));
        }
        else if constexpr (std::is_same_v<OpType, MaskOp>)
        {
            Bit const lhs = static_cast<Bit>((vector_elements[vs2_base + (i / sew)] >> (i % sew)) & 1);
            Bit const rhs = static_cast<Bit>((vector_elements[vs1_base + (i / sew)] >> (i % sew)) & 1);
            // Clear bit
            vector_elements[vd_base + (i / sew)] &= ~(1_u64 << (i % sew));
            // Conditionally set bit
            vector_elements[vd_base + (i / sew)] |= op(lhs, rhs) << (i % sew);
        }
        else
        {
            static_assert(false, "Invalid operation for vv");
        }
    }
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_vv_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                          unsigned vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                          OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto *const wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            wide_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpSewData>)
        {
            wide_elements[vd_base + i] =
                op(vector_elements[vs2_base + i], vector_elements[vs1_base + i], static_cast<SewType>(sew));
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOp>)
        {
            // Accumulator instructions have vs2 and vs1 elements switched compared to e.g. vadd.vv
            // I.e. here vs2 is rhs and vs1 lhs
            wide_elements[vd_base + i] =
                op(vector_elements[vs1_base + i], vector_elements[vs2_base + i], wide_elements[vd_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOpSewData>)
        {
            // Accumulator instructions have vs2 and vs1 elements switched compared to e.g. vadd.vv
            // I.e. here vs2 is rhs and vs1 lhs
            wide_elements[vd_base + i] = op(vector_elements[vs1_base + i], vector_elements[vs2_base + i],
                                            wide_elements[vd_base + i], static_cast<SewType>(sew));
        }
        else
        {
            static_assert(false, "Invalid operation for widening vv");
        }
    }
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_wv_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                          unsigned vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                          OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto *const wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            wide_elements[vd_base + i] = op(wide_elements[vs2_base + i], vector_elements[vs1_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpSewData>)
        {
            wide_elements[vd_base + i] =
                op(wide_elements[vs2_base + i], vector_elements[vs1_base + i], static_cast<SewType>(sew));
        }
        else
        {
            static_assert(false, "Invalid operation for widening wx");
        }
    }
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, ValueResultOpSewData>
inline constexpr void narrowing_wv_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                           unsigned vd_base, unsigned const vs1_base, unsigned const vs2_base,
                                           OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto *const wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }
        // Only ValueResultOpSewData for now
        vector_elements[vd_base + i] =
            op(wide_elements[vs2_base + i], vector_elements[vs1_base + i], static_cast<SewType>(sew * 2));
    }
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, ValueResultOpSewData>
inline constexpr void narrowing_wxi_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                            unsigned const vd_base, unsigned const vs2_base, uint64_t const scalar,
                                            OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto *const wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }
        // Only ValueResultOpSewData for now
        vector_elements[vd_base + i] = op(wide_elements[vs2_base + i], scalar, static_cast<SewType>(sew * 2));
    }
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void vxi_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                  unsigned const vd_base, unsigned const vs2_base, uint64_t const scalar,
                                  OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar);
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpSewData>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar, static_cast<SewType>(sew));
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpMaskData>)
        {
            auto const mask_bit = static_cast<Bit>((vector_elements[i / sew] >> (i % sew)) & 1);
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar, mask_bit);
        }
        else if constexpr (std::is_same_v<OpType, BitResultOp>)
        {
            // Clear bit
            vector_elements[vd_base + (i / sew)] &= ~(1_u64 << (i % sew));
            // Conditionally set bit
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], scalar) << (i % sew);
        }
        else if constexpr (std::is_same_v<OpType, BitResultOpSewData>)
        {
            // Clear bit
            vector_elements[vd_base + (i / sew)] &= ~(1_u64 << (i % sew));
            // Conditionally set bit
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], scalar, static_cast<SewType>(sew))
                                                    << (i % sew);
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOp>)
        {
            // Accumulator instructions have vs2 and vs1 elements switched compared to e.g. vadd.vv
            // I.e. here vs2 is rhs and the scalar is lhs
            vector_elements[vd_base + i] = op(scalar, vector_elements[vs2_base + i], vector_elements[vd_base + i]);
        }
    }
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_vx_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                          unsigned const vd_base, unsigned const vs2_base, uint64_t const scalar,
                                          OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto *const wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            wide_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar);
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpSewData>)
        {
            wide_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar, static_cast<SewType>(sew));
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOp>)
        {
            wide_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar, wide_elements[vd_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOpSewData>)
        {
            wide_elements[vd_base + i] =
                op(vector_elements[vs2_base + i], scalar, wide_elements[vd_base + i], static_cast<SewType>(sew));
        }
        else
        {
            static_assert(false, "This operation is not supported");
        }
    }
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void widening_wx_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                          unsigned const vd_base, unsigned const vs2_base, uint64_t const scalar,
                                          OpType const op)
{
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto *const wide_elements = static_cast<WideElementType *>(vector_field);
    constexpr auto wide_sew = sizeof(WideElementType) * 8;

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((wide_elements[i / (wide_sew)] >> (i % wide_sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            wide_elements[vd_base + i] = op(wide_elements[vs2_base + i], scalar);
        }
        else if constexpr (std::is_same_v<OpType, ValueResultOpSewData>)
        {
            constexpr auto sew = static_cast<SewType>(sizeof(VectorElementType) * 8);
            wide_elements[vd_base + i] = op(wide_elements[vs2_base + i], scalar, sew);
        }
        else
        {
            static_assert(false, "Invalid operation for widening wx");
        }
    }
}

template <typename SourceType, typename DestType, MaskType Masked>
    requires ValidVectorElementType<SourceType> and ValidVectorElementType<DestType>
inline constexpr void vext_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                   unsigned const vd_base, unsigned const vs2_base)
{
    auto *const source_elements = static_cast<SourceType *>(vector_field);
    auto *const dest_elements = static_cast<DestType *>(vector_field);
    constexpr auto sew = sizeof(DestType) << 3;
    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Masked == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((dest_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }
        dest_elements[vd_base + i] = static_cast<DestType>(source_elements[vs2_base + i]);
    }
}

template <typename VectorElementType, MaskType Mask, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline constexpr void v_unary_iterate(void *const vector_field, uint16_t const vstart, uint16_t const vl,
                                      unsigned const vd_base, unsigned const vs2_base, OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Mask == MaskType::Masked)
        {
            auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
            if (is_masked_element(mask_bit))
            {
                continue;
            }
        }

        if constexpr (std::is_same_v<OpType, UnaryOpSewData>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], static_cast<SewType>(sew));
        }
        else
        {
            static_assert(false, "Invalid operation for vv");
        }
    }
}

#define VECTOR_ITERATOR_SWITCH(iterator)                                                                               \
    using elm_8_t = ElementTypeMap<8, Sign>::element_type;                                                             \
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;                                                           \
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;                                                           \
    using elm_64_t = ElementTypeMap<64, Sign>::element_type;                                                           \
                                                                                                                       \
    switch (sew)                                                                                                       \
    {                                                                                                                  \
    case sew_8:                                                                                                        \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_8_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_8_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base,     \
                                                            op);                                                       \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_16:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_16_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op); \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_16_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base,    \
                                                             op);                                                      \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_32:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_32_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op); \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_32_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base,    \
                                                             op);                                                      \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_64:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_64_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op); \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_64_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base,    \
                                                             op);                                                      \
        }                                                                                                              \
        break;                                                                                                         \
    default:                                                                                                           \
        break;                                                                                                         \
    }

#define WIDE_VECTOR_ITERATOR_SWITCH(iterator)                                                                          \
    using elm_8_t = ElementTypeMap<8, Sign>::element_type;                                                             \
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;                                                           \
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;                                                           \
                                                                                                                       \
    switch (sew)                                                                                                       \
    {                                                                                                                  \
    case sew_8:                                                                                                        \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_8_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_8_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base,     \
                                                            op);                                                       \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_16:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_16_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op); \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_16_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base,    \
                                                             op);                                                      \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_32:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_32_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op); \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_32_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base,    \
                                                             op);                                                      \
        }                                                                                                              \
        break;                                                                                                         \
    default:                                                                                                           \
        break;                                                                                                         \
    }

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_vv(void *const vector_field, uint16_t const vtype,
                                                  uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                  uint8_t const vs2, uint16_t const vstart, uint16_t const vlen,
                                                  uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs1_base = vs1 * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    VECTOR_ITERATOR_SWITCH(vv)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_vv(void *const vector_field, uint16_t const vtype,
                                                           uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                           uint8_t const vs2, uint16_t const vstart,
                                                           uint16_t const vlen, uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * (elements_per_register >> 1);
    auto const vs1_base = vs1 * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    WIDE_VECTOR_ITERATOR_SWITCH(widening_vv)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_reduce(void *const vector_field, uint16_t const vtype,
                                                      uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                      uint8_t const vs2, uint16_t const vstart, uint16_t const vlen,
                                                      uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs1_base = vs1 * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    VECTOR_ITERATOR_SWITCH(reduce)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_reduce(void *const vector_field, uint16_t const vtype,
                                                               uint8_t const mask_bit, uint8_t const vd,
                                                               uint8_t const vs1, uint8_t const vs2,
                                                               uint16_t const vstart, uint16_t const vlen,
                                                               uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * (elements_per_register >> 1);
    auto const vs1_base = vs1 * (elements_per_register >> 1);
    auto const vs2_base = vs2 * elements_per_register;

    WIDE_VECTOR_ITERATOR_SWITCH(widening_reduce)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_wv(void *const vector_field, uint16_t const vtype,
                                                           uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                           uint8_t const vs2, uint16_t const vstart,
                                                           uint16_t const vlen, uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * (elements_per_register >> 1);
    auto const vs1_base = vs1 * elements_per_register;
    auto const vs2_base = vs2 * (elements_per_register >> 1);

    WIDE_VECTOR_ITERATOR_SWITCH(widening_wv)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_narrowing_wv(void *const vector_field, uint16_t const vtype,
                                                            uint8_t const mask_bit, uint8_t const vd, uint8_t const vs1,
                                                            uint8_t const vs2, uint16_t const vstart,
                                                            uint16_t const vlen, uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs1_base = vs1 * elements_per_register;
    auto const vs2_base = vs2 * (elements_per_register >> 1);

    WIDE_VECTOR_ITERATOR_SWITCH(narrowing_wv)
}

#define SCALAR_ITERATOR_SWITCH(iterator)                                                                               \
    using elm_8_t = ElementTypeMap<8, Sign>::element_type;                                                             \
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;                                                           \
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;                                                           \
    using elm_64_t = ElementTypeMap<64, Sign>::element_type;                                                           \
    switch (sew)                                                                                                       \
    {                                                                                                                  \
    case sew_8:                                                                                                        \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_8_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);    \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_8_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);  \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_16:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_16_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_16_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op); \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_32:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_32_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_32_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op); \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_64:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_64_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_64_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op); \
        }                                                                                                              \
        break;                                                                                                         \
    default:                                                                                                           \
        break;                                                                                                         \
    }

#define WIDE_SCALAR_ITERATOR_SWITCH(iterator)                                                                          \
    using elm_8_t = ElementTypeMap<8, Sign>::element_type;                                                             \
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;                                                           \
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;                                                           \
    switch (sew)                                                                                                       \
    {                                                                                                                  \
    case sew_8:                                                                                                        \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_8_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);    \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_8_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);  \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_16:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_16_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_16_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op); \
        }                                                                                                              \
        break;                                                                                                         \
    case sew_32:                                                                                                       \
        if (is_masked_instruction(static_cast<bool>(mask_bit)))                                                        \
        {                                                                                                              \
            iterator##_iterate<elm_32_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            iterator##_iterate<elm_32_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op); \
        }                                                                                                              \
        break;                                                                                                         \
    default:                                                                                                           \
        break;                                                                                                         \
    }

template <SignType Sign, ImmExtensionType ImmExtension, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_vi(void *const vector_field, uint16_t const vtype,
                                                  uint8_t const mask_bit, uint8_t const vd, uint8_t const vs2,
                                                  uint8_t immediate, uint16_t const vstart, uint16_t const vlen,
                                                  uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    uint64_t scalar = immediate;
    if constexpr (ImmExtension == ImmExtensionType::SignExtend)
    {
        scalar = sign_extend_immediate(scalar);
    }

    SCALAR_ITERATOR_SWITCH(vxi)
}

template <SignType Sign, ImmExtensionType ImmExtension, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_narrowing_wi(void *const vector_field, uint16_t const vtype,
                                                            uint8_t const mask_bit, uint8_t const vd, uint8_t const vs2,
                                                            uint8_t immediate, uint16_t const vstart,
                                                            uint16_t const vlen, uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * (elements_per_register >> 1);

    uint64_t scalar = immediate;
    if constexpr (ImmExtension == ImmExtensionType::SignExtend)
    {
        scalar = sign_extend_immediate(scalar);
    }

    WIDE_SCALAR_ITERATOR_SWITCH(narrowing_wxi)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_vx(void *const vector_field, void *const scalar_field,
                                                  uint16_t const vtype, uint8_t const mask_bit, uint8_t const vd,
                                                  uint8_t const vs2, uint8_t const rs1, uint16_t const vstart,
                                                  uint16_t const vlen, uint16_t xlen, uint16_t const vl,
                                                  OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    auto const scalar = get_scalar<Sign>(scalar_field, sew, xlen, rs1);

    SCALAR_ITERATOR_SWITCH(vxi)
}

template <typename OpType, SignType Sign>
inline constexpr GO_FAST void dispatch_iterate_vf(void *const vector_field, void *const scalar_field,
                                                  uint16_t const vtype, uint8_t const mask_bit, uint8_t const vd,
                                                  uint8_t const vs2, uint8_t const rs1, uint16_t const vstart,
                                                  uint16_t const vlen, uint16_t flen, uint16_t const vl,
                                                  OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    auto const scalar = get_float_scalar(scalar_field, sew, flen, rs1);

    SCALAR_ITERATOR_SWITCH(vxi)
}

template <typename OpType, SignType Sign>
inline constexpr GO_FAST void dispatch_iterate_widening_vf(void *const vector_field, void *const scalar_field,
                                                           uint16_t const vtype, uint8_t const mask_bit,
                                                           uint8_t const vd, uint8_t const vs2, uint8_t const rs1,
                                                           uint16_t const vstart, uint16_t const vlen, uint16_t flen,
                                                           uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    auto const scalar = get_float_scalar(scalar_field, sew, flen, rs1);

    WIDE_SCALAR_ITERATOR_SWITCH(widening_vx)
}

template <typename OpType, SignType Sign>
inline constexpr GO_FAST void dispatch_iterate_widening_wf(void *const vector_field, void *const scalar_field,
                                                           uint16_t const vtype, uint8_t const mask_bit,
                                                           uint8_t const vd, uint8_t const vs2, uint8_t const rs1,
                                                           uint16_t const vstart, uint16_t const vlen, uint16_t flen,
                                                           uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    auto const scalar = get_float_scalar(scalar_field, sew, flen, rs1);

    WIDE_SCALAR_ITERATOR_SWITCH(widening_wx)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_vx(void *const vector_field, void *const scalar_field,
                                                           uint16_t const vtype, uint8_t const mask_bit,
                                                           uint8_t const vd, uint8_t const vs2, uint8_t rs1,
                                                           uint16_t const vstart, uint16_t const vlen, uint16_t xlen,
                                                           uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * (elements_per_register >> 1);
    auto const vs2_base = vs2 * elements_per_register;

    auto const scalar = get_scalar<Sign>(scalar_field, sew, xlen, rs1);

    WIDE_SCALAR_ITERATOR_SWITCH(widening_vx)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_widening_wx(void *const vector_field, void *const scalar_field,
                                                           uint16_t const vtype, uint8_t const mask_bit,
                                                           uint8_t const vd, uint8_t const vs2, uint8_t rs1,
                                                           uint16_t const vstart, uint16_t const vlen, uint16_t xlen,
                                                           uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * (elements_per_register >> 1);
    auto const vs2_base = vs2 * (elements_per_register >> 1);

    auto const scalar = get_scalar<Sign>(scalar_field, sew, xlen, rs1);

    WIDE_SCALAR_ITERATOR_SWITCH(widening_wx)
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_narrowing_wx(void *const vector_field, void *const scalar_field,
                                                            uint16_t const vtype, uint8_t const mask_bit,
                                                            uint8_t const vd, uint8_t const vs2, uint8_t rs1,
                                                            uint16_t const vstart, uint16_t const vlen, uint16_t xlen,
                                                            uint16_t const vl, OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * (elements_per_register >> 1);

    auto const scalar = get_scalar<Sign>(scalar_field, sew, xlen, rs1);

    WIDE_SCALAR_ITERATOR_SWITCH(narrowing_wxi)
}

template <unsigned Sew, unsigned Factor>
inline constexpr GO_FAST void dispatch_iterate_vext(void *const vector_field, uint16_t const vtype,
                                                    bool const is_masked, uint8_t const vd, uint8_t const vs2,
                                                    bool const is_signed, uint16_t const vstart, uint16_t const vlen,
                                                    uint16_t const vl)
{
    auto const src_elements_per_register = (Factor * vlen) / Sew;
    auto const dest_elements_per_register = vlen / Sew;
    auto const vd_base = vd * dest_elements_per_register;
    auto const vs2_base = vs2 * src_elements_per_register;

    using SourceTypeSigned = ExtTypes<Sew, SignType::Signed, Factor>::source_type;
    using DestTypeSigned = ExtTypes<Sew, SignType::Signed, Factor>::dest_type;
    using SourceTypeUnsigned = ExtTypes<Sew, SignType::Unsigned, Factor>::source_type;
    using DestTypeUnsigned = ExtTypes<Sew, SignType::Unsigned, Factor>::dest_type;

    if (is_signed)
    {
        if (is_masked)
        {
            vext_iterate<SourceTypeSigned, DestTypeSigned, MaskType::Masked>(vector_field, vstart, vl, vd_base,
                                                                             vs2_base);
        }
        else
        {
            vext_iterate<SourceTypeSigned, DestTypeSigned, MaskType::Unmasked>(vector_field, vstart, vl, vd_base,
                                                                               vs2_base);
        }
    }
    else
    {
        if (is_masked)
        {
            vext_iterate<SourceTypeUnsigned, DestTypeUnsigned, MaskType::Masked>(vector_field, vstart, vl, vd_base,
                                                                                 vs2_base);
        }
        else
        {
            vext_iterate<SourceTypeUnsigned, DestTypeUnsigned, MaskType::Unmasked>(vector_field, vstart, vl, vd_base,
                                                                                   vs2_base);
        }
    }
}

template <SignType Sign, typename OpType>
inline constexpr GO_FAST void dispatch_iterate_v_unary(void *const vector_field, uint16_t const vtype,
                                                       uint8_t const mask_bit, uint8_t const vd, uint8_t const vs2,
                                                       uint16_t const vstart, uint16_t const vlen, uint16_t const vl,
                                                       OpType const op)
{
    auto const sew = decode_sew(vtype);
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    using elm_8_t = ElementTypeMap<8, Sign>::element_type;
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;
    using elm_64_t = ElementTypeMap<64, Sign>::element_type;

    switch (sew)
    {
    case sew_8:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            v_unary_iterate<elm_8_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, op);
        }
        else
        {
            v_unary_iterate<elm_8_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, op);
        }
        break;
    case sew_16:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            v_unary_iterate<elm_16_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, op);
        }
        else
        {
            v_unary_iterate<elm_16_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, op);
        }
        break;
    case sew_32:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            v_unary_iterate<elm_32_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, op);
        }
        else
        {
            v_unary_iterate<elm_32_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, op);
        }
        break;
    case sew_64:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            v_unary_iterate<elm_64_t, MaskType::Masked>(vector_field, vstart, vl, vd_base, vs2_base, op);
        }
        else
        {
            v_unary_iterate<elm_64_t, MaskType::Unmasked>(vector_field, vstart, vl, vd_base, vs2_base, op);
        }
        break;
    default:
        break;
    }
}

std::int8_t vtype_decode(std::uint16_t const vtype, std::uint8_t *ta, std::uint8_t *ma, std::uint32_t *sew,
                         std::uint8_t *z_lmul, std::uint8_t *n_lmul)
{
    return (VTYPE::decode(vtype, ta, ma, sew, z_lmul, n_lmul));
}

std::uint16_t vtype_encode(std::uint16_t sew, std::uint8_t z_lmul, std::uint8_t n_lmul, std::uint8_t ta,
                           std::uint8_t ma)
{
    return VTYPE::encode(sew, z_lmul, n_lmul, ta, ma);
}

std::uint8_t vtype_extractSEW(std::uint16_t pVTYPE)
{
    return VTYPE::extractSEW(pVTYPE);
}

std::uint8_t vtype_extractLMUL(std::uint16_t pVTYPE)
{
    return VTYPE::extractLMUL(pVTYPE);
}

std::uint8_t vtype_extractTA(std::uint16_t pVTYPE)
{
    return VTYPE::extractTA(pVTYPE);
}

std::uint8_t vtype_extractMA(std::uint16_t pVTYPE)
{
    return VTYPE::extractMA(pVTYPE);
}

std::uint16_t vcfg_concatEEW(std::uint8_t mew, std::uint8_t width)
{
    return (VTYPE::concatEEW(mew, width));
}

std::uint8_t vload_encoded_unitstride(void *pV, std::uint8_t *pM, std::uint16_t pVTYPE, std::uint8_t pVm,
                                      std::uint16_t pEEW, std::uint8_t pVd, std::uint16_t pVSTART, std::uint16_t pVLEN,
                                      std::uint16_t pVL, std::uint64_t pMSTART)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint64_t _z_emul = pEEW * _vt._z_lmul;
    std::uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * 8) || (_z_emul > _n_emul * 8))
        return 1;

    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    std::function<void(std::size_t, std::uint8_t *, std::size_t)> f_readMem =
        [pM](std::size_t addr, std::uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            buff[i] = pM[addr + i];
    };

    VLSU::load_eew(f_readMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, pVL, pVLEN / 8, pVd, pMSTART, pVSTART, pVm,
                   pEEW / 8);

    return (0);
}

std::uint8_t vload_encoded_stride(void *pV, std::uint8_t *pM, std::uint16_t pVTYPE, std::uint8_t pVm,
                                  std::uint16_t pEEW, std::uint8_t pVd, std::uint16_t pVSTART, std::uint16_t pVLEN,
                                  std::uint16_t pVL, std::uint64_t pMSTART, std::int16_t pSTRIDE)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint64_t _z_emul = pEEW * _vt._z_lmul;
    std::uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * 8) || (_z_emul > _n_emul * 8))
    {
        return 1;
    }

    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    std::function<void(std::size_t, std::uint8_t *, std::size_t)> f_readMem =
        [pM](std::size_t addr, std::uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            buff[i] = pM[addr + i];
    };

    VLSU::load_eew(f_readMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, pVL, pVLEN / 8, pVd, pMSTART, pVSTART, pVm,
                   pSTRIDE);

    return (0);
}

std::uint8_t vload_segment_unitstride(void *pV, std::uint8_t *pM, std::uint16_t pVTYPE, std::uint8_t pVm,
                                      std::uint16_t pEEW, std::uint8_t pNF, std::uint8_t pVd, std::uint16_t pVSTART,
                                      std::uint16_t pVLEN, std::uint16_t pVL, std::uint64_t pMSTART)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint64_t _z_emul = pEEW * _vt._z_lmul;
    std::uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * pNF * 8) || (_z_emul * pNF > _n_emul * 8))
        return 1;
    if ((pVd + pNF * _z_emul / _n_emul) > 32)
        return 1;
    if (pVSTART >= pVL)
        return (0);

    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    std::function<void(std::size_t, std::uint8_t *, std::size_t)> f_readMem =
        [pM](std::size_t addr, std::uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            buff[i] = pM[addr + i];
    };

    std::uint16_t _vstart = pVSTART;
    std::uint64_t _moffset = pMSTART;

    for (int i = 0; i < pNF; ++i)
    {
        VLSU::load_eew(f_readMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, pVL, pVLEN / 8,
                       pVd + (i * _z_emul / _n_emul), _moffset, _vstart, pVm, pEEW / 8);
        _moffset += (pVL - _vstart) * pEEW / 8;
        _vstart = 0;
    }

    return (0);
}

std::uint8_t vload_segment_stride(void *pV, std::uint8_t *pM, std::uint16_t pVTYPE, std::uint8_t pVm,
                                  std::uint16_t pEEW, std::uint8_t pNF, std::uint8_t pVd, std::uint16_t pVSTART,
                                  std::uint16_t pVLEN, std::uint16_t pVL, std::uint64_t pMSTART, std::int16_t pSTRIDE)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint64_t _z_emul = pEEW * _vt._z_lmul;
    std::uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * pNF * 8) || (_z_emul * pNF > _n_emul * 8))
        return 1;
    if ((pVd + pNF * _z_emul / _n_emul) > 32)
        return 1;
    if (pVSTART >= pVL)
        return (0);

    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    std::function<void(std::size_t, std::uint8_t *, std::size_t)> f_readMem =
        [pM](std::size_t addr, std::uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            buff[i] = pM[addr + i];
    };

    std::uint16_t _vstart = pVSTART;
    std::uint64_t _moffset = pMSTART;

    for (int i = 0; i < pNF; ++i)
    {
        _moffset = pMSTART + i * pEEW / 8;
        VLSU::load_eew(f_readMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, pVL, pVLEN / 8,
                       pVd + (i * _z_emul / _n_emul), _moffset, _vstart, pVm, pSTRIDE);
        _vstart = 0;
    }

    return (0);
}

std::uint8_t vstore_encoded_unitstride(void *pV, std::uint8_t *pM, std::uint16_t pVTYPE, std::uint8_t pVm,
                                       std::uint16_t pEEW, std::uint8_t pVd, std::uint16_t pVSTART, std::uint16_t pVLEN,
                                       std::uint16_t pVL, std::uint64_t pMSTART)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint64_t _z_emul = pEEW * _vt._z_lmul;
    std::uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * 8) || (_z_emul > _n_emul * 8))
        return 1;

    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    std::function<void(std::size_t, std::uint8_t *, std::size_t)> f_writeMem =
        [pM](std::size_t addr, std::uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            pM[addr + i] = buff[i];
    };

    VLSU::store_eew(f_writeMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, pVL, pVLEN / 8, pVd, pMSTART, pVSTART, pVm,
                    pEEW / 8);

    return (0);
}

std::uint8_t vstore_encoded_stride(void *pV, std::uint8_t *pM, std::uint16_t pVTYPE, std::uint8_t pVm,
                                   std::uint16_t pEEW, std::uint8_t pVd, std::uint16_t pVSTART, std::uint16_t pVLEN,
                                   std::uint16_t pVL, std::uint64_t pMSTART, std::int16_t pStride)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint64_t _z_emul = pEEW * _vt._z_lmul;
    std::uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * 8) || (_z_emul > _n_emul * 8))
        return 1;

    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    std::function<void(std::size_t, std::uint8_t *, std::size_t)> f_writeMem =
        [pM](std::size_t addr, std::uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            pM[addr + i] = buff[i];
    };
    VLSU::store_eew(f_writeMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, pVL, pVLEN / 8, pVd, pMSTART, pVSTART, pVm,
                    pStride);

    return (0);
}

std::uint8_t vstore_segment_unitstride(void *pV, std::uint8_t *pM, std::uint16_t pVTYPE, std::uint8_t pVm,
                                       std::uint16_t pEEW, std::uint8_t pNF, std::uint8_t pVd, std::uint16_t pVSTART,
                                       std::uint16_t pVLEN, std::uint16_t pVL, std::uint64_t pMSTART)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint64_t _z_emul = pEEW * _vt._z_lmul;
    std::uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * pNF * 8) || (_z_emul * pNF > _n_emul * 8))
        return 1;
    if ((pVd + pNF * _z_emul / _n_emul) > 32)
        return 1;
    if (pVSTART >= pVL)
        return (0);

    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    std::function<void(std::size_t, std::uint8_t *, std::size_t)> f_writeMem =
        [pM](std::size_t addr, std::uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            pM[addr + i] = buff[i];
    };

    std::uint16_t _vstart = pVSTART;
    std::uint64_t _moffset = pMSTART;

    for (int i = 0; i < pNF; ++i)
    {
        VLSU::store_eew(f_writeMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, pVL, pVLEN / 8,
                        pVd + (i * _z_emul / _n_emul), _moffset, _vstart, pVm, pEEW / 8);
        _moffset += (pVL - _vstart) * pEEW / 8;
        _vstart = 0;
    }

    return (0);
}

std::uint8_t vstore_segment_stride(void *pV, std::uint8_t *pM, std::uint16_t pVTYPE, std::uint8_t pVm,
                                   std::uint16_t pEEW, std::uint8_t pNF, std::uint8_t pVd, std::uint16_t pVSTART,
                                   std::uint16_t pVLEN, std::uint16_t pVL, std::uint64_t pMSTART, std::int16_t pStride)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint64_t _z_emul = pEEW * _vt._z_lmul;
    std::uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * pNF * 8) || (_z_emul * pNF > _n_emul * 8))
        return 1;
    if ((pVd + pNF * _z_emul / _n_emul) > 32)
        return 1;
    if (pVSTART >= pVL)
        return (0);

    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    std::function<void(std::size_t, std::uint8_t *, std::size_t)> f_writeMem =
        [pM](std::size_t addr, std::uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            pM[addr + i] = buff[i];
    };

    std::uint16_t _vstart = pVSTART;
    std::uint64_t _moffset = pMSTART;
    for (int i = 0; i < pNF; ++i)
    {
        _moffset = pMSTART + i * pEEW / 8;
        VLSU::store_eew(f_writeMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, pVL, pVLEN / 8,
                        pVd + (i * _z_emul / _n_emul), _moffset, _vstart, pVm, pStride);
        _moffset += (pVL - _vstart) * pEEW / 8;
        _vstart = 0;
    }

    return (0);
}

/* 11. Vector Integer Arithmetic Instructions */

/* End 11.8. */

std::uint8_t vmv_xs(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pRd, std::uint8_t pVs2, std::uint16_t pVLEN,
                    std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRd * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRd * 8]);

    VPERM::mv_xs(VectorRegField, _vt._sew / 8, pVL, pVLEN / 8, pVs2, ScalarReg, pXLEN / 8);

    return (0);
}

std::uint8_t vmv_sx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pRs1,
                    std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VPERM::mv_sx(VectorRegField, _vt._sew / 8, pVL, pVLEN / 8, pVd, ScalarReg, pVSTART, pXLEN / 8);

    return (0);
}

std::uint8_t vslideup_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                         std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                         std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VPERM::slideup_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                      pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vslideup_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                         std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VPERM::slideup_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm, pVSTART,
                      pVm);

    return (0);
}

std::uint8_t vslidedown_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                           std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                           std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VPERM::slidedown_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                        pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vslidedown_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                           std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VPERM::slidedown_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vslide1up_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                          std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                          std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto perm_instr_info = VPERM::PermInstrInfo{};

    VPERM::perm_op_slide_vx(VectorRegField, v_instr_info, perm_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3);

    return (0);
}

std::uint8_t vfslide1up_vf(void *pV, void *pF, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                           std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                           std::uint16_t pVL, std::uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pF))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pF)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto perm_instr_info = VPERM::PermInstrInfo{ .float_instr = true };

    VPERM::perm_op_slide_vx(VectorRegField, v_instr_info, perm_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3);

    return (0);
}

std::uint8_t vslide1down_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                            std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                            std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto perm_instr_info = VPERM::PermInstrInfo{ .slide_down = true };

    VPERM::perm_op_slide_vx(VectorRegField, v_instr_info, perm_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3);

    return (0);
}

std::uint8_t vfslide1down_vf(void *pV, void *pF, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                             std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                             std::uint16_t pVL, std::uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pF))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pF)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto perm_instr_info = VPERM::PermInstrInfo{ .slide_down = true, .float_instr = true };

    VPERM::perm_op_slide_vx(VectorRegField, v_instr_info, perm_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3);

    return (0);
}

/* End 11.10. */
/* End 11.11. */

/* 11.12. Vector Widening Integer Multiply Instructions */
/* End 11.12. */

/* 11.4 Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
std::uint8_t vadc_vvm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs1, std::uint8_t pVs2,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vadc_vvm(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART);

    return (0);
}

std::uint8_t vadc_vxm(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs2, std::uint8_t pRs1,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VARITH_INT::vadc_vxm(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pXLEN / 8);

    return (0);
}

std::uint8_t vadc_vim(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs2, std::uint8_t pVimm,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vadc_vim(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                         pVSTART);

    return (0);
}

std::uint8_t vmadc_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vv_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2,
                                      VARITH_INT::deprecated::produce_carry_out);

    return (0);
}

std::uint8_t vmadc_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::produce_carry_out);

    return (0);
}

std::uint8_t vmadc_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vi_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, pVimm,
                                      VARITH_INT::deprecated::produce_carry_out);

    return (0);
}

std::uint8_t vsbc_vvm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs1, std::uint8_t pVs2,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = true,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::sub);

    return (0);
}

std::uint8_t vsbc_vxm(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs2, std::uint8_t pRs1,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = true,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                          VARITH_INT::deprecated::sub);

    return (0);
}

std::uint8_t vmsbc_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vv_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2,
                                      VARITH_INT::deprecated::produce_borrow_out);

    return (0);
}

std::uint8_t vmsbc_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::produce_borrow_out);

    return (0);
}
/* End 11.4 */

/* 11.13. Vector Single-Width Integer Multiply-Add Instructions */

/* 11.14. Vector Widening Integer Multiply-Add Instructions  */

/* End 11.14. */

/* 11.15. Vector Integer Merge Instructions */
std::uint8_t vmerge_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs1, std::uint8_t pVs2,
                       std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vmerge_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART);

    return (0);
}

std::uint8_t vmerge_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs2, std::uint8_t pVimm,
                       std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vmerge_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                          pVSTART);

    return (0);
}

std::uint8_t vmerge_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs2, std::uint8_t pRs1,
                       std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VARITH_INT::vmerge_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                          pVSTART, pXLEN / 8);

    return (0);
}
/* End 11.15. */

/* 11.16. Vector Integer Move Instructions */
std::uint8_t vmv_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs1, std::uint16_t pVSTART,
                    std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::mv_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVSTART);

    return (0);
}

std::uint8_t vmv_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVimm, std::uint16_t pVSTART,
                    std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::mv_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVimm, pVSTART);

    return (0);
}

std::uint8_t vmv_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pRs1,
                    std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VARITH_INT::mv_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, ScalarReg, pVSTART,
                      pXLEN / 8);

    return (0);
}
/* End 11.16. */
/* End 11. */

/* 12. Vector Fixed-Point Arithmetic Instructions */
/* 12.1. Vector Single-Width Saturating Add and Subtract */
std::uint8_t vsaddu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = 0, .narrowing_op = false };

    auto ret = VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs1, pVs2,
                                       VARITH_FIXP::saddu);

    // auto ret = VARITH_FIXP::vsadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd,
    //                                  pVs1, pVs2, pVSTART, pVm, false);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vsaddu_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false,
                             .zero_extend_immediate = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = 0, .narrowing_op = false };

    auto ret = VARITH_FIXP::fixp_op_vi(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, pVimm,
                                       VARITH_FIXP::saddu);

    // auto ret = VARITH_FIXP::vsadd_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd,
    //                                  pVs2, pVimm, pVSTART, pVm, false);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vsaddu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = 0, .narrowing_op = false };

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                                       VARITH_FIXP::saddu);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vsadd_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    auto ret = VARITH_FIXP::vsadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1,
                                     pVs2, pVSTART, pVm, true);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vsadd_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    auto ret = VARITH_FIXP::vsadd_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
                                     pVimm, pVSTART, pVm, true);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vsadd_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    auto ret = VARITH_FIXP::vsadd_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
                                     ScalarReg, pVSTART, pVm, true, pXLEN / 8);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vssubu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    auto ret = VARITH_FIXP::vssub_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1,
                                     pVs2, pVSTART, pVm, false);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vssubu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    auto ret = VARITH_FIXP::vssub_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
                                     ScalarReg, pVSTART, pVm, false, pXLEN / 8);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vssub_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    auto ret = VARITH_FIXP::vssub_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1,
                                     pVs2, pVSTART, pVm, true);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vssub_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info;

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                                       VARITH_FIXP::ssub);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}
/* End 12.1. */

/* 12.2. Vector Single-Width Averaging Add and Subtract */
/* TODO: Check for illegal rounding mode values */
std::uint8_t vaaddu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_FIXP::vaadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm, false, pRm);

    return (0);
}

std::uint8_t vaaddu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                            VARITH_FIXP::aaddu);

    return (0);
}

std::uint8_t vaadd_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_FIXP::vaadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm, true, pRm);

    return (0);
}

std::uint8_t vaadd_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                            VARITH_FIXP::aadd);
    return (0);
}

std::uint8_t vasubu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = false };

    auto ret = VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs1, pVs2,
                                       VARITH_FIXP::asubu);

    return (0);
}

std::uint8_t vasubu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                            VARITH_FIXP::asubu);
    return (0);
}

std::uint8_t vasub_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_FIXP::vasub_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm, true, pRm);

    return (0);
}

std::uint8_t vasub_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                            VARITH_FIXP::asub);
    return (0);
}
/* End 12.2. */

/* 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation */
std::uint8_t vsmul_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true,
                             .zero_extend_immediate = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = false };

    auto ret = VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs1, pVs2,
                                       VARITH_FIXP::smul);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vsmul_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = false };

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                                       VARITH_FIXP::smul);

    // auto ret = VARITH_FIXP::vsmul_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd,
    //                                  pVs2, ScalarReg, pVSTART, pVm, pXLEN / 8, pRm);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}
/* End 12.3. */

/* 12.4. Vector Single-Width Scaling Shift Instructions */
std::uint8_t vssrl_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_FIXP::vssrl_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm, pRm);

    return 0;
}

std::uint8_t vssrl_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false,
                             .zero_extend_immediate = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vi(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, pVimm, VARITH_FIXP::ssrl);

    // VARITH_FIXP::vssrl_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
    // pVimm,
    //                       pVSTART, pVm, pRm);

    return 0;
}

std::uint8_t vssrl_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VARITH_FIXP::vssrl_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                          pVSTART, pVm, pXLEN / 8, pRm);

    return 0;
}

std::uint8_t vssra_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_FIXP::vssra_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm, pRm);

    return 0;
}

std::uint8_t vssra_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_FIXP::vssra_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                          pVSTART, pVm, pRm);

    return 0;
}

std::uint8_t vssra_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VARITH_FIXP::vssra_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                          pVSTART, pVm, pXLEN / 8, pRm);

    return 0;
}
/* End 12.4. */

/* 12.5. Vector Narrowing Fixed-Point Clip Instructions */
std::uint8_t vnclipu_wv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                        std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = true };

    auto ret = VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs1, pVs2,
                                       VARITH_FIXP::clipu);

    // VARITH_FIXP::vnclipu_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1,
    // pVs2,
    //                         pVSTART, pVm, pRm);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vnclipu_wi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false,
                             .zero_extend_immediate = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = true };

    auto ret = VARITH_FIXP::fixp_op_vi(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, pVimm,
                                       VARITH_FIXP::clipu);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vnclipu_wx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = true };

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                                       VARITH_FIXP::clipu);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;

    // VARITH_FIXP::vnclipu_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
    //                         ScalarReg, pVSTART, pVm, pXLEN / 8, pRm);
}

std::uint8_t vnclip_wv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = true };

    auto ret = VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs1, pVs2,
                                       VARITH_FIXP::clip);

    // auto ret = VARITH_FIXP::vnclip_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8,
    // pVd,
    //                                   pVs1, pVs2, pVSTART, pVm, pRm);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;

    return 0;
}

std::uint8_t vnclip_wi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = false,
                             .zero_extend_immediate = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = true };

    auto ret = VARITH_FIXP::fixp_op_vi(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, pVimm,
                                       VARITH_FIXP::clip);

    // auto ret = VARITH_FIXP::vnclip_wi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8,
    // pVd,
    //                                   pVs2, pVimm, pVSTART, pVm, pRm);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

std::uint8_t vnclip_wx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pXLEN, std::uint8_t pRm)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = pRm, .narrowing_op = true };

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, pVd, pVs2, ScalarReg, pXLEN,
                                       VARITH_FIXP::clip);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

/* 13.15. Vector Floating-Point Merge Instruction */
std::uint8_t vfmerge_vfm(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs2,
                         std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                         std::uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART };

    VARITH_FLOAT::vf_merge(VectorRegField, v_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3);

    return 0;
}
/* End 13.15. */

/* 13.16. Vector Floating-Point Move Instruction */
std::uint8_t vfmv_v_f(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pRs1,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART };

    VARITH_FLOAT::vf_move(VectorRegField, v_instr_info, pVd, ScalarReg, pFLEN >> 3);

    return 0;
}
/* End 13.16. */

/* 15.2. Vector count population in mask vcpop.m */
std::uint8_t vcpop_m(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pRd, std::uint8_t pVs2,
                     std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRd * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRd * 8]);

    VMASK::mask_op_to_scalar(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVs2, ScalarReg,
                             pVSTART, pVm, pXLEN / 8, /* is_vcpop = */ true);

    return (0);
}
/* End 15.2. */
/* 15.3. vfirst find-first-set mask bit */
std::uint8_t vfirst_m(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pRd, std::uint8_t pVs2,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRd * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRd * 8]);

    VMASK::mask_op_to_scalar(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVs2, ScalarReg,
                             pVSTART, pVm, pXLEN / 8, /* is_vcpop = */ false);

    return (0);
}
/* End 15.3. */
/* 15.4. vmsbf.m set-before-first mask bit */
std::uint8_t vmsbf_m(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                     std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_sxf(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVSTART, pVm,
                       /* including_first = */ false, /* only_first = */ false);

    return 0;
}
/* End 15.4. */
/* 15.5. vmsif.m set-including-first mask bit */
std::uint8_t vmsif_m(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                     std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_sxf(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVSTART, pVm,
                       /* including_first = */ true, /* only_first = */ false);

    return 0;
}
/* End 15.5. */
/* 15.6. vmsof.m set-only-first mask bit */
std::uint8_t vmsof_m(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                     std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_sxf(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVSTART, pVm,
                       /* including_first = */ false, /* only_first = */ true);

    return 0;
}
/* End 15.6. */
/* 15.8. Vector Iota Instruction */
std::uint8_t viota_m(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                     std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_viota(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVSTART, pVm);

    return 0;
}
/* End 15.8. */
/* 15.9. Vector Element Index Instruction */
std::uint8_t vid_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint16_t pVSTART,
                   std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_vid(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVSTART, pVm);

    return 0;
}
/* End 15.9. */
/* End 15. */
/* 16. Vector Permutation Instructions */
/* 16.1. Integer Scalar Move Instructions */
/* End 16.1. */

/* 16.2. Floating-Point Scalar Move Instructions */
uint8_t vfmv_f_s(void *pV, void *pF, uint16_t pVTYPE, uint8_t pRd, uint8_t pVs2, uint16_t pVSTART, uint16_t pVLEN,
                 uint16_t pVL, uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;
    std::uint8_t *ScalarReg;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pFLEN <= 32)
    {
        ScalarReg = &((static_cast<std::uint8_t *>(pF))[pRd * 4]);
    }
    else
    {
        ScalarReg = &(static_cast<std::uint8_t *>(pF)[pRd * 8]);
    }

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART };

    VPERM::perm_op_move_float(VectorRegField, v_instr_info, pVs2, ScalarReg, pFLEN, false);

    return 0;
}

uint8_t vfmv_s_f(void *pV, void *pF, uint16_t pVTYPE, uint8_t pVd, uint8_t pRs1, uint16_t pVSTART, uint16_t pVLEN,
                 uint16_t pVL, uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);

    if (pVL == 0)
    {
        return 0;
    }

    std::uint8_t *VectorRegField;
    std::uint8_t *ScalarReg;
    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pFLEN <= 32)
    {
        ScalarReg = &((static_cast<std::uint8_t *>(pF))[pRs1 * 4]);
    }
    else
    {
        ScalarReg = &(static_cast<std::uint8_t *>(pF)[pRs1 * 8]);
    }

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = pVL,
                             .vector_register_length = pVLEN,
                             .start_element = pVSTART };

    VPERM::perm_op_move_float(VectorRegField, v_instr_info, pVd, ScalarReg, pFLEN, true);

    return 0;
}
/* End 16.2. */

/* 16.3. Vector Slide Instructions */
/* End 16.3. */
/* 16.4. Vector Register Gather Instructions */
std::uint8_t vrgather_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                         std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VPERM::vrgather_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, /* ei16 = */ false);

    return (0);
}

std::uint8_t vrgatherei16_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                             std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VPERM::vrgather_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, /* ei16 = */ true);

    return (0);
}

std::uint8_t vrgather_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                         std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VPERM::vrgather_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                       pVSTART, pVm);

    return (0);
}

std::uint8_t vrgather_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                         std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                         std::uint16_t pVL, std::uint8_t pXLEN)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *ScalarReg;
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    if (pXLEN <= 32)
        ScalarReg = &((static_cast<std::uint8_t *>(pR))[pRs1 * 4]);
    else
        ScalarReg = &(static_cast<std::uint8_t *>(pR)[pRs1 * 8]);

    VPERM::vrgather_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                       pVSTART, pVm, pXLEN / 8);

    return (0);
}
/* End 16.4. */
/* 16.5. Vector Compress Instruction */
std::uint8_t vcompress_vm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs1, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VPERM::vcompress_vm(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                        pVSTART);

    return (0);
}
/* End 16.5. */
/* 16.6. Whole Vector Register Move */
std::uint8_t vmvr_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVd, std::uint8_t pVs2, std::uint8_t simm5,
                    std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VPERM::vmvr_v(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, simm5, pVSTART);

    return 0;
}
/* End 16.6. */
/* End 16. */

//} // extern "C"
