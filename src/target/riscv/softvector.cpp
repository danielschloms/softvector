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

#include <cstdint>
#include <cstddef>
#include <cstdio>

#include "softvector.h"
#include "operations.hpp"

#include "base/base.hpp"
#include "lsu/lsu.hpp"
#include "arithmetic/integer.hpp"
#include "arithmetic/floatingpoint.hpp"
#include "arithmetic/fixedpoint.hpp"
#include "misc/mask.hpp"
#include "misc/permutation.hpp"
#include "misc/reduction.hpp"

// extern "C"
// {

/* --- Private enums --- */
enum class SignType
{
    Signed,
    Unsigned
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

/* --- Private globals --- */

constexpr auto masked_instruction_value = false;
constexpr auto masked_element_value = false;
constexpr auto sew_8_bytes = 1;
constexpr auto sew_16_bytes = 2;
constexpr auto sew_32_bytes = 4;
constexpr auto sew_64_bytes = 8;

/* --- Private function declarations --- */

template <typename VectorElementType, bool IsSigned>
auto dump_v_register(unsigned v_register, unsigned vlen, void *vector_field) -> void;

inline constexpr auto decode_sew(uint32_t vtype) -> unsigned;

inline constexpr auto is_masked_instruction(bool const instruction_mask_bit) -> bool;

inline constexpr auto is_masked_element(bool const element_mask_bit) -> bool;

template <SignType Sign, typename OpType>
void dispatch_iterate_vv(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs1, uint8_t vs2,
                         uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op);

// vd        vs2    vs1
// 2 * SEW = SEW op SEW
template <SignType Sign, typename OpType>
void dispatch_iterate_widening_vv(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs1,
                                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op);

// vd        vs2        vs1
// 2 * SEW = 2 * SEW op SEW
template <SignType Sign, typename OpType>
void dispatch_iterate_widening_wv(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs1,
                                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op);

template <SignType Sign, ImmExtensionType ImmExtension, typename OpType>
void dispatch_iterate_vi(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs2,
                         uint8_t immediate, uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op);

template <SignType Sign, typename OpType>
void dispatch_iterate_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd,
                         uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t xlen, uint16_t vl,
                         OpType op);

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline void iterate_vv(void *const vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                       unsigned const vs1_base, unsigned const vs2_base, OpType op);

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vxi(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs2_base, uint64_t scalar,
                 OpType op);

/* --- Public function definitions --- */
#define VV_OP(name, inner_op, sign)                                                                            \
    uint8_t name(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,  \
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)                                     \
    {                                                                                                          \
        dispatch_iterate_vv<sign>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen, vl, \
                                  inner_op);                                                                   \
        return 0;                                                                                              \
    }

#define VI_OP(name, inner_op, sign, imm_extension)                                                                  \
    uint8_t name(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,       \
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)                                          \
    {                                                                                                               \
        dispatch_iterate_vi<sign, imm_extension>(vector_field, vtype, masked_instruction_bit, vd, vs2, imm, vstart, \
                                                 vlen, vl, inner_op);                                               \
        return 0;                                                                                                   \
    }

#define VX_OP(name, inner_op, sign)                                                                                  \
    uint8_t name(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, \
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)                \
    {                                                                                                                \
        dispatch_iterate_vx<sign>(vector_field, scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1, vstart,   \
                                  vlen, xlen, vl, inner_op);                                                         \
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

// 11.3. Vector Integer Extension

// 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions

// 11.5. Vector Bitwise Logical Instructions

// 11.6. Vector Single-Width Shift Instructions
uint8_t vsll_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    dispatch_iterate_vv<SignType::Signed>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen, vl,
                                          sll_int);
    return 0;
}

// 11.7. Vector Narrowing Integer Right Shift Instructions

// 11.8. Vector Integer Compare Instructions

// 11.9. Vector Integer Min/Max Instructions

// 11.10. Vector Single-Width Integer Multiply Instructions

// 11.11. Vector Integer Divide Instructions

// 11.12. Vector Widening Integer Multiply Instructions

// 11.13. Vector Single-Width Integer Multiply-Add Instructions
uint8_t vmacc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    std::printf("vmacc.vv\n");
    dispatch_iterate_vv<SignType::Signed>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen, vl,
                                          static_cast<AccumulatorOp>(macc));
    return 0;
}

// uint8_t vmacc_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
//                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
// {
//     return 0;
// }

/* --- Private function definitions --- */

template <typename VectorElementType, bool IsSigned>
auto dump_v_register(unsigned v_register, unsigned vlen, void *vector_field) -> void
{
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);
    auto const sew = sizeof(VectorElementType) * 8;
    auto const elements_per_register = vlen / sew;
    auto const v_base = v_register * elements_per_register;

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
            std::printf("%li | ", vector_elements[v_base + i]);
        }
        std::printf("\n");
    }
}

/**
 * @brief Extract SEW in bit from VTYPE
 */
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

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
inline void iterate_vv(void *const vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                       unsigned const vs1_base, unsigned const vs2_base, OpType op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto *const vector_elements = static_cast<VectorElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (std::is_same_v<OpType, BinaryValueResultOp>)
        {
            // Casting signed to larger unsigned will sign extend.
            // As vector elements can be interpreted as int or uint, this should already take care of signed/unsigned
            // instructions
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, BinaryBitResultOp>)
        {
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], vector_elements[vs1_base + i])
                                                    << (i % sew);
        }
        else if constexpr (std::is_same_v<OpType, ShiftOp>)
        {
            vector_elements[vd_base + i] =
                op(vector_elements[vs2_base + i], vector_elements[vs1_base + i], static_cast<SewType>(sew));
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOp>)
        {
            // Accumulator instructions have vs2 and vs1 elements switched compared to e.g. vadd.vv
            // I.e. here vs2 is rhs and vs1 lhs
            vector_elements[vd_base + i] =
                op(vector_elements[vs1_base + i], vector_elements[vs2_base + i], vector_elements[vd_base + i]);
        }
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vv_masked(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                       unsigned const vs1_base, unsigned const vs2_base, OpType const op)
{
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    static constexpr auto sew = sizeof(VectorElementType) * 8;

    for (size_t i = vstart; i < vl; ++i)
    {
        auto mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        if constexpr (std::is_same_v<OpType, BinaryValueResultOp>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, BinaryBitResultOp>)
        {
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], vector_elements[vs1_base + i])
                                                    << (i % sew);
        }
        else if constexpr (std::is_same_v<OpType, ShiftOp>)
        {
            vector_elements[vd_base + i] =
                op(vector_elements[vs2_base + i], vector_elements[vs1_base + i], static_cast<SewType>(sew));
        }
        else if constexpr (std::is_same_v<OpType, AccumulatorOp>)
        {
            // Accumulator instructions have vs2 and vs1 elements switched compared to e.g. vadd.vv
            // I.e. here vs2 is rhs and vs1 lhs
            vector_elements[vd_base + i] =
                op(vector_elements[vs1_base + i], vector_elements[vs2_base + i], vector_elements[vd_base + i]);
        }
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_widening_vv(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                         unsigned vs2_base, OpType op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        wide_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i]);
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_widening_vv_masked(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                                unsigned vs2_base, OpType op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        auto mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        wide_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i]);
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_widening_wv(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                         unsigned vs2_base, OpType op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        wide_elements[vd_base + i] = op(wide_elements[vs2_base + i], vector_elements[vs1_base + i]);
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_widening_wv_masked(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                                unsigned vs2_base, OpType op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        auto mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        wide_elements[vd_base + i] = op(wide_elements[vs2_base + i], vector_elements[vs1_base + i]);
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, ShiftOp>
void iterate_narrowing_wv(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                          unsigned vs2_base, OpType op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        // Only ShiftOp for now
        vector_elements[vd_base + i] = op(wide_elements[vs2_base + i], vector_elements[vs1_base + i], sew * 2);
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_narrowing_wv_masked(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                                 unsigned vs2_base, OpType op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        auto mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        // Only ShiftOp for now
        vector_elements[vd_base + i] = op(wide_elements[vs2_base + i], vector_elements[vs1_base + i], sew * 2);
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vxi(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                 unsigned const vs2_base, uint64_t const scalar, OpType op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (std::is_same_v<OpType, BinaryValueResultOp>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar);
        }
        else if constexpr (std::is_same_v<OpType, BinaryBitResultOp>)
        {
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], scalar) << (i % sew);
        }
        else if constexpr (std::is_same_v<OpType, ShiftOp>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar, sew);
        }
    }
}

template <typename VectorElementType, typename OpType, typename ScalarType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType> and ValidScalarType<ScalarType>
void iterate_vxi_masked(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                        unsigned const vs2_base, ScalarType const scalar, OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        if constexpr (std::is_same_v<OpType, BinaryValueResultOp>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar);
        }
        else if constexpr (std::is_same_v<OpType, BinaryBitResultOp>)
        {
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], scalar) << (i % sew);
        }
        else
        {
            static_assert(false, "Operation not implemented");
        }
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_widening_vx(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                         unsigned const vs2_base, uint64_t const scalar, OpType op)
{
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        wide_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar);
    }
}

template <typename VectorElementType, typename OpType, typename ScalarType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_widening_vx_masked(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                                unsigned const vs2_base, uint64_t const scalar, OpType const op)
{
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        wide_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar);
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_widening_wx(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                         unsigned const vs2_base, uint64_t const scalar, OpType op)
{
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        wide_elements[vd_base + i] = op(wide_elements[vs2_base + i], scalar);
    }
}

template <typename VectorElementType, typename OpType, typename ScalarType>
    requires ValidVectorElementType<VectorElementType> and std::is_same_v<OpType, BinaryValueResultOp>
void iterate_widening_wx_masked(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                                unsigned const vs2_base, uint64_t const scalar, OpType const op)
{
    static constexpr auto wide_sew = sizeof(VectorElementType) * 8 * 2;
    using WideElementType = TypeWidener<VectorElementType>::wide_type;
    auto wide_elements = static_cast<WideElementType *>(vector_field);

    for (size_t i = vstart; i < vl; ++i)
    {
        auto const mask_bit = static_cast<bool>((wide_elements[i / wide_sew] >> (i % wide_sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        wide_elements[vd_base + i] = op(wide_elements[vs2_base + i], scalar);
    }
}

template <SignType Sign, typename OpType>
void dispatch_iterate_vv(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs1, uint8_t vs2,
                         uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op)
{
    auto const sew = decode_sew(vtype);
    auto const sew_bytes = sew >> 3;
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs1_base = vs1 * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    using elm_8_t = ElementTypeMap<8, Sign>::element_type;
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;
    using elm_64_t = ElementTypeMap<64, Sign>::element_type;

    switch (sew_bytes)
    {
    case sew_8_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vv_masked<elm_8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_vv<elm_8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    case sew_16_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vv_masked<elm_16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_vv<elm_16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    case sew_32_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vv_masked<elm_32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_vv<elm_32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    case sew_64_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vv_masked<elm_64_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_vv<elm_64_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    default:
        // Invalid SEW
        break;
    }
}

template <SignType Sign, typename OpType>
void dispatch_iterate_widening_vv(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs1,
                                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op)
{
    auto const sew = decode_sew(vtype);
    auto const sew_bytes = sew >> 3;
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register * 2;
    auto const vs1_base = vs1 * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    using elm_8_t = ElementTypeMap<8, Sign>::element_type;
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;

    switch (sew_bytes)
    {
    case sew_8_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_widening_vv_masked<elm_8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_widening_vv<elm_8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    case sew_16_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_widening_vv_masked<elm_16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_widening_vv<elm_16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    case sew_32_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_widening_vv_masked<elm_32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_widening_vv<elm_32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    default:
        // Invalid SEW
        break;
    }
}

template <SignType Sign, typename OpType>
void dispatch_iterate_widening_wv(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs1,
                                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op)
{
    auto const sew = decode_sew(vtype);
    auto const sew_bytes = sew >> 3;
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register * 2;
    auto const vs1_base = vs1 * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register * 2;

    using elm_8_t = ElementTypeMap<8, Sign>::element_type;
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;

    switch (sew_bytes)
    {
    case sew_8_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_widening_wv_masked<elm_8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_widening_wv<elm_8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    case sew_16_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_widening_wv_masked<elm_16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_widening_wv<elm_16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    case sew_32_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_widening_wv_masked<elm_32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        else
        {
            iterate_widening_wv<elm_32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
        }
        break;
    default:
        // Invalid SEW
        break;
    }
}

template <SignType Sign, ImmExtensionType ImmExtension, typename OpType>
void dispatch_iterate_vi(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs2,
                         uint8_t immediate, uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op)
{
    auto const sew = decode_sew(vtype);
    auto const sew_bytes = sew >> 3;
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    using elm_8_t = ElementTypeMap<8, Sign>::element_type;
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;
    using elm_64_t = ElementTypeMap<64, Sign>::element_type;

    uint64_t scalar = immediate;
    if constexpr (ImmExtension == ImmExtensionType::SignExtend)
    {
        scalar = sign_extend_immediate(scalar);
    }

    switch (sew_bytes)
    {
    case sew_8_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    case sew_16_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    case sew_32_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            std::printf("elm32imm\n");
            iterate_vxi<elm_32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    case sew_64_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    default:
        // Invalid SEW
        break;
    }
}

template <SignType Sign, typename OpType>
void dispatch_iterate_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd,
                         uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t xlen, uint16_t vl,
                         OpType op)
{
    auto const sew = decode_sew(vtype);
    auto const sew_bytes = sew >> 3;
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    using elm_8_t = ElementTypeMap<8, Sign>::element_type;
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;
    using elm_64_t = ElementTypeMap<64, Sign>::element_type;

    using scalar_32_t = ElementTypeMap<32, Sign>::element_type;
    using scalar_64_t = ElementTypeMap<64, Sign>::element_type;
    uint64_t scalar = 0;
    switch (xlen)
    {
    case 32:
        scalar = static_cast<uint64_t>((static_cast<scalar_32_t *>(scalar_field))[rs1]);
        break;
    case 64:
        scalar = static_cast<uint64_t>((static_cast<scalar_64_t *>(scalar_field))[rs1]);
        break;
    default:
        // Invalid XLEN!
        break;
    }

    switch (sew_bytes)
    {
    case sew_8_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    case sew_16_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    case sew_32_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    case sew_64_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    default:
        // Invalid SEW
        break;
    }
}

template <SignType Sign, typename OpType>
void dispatch_iterate_widening_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd,
                                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t xlen, uint16_t vl,
                                  OpType op)
{
    auto const sew = decode_sew(vtype);
    auto const sew_bytes = sew >> 3;
    auto const elements_per_register = vlen / sew;
    auto const vd_base = vd * elements_per_register;
    auto const vs2_base = vs2 * elements_per_register;

    using elm_8_t = ElementTypeMap<8, Sign>::element_type;
    using elm_16_t = ElementTypeMap<16, Sign>::element_type;
    using elm_32_t = ElementTypeMap<32, Sign>::element_type;

    using scalar_32_t = ElementTypeMap<32, Sign>::element_type;
    using scalar_64_t = ElementTypeMap<64, Sign>::element_type;

    uint64_t scalar = 0;
    switch (xlen)
    {
    case 32:
        scalar = static_cast<uint64_t>((static_cast<scalar_32_t *>(scalar_field))[rs1]);
        break;
    case 64:
        scalar = static_cast<uint64_t>((static_cast<scalar_64_t *>(scalar_field))[rs1]);
        break;
    default:
        // Invalid XLEN!
        break;
    }

    switch (sew_bytes)
    {
    case sew_8_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    case sew_16_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    case sew_32_bytes:
        if (is_masked_instruction(static_cast<bool>(mask_bit)))
        {
            iterate_vxi_masked<elm_32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        else
        {
            iterate_vxi<elm_32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
        }
        break;
    default:
        // Invalid SEW
        break;
    }
}

std::int8_t vtype_decode(std::uint16_t vtype, std::uint8_t *ta, std::uint8_t *ma, std::uint32_t *sew,
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

/* 11.2. Vector Widening Integer Add/Subtract */
std::uint8_t vwaddu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::wop_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, true, false);

    return (0);
}

std::uint8_t vwadd_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::wop_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, true, true);

    return (0);
}

std::uint8_t vwsubu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::wop_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, false, false);

    return (0);
}

std::uint8_t vwsub_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::wop_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, false, true);

    return (0);
}

std::uint8_t vwaddu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::wop_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                       pVSTART, pVm, true, false, pXLEN / 8);

    return (0);
}

std::uint8_t vwadd_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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
                             .signed_op = true,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::add);

    // VARITH_INT::wop_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
    // ScalarReg,
    //                    pVSTART, pVm, true, true, pXLEN / 8);

    return (0);
}

std::uint8_t vwsubu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::wop_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                       pVSTART, pVm, false, false, pXLEN / 8);

    return (0);
}

std::uint8_t vwsub_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::wop_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                       pVSTART, pVm, false, true, pXLEN / 8);

    return (0);
}

std::uint8_t vwaddu_w_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                         std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::wop_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, true, false);

    return (0);
}

std::uint8_t vwadd_w_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                        std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::wop_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, true, true);

    return (0);
}

std::uint8_t vwsubu_w_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                         std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::wop_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, false, false);

    return (0);
}

std::uint8_t vwsub_w_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                        std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::wop_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2, pVSTART,
                       pVm, false, true);

    return (0);
}

std::uint8_t vwaddu_w_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
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
                             .masked = !pVm,
                             .signed_op = false,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::add);

    // VARITH_INT::wop_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
    // ScalarReg,
    //                    pVSTART, pVm, true, false, pXLEN / 8);

    return (0);
}

std::uint8_t vwadd_w_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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
                             .signed_op = true,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::add);

    // VARITH_INT::wop_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
    // ScalarReg,
    //                    pVSTART, pVm, true, true, pXLEN / 8);

    return (0);
}

std::uint8_t vwsubu_w_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
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

    VARITH_INT::wop_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                       pVSTART, pVm, false, false, pXLEN / 8);

    return (0);
}

std::uint8_t vwsub_w_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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
                             .signed_op = true,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::sub);

    // VARITH_INT::wop_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
    // ScalarReg,
    //                    pVSTART, pVm, false, true, pXLEN / 8);

    return (0);
}
/* End 11.2. */

/* 11.3. Vector Integer Extension */
std::uint8_t vext_vf(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                     std::uint8_t extension_encoding, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vext_vf(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
                        extension_encoding, pVSTART, pVm);

    return (0);
}
/* End 11.3. */

/* 11.5. Vector Bitwise Logical Instructions */
std::uint8_t vand_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2,
                          VARITH_INT::deprecated::logical_and);

    return (0);
}

std::uint8_t vand_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, pVimm,
                          VARITH_INT::deprecated::logical_and);

    return (0);
}

std::uint8_t vand_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::logical_and);

    // VARITH_INT::int_instr_info_t int_instr_info; VARITH_INT::int_op_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul,
    // _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2,
    //                       ScalarReg, pVSTART, pVm, pXLEN / 8, VARITH_INT::logical_and, /* signed_vs2 = */ true,
    //                       /* signed_scalar = */ true);

    return (0);
}

std::uint8_t vor_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2,
                          VARITH_INT::deprecated::logical_or);

    return (0);
}

std::uint8_t vor_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, pVimm,
                          VARITH_INT::deprecated::logical_or);

    return (0);
}

std::uint8_t vor_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::logical_or);

    return (0);
}

std::uint8_t vxor_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2,
                          VARITH_INT::deprecated::logical_xor);

    return (0);
}

std::uint8_t vxor_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, pVimm,
                          VARITH_INT::deprecated::logical_xor);

    return (0);
}

std::uint8_t vxor_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::logical_xor);

    return (0);
}
/* End 11.5. */

/* 11.6. Vector Single-Width Shift Instructions */
// std::uint8_t vsll_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
//                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
// {
//     VTYPE::VTYPE _vt(pVTYPE);
//     std::uint8_t *VectorRegField;
//
//     VectorRegField = static_cast<std::uint8_t *>(pV);
//
//     VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
//                              .lmul_denom = _vt._n_lmul,
//                              .sew = _vt._sew,
//                              .vector_length = pVL,
//                              .vector_register_length = pVLEN,
//                              .start_element = pVSTART,
//                              .masked = !pVm,
//                              .signed_op = true,
//                              .zero_extend_immediate = true };
//
//     auto int_instr_info = VARITH_INT::IntInstrInfo{};
//
//     VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2,
//     VARITH_INT::deprecated::sll);
//
//     return (0);
// }

std::uint8_t vsll_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                     std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);
    VARITH_INT::sll_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                       pVSTART, pVm);

    return (0);
}

std::uint8_t vsll_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::sll_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                       pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vsrl_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .signed_op = false,
                             .zero_extend_immediate = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::srl);

    return (0);
}

std::uint8_t vsrl_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                     std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::srl_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                       pVSTART, pVm);

    return (0);
}

std::uint8_t vsrl_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::srl_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                       pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vsra_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .signed_op = true,
                             .zero_extend_immediate = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::sra);

    return (0);
}

std::uint8_t vsra_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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
                             .signed_op = true,
                             .zero_extend_immediate = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, pVimm, VARITH_INT::deprecated::sra);

    return (0);
}

std::uint8_t vsra_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::sra_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                       pVSTART, pVm, pXLEN / 8);

    return (0);
}
/* End 11.6. */

/* 11.7. Vector Narrowing Integer Right Shift Instructions */
std::uint8_t vnsrl_wv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vnsrl_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vnsrl_wi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vnsrl_wi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vnsrl_wx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vnsrl_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vnsra_wv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vnsra_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vnsra_wi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vnsra_wi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vnsra_wx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vnsra_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8);

    return (0);
}
/* End 11.7. */

/* 11.8. Vector Integer Compare Instructions */
std::uint8_t vmseq_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::mseq_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vmseq_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::mseq_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vmseq_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::eq);

    return (0);
}

std::uint8_t vmsne_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msne_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vmsne_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msne_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vmsne_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::ne);

    return (0);
}

std::uint8_t vmsltu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msltu_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vmsltu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::ltu);

    return (0);
}

std::uint8_t vmslt_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::mslt_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vmslt_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::lt);

    return (0);
}

std::uint8_t vmsleu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msleu_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vmsleu_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, pVimm,
                                      VARITH_INT::deprecated::leu);

    return (0);
}

std::uint8_t vmsleu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::leu);

    return (0);
}

std::uint8_t vmsle_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msle_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vmsle_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msle_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vmsle_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::le);

    return (0);
}

std::uint8_t vmsgtu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msgtu_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vmsgtu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::gtu);

    return (0);
}

std::uint8_t vmsgtu_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msgtu_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vmsgt_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msgt_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                        pVSTART, pVm);

    return (0);
}

std::uint8_t vmsgt_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                                      VARITH_INT::deprecated::gt);

    return (0);
}

std::uint8_t vmsgt_vi(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pVimm, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::msgt_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, pVimm,
                        pVSTART, pVm);

    return (0);
}
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

/* 11.10. Vector Single-Width Integer Multiply Instructions */
std::uint8_t vmul_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::mul);

    return (0);
}

std::uint8_t vmul_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vmul_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                        pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vmulh_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::mulh);

    return (0);
}

std::uint8_t vmulh_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vmulh_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vmulhu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::mulhu);

    return (0);
}

std::uint8_t vmulhu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vmulhu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                          pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vmulhsu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                        std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    // TODO: new version not passing test
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    // VARITH_INT::int_op_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1,
    // pVs2,
    //                       pVSTART, pVm, VARITH_INT::mulh, /* signed_vs2 = */ true, /* signed_vs1 = */ false);
    VARITH_INT::vmulhsu_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm);

    return (0);
}

std::uint8_t vmulhsu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vmulhsu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                           pVSTART, pVm, pXLEN / 8);

    return (0);
}
/* End 11.10. */

/* 11.11. Vector Integer Divide Instructions */
std::uint8_t vdiv_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto v_instr_info = VInstrInfo{ .lmul_num = _vt._z_lmul,
                                    .lmul_denom = _vt._n_lmul,
                                    .sew = _vt._sew,
                                    .vector_length = pVL,
                                    .vector_register_length = pVLEN,
                                    .start_element = pVSTART,
                                    .masked = !pVm,
                                    .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN >> 3,
                          VARITH_INT::deprecated::div);

    return (0);
}

std::uint8_t vdiv_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::div);

    return (0);
}

std::uint8_t vdivu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vdivu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vdivu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::divu);

    return (0);
}

std::uint8_t vrem_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::rem);

    return (0);
}

std::uint8_t vrem_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::rem);

    return (0);
}

std::uint8_t vremu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vremu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vremu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::remu);

    return (0);
}
/* End 11.11. */

/* 11.12. */
std::uint8_t vwmul_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .signed_op = true,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::mul);

    return (0);
}

std::uint8_t vwmul_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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
                             .signed_op = true,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::mul);

    return (0);
}

std::uint8_t vwmulu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vwmul_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART, pVm, VARITH_INT::VWMUL_TYPE::U_U);

    return (0);
}

std::uint8_t vwmulu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vwmul_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8, VARITH_INT::VWMUL_TYPE::U_U);

    return (0);
}

std::uint8_t vwmulsu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                        std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vwmul_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART, pVm, VARITH_INT::VWMUL_TYPE::S_U);

    return (0);
}

std::uint8_t vwmulsu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vwmul_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8, VARITH_INT::VWMUL_TYPE::S_U);

    return (0);
}
/* End 11.12. */

std::uint8_t vmax_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::max);

    return (0);
}

std::uint8_t vmax_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::max);

    return (0);
}

std::uint8_t vmaxu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::maxu);

    return (0);
}

std::uint8_t vmaxu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vmaxu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vmin_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::min);

    return (0);
}

std::uint8_t vmin_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::min);

    return (0);
}

std::uint8_t vminu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, pVd, pVs1, pVs2, VARITH_INT::deprecated::minu);

    return (0);
}

std::uint8_t vminu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vminu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8);

    return (0);
}

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
// std::uint8_t vmacc_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
//                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
// {
//     VTYPE::VTYPE _vt(pVTYPE);
//     std::uint8_t *VectorRegField;
//
//     VectorRegField = static_cast<std::uint8_t *>(pV);
//
//     VARITH_INT::vmacc_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
//                          pVSTART, pVm);
//
//     return (0);
// }
//
std::uint8_t vmacc_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vmacc_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                         pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vnmsac_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vnmsac_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm);

    return (0);
}

std::uint8_t vnmsac_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vnmsac_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                          pVSTART, pVm, pXLEN / 8);

    return (0);
}

std::uint8_t vmadd_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vmadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                         pVSTART, pVm);

    return (0);
}

std::uint8_t vmadd_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::madd);

    return (0);
}

std::uint8_t vnmsub_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vnmsub_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm);

    return (0);
}

std::uint8_t vnmsub_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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

    VARITH_INT::vnmsub_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                          pVSTART, pVm, pXLEN / 8);

    return (0);
}
/* End 11.13. */

/* 11.14. Vector Widening Integer Multiply-Add Instructions  */
std::uint8_t vwmaccu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                        std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vwmacc_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm, VARITH_INT::VWMACC_TYPE::U_U);

    return (0);
}

std::uint8_t vwmaccu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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
                             .signed_op = false,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::maccu);

    return (0);
}

std::uint8_t vwmacc_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vwmacc_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm, VARITH_INT::VWMACC_TYPE::S_S);

    return (0);
}

std::uint8_t vwmacc_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
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
                             .signed_op = true,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::macc);

    return (0);
}

std::uint8_t vwmaccsu_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                         std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VARITH_INT::vwmacc_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                          pVSTART, pVm, VARITH_INT::VWMACC_TYPE::S_U);

    return (0);
}

std::uint8_t vwmaccsu_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
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
                             .masked = !pVm,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mixed_signed = true };

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, pVd, pVs2, ScalarReg, pXLEN / 8,
                          VARITH_INT::deprecated::macc);

    return (0);
}

std::uint8_t vwmaccus_vx(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
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

    VARITH_INT::vwmacc_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs2, ScalarReg,
                          pVSTART, pVm, pXLEN / 8, VARITH_INT::VWMACC_TYPE::U_S);

    return (0);
}
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
/* End 12.5. */
/* End 12. */

/* 13. Vector Floating-Point Instructions */
/* 13.1. Vector Floating-Point Exception Flags */
/* End 13.1. */
/* 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions */

std::uint8_t vfadd_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::add);
    return 0;
}

std::uint8_t vfadd_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::add);

    return 0;
}

std::uint8_t vfsub_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::sub);

    return 0;
}

std::uint8_t vfsub_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sub);

    return 0;
}

std::uint8_t vfrsub_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::rsub);

    return 0;
}
/* End 13.2. */
/* 13.3. Vector Widening Floating-Point Add/Subtract Instructions */
std::uint8_t vfwadd_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::add);

    return 0;
}

std::uint8_t vfwadd_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::add);

    return 0;
}

std::uint8_t vfwsub_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::sub);

    return 0;
}

std::uint8_t vfwsub_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sub);

    return 0;
}

std::uint8_t vfwadd_wv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::add);

    return 0;
}

std::uint8_t vfwadd_wf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::add);

    return 0;
}

std::uint8_t vfwsub_wv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::sub);

    return 0;
}

std::uint8_t vfwsub_wf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sub);

    return 0;
}
/* End 13.3. */
/* 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions */
std::uint8_t vfmul_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::mul);

    return 0;
}

std::uint8_t vfmul_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::mul);

    return 0;
}

std::uint8_t vfdiv_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::div);

    return 0;
}

std::uint8_t vfdiv_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::div);

    return 0;
}

std::uint8_t vfrdiv_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::rdiv);

    return 0;
}
/* End 13.4. */
/* 13.5. Vector Widening Floating-Point Multiply */
std::uint8_t vfwmul_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::mul);

    return 0;
}

std::uint8_t vfwmul_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::mul);

    return 0;
}
/* End 13.5. */
/* 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions */
std::uint8_t vfmacc_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::macc);

    return 0;
}

std::uint8_t vfmacc_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::macc);

    return 0;
}

std::uint8_t vfnmacc_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::nmacc);

    return 0;
}

std::uint8_t vfnmacc_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmacc);

    return 0;
}

std::uint8_t vfmsac_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::msac);

    return 0;
}

std::uint8_t vfmsac_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::msac);

    return 0;
}

std::uint8_t vfnmsac_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::nmsac);

    return 0;
}

std::uint8_t vfnmsac_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmsac);

    return 0;
}

std::uint8_t vfmadd_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::madd);

    return 0;
}

std::uint8_t vfmadd_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::madd);

    return 0;
}

std::uint8_t vfnmadd_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::nmadd);

    return 0;
}

std::uint8_t vfnmadd_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmadd);

    return 0;
}

std::uint8_t vfmsub_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::msub);

    return 0;
}

std::uint8_t vfmsub_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::msub);

    return 0;
}

std::uint8_t vfnmsub_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::nmsub);

    return 0;
}

std::uint8_t vfnmsub_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmsub);

    return 0;
}
/* End 13.6. */
/* 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions */
std::uint8_t vfwmacc_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::macc);

    return 0;
}

std::uint8_t vfwmacc_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::macc);

    return 0;
}

std::uint8_t vfwnmacc_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::nmacc);

    return 0;
}

std::uint8_t vfwnmacc_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                         std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                         std::uint16_t pVL, std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmacc);

    return 0;
}

std::uint8_t vfwmsac_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::msac);

    return 0;
}

std::uint8_t vfwmsac_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::msac);

    return 0;
}

std::uint8_t vfwnmsac_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::nmsac);

    return 0;
}

std::uint8_t vfwnmsac_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd,
                         std::uint8_t pVs2, std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN,
                         std::uint16_t pVL, std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmsac);

    return 0;
}
/* End 13.7. */

/* 13.8. Vector Floating-Point Square-Root Instruction */
std::uint8_t vfsqrt_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_unary(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::sqrt);

    return 0;
}
/* End 13.8. */

/* 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction */
std::uint8_t vfrsqrt7_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_unary(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::rsqrt7);

    return 0;
}
/* End 13.9. */

/* 13.10. Vector Floating-Point Reciprocal Estimate Instruction */
std::uint8_t vfrec7_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_unary(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::rec7);

    return 0;
}
/* End 13.10. */

/* 13.11. Vector Floating-Point MIN/MAX Instructions */
std::uint8_t vfmin_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::min);

    return 0;
}

std::uint8_t vfmin_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::min);

    return 0;
}

std::uint8_t vfmax_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::max);

    return 0;
}

std::uint8_t vfmax_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::max);

    return 0;
}
/* End 13.11. */

/* 13.12. Vector Floating-Point Sign-Injection Instructions */
std::uint8_t vfsgnj_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::sgnj);

    return 0;
}

std::uint8_t vfsgnj_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                       std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sgnj);

    return 0;
}

std::uint8_t vfsgnjn_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::sgnjn);

    return 0;
}

std::uint8_t vfsgnjn_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sgnjn);

    return 0;
}

std::uint8_t vfsgnjx_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::sgnjx);

    return 0;
}

std::uint8_t vfsgnjx_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                        std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                        std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sgnjx);

    return 0;
}
/* End 13.12. */

/* 13.13. Vector Floating-Point Compare Instructions */
std::uint8_t vmfeq_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::eq);

    return 0;
}

std::uint8_t vmfeq_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::eq);

    return 0;
}

std::uint8_t vmfne_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::ne);

    return 0;
}

std::uint8_t vmfne_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::ne);

    return 0;
}

std::uint8_t vmflt_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::lt);

    return 0;
}

std::uint8_t vmflt_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::lt);

    return 0;
}

std::uint8_t vmfle_vv(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vv_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::le);

    return 0;
}

std::uint8_t vmfle_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::le);

    return 0;
}

std::uint8_t vmfgt_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::gt);

    return 0;
}

std::uint8_t vmfge_vf(void *pV, void *pR, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                      std::uint8_t pRs1, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL,
                      std::uint8_t pFLEN, std::uint8_t pRm)
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
                             .start_element = pVSTART,
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::ge);

    return 0;
}
/* End 13.13. */

/* 13.14. Vector Floating-Point Classify Instruction */
std::uint8_t vfclass_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                       std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_op_unary(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::classify);

    return 0;
}
/* End 13.14. */

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

/* 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions */
std::uint8_t vfcvt_xu_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::convert_x_f);

    return 0;
}

std::uint8_t vfcvt_x_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                         std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::convert_x_f);

    return 0;
}

std::uint8_t vfcvt_rtz_xu_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                              std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::convert_x_f);

    return 0;
}

std::uint8_t vfcvt_rtz_x_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                             std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::convert_x_f);

    return 0;
}

std::uint8_t vfcvt_f_xu_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_vs2_is_int = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::convert_f_x);

    return 0;
}

std::uint8_t vfcvt_f_x_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                         std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_vs2_is_int = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2, VARITH_FLOAT::convert_f_x);

    return 0;
}
/* End 13.17. */

/* 13.18. Widening Floating-Point/Integer Type-Convert Instructions */
std::uint8_t vfwcvt_xu_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                           std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_widening_x_f);

    return 0;
}

std::uint8_t vfwcvt_x_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_widening_x_f);

    return 0;
}

std::uint8_t vfwcvt_rtz_xu_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                               std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_widening_x_f);

    return 0;
}

std::uint8_t vfwcvt_rtz_x_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                              std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_widening_x_f);

    return 0;
}

std::uint8_t vfwcvt_f_xu_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                           std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_vs2_is_int = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_widening_f_x);

    return 0;
}

std::uint8_t vfwcvt_f_x_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_vs2_is_int = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_widening_f_x);

    return 0;
}

std::uint8_t vfwcvt_f_f_v(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_widening_f_f);

    return 0;
}
/* End 13.18. */

/* 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions */
std::uint8_t vfncvt_xu_f_w(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                           std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_narrowing_x_f);

    return 0;
}

std::uint8_t vfncvt_x_f_w(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_narrowing_x_f);

    return 0;
}

std::uint8_t vfncvt_rtz_xu_f_w(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                               std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_narrowing_x_f);

    return 0;
}

std::uint8_t vfncvt_rtz_x_f_w(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                              std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_narrowing_x_f);

    return 0;
}

std::uint8_t vfncvt_f_xu_w(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                           std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_narrowing_f_x);

    return 0;
}

std::uint8_t vfncvt_f_x_w(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_narrowing_f_x);

    return 0;
}

std::uint8_t vfncvt_f_f_w(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                          std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_narrowing_f_f);

    return 0;
}

std::uint8_t vfncvt_rod_f_f_w(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs2,
                              std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL, std::uint8_t pRm)
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
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm, .ncvt_rod = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, pVd, pVs2,
                             VARITH_FLOAT::convert_narrowing_f_f);

    return 0;
}
/* End 13.19. */
/* End 13. */

/* 14. Vector Reduction Operations */
/* 14.1. Vector Single-Width Integer Reduction Instructions */
std::uint8_t vredsum_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::add);

    return 0;
}

std::uint8_t vredmaxu_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::maxu);

    return 0;
}

std::uint8_t vredmax_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::max);

    return 0;
}

std::uint8_t vredminu_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::minu);

    return 0;
}

std::uint8_t vredmin_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::min);

    return 0;
}

std::uint8_t vredand_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::logical_and);

    return 0;
}

std::uint8_t vredor_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::logical_or);

    return 0;
}

std::uint8_t vredxor_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::logical_xor);

    return 0;
}

/* End 14.1. */
/* 14.2. Vector Widening Integer Reduction Instructions */
std::uint8_t vwredsumu_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true,
                             .wide_vs1 = true };

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::add);

    return 0;
}

std::uint8_t vwredsum_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .signed_op = true,
                             .wide_vd = true,
                             .wide_vs1 = true };

    VREDUC::red_op_int(VectorRegField, v_instr_info, pVd, pVs1, pVs2, VARITH_INT::add);

    return 0;
}

/* End 14.2. */
/* 14.3. Vector Single-Width Floating-Point Reduction Instructions */
std::uint8_t vfredosum_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::add);

    return 0;
}

std::uint8_t vfredusum_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::add);

    return 0;
}

std::uint8_t vfredmax_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::max);

    return 0;
}

std::uint8_t vfredmin_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .masked = !pVm };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::min);

    return 0;
}

/* End 14.3. */
/* 14.4. Vector Widening Floating-Point Reduction Instructions */
std::uint8_t vfwredosum_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true,
                             .wide_vs1 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::wadd);

    return 0;
}

std::uint8_t vfwredusum_vs(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
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
                             .wide_vd = true,
                             .wide_vs1 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = pRm };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, pVd, pVs1, pVs2, VARITH_FLOAT::wadd);

    return 0;
}

/* 15. Vector Mask Instructions */
/* 15.1. Vector Mask-Register Logical Instructions */
std::uint8_t vmand_mm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm, VMASK::logical_and);

    return 0;
}

std::uint8_t vmnand_mm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm, VMASK::logical_nand);

    return 0;
}

std::uint8_t vmandn_mm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm, VMASK::logical_andn);

    return 0;
}

std::uint8_t vmxor_mm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm, VMASK::logical_xor);

    return 0;
}

std::uint8_t vmor_mm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                     std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm, VMASK::logical_or);

    return 0;
}

std::uint8_t vmnor_mm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm, VMASK::logical_nor);

    return 0;
}

std::uint8_t vmorn_mm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                      std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm, VMASK::logical_orn);

    return 0;
}

std::uint8_t vmxnor_mm(void *pV, std::uint16_t pVTYPE, std::uint8_t pVm, std::uint8_t pVd, std::uint8_t pVs1,
                       std::uint8_t pVs2, std::uint16_t pVSTART, std::uint16_t pVLEN, std::uint16_t pVL)
{
    VTYPE::VTYPE _vt(pVTYPE);
    std::uint8_t *VectorRegField;

    VectorRegField = static_cast<std::uint8_t *>(pV);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, pVL, pVLEN / 8, pVd, pVs1, pVs2,
                           pVSTART, pVm, VMASK::logical_xnor);

    return 0;
}
/* End 15.1. */
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
