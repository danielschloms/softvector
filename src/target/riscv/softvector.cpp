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

#include "softvector.h"
#include "operations.hpp"

#include "base/base.hpp"
#include "lsu/lsu.hpp"
#include "arithmetic/integer.hpp"
#include "arithmetic/floatingpoint.hpp"
#include "arithmetic/fixedpoint.hpp"
#include "arithmetic/softfloat-extension.hpp"
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

/* --- Private globals --- */

constexpr auto masked_instruction_value = false;
constexpr auto masked_element_value = false;
constexpr auto sew_8_bytes = 1;
constexpr auto sew_16_bytes = 2;
constexpr auto sew_32_bytes = 4;
constexpr auto sew_64_bytes = 8;

/* --- Private function declarations --- */

inline unsigned decode_sew(uint32_t const vtype);

template <SignType Sign, typename OpType>
void dispatch_iterate_vv(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs1, uint8_t vs2,
                         uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op);

template <SignType Sign, ImmExtensionType ImmExtension, typename OpType>
void dispatch_iterate_vi(void *vector_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd, uint8_t vs2,
                         uint8_t immediate, uint16_t vstart, uint16_t vlen, uint16_t vl, OpType op);

template <SignType Sign, typename OpType>
void dispatch_iterate_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t mask_bit, uint8_t vd,
                         uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t xlen, uint16_t vl,
                         OpType op);

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vv(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                unsigned vs2_base, OpType op);

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vxi(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs2_base, uint64_t scalar,
                 OpType op);

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vv_masked(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                       unsigned vs2_base, OpType op);

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vxi_masked(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs2_base,
                        uint64_t scalar, OpType op);

/* --- Public function definitions --- */

// 11. Vector Integer Arithmetic Instructions
uint8_t vadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    dispatch_iterate_vv<SignType::Signed>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen, vl,
                                          add_int);
    return 0;
}

uint8_t vadd_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    dispatch_iterate_vi<SignType::Signed, ImmExtensionType::SignExtend>(vector_field, vtype, masked_instruction_bit, vd,
                                                                        vs2, imm, vstart, vlen, vl, add_int);
    return 0;
}

uint8_t vadd_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    dispatch_iterate_vx<SignType::Signed>(vector_field, scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1,
                                          vstart, vlen, xlen, vl, add_int);
    return 0;
}

/* 11.1. Vector Single-Width Integer Add and Subtract */
uint8_t vsub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    dispatch_iterate_vv<SignType::Signed>(vector_field, vtype, masked_instruction_bit, vd, vs1, vs2, vstart, vlen, vl,
                                          sub_int);
    return 0;
}

uint8_t vsub_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    dispatch_iterate_vx<SignType::Signed>(vector_field, scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1,
                                          vstart, vlen, xlen, vl, sub_int);
    return 0;
}

uint8_t vrsub_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    dispatch_iterate_vx<SignType::Signed>(vector_field, scalar_field, vtype, masked_instruction_bit, vd, vs2, rs1,
                                          vstart, vlen, xlen, vl, rsub_int);
    return 0;
}

uint8_t vrsub_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    dispatch_iterate_vi<SignType::Signed, ImmExtensionType::SignExtend>(vector_field, vtype, masked_instruction_bit, vd,
                                                                        vs2, imm, vstart, vlen, vl, add_int);
    return 0;
}
/* End 11.1. */

/* 11.2. Vector Widening Integer Add/Subtract */
uint8_t vwaddu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::wop_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, true, false);

    return (0);
}

uint8_t vwadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::wop_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, true, true);

    return (0);
}

uint8_t vwsubu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::wop_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, false, false);

    return (0);
}

uint8_t vwsub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::wop_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, false, true);

    return (0);
}

uint8_t vwaddu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::wop_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                       masked_instruction_bit, true, false, xlen / 8);

    return (0);
}

uint8_t vwadd_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::add);

    // VARITH_INT::wop_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
    // ScalarReg,
    //                    vstart, masked_instruction_bit, true, true, xlen / 8);

    return (0);
}

uint8_t vwsubu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::wop_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                       masked_instruction_bit, false, false, xlen / 8);

    return (0);
}

uint8_t vwsub_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::wop_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                       masked_instruction_bit, false, true, xlen / 8);

    return (0);
}

uint8_t vwaddu_w_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::wop_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, true, false);

    return (0);
}

uint8_t vwadd_w_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::wop_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, true, true);

    return (0);
}

uint8_t vwsubu_w_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::wop_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, false, false);

    return (0);
}

uint8_t vwsub_w_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::wop_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, false, true);

    return (0);
}

uint8_t vwaddu_w_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                    uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::add);

    // VARITH_INT::wop_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
    // ScalarReg,
    //                    vstart, masked_instruction_bit, true, false, xlen / 8);

    return (0);
}

uint8_t vwadd_w_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::add);

    // VARITH_INT::wop_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
    // ScalarReg,
    //                    vstart, masked_instruction_bit, true, true, xlen / 8);

    return (0);
}

uint8_t vwsubu_w_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                    uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::wop_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                       masked_instruction_bit, false, false, xlen / 8);

    return (0);
}

uint8_t vwsub_w_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::sub);

    // VARITH_INT::wop_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
    // ScalarReg,
    //                    vstart, masked_instruction_bit, false, true, xlen / 8);

    return (0);
}
/* End 11.2. */

/* 11.3. Vector Integer Extension */
uint8_t vext_vf(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint8_t extension_encoding, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vext_vf(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
                        extension_encoding, vstart, masked_instruction_bit);

    return (0);
}
/* End 11.3. */

/* 11.5. Vector Bitwise Logical Instructions */
uint8_t vand_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2,
                          VARITH_INT::deprecated::logical_and);

    return (0);
}

uint8_t vand_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi(VectorRegField, v_instr_info, int_instr_info, vd, vs2, imm,
                          VARITH_INT::deprecated::logical_and);

    return (0);
}

uint8_t vand_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};
    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::logical_and);

    // VARITH_INT::int_instr_info_t int_instr_info; VARITH_INT::int_op_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul,
    // _vt._sew / 8, vl, vlen / 8, vd, vs2,
    //                       ScalarReg, vstart, masked_instruction_bit, xlen / 8, VARITH_INT::logical_and, /*
    //                       signed_vs2 = */ true,
    //                       /* signed_scalar = */ true);

    return (0);
}

uint8_t vor_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1, uint8_t vs2,
               uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2,
                          VARITH_INT::deprecated::logical_or);

    return (0);
}

uint8_t vor_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2, uint8_t imm,
               uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi(VectorRegField, v_instr_info, int_instr_info, vd, vs2, imm,
                          VARITH_INT::deprecated::logical_or);

    return (0);
}

uint8_t vor_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
               uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::logical_or);

    return (0);
}

uint8_t vxor_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2,
                          VARITH_INT::deprecated::logical_xor);

    return (0);
}

uint8_t vxor_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi(VectorRegField, v_instr_info, int_instr_info, vd, vs2, imm,
                          VARITH_INT::deprecated::logical_xor);

    return (0);
}

uint8_t vxor_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::logical_xor);

    return (0);
}
/* End 11.5. */

/* 11.6. Vector Single-Width Shift Instructions */
uint8_t vsll_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .zero_extend_immediate = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::sll);

    return (0);
}

uint8_t vsll_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    VARITH_INT::sll_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                       masked_instruction_bit);

    return (0);
}

uint8_t vsll_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::sll_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                       masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vsrl_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .zero_extend_immediate = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::srl);

    return (0);
}

uint8_t vsrl_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::srl_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                       masked_instruction_bit);

    return (0);
}

uint8_t vsrl_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::srl_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                       masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vsra_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .zero_extend_immediate = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::sra);

    return (0);
}

uint8_t vsra_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .zero_extend_immediate = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi(VectorRegField, v_instr_info, int_instr_info, vd, vs2, imm, VARITH_INT::deprecated::sra);

    return (0);
}

uint8_t vsra_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::sra_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                       masked_instruction_bit, xlen / 8);

    return (0);
}
/* End 11.6. */

/* 11.7. Vector Narrowing Integer Right Shift Instructions */
uint8_t vnsrl_wv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vnsrl_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vnsrl_wi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vnsrl_wi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vnsrl_wx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vnsrl_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vnsra_wv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vnsra_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vnsra_wi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vnsra_wi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vnsra_wx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vnsra_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8);

    return (0);
}
/* End 11.7. */

/* 11.8. Vector Integer Compare Instructions */
uint8_t vmseq_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::mseq_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vmseq_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::mseq_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vmseq_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::eq);

    return (0);
}

uint8_t vmsne_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msne_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vmsne_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msne_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vmsne_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::ne);

    return (0);
}

uint8_t vmsltu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msltu_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vmsltu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::ltu);

    return (0);
}

uint8_t vmslt_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::mslt_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vmslt_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::lt);

    return (0);
}

uint8_t vmsleu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msleu_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vmsleu_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                  uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vi_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, imm,
                                      VARITH_INT::deprecated::leu);

    return (0);
}

uint8_t vmsleu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::leu);

    return (0);
}

uint8_t vmsle_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msle_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vmsle_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msle_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vmsle_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::le);

    return (0);
}

uint8_t vmsgtu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msgtu_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vmsgtu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::gtu);

    return (0);
}

uint8_t vmsgtu_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                  uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msgtu_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vmsgt_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msgt_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vmsgt_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::gt);

    return (0);
}

uint8_t vmsgt_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::msgt_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                        masked_instruction_bit);

    return (0);
}
/* End 11.8. */

uint8_t vmv_xs(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t scalar_fieldd, uint8_t vs2,
               uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[scalar_fieldd * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[scalar_fieldd * 8]);

    VPERM::mv_xs(VectorRegField, _vt._sew / 8, vl, vlen / 8, vs2, ScalarReg, xlen / 8);

    return (0);
}

uint8_t vmv_sx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t vd, uint8_t rs1, uint16_t vstart,
               uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VPERM::mv_sx(VectorRegField, _vt._sew / 8, vl, vlen / 8, vd, ScalarReg, vstart, xlen / 8);

    return (0);
}

uint8_t vslideup_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                    uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VPERM::slideup_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                      masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vslideup_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                    uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VPERM::slideup_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                      masked_instruction_bit);

    return (0);
}

uint8_t vslidedown_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit,
                      uint8_t vd, uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VPERM::slidedown_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                        vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vslidedown_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                      uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VPERM::slidedown_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                        masked_instruction_bit);

    return (0);
}

uint8_t vslide1up_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                     uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto perm_instr_info = VPERM::PermInstrInfo{};

    VPERM::perm_op_slide_vx(VectorRegField, v_instr_info, perm_instr_info, vd, vs2, ScalarReg, xlen >> 3);

    return (0);
}

uint8_t vfslide1up_vf(void *vector_field, void *pF, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                      uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(pF))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(pF)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto perm_instr_info = VPERM::PermInstrInfo{ .float_instr = true };

    VPERM::perm_op_slide_vx(VectorRegField, v_instr_info, perm_instr_info, vd, vs2, ScalarReg, pFLEN >> 3);

    return (0);
}

uint8_t vslide1down_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit,
                       uint8_t vd, uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto perm_instr_info = VPERM::PermInstrInfo{ .slide_down = true };

    VPERM::perm_op_slide_vx(VectorRegField, v_instr_info, perm_instr_info, vd, vs2, ScalarReg, xlen >> 3);

    return (0);
}

uint8_t vfslide1down_vf(void *vector_field, void *pF, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                        uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(pF))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(pF)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto perm_instr_info = VPERM::PermInstrInfo{ .slide_down = true, .float_instr = true };

    VPERM::perm_op_slide_vx(VectorRegField, v_instr_info, perm_instr_info, vd, vs2, ScalarReg, pFLEN >> 3);

    return (0);
}

/* 11.10. Vector Single-Width Integer Multiply Instructions */
uint8_t vmul_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::mul);

    return (0);
}

uint8_t vmul_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vmul_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                        vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vmulh_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::mulh);

    return (0);
}

uint8_t vmulh_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vmulh_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vmulhu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::mulhu);

    return (0);
}

uint8_t vmulhu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vmulhu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                          vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vmulhsu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    // TODO: new version not passing test
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    // VARITH_INT::int_op_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1,
    // vs2,
    //                       vstart, masked_instruction_bit, VARITH_INT::mulh, /* signed_vs2 = */ true, /* signed_vs1 =
    //                       */ false);
    VARITH_INT::vmulhsu_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit);

    return (0);
}

uint8_t vmulhsu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vmulhsu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                           vstart, masked_instruction_bit, xlen / 8);

    return (0);
}
/* End 11.10. */

/* 11.11. Vector Integer Divide Instructions */
uint8_t vdiv_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    auto v_instr_info = VInstrInfo{ .lmul_num = _vt._z_lmul,
                                    .lmul_denom = _vt._n_lmul,
                                    .sew = _vt._sew,
                                    .vector_length = vl,
                                    .vector_register_length = vlen,
                                    .start_element = vstart,
                                    .masked = !masked_instruction_bit,
                                    .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                          VARITH_INT::deprecated::div);

    return (0);
}

uint8_t vdiv_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::div);

    return (0);
}

uint8_t vdivu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vdivu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vdivu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::divu);

    return (0);
}

uint8_t vrem_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::rem);

    return (0);
}

uint8_t vrem_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::rem);

    return (0);
}

uint8_t vremu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vremu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vremu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::remu);

    return (0);
}
/* End 11.11. */

/* 11.12. */
uint8_t vwmul_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::mul);

    return (0);
}

uint8_t vwmul_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::mul);

    return (0);
}

uint8_t vwmulu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vwmul_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit, VARITH_INT::VWMUL_TYPE::U_U);

    return (0);
}

uint8_t vwmulu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vwmul_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8, VARITH_INT::VWMUL_TYPE::U_U);

    return (0);
}

uint8_t vwmulsu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vwmul_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit, VARITH_INT::VWMUL_TYPE::S_U);

    return (0);
}

uint8_t vwmulsu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vwmul_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8, VARITH_INT::VWMUL_TYPE::S_U);

    return (0);
}
/* End 11.12. */

uint8_t vmax_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::max);

    return (0);
}

uint8_t vmax_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::max);

    return (0);
}

uint8_t vmaxu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::maxu);

    return (0);
}

uint8_t vmaxu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vmaxu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vmin_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::min);

    return (0);
}

uint8_t vmin_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::min);

    return (0);
}

uint8_t vminu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::minu);

    return (0);
}

uint8_t vminu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vminu_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

/* 11.4 Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
uint8_t vadc_vvm(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t vs1, uint8_t vs2, uint16_t vstart,
                 uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vadc_vvm(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart);

    return (0);
}

uint8_t vadc_vxm(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t vd, uint8_t vs2, uint8_t rs1,
                 uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vadc_vxm(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, xlen / 8);

    return (0);
}

uint8_t vadc_vim(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t vs2, uint8_t imm, uint16_t vstart,
                 uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vadc_vim(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart);

    return (0);
}

uint8_t vmadc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vv_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2,
                                      VARITH_INT::deprecated::produce_carry_out);

    return (0);
}

uint8_t vmadc_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::produce_carry_out);

    return (0);
}

uint8_t vmadc_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vi_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, imm,
                                      VARITH_INT::deprecated::produce_carry_out);

    return (0);
}

uint8_t vsbc_vvm(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t vs1, uint8_t vs2, uint16_t vstart,
                 uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = true,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vv(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2, VARITH_INT::deprecated::sub);

    return (0);
}

uint8_t vsbc_vxm(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t vd, uint8_t vs2, uint8_t rs1,
                 uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = true,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                          VARITH_INT::deprecated::sub);

    return (0);
}

uint8_t vmsbc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vv_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs1, vs2,
                                      VARITH_INT::deprecated::produce_borrow_out);

    return (0);
}

uint8_t vmsbc_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mask_is_data = true };

    VARITH_INT::int_op_vx_to_register(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen >> 3,
                                      VARITH_INT::deprecated::produce_borrow_out);

    return (0);
}
/* End 11.4 */

/* 11.13. Vector Single-Width Integer Multiply-Add Instructions */
uint8_t vmacc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vmacc_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vmacc_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vmacc_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                         vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vnmsac_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vnmsac_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit);

    return (0);
}

uint8_t vnmsac_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vnmsac_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                          vstart, masked_instruction_bit, xlen / 8);

    return (0);
}

uint8_t vmadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vmadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                         masked_instruction_bit);

    return (0);
}

uint8_t vmadd_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::madd);

    return (0);
}

uint8_t vnmsub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vnmsub_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit);

    return (0);
}

uint8_t vnmsub_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vnmsub_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                          vstart, masked_instruction_bit, xlen / 8);

    return (0);
}
/* End 11.13. */

/* 11.14. Vector Widening Integer Multiply-Add Instructions  */
uint8_t vwmaccu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vwmacc_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit, VARITH_INT::VWMACC_TYPE::U_U);

    return (0);
}

uint8_t vwmaccu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::maccu);

    return (0);
}

uint8_t vwmacc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vwmacc_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit, VARITH_INT::VWMACC_TYPE::S_S);

    return (0);
}

uint8_t vwmacc_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{};

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::macc);

    return (0);
}

uint8_t vwmaccsu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vwmacc_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit, VARITH_INT::VWMACC_TYPE::S_U);

    return (0);
}

uint8_t vwmaccsu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                    uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto int_instr_info = VARITH_INT::IntInstrInfo{ .mixed_signed = true };

    VARITH_INT::int_op_vx(VectorRegField, v_instr_info, int_instr_info, vd, vs2, ScalarReg, xlen / 8,
                          VARITH_INT::deprecated::macc);

    return (0);
}

uint8_t vwmaccus_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                    uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vwmacc_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                          vstart, masked_instruction_bit, xlen / 8, VARITH_INT::VWMACC_TYPE::U_S);

    return (0);
}
/* End 11.14. */

/* 11.15. Vector Integer Merge Instructions */
uint8_t vmerge_vv(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t vs1, uint8_t vs2, uint16_t vstart,
                  uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vmerge_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart);

    return (0);
}

uint8_t vmerge_vi(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t vs2, uint8_t imm, uint16_t vstart,
                  uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::vmerge_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart);

    return (0);
}

uint8_t vmerge_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t vd, uint8_t vs2, uint8_t rs1,
                  uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::vmerge_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                          vstart, xlen / 8);

    return (0);
}
/* End 11.15. */

/* 11.16. Vector Integer Move Instructions */
uint8_t vmv_vv(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t vs1, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::mv_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vstart);

    return (0);
}

uint8_t vmv_vi(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_INT::mv_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, imm, vstart);

    return (0);
}

uint8_t vmv_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t vd, uint8_t rs1, uint16_t vstart,
               uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_INT::mv_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, ScalarReg, vstart,
                      xlen / 8);

    return (0);
}
/* End 11.16. */
/* End 11. */

/* 12. Vector Fixed-Point Arithmetic Instructions */
/* 12.1. Vector Single-Width Saturating Add and Subtract */
uint8_t vsaddu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = 0, .narrowing_op = false };

    auto ret =
        VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs1, vs2, VARITH_FIXP::saddu);

    // auto ret = VARITH_FIXP::vsadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd,
    //                                  vs1, vs2, vstart, masked_instruction_bit, false);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vsaddu_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                  uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .zero_extend_immediate = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = 0, .narrowing_op = false };

    auto ret =
        VARITH_FIXP::fixp_op_vi(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, imm, VARITH_FIXP::saddu);

    // auto ret = VARITH_FIXP::vsadd_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd,
    //                                  vs2, imm, vstart, masked_instruction_bit, false);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vsaddu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = 0, .narrowing_op = false };

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                                       VARITH_FIXP::saddu);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vsadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    auto ret = VARITH_FIXP::vsadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2,
                                     vstart, masked_instruction_bit, true);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vsadd_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    auto ret = VARITH_FIXP::vsadd_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm,
                                     vstart, masked_instruction_bit, true);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vsadd_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    auto ret = VARITH_FIXP::vsadd_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
                                     ScalarReg, vstart, masked_instruction_bit, true, xlen / 8);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vssubu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    auto ret = VARITH_FIXP::vssub_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2,
                                     vstart, masked_instruction_bit, false);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vssubu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    auto ret = VARITH_FIXP::vssub_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
                                     ScalarReg, vstart, masked_instruction_bit, false, xlen / 8);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vssub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    auto ret = VARITH_FIXP::vssub_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2,
                                     vstart, masked_instruction_bit, true);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vssub_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info;

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                                       VARITH_FIXP::ssub);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}
/* End 12.1. */

/* 12.2. Vector Single-Width Averaging Add and Subtract */
/* TODO: Check for illegal rounding mode values */
uint8_t vaaddu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_FIXP::vaadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit, false, rounding_mode);

    return (0);
}

uint8_t vaaddu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                            VARITH_FIXP::aaddu);

    return (0);
}

uint8_t vaadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_FIXP::vaadd_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit, true, rounding_mode);

    return (0);
}

uint8_t vaadd_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                            VARITH_FIXP::aadd);
    return (0);
}

uint8_t vasubu_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = false };

    auto ret =
        VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs1, vs2, VARITH_FIXP::asubu);

    return (0);
}

uint8_t vasubu_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                            VARITH_FIXP::asubu);
    return (0);
}

uint8_t vasub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_FIXP::vasub_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit, true, rounding_mode);

    return (0);
}

uint8_t vasub_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                            VARITH_FIXP::asub);
    return (0);
}
/* End 12.2. */

/* 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation */
uint8_t vsmul_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .zero_extend_immediate = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = false };

    auto ret =
        VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs1, vs2, VARITH_FIXP::smul);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vsmul_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = false };

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                                       VARITH_FIXP::smul);

    // auto ret = VARITH_FIXP::vsmul_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd,
    //                                  vs2, ScalarReg, vstart, masked_instruction_bit, xlen / 8, rounding_mode);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}
/* End 12.3. */

/* 12.4. Vector Single-Width Scaling Shift Instructions */
uint8_t vssrl_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_FIXP::vssrl_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit, rounding_mode);

    return 0;
}

uint8_t vssrl_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .zero_extend_immediate = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = false };

    VARITH_FIXP::fixp_op_vi(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, imm, VARITH_FIXP::ssrl);

    // VARITH_FIXP::vssrl_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
    // imm,
    //                       vstart, masked_instruction_bit, rounding_mode);

    return 0;
}

uint8_t vssrl_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_FIXP::vssrl_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                          vstart, masked_instruction_bit, xlen / 8, rounding_mode);

    return 0;
}

uint8_t vssra_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_FIXP::vssra_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                          masked_instruction_bit, rounding_mode);

    return 0;
}

uint8_t vssra_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VARITH_FIXP::vssra_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                          masked_instruction_bit, rounding_mode);

    return 0;
}

uint8_t vssra_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VARITH_FIXP::vssra_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg,
                          vstart, masked_instruction_bit, xlen / 8, rounding_mode);

    return 0;
}
/* End 12.4. */

/* 12.5. Vector Narrowing Fixed-Point Clip Instructions */
uint8_t vnclipu_wv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = true };

    auto ret =
        VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs1, vs2, VARITH_FIXP::clipu);

    // VARITH_FIXP::vnclipu_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1,
    // vs2,
    //                         vstart, masked_instruction_bit, rounding_mode);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vnclipu_wi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                   uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .zero_extend_immediate = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = true };

    auto ret =
        VARITH_FIXP::fixp_op_vi(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, imm, VARITH_FIXP::clipu);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vnclipu_wx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = true };

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                                       VARITH_FIXP::clipu);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;

    // VARITH_FIXP::vnclipu_wx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2,
    //                         ScalarReg, vstart, masked_instruction_bit, xlen / 8, rounding_mode);
}

uint8_t vnclip_wv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = true };

    auto ret =
        VARITH_FIXP::fixp_op_vv(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs1, vs2, VARITH_FIXP::clip);

    // auto ret = VARITH_FIXP::vnclip_wv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8,
    // vd,
    //                                   vs1, vs2, vstart, masked_instruction_bit, rounding_mode);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;

    return 0;
}

uint8_t vnclip_wi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                  uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .zero_extend_immediate = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = true };

    auto ret =
        VARITH_FIXP::fixp_op_vi(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, imm, VARITH_FIXP::clip);

    // auto ret = VARITH_FIXP::vnclip_wi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8,
    // vd,
    //                                   vs2, imm, vstart, masked_instruction_bit, rounding_mode);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}

uint8_t vnclip_wx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VARITH_FIXP::FpInstrInfo fixedpoint_instr_info{ .rounding_mode = rounding_mode, .narrowing_op = true };

    auto ret = VARITH_FIXP::fixp_op_vx(VectorRegField, v_instr_info, fixedpoint_instr_info, vd, vs2, ScalarReg, xlen,
                                       VARITH_FIXP::clip);

    return ret == VILL::VPU_RETURN::NO_EXCEPT_FP_SAT ? 1 : 0;
}
/* End 12.5. */
/* End 12. */

/* 13. Vector Floating-Point Instructions */
/* 13.1. Vector Floating-Point Exception Flags */
/* End 13.1. */
/* 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions */

uint8_t vfadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::add);
    return 0;
}

uint8_t vfadd_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::add);

    return 0;
}

uint8_t vfsub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::sub);

    return 0;
}

uint8_t vfsub_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sub);

    return 0;
}

uint8_t vfrsub_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::rsub);

    return 0;
}
/* End 13.2. */
/* 13.3. Vector Widening Floating-Point Add/Subtract Instructions */
uint8_t vfwadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::add);

    return 0;
}

uint8_t vfwadd_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::add);

    return 0;
}

uint8_t vfwsub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::sub);

    return 0;
}

uint8_t vfwsub_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sub);

    return 0;
}

uint8_t vfwadd_wv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::add);

    return 0;
}

uint8_t vfwadd_wf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::add);

    return 0;
}

uint8_t vfwsub_wv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::sub);

    return 0;
}

uint8_t vfwsub_wf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sub);

    return 0;
}
/* End 13.3. */
/* 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions */
uint8_t vfmul_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::mul);

    return 0;
}

uint8_t vfmul_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::mul);

    return 0;
}

uint8_t vfdiv_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::div);

    return 0;
}

uint8_t vfdiv_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::div);

    return 0;
}

uint8_t vfrdiv_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::rdiv);

    return 0;
}
/* End 13.4. */
/* 13.5. Vector Widening Floating-Point Multiply */
uint8_t vfwmul_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::mul);

    return 0;
}

uint8_t vfwmul_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::mul);

    return 0;
}
/* End 13.5. */
/* 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions */
uint8_t vfmacc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::macc);

    return 0;
}

uint8_t vfmacc_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::macc);

    return 0;
}

uint8_t vfnmacc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::nmacc);

    return 0;
}

uint8_t vfnmacc_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmacc);

    return 0;
}

uint8_t vfmsac_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::msac);

    return 0;
}

uint8_t vfmsac_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::msac);

    return 0;
}

uint8_t vfnmsac_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::nmsac);

    return 0;
}

uint8_t vfnmsac_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmsac);

    return 0;
}

uint8_t vfmadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::madd);

    return 0;
}

uint8_t vfmadd_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::madd);

    return 0;
}

uint8_t vfnmadd_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::nmadd);

    return 0;
}

uint8_t vfnmadd_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmadd);

    return 0;
}

uint8_t vfmsub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::msub);

    return 0;
}

uint8_t vfmsub_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::msub);

    return 0;
}

uint8_t vfnmsub_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::nmsub);

    return 0;
}

uint8_t vfnmsub_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmsub);

    return 0;
}
/* End 13.6. */
/* 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions */
uint8_t vfwmacc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::macc);

    return 0;
}

uint8_t vfwmacc_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::macc);

    return 0;
}

uint8_t vfwnmacc_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::nmacc);

    return 0;
}

uint8_t vfwnmacc_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                    uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                    uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmacc);

    return 0;
}

uint8_t vfwmsac_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::msac);

    return 0;
}

uint8_t vfwmsac_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::msac);

    return 0;
}

uint8_t vfwnmsac_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::nmsac);

    return 0;
}

uint8_t vfwnmsac_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                    uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                    uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::nmsac);

    return 0;
}
/* End 13.7. */

/* 13.8. Vector Floating-Point Square-Root Instruction */
uint8_t vfsqrt_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_unary(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::sqrt);

    return 0;
}
/* End 13.8. */

/* 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction */
uint8_t vfrsqrt7_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                   uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_unary(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::rsqrt7);

    return 0;
}
/* End 13.9. */

/* 13.10. Vector Floating-Point Reciprocal Estimate Instruction */
uint8_t vfrec7_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                 uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_unary(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::rec7);

    return 0;
}
/* End 13.10. */

/* 13.11. Vector Floating-Point MIN/MAX Instructions */
uint8_t vfmin_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::min);

    return 0;
}

uint8_t vfmin_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::min);

    return 0;
}

uint8_t vfmax_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::max);

    return 0;
}

uint8_t vfmax_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::max);

    return 0;
}
/* End 13.11. */

/* 13.12. Vector Floating-Point Sign-Injection Instructions */
uint8_t vfsgnj_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::sgnj);

    return 0;
}

uint8_t vfsgnj_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                  uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                  uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sgnj);

    return 0;
}

uint8_t vfsgnjn_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::sgnjn);

    return 0;
}

uint8_t vfsgnjn_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sgnjn);

    return 0;
}

uint8_t vfsgnjx_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::sgnjx);

    return 0;
}

uint8_t vfsgnjx_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                   uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                   uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                           VARITH_FLOAT::sgnjx);

    return 0;
}
/* End 13.12. */

/* 13.13. Vector Floating-Point Compare Instructions */
uint8_t vmfeq_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::eq);

    return 0;
}

uint8_t vmfeq_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::eq);

    return 0;
}

uint8_t vmfne_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::ne);

    return 0;
}

uint8_t vmfne_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::ne);

    return 0;
}

uint8_t vmflt_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::lt);

    return 0;
}

uint8_t vmflt_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::lt);

    return 0;
}

uint8_t vmfle_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vv_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::le);

    return 0;
}

uint8_t vmfle_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::le);

    return 0;
}

uint8_t vmfgt_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::gt);

    return 0;
}

uint8_t vmfge_vf(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                 uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN,
                 uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_vf_to_reg(VectorRegField, v_instr_info, float_instr_info, vd, vs2, ScalarReg, pFLEN >> 3,
                                  VARITH_FLOAT::ge);

    return 0;
}
/* End 13.13. */

/* 13.14. Vector Floating-Point Classify Instruction */
uint8_t vfclass_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                  uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_op_unary(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::classify);

    return 0;
}
/* End 13.14. */

/* 13.15. Vector Floating-Point Merge Instruction */
uint8_t vfmerge_vfm(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t vd, uint8_t vs2, uint8_t rs1,
                    uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart };

    VARITH_FLOAT::vf_merge(VectorRegField, v_instr_info, vd, vs2, ScalarReg, pFLEN >> 3);

    return 0;
}
/* End 13.15. */

/* 13.16. Vector Floating-Point Move Instruction */
uint8_t vfmv_v_f(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t vd, uint8_t rs1, uint16_t vstart,
                 uint16_t vlen, uint16_t vl, uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart };

    VARITH_FLOAT::vf_move(VectorRegField, v_instr_info, vd, ScalarReg, pFLEN >> 3);

    return 0;
}
/* End 13.16. */

/* 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions */
uint8_t vfcvt_xu_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                     uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::convert_x_f);

    return 0;
}

uint8_t vfcvt_x_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                    uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::convert_x_f);

    return 0;
}

uint8_t vfcvt_rtz_xu_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                         uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::convert_x_f);

    return 0;
}

uint8_t vfcvt_rtz_x_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                        uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::convert_x_f);

    return 0;
}

uint8_t vfcvt_f_xu_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                     uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_vs2_is_int = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::convert_f_x);

    return 0;
}

uint8_t vfcvt_f_x_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                    uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_vs2_is_int = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2, VARITH_FLOAT::convert_f_x);

    return 0;
}
/* End 13.17. */

/* 13.18. Widening Floating-Point/Integer Type-Convert Instructions */
uint8_t vfwcvt_xu_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                      uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_widening_x_f);

    return 0;
}

uint8_t vfwcvt_x_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                     uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_widening_x_f);

    return 0;
}

uint8_t vfwcvt_rtz_xu_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                          uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_widening_x_f);

    return 0;
}

uint8_t vfwcvt_rtz_x_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                         uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_widening_x_f);

    return 0;
}

uint8_t vfwcvt_f_xu_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                      uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_vs2_is_int = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_widening_f_x);

    return 0;
}

uint8_t vfwcvt_f_x_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                     uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_vs2_is_int = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_widening_f_x);

    return 0;
}

uint8_t vfwcvt_f_f_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                     uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_widening_f_f);

    return 0;
}
/* End 13.18. */

/* 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions */
uint8_t vfncvt_xu_f_w(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                      uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_narrowing_x_f);

    return 0;
}

uint8_t vfncvt_x_f_w(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                     uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_narrowing_x_f);

    return 0;
}

uint8_t vfncvt_rtz_xu_f_w(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                          uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_narrowing_x_f);

    return 0;
}

uint8_t vfncvt_rtz_x_f_w(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                         uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .cvt_rtz = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_narrowing_x_f);

    return 0;
}

uint8_t vfncvt_f_xu_w(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                      uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = false,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_narrowing_f_x);

    return 0;
}

uint8_t vfncvt_f_x_w(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                     uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_narrowing_f_x);

    return 0;
}

uint8_t vfncvt_f_f_w(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                     uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_narrowing_f_f);

    return 0;
}

uint8_t vfncvt_rod_f_f_w(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                         uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vs2 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode, .ncvt_rod = true };

    VARITH_FLOAT::vf_convert(VectorRegField, v_instr_info, float_instr_info, vd, vs2,
                             VARITH_FLOAT::convert_narrowing_f_f);

    return 0;
}
/* End 13.19. */
/* End 13. */

/* 14. Vector Reduction Operations */
/* 14.1. Vector Single-Width Integer Reduction Instructions */
uint8_t vredsum_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::add);

    return 0;
}

uint8_t vredmaxu_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::maxu);

    return 0;
}

uint8_t vredmax_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::max);

    return 0;
}

uint8_t vredminu_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::minu);

    return 0;
}

uint8_t vredmin_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::min);

    return 0;
}

uint8_t vredand_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::logical_and);

    return 0;
}

uint8_t vredor_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::logical_or);

    return 0;
}

uint8_t vredxor_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                   uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::logical_xor);

    return 0;
}

/* End 14.1. */
/* 14.2. Vector Widening Integer Reduction Instructions */
uint8_t vwredsumu_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                     uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true,
                             .wide_vs1 = true };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::add);

    return 0;
}

uint8_t vwredsum_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .signed_op = true,
                             .wide_vd = true,
                             .wide_vs1 = true };

    VREDUC::red_op_int(VectorRegField, v_instr_info, vd, vs1, vs2, VARITH_INT::add);

    return 0;
}

/* End 14.2. */
/* 14.3. Vector Single-Width Floating-Point Reduction Instructions */
uint8_t vfredosum_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                     uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::add);

    return 0;
}

uint8_t vfredusum_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                     uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::add);

    return 0;
}

uint8_t vfredmax_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::max);

    return 0;
}

uint8_t vfredmin_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::min);

    return 0;
}

/* End 14.3. */
/* 14.4. Vector Widening Floating-Point Reduction Instructions */
uint8_t vfwredosum_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                      uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true,
                             .wide_vs1 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::wadd);

    return 0;
}

uint8_t vfwredusum_vs(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                      uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t rounding_mode)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart,
                             .masked = !masked_instruction_bit,
                             .wide_vd = true,
                             .wide_vs1 = true };

    auto float_instr_info = VARITH_FLOAT::FloatInstrInfo{ .rounding_mode = rounding_mode };

    VREDUC::red_op_float(VectorRegField, v_instr_info, float_instr_info, vd, vs1, vs2, VARITH_FLOAT::wadd);

    return 0;
}

/* 15. Vector Mask Instructions */
/* 15.1. Vector Mask-Register Logical Instructions */
uint8_t vmand_mm(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit, VMASK::logical_and);

    return 0;
}

uint8_t vmnand_mm(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit, VMASK::logical_nand);

    return 0;
}

uint8_t vmandn_mm(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit, VMASK::logical_andn);

    return 0;
}

uint8_t vmxor_mm(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit, VMASK::logical_xor);

    return 0;
}

uint8_t vmor_mm(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit, VMASK::logical_or);

    return 0;
}

uint8_t vmnor_mm(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit, VMASK::logical_nor);

    return 0;
}

uint8_t vmorn_mm(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                 uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit, VMASK::logical_orn);

    return 0;
}

uint8_t vmxnor_mm(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                  uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_logical(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                           masked_instruction_bit, VMASK::logical_xnor);

    return 0;
}
/* End 15.1. */
/* 15.2. Vector count population in mask vcpop.m */
uint8_t vcpop_m(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit,
                uint8_t scalar_fieldd, uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[scalar_fieldd * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[scalar_fieldd * 8]);

    VMASK::mask_op_to_scalar(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vs2, ScalarReg,
                             vstart, masked_instruction_bit, xlen / 8, /* is_vcpop = */ true);

    return (0);
}
/* End 15.2. */
/* 15.3. vfirst find-first-set mask bit */
uint8_t vfirst_m(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit,
                 uint8_t scalar_fieldd, uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[scalar_fieldd * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[scalar_fieldd * 8]);

    VMASK::mask_op_to_scalar(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vs2, ScalarReg,
                             vstart, masked_instruction_bit, xlen / 8, /* is_vcpop = */ false);

    return (0);
}
/* End 15.3. */
/* 15.4. vmsbf.m set-before-first mask bit */
uint8_t vmsbf_m(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_sxf(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, vstart,
                       masked_instruction_bit,
                       /* including_first = */ false, /* only_first = */ false);

    return 0;
}
/* End 15.4. */
/* 15.5. vmsif.m set-including-first mask bit */
uint8_t vmsif_m(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_sxf(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, vstart,
                       masked_instruction_bit,
                       /* including_first = */ true, /* only_first = */ false);

    return 0;
}
/* End 15.5. */
/* 15.6. vmsof.m set-only-first mask bit */
uint8_t vmsof_m(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_op_sxf(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, vstart,
                       masked_instruction_bit,
                       /* including_first = */ false, /* only_first = */ true);

    return 0;
}
/* End 15.6. */
/* 15.8. Vector Iota Instruction */
uint8_t viota_m(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_viota(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, vstart,
                      masked_instruction_bit);

    return 0;
}
/* End 15.8. */
/* 15.9. Vector Element Index Instruction */
uint8_t vid_v(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint16_t vstart,
              uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VMASK::mask_vid(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vstart,
                    masked_instruction_bit);

    return 0;
}
/* End 15.9. */
/* End 15. */
/* 16. Vector Permutation Instructions */
/* 16.1. Integer Scalar Move Instructions */
/* End 16.1. */

/* 16.2. Floating-Point Scalar Move Instructions */
uint8_t vfmv_f_s(void *vector_field, void *pF, uint16_t vtype, uint8_t scalar_fieldd, uint8_t vs2, uint16_t vstart,
                 uint16_t vlen, uint16_t vl, uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;
    uint8_t *ScalarReg;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
    {
        ScalarReg = &((static_cast<uint8_t *>(pF))[scalar_fieldd * 4]);
    }
    else
    {
        ScalarReg = &(static_cast<uint8_t *>(pF)[scalar_fieldd * 8]);
    }

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart };

    VPERM::perm_op_move_float(VectorRegField, v_instr_info, vs2, ScalarReg, pFLEN, false);

    return 0;
}

uint8_t vfmv_s_f(void *vector_field, void *pF, uint16_t vtype, uint8_t vd, uint8_t rs1, uint16_t vstart, uint16_t vlen,
                 uint16_t vl, uint8_t pFLEN)
{
    VTYPE::VTYPE _vt(vtype);

    if (vl == 0)
    {
        return 0;
    }

    uint8_t *VectorRegField;
    uint8_t *ScalarReg;
    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (pFLEN <= 32)
    {
        ScalarReg = &((static_cast<uint8_t *>(pF))[rs1 * 4]);
    }
    else
    {
        ScalarReg = &(static_cast<uint8_t *>(pF)[rs1 * 8]);
    }

    VInstrInfo v_instr_info{ .lmul_num = _vt._z_lmul,
                             .lmul_denom = _vt._n_lmul,
                             .sew = _vt._sew,
                             .vector_length = vl,
                             .vector_register_length = vlen,
                             .start_element = vstart };

    VPERM::perm_op_move_float(VectorRegField, v_instr_info, vd, ScalarReg, pFLEN, true);

    return 0;
}
/* End 16.2. */

/* 16.3. Vector Slide Instructions */
/* End 16.3. */
/* 16.4. Vector Register Gather Instructions */
uint8_t vrgather_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                    uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VPERM::vrgather_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, /* ei16 = */ false);

    return (0);
}

uint8_t vrgatherei16_vv(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs1,
                        uint8_t vs2, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VPERM::vrgather_vv(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart,
                       masked_instruction_bit, /* ei16 = */ true);

    return (0);
}

uint8_t vrgather_vi(void *vector_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd, uint8_t vs2,
                    uint8_t imm, uint16_t vstart, uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VPERM::vrgather_vi(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, imm, vstart,
                       masked_instruction_bit);

    return (0);
}

uint8_t vrgather_vx(void *vector_field, void *scalar_field, uint16_t vtype, uint8_t masked_instruction_bit, uint8_t vd,
                    uint8_t vs2, uint8_t rs1, uint16_t vstart, uint16_t vlen, uint16_t vl, uint8_t xlen)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *ScalarReg;
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);
    if (xlen <= 32)
        ScalarReg = &((static_cast<uint8_t *>(scalar_field))[rs1 * 4]);
    else
        ScalarReg = &(static_cast<uint8_t *>(scalar_field)[rs1 * 8]);

    VPERM::vrgather_vx(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, ScalarReg, vstart,
                       masked_instruction_bit, xlen / 8);

    return (0);
}
/* End 16.4. */
/* 16.5. Vector Compress Instruction */
uint8_t vcompress_vm(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t vs1, uint8_t vs2, uint16_t vstart,
                     uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VPERM::vcompress_vm(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs1, vs2, vstart);

    return (0);
}
/* End 16.5. */
/* 16.6. Whole Vector Register Move */
uint8_t vmvr_v(void *vector_field, uint16_t vtype, uint8_t vd, uint8_t vs2, uint8_t simm5, uint16_t vstart,
               uint16_t vlen, uint16_t vl)
{
    VTYPE::VTYPE _vt(vtype);
    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    VPERM::vmvr_v(VectorRegField, _vt._z_lmul, _vt._n_lmul, _vt._sew / 8, vl, vlen / 8, vd, vs2, simm5, vstart);

    return 0;
}
/* End 16.6. */
/* End 16. */

/* --- Private function definitions --- */

/**
 * @brief Extract SEW in bit from VTYPE
 */
inline unsigned decode_sew(uint32_t const vtype)
{
    static constexpr auto sew_offset = 3;
    static constexpr auto sew_bitmask = 0b11;

    auto const vsew = (vtype >> sew_offset) & sew_bitmask;
    return 8 << vsew;
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vv(void *vector_field, uint16_t vstart, uint16_t vl, unsigned vd_base, unsigned vs1_base,
                unsigned vs2_base, OpType op)
{
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            // Casting signed to larger unsigned will sign extend.
            // As vector elements can be interpreted as int or uint, this should already take care of signed/unsigned
            // instructions
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, BitResultOp>)
        {
            static constexpr auto sew = sizeof(VectorElementType) * 8;
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], vector_elements[vs1_base + i])
                                                    << (i % sew);
        }
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vxi(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                 unsigned const vs2_base, uint64_t const scalar, OpType const op)
{
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar);
        }
        else if constexpr (std::is_same_v<OpType, BitResultOp>)
        {
            static constexpr auto sew = sizeof(VectorElementType) * 8;
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], scalar) << (i % sew);
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
        auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], vector_elements[vs1_base + i]);
        }
        else if constexpr (std::is_same_v<OpType, BitResultOp>)
        {
            static constexpr auto sew_bytes = sizeof(VectorElementType);
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], vector_elements[vs1_base + i])
                                                    << (i % sew);
        }
    }
}

template <typename VectorElementType, typename OpType>
    requires ValidVectorElementType<VectorElementType> and ValidOperation<OpType>
void iterate_vxi_masked(void *vector_field, uint16_t const vstart, uint16_t const vl, unsigned const vd_base,
                        unsigned const vs2_base, uint64_t const scalar, OpType const op)
{
    auto vector_elements = static_cast<VectorElementType *>(vector_field);
    static constexpr auto sew = sizeof(VectorElementType) * 8;
    for (size_t i = vstart; i < vl; ++i)
    {
        auto const mask_bit = static_cast<bool>((vector_elements[i / sew] >> (i % sew)) & 1);
        if (mask_bit == masked_element_value)
        {
            continue;
        }
        if constexpr (std::is_same_v<OpType, ValueResultOp>)
        {
            vector_elements[vd_base + i] = op(vector_elements[vs2_base + i], scalar);
        }
        else if constexpr (std::is_same_v<OpType, BitResultOp>)
        {
            vector_elements[vd_base + (i / sew)] |= op(vector_elements[vs2_base + i], scalar) << (i % sew);
        }
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

    if (static_cast<bool>(mask_bit) == masked_instruction_value)
    {
        switch (sew_bytes)
        {
        case sew_8_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vv_masked<int8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            else
            {
                iterate_vv_masked<uint8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            break;
        case sew_16_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vv_masked<int16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            else
            {
                iterate_vv_masked<uint16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            break;
        case sew_32_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vv_masked<int32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            else
            {
                iterate_vv_masked<uint32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            break;
        case sew_64_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vv_masked<int64_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            else
            {
                iterate_vv_masked<uint64_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            break;
        default:
            // Invalid SEW
            break;
        }
    }
    else
    {
        switch (sew_bytes)
        {
        case sew_8_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vv<int8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            else
            {
                iterate_vv<uint8_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            break;
        case sew_16_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vv<int16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            else
            {
                iterate_vv<uint16_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            break;
        case sew_32_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vv<int32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            else
            {
                iterate_vv<uint32_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            break;
        case sew_64_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vv<int64_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            else
            {
                iterate_vv<uint64_t>(vector_field, vstart, vl, vd_base, vs1_base, vs2_base, op);
            }
            break;
        default:
            // Invalid SEW
            break;
        }
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

    uint64_t scalar = 0;
    if constexpr (Sign == SignType::Signed)
    {
        switch (xlen)
        {
        case 32:
            scalar = (static_cast<int32_t *>(scalar_field))[rs1];
            break;
        case 64:
            scalar = (static_cast<int64_t *>(scalar_field))[rs1];
            break;
        default:
            // Invalid XLEN!
            break;
        }
    }
    else
    {
        switch (xlen)
        {
        case 32:
            scalar = (static_cast<uint32_t *>(scalar_field))[rs1];
            break;
        case 64:
            scalar = (static_cast<uint64_t *>(scalar_field))[rs1];
            break;
        default:
            // Invalid XLEN!
            break;
        }
    }

    if (static_cast<bool>(mask_bit) == masked_instruction_value)
    {
        switch (sew_bytes)
        {
        case sew_8_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi_masked<int8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi_masked<uint8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_16_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi_masked<int16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi_masked<uint16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_32_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi_masked<int32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi_masked<uint32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_64_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi_masked<int64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi_masked<uint64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        default:
            // Invalid SEW
            break;
        }
    }
    else
    {
        switch (sew_bytes)
        {
        case sew_8_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi<int8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi<uint8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_16_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi<int16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi<uint16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_32_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi<int32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi<uint32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_64_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi<int64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi<uint64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        default:
            // Invalid SEW
            break;
        }
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

    uint64_t scalar = immediate;
    if constexpr (ImmExtension == ImmExtensionType::SignExtend)
    {
        scalar = sign_extend_immediate(scalar);
    }

    if (static_cast<bool>(mask_bit) == masked_instruction_value)
    {
        switch (sew_bytes)
        {
        case sew_8_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi_masked<int8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi_masked<uint8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_16_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi_masked<int16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi_masked<uint16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_32_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi_masked<int32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi_masked<uint32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_64_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi_masked<int64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi_masked<uint64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        default:
            // Invalid SEW
            break;
        }
    }
    else
    {
        switch (sew_bytes)
        {
        case sew_8_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi<int8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi<uint8_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_16_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi<int16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi<uint16_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_32_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi<int32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi<uint32_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        case sew_64_bytes:
            if constexpr (Sign == SignType::Signed)
            {
                iterate_vxi<int64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            else
            {
                iterate_vxi<uint64_t>(vector_field, vstart, vl, vd_base, vs2_base, scalar, op);
            }
            break;
        default:
            // Invalid SEW
            break;
        }
    }
}

int8_t vtype_decode(uint16_t vtype, uint8_t *ta, uint8_t *ma, uint32_t *sew, uint8_t *z_lmul, uint8_t *n_lmul)
{
    return (VTYPE::decode(vtype, ta, ma, sew, z_lmul, n_lmul));
}

uint16_t vtype_encode(uint16_t sew, uint8_t z_lmul, uint8_t n_lmul, uint8_t ta, uint8_t ma)
{
    return VTYPE::encode(sew, z_lmul, n_lmul, ta, ma);
}

uint8_t vtype_extractSEW(uint16_t vtype)
{
    return VTYPE::extractSEW(vtype);
}

uint8_t vtype_extractLMUL(uint16_t vtype)
{
    return VTYPE::extractLMUL(vtype);
}

uint8_t vtype_extractTA(uint16_t vtype)
{
    return VTYPE::extractTA(vtype);
}

uint8_t vtype_extractMA(uint16_t vtype)
{
    return VTYPE::extractMA(vtype);
}

uint16_t vcfg_concatEEW(uint8_t mew, uint8_t width)
{
    return (VTYPE::concatEEW(mew, width));
}

uint8_t vload_encoded_unitstride(void *vector_field, uint8_t *pM, uint16_t vtype, uint8_t masked_instruction_bit,
                                 uint16_t pEEW, uint8_t vd, uint16_t vstart, uint16_t vlen, uint16_t vl,
                                 uint64_t pMSTART)
{
    VTYPE::VTYPE _vt(vtype);
    uint64_t _z_emul = pEEW * _vt._z_lmul;
    uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * 8) || (_z_emul > _n_emul * 8))
        return 1;

    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    std::function<void(std::size_t, uint8_t *, std::size_t)> f_readMem =
        [pM](std::size_t addr, uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            buff[i] = pM[addr + i];
    };

    VLSU::load_eew(f_readMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, vl, vlen / 8, vd, pMSTART, vstart,
                   masked_instruction_bit, pEEW / 8);

    return (0);
}

uint8_t vload_encoded_stride(void *vector_field, uint8_t *pM, uint16_t vtype, uint8_t masked_instruction_bit,
                             uint16_t pEEW, uint8_t vd, uint16_t vstart, uint16_t vlen, uint16_t vl,
                             uint64_t pMSTART, int16_t pSTRIDE)
{
    VTYPE::VTYPE _vt(vtype);
    uint64_t _z_emul = pEEW * _vt._z_lmul;
    uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * 8) || (_z_emul > _n_emul * 8))
    {
        return 1;
    }

    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    std::function<void(std::size_t, uint8_t *, std::size_t)> f_readMem =
        [pM](std::size_t addr, uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            buff[i] = pM[addr + i];
    };

    VLSU::load_eew(f_readMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, vl, vlen / 8, vd, pMSTART, vstart,
                   masked_instruction_bit, pSTRIDE);

    return (0);
}

uint8_t vload_segment_unitstride(void *vector_field, uint8_t *pM, uint16_t vtype, uint8_t masked_instruction_bit,
                                 uint16_t pEEW, uint8_t pNF, uint8_t vd, uint16_t vstart, uint16_t vlen, uint16_t vl,
                                 uint64_t pMSTART)
{
    VTYPE::VTYPE _vt(vtype);
    uint64_t _z_emul = pEEW * _vt._z_lmul;
    uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * pNF * 8) || (_z_emul * pNF > _n_emul * 8))
        return 1;
    if ((vd + pNF * _z_emul / _n_emul) > 32)
        return 1;
    if (vstart >= vl)
        return (0);

    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    std::function<void(std::size_t, uint8_t *, std::size_t)> f_readMem =
        [pM](std::size_t addr, uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            buff[i] = pM[addr + i];
    };

    uint16_t _vstart = vstart;
    uint64_t _moffset = pMSTART;

    for (int i = 0; i < pNF; ++i)
    {
        VLSU::load_eew(f_readMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, vl, vlen / 8,
                       vd + (i * _z_emul / _n_emul), _moffset, _vstart, masked_instruction_bit, pEEW / 8);
        _moffset += (vl - _vstart) * pEEW / 8;
        _vstart = 0;
    }

    return (0);
}

uint8_t vload_segment_stride(void *vector_field, uint8_t *pM, uint16_t vtype, uint8_t masked_instruction_bit,
                             uint16_t pEEW, uint8_t pNF, uint8_t vd, uint16_t vstart, uint16_t vlen, uint16_t vl,
                             uint64_t pMSTART, int16_t pSTRIDE)
{
    VTYPE::VTYPE _vt(vtype);
    uint64_t _z_emul = pEEW * _vt._z_lmul;
    uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * pNF * 8) || (_z_emul * pNF > _n_emul * 8))
        return 1;
    if ((vd + pNF * _z_emul / _n_emul) > 32)
        return 1;
    if (vstart >= vl)
        return (0);

    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    std::function<void(std::size_t, uint8_t *, std::size_t)> f_readMem =
        [pM](std::size_t addr, uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            buff[i] = pM[addr + i];
    };

    uint16_t _vstart = vstart;
    uint64_t _moffset = pMSTART;

    for (int i = 0; i < pNF; ++i)
    {
        _moffset = pMSTART + i * pEEW / 8;
        VLSU::load_eew(f_readMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, vl, vlen / 8,
                       vd + (i * _z_emul / _n_emul), _moffset, _vstart, masked_instruction_bit, pSTRIDE);
        _vstart = 0;
    }

    return (0);
}

uint8_t vstore_encoded_unitstride(void *vector_field, uint8_t *pM, uint16_t vtype, uint8_t masked_instruction_bit,
                                  uint16_t pEEW, uint8_t vd, uint16_t vstart, uint16_t vlen, uint16_t vl,
                                  uint64_t pMSTART)
{
    VTYPE::VTYPE _vt(vtype);
    uint64_t _z_emul = pEEW * _vt._z_lmul;
    uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * 8) || (_z_emul > _n_emul * 8))
        return 1;

    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    std::function<void(std::size_t, uint8_t *, std::size_t)> f_writeMem =
        [pM](std::size_t addr, uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            pM[addr + i] = buff[i];
    };

    VLSU::store_eew(f_writeMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, vl, vlen / 8, vd, pMSTART, vstart,
                    masked_instruction_bit, pEEW / 8);

    return (0);
}

uint8_t vstore_encoded_stride(void *vector_field, uint8_t *pM, uint16_t vtype, uint8_t masked_instruction_bit,
                              uint16_t pEEW, uint8_t vd, uint16_t vstart, uint16_t vlen, uint16_t vl,
                              uint64_t pMSTART, int16_t pStride)
{
    VTYPE::VTYPE _vt(vtype);
    uint64_t _z_emul = pEEW * _vt._z_lmul;
    uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * 8) || (_z_emul > _n_emul * 8))
        return 1;

    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    std::function<void(std::size_t, uint8_t *, std::size_t)> f_writeMem =
        [pM](std::size_t addr, uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            pM[addr + i] = buff[i];
    };
    VLSU::store_eew(f_writeMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, vl, vlen / 8, vd, pMSTART, vstart,
                    masked_instruction_bit, pStride);

    return (0);
}

uint8_t vstore_segment_unitstride(void *vector_field, uint8_t *pM, uint16_t vtype, uint8_t masked_instruction_bit,
                                  uint16_t pEEW, uint8_t pNF, uint8_t vd, uint16_t vstart, uint16_t vlen, uint16_t vl,
                                  uint64_t pMSTART)
{
    VTYPE::VTYPE _vt(vtype);
    uint64_t _z_emul = pEEW * _vt._z_lmul;
    uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * pNF * 8) || (_z_emul * pNF > _n_emul * 8))
        return 1;
    if ((vd + pNF * _z_emul / _n_emul) > 32)
        return 1;
    if (vstart >= vl)
        return (0);

    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    std::function<void(std::size_t, uint8_t *, std::size_t)> f_writeMem =
        [pM](std::size_t addr, uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            pM[addr + i] = buff[i];
    };

    uint16_t _vstart = vstart;
    uint64_t _moffset = pMSTART;

    for (int i = 0; i < pNF; ++i)
    {
        VLSU::store_eew(f_writeMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, vl, vlen / 8,
                        vd + (i * _z_emul / _n_emul), _moffset, _vstart, masked_instruction_bit, pEEW / 8);
        _moffset += (vl - _vstart) * pEEW / 8;
        _vstart = 0;
    }

    return (0);
}

uint8_t vstore_segment_stride(void *vector_field, uint8_t *pM, uint16_t vtype, uint8_t masked_instruction_bit,
                              uint16_t pEEW, uint8_t pNF, uint8_t vd, uint16_t vstart, uint16_t vlen, uint16_t vl,
                              uint64_t pMSTART, int16_t pStride)
{
    VTYPE::VTYPE _vt(vtype);
    uint64_t _z_emul = pEEW * _vt._z_lmul;
    uint64_t _n_emul = _vt._sew * _vt._n_lmul;

    if ((_n_emul > _z_emul * pNF * 8) || (_z_emul * pNF > _n_emul * 8))
        return 1;
    if ((vd + pNF * _z_emul / _n_emul) > 32)
        return 1;
    if (vstart >= vl)
        return (0);

    uint8_t *VectorRegField;

    VectorRegField = static_cast<uint8_t *>(vector_field);

    std::function<void(std::size_t, uint8_t *, std::size_t)> f_writeMem =
        [pM](std::size_t addr, uint8_t *buff, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            pM[addr + i] = buff[i];
    };

    uint16_t _vstart = vstart;
    uint64_t _moffset = pMSTART;
    for (int i = 0; i < pNF; ++i)
    {
        _moffset = pMSTART + i * pEEW / 8;
        VLSU::store_eew(f_writeMem, VectorRegField, _z_emul, _n_emul, pEEW / 8, vl, vlen / 8,
                        vd + (i * _z_emul / _n_emul), _moffset, _vstart, masked_instruction_bit, pStride);
        _moffset += (vl - _vstart) * pEEW / 8;
        _vstart = 0;
    }

    return (0);
}

//} // extern "C"
