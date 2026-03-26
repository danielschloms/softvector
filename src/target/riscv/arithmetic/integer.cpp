/*
 * Copyright [2020] [Technical University of Munich]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//////////////////////////////////////////////////////////////////////////////////////
/// \file integer.cpp
/// \brief C++ Source for vector integer arithmetic helpers for RISC-V ISS
/// \date 06/23/2020
//////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cstddef>

#include "arithmetic/integer.hpp"
#include "base/base.hpp"
#include "vpu/softvector-types.hpp"
#include "base/softvector-platform-types.hpp"

// Private function declarations

auto iterate_vector(SVector const &vs2, SVector const &vs1, SVector &vd, SVRegister const &vm, bool mask,
                    VARITH_INT::deprecated::IntFunctionDeprecated func, std::size_t start_index, bool signed_vs2,
                    bool signed_vs1, bool mask_is_data) -> void;

auto iterate_vector(SVector const &vs2, uint64_t scalar, SVector &vd, SVRegister const &vm, bool mask,
                    VARITH_INT::deprecated::IntFunctionDeprecated func, std::size_t start_index, bool signed_vs2,
                    bool mask_is_data) -> void;

auto iterate_vector_to_register(SVector const &vs2, SVector const &vs1, SVRegister &vd, SVRegister const &vm, bool mask,
                                VARITH_INT::deprecated::IntRegisterFunction func, std::size_t start_index,
                                bool signed_vs2, bool signed_vs1, bool mask_is_data, std::size_t sew) -> void;

auto iterate_vector_to_register(SVector const &vs2, uint64_t scalar, SVRegister &vd, SVRegister const &vm,
                                bool mask, VARITH_INT::deprecated::IntRegisterFunction func, std::size_t start_index,
                                bool signed_vs2, bool mask_is_data, std::size_t sew) -> void;

// Private function definitions

auto iterate_vector(SVector const &vs2, SVector const &vs1, SVector &vd, SVRegister const &vm, bool mask,
                    VARITH_INT::deprecated::IntFunctionDeprecated func, std::size_t start_index, bool signed_vs2,
                    bool signed_vs1, bool mask_is_data) -> void
{
    for (std::size_t i_element = start_index; i_element < vd.length_; ++i_element)
    {
        auto mask_bit = vm.get_bit(i_element);
        if (!mask || mask_bit || mask_is_data)
        {
            auto lhs = signed_vs2 ? vs2[i_element].to_i64() : vs2[i_element].to_u64();
            auto rhs = signed_vs1 ? vs1[i_element].to_i64() : vs1[i_element].to_u64();
            func(lhs, rhs, vd[i_element], mask && mask_bit && mask_is_data);
        }
    }
}

auto iterate_vector(SVector const &vs2, uint64_t scalar, SVector &vd, SVRegister const &vm, bool mask,
                    VARITH_INT::deprecated::IntFunctionDeprecated func, std::size_t start_index, bool signed_vs2,
                    bool mask_is_data) -> void
{
    for (std::size_t i_element = start_index; i_element < vd.length_; ++i_element)
    {
        auto mask_bit = vm.get_bit(i_element);
        if (!mask || mask_bit || mask_is_data)
        {
            uint64_t lhs = signed_vs2 ? vs2[i_element].to_i64() : vs2[i_element].to_u64();
            func(lhs, scalar, vd[i_element], mask && mask_bit && mask_is_data);
        }
    }
}

auto iterate_vector_to_register(SVector const &vs2, SVector const &vs1, SVRegister &vd, SVRegister const &vm, bool mask,
                                VARITH_INT::deprecated::IntRegisterFunction func, std::size_t start_index,
                                bool signed_vs2, bool signed_vs1, bool mask_is_data, std::size_t sew) -> void
{
    for (std::size_t i_element = start_index; i_element < vs2.length_; ++i_element)
    {
        auto mask_bit = vm.get_bit(i_element);
        if (!mask || mask_bit || mask_is_data)
        {
            auto lhs = signed_vs2 ? vs2[i_element].to_i64() : vs2[i_element].to_u64();
            auto rhs = signed_vs1 ? vs1[i_element].to_i64() : vs1[i_element].to_u64();
            auto ret = func(lhs, rhs, sew, mask && mask_bit && mask_is_data);
            ret ? vd.set_bit(i_element) : vd.reset_bit(i_element);
        }
    }
}

auto iterate_vector_to_register(SVector const &vs2, uint64_t const scalar, SVRegister &vd, SVRegister const &vm,
                                bool const mask, VARITH_INT::deprecated::IntRegisterFunction func,
                                std::size_t const start_index, bool const signed_vs2, bool const mask_is_data,
                                std::size_t const sew) -> void
{
    for (std::size_t i_element = start_index; i_element < vs2.length_; ++i_element)
    {
        auto mask_bit = vm.get_bit(i_element);
        if (!mask || mask_bit || mask_is_data)
        {
            auto lhs = signed_vs2 ? vs2[i_element].to_i64() : vs2[i_element].to_u64();
            auto ret = func(lhs, scalar, sew, mask && mask_bit && mask_is_data);
            ret ? vd.set_bit(i_element) : vd.reset_bit(i_element);
        }
    }
}

// Public function definitions

VILL::vpu_return_t VARITH_INT::int_op_vv(uint8_t *vec_reg_mem, VInstrInfo const &v_instr_info,
                                         const IntInstrInfo &int_instr_info, uint16_t const reg_vd,
                                         uint16_t const reg_vs1, uint16_t const reg_vs2,
                                         deprecated::IntFunctionDeprecated func)
{
    RVVRegField V(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
                  SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    RVVRegField V_wide(v_instr_info.vector_register_length, v_instr_info.vector_length, 2 * v_instr_info.sew,
                       SVMul(2 * v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    auto alignment_exception =
        check_alignment(V, V_wide, reg_vd, reg_vs2, reg_vs1, v_instr_info.wide_vd, v_instr_info.wide_vs2);
    if (alignment_exception != VILL::vpu_return_t::NO_EXCEPT)
    {
        return alignment_exception;
    }

    V.init();
    if (v_instr_info.wide_vd || v_instr_info.wide_vs2)
    {
        V_wide.init();
    }

    RVVector &vs1 = V.get_vec(reg_vs1);
    RVVector &vs2 = v_instr_info.wide_vs2 ? V_wide.get_vec(reg_vs2) : V.get_vec(reg_vs2);
    RVVector &vd = v_instr_info.wide_vd ? V_wide.get_vec(reg_vd) : V.get_vec(reg_vd);

    // Mixed-signed: vs2 is signed if it is a signed-unsigned instruction
    auto signed_vs2 = int_instr_info.mixed_signed ? int_instr_info.mixed_signed_vs2_signed : v_instr_info.signed_op;
    // Mixed-signed: vs1 is signed if vs2 is unsigned and vice versa
    auto signed_vs1 = int_instr_info.mixed_signed ? !signed_vs2 : v_instr_info.signed_op;

    iterate_vector(vs2, vs1, vd, V.get_mask_reg(), v_instr_info.masked, func, v_instr_info.start_element, signed_vs2,
                   signed_vs1, int_instr_info.mask_is_data);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::int_op_vi(uint8_t *vec_reg_mem, VInstrInfo const &v_instr_info,
                                         IntInstrInfo const &int_instr_info, uint16_t const reg_vd,
                                         uint16_t const reg_vs2, uint8_t imm5,
                                         deprecated::IntFunctionDeprecated func)
{
    RVVRegField V(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
                  SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(reg_vs2))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    if (!V.vec_reg_is_aligned(reg_vd))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }

    V.init();

    // For instructions with specific uimm, just zero extend, otherwise sign extend
    uint64_t imm = v_instr_info.zero_extend_immediate ? zero_extend_immediate(imm5) : sign_extend_immediate(imm5);

    RVVector &vs2 = V.get_vec(reg_vs2);
    RVVector &vd = V.get_vec(reg_vd);

    iterate_vector(vs2, imm, vd, V.get_mask_reg(), v_instr_info.masked, func, v_instr_info.start_element,
                   v_instr_info.signed_op, int_instr_info.mask_is_data);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::int_op_vx(uint8_t *vec_reg_mem, const VInstrInfo &v_instr_info,
                                         const IntInstrInfo &int_instr_info, uint16_t const reg_vd,
                                         uint16_t const reg_vs2, uint8_t *scalar_reg_mem,
                                         uint8_t scalar_reg_len_bytes, deprecated::IntFunctionDeprecated func)
{
    RVVRegField V(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
                  SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    RVVRegField V_wide(v_instr_info.vector_register_length, v_instr_info.vector_length, 2 * v_instr_info.sew,
                       SVMul(2 * v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    auto alignment_exception = check_alignment(V, V_wide, reg_vd, reg_vs2, v_instr_info.wide_vd, v_instr_info.wide_vs2);
    if (alignment_exception != VILL::vpu_return_t::NO_EXCEPT)
    {
        return alignment_exception;
    }

    V.init();
    if (v_instr_info.wide_vd || v_instr_info.wide_vs2)
    {
        V_wide.init();
    }

    uint64_t scalar = (scalar_reg_len_bytes > xlen_32_bytes)
                               ? *(reinterpret_cast<uint64_t *>(scalar_reg_mem))
                               : *(reinterpret_cast<uint32_t *>(scalar_reg_mem));

    // Mixed-signed: vs2 is signed if it is a signed-unsigned instruction
    auto signed_vs2 = int_instr_info.mixed_signed ? int_instr_info.mixed_signed_vs2_signed : v_instr_info.signed_op;
    // Mixed-signed: Scalar is signed if vs2 is unsigned and vice versa
    auto signed_scalar = int_instr_info.mixed_signed ? !signed_vs2 : v_instr_info.signed_op;

    scalar = mask_and_sign_extend_scalar(scalar, v_instr_info.sew, signed_scalar);

    RVVector &vs2 = v_instr_info.wide_vs2 ? V_wide.get_vec(reg_vs2) : V.get_vec(reg_vs2);
    RVVector &vd = v_instr_info.wide_vd ? V_wide.get_vec(reg_vd) : V.get_vec(reg_vd);

    iterate_vector(vs2, scalar, vd, V.get_mask_reg(), v_instr_info.masked, func, v_instr_info.start_element, signed_vs2,
                   int_instr_info.mask_is_data);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}

auto VARITH_INT::int_op_vv_to_register(uint8_t *vec_reg_mem, VInstrInfo const &v_instr_info,
                                       IntInstrInfo const &int_instr_info, uint16_t const reg_vd,
                                       uint16_t const reg_vs1, uint16_t const reg_vs2,
                                       deprecated::IntRegisterFunction func) -> VILL::vpu_return_t
{
    RVVRegField V(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
                  SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    RVVRegField V_wide(v_instr_info.vector_register_length, v_instr_info.vector_length, 2 * v_instr_info.sew,
                       SVMul(2 * v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    auto alignment_exception =
        check_alignment(V, V_wide, reg_vd, reg_vs2, reg_vs1, v_instr_info.wide_vd, v_instr_info.wide_vs2);
    if (alignment_exception != VILL::vpu_return_t::NO_EXCEPT)
    {
        return alignment_exception;
    }

    V.init();
    if (v_instr_info.wide_vd || v_instr_info.wide_vs2)
    {
        V_wide.init();
    }

    RVVector &vs1 = V.get_vec(reg_vs1);
    RVVector &vs2 = v_instr_info.wide_vs2 ? V_wide.get_vec(reg_vs2) : V.get_vec(reg_vs2);
    SVRegister &vd = v_instr_info.wide_vd ? V_wide.get_vecreg(reg_vd) : V.get_vecreg(reg_vd);

    // Mixed-signed: vs2 is signed if it is a signed-unsigned instruction
    auto signed_vs2 = int_instr_info.mixed_signed ? int_instr_info.mixed_signed_vs2_signed : v_instr_info.signed_op;
    // Mixed-signed: vs1 is signed if vs2 is unsigned and vice versa
    auto signed_vs1 = int_instr_info.mixed_signed ? !signed_vs2 : v_instr_info.signed_op;

    iterate_vector_to_register(vs2, vs1, vd, V.get_mask_reg(), v_instr_info.masked, func, v_instr_info.start_element,
                               signed_vs2, signed_vs1, int_instr_info.mask_is_data, v_instr_info.sew);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}

auto VARITH_INT::int_op_vi_to_register(uint8_t *vec_reg_mem, VInstrInfo const &v_instr_info,
                                       IntInstrInfo const &int_instr_info, uint16_t const reg_vd,
                                       uint16_t const reg_vs2, uint8_t const imm5,
                                       deprecated::IntRegisterFunction func) -> VILL::vpu_return_t
{
    RVVRegField V(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
                  SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(reg_vs2))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    if (!V.vec_reg_is_aligned(reg_vd))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }

    V.init();

    uint64_t imm = v_instr_info.zero_extend_immediate ? zero_extend_immediate(imm5) : sign_extend_immediate(imm5);

    RVVector &vs2 = V.get_vec(reg_vs2);
    SVRegister &vd = V.get_vecreg(reg_vd);

    iterate_vector_to_register(vs2, imm, vd, V.get_mask_reg(), v_instr_info.masked, func, v_instr_info.start_element,
                               v_instr_info.signed_op, int_instr_info.mask_is_data, v_instr_info.sew);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}

auto VARITH_INT::int_op_vx_to_register(uint8_t *vec_reg_mem, VInstrInfo const &v_instr_info,
                                       IntInstrInfo const &int_instr_info, uint16_t const reg_vd, uint16_t const reg_vs2,
                                       uint8_t *scalar_reg_mem, uint8_t const scalar_reg_len_bytes,
                                       deprecated::IntRegisterFunction func) -> VILL::vpu_return_t
{
    RVVRegField V(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
                  SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(reg_vs2))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    if (!V.vec_reg_is_aligned(reg_vd))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }

    V.init();

    uint64_t imm = (scalar_reg_len_bytes > xlen_32_bytes) ? *(reinterpret_cast<uint64_t *>(scalar_reg_mem))
                                                               : *(reinterpret_cast<uint32_t *>(scalar_reg_mem));
    imm = mask_and_sign_extend_scalar(imm, v_instr_info.sew, v_instr_info.signed_op);
    RVVector &vs2 = V.get_vec(reg_vs2);
    SVRegister &vd = V.get_vecreg(reg_vd);

    iterate_vector_to_register(vs2, imm, vd, V.get_mask_reg(), v_instr_info.masked, func, v_instr_info.start_element,
                               v_instr_info.signed_op, int_instr_info.mask_is_data, v_instr_info.sew);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}

/* End 11.2. */

/* 11.16. Vector Integer Move Instructions */
VILL::vpu_return_t VARITH_INT::mv_vv(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                     uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                     uint16_t dst_vec_reg, uint16_t src_vec_reg, uint16_t vec_elem_start)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg))
    {
        return (VILL::VPU_RETURN::SRC1_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        RVVector &vs1 = V.get_vec(src_vec_reg);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_assign(vs1, V.get_mask_reg(), false, vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::mv_vx(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                     uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                     uint16_t dst_vec_reg, uint8_t *scalar_reg_mem,
                                     uint16_t vec_elem_start, uint8_t scalar_reg_len_bytes)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        int64_t imm = (scalar_reg_len_bytes > 4) ? *(reinterpret_cast<int64_t *>(scalar_reg_mem))
                                                 : *(reinterpret_cast<int32_t *>(scalar_reg_mem));
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_assign(imm, V.get_mask_reg(), false, vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::mv_vi(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                     uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                     uint16_t dst_vec_reg, uint8_t s_imm, uint16_t vec_elem_start)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        int64_t imm = static_cast<int64_t>(s_imm & 0x10 ? s_imm | ~0x1F : s_imm);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_assign(imm, V.get_mask_reg(), false, vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}
/* End 11.16. */

/* 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
VILL::vpu_return_t VARITH_INT::vadc_vvm(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_rhs,
                                        uint16_t src_vec_reg_lhs, uint16_t vec_elem_start)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_rhs))
    {
        return (VILL::VPU_RETURN::SRC1_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        RVVector &vs1 = V.get_vec(src_vec_reg_rhs);
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_adc(vs2, vs1, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vadc_vim(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_lhs, uint8_t s_imm,
                                        uint16_t vec_elem_start)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        int64_t imm = static_cast<int64_t>(s_imm & 0x10 ? s_imm | ~0x1F : s_imm);
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_adc(vs2, imm, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vadc_vxm(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_lhs,
                                        uint8_t *scalar_reg_mem, uint16_t vec_elem_start,
                                        uint8_t scalar_reg_len_bytes)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        int64_t imm = (scalar_reg_len_bytes > 4) ? *(reinterpret_cast<int64_t *>(scalar_reg_mem))
                                                 : *(reinterpret_cast<int32_t *>(scalar_reg_mem));
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_adc(vs2, imm, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vmadc_vv(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_rhs,
                                        uint16_t src_vec_reg_lhs, uint16_t vec_elem_start, bool mask_f)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_rhs))
    {
        return (VILL::VPU_RETURN::SRC1_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        RVVector &vs1 = V.get_vec(src_vec_reg_rhs);
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        SVRegister &vd = V.get_vecreg(dst_vec_reg);

        vd.m_madc(vs2, vs1, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vmadc_vi(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_lhs, uint8_t s_imm,
                                        uint16_t vec_elem_start, bool mask_f)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        int64_t imm = static_cast<int64_t>(s_imm & 0x10 ? s_imm | ~0x1F : s_imm);
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        SVRegister &vd = V.get_vecreg(dst_vec_reg);

        vd.m_madc(vs2, imm, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vmadc_vx(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_lhs,
                                        uint8_t *scalar_reg_mem, uint16_t vec_elem_start, bool mask_f,
                                        uint8_t scalar_reg_len_bytes)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        int64_t imm = (scalar_reg_len_bytes > 4) ? *(reinterpret_cast<int64_t *>(scalar_reg_mem))
                                                 : *(reinterpret_cast<int32_t *>(scalar_reg_mem));
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        SVRegister &vd = V.get_vecreg(dst_vec_reg);

        vd.m_madc(vs2, imm, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vsbc_vvm(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_rhs,
                                        uint16_t src_vec_reg_lhs, uint16_t vec_elem_start)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_rhs))
    {
        return (VILL::VPU_RETURN::SRC1_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        RVVector &vs1 = V.get_vec(src_vec_reg_rhs);
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_sbc(vs2, vs1, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vsbc_vxm(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_lhs,
                                        uint8_t *scalar_reg_mem, uint16_t vec_elem_start,
                                        uint8_t scalar_reg_len_bytes)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        int64_t imm = (scalar_reg_len_bytes > 4) ? *(reinterpret_cast<int64_t *>(scalar_reg_mem))
                                                 : *(reinterpret_cast<int32_t *>(scalar_reg_mem));
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_sbc(vs2, imm, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vmsbc_vv(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_rhs,
                                        uint16_t src_vec_reg_lhs, uint16_t vec_elem_start, bool mask_f)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_rhs))
    {
        return (VILL::VPU_RETURN::SRC1_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        RVVector &vs1 = V.get_vec(src_vec_reg_rhs);
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        SVRegister &vd = V.get_vecreg(dst_vec_reg);

        vd.m_msbc(vs2, vs1, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vmsbc_vx(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                        uint16_t sew_bytes, uint16_t vec_len, uint16_t vec_reg_len_bytes,
                                        uint16_t dst_vec_reg, uint16_t src_vec_reg_lhs,
                                        uint8_t *scalar_reg_mem, uint16_t vec_elem_start, bool mask_f,
                                        uint8_t scalar_reg_len_bytes)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    else if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }
    else
    {
        V.init();

        int64_t imm = (scalar_reg_len_bytes > 4) ? *(reinterpret_cast<int64_t *>(scalar_reg_mem))
                                                 : *(reinterpret_cast<int32_t *>(scalar_reg_mem));
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        SVRegister &vd = V.get_vecreg(dst_vec_reg);

        vd.m_msbc(vs2, imm, V.get_mask_reg(), vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}
/* End 11.4 */

/* 11.15. Vector Integer Merge Instructions */
VILL::vpu_return_t VARITH_INT::vmerge_vv(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                         uint16_t sew_bytes, uint16_t vec_len,
                                         uint16_t vec_reg_len_bytes, uint16_t dst_vec_reg,
                                         uint16_t src_vec_reg_rhs, uint16_t src_vec_reg_lhs,
                                         uint16_t vec_elem_start)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC1_VEC_ILL);
    }
    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }

    V.init();

    RVVector &vs1 = V.get_vec(src_vec_reg_rhs);
    RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
    RVVector &vd = V.get_vec(dst_vec_reg);

    vd.m_merge(vs2, vs1, V.get_mask_reg(), vec_elem_start);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vmerge_vx(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                         uint16_t sew_bytes, uint16_t vec_len,
                                         uint16_t vec_reg_len_bytes, uint16_t dst_vec_reg,
                                         uint16_t src_vec_reg_lhs, uint8_t *scalar_reg_mem,
                                         uint16_t vec_elem_start, uint8_t scalar_reg_len_bytes)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);
    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }

    V.init();

    RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
    RVVector &vd = V.get_vec(dst_vec_reg);

    int64_t imm = (scalar_reg_len_bytes > 4) ? *(reinterpret_cast<int64_t *>(scalar_reg_mem))
                                             : *(reinterpret_cast<int32_t *>(scalar_reg_mem));

    vd.m_merge(vs2, imm, V.get_mask_reg(), vec_elem_start);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_INT::vmerge_vi(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom,
                                         uint16_t sew_bytes, uint16_t vec_len,
                                         uint16_t vec_reg_len_bytes, uint16_t dst_vec_reg,
                                         uint16_t src_vec_reg_lhs, uint8_t s_imm,
                                         uint16_t vec_elem_start)
{
    RVVRegField V(vec_reg_len_bytes * 8, vec_len, sew_bytes * 8, SVMul(emul_num, emul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(src_vec_reg_lhs))
    {
        return (VILL::VPU_RETURN::SRC2_VEC_ILL);
    }
    if (!V.vec_reg_is_aligned(dst_vec_reg))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }

    V.init();

    RVVector &vs2 = V.get_vec(src_vec_reg_lhs);

    int64_t imm = static_cast<int64_t>(s_imm & 0x10 ? s_imm | ~0x1F : s_imm);
    RVVector &vd = V.get_vec(dst_vec_reg);

    vd.m_merge(vs2, imm, V.get_mask_reg(), vec_elem_start);

    return (VILL::VPU_RETURN::NO_EXCEPT);
}
/* End 11.15. */