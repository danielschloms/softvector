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
/// \file integer.hpp
/// \brief Defines helpers implementing integer arithmetics after
/// https://github.com/riscv/riscv-v-spec/blob/0.9/v-spec.adoc#vector-arithmetic-instruction-formats
/// \date 06/23/2020
//////////////////////////////////////////////////////////////////////////////////////

#ifndef __RVVHL_ARITH_INTEGER_H__
#define __RVVHL_ARITH_INTEGER_H__

#include <functional>

#include "stdint.h"
#include "base/base.hpp"
#include "vpu/softvector-types.hpp"

//////////////////////////////////////////////////////////////////////////////////////
/// \brief This space concludes integer arithmetic helpers
namespace VARITH_INT
{

struct IntInstrInfo
{
    bool mixed_signed = false;            //!< True if instruction is mixed-signed (e.g. vmulhsu)
    bool mixed_signed_vs2_signed = false; //!< True if vs2 is signed in mixed-signed instruction
    bool mask_is_data = false;            //!< True if mask register bits are used as data instead of element masks
};

// TODO: Rewrite functions to return the value, not write into a register

using IntFunction = std::function<uint64_t(uint64_t const /* lhs */, uint64_t const /* rhs */,
                                                bool const /* mask_bit */)>;

namespace deprecated
{
using IntFunctionDeprecated =
    std::function<void(uint64_t /* lhs */, uint64_t /* rhs */, SVElement & /* vd */, bool /* mask_bit */)>;
using IntRegisterFunction = std::function<bool(uint64_t /* lhs */, uint64_t /* rhs */, std::size_t /* sew */,
                                               [[maybe_unused]] bool /* mask_bit */)>;

/* 11.1. Vector Single-Width Integer Add and Subtract */
inline IntFunctionDeprecated add = [](uint64_t lhs, uint64_t rhs, SVElement &vd, bool carry_in) -> void
{ vd = lhs + rhs + carry_in; };

inline IntFunctionDeprecated sub = [](uint64_t lhs, uint64_t rhs, SVElement &vd, bool borrow_in) -> void
{ vd = lhs - rhs - borrow_in; };

/* 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
inline IntRegisterFunction produce_carry_out = [](uint64_t lhs, uint64_t rhs, std::size_t sew,
                                                  bool carry_in) -> bool
{
    auto result = lhs + rhs + carry_in;

    auto msb_lhs = msb_is_set(lhs, sew);
    auto msb_rhs = msb_is_set(rhs, sew);
    auto msb_result = msb_is_set(result, sew);

    // Carry out if:
    // - MSB of both operands are set
    // - MSB of one operand is set, but result MSB is not set
    auto carry_out =
        (msb_lhs && msb_rhs) || (msb_lhs && !msb_rhs && !msb_result) || (!msb_lhs && msb_rhs && !msb_result);

    return carry_out;
};

/* 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
inline IntRegisterFunction produce_borrow_out = [](uint64_t lhs, uint64_t rhs, std::size_t sew,
                                                   bool borrow_in) -> bool
{
    auto result = lhs - rhs - borrow_in;

    auto msb_lhs = msb_is_set(lhs, sew);
    auto msb_rhs = msb_is_set(rhs, sew);
    auto msb_result = msb_is_set(result, sew);

    // Borrow out if:
    // - MSB of rhs is set and MSB of lhs is not set
    // - MSB of result is set and MSB of lhs = MSB of rhs
    auto borrow_out =
        (!msb_lhs && msb_rhs) || (msb_lhs && msb_rhs && msb_result) || (!msb_lhs && !msb_rhs && msb_result);

    return borrow_out;
};

} // namespace deprecated

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Regular vector integer arithmetic operation vector-vector
/// \details For all i: vd[i] = vs2[i] op vs1[i]
VILL::vpu_return_t int_op_vv(
    uint8_t *vec_reg_mem,                  //!< Vector register file memory space. One dimensional
    const VInstrInfo &v_instr_info,        //!< Struct containing vector instruction information
    const IntInstrInfo &int_instr_info,    //!< Struct containint integer instruction specific information
    uint16_t reg_vd,                       //!< Destination vector D [index]
    uint16_t reg_vs1,                      //!< Source vector R [index]
    uint16_t reg_vs2,                      //!< Source vector L [index]
    deprecated::IntFunctionDeprecated func //!< Integer arithmetic function
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Regular vector integer arithmetic operation vector-immediate
/// \details For all i: vd[i] = vs2[i] op sign_extend(imm5)
VILL::vpu_return_t int_op_vi(
    uint8_t *vec_reg_mem,                  //!< Vector register file memory space. One dimensional
    VInstrInfo const &v_instr_info,        //!< Struct containing vector instruction information
    IntInstrInfo const &int_instr_info,    //!< Struct containint integer instruction specific information
    uint16_t reg_vd,                       //!< Destination vector D [index]
    uint16_t reg_vs2,                      //!< Source vector L [index]
    uint8_t imm5,                          //!< Sign or zero extending 5-bit immediate
    deprecated::IntFunctionDeprecated func //!< Integer arithmetic function
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Regular vector integer arithmetic operation vector-scalar
/// \details For all i: vd[i] = vs2[i] op sign_extend(X[rs1])
VILL::vpu_return_t int_op_vx(
    uint8_t *vec_reg_mem,                  //!< Vector register file memory space. One dimensional
    VInstrInfo const &v_instr_info,        //!< Struct containing vector instruction information
    IntInstrInfo const &int_instr_info,    //!< Struct containint integer instruction specific information
    uint16_t reg_vd,                       //!< Destination vector D [index]
    uint16_t reg_vs2,                      //!< Source vector L [index]
    uint8_t *scalar_reg_mem,               //!< Memory space holding scalar data (min. _xlenb bytes)
    uint8_t scalar_reg_len_bytes,          //!< Length of scalar [bytes]
    deprecated::IntFunctionDeprecated func //!< Integer arithmetic function
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief  Integer operation vector-vector with register destination
/// \details For all i: vd.mask[i] = vs2[i] op vs1[i]
auto int_op_vv_to_register(
    uint8_t *vec_reg_mem,                //!< Vector register file memory space. One dimensional
    VInstrInfo const &v_instr_info,      //!< Struct containing vector instruction information
    IntInstrInfo const &int_instr_info,  //!< Struct containint integer instruction specific information
    uint16_t reg_vd,                     //!< Destination vector D [index]
    uint16_t reg_vs1,                    //!< Source vector R [index]
    uint16_t reg_vs2,                    //!< Source vector L [index]
    deprecated::IntRegisterFunction func //!< Integer comparison function
    ) -> VILL::vpu_return_t;

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Integer operation vector-immediate with register destination
/// \details For all i: vd.mask[i] = vs2[i] op sign_extend(imm5)
auto int_op_vi_to_register(
    uint8_t *vec_reg_mem,                //!< Vector register file memory space. One dimensional
    VInstrInfo const &v_instr_info,      //!< Struct containing vector instruction information
    IntInstrInfo const &int_instr_info,  //!< Struct containint integer instruction specific information
    uint16_t reg_vd,                     //!< Destination vector D [index]
    uint16_t reg_vs2,                    //!< Source vector L [index]
    uint8_t imm5,                        //!< Sign or zero extending 5-bit immediate
    deprecated::IntRegisterFunction func //!< Integer comparison function
    ) -> VILL::vpu_return_t;

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Integer operation vector-scalar with register destination
/// \details For all i: vd.mask[i] = vs2[i] op zero/sign_extend(X[rs1])
VILL::vpu_return_t int_op_vx_to_register(
    uint8_t *vec_reg_mem,                //!< Vector register file memory space. One dimensional
    VInstrInfo const &v_instr_info,      //!< Struct containing vector instruction information
    IntInstrInfo const &int_instr_info,  //!< Struct containint integer instruction specific information
    uint16_t reg_vd,                     //!< Destination vector D [index]
    uint16_t reg_vs2,                    //!< Source vector L [index]
    uint8_t *scalar_reg_mem,             //!< Memory space holding scalar data (min. _xlenb bytes)
    uint8_t scalar_reg_len_bytes,        //!< Length of scalar [bytes]
    deprecated::IntRegisterFunction func //!< Integer comparison function
);

/* 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
VILL::vpu_return_t vadc_vvm(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs1,
                            uint16_t reg_vs2, uint16_t vec_elem_start);

VILL::vpu_return_t vadc_vim(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs2,
                            uint8_t s_imm, uint16_t vec_elem_start);

VILL::vpu_return_t vadc_vxm(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs2,
                            uint8_t *scalar_reg_mem, uint16_t vec_elem_start, uint8_t scalar_reg_len_bytes);

VILL::vpu_return_t vmadc_vv(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs1,
                            uint16_t reg_vs2, uint16_t vec_elem_start, bool mask_f);

VILL::vpu_return_t vmadc_vi(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs2,
                            uint8_t s_imm, uint16_t vec_elem_start, bool mask_f);

VILL::vpu_return_t vmadc_vx(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs2,
                            uint8_t *scalar_reg_mem, uint16_t vec_elem_start, bool mask_f,
                            uint8_t scalar_reg_len_bytes);

VILL::vpu_return_t vsbc_vvm(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs1,
                            uint16_t reg_vs2, uint16_t vec_elem_start);

VILL::vpu_return_t vsbc_vxm(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs2,
                            uint8_t *scalar_reg_mem, uint16_t vec_elem_start, uint8_t scalar_reg_len_bytes);

VILL::vpu_return_t vmsbc_vv(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs1,
                            uint16_t reg_vs2, uint16_t vec_elem_start, bool mask_f);

VILL::vpu_return_t vmsbc_vx(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                            uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t reg_vd, uint16_t reg_vs2,
                            uint8_t *scalar_reg_mem, uint16_t vec_elem_start, bool mask_f,
                            uint8_t scalar_reg_len_bytes);
/* End 11.4 */

/* 11.15. Vector Integer Merge Instructions */
//////////////////////////////////////////////////////////////////////////////////////
/// \brief Merge vector-vector: vd[i] = v0.mask[i] ? vs1[i] : vs2[i]
VILL::vpu_return_t vmerge_vv(uint8_t *vec_reg_mem,       //!< Vector register file memory space. One dimensional
                             uint64_t emul_num,     //!< Register multiplicity numerator
                             uint64_t emul_denom,   //!< Register multiplicity denominator
                             uint16_t sew_bytes,         //!< Element width [bytes]
                             uint16_t vec_len,           //!< Vector length [elements]
                             uint16_t vec_reg_len_bytes, //!< Vector register length [bytes]
                             uint16_t reg_vd,            //!< Destination vector A [index]
                             uint16_t reg_vs1,           //!< Source vector vs1 [index]
                             uint16_t reg_vs2,           //!< Source vector vs2 [index]
                             uint16_t vec_elem_start     //!< Starting element [index]
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Merge vector-scalar: vd[i] = v0.mask[i] ? x[rs1] : vs2[i]
VILL::vpu_return_t vmerge_vx(uint8_t *vec_reg_mem,        //!< Vector register file memory space. One dimensional
                             uint64_t emul_num,      //!< Register multiplicity numerator
                             uint64_t emul_denom,    //!< Register multiplicity denominator
                             uint16_t sew_bytes,          //!< Element width [bytes]
                             uint16_t vec_len,            //!< Vector length [elements]
                             uint16_t vec_reg_len_bytes,  //!< Vector register length [bytes]
                             uint16_t reg_vd,             //!< Destination vector A [index]
                             uint16_t reg_vs2,            //!< Source vector vs2 [index]
                             uint8_t *scalar_reg_mem,     //!< Memory space holding scalar data (min. _xlenb bytes)
                             uint16_t vec_elem_start,     //!< Starting element [index]
                             uint8_t scalar_reg_len_bytes //!< Length of scalar [bytes]
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Merge vector-scalar: vd[i] = v0.mask[i] ? imm : vs2[i]
VILL::vpu_return_t vmerge_vi(uint8_t *vec_reg_mem,       //!< Vector register file memory space. One dimensional
                             uint64_t emul_num,     //!< Register multiplicity numerator
                             uint64_t emul_denom,   //!< Register multiplicity denominator
                             uint16_t sew_bytes,         //!< Element width [bytes]
                             uint16_t vec_len,           //!< Vector length [elements]
                             uint16_t vec_reg_len_bytes, //!< Vector register length [bytes]
                             uint16_t reg_vd,            //!< Destination vector D [index]
                             uint16_t reg_vs2,           //!< Source vector vs2 [index]
                             uint8_t s_imm,              //!< Sign extending 5-bit immediate
                             uint16_t vec_elem_start     //!< Starting element [index]
);
/* End 11.15. */

/* 11.16. Vector Integer Move Instructions */
//////////////////////////////////////////////////////////////////////////////////////
/// \brief Move vector vd[i] = vs1[i]
VILL::vpu_return_t mv_vv(uint8_t *vec_reg_mem,                  //!< Vector register file memory space. One dimensional
                         uint64_t const emul_num,          //!< Register multiplicity numerator
                         uint64_t const emul_denom,        //!< Register multiplicity denominator
                         uint16_t const sew_bytes,         //!< Element width [bytes]
                         uint16_t const vec_len,           //!< Vector length [elements]
                         uint16_t const vec_reg_len_bytes, //!< Vector register length [bytes]
                         uint16_t const reg_vd,            //!< Destination vector A [index]
                         uint16_t const src_vec_reg,       //!< Source vector A [index]
                         uint16_t const vec_elem_start     //!< Starting element [index]
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Move vector vd[i] = X[rs1]
VILL::vpu_return_t mv_vx(uint8_t *vec_reg_mem,                  //!< Vector register file memory space. One dimensional
                         uint64_t const emul_num,          //!< Register multiplicity numerator
                         uint64_t const emul_denom,        //!< Register multiplicity denominator
                         uint16_t const sew_bytes,         //!< Element width [bytes]
                         uint16_t const vec_len,           //!< Vector length [elements]
                         uint16_t const vec_reg_len_bytes, //!< Vector register length [bytes]
                         uint16_t const reg_vd,            //!< Destination vector A [index]
                         uint8_t *scalar_reg_mem,       //!< Memory space holding scalar data (min. _xlenb bytes)
                         uint16_t const vec_elem_start, //!< Starting element [index]
                         uint8_t const scalar_reg_len_bytes //!< Length of scalar [bytes]
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Move signed immediate to vector vd[i] = simm
VILL::vpu_return_t mv_vi(uint8_t *vec_reg_mem,                  //!< Vector register file memory space. One dimensional
                         uint64_t const emul_num,          //!< Register multiplicity numerator
                         uint64_t const emul_denom,        //!< Register multiplicity denominator
                         uint16_t const sew_bytes,         //!< Element width [bytes]
                         uint16_t const vec_len,           //!< Vector length [elements]
                         uint16_t const vec_reg_len_bytes, //!< Vector register length [bytes]
                         uint16_t const reg_vd,            //!< Destination vector D [index]
                         uint8_t const s_imm,              //!< Sign extending 5-bit immediate
                         uint16_t const vec_elem_start     //!< Starting element [index]
);
/* End 11.16. */

}; // namespace VARITH_INT

#endif /* __RVVHL_ARITH_INTEGER_H__ */
