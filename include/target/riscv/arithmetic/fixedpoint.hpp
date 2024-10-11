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
/// \file fixedpoint.hpp
/// \brief Defines helpers implementing fixed-point arithmetics after
/// https://github.com/riscv/riscv-v-spec/blob/0.9/v-spec.adoc#vector-arithmetic-instruction-formats
/// \date 09/09/2020
//////////////////////////////////////////////////////////////////////////////////////

#ifndef __RVVHL_ARITH_FIXEDPOINT_H__
#define __RVVHL_ARITH_FIXEDPOINT_H__

#include "stdint.h"
#include "base/base.hpp"

//////////////////////////////////////////////////////////////////////////////////////
/// \brief This space concludes fixed-point arithmetic helpers
namespace VARITH_FIXP
{

/* 12.1. Vector Single-Width Saturating Add and Subtract */
//////////////////////////////////////////////////////////////////////////////////////
/// \brief Saturating addition vector-vector
/// \details For all i: D[i] = L[i] + R[i]
VILL::vpu_return_t vsadd_vv(uint8_t *vec_reg_mem,       //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,          //!< Register multiplicity numerator
                            uint64_t emul_denom,        //!< Register multiplicity denominator
                            uint16_t sew_bytes,         //!< Element width [bytes]
                            uint16_t vec_len,           //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes, //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,       //!< Destination vector D [index]
                            uint16_t src_vec_reg_rhs,   //!< Source vector R [index]
                            uint16_t src_vec_reg_lhs,   //!< Source vector L [index]
                            uint16_t vec_elem_start,    //!< Starting element [index]
                            bool mask_f,                //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed              //!< Signed or unsigned operation
);
//////////////////////////////////////////////////////////////////////////////////////
/// \brief Saturating addition vector-immediate
/// \details For all i: D[i] = L[i] + sign_extend(imm)
VILL::vpu_return_t vsadd_vi(uint8_t *vec_reg_mem,       //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,          //!< Register multiplicity numerator
                            uint64_t emul_denom,        //!< Register multiplicity denominator
                            uint16_t sew_bytes,         //!< Element width [bytes]
                            uint16_t vec_len,           //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes, //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,       //!< Destination vector D [index]
                            uint16_t src_vec_reg_lhs,   //!< Source vector L [index]
                            uint8_t imm,                //!< Sign or zero extending 5-bit immediate
                            uint16_t vec_elem_start,    //!< Starting element [index]
                            bool mask_f,                //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed              //!< Signed or unsigned operation
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Saturating addition vector-scalar
/// \details For all i: D[i] = L[i] + sign_extend(*X)
VILL::vpu_return_t vsadd_vx(uint8_t *vec_reg_mem,        //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,           //!< Register multiplicity numerator
                            uint64_t emul_denom,         //!< Register multiplicity denominator
                            uint16_t sew_bytes,          //!< Element width [bytes]
                            uint16_t vec_len,            //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes,  //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,        //!< Destination vector D [index]
                            uint16_t src_vec_reg_lhs,    //!< Source vector L [index]
                            uint8_t *scalar_reg_mem,     //!< Memory space holding scalar data (min. _xlenb bytes)
                            uint16_t vec_elem_start,     //!< Starting element [index]
                            bool mask_f,                 //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed,              //!< Signed or unsigned operation
                            uint8_t scalar_reg_len_bytes //!< Length of scalar [bytes]
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Saturating subtraction vector-vector
/// \details For all i: D[i] = L[i] + R[i]
VILL::vpu_return_t vssub_vv(uint8_t *vec_reg_mem,       //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,          //!< Register multiplicity numerator
                            uint64_t emul_denom,        //!< Register multiplicity denominator
                            uint16_t sew_bytes,         //!< Element width [bytes]
                            uint16_t vec_len,           //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes, //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,       //!< Destination vector D [index]
                            uint16_t src_vec_reg_rhs,   //!< Source vector R [index]
                            uint16_t src_vec_reg_lhs,   //!< Source vector L [index]
                            uint16_t vec_elem_start,    //!< Starting element [index]
                            bool mask_f,                //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed              //!< Signed or unsigned operation
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Saturating subtraction vector-scalar
/// \details For all i: D[i] = L[i] + sign_extend(*X)
VILL::vpu_return_t vssub_vx(uint8_t *vec_reg_mem,        //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,           //!< Register multiplicity numerator
                            uint64_t emul_denom,         //!< Register multiplicity denominator
                            uint16_t sew_bytes,          //!< Element width [bytes]
                            uint16_t vec_len,            //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes,  //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,        //!< Destination vector D [index]
                            uint16_t src_vec_reg_lhs,    //!< Source vector L [index]
                            uint8_t *scalar_reg_mem,     //!< Memory space holding scalar data (min. _xlenb bytes)
                            uint16_t vec_elem_start,     //!< Starting element [index]
                            bool mask_f,                 //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed,              //!< Signed or unsigned operation
                            uint8_t scalar_reg_len_bytes //!< Length of scalar [bytes]
);
/* End 12.1. */

/* 12.2. Vector Single-Width Averaging Add and Subtract */
//////////////////////////////////////////////////////////////////////////////////////
/// \brief Averaging addition vector-vector
/// \details For all i: D[i] = L[i] + R[i]
VILL::vpu_return_t vaadd_vv(uint8_t *vec_reg_mem,       //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,          //!< Register multiplicity numerator
                            uint64_t emul_denom,        //!< Register multiplicity denominator
                            uint16_t sew_bytes,         //!< Element width [bytes]
                            uint16_t vec_len,           //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes, //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,       //!< Destination vector D [index]
                            uint16_t src_vec_reg_rhs,   //!< Source vector R [index]
                            uint16_t src_vec_reg_lhs,   //!< Source vector L [index]
                            uint16_t vec_elem_start,    //!< Starting element [index]
                            bool mask_f,                //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed,             //!< Signed or unsigned operation
                            uint8_t rounding_mode       //!< Rounding mode
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Averaging addition vector-scalar
/// \details For all i: D[i] = L[i] + sign_extend(*X)
VILL::vpu_return_t vaadd_vx(uint8_t *vec_reg_mem,         //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,            //!< Register multiplicity numerator
                            uint64_t emul_denom,          //!< Register multiplicity denominator
                            uint16_t sew_bytes,           //!< Element width [bytes]
                            uint16_t vec_len,             //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes,   //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,         //!< Destination vector D [index]
                            uint16_t src_vec_reg_lhs,     //!< Source vector L [index]
                            uint8_t *scalar_reg_mem,      //!< Memory space holding scalar data (min. _xlenb bytes)
                            uint16_t vec_elem_start,      //!< Starting element [index]
                            bool mask_f,                  //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed,               //!< Signed or unsigned operation
                            uint8_t scalar_reg_len_bytes, //!< Length of scalar [bytes]
                            uint8_t rounding_mode         //!< Rounding mode
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Averaging subtraction vector-vector
/// \details For all i: D[i] = L[i] + R[i]
VILL::vpu_return_t vasub_vv(uint8_t *vec_reg_mem,       //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,          //!< Register multiplicity numerator
                            uint64_t emul_denom,        //!< Register multiplicity denominator
                            uint16_t sew_bytes,         //!< Element width [bytes]
                            uint16_t vec_len,           //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes, //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,       //!< Destination vector D [index]
                            uint16_t src_vec_reg_rhs,   //!< Source vector R [index]
                            uint16_t src_vec_reg_lhs,   //!< Source vector L [index]
                            uint16_t vec_elem_start,    //!< Starting element [index]
                            bool mask_f,                //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed,             //!< Signed or unsigned operation
                            uint8_t rounding_mode       //!< Rounding mode
);

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Averaging subtraction vector-scalar
/// \details For all i: D[i] = L[i] + sign_extend(*X)
VILL::vpu_return_t vasub_vx(uint8_t *vec_reg_mem,         //!< Vector register file memory space. One dimensional
                            uint64_t emul_num,            //!< Register multiplicity numerator
                            uint64_t emul_denom,          //!< Register multiplicity denominator
                            uint16_t sew_bytes,           //!< Element width [bytes]
                            uint16_t vec_len,             //!< Vector length [elements]
                            uint16_t vec_reg_len_bytes,   //!< Vector register length [bytes]
                            uint16_t dst_vec_reg,         //!< Destination vector D [index]
                            uint16_t src_vec_reg_lhs,     //!< Source vector L [index]
                            uint8_t *scalar_reg_mem,      //!< Memory space holding scalar data (min. _xlenb bytes)
                            uint16_t vec_elem_start,      //!< Starting element [index]
                            bool mask_f,                  //!< Vector mask flag. 1: masking 0: no masking
                            bool is_signed,               //!< Signed or unsigned operation
                            uint8_t scalar_reg_len_bytes, //!< Length of scalar [bytes]
                            uint8_t rounding_mode         //!< Rounding mode
);
/* End 12.2. */
/* End 12. */
/* rvv spec. 13.2 - Vector Single-Width Averaging Add and Substract */
// TODO: ...
/* rvv spec. 13.3 - Vector Single-Width Fractional Multiply with Rounding and Saturation */
// TODO: ...
/* rvv spec. 13.4 - Vector Single-Width Scaling Shift Instructions */
// TODO: ...
/* rvv spec. 13.5 - Vector Narrowing Fixed-Point Clip Instructions */
// TODO: ...

} // namespace VARITH_FIXP
#endif /* __RVVHL_ARITH_FIXEDPOINT_H__ */
