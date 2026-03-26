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
/// \file floatingpoint.hpp
/// \brief Defines helpers implementing floating-point arithmetics after
/// https://github.com/riscv/riscv-v-spec/blob/0.9/v-spec.adoc#vector-arithmetic-instruction-formats
/// \date 09/09/2020
//////////////////////////////////////////////////////////////////////////////////////

#ifndef __RVVHL_ARITH_FLOATINGPOINT_H__
#define __RVVHL_ARITH_FLOATINGPOINT_H__

#include <functional>

#include "stdint.h"
#include "base/base.hpp"
#include "vpu/softvector-types.hpp"

//////////////////////////////////////////////////////////////////////////////////////
/// \brief This space concludes floating-point arithmetic helpers
namespace VARITH_FLOAT
{

using FloatFunction = std::function<bool(uint64_t, uint64_t, SVElement &, size_t)>;

VILL::vpu_return_t vf_merge(uint8_t *vec_reg_mem, const VInstrInfo &v_instr_info, uint16_t reg_vd, uint16_t reg_vs2,
                            uint8_t *scalar_reg_mem, uint8_t scalar_reg_len_bytes);

VILL::vpu_return_t vf_move(uint8_t *vec_reg_mem, const VInstrInfo &v_instr_info, uint16_t reg_vd,
                           uint8_t *scalar_reg_mem, uint8_t scalar_reg_len_bytes);

}; // namespace VARITH_FLOAT
#endif /* __RVVHL_ARITH_FLOATINGPOINT_H__ */
