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
/// \file lsu.hpp
/// \brief Defines helpers implementing load/stores after
/// https://github.com/riscv/riscv-v-spec/blob/0.9/v-spec.adoc#vector-loads-and-stores
/// \date 06/23/2020
//////////////////////////////////////////////////////////////////////////////////////

#ifndef __RVVHL_VLSU_H__
#define __RVVHL_VLSU_H__

#include <cstdint>
#include <functional>
#include "base/base.hpp"

namespace VLSU
{

using MemoryAccessFunction = std::function<void(size_t, uint8_t *, size_t)>;

/* EEW-based */
//////////////////////////////////////////////////////////////////////////////////////
/// @brief Load <vl>-times <eew>-elements through readMem function into vector register file
VILL::vpu_return_t load_eew(
    MemoryAccessFunction func_read_mem, //!< Function for memory read access
    uint8_t *const vec_reg_mem, //!< Vector register file memory space. One dimensional [0..32*VLEN-1] byte array
    [[maybe_unused]] uint64_t const emul_num,   //!< Effective register multiplicity numerator
    [[maybe_unused]] uint64_t const emul_denom, //!< Effective register multiplicity denominator
    uint16_t const eew_bytes,                   //!< Effective element width [bytes]
    uint16_t const vec_len,                     //!< Vector length [elements]
    uint16_t const vec_reg_len_bytes,           //!< Vector register length [bytes]
    uint16_t const vd,                          //!< Destination vector [index]
    uint64_t const src_mem_start,               //!< Source memory start address
    uint16_t const vec_elem_start,              //!< Starting element [index]
    uint8_t const mask_f,                       //!< Vector mask flag. 1: masking 0: no masking
    int16_t const stride_bytes                  //!< Stride length [bytes]
);

template <bool Masked>
VILL::vpu_return_t __attribute__((always_inline)) load_unitstride(
    MemoryAccessFunction func_read_mem, //!< Function for memory read access
    uint8_t *const vector_field, //!< Vector register file memory space. One dimensional [0..32*VLEN-1] byte array
    uint16_t const eew_bytes,    //!< Effective element width [bytes]
    uint16_t const vl,           //!< Vector length [elements]
    uint16_t const vlen_bytes,   //!< Vector register length [bytes]
    uint16_t const vd,           //!< Destination vector [index]
    uint64_t src_mem_offset,     //!< Source memory start address
    uint16_t const vstart        //!< Starting element [index]
)
{
    auto const vd_base = vd * vlen_bytes;
    src_mem_offset += (vstart * eew_bytes);

    // Fast path for common case
    // Unmasked loads with stride = eew can be done in one go
    if constexpr (!Masked)
    {
        // We can do it with one request
        func_read_mem(src_mem_offset, vector_field + vd_base + (vstart * eew_bytes), (vl - vstart) * eew_bytes);
        return VILL::VPU_RETURN::NO_EXCEPT;
    }

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Masked)
        {
            if (!(vector_field[i >> 3] >> (i & 0b111) & 1))
            {
                src_mem_offset += eew_bytes;
                continue;
            }
        }
        func_read_mem(src_mem_offset, vector_field + vd_base + (i * eew_bytes), eew_bytes);
        src_mem_offset += eew_bytes;
    }

    return VILL::VPU_RETURN::NO_EXCEPT;
}

template <bool Masked>
VILL::vpu_return_t __attribute__((always_inline)) load_eew_v2(
    MemoryAccessFunction func_read_mem, //!< Function for memory read access
    uint8_t *const vector_field, //!< Vector register file memory space. One dimensional [0..32*VLEN-1] byte array
    uint16_t const eew_bytes,    //!< Effective element width [bytes]
    uint16_t const vl,           //!< Vector length [elements]
    uint16_t const vlen_bytes,   //!< Vector register length [bytes]
    uint16_t const vd,           //!< Destination vector [index]
    uint64_t src_mem_offset,     //!< Source memory start address
    uint16_t const vstart,       //!< Starting element [index]
    int16_t const stride_bytes   //!< Stride length [bytes]
)
{
    auto const vd_base = vd * vlen_bytes;
    src_mem_offset += (vstart * stride_bytes);

    // Fast path for common case
    // Unmasked loads with stride = eew can be done in one go
    if constexpr (!Masked)
    {
        if (eew_bytes == stride_bytes)
        {
            // We can do it with one request
            func_read_mem(src_mem_offset, vector_field + vd_base + (vstart * eew_bytes), (vl - vstart) * eew_bytes);
            return VILL::VPU_RETURN::NO_EXCEPT;
        }
    }

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Masked)
        {
            if (!(vector_field[i >> 3] >> (i & 0b111) & 1))
            {
                src_mem_offset += stride_bytes;
                continue;
            }
        }
        func_read_mem(src_mem_offset, vector_field + vd_base + (i * eew_bytes), eew_bytes);
        src_mem_offset += stride_bytes;
    }

    return VILL::VPU_RETURN::NO_EXCEPT;
}

//////////////////////////////////////////////////////////////////////////////////////
/// \brief Store <vl>-times <eew>-elements through func_write_mem function from vector register file
VILL::vpu_return_t store_eew(
    MemoryAccessFunction func_write_mem, //!< Function for memory write access
    uint8_t *vec_reg_mem,       //!< Vector register file memory space. One dimensional [0..32*VLEN-1] byte array
    uint64_t emul_num,          //!< Effective register multiplicity numerator
    uint64_t emul_denom,        //!< Effective register multiplicity denominator
    uint16_t eew_bytes,         //!< Effective element width [bytes]
    uint16_t vec_len,           //!< Vector length [elements]
    uint16_t vec_reg_len_bytes, //!< Vector register length [bytes]
    uint16_t src_vec_reg,       //!< Source vector register [index]
    uint64_t dst_mem_start,     //!< Destination memory start address
    uint16_t vec_elem_start,    //!< Starting element [index]
    uint8_t mask_f,             //!< Vector mask flag. 1: masking 0: no masking
    int16_t stride_bytes        //!< Stride length [bytes]
);

template <bool Masked>
VILL::vpu_return_t __attribute__((always_inline)) store_eew_v2(
    MemoryAccessFunction func_write_mem, //!< Function for memory read access
    uint8_t *const vector_field, //!< Vector register file memory space. One dimensional [0..32*VLEN-1] byte array
    uint16_t const eew_bytes,    //!< Effective element width [bytes]
    uint16_t const vl,           //!< Vector length [elements]
    uint16_t const vlen_bytes,   //!< Vector register length [bytes]
    uint16_t const vs3,          //!< Destination vector [index]
    uint64_t dest_mem_offset,    //!< Source memory start address
    uint16_t const vstart,       //!< Starting element [index]
    int16_t const stride_bytes   //!< Stride length [bytes]
)
{
    auto const vs3_base = vs3 * vlen_bytes;
    dest_mem_offset += (vstart * stride_bytes);

    // Fast path for common case
    // Unmasked stores with stride = eew can be done in one go
    if constexpr (!Masked)
    {
        if (eew_bytes == stride_bytes)
        {
            // We can do it with one request
            func_write_mem(dest_mem_offset, vector_field + vs3_base + (vstart * eew_bytes), (vl - vstart) * eew_bytes);
            return VILL::VPU_RETURN::NO_EXCEPT;
        }
    }

    for (size_t i = vstart; i < vl; ++i)
    {
        if constexpr (Masked)
        {
            if (!(vector_field[i >> 3] >> (i & 0b111) & 1))
            {
                dest_mem_offset += stride_bytes;
                continue;
            }
        }
        func_write_mem(dest_mem_offset, vector_field + vs3_base + (i * eew_bytes), eew_bytes);
        dest_mem_offset += stride_bytes;
    }

    return VILL::VPU_RETURN::NO_EXCEPT;
}

//////////////////////////////////////////////////////////////////////////////////////
/// @brief Load <vl>-times <sew>-elements with <eew>-offsets (vs2) through readMem function into vector register
/// file
auto load_indices(
    MemoryAccessFunction func_read_mem, //!< Function for memory read access
    uint8_t *vec_reg_mem,           //!< Vector register file memory space. One dimensional [0..32*VLEN-1] byte array
    VInstrInfo const &v_instr_info, //!< Struct containing vector instruction information
    uint16_t reg_vd,                //!< Destination vector register [index]
    uint16_t reg_vs2,               //!< Index source vector register [index]
    uint64_t src_mem_start,         //!< Source memory start address
    uint16_t eew                    //!< Effective element width [bits]
    ) -> VILL::vpu_return_t;

//////////////////////////////////////////////////////////////////////////////////////
/// @brief Store <vl>-times <sew>-elements with <eew>-offsets (vs2) through func_write_mem function from vector
/// register file
auto store_indices(
    MemoryAccessFunction func_write_mem, //!< Function for memory read access
    uint8_t *vec_reg_mem,           //!< Vector register file memory space. One dimensional [0..32*VLEN-1] byte array
    VInstrInfo const &v_instr_info, //!< Struct containing vector instruction information
    uint16_t reg_vs3,               //!< Source vector register [index]
    uint16_t reg_vs2,               //!< Index source vector registers [index]
    uint64_t dst_mem_start,         //!< Source memory start address
    uint16_t eew,                   //!< Effective element width [bits]
    uint8_t nf                      //!< Number of fields
    ) -> VILL::vpu_return_t;

// auto VLSU::load_indices(VLSU::MemoryAccessFunction func_read_mem, uint8_t *vec_reg_mem, VInstrInfo const
// &v_instr_info,
//                         uint16_t reg_vd, uint16_t reg_vs2, uint64_t src_mem_start, uint16_t eew) ->
//                         VILL::vpu_return_t
// {
//     RVVRegField V_dest(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
//                        SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

//     auto const emul_num = eew * v_instr_info.lmul_num;
//     auto const emul_denom = v_instr_info.sew * v_instr_info.lmul_denom;

//     RVVRegField V_indices(v_instr_info.vector_register_length, v_instr_info.vector_length, eew,
//                           SVMul(emul_num, emul_denom), vec_reg_mem);

//     if (!V_dest.vec_reg_is_aligned(reg_vd))
//     {
//         return (VILL::VPU_RETURN::DST_VEC_ILL);
//     }
//     if (!V_indices.vec_reg_is_aligned(reg_vs2))
//     {
//         return (VILL::VPU_RETURN::SRC2_VEC_ILL);
//     }

//     V_dest.init();
//     V_indices.init();

//     RVVector &vd = V_dest.get_vec(reg_vd);
//     RVVector &vs2 = V_indices.get_vec(reg_vs2);

//     // auto const eew_bytes = eew >> 3;
//     auto const sew_bytes = v_instr_info.sew >> 3;

//     for (size_t i = 0; i < v_instr_info.vector_length; ++i)
//     {
//         // TODO: Overflow possible? Checking?
//         auto index_offset = vs2[i].to_u64();
//         size_t mem_offset = src_mem_start + index_offset;
//         if (i >= v_instr_info.start_element && (!v_instr_info.masked || V_dest.get_mask_reg().get_bit(i)))
//         {
//             func_read_mem(mem_offset, vd[i].mem_, sew_bytes);
//         }
//     }

//     return VILL::VPU_RETURN::NO_EXCEPT;
// }

// VILL::vpu_return_t store_indices_v2(VLSU::MemoryAccessFunction func_write_mem, uint8_t *vec_reg_mem,
//                                     VInstrInfo const &v_instr_info, uint16_t vs3, uint16_t vs2, uint64_t dst_mem_start,
//                                     uint16_t eew, uint8_t nf)
// {
//     auto const sew = v_instr_info.sew;
//     auto const vlen = v_instr_info.vector_register_length;
//     auto const vs2_base = vs2 * (vlen / eew); // Indices
//     auto const vs3_base = vs3 * (vlen / sew); // Source

//     RVVRegField V_src(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
//                       SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

//     auto const emul_num = eew * v_instr_info.lmul_num;
//     auto const emul_denom = v_instr_info.sew * v_instr_info.lmul_denom;

//     RVVRegField V_indices(v_instr_info.vector_register_length, v_instr_info.vector_length, eew,
//                           SVMul(emul_num, emul_denom), vec_reg_mem);

//     V_src.init();
//     V_indices.init();

//     // This is done to match the store order in the testing repository.
//     // However, the speficiation states that stores can occur in any order,
//     // so the tests should reflect that in the future.
//     auto vectors = std::vector<std::reference_wrapper<RVVector>>();
//     for (size_t i = 0; i < nf; i++)
//     {
//         auto reg = vs3 + std::max(i, i * (v_instr_info.lmul_num / v_instr_info.lmul_denom));
//         RVVector &v = V_src.get_vec(reg);
//         vectors.push_back(v);
//     }

//     RVVector &vs2 = V_indices.get_vec(vs2);

//     auto const sew_bytes = v_instr_info.sew >> 3;

//     for (size_t i = 0; i < v_instr_info.vector_length; ++i)
//     {
//         if (i >= v_instr_info.start_element && (!v_instr_info.masked || V_src.get_mask_reg().get_bit(i)))
//         {
//             size_t mem_offset = dst_mem_start + vs2[i].to_u64();
//             for (size_t field = 0; field < nf; field++)
//             {
//                 func_write_mem(mem_offset, vectors[field].get()[i].mem_, sew_bytes);
//                 mem_offset += sew_bytes;
//             }
//         }
//     }

//     return VILL::VPU_RETURN::NO_EXCEPT;
// }

} // namespace VLSU

#endif /* __RVVHL_VLSU_H__ */
