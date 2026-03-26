#include "arithmetic/floatingpoint.hpp"
#include "arithmetic/softfloat-extension.hpp"
#include "base/base.hpp"
#include "vpu/softvector-types.hpp"
#include "base/softvector-platform-types.hpp"

// Private function declarations

void iterate_vector_merge(const SVector &vs2, uint64_t scalar, SVector &vd, const SVRegister &vm, size_t sew,
                          size_t start_index);

void iterate_vector_move(uint64_t scalar, SVector &vd, const SVRegister &vm, size_t sew, size_t start_index);

// Private function definitions

void iterate_vector_merge(const SVector &vs2, uint64_t scalar, SVector &vd, const SVRegister &vm, size_t sew,
                          size_t start_index)
{
    for (size_t i_element = start_index; i_element < vd.length_; ++i_element)
    {
        // 0: use vs2[i], f[rs1] otherwise
        vd[i_element] = vm.get_bit(i_element) ? scalar : vs2[i_element].to_u64();
    }
}

void iterate_vector_move(uint64_t scalar, SVector &vd, const SVRegister &vm, size_t sew, size_t start_index)
{
    for (size_t i_element = start_index; i_element < vd.length_; ++i_element)
    {
        vd[i_element] = scalar;
    }
}

// Public function definitions

VILL::vpu_return_t VARITH_FLOAT::vf_merge(uint8_t *vec_reg_mem, VInstrInfo const &v_instr_info, uint16_t reg_vd,
                                          uint16_t reg_vs2, uint8_t *scalar_reg_mem, uint8_t scalar_reg_len_bytes)
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

    RVVector &vs2 = V.get_vec(reg_vs2);
    RVVector &vd = V.get_vec(reg_vd);

    uint64_t scalar = (scalar_reg_len_bytes > 4) ? *(reinterpret_cast<uint64_t *>(scalar_reg_mem))
                                                 : *(reinterpret_cast<uint32_t *>(scalar_reg_mem));

    if (scalar_reg_len_bytes > (v_instr_info.sew >> 3))
    {
        scalar =
            ((v_instr_info.sew >> 3) == 2) ? check_and_unbox_f16(f64(scalar)).v : check_and_unbox_f32(f64(scalar)).v;
    }

    iterate_vector_merge(vs2, scalar, vd, V.get_mask_reg(), v_instr_info.sew, v_instr_info.start_element);

    return VILL::VPU_RETURN::NO_EXCEPT;
}

VILL::vpu_return_t VARITH_FLOAT::vf_move(uint8_t *vec_reg_mem, VInstrInfo const &v_instr_info, uint16_t reg_vd,
                                         uint8_t *scalar_reg_mem, uint8_t scalar_reg_len_bytes)
{
    RVVRegField V(v_instr_info.vector_register_length, v_instr_info.vector_length, v_instr_info.sew,
                  SVMul(v_instr_info.lmul_num, v_instr_info.lmul_denom), vec_reg_mem);

    if (!V.vec_reg_is_aligned(reg_vd))
    {
        return (VILL::VPU_RETURN::DST_VEC_ILL);
    }

    V.init();

    RVVector &vd = V.get_vec(reg_vd);

    uint64_t scalar = (scalar_reg_len_bytes > 4) ? *(reinterpret_cast<uint64_t *>(scalar_reg_mem))
                                                 : *(reinterpret_cast<uint32_t *>(scalar_reg_mem));

    if (scalar_reg_len_bytes > (v_instr_info.sew >> 3))
    {
        scalar =
            ((v_instr_info.sew >> 3) == 2) ? check_and_unbox_f16(f64(scalar)).v : check_and_unbox_f32(f64(scalar)).v;
    }

    iterate_vector_move(scalar, vd, V.get_mask_reg(), v_instr_info.sew, v_instr_info.start_element);

    return VILL::VPU_RETURN::NO_EXCEPT;
}