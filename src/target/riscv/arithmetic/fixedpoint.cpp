#include "arithmetic/fixedpoint.hpp"
#include "base/base.hpp"
#include "vpu/softvector-types.hpp"
#include "base/softvector-platform-types.hpp"

VILL::vpu_return_t VARITH_FIXP::vsaddu_vv(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                                      uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t dst_vec_reg,
                                      uint16_t src_vec_reg_rhs, uint16_t src_vec_reg_lhs, uint16_t vec_elem_start,
                                      bool mask_f)
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

        vd.m_sat_addu(vs2, vs1, V.get_mask_reg(), !mask_f, vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_FIXP::vsaddu_vi(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                                      uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t dst_vec_reg,
                                      uint16_t src_vec_reg_lhs, uint8_t s_imm, uint16_t vec_elem_start, bool mask_f)
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

        uint64_t imm = static_cast<uint64_t>(s_imm & 0x10 ? s_imm | ~0x1F : s_imm);
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_sat_addu(vs2, imm, V.get_mask_reg(), !mask_f, vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}

VILL::vpu_return_t VARITH_FIXP::vsaddu_vx(uint8_t *vec_reg_mem, uint64_t emul_num, uint64_t emul_denom, uint16_t sew_bytes,
                                      uint16_t vec_len, uint16_t vec_reg_len_bytes, uint16_t dst_vec_reg,
                                      uint16_t src_vec_reg_lhs, uint8_t *scalar_reg_mem, uint16_t vec_elem_start,
                                      bool mask_f, uint8_t scalar_reg_len_bytes)
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

        uint64_t imm = (scalar_reg_len_bytes > 32) ? *(reinterpret_cast<uint64_t *>(scalar_reg_mem))
                                                  : *(reinterpret_cast<uint32_t *>(scalar_reg_mem));
        RVVector &vs2 = V.get_vec(src_vec_reg_lhs);
        RVVector &vd = V.get_vec(dst_vec_reg);

        vd.m_sat_addu(vs2, imm, V.get_mask_reg(), !mask_f, vec_elem_start);
    }
    return (VILL::VPU_RETURN::NO_EXCEPT);
}