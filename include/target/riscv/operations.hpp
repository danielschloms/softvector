#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "base/base.hpp"
#include "arithmetic/softfloat-extension.hpp"

#ifdef ETISS_SOFTFLOAT
extern "C"
{
#include "softfloat_orig.h"
}
#else
#include "softfloat.h"
#endif

enum class SewType : uint8_t
{
    sew_8 = 8,
    sew_16 = 16,
    sew_32 = 32,
    sew_64 = 64
};

template <unsigned Sew>
concept ValidSew = (Sew == 8) or (Sew == 16) or (Sew == 32) or (Sew == 64);

template <typename T>
concept ValidVectorElementType =
    std::is_same_v<T, uint8_t> or std::is_same_v<T, uint16_t> or std::is_same_v<T, uint32_t> or
    std::is_same_v<T, uint64_t> or std::is_same_v<T, int8_t> or std::is_same_v<T, int16_t> or
    std::is_same_v<T, int32_t> or std::is_same_v<T, int64_t>;

template <typename T>
concept ValidScalarType = std::is_same_v<T, uint32_t> or std::is_same_v<T, uint64_t> or std::is_same_v<T, int32_t> or
                          std::is_same_v<T, int64_t>;

using Bit = bool;

using ValueResultOp = uint64_t (*)(uint64_t const /* lhs */, uint64_t const /* rhs */);
using ValueResultOpMaskData = uint64_t (*)(uint64_t const /* lhs */, uint64_t const /* rhs */,
                                           Bit const /* mask data bit */);
using ValueResultOpSewData = uint64_t (*)(uint64_t const /* lhs */, uint64_t const /* rhs */, SewType const /* sew */);

using UnaryOpSewData = uint64_t (*)(uint64_t const /* val */, SewType const /* sew */);

using BitResultOp = Bit (*)(uint64_t const /* lhs */, uint64_t const /* rhs */);
using BitResultOpSewData = Bit (*)(uint64_t const /* lhs */, uint64_t const /* rhs */, SewType const);

using AccumulatorOp = uint64_t (*)(uint64_t const /* lhs */, uint64_t const /* rhs */,
                                   uint64_t const /* accumulator */);
using AccumulatorOpSewData = uint64_t (*)(uint64_t const /* lhs */, uint64_t const /* rhs */,
                                          uint64_t const /* accumulator */, SewType const /* sew */);

using MaskOp = Bit (*)(Bit const /* lhs */, Bit const /* rhs */);

using SaturatingFpOp = uint64_t (*)(uint64_t const /* lhs */, uint64_t const /* rhs */, SewType const /* sew */,
                                    bool & /* overflow */);
using AveragingFpOp = uint64_t (*)(uint64_t const /* lhs */, uint64_t const /* rhs */, SewType const /* sew */,
                                   uint8_t const /* rounding_mode */);

using CarryBorrowOp = Bit (*)(SewType const /* sew */, uint64_t const /* lhs */, uint64_t const /* rhs */);
using CarryBorrowOpMaskData = Bit (*)(SewType const /* sew */, uint64_t const /* lhs */, uint64_t const /* rhs */,
                                      Bit const /* mask data bit */);

template <typename F>
concept ValidOperation =
    std::is_same_v<F, ValueResultOp> or std::is_same_v<F, BitResultOp> or std::is_same_v<F, ValueResultOpSewData> or
    std::is_same_v<F, ValueResultOpMaskData> or std::is_same_v<F, AccumulatorOp> or
    std::is_same_v<F, AccumulatorOpSewData> or std::is_same_v<F, MaskOp> or std::is_same_v<F, UnaryOpSewData> or
    std::is_same_v<F, BitResultOpSewData>;

inline constexpr uint64_t add_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs + rhs;
}

inline constexpr uint64_t sub_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs - rhs;
}

inline constexpr uint64_t rsub_int(uint64_t const lhs, uint64_t const rhs)
{
    return rhs - lhs;
}

inline constexpr uint64_t adc_int(uint64_t const lhs, uint64_t const rhs, Bit carry)
{
    return lhs + rhs + carry;
}

inline constexpr Bit madc_carry_in(SewType const sew, uint64_t const lhs, uint64_t const rhs, Bit const carry)
{
    auto const result = lhs + rhs + carry;

    auto const msb_lhs = msb_is_set(lhs, std::to_underlying(sew));
    auto const msb_rhs = msb_is_set(rhs, std::to_underlying(sew));
    auto const msb_result = msb_is_set(result, std::to_underlying(sew));

    // Carry out if:
    // - MSB of both operands are set
    // - MSB of one operand is set, but result MSB is not set
    return (msb_lhs && msb_rhs) || (msb_lhs && !msb_rhs && !msb_result) || (!msb_lhs && msb_rhs && !msb_result);
}

inline constexpr Bit madc_no_carry_in(SewType const sew, uint64_t const lhs, uint64_t const rhs)
{
    auto const result = lhs + rhs;

    auto const msb_lhs = msb_is_set(lhs, std::to_underlying(sew));
    auto const msb_rhs = msb_is_set(rhs, std::to_underlying(sew));
    auto const msb_result = msb_is_set(result, std::to_underlying(sew));

    // Carry out if:
    // - MSB of both operands are set
    // - MSB of one operand is set, but result MSB is not set
    return (msb_lhs && msb_rhs) || (msb_lhs && !msb_rhs && !msb_result) || (!msb_lhs && msb_rhs && !msb_result);
}

inline constexpr uint64_t sbc_int(uint64_t const lhs, uint64_t const rhs, Bit borrow)
{
    return lhs - rhs - borrow;
}

inline constexpr Bit msbc_borrow_in(SewType const sew, uint64_t const lhs, uint64_t const rhs, Bit const borrow)
{
    auto const result = lhs - rhs - borrow;

    auto const msb_lhs = msb_is_set(lhs, std::to_underlying(sew));
    auto const msb_rhs = msb_is_set(rhs, std::to_underlying(sew));
    auto const msb_result = msb_is_set(result, std::to_underlying(sew));

    // Borrow out if:
    // - MSB of rhs is set and MSB of lhs is not set
    // - MSB of result is set and MSB of lhs = MSB of rhs
    return (!msb_lhs && msb_rhs) || (msb_lhs && msb_rhs && msb_result) || (!msb_lhs && !msb_rhs && msb_result);
}

inline constexpr Bit msbc_no_borrow_in(SewType const sew, uint64_t const lhs, uint64_t const rhs)
{
    auto const result = lhs - rhs;

    auto const msb_lhs = msb_is_set(lhs, std::to_underlying(sew));
    auto const msb_rhs = msb_is_set(rhs, std::to_underlying(sew));
    auto const msb_result = msb_is_set(result, std::to_underlying(sew));

    // Borrow out if:
    // - MSB of rhs is set and MSB of lhs is not set
    // - MSB of result is set and MSB of lhs = MSB of rhs
    return (!msb_lhs && msb_rhs) || (msb_lhs && msb_rhs && msb_result) || (!msb_lhs && !msb_rhs && msb_result);
}

inline constexpr uint64_t and_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs & rhs;
}

inline constexpr uint64_t or_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs | rhs;
}

inline constexpr uint64_t xor_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs ^ rhs;
}

inline constexpr uint64_t sll_int(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    return lhs << (rhs & (std::to_underlying(sew) - 1));
}

inline constexpr uint64_t srl_int(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    return lhs >> (rhs & (std::to_underlying(sew) - 1));
}

inline constexpr uint64_t sra_int(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    return static_cast<int64_t>(lhs) >> (rhs & (std::to_underlying(sew) - 1));
}

inline constexpr Bit eq_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs == rhs;
};

inline constexpr Bit ne_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs != rhs;
};

inline constexpr Bit ltu_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs < rhs;
};

inline constexpr Bit lt_int(uint64_t const lhs, uint64_t const rhs)
{
    return static_cast<int64_t>(lhs) < static_cast<int64_t>(rhs);
};

inline constexpr Bit leu_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs <= rhs;
};

inline constexpr Bit le_int(uint64_t const lhs, uint64_t const rhs)
{
    return static_cast<int64_t>(lhs) <= static_cast<int64_t>(rhs);
};

inline constexpr Bit gtu_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs > rhs;
};

inline constexpr Bit gt_int(uint64_t const lhs, uint64_t const rhs)
{
    return static_cast<int64_t>(lhs) > static_cast<int64_t>(rhs);
};

// 11.9. Vector Integer Min/Max Instructions

inline constexpr uint64_t minu_int(uint64_t const lhs, uint64_t const rhs)
{
    return std::min(lhs, rhs);
}

inline constexpr uint64_t min_int(uint64_t const lhs, uint64_t const rhs)
{
    return std::min(static_cast<int64_t>(lhs), static_cast<int64_t>(rhs));
}

inline constexpr uint64_t maxu_int(uint64_t const lhs, uint64_t const rhs)
{
    return std::max(lhs, rhs);
}

inline constexpr uint64_t max_int(uint64_t const lhs, uint64_t const rhs)
{
    return std::max(static_cast<int64_t>(lhs), static_cast<int64_t>(rhs));
}

// 11.10. Vector Single-Width Integer Multiply Instructions &
// 11.12. Vector Widening Integer Multiply Instructions
inline constexpr uint64_t mul_int(uint64_t const lhs, uint64_t const rhs)
{
    return static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
};

inline constexpr uint64_t mulu_int(uint64_t const lhs, uint64_t const rhs)
{
    return lhs * rhs;
};

inline constexpr uint64_t mulsu_int(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    auto const sew_mask = (1_u64 << std::to_underlying(sew)) - 1;
    // rhs is sign extended when read from the vector register, so mask it off again
    return (static_cast<int64_t>(lhs) * (rhs & sew_mask));
};

inline constexpr uint64_t mulh_int(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    return (static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs)) >> std::to_underlying(sew);
};

inline constexpr uint64_t mulhu_int(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    return (lhs * rhs) >> std::to_underlying(sew);
};

inline constexpr uint64_t mulhsu_int(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    auto const sew_mask = (1_u64 << std::to_underlying(sew)) - 1;
    // rhs is sign extended when read from the vector register, so mask it off again
    return (static_cast<int64_t>(lhs) * (rhs & sew_mask)) >> std::to_underlying(sew);
};

/* 11.11. Vector Integer Divide Instructions */
inline constexpr uint64_t divu_int(uint64_t const lhs, uint64_t const rhs)
{
    // Divide by zero case
    if (rhs == 0)
    {
        return -1;
    }
    return lhs / rhs;
};

inline constexpr uint64_t div_int(uint64_t const lhs, uint64_t const rhs)
{
    // Divide by zero case
    if (rhs == 0)
    {
        return -1;
    }

    // Overflow case
    if ((static_cast<int64_t>(lhs) == std::numeric_limits<int64_t>::min()) && (static_cast<int64_t>(rhs) == -1))
    {
        return lhs;
    }
    return static_cast<int64_t>(lhs) / static_cast<int64_t>(rhs);
};

inline constexpr uint64_t remu_int(uint64_t const lhs, uint64_t const rhs)
{
    // Divide by zero case
    if (rhs == 0)
    {
        return lhs;
    }
    return lhs % rhs;
};

inline constexpr uint64_t rem_int(uint64_t const lhs, uint64_t const rhs)
{
    // Divide by zero case
    if (rhs == 0)
    {
        return lhs;
    }
    // Overflow case
    if ((static_cast<int64_t>(lhs) == std::numeric_limits<int64_t>::min()) && (static_cast<int64_t>(rhs) == -1))
    {
        return 0;
    }
    return static_cast<int64_t>(lhs) % static_cast<int64_t>(rhs);
};

inline constexpr uint64_t macc(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator)
{
    return accumulator + (lhs * rhs);
}

inline constexpr uint64_t maccsu(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator, SewType const sew)
{
    auto const sew_mask = (1_u64 << std::to_underlying(sew)) - 1;
    // rhs is sign extended when read from the vector register, so mask it off again
    return accumulator + (lhs * (rhs & sew_mask));
}

inline constexpr uint64_t maccus(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator, SewType const sew)
{
    auto const sew_mask = (1_u64 << std::to_underlying(sew)) - 1;
    // rhs is sign extended when read from the vector register, so mask it off again
    return accumulator + ((lhs & sew_mask) * rhs);
}

inline constexpr uint64_t nmsac(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator)
{
    return accumulator - (lhs * rhs);
}

inline constexpr uint64_t madd(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator)
{
    return rhs + (lhs * accumulator);
}

inline constexpr uint64_t nmsub(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator)
{
    return rhs - (lhs * accumulator);
}

// 12. Vector Fixed-Point Arithmetic Instructions

// Helpers

/**
 * @brief Saturates a value to the nearest sew bit boundary
 */
inline constexpr int64_t saturate_boundary_signed(int64_t value, uint8_t sew)
{
    // E.g. 8 bit:
    // Upper bound = 0111 1111
    // Lower bound = 1000 0000
    // Lower bound is extended to 64 bit, so just AND -1 with the inverted upper bound
    int64_t upper_bound = get_n_bit_mask(sew - 1);
    int64_t lower_bound = -1 & (~upper_bound);
    return std::clamp(value, lower_bound, upper_bound);
}

/**
 * @brief Saturates a value to the unsigned sew bit boundary
 */
inline constexpr uint64_t saturate_boundary_unsigned(uint64_t value, uint8_t sew)
{
    auto upper_bound = get_n_bit_mask(sew);
    return (value <= upper_bound) ? value : upper_bound;
}

inline constexpr uint64_t sadd(uint64_t const lhs, uint64_t const rhs, SewType const sew, bool &overflow)
{

    auto const res = static_cast<int64_t>(lhs) + static_cast<int64_t>(rhs);
    auto msb_lhs = msb_is_set(lhs, std::to_underlying(sew));
    auto msb_rhs = msb_is_set(rhs, std::to_underlying(sew));
    auto msb_res = msb_is_set(res, std::to_underlying(sew));

    if (msb_lhs && msb_rhs && !msb_res)
    {
        // Negative overflow
        overflow = true;
        return get_min_signed(std::to_underlying(sew));
    }

    if (!msb_lhs && !msb_rhs && msb_res)
    {
        // Positive overflow
        overflow = true;
        return get_n_bit_mask(std::to_underlying(sew) - 1);
    }

    overflow = false;
    return res;
};

// 15. Vector Mask Instructions
// 15.1. Vector Mask-Register Logical Instructions
inline constexpr Bit and_mask(Bit lhs, Bit rhs)
{
    return lhs && rhs;
}

inline constexpr Bit nand_mask(Bit lhs, Bit rhs)
{
    return !(lhs && rhs);
}

inline constexpr Bit andn_mask(Bit lhs, Bit rhs)
{
    return lhs && (!rhs);
}

inline constexpr Bit xor_mask(Bit lhs, Bit rhs)
{
    return lhs != rhs;
}

inline constexpr Bit or_mask(Bit lhs, Bit rhs)
{
    return lhs || rhs;
}

inline constexpr Bit nor_mask(Bit lhs, Bit rhs)
{
    return !(lhs || rhs);
}

inline constexpr Bit orn_mask(Bit lhs, Bit rhs)
{
    return lhs || (!rhs);
}

inline constexpr Bit xnor_mask(Bit lhs, Bit rhs)
{
    return lhs == rhs;
}

inline constexpr uint64_t add_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_add(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_add(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_add(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t sub_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_sub(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_sub(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_sub(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t rsub_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_sub(f16(rhs), f16(lhs)).v;
        break;
    case 32:
        return f32_sub(f32(rhs), f32(lhs)).v;
        break;
    case 64:
        return f64_sub(f64(rhs), f64(lhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t wadd_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_add(f16_to_f32(f16(lhs)), f16_to_f32(f16(rhs))).v;
        break;
    case 32:
        return f64_add(f32_to_f64(f32(lhs)), f32_to_f64(f32(rhs))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t wsub_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_sub(f16_to_f32(f16(lhs)), f16_to_f32(f16(rhs))).v;
        break;
    case 32:
        return f64_sub(f32_to_f64(f32(lhs)), f32_to_f64(f32(rhs))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t wadd_w_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_add(f32(lhs), f16_to_f32(f16(rhs))).v;
        break;
    case 32:
        return f64_add(f64(lhs), f32_to_f64(f32(rhs))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t wsub_w_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_sub(f32(lhs), f16_to_f32(f16(rhs))).v;
        break;
    case 32:
        return f64_sub(f64(lhs), f32_to_f64(f32(rhs))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

/* 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions */
inline constexpr uint64_t mul_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mul(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_mul(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_mul(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t div_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_div(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_div(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_div(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t rdiv_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_div(f16(rhs), f16(lhs)).v;
        break;
    case 32:
        return f32_div(f32(rhs), f32(lhs)).v;
        break;
    case 64:
        return f64_div(f64(rhs), f64(lhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

/* 13.5. Vector Widening Floating-Point Multiply */
inline constexpr uint64_t wmul_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_mul(f16_to_f32(f16(lhs)), f16_to_f32(f16(rhs))).v;
        break;
    case 32:
        return f64_mul(f32_to_f64(f32(lhs)), f32_to_f64(f32(rhs))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

/* 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions */
inline constexpr uint64_t macc_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                     SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mulAdd(f16(lhs), f16(rhs), f16(accumulator)).v;
        break;
    case 32:
        return f32_mulAdd(f32(lhs), f32(rhs), f32(accumulator)).v;
        break;
    case 64:
        return f64_mulAdd(f64(lhs), f64(rhs), f64(accumulator)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t nmacc_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                      SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mulAdd(f16(lhs), f16_neg(f16(rhs)), f16_neg(f16(accumulator))).v;
        break;
    case 32:
        return f32_mulAdd(f32(lhs), f32_neg(f32(rhs)), f32_neg(f32(accumulator))).v;
        break;
    case 64:
        return f64_mulAdd(f64(lhs), f64_neg(f64(rhs)), f64_neg(f64(accumulator))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t msac_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                     SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mulAdd(f16(lhs), f16(rhs), f16_neg(f16(accumulator))).v;
        break;
    case 32:
        return f32_mulAdd(f32(lhs), f32(rhs), f32_neg(f32(accumulator))).v;
        break;
    case 64:
        return f64_mulAdd(f64(lhs), f64(rhs), f64_neg(f64(accumulator))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t nmsac_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                      SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mulAdd(f16(lhs), f16_neg(f16(rhs)), f16(accumulator)).v;
        break;
    case 32:
        return f32_mulAdd(f32(lhs), f32_neg(f32(rhs)), f32(accumulator)).v;
        break;
    case 64:
        return f64_mulAdd(f64(lhs), f64_neg(f64(rhs)), f64(accumulator)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t madd_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                     SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mulAdd(f16(accumulator), f16(rhs), f16(lhs)).v;
        break;
    case 32:
        return f32_mulAdd(f32(accumulator), f32(rhs), f32(lhs)).v;
        break;
    case 64:
        return f64_mulAdd(f64(accumulator), f64(rhs), f64(lhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t nmadd_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                      SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mulAdd(f16(accumulator), f16_neg(f16(rhs)), f16_neg(f16(lhs))).v;
        break;
    case 32:
        return f32_mulAdd(f32(accumulator), f32_neg(f32(rhs)), f32_neg(f32(lhs))).v;
        break;
    case 64:
        return f64_mulAdd(f64(accumulator), f64_neg(f64(rhs)), f64_neg(f64(lhs))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t msub_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                     SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mulAdd(f16(accumulator), f16(rhs), f16_neg(f16(lhs))).v;
        break;
    case 32:
        return f32_mulAdd(f32(accumulator), f32(rhs), f32_neg(f32(lhs))).v;
        break;
    case 64:
        return f64_mulAdd(f64(accumulator), f64(rhs), f64_neg(f64(lhs))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t nmsub_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                      SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_mulAdd(f16(accumulator), f16_neg(f16(rhs)), f16(lhs)).v;
        break;
    case 32:
        return f32_mulAdd(f32(accumulator), f32_neg(f32(rhs)), f32(lhs)).v;
        break;
    case 64:
        return f64_mulAdd(f64(accumulator), f64_neg(f64(rhs)), f64(lhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

/* 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions */
inline constexpr uint64_t wmacc_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                      SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_mulAdd(f16_to_f32(f16(lhs)), f16_to_f32(f16(rhs)), f32(accumulator)).v;
        break;
    case 32:
        return f64_mulAdd(f32_to_f64(f32(lhs)), f32_to_f64(f32(rhs)), f64(accumulator)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t wnmacc_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                       SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_mulAdd(f16_to_f32(f16(lhs)), f16_to_f32(f16_neg(f16(rhs))), f32_neg(f32(accumulator))).v;
        break;
    case 32:
        return f64_mulAdd(f32_to_f64(f32(lhs)), f32_to_f64(f32_neg(f32(rhs))), f64_neg(f64(accumulator))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t wmsac_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                      SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_mulAdd(f16_to_f32(f16(lhs)), f16_to_f32(f16(rhs)), f32_neg(f32(accumulator))).v;
        break;
    case 32:
        return f64_mulAdd(f32_to_f64(f32(lhs)), f32_to_f64(f32(rhs)), f64_neg(f64(accumulator))).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t wnmsac_float(uint64_t const lhs, uint64_t const rhs, uint64_t const accumulator,
                                       SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f32_mulAdd(f16_to_f32(f16(lhs)), f16_to_f32(f16_neg(f16(rhs))), f32(accumulator)).v;
        break;
    case 32:
        return f64_mulAdd(f32_to_f64(f32(lhs)), f32_to_f64(f32_neg(f32(rhs))), f64(accumulator)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

/* 13.8. Vector Floating-Point Square-Root Instruction */
inline constexpr uint64_t sqrt_float(uint64_t const lhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_sqrt(f16(lhs)).v;
        break;
    case 32:
        return f32_sqrt(f32(lhs)).v;
        break;
    case 64:
        return f64_sqrt(f64(lhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

/* 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction */
inline constexpr uint64_t rsqrt7_float(uint64_t const lhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_rsqrte7(f16(lhs)).v;
        break;
    case 32:
        return f32_rsqrte7(f32(lhs)).v;
        break;
    case 64:
        return f64_rsqrte7(f64(lhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

/* 13.10. Vector Floating-Point Reciprocal Estimate Instruction */
inline constexpr uint64_t rec7_float(uint64_t const lhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_recip7(f16(lhs)).v;
        break;
    case 32:
        return f32_recip7(f32(lhs)).v;
        break;
    case 64:
        return f64_recip7(f64(lhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

/* 13.11. Vector Floating-Point MIN/MAX Instructions */
inline constexpr uint64_t min_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_min(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_min(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_min(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t max_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_max(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_max(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_max(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};
/* End 13.11. */

/* 13.12. Vector Floating-Point Sign-Injection Instructions */
inline constexpr uint64_t sgnj_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_sgnj(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_sgnj(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_sgnj(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t sgnjn_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_sgnjn(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_sgnjn(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_sgnjn(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

inline constexpr uint64_t sgnjx_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_sgnjx(f16(lhs), f16(rhs)).v;
        break;
    case 32:
        return f32_sgnjx(f32(lhs), f32(rhs)).v;
        break;
    case 64:
        return f64_sgnjx(f64(lhs), f64(rhs)).v;
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};
/* End 13.12. */

/* 13.13. Vector Floating-Point Compare Instructions */
inline constexpr Bit eq_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_eq(f16(lhs), f16(rhs));
    case 32:
        return f32_eq(f32(lhs), f32(rhs));
    case 64:
        return f64_eq(f64(lhs), f64(rhs));
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
};

inline constexpr Bit ne_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return !f16_eq(f16(lhs), f16(rhs));
    case 32:
        return !f32_eq(f32(lhs), f32(rhs));
    case 64:
        return !f64_eq(f64(lhs), f64(rhs));
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
};

inline constexpr Bit lt_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_lt(f16(lhs), f16(rhs));
    case 32:
        return f32_lt(f32(lhs), f32(rhs));
    case 64:
        return f64_lt(f64(lhs), f64(rhs));
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
};

inline constexpr Bit le_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_le(f16(lhs), f16(rhs));
    case 32:
        return f32_le(f32(lhs), f32(rhs));
    case 64:
        return f64_le(f64(lhs), f64(rhs));
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
};

inline constexpr Bit gt_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_lt(f16(rhs), f16(lhs));
    case 32:
        return f32_lt(f32(rhs), f32(lhs));
    case 64:
        return f64_lt(f64(rhs), f64(lhs));
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
};

inline constexpr Bit ge_float(uint64_t const lhs, uint64_t const rhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return f16_le(f16(rhs), f16(lhs));
    case 32:
        return f32_le(f32(rhs), f32(lhs));
    case 64:
        return f64_le(f64(rhs), f64(lhs));
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
};
/* End 13.13. */

/* 13.14. Vector Floating-Point Classify Instruction */
inline constexpr uint64_t classify_float(uint64_t const lhs, SewType const sew)
{
    switch (std::to_underlying(sew))
    {
    case 16:
        return 0 | f16_classify(f16(lhs));
        break;
    case 32:
        return 0 | f32_classify(f32(lhs));
        break;
    case 64:
        return 0 | f64_classify(f64(lhs));
        break;
    default:
        // Illegal
        exit(EXIT_FAILURE);
    }
    return 0;
};

// /* 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions */
// // Float to (un)signed int, same width
// inline FloatConversionFunction convert_x_f = [](uint64_t opL, SVElement &vd, size_t sew, bool signed_x, bool rtz,
//                                                 bool rod = false) -> void {
//     auto rounding_mode = rtz ? softfloat_round_minMag : softfloat_roundingMode;
//     switch (sew)
//     {
//     case 16:
//         vd = signed_x ? f16_to_i16(f16(opL), rounding_mode, true) : f16_to_ui16(f16(opL), rounding_mode, true);
//         return;
//     case 32:
//         vd = signed_x ? f32_to_i32(f32(opL), rounding_mode, true) : f32_to_ui32(f32(opL), rounding_mode, true);
//         return;
//     case 64:
//         vd = signed_x ? f64_to_i64(f64(opL), rounding_mode, true) : f64_to_ui64(f64(opL), rounding_mode, true);
//         return;
//     default:
//         break;
//     }
// };

// // (Un)signed int to float, same width
// inline FloatConversionFunction convert_f_x = [](uint64_t opL, SVElement &vd, size_t sew, bool signed_x,
//                                                 bool rtz = false, bool rod = false) -> void {
//     switch (sew)
//     {
//     case 16:
//         vd = signed_x ? i32_to_f16(opL).v : ui32_to_f16(opL).v;
//         return;
//     case 32:
//         vd = signed_x ? i32_to_f32(opL).v : ui32_to_f32(opL).v;
//         return;
//     case 64:
//         vd = signed_x ? i64_to_f64(opL).v : ui64_to_f64(opL).v;
//         return;
//     default:
//         break;
//     }
// };
// /* End 13.17. */

// /* 13.18. Widening Floating-Point/Integer Type-Convert Instructions */
// // Float to (un)signed int, widening
// inline FloatConversionFunction convert_widening_x_f = [](uint64_t opL, SVElement &vd, size_t sew, bool signed_x,
//                                                          bool rtz, bool rod = false) -> void {
//     auto rounding_mode = rtz ? softfloat_round_minMag : softfloat_roundingMode;
//     switch (sew)
//     {
//     case 16:
//         vd = signed_x ? f16_to_i32(f16(opL), rounding_mode, true) : f16_to_ui32(f16(opL), rounding_mode, true);
//         return;
//     case 32:
//         vd = signed_x ? f32_to_i64(f32(opL), rounding_mode, true) : f32_to_ui64(f32(opL), rounding_mode, true);
//         return;
//     default:
//         break;
//     }
// };

// // (Un)signed int to float, widening
// inline FloatConversionFunction convert_widening_f_x = [](uint64_t opL, SVElement &vd, size_t sew, bool signed_x,
//                                                          bool rtz = false, bool rod = false) -> void {
//     switch (sew)
//     {
//     case 16:
//         vd = signed_x ? i32_to_f32(opL).v : ui32_to_f32(opL).v;
//         return;
//     case 32:
//         vd = signed_x ? i32_to_f64(opL).v : ui32_to_f64(opL).v;
//         return;
//     default:
//         break;
//     }
// };

// // Float to float, widening                         pVSTART, pVm
// inline FloatConversionFunction convert_widening_f_f = [](uint64_t opL, SVElement &vd, size_t sew, bool signed_x,
//                                                          bool rtz = false, bool rod = false) -> void {
//     switch (sew)
//     {
//     case 16:
//         vd = f16_to_f32(f16(opL)).v;
//         return;
//     case 32:
//         vd = f32_to_f64(f32(opL)).v;
//         return;
//     default:
//         break;
//     }
// };
// /* End 13.18. */

// /* 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions */
// // Float to (un)signed int, narrowing
// inline FloatConversionFunction convert_narrowing_x_f = [](uint64_t opL, SVElement &vd, size_t sew, bool signed_x,
//                                                           bool rtz, bool rod = false) -> void {
//     auto rounding_mode = rtz ? softfloat_round_minMag : softfloat_roundingMode;
//     switch (sew)
//     {
//     case 16:
//         vd = signed_x ? f32_to_i16(f32(opL), rounding_mode, true) : f32_to_ui16(f32(opL), rounding_mode, true);
//         return;
//     case 32:
//         vd = signed_x ? f64_to_i32(f64(opL), rounding_mode, true) : f64_to_ui32(f64(opL), rounding_mode, true);
//         return;
//     default:
//         break;
//     }
// };

// // (Un)signed int to float, narrowing
// inline FloatConversionFunction convert_narrowing_f_x = [](uint64_t opL, SVElement &vd, size_t sew, bool signed_x,
//                                                           bool rtz = false, bool rod = false) -> void {
//     switch (sew)
//     {
//     case 16:
//         vd = signed_x ? i32_to_f16(opL).v : ui32_to_f16(opL).v;
//         return;
//     case 32:
//         vd = signed_x ? i64_to_f32(opL).v : ui64_to_f32(opL).v;
//         return;
//     default:
//         break;
//     }
// };

// // Float to float, narrowing
// inline FloatConversionFunction convert_narrowing_f_f = [](uint64_t opL, SVElement &vd, size_t sew, bool signed_x,
//                                                           bool rtz, bool rod) -> void {
//     if (rod)
//     {
//         softfloat_roundingMode = softfloat_round_odd;
//     }
//     switch (sew)
//     {
//     case 16:
//         vd = f32_to_f16(f32(opL)).v;
//         return;
//     case 32:
//         vd = f64_to_f32(f64(opL)).v;
//         return;
//     default:
//         break;
//     }
// };
// /* End 13.19. */