#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "base/base.hpp"

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

using BinaryValueResultOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */);
using BinaryValueResultOpMaskData = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, Bit /* mask data bit */);
using BitResultOp = Bit (*)(uint64_t /* lhs */, uint64_t /* rhs */);
using ShiftOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, SewType);
using AccumulatorOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, uint64_t /* accumulator */);
using MixedSignAccumulatorOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, uint64_t /* accumulator */,
                                            SewType /* sew */);
using SaturatingFpOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, SewType /* sew */, bool & /* overflow */);
using AveragingFpOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, SewType /* sew */, uint8_t /* rounding_mode */);

using CarryBorrowOp = Bit (*)(SewType /* sew */, uint64_t /* lhs */, uint64_t /* rhs */);
using CarryBorrowOpMaskData = Bit (*)(SewType /* sew */, uint64_t /* lhs */, uint64_t /* rhs */,
                                      Bit /* mask data bit */);

template <typename F>
concept ValidOperation = std::is_same_v<F, BinaryValueResultOp> or std::is_same_v<F, BitResultOp> or
                         std::is_same_v<F, ShiftOp> or std::is_same_v<F, BinaryValueResultOpMaskData> or
                         std::is_same_v<F, AccumulatorOp> or std::is_same_v<F, MixedSignAccumulatorOp>;

inline constexpr uint64_t add_int(uint64_t lhs, uint64_t rhs)
{
    return lhs + rhs;
}

inline constexpr uint64_t sub_int(uint64_t lhs, uint64_t rhs)
{
    return lhs - rhs;
}

inline constexpr uint64_t rsub_int(uint64_t lhs, uint64_t rhs)
{
    return rhs - lhs;
}

inline constexpr uint64_t adc_int(uint64_t lhs, uint64_t rhs, Bit carry)
{
    return lhs + rhs + carry;
}

inline constexpr Bit madc_carry_in(SewType sew, uint64_t lhs, uint64_t rhs, Bit carry)
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

inline constexpr Bit madc_no_carry_in(SewType sew, uint64_t lhs, uint64_t rhs)
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

inline constexpr uint64_t sbc_int(uint64_t lhs, uint64_t rhs, Bit borrow)
{
    return lhs - rhs - borrow;
}

inline constexpr Bit msbc_borrow_in(SewType sew, uint64_t lhs, uint64_t rhs, Bit borrow)
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

inline constexpr Bit msbc_no_borrow_in(SewType sew, uint64_t lhs, uint64_t rhs)
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

inline constexpr uint64_t and_int(uint64_t lhs, uint64_t rhs)
{
    return lhs & rhs;
}

inline constexpr uint64_t or_int(uint64_t lhs, uint64_t rhs)
{
    return lhs | rhs;
}

inline constexpr uint64_t xor_int(uint64_t lhs, uint64_t rhs)
{
    return lhs ^ rhs;
}

inline constexpr uint64_t sll_int(uint64_t lhs, uint64_t rhs, SewType sew)
{
    return lhs << (rhs & (std::to_underlying(sew) - 1));
}

inline constexpr uint64_t srl_int(uint64_t lhs, uint64_t rhs, SewType sew)
{
    return lhs >> (rhs & (std::to_underlying(sew) - 1));
}

inline constexpr uint64_t sra_int(uint64_t lhs, uint64_t rhs, SewType sew)
{
    return static_cast<int64_t>(lhs) >> (rhs & (std::to_underlying(sew) - 1));
}

inline constexpr Bit eq_int(uint64_t lhs, uint64_t rhs)
{
    return lhs == rhs;
};

inline constexpr Bit ne_int(uint64_t lhs, uint64_t rhs)
{
    return lhs != rhs;
};

inline constexpr Bit ltu_int(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs;
};

inline constexpr Bit lt_int(uint64_t lhs, uint64_t rhs)
{
    return static_cast<int64_t>(lhs) < static_cast<int64_t>(rhs);
};

inline constexpr Bit leu_int(uint64_t lhs, uint64_t rhs)
{
    return lhs <= rhs;
};

inline constexpr Bit le_int(uint64_t lhs, uint64_t rhs)
{
    return static_cast<int64_t>(lhs) <= static_cast<int64_t>(rhs);
};

inline constexpr Bit gtu_int(uint64_t lhs, uint64_t rhs)
{
    return lhs > rhs;
};

inline constexpr Bit gt_int(uint64_t lhs, uint64_t rhs)
{
    return static_cast<int64_t>(lhs) > static_cast<int64_t>(rhs);
};

// 11.9. Vector Integer Min/Max Instructions

inline constexpr uint64_t minu_int(uint64_t lhs, uint64_t rhs)
{
    return std::min(lhs, rhs);
}

inline constexpr uint64_t min_int(uint64_t lhs, uint64_t rhs)
{
    return std::min(static_cast<int64_t>(lhs), static_cast<int64_t>(rhs));
}

inline constexpr uint64_t maxu_int(uint64_t lhs, uint64_t rhs)
{
    return std::max(lhs, rhs);
}

inline constexpr uint64_t max_int(uint64_t lhs, uint64_t rhs)
{
    return std::max(static_cast<int64_t>(lhs), static_cast<int64_t>(rhs));
}

// 11.10. Vector Single-Width Integer Multiply Instructions &
// 11.12. Vector Widening Integer Multiply Instructions
inline constexpr uint64_t mul_int(uint64_t lhs, uint64_t rhs)
{
    return static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
};

inline constexpr uint64_t mulu_int(uint64_t lhs, uint64_t rhs)
{
    return lhs * rhs;
};

inline constexpr uint64_t mulsu_int(uint64_t lhs, uint64_t rhs, SewType sew)
{
    auto const sew_mask = (1_u64 << std::to_underlying(sew)) - 1;
    // rhs is sign extended when read from the vector register, so mask it off again
    return (static_cast<int64_t>(lhs) * (rhs & sew_mask));
};

inline constexpr uint64_t mulh_int(uint64_t lhs, uint64_t rhs, SewType sew)
{
    return (static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs)) >> std::to_underlying(sew);
};

inline constexpr uint64_t mulhu_int(uint64_t lhs, uint64_t rhs, SewType sew)
{
    return (lhs * rhs) >> std::to_underlying(sew);
};

inline constexpr uint64_t mulhsu_int(uint64_t lhs, uint64_t rhs, SewType sew)
{
    auto const sew_mask = (1_u64 << std::to_underlying(sew)) - 1;
    // rhs is sign extended when read from the vector register, so mask it off again
    return (static_cast<int64_t>(lhs) * (rhs & sew_mask)) >> std::to_underlying(sew);
};

/* 11.11. Vector Integer Divide Instructions */
inline constexpr uint64_t divu_int(uint64_t lhs, uint64_t rhs)
{
    // Divide by zero case
    if (rhs == 0)
    {
        return -1;
    }
    return lhs / rhs;
};

inline constexpr uint64_t div_int(uint64_t lhs, uint64_t rhs)
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

inline constexpr uint64_t remu_int(uint64_t lhs, uint64_t rhs)
{
    // Divide by zero case
    if (rhs == 0)
    {
        return lhs;
    }
    return lhs % rhs;
};

inline constexpr uint64_t rem_int(uint64_t lhs, uint64_t rhs)
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

inline constexpr uint64_t macc(uint64_t lhs, uint64_t rhs, uint64_t accumulator)
{
    return accumulator + (lhs * rhs);
}

inline constexpr uint64_t maccsu(uint64_t lhs, uint64_t rhs, uint64_t accumulator, SewType sew)
{
    auto const sew_mask = (1_u64 << std::to_underlying(sew)) - 1;
    // rhs is sign extended when read from the vector register, so mask it off again
    return accumulator + (lhs * (rhs & sew_mask));
}

inline constexpr uint64_t maccus(uint64_t lhs, uint64_t rhs, uint64_t accumulator, SewType sew)
{
    auto const sew_mask = (1_u64 << std::to_underlying(sew)) - 1;
    // rhs is sign extended when read from the vector register, so mask it off again
    return accumulator + ((lhs & sew_mask) * rhs);
}

inline constexpr uint64_t nmsac(uint64_t lhs, uint64_t rhs, uint64_t accumulator)
{
    return accumulator - (lhs * rhs);
}

inline constexpr uint64_t madd(uint64_t lhs, uint64_t rhs, uint64_t accumulator)
{
    return rhs + (lhs * accumulator);
}

inline constexpr uint64_t nmsub(uint64_t lhs, uint64_t rhs, uint64_t accumulator)
{
    return rhs - (lhs * accumulator);
}

// 12. Vector Fixed-Point Arithmetic Instructions

// Helpers

/**
 * @brief Saturates a value to the nearest sew bit boundary
 */
inline constexpr int64_t saturate_boundary_signed(int64_t value, size_t sew)
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
inline constexpr uint64_t saturate_boundary_unsigned(uint64_t value, size_t sew)
{
    auto upper_bound = get_n_bit_mask(sew);
    return (value <= upper_bound) ? value : upper_bound;
}

inline constexpr uint64_t sadd(uint64_t lhs, uint64_t rhs, SewType sew, bool &overflow)
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
        return true;
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

// inline FixpointFunction saddu = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                    uint8_t rounding_mode) -> bool {
//     auto sew_mask = get_n_bit_mask(sew);
//     auto res = (lhs + rhs) & sew_mask;
//     auto sat = false;
//     if (res < lhs)
//     {
//         // Overflow
//         res = sew_mask;
//         sat = true;
//     }
//     vd = res;
//     return sat;
// };

// inline FixpointFunction ssub = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                   uint8_t rounding_mode) -> bool {
//     auto res = static_cast<int64_t>(lhs) - static_cast<int64_t>(rhs);
//     auto msb_lhs = msb_is_set(lhs, sew);
//     auto msb_rhs = msb_is_set(rhs, sew);
//     auto msb_res = msb_is_set(res, sew);

//     if (msb_lhs && !msb_rhs && !msb_res)
//     {
//         // Negative overflow
//         vd = get_min_signed(sew);
//         return true;
//     }

//     if (!msb_lhs && msb_rhs && msb_res)
//     {
//         // Positive overflow
//         vd = get_n_bit_mask(sew - 1);
//         return true;
//     }

//     vd = res;
//     return false;
// };

// inline FixpointFunction ssubu = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                    uint8_t rounding_mode) -> bool {
//     auto sew_mask = get_n_bit_mask(sew);
//     auto res = (lhs - rhs) & sew_mask;
//     auto sat = false;
//     if (res > lhs)
//     {
//         // Overflow
//         res = 1_i64 << (sew - 1);
//         sat = true;
//     }
//     vd = res;
//     return sat;
// };

// /* 12.2. Vector Single-Width Averaging Add and Subtract */

// inline FixpointFunction aadd = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                   uint8_t rounding_mode) -> bool {
//     static constexpr auto rounding_bits = 1;
//     auto res = static_cast<int64_t>(lhs) + static_cast<int64_t>(rhs);
//     vd = roundoff_signed(res, rounding_bits, rounding_mode);
//     return false;
// };

// inline FixpointFunction aaddu = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                    uint8_t rounding_mode) -> bool {
//     static constexpr auto rounding_bits = 1;
//     auto res = lhs + rhs;
//     vd = roundoff_unsigned(res, rounding_bits, rounding_mode);
//     return false;
// };

// inline FixpointFunction asub = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                   uint8_t rounding_mode) -> bool {
//     static constexpr auto rounding_bits = 1;
//     auto res = static_cast<int64_t>(lhs) - static_cast<int64_t>(rhs);
//     vd = roundoff_signed(res, rounding_bits, rounding_mode);
//     return false;
// };

// inline FixpointFunction asubu = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                    uint8_t rounding_mode) -> bool {
//     static constexpr auto rounding_bits = 1;
//     auto res = lhs - rhs;
//     vd = roundoff_unsigned(res, rounding_bits, rounding_mode);
//     return false;
// };

// /* 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation */

// inline FixpointFunction smul = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                   uint8_t rounding_mode) -> bool {
//     auto res = (static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs));
//     res = roundoff_signed(res, sew - 1, rounding_mode);
//     auto clamped_res = saturate_boundary_signed(res, sew);
//     vd = clamped_res;
//     return clamped_res != res;
// };

// /* 12.4. Vector Single-Width Scaling Shift Instructions */

// inline FixpointFunction ssrl = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                   uint8_t rounding_mode) -> bool {
//     // Masking with sew - 1 will provide a bitmask that only uses the lower lg2(SEW) bits.
//     auto shiftamount = rhs & (sew - 1);
//     auto res = roundoff_signed(lhs, shiftamount, rounding_mode);
//     vd = res;
//     return false;
// };

// /* 12.5. Vector Narrowing Fixed-Point Clip Instructions */

// inline FixpointFunction clip = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                   uint8_t rounding_mode) -> bool {
//     // Masking with (sew << 1) - 1 will provide a bitmask that only uses the lower lg2(2*SEW) bits.
//     auto shiftamount = rhs & ((sew << 1) - 1);
//     auto res = roundoff_signed(sign_extend(lhs, 2 * sew), shiftamount, rounding_mode);
//     auto clamped_res = saturate_boundary_signed(res, sew);
//     vd = clamped_res;
//     return clamped_res != res;
// };

// inline FixpointFunction clipu = [](uint64_t lhs, uint64_t rhs, SVElement &vd, size_t sew,
//                                    uint8_t rounding_mode) -> bool {
//     auto shiftamount = rhs & ((sew << 1) - 1);
//     auto res = roundoff_unsigned(lhs, shiftamount, rounding_mode);
//     auto clamped_res = saturate_boundary_unsigned(res, sew);
//     vd = clamped_res;
//     return clamped_res != res;
// };
