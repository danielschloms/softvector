#pragma once

#include <algorithm>
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
using AccumulatorOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, uint64_t /* shift amount mask */);
using CarryBorrowOp = Bit (*)(SewType /* sew */, uint64_t /* lhs */, uint64_t /* rhs */);
using CarryBorrowOpMaskData = Bit (*)(SewType /* sew */, uint64_t /* lhs */, uint64_t /* rhs */,
                                      Bit /* mask data bit */);

template <typename F>
concept ValidOperation =
    std::is_same_v<F, BinaryValueResultOp> or std::is_same_v<F, CarryBorrowOp> or std::is_same_v<F, BitResultOp> or
    std::is_same_v<F, ShiftOp> or std::is_same_v<F, BinaryValueResultOpMaskData> or
    std::is_same_v<F, CarryBorrowOpMaskData> or std::is_same_v<F, AccumulatorOp>;

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

inline constexpr uint64_t mulsu_int(uint64_t lhs, uint64_t rhs)
{
    return static_cast<int64_t>(lhs) * rhs;
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
    return (static_cast<int64_t>(lhs) * rhs) >> std::to_underlying(sew);
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
