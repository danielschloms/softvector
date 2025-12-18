#pragma once

#include <cstdint>
#include <type_traits>

template <typename T>
concept ValidVectorElementType =
    std::is_same_v<T, uint8_t> or std::is_same_v<T, uint16_t> or std::is_same_v<T, uint32_t> or
    std::is_same_v<T, uint64_t> or std::is_same_v<T, int8_t> or std::is_same_v<T, int16_t> or
    std::is_same_v<T, int32_t> or std::is_same_v<T, int64_t>;

template <typename T>
concept ValidScalarType = std::is_same_v<T, uint32_t> or std::is_same_v<T, uint64_t> or std::is_same_v<T, int32_t> or
                          std::is_same_v<T, int64_t>;

using Bit = bool;

using ValueResultOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */);
using ShiftOp = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, uint64_t /* shift amount mask */);
using ValueResultOpMaskData = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, Bit /* mask data bit*/);
using BitResultOp = Bit (*)(uint64_t /* lhs */, uint64_t /* rhs */);
using BitResultOpMaskData = Bit (*)(uint64_t /* lhs */, uint64_t /* rhs */, Bit /* mask data bit*/);

template <typename F>
concept ValidOperation =
    std::is_same_v<F, ValueResultOp> or std::is_same_v<F, BitResultOp> or std::is_same_v<F, ShiftOp> or
    std::is_same_v<F, ValueResultOpMaskData> or std::is_same_v<F, BitResultOpMaskData>;

inline uint64_t add_int(uint64_t lhs, uint64_t rhs)
{
    return lhs + rhs;
}

inline uint64_t sub_int(uint64_t lhs, uint64_t rhs)
{
    return lhs - rhs;
}

inline uint64_t rsub_int(uint64_t lhs, uint64_t rhs)
{
    return rhs - lhs;
}

inline uint64_t and_int(uint64_t lhs, uint64_t rhs)
{
    return lhs & rhs;
}

inline uint64_t or_int(uint64_t lhs, uint64_t rhs)
{
    return lhs | rhs;
}

inline uint64_t xor_int(uint64_t lhs, uint64_t rhs)
{
    return lhs ^ rhs;
}

inline uint64_t sll_int(uint64_t lhs, uint64_t rhs, uint64_t sew)
{
    return lhs << (rhs & (sew - 1));
}

inline uint64_t srl_int(uint64_t lhs, uint64_t rhs, uint64_t sew)
{
    return lhs >> (rhs & (sew - 1));
}

inline uint64_t sra_int(uint64_t lhs, uint64_t rhs, uint64_t sew)
{
    return static_cast<int64_t>(lhs) >> (rhs & (sew - 1));
}

inline Bit eq_int(uint64_t lhs, uint64_t rhs)
{
    return lhs == rhs;
};

inline Bit ne_int(uint64_t lhs, uint64_t rhs)
{
    return lhs != rhs;
};

inline Bit ltu_int(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs;
};

inline Bit lt_int(uint64_t lhs, uint64_t rhs)
{
    return static_cast<int64_t>(lhs) < static_cast<int64_t>(rhs);
};

inline Bit leu_int(uint64_t lhs, uint64_t rhs)
{
    return lhs <= rhs;
};

inline Bit le_int(uint64_t lhs, uint64_t rhs)
{
    return static_cast<int64_t>(lhs) <= static_cast<int64_t>(rhs);
};

inline Bit gtu_int(uint64_t lhs, uint64_t rhs)
{
    return lhs > rhs;
};

inline Bit gt_int(uint64_t lhs, uint64_t rhs)
{
    return static_cast<int64_t>(lhs) > static_cast<int64_t>(rhs);
};