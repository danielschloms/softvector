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
using ValueResultOpMaskData = uint64_t (*)(uint64_t /* lhs */, uint64_t /* rhs */, Bit /* mask data bit*/);
using BitResultOp = Bit (*)(uint64_t /* lhs */, uint64_t /* rhs */);
using BitResultOpMaskData = Bit (*)(uint64_t /* lhs */, uint64_t /* rhs */, Bit /* mask data bit*/);

template <typename F>
concept ValidOperation = std::is_same_v<F, ValueResultOp> or std::is_same_v<F, BitResultOp> or
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