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
/// \file softvector-vector.cpp
/// \brief Extended vector type for softvector representation
/// \date 07/03/2020
//////////////////////////////////////////////////////////////////////////////////////

#include "vpu/softvector-types.hpp"
#include <cassert>
#include <cstdint>
#include <cstddef>

namespace Deprecated
{

auto roundoff_unsigned(uint64_t value, uint8_t rounding_bits, uint8_t rounding_mode) -> uint64_t;

auto roundoff_signed(int64_t value, uint8_t rounding_bits, uint8_t rounding_mode) -> int64_t;

auto roundoff_unsigned(uint64_t value, uint8_t rounding_bits, uint8_t rounding_mode) -> uint64_t
{
    // Only lower 2 bits are used
    rounding_mode &= 0b11;

    if (rounding_bits == 0)
    {
        return value;
    }
    auto rounding_increment = false;
    auto range_zero_check = false;
    auto bitmask = 0U;

    switch (rounding_mode)
    {
    case 0:
    {
        rounding_increment = static_cast<bool>(value & (1U << (rounding_bits - 1)));
        break;
    }
    case 1:
    {
        // Needs check v[d-2:0] != 0
        if (rounding_bits >= 2)
        {
            // Bitmask for v[d-2 : 0]
            bitmask = (1 << (rounding_bits - 1)) - 1;
            range_zero_check = value & bitmask;
        }
        // v[d-1] & (v[d-2:0] != 0 | v[d])
        bool condition_1 = (value & (1 << (rounding_bits - 1)));
        bool condition_2 = static_cast<bool>(range_zero_check || (value & (1 << rounding_bits)));
        rounding_increment = condition_1 && condition_2;
        break;
    }
    case 2:
    {
        // rounding_increment = 0;
        break;
    }
    case 3:
    {
        // Bitmask for v[d-1 : 0]
        bitmask = (1 << (rounding_bits)) - 1;
        // Needs check v[d-1:0] != 0
        range_zero_check = value & bitmask;
        rounding_increment = !static_cast<bool>(value & (1 << rounding_bits)) && range_zero_check;
        break;
    }
    default:
    {
        // Illegal!
        break;
    }
    }

    return (value >> rounding_bits) + rounding_increment;
}

auto roundoff_signed(int64_t value, uint8_t rounding_bits, uint8_t rounding_mode) -> int64_t
{
    if (rounding_bits == 0)
    {
        return value;
    }

    // Only lower 2 bits are used
    rounding_mode &= 0b11;
    auto range_zero_check = false;
    int64_t bitmask = 0;

    auto rounding_increment = false;
    switch (rounding_mode)
    {
    case 0:
    {
        rounding_increment = static_cast<bool>(value & (1 << (rounding_bits - 1)));
        break;
    }
    case 1:
    {
        // Needs check v[d-2:0] != 0
        if (rounding_bits >= 2)
        {
            // Bitmask for v[d-2 : 0]
            bitmask = (1 << (rounding_bits - 1)) - 1;
            range_zero_check = value & bitmask;
        }
        // v[d-1] & (v[d-2:0] != 0 | v[d])
        bool condition_1 = (value & (1 << (rounding_bits - 1)));
        bool condition_2 = static_cast<bool>(range_zero_check || (value & (1 << rounding_bits)));
        rounding_increment = condition_1 && condition_2;
        break;
    }
    case 2:
    {
        // rounding_increment = 0;
        break;
    }
    case 3:
    {
        // Bitmask for v[d-1 : 0]
        bitmask = (1 << (rounding_bits)) - 1;
        // Needs check v[d-1:0] != 0
        range_zero_check = value & bitmask;
        rounding_increment = !static_cast<bool>(value & (1 << rounding_bits)) && range_zero_check;
        break;
    }
    default:
    {
        // Illegal!
        break;
    }
    }

    return (value >> rounding_bits) + rounding_increment;
}
} // namespace Deprecated

void SVector::assign(const SVector &vin, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        (*this)[i_element] = vin[i_element];
    }
}

SVector &SVector::operator=(const SVector &rhs)
{
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        (*this)[i_element] = rhs[i_element];
    }
    return (*this);
}

SVector SVector::operator+(const SVector &rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] + rhs[i_element];
    }
    return (ret);
}

SVector SVector::operator+(const int64_t rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] + rhs;
    }
    return (ret);
}

SVector SVector::operator&(const SVector &rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] & rhs[i_element];
    }
    return (ret);
}

SVector SVector::operator&(const int64_t rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] & rhs;
    }
    return (ret);
}

SVector SVector::operator|(const SVector &rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] | rhs[i_element];
    }
    return (ret);
}

SVector SVector::operator|(const int64_t rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] | rhs;
    }
    return (ret);
}

SVector SVector::operator^(const SVector &rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] ^ rhs[i_element];
    }
    return (ret);
}

SVector SVector::operator^(const int64_t rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] ^ rhs;
    }
    return (ret);
}

SVector SVector::operator-(const SVector &rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] - rhs[i_element];
    }
    return (ret);
}

SVector SVector::operator-(const int64_t rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] - rhs;
    }
    return (ret);
}

SVector SVector::operator<<(const SVector &rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] << rhs[i_element];
    }
    return (ret);
}

SVector SVector::operator<<(const uint64_t rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] << rhs;
    }
    return (ret);
}

SVector SVector::operator>>(const SVector &rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] >> rhs[i_element];
    }
    return (ret);
}

SVector SVector::operator>>(const uint64_t rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element] >> rhs;
    }
    return (ret);
}

SVector SVector::srl(const SVector &rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element].srl(rhs[i_element]);
    }
    return (ret);
}

SVector SVector::srl(const uint64_t rhs) const
{
    SVector ret(length_, elements_[0]->width_in_bits_, start_reg_index_);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        ret[i_element] = (*this)[i_element].srl(rhs);
    }
    return (ret);
}

SVRegister SVector::operator==(const SVector &rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] == rhs[i_element])
            ret.toggle_bit(i_element);
    }
    return (ret);
}

SVRegister SVector::operator==(const int64_t rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] == rhs)
            ret.toggle_bit(i_element);
    }
    return (ret);
}

SVRegister SVector::operator!=(const SVector &rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] != rhs[i_element])
            ret.toggle_bit(i_element);
    }
    return (ret);
}

SVRegister SVector::operator!=(const int64_t rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] != rhs)
            ret.toggle_bit(i_element);
    }
    return (ret);
}

// signed comparisons
SVRegister SVector::operator<(const SVector &rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] < rhs[i_element])
            ret.toggle_bit(i_element);
    }
    return (ret);
}

SVRegister SVector::operator<(const int64_t rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] < rhs)
            ret.toggle_bit(i_element);
    }
    return (ret);
}

SVRegister SVector::operator<=(const SVector &rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] <= rhs[i_element])
            ret.toggle_bit(i_element);
    }
    return (ret);
}

SVRegister SVector::operator<=(const int64_t rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] <= rhs)
            ret.toggle_bit(i_element);
    }
    return (ret);
}

SVRegister SVector::operator>(const SVector &rhs) const
{
    return (rhs < *this);
}

SVRegister SVector::operator>(const int64_t rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] > rhs)
            ret.toggle_bit(i_element);
    }
    return (ret);
}

SVRegister SVector::operator>=(const SVector &rhs) const
{
    return (rhs <= *this);
}

SVRegister SVector::operator>=(const int64_t rhs) const
{
    SVRegister ret(length_ * (*this)[0].width_in_bits_ / 8);
    for (size_t i_element = 0; i_element < length_; ++i_element)
    {
        if ((*this)[i_element] >= rhs)
            ret.toggle_bit(i_element);
    }
    return (ret);
}

// masked ops
void SVector::m_assign(const SVector &vin, const SVRegister &vm, bool mask, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
            (*this)[i_element] = vin[i_element];
    }
}

void SVector::m_assign(const int64_t rhs, const SVRegister &vm, bool mask, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
            (*this)[i_element] = rhs;
    }
}

SVector &SVector::m_slideup(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask, size_t start_index)
{
    size_t max = rhs > start_index ? rhs : start_index;
    for (size_t i_element = max; i_element < length_; ++i_element)
    {
        // i_element >= max -> i_element - max >= 0
        if (!mask || vm.get_bit(i_element))
            (*this)[i_element] = opL[i_element - max];
    }
    return (*this);
}

SVector &SVector::m_slidedown(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask, size_t vlmax,
                              size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            size_t i_src_element = i_element + rhs;
            if (i_src_element < length_)
            {
                (*this)[i_element] = opL[i_element + rhs];
            }
            else
            {
                if (i_src_element < vlmax)
                {
                    (*this)[i_element] =
                        SVElement(opL[0].width_in_bits_, opL[0].mem_ + i_src_element * opL[0].width_in_bits_ / 8);
                }
                else
                {
                    (*this)[i_element] = 0;
                }
            }
        }
    }
    return (*this);
}

SVector &SVector::m_vrgather(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask, size_t vlmax,
                             size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            // Risky?
            auto i_rhs = rhs[i_element].to_u64();
            (*this)[i_element] = i_rhs >= vlmax ? 0 : opL[i_rhs].to_i64();
        }
    }
    return (*this);
}

SVector &SVector::m_vrgather(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask, size_t vlmax,
                             size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            // Risky?
            (*this)[i_element] = rhs >= vlmax ? 0 : opL[rhs].to_i64();
        }
    }
    return (*this);
}

SVector &SVector::m_vcompress(const SVector &opL, const SVRegister &vm, size_t start_index)
{
    size_t i_dest = 0;
    for (size_t i_element = start_index; i_element < this->length_; ++i_element)
    {
        if (vm.get_bit(i_element))
        {
            (*this)[i_dest] = opL[i_element];
            i_dest++;
        }
    }
    return (*this);
}

// 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
SVector &SVector::m_adc(const SVector &opL, const SVector &rhs, const SVRegister &vm, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        (*this)[i_element] = opL[i_element].to_i64() + rhs[i_element].to_i64() + vm.get_bit(i_element);
    }
    return (*this);
}

SVector &SVector::m_adc(const SVector &opL, const int64_t rhs, const SVRegister &vm, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        (*this)[i_element] = opL[i_element].to_i64() + rhs + vm.get_bit(i_element);
    }
    return (*this);
}

SVector &SVector::m_sbc(const SVector &opL, const SVector &rhs, const SVRegister &vm, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        (*this)[i_element] = opL[i_element].to_i64() - rhs[i_element].to_i64() - vm.get_bit(i_element);
    }
    return (*this);
}

SVector &SVector::m_sbc(const SVector &opL, const int64_t rhs, const SVRegister &vm, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        (*this)[i_element] = opL[i_element].to_i64() - rhs - vm.get_bit(i_element);
    }
    return (*this);
}
/* End 11.4. */

/* 11.15. Vector Integer Merge Instructions */
SVector &SVector::m_merge(const SVector &opL, const SVector &rhs, const SVRegister &vm, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        (*this)[i_element] = vm.get_bit(i_element) ? rhs[i_element] : opL[i_element];
    }
    return (*this);
}

SVector &SVector::m_merge(const SVector &opL, const int64_t rhs, const SVRegister &vm, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        (*this)[i_element] = vm.get_bit(i_element) ? rhs : opL[i_element].to_i64();
    }
    return (*this);
}
/* End 11.15. */

/* 12. Vector Fixed-Point Arithmetic Instructions */
/* 12.1. Vector Single-Width Saturating Add and Subtract */
SVector &SVector::m_sat_addu(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask, bool *sat,
                             size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_u64();
            auto rhs_u64 = rhs[i_element].to_u64();
            auto result = opL_u64 + rhs_u64;
            uint64_t msb = static_cast<uint64_t>(1U) << (opL[i_element].width_in_bits_ - 1);
            bool msb_result = result & msb;
            if ((opL[i_element].msb_is_set() || rhs[i_element].msb_is_set()) && !msb_result)
            {
                // Saturation, use max. uint
                (*this)[i_element] = -1;
                (*sat) = true;
                continue;
            }
            (*this)[i_element] = result;
        }
    }
    return (*this);
}

SVector &SVector::m_sat_addu(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask, bool *sat,
                             size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_u64();
            auto result = opL_u64 + rhs;
            uint64_t msb = static_cast<uint64_t>(1U) << (opL[i_element].width_in_bits_ - 1);
            bool msb_rhs = rhs & msb;
            bool msb_result = result & msb;
            if ((opL[i_element].msb_is_set() || msb_rhs) && !msb_result)
            {
                // Saturation, use max. uint
                (*this)[i_element] = -1;
                (*sat) = true;
                continue;
            }
            (*this)[i_element] = result;
        }
    }
    return (*this);
}

SVector &SVector::m_sat_add(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask, bool *sat,
                            size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_i64 = opL[i_element].to_i64();
            auto rhs_i64 = rhs[i_element].to_i64();
            auto result = opL_i64 + rhs_i64;
            int64_t msb = static_cast<int64_t>(1) << (opL[i_element].width_in_bits_ - 1);
            bool msb_result = result & msb;
            bool msb_opL = opL[i_element].msb_is_set();
            bool msb_rhs = rhs[i_element].msb_is_set();
            if ((msb_opL && msb_rhs && !msb_result))
            {
                // Saturation to min. signed value
                (*this)[i_element].set_min_signed();
                (*sat) = true;
                continue;
            }
            if (!msb_opL && !msb_rhs && msb_result)
            {
                // Saturation to max. signed value
                (*this)[i_element].set_max_signed();
                (*sat) = true;
                continue;
            }
            (*this)[i_element] = result;
        }
    }
    return (*this);
}

SVector &SVector::m_sat_add(const SVector &opL, const int64_t rhs, const SVRegister &vm, bool mask, bool *sat,
                            size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_i64();
            auto result = opL_u64 + rhs;
            int64_t msb = static_cast<int64_t>(1U) << (opL[i_element].width_in_bits_ - 1);
            bool msb_opL = opL[i_element].msb_is_set();
            bool msb_rhs = rhs & msb;
            bool msb_result = result & msb;
            if ((msb_opL && msb_rhs && !msb_result))
            {
                // Saturation to min. signed value
                (*this)[i_element].set_min_signed();
                (*sat) = true;
                continue;
            }
            if (!msb_opL && !msb_rhs && msb_result)
            {
                // Saturation to max. signed value
                (*this)[i_element].set_max_signed();
                (*sat) = true;
                continue;
            }
            (*this)[i_element] = result;
        }
    }
    return (*this);
}

SVector &SVector::m_sat_subu(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask, bool *sat,
                             size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_u64();
            auto rhs_u64 = rhs[i_element].to_u64();
            auto result = opL_u64 - rhs_u64;
            uint64_t msb = static_cast<uint64_t>(1U) << (opL[i_element].width_in_bits_ - 1);
            if (opL_u64 < rhs_u64)
            {
                // Saturation, use min. uint
                (*this)[i_element] = 0;
                (*sat) = true;
                continue;
            }
            (*this)[i_element] = result;
        }
    }
    return (*this);
}

SVector &SVector::m_sat_subu(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask, bool *sat,
                             size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_u64();
            auto result = opL_u64 - rhs;
            uint64_t msb = static_cast<uint64_t>(1U) << (opL[i_element].width_in_bits_ - 1);
            if (opL_u64 < rhs)
            {
                // Saturation, use max. uint
                (*this)[i_element] = 0;
                (*sat) = true;
                continue;
            }
            (*this)[i_element] = result;
        }
    }
    return (*this);
}

SVector &SVector::m_sat_sub(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask, bool *sat,
                            size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_i64 = opL[i_element].to_i64();
            auto rhs_i64 = rhs[i_element].to_i64();
            auto result = opL_i64 - rhs_i64;
            int64_t msb = static_cast<int64_t>(1) << (opL[i_element].width_in_bits_ - 1);
            bool msb_result = result & msb;
            bool msb_opL = opL[i_element].msb_is_set();
            bool msb_rhs = rhs[i_element].msb_is_set();
            if ((msb_opL && !msb_rhs && !msb_result))
            {
                // Neg - Pos = Pos -> Negative Overflow
                // Saturation to min. signed value
                (*this)[i_element].set_min_signed();
                (*sat) = true;
                continue;
            }
            if (!msb_opL && msb_rhs && msb_result)
            {
                // Pos - Neg = Neg -> Positive Overflow
                // Saturation to max. signed value
                (*this)[i_element].set_max_signed();
                (*sat) = true;
                continue;
            }
            (*this)[i_element] = result;
        }
    }
    return (*this);
}

SVector &SVector::m_sat_sub(const SVector &opL, const int64_t rhs, const SVRegister &vm, bool mask, bool *sat,
                            size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_i64();
            auto result = opL_u64 - rhs;
            int64_t msb = static_cast<int64_t>(1U) << (opL[i_element].width_in_bits_ - 1);
            bool msb_opL = opL[i_element].msb_is_set();
            bool msb_rhs = rhs & msb;
            bool msb_result = result & msb;
            if ((msb_opL && !msb_rhs && !msb_result))
            {
                // Neg - Pos = Pos -> Negative Overflow
                // Saturation to min. signed value
                (*this)[i_element].set_min_signed();
                (*sat) = true;
                continue;
            }
            if (!msb_opL && msb_rhs && msb_result)
            {
                // Pos - Neg = Neg -> Positive Overflow
                // Saturation to max. signed value
                (*this)[i_element].set_max_signed();
                (*sat) = true;
                continue;
            }
            (*this)[i_element] = result;
        }
    }
    return (*this);
}
/* End 12.1. */

/* 12.2. Vector Single-Width Averaging Add and Subtract */
SVector &SVector::m_avg_addu(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                             uint8_t rounding_mode, size_t start_index)
{
    constexpr auto rounding_bits = 1;
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_u64();
            auto rhs_u64 = rhs[i_element].to_u64();
            (*this)[i_element] = Deprecated::roundoff_unsigned(opL_u64 + rhs_u64, rounding_bits, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_avg_addu(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask,
                             uint8_t rounding_mode, size_t start_index)
{
    constexpr auto rounding_bits = 1;
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_u64();
            (*this)[i_element] = Deprecated::roundoff_unsigned(opL_u64 + rhs, rounding_bits, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_avg_add(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                            uint8_t rounding_mode, size_t start_index)
{
    constexpr auto rounding_bits = 1;
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_i64 = opL[i_element].to_i64();
            auto rhs_i64 = rhs[i_element].to_i64();
            (*this)[i_element] = Deprecated::roundoff_signed(opL_i64 + rhs_i64, rounding_bits, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_avg_add(const SVector &opL, const int64_t rhs, const SVRegister &vm, bool mask,
                            uint8_t rounding_mode, size_t start_index)
{
    constexpr auto rounding_bits = 1;
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_i64 = opL[i_element].to_i64();
            (*this)[i_element] = Deprecated::roundoff_signed(opL_i64 + rhs, rounding_bits, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_avg_subu(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                             uint8_t rounding_mode, size_t start_index)
{
    constexpr auto rounding_bits = 1;
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_u64();
            auto rhs_u64 = rhs[i_element].to_u64();
            (*this)[i_element] = Deprecated::roundoff_unsigned(opL_u64 + rhs_u64, rounding_bits, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_avg_subu(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask,
                             uint8_t rounding_mode, size_t start_index)
{
    constexpr auto rounding_bits = 1;
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_u64 = opL[i_element].to_u64();
            (*this)[i_element] = Deprecated::roundoff_unsigned(opL_u64 + rhs, rounding_bits, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_avg_sub(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                            uint8_t rounding_mode, size_t start_index)
{
    constexpr auto rounding_bits = 1;
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_i64 = opL[i_element].to_i64();
            auto rhs_i64 = rhs[i_element].to_i64();
            (*this)[i_element] = Deprecated::roundoff_signed(opL_i64 - rhs_i64, rounding_bits, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_avg_sub(const SVector &opL, const int64_t rhs, const SVRegister &vm, bool mask,
                            uint8_t rounding_mode, size_t start_index)
{
    constexpr auto rounding_bits = 1;
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto opL_i64 = opL[i_element].to_i64();
            (*this)[i_element] = Deprecated::roundoff_signed(opL_i64 - rhs, rounding_bits, rounding_mode);
        }
    }
    return (*this);
}
/* End 12.2. */

/* 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation */
SVector &SVector::m_round_sat_mul(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                                  uint8_t rounding_mode, bool *sat, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            // vsmul.vv vd, vs2, vs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1))
            // vsmul.vx vd, vs2, rs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))
            auto rounding_bits = opL[i_element].width_in_bits_ - 1;
            auto opL_i64 = opL[i_element].to_i64();
            auto rhs_i64 = rhs[i_element].to_i64();
            auto result = Deprecated::roundoff_signed(opL_i64 * rhs_i64, rounding_bits, rounding_mode);

            int64_t msb = static_cast<int64_t>(1U) << (opL[i_element].width_in_bits_ - 1);
            bool msb_opL = opL_i64 & msb;
            bool msb_rhs = rhs_i64 & msb;
            bool msb_result = result & msb;
            if ((msb_opL != msb_rhs) && !msb_result)
            {
                // Different sign multiplication = Pos -> overflow
                // Saturation to min. signed value
                (*this)[i_element].set_min_signed();
                (*sat) = true;
                continue;
            }
            if ((msb_opL == msb_rhs) && msb_result)
            {
                // Same sign multiplication = Neg -> overflow
                // Saturation to max. signed value
                (*this)[i_element].set_max_signed();
                (*sat) = true;
                continue;
            }

            (*this)[i_element] = result;
        }
    }
    return (*this);
}

SVector &SVector::m_round_sat_mul(const SVector &opL, const int64_t rhs, const SVRegister &vm, bool mask,
                                  uint8_t rounding_mode, bool *sat, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            // vsmul.vv vd, vs2, vs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1))
            // vsmul.vx vd, vs2, rs1, vm  # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))
            auto rounding_bits = opL[i_element].width_in_bits_ - 1;
            auto opL_i64 = opL[i_element].to_i64();
            auto result = Deprecated::roundoff_signed(opL_i64 * rhs, rounding_bits, rounding_mode);

            int64_t msb = static_cast<int64_t>(1U) << (opL[i_element].width_in_bits_ - 1);
            bool msb_opL = opL_i64 & msb;
            bool msb_rhs = rhs & msb;
            bool msb_result = result & msb;
            if ((msb_opL != msb_rhs) && !msb_result)
            {
                // Different sign multiplication = Pos -> overflow
                // Saturation to min. signed value
                (*this)[i_element].set_min_signed();
                (*sat) = true;
                continue;
            }
            if ((msb_opL == msb_rhs) && msb_result)
            {
                // Same sign multiplication = Neg -> overflow
                // Saturation to max. signed value
                (*this)[i_element].set_max_signed();
                (*sat) = true;
                continue;
            }

            (*this)[i_element] = result;
        }
    }
    return (*this);
}
/* End 12.3. */

/* 12.4. Vector Single-Width Scaling Shift Instructions */
SVector &SVector::m_scaling_srl(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                                uint8_t rounding_mode, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto bitmask = opL[i_element].width_in_bits_ - 1;
            (*this)[i_element] = Deprecated::roundoff_unsigned(opL[i_element].to_u64(),
                                                               rhs[i_element].to_u64() & bitmask, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_scaling_srl(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask,
                                uint8_t rounding_mode, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto bitmask = opL[i_element].width_in_bits_ - 1;
            (*this)[i_element] = Deprecated::roundoff_unsigned(opL[i_element].to_u64(), rhs & bitmask, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_scaling_sra(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                                uint8_t rounding_mode, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto bitmask = opL[i_element].width_in_bits_ - 1;
            (*this)[i_element] =
                Deprecated::roundoff_signed(opL[i_element].to_i64(), rhs[i_element].to_u64() & bitmask, rounding_mode);
        }
    }
    return (*this);
}

SVector &SVector::m_scaling_sra(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask,
                                uint8_t rounding_mode, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            auto bitmask = opL[i_element].width_in_bits_ - 1;
            (*this)[i_element] = Deprecated::roundoff_signed(opL[i_element].to_i64(), rhs & bitmask, rounding_mode);
        }
    }
    return (*this);
}
/* End 12.4. */

/* 12.5. Vector Narrowing Fixed-Point Clip Instructions */
SVector &SVector::m_narrowing_clipu(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                                    uint8_t rounding_mode, bool *sat, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            // opL: 2*SEW
            auto bitmask = opL[i_element].width_in_bits_ - 1;
            auto result = Deprecated::roundoff_unsigned(opL[i_element].to_u64(), rhs[i_element].to_u64() & bitmask,
                                                        rounding_mode);
            if (result > (*this)[i_element].get_max_unsigned())
            {
                // Overflow
                *sat = true;
                (*this)[i_element] = -1;
            }
        }
    }
    return (*this);
}

SVector &SVector::m_narrowing_clipu(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask,
                                    uint8_t rounding_mode, bool *sat, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            // opL: 2*SEW
            auto bitmask = opL[i_element].width_in_bits_ - 1;
            auto result = Deprecated::roundoff_signed(opL[i_element].to_u64(), rhs & bitmask, rounding_mode);
            if (result > (*this)[i_element].get_max_unsigned())
            {
                // Overflow
                *sat = true;
                (*this)[i_element] = -1;
            }
        }
    }
    return (*this);
}

SVector &SVector::m_narrowing_clip(const SVector &opL, const SVector &rhs, const SVRegister &vm, bool mask,
                                   uint8_t rounding_mode, bool *sat, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            // opL: 2*SEW
            auto bitmask = opL[i_element].width_in_bits_ - 1;
            auto result =
                Deprecated::roundoff_signed(opL[i_element].to_i64(), rhs[i_element].to_u64() & bitmask, rounding_mode);
            if (result > (*this)[i_element].get_max_signed())
            {
                // Overflow
                *sat = true;
                (*this)[i_element].set_max_signed();
            }
        }
    }
    return (*this);
}

SVector &SVector::m_narrowing_clip(const SVector &opL, const uint64_t rhs, const SVRegister &vm, bool mask,
                                   uint8_t rounding_mode, bool *sat, size_t start_index)
{
    for (size_t i_element = start_index; i_element < length_; ++i_element)
    {
        if (!mask || vm.get_bit(i_element))
        {
            // opL: 2*SEW
            auto bitmask = opL[i_element].width_in_bits_ - 1;
            auto result = Deprecated::roundoff_signed(opL[i_element].to_i64(), rhs & bitmask, rounding_mode);
            if (result > (*this)[i_element].get_max_signed())
            {
                // Overflow
                *sat = true;
                (*this)[i_element].set_max_signed();
            }
        }
    }
    return (*this);
}
/* End 12.5. */

/* End 12. */
