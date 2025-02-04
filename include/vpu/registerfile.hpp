#pragma once

#include <array>
#include <cstdint>
#include <stdlib.h>
#include <span>

#include "serial/flatarray.hpp"
#include "serial/serial.hpp"
#include "serial/span.hpp"

constexpr auto nRegisters = 32;
#ifndef VLEN
constexpr auto vlen = 1024;
#else
constexpr auto vlen = VLEN;
#endif
constexpr auto vlenBytes = vlen >> 3;

namespace regfile
{
using VectorRegister = std::array<std::uint8_t, vlenBytes>;

enum class Eew : std::uint8_t
{
    e8 = 8,
    e16 = 16,
    e32 = 32,
    e64 = 64
};

enum class Emul
{
    mf8,
    mf4,
    mf2,
    m1,
    m2,
    m4,
    m8
};

class VectorRegisterFile
{
  private:
    std::array<Byte, vlenBytes * nRegisters> memory;

  public:
    VectorRegisterFile(/* args */);
    ~VectorRegisterFile();

    template <Emul emul, Eew eew>
    auto getVector(std::size_t baseRegister, std::size_t vectorLength)
    {
        auto offset = baseRegister * vlenBytes;
        if (eew == Eew::e16)
        {
            return Deserialize<std::vector<std::uint16_t>>(Span(memory).subspan(offset, vectorLength));
        }
    }

    template <typename T, size_t extent>
    auto writeBack(std::span<T, extent> source, size_t destinationBaseRegister) -> void
    {
    }
};

VectorRegisterFile::VectorRegisterFile(/* args */) {}

VectorRegisterFile::~VectorRegisterFile() {}
} // namespace regfile
