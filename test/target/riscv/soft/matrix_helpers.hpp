#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

inline constexpr auto LMUL_M1 = 0b000;
inline constexpr auto LMUL_M2 = 0b001;
inline constexpr auto LMUL_M4 = 0b010;
inline constexpr auto LMUL_M8 = 0b011;

inline constexpr auto SEW_OFFSET = 3;

inline constexpr auto SEW_E8 = 0b000;
inline constexpr auto SEW_E16 = 0b001;
inline constexpr auto SEW_E32 = 0b010;
inline constexpr auto SEW_E64 = 0b011;

inline constexpr auto LAMBDA_1 = 0b001;
inline constexpr auto LAMBDA_2 = 0b010;
inline constexpr auto LAMBDA_4 = 0b011;
inline constexpr auto LAMBDA_8 = 0b100;
inline constexpr auto LAMBDA_16 = 0b101;
inline constexpr auto LAMBDA_32 = 0b110;
inline constexpr auto LAMBDA_64 = 0b111;

struct MatrixVtype
{
    unsigned lmul = 0;
    unsigned sew = 0;
    unsigned lambda = 0;
    bool altfmt_A = false;
    bool altfmt_B = false;
    bool bs = false;
};

template <typename T>
    requires std::is_integral_v<T>
unsigned constexpr max_decimal_width()
{
    auto max = std::numeric_limits<T>::max();
    auto w = 0;
    while (max)
    {
        max /= 10;
        w++;
    }
    return w;
}

inline constexpr MatrixVtype decode_matrix_vtype(uint32_t vtype)
{
    return {
        .lmul = 1U << (vtype & 0b11),
        .sew = 8U << ((vtype >> 3) & 0b11),
        .lambda = 1U << (((vtype >> 28) & 0b111) - 1),
        .altfmt_A = static_cast<bool>((vtype >> 27) & 1),
        .altfmt_B = static_cast<bool>((vtype >> 26) & 1),
        .bs = static_cast<bool>((vtype >> 25) & 1),
    };
}

// Row-major order
template <typename T>
    requires std::is_integral_v<T>
void convert(T *const vector_elements, unsigned const lambda, unsigned const vlen, unsigned v_register,
             unsigned widening, unsigned lmul, bool trans, std::vector<T> &serial, bool to_serial)
{
    auto const sew = sizeof(T) * 8;
    auto const elements_per_register = vlen / sew;
    auto const v_base = v_register * elements_per_register;
    auto const cols = lambda * lmul * widening;
    auto const rows = (vlen / sew) / lambda;

    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            auto const used_col = trans ? row : col;
            auto const used_row = trans ? col : row;
            auto const v_offset = (used_col / lambda) * elements_per_register;
            auto const v_element = (used_row * lambda) + (used_col % lambda);
            if (to_serial)
            {
                serial.push_back(vector_elements[v_base + v_offset + v_element]);
            }
            else
            {
                vector_elements[v_base + v_offset + v_element] = serial[row * col];
            }
        }
    }
}

template <typename T>
    requires std::is_integral_v<T>
bool is_equal(std::vector<T> const &A, std::vector<T> const &B)
{
    assert(A.size() == B.size());
    for (size_t i = 0; i < A.size(); ++i)
    {
        if (!(A[i] == B[i]))
        {
            return false;
        }
    }
    return true;
}

template <typename T>
    requires std::is_integral_v<T>
void print_matrix(std::vector<T> const &vec, unsigned const lambda, unsigned const lmul, unsigned const widening)
{
    auto const cols = lambda * lmul * widening;
    for (size_t row = 0; row < vec.size() / cols; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            if constexpr (std::is_signed_v<T>)
            {
                std::printf("| %-*i ", max_decimal_width<T>() + 1, vec.at(row * cols + col));
            }
            else
            {
                std::printf("| %-*u ", max_decimal_width<T>(), vec.at(row * cols + col));
            }
        }
        std::printf("|\n\n");
    }
}

template <typename T>
    requires std::is_integral_v<T>
void print_rv_matrix(T *const vector_elements, unsigned const lambda, unsigned const vlen, unsigned v_register,
                     unsigned widening, unsigned lmul, bool trans)
{
    auto const sew = sizeof(T) * 8;
    auto const elements_per_register = vlen / sew;
    auto const v_base = v_register * elements_per_register;

    std::printf("v%u: ", v_register);

    auto const cols = lambda * lmul * widening;
    auto const rows = (vlen / sew) / lambda;
    std::printf("%u cols, %lu rows\n", cols, rows);
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            auto const used_col = trans ? row : col;
            auto const used_row = trans ? col : row;
            auto const v_offset = (used_col / lambda) * elements_per_register;
            auto const v_element = (used_row * lambda) + (used_col % lambda);
            if constexpr (std::is_signed_v<T>)
            {
                std::printf("| %-*i ", max_decimal_width<T>() + 1, vector_elements[v_base + v_offset + v_element]);
            }
            else
            {
                std::printf("| %-*u ", max_decimal_width<T>(), vector_elements[v_base + v_offset + v_element]);
            }
        }
        std::printf("|\n\n");
    }
}

inline constexpr uint32_t encode_matrix_vtype(unsigned sew, unsigned lmul, unsigned lambda, bool altfmt_A,
                                              bool altfmt_B, bool bs)
{
    // Ignore VILL
    return (sew << SEW_OFFSET) | (lmul) | (lambda << 28) | (altfmt_A << 27) | (altfmt_B << 26) | (bs << 25);
}

inline constexpr void zero_vectors(uint8_t *vector_field, size_t size)
{
    std::memset(vector_field, 0, size);
}

inline constexpr bool check(unsigned mul_C, unsigned sew)
{
    auto valid_mul_C = (mul_C == 1 || mul_C == 2 || mul_C == 4 || mul_C == 8 || mul_C == 16);
    // std::printf("[Info] Currently only allowing SEW 8");
    auto valid_sew = (sew == SEW_E8);
    return valid_mul_C && valid_sew;
}

template <typename T>
    requires std::is_integral_v<T>
void mmacc(std::vector<T> const &A, std::vector<T> const &B, std::vector<T> &C, unsigned lmul, unsigned lambda,
           unsigned widening)
{
    auto const inner_dim = lmul * lambda * widening;
    auto const C_dim = A.size() / inner_dim;
    // std::printf("inner_dim %u, Cdim %lu\n", inner_dim, C_dim);
    for (size_t row = 0; row < C_dim; ++row)
    {
        for (size_t col = 0; col < C_dim; ++col)
        {
            uint64_t accumulator = 0;
            for (size_t i = 0; i < inner_dim; ++i)
            {
                // std::printf("at %lu\n", row * inner_dim + i);
                // auto const a_v = A.at(row * inner_dim + i);
                // auto const b_v = B.at(row * inner_dim + i);
                // std::printf("%lu += %lu * %lu\n", accumulator, a_v, b_v);
                accumulator += A.at(row * inner_dim + i) * B.at(row * inner_dim + i);
            }
            // std::printf("acc %u\n", accumulator);
            C.at(row * C_dim + col) += accumulator;
        }
    }
}

template <typename T>
    requires std::is_integral_v<T>
void zero_vec(std::vector<T> &vec)
{
    std::memset(vec.data(), 0, vec.size() * sizeof(T));
}

template <typename T>
    requires std::is_integral_v<T>
void seq_fill(std::vector<T> &vec, unsigned lmul, unsigned lambda, unsigned widening, unsigned total_elements)
{
    auto const cols = lmul * lambda * widening;
    auto const rows = total_elements / cols;
    // std::printf("seqfill rows %u cols %u\n", rows, cols);
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            auto const fill_val = (col % (lambda * widening)) + (row * lambda * widening);
            // std::printf("col %u, row %u, Fill %u,\n", col, row, fval);
            vec.push_back(fill_val);
        }
    }
}