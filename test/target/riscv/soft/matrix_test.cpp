#include "softvector.h"
#include "matrix_helpers.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

static constexpr auto max_vlen = 1U << 16;
static constexpr auto max_vlen_bytes = max_vlen >> 3;
static constexpr auto n_vector_registers = 32;
static constexpr auto vector_field_bytes = max_vlen_bytes * n_vector_registers;

alignas(64) auto vector_field = std::array<uint8_t, vector_field_bytes>{ 0 };

template <typename T>
    requires std::is_integral_v<T>
bool seq_increase_test(unsigned sew, unsigned lmul, unsigned lambda, unsigned vd, unsigned vs1, unsigned vs2,
                       unsigned vlen)
{
    auto const vtype = encode_matrix_vtype(sew, lmul, lambda, false, false, false);
    auto const decoded_vtype = decode_matrix_vtype(vtype);
    auto const elements_per_register = vlen / decoded_vtype.sew;
    auto const mul_C = elements_per_register / (decoded_vtype.lambda * decoded_vtype.lambda);
    if (!check_mul_C(mul_C))
    {
        std::printf("Illegal MUL_C\n");
        return false;
    }

    std::vector<T> A;
    A.reserve(elements_per_register * lmul);

    std::vector<T> B;
    B.reserve(elements_per_register * lmul);

    std::vector<T> C;
    // C.reserve(elements_per_register * mul_C);
    C.resize(elements_per_register * mul_C);
    zero_vec(C);

    // Fill vectors sequentially
    // RISC-V
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype), 1, vs1, 0, vlen,
          (vlen / decoded_vtype.sew) * decoded_vtype.lmul);
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype), 1, vs2, 0, vlen,
          (vlen / decoded_vtype.sew) * decoded_vtype.lmul);

    // Golden
    seq_fill(A, decoded_vtype.lmul, decoded_vtype.lambda, 1, elements_per_register * decoded_vtype.lmul);
    seq_fill(B, decoded_vtype.lmul, decoded_vtype.lambda, 1, elements_per_register * decoded_vtype.lmul);

    // MMACC
    vmmacc_vv(vector_field.data(), vtype, vd, vs1, vs2, 0, vlen);
    mmacc(A, B, C, decoded_vtype.lmul, decoded_vtype.lambda, 1);

    std::vector<T> C_from_RV;

    convert(static_cast<T *>(static_cast<void *>(vector_field.data())), decoded_vtype.lambda, vlen, vd, 1, mul_C, false,
            C_from_RV, true);
    auto is_ok = is_equal(C, C_from_RV);
    if (!is_ok)
    {
        std::printf("Result not equal to golden result!\n");
        // std::printf("A\n");
        // print_matrix(A, decoded_vtype.lambda, decoded_vtype.lmul, 1);
        // std::printf("ARV\n");
        // print_rv_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vs1, 1, decoded_vtype.lmul, false);
        // std::printf("B\n");
        // print_matrix(B, decoded_vtype.lambda, decoded_vtype.lmul, 1);
        // std::printf("C\n");
        // print_matrix(C, decoded_vtype.lambda, mul_C, 1);
        // std::printf("C from RV\n");
        // print_matrix(C_from_RV, decoded_vtype.lambda, mul_C, 1);
        // std::printf("RV\n");
        // print_rv_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vd, 1, mul_C, false);
    }
    else
    {
        std::printf("Test success\n");
    }
    zero_vectors(vector_field.data(), vector_field.size());
    return is_ok;
}

int main()
{
    auto const lambdas = std::to_array({ LAMBDA_1, LAMBDA_2, LAMBDA_4, LAMBDA_8, LAMBDA_16, LAMBDA_32, LAMBDA_64 });
    auto const lmuls = std::to_array({ LMUL_M1, LMUL_M2, LMUL_M8, LMUL_M8 });
    auto const sews = std::to_array({ SEW_E8, SEW_E16, SEW_E32, SEW_E64 });

    auto vlen = 64;
    while (vlen <= max_vlen)
    {
        for (auto &&lmul : lmuls)
        {
            for (auto &&lambda : lambdas)
            {
                for (auto &&sew : sews)
                {
                    auto const sew_val = 8U << sew;
                    auto const elements_per_register = vlen / sew_val;
                    auto const lambda_val = 1U << (lambda - 1);
                    auto const mul_C = elements_per_register / (lambda_val * lambda_val);
                    if (!check_mul_C(mul_C))
                    {
                        continue;
                    }
                    std::printf("SEW: %u, LMUL: %u, LAMBDA: %u, VLEN: %u\n", sew_val, 1U << lmul, lambda_val, vlen);
                    switch (sew_val)
                    {
                    case 8:
                        seq_increase_test<uint8_t>(sew, lmul, lambda, 16, 0, 8, vlen);
                        break;
                    case 16:
                        seq_increase_test<uint16_t>(sew, lmul, lambda, 16, 0, 8, vlen);
                        break;
                    case 32:
                        seq_increase_test<uint32_t>(sew, lmul, lambda, 16, 0, 8, vlen);
                        break;
                    case 64:
                        seq_increase_test<uint64_t>(sew, lmul, lambda, 16, 0, 8, vlen);
                        break;
                    }
                }
            }
        }
        vlen <<= 1;
    }
}