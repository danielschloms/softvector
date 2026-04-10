#include "softvector.h"
#include "matrix_helpers.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

static constexpr auto max_vlen = 1024;
static constexpr auto max_vlen_bytes = max_vlen >> 3;
static constexpr auto n_vector_registers = 32;
static constexpr auto vector_field_bytes = max_vlen_bytes * n_vector_registers;

alignas(64) auto vector_field = std::array<uint8_t, vector_field_bytes>{ 0 };

template <typename T>
    requires std::is_integral_v<T>
void seq_increase_test(unsigned sew, unsigned lmul, unsigned lambda, unsigned vd, unsigned vs1, unsigned vs2,
                       unsigned vlen)
{
    auto const vtype = encode_matrix_vtype(sew, lmul, lambda, false, false, false);
    auto const decoded_vtype = decode_matrix_vtype(vtype);
    auto const elements_per_register = vlen / decoded_vtype.sew;
    auto const mul_C = elements_per_register / (decoded_vtype.lambda * decoded_vtype.lambda);
    // std::printf("# of C registers: %u\n", mul_C);
    if (!check(mul_C, sew))
    {
        std::printf("Illegal MUL_C or SEW\n");
        return;
    }

    // std::printf("Setup references\n");
    std::vector<T> A;
    // A.reserve(elements_per_register * lmul);

    std::vector<T> B;
    // B.reserve(elements_per_register * lmul);

    std::vector<T> C;
    // C.reserve(elements_per_register * mul_C);
    C.resize(elements_per_register * mul_C);
    zero_vec(C);

    // std::printf("Fill RV\n");

    // Fill vectors sequentially
    // RISC-V
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype), 1, vs1, 0, vlen,
          (vlen / decoded_vtype.sew) * decoded_vtype.lmul);
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype), 1, vs2, 0, vlen,
          (vlen / decoded_vtype.sew) * decoded_vtype.lmul);

    // std::printf("Fill golden\n");
    // Golden
    seq_fill(A, decoded_vtype.lmul, decoded_vtype.lambda, 1, elements_per_register * decoded_vtype.lmul);
    seq_fill(B, decoded_vtype.lmul, decoded_vtype.lambda, 1, elements_per_register * decoded_vtype.lmul);
    print_matrix(A, decoded_vtype.lambda, decoded_vtype.lmul, 1);

    // std::printf("AS %lu\n", A.size());

    // print_rv_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vs1, 1, decoded_vtype.lmul, false);
    // print_rv_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vs2, 1, decoded_vtype.lmul, true);

    // MMACC
    vmmacc_vv(vector_field.data(), vtype, vd, vs1, vs2, 0, vlen);
    mmacc(A, B, C, decoded_vtype.lmul, decoded_vtype.lambda, 1);

    std::vector<T> C_from_RV;

    convert(vector_field.data(), decoded_vtype.lambda, vlen, vd, 1, mul_C, false, C_from_RV, true);
    if (!is_equal(C, C_from_RV))
    {
        std::printf("Result not equal to golden result!\n");
        std::printf("C\n");
        print_matrix(C, decoded_vtype.lambda, mul_C, 1);
        std::printf("C from RV\n");
        print_matrix(C_from_RV, decoded_vtype.lambda, mul_C, 1);
        std::printf("RV\n");
        print_rv_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vd, 1, mul_C, false);
    }
    else
    {
        std::printf("Test success\n");
    }
    // print_rv_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vd, 1, mul_C, false);
    zero_vectors(vector_field.data(), vector_field.size());
}

int main()
{
    auto const lambdas = std::to_array({ LAMBDA_1, LAMBDA_2, LAMBDA_4, LAMBDA_8, LAMBDA_16, LAMBDA_32, LAMBDA_64 });
    auto const vlens = std::to_array({ 64, 128, 256, 512, 1024 });
    auto const lmuls = std::to_array({ LMUL_M1, LMUL_M2, LMUL_M8, LMUL_M8 });
    auto const sew = 8;

    for (auto &&vlen : vlens)
    {
        for (auto &&lmul : lmuls)
        {
            for (auto &&lambda : lambdas)
            {
                auto const elements_per_register = vlen / 8;
                auto const lambda_val = (1U << lambda) - 1;
                auto const mul_C = elements_per_register / (lambda_val * lambda_val);
                if (!check(mul_C, SEW_E8))
                {
                    continue;
                }
                std::printf("SEW: %u, LMUL: %u, LAMBDA: %u, VLEN: %u\n", 8, 1U << lmul, 1U << (lambda - 1), vlen);
                seq_increase_test<uint8_t>(SEW_E8, lmul, lambda, 0, 1, 2, vlen);
                exit(EXIT_SUCCESS);
            }
        }
    }
}