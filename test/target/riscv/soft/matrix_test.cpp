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

template <typename BaseType>
struct ResultWidener;

template <>
struct ResultWidener<uint8_t>
{
    using doublewide = int16_t;
    using quadwide = int32_t;
    using octwide = int64_t;
};

template <>
struct ResultWidener<int8_t>
{
    using doublewide = int16_t;
    using quadwide = int32_t;
    using octwide = int64_t;
};

template <>
struct ResultWidener<uint16_t>
{
    using doublewide = int32_t;
    using quadwide = int64_t;
};

template <>
struct ResultWidener<int16_t>
{
    using doublewide = int32_t;
    using quadwide = int64_t;
};

template <>
struct ResultWidener<uint32_t>
{
    using doublewide = int64_t;
};

template <>
struct ResultWidener<int32_t>
{
    using doublewide = int64_t;
};

static constexpr auto max_vlen = 1U << 16;
static constexpr auto max_vlen_bytes = max_vlen >> 3;
static constexpr auto n_vector_registers = 32;
static constexpr auto vector_field_bytes = max_vlen_bytes * n_vector_registers;

alignas(64) auto vector_field = std::array<uint8_t, vector_field_bytes>{ 0 };

template <typename T>
    requires std::is_integral_v<T>
bool seq_increase_test(unsigned sew, unsigned lmul, unsigned lambda, unsigned vd, unsigned vs1, unsigned vs2,
                       unsigned vlen, bool signed_A, bool signed_B)
{
    auto const vtype = encode_matrix_vtype(sew, lmul, lambda, !signed_A, !signed_B, false);
    auto const decoded_vtype = decode_matrix_vtype(vtype);
    auto const elements_per_register = vlen / decoded_vtype.sew;
    auto const mul_C = elements_per_register / (decoded_vtype.lambda * decoded_vtype.lambda);
    if (!check_mul_C(mul_C))
    {
        std::printf("Illegal MUL_C\n");
        return false;
    }

    using ResultType = std::make_signed_t<T>;

    std::vector<T> A;
    A.reserve(elements_per_register * lmul);

    std::vector<T> B;
    B.reserve(elements_per_register * lmul);

    std::vector<ResultType> C;
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
    vmmacc_vv(vector_field.data(), vtype, vd, vs1, vs2, 0, vlen, elements_per_register * decoded_vtype.lmul);
    mmacc(A, B, C, decoded_vtype.lmul, decoded_vtype.lambda, 1, signed_A, signed_B);

    std::vector<ResultType> C_from_RV;

    convert(static_cast<ResultType *>(static_cast<void *>(vector_field.data())), decoded_vtype.lambda, vlen, vd, 1,
            mul_C, false, C_from_RV, true);
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
        std::exit(EXIT_FAILURE);
    }
    else
    {
        std::printf("\t%sSINGLE WIDTH: Test success%s\n", GREEN, RESET);
    }
    zero_vectors(vector_field.data(), vector_field.size());
    return is_ok;
}

template <typename T>
    requires std::is_integral_v<T>
bool seq_increase_test_double(unsigned sew, unsigned lmul, unsigned lambda, unsigned vd, unsigned vs1, unsigned vs2,
                              unsigned vlen, bool signed_A, bool signed_B)
{
    auto const vtype_wmmacc = encode_matrix_vtype(sew, lmul, lambda, !signed_A, !signed_B, false);
    auto const vtype_vid = encode_matrix_vtype(sew - 1, lmul, lambda, !signed_A, !signed_B, false);
    auto const decoded_vtype = decode_matrix_vtype(vtype_wmmacc);
    auto const elements_per_register = vlen / decoded_vtype.sew;
    auto const mul_C = elements_per_register / (decoded_vtype.lambda * decoded_vtype.lambda);
    if (!check_mul_C(mul_C))
    {
        std::printf("Illegal MUL_C\n");
        return false;
    }
    auto const lambda_val = 1U << (lambda - 1);
    // std::printf("QUAD: SEW: %u, LMUL: %u, LAMBDA: %u, VLEN: %u\n", 8u << sew, 1U << lmul, lambda_val, vlen);

    using ResultType = ResultWidener<T>::doublewide;
    static constexpr auto widening = 2;
    std::vector<T> A;
    A.reserve(elements_per_register * lmul * widening);

    std::vector<T> B;
    B.reserve(elements_per_register * lmul * widening);

    std::vector<ResultType> C;
    // C.reserve(elements_per_register * mul_C);
    C.resize(elements_per_register * mul_C);
    zero_vec(C);

    // Fill vectors sequentially
    // RISC-V
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype_vid), 1, vs1, 0, vlen,
          elements_per_register * decoded_vtype.lmul * widening);
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype_vid), 1, vs2, 0, vlen,
          elements_per_register * decoded_vtype.lmul * widening);


    // Golden
    seq_fill(A, decoded_vtype.lmul, decoded_vtype.lambda, widening,
             elements_per_register * decoded_vtype.lmul * widening);
    seq_fill(B, decoded_vtype.lmul, decoded_vtype.lambda, widening,
             elements_per_register * decoded_vtype.lmul * widening);

    // MMACC
    vwmmacc_vv(vector_field.data(), vtype_wmmacc, vd, vs1, vs2, 0, vlen, elements_per_register * decoded_vtype.lmul);
    mmacc(A, B, C, decoded_vtype.lmul, decoded_vtype.lambda, widening, signed_A, signed_B);

    std::vector<ResultType> C_from_RV;

    convert(reinterpret_cast<ResultType *>(vector_field.data()), decoded_vtype.lambda, vlen, vd, 1, mul_C, false,
            C_from_RV, true);
    auto is_ok = is_equal(C, C_from_RV);
    if (!is_ok)
    {
        std::printf("A\n");
        print_matrix(A, decoded_vtype.lambda, decoded_vtype.lmul, widening);
        std::printf("ARV\n");
        print_rv_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vs1, widening, decoded_vtype.lmul, false);
        std::printf("B\n");
        print_matrix(B, decoded_vtype.lambda, decoded_vtype.lmul, widening);
        std::printf("C\n");
        print_matrix(C, decoded_vtype.lambda, mul_C, 1);
        //std::printf("C from RV\n");
        //print_matrix(C_from_RV, decoded_vtype.lambda, mul_C, 1);
        std::printf("RV\n");
        print_rv_matrix(reinterpret_cast<ResultType *>(vector_field.data()), decoded_vtype.lambda, vlen, vd, 1, mul_C,
                        false);
        // for (int i = 0; i < 16; i++)
        // {
        //     std::printf("%i\n", reinterpret_cast<ResultType *>(vector_field.data())[16 * 4 + i]);
        // }
        std::exit(EXIT_FAILURE);
    }
    else
    {
        std::printf("\t%sDOUBLE:       Test success%s\n", GREEN, RESET);
    }
    zero_vectors(vector_field.data(), vector_field.size());
    return is_ok;
}

template <typename T>
    requires std::is_integral_v<T>
bool seq_increase_test_quad(unsigned sew, unsigned lmul, unsigned lambda, unsigned vd, unsigned vs1, unsigned vs2,
                            unsigned vlen, bool signed_A, bool signed_B)
{
    auto const vtype_qwmmacc = encode_matrix_vtype(sew, lmul, lambda, !signed_A, !signed_B, false);
    auto const vtype_vid = encode_matrix_vtype(sew - 2, lmul, lambda, !signed_A, !signed_B, false);
    auto const decoded_vtype = decode_matrix_vtype(vtype_qwmmacc);
    auto const elements_per_register = vlen / decoded_vtype.sew;
    auto const mul_C = elements_per_register / (decoded_vtype.lambda * decoded_vtype.lambda);
    if (!check_mul_C(mul_C))
    {
        std::printf("Illegal MUL_C\n");
        return false;
    }
    auto const lambda_val = 1U << (lambda - 1);
    // std::printf("QUAD: SEW: %u, LMUL: %u, LAMBDA: %u, VLEN: %u\n", 8u << sew, 1U << lmul, lambda_val, vlen);

    using ResultType = ResultWidener<T>::quadwide;
    static constexpr auto widening = 4;
    std::vector<T> A;
    A.reserve(elements_per_register * lmul * widening);

    std::vector<T> B;
    B.reserve(elements_per_register * lmul * widening);

    std::vector<ResultType> C;
    // C.reserve(elements_per_register * mul_C);
    C.resize(elements_per_register * mul_C);
    zero_vec(C);

    // Fill vectors sequentially
    // RISC-V
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype_vid), 1, vs1, 0, vlen,
          elements_per_register * decoded_vtype.lmul * widening);
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype_vid), 1, vs2, 0, vlen,
          elements_per_register * decoded_vtype.lmul * widening);

    // Golden
    seq_fill(A, decoded_vtype.lmul, decoded_vtype.lambda, widening,
             elements_per_register * decoded_vtype.lmul * widening);
    seq_fill(B, decoded_vtype.lmul, decoded_vtype.lambda, widening,
             elements_per_register * decoded_vtype.lmul * widening);

    // MMACC
    vqwmmacc_vv(vector_field.data(), vtype_qwmmacc, vd, vs1, vs2, 0, vlen, elements_per_register* decoded_vtype.lmul);
    mmacc(A, B, C, decoded_vtype.lmul, decoded_vtype.lambda, widening, signed_A, signed_B);

    std::vector<ResultType> C_from_RV;

    convert(reinterpret_cast<ResultType *>(vector_field.data()), decoded_vtype.lambda, vlen, vd, 1, mul_C, false,
            C_from_RV, true);
    auto is_ok = is_equal(C, C_from_RV);
    if (!is_ok)
    {
        std::printf("QUAD: Result not equal to golden result!\n");

        std::printf("A\n");
        print_matrix(A, decoded_vtype.lambda, decoded_vtype.lmul, widening);
        std::printf("ARV\n");
        print_rv_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vs1, widening, decoded_vtype.lmul, false);
        std::printf("B\n");
        print_matrix(B, decoded_vtype.lambda, decoded_vtype.lmul, widening);
        std::printf("C\n");
        print_matrix(C, decoded_vtype.lambda, mul_C, 1);
        std::printf("C from RV\n");
        print_matrix(C_from_RV, decoded_vtype.lambda, mul_C, 1);
        std::printf("RV\n");
        print_rv_matrix(reinterpret_cast<ResultType *>(vector_field.data()), decoded_vtype.lambda, vlen, vd, 1, mul_C,
                        false);
        // for (int i = 0; i < 16; i++)
        // {
        //     std::printf("%i\n", reinterpret_cast<ResultType *>(vector_field.data())[16 * 4 + i]);
        // }
        std::exit(EXIT_FAILURE);
    }
    else
    {
        std::printf("\t%sQUAD:         Test success%s\n", GREEN, RESET);
    }
    zero_vectors(vector_field.data(), vector_field.size());
    return is_ok;
}


template <typename T>
    requires std::is_integral_v<T>
bool load_store_test(unsigned sew, unsigned lmul, unsigned lambda, unsigned vd, unsigned vlen)
{
    auto const vtype = encode_matrix_vtype(sew, lmul, lambda, 0, 0, false);
    auto const decoded_vtype = decode_matrix_vtype(vtype);
    auto const elements_per_register = vlen / decoded_vtype.sew;
    
    using ResultType = std::make_signed_t<T>;

    // create array of memory to load from ( dont use vector!)
    std::vector<ResultType> C;
    C.reserve(elements_per_register * decoded_vtype.lmul);
    seq_fill(C, decoded_vtype.lmul, decoded_vtype.lambda, 1,
             elements_per_register * decoded_vtype.lmul);

    auto const res = vmtl_v(
        vector_field.data(), reinterpret_cast<uint8_t *>(&C[0]), vtype, sizeof(T), vd, lambda,
        decoded_vtype.lmul*decoded_vtype.lambda, 0, vlen, vlen/decoded_vtype.sew * decoded_vtype.lmul);

    std::vector<ResultType> C_from_RV_store(elements_per_register * decoded_vtype.lmul, 0);
    auto const res2 = vmts_v(
        vector_field.data(), reinterpret_cast<uint8_t *>(&C_from_RV_store[0]), vtype, 1, vd, lambda,
        decoded_vtype.lmul*decoded_vtype.lambda, 0, vlen, vlen/decoded_vtype.sew * decoded_vtype.lmul);

    //print_rv_matrix(reinterpret_cast<ResultType *>(vector_field.data()), decoded_vtype.lambda, vlen, vd, 1, decoded_vtype.lmul, false);

    std::vector<ResultType> C_from_RV;

    convert(reinterpret_cast<ResultType *>(vector_field.data()), decoded_vtype.lambda, vlen, vd, 1, decoded_vtype.lmul, false,
            C_from_RV, true);
    
    auto is_ok = is_equal(C, C_from_RV);
    if (!is_ok)
    {
        std::printf("Loaded Result not equal to golden result!\n");
        std::printf("C\n");
        print_matrix(C, decoded_vtype.lambda, decoded_vtype.lmul, 1);
        std::printf("C from RV\n");
        print_matrix(C_from_RV, decoded_vtype.lambda, decoded_vtype.lmul, 1);
        std::exit(EXIT_FAILURE);
    } else if (!is_equal(C, C_from_RV_store)) {
        std::printf("Stored Result not equal to golden result!\n");
        print_matrix(C, decoded_vtype.lambda, decoded_vtype.lmul, 1);
        printf("\n\n\n\n");
        print_matrix(C_from_RV_store, decoded_vtype.lambda, decoded_vtype.lmul, 1);
        std::exit(EXIT_FAILURE);
    } else {
        std::printf("\t%sLOAD : Test success%s\n", GREEN, RESET);
    }
    zero_vectors(vector_field.data(), vector_field.size());
    return is_ok;
}

struct Signs
{
    bool signed_A = false;
    bool signed_B = false;
};

int main()
{
    auto const lambdas = std::to_array({ LAMBDA_1, LAMBDA_2, LAMBDA_4, LAMBDA_8, LAMBDA_16, LAMBDA_32, LAMBDA_64 });
    auto const lmuls = std::to_array({ LMUL_M1, LMUL_M2, LMUL_M4, LMUL_M8 });
    auto const sews = std::to_array({ SEW_E8, SEW_E16, SEW_E32, SEW_E64 });
    auto vlen = 64U;

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
                    switch (sew)
                    {
                    case SEW_E8:
                        load_store_test<uint8_t>(sew, lmul, lambda , 0, vlen);
                        break;
                    case SEW_E16:
                        load_store_test<uint16_t>(sew, lmul, lambda , 0, vlen);
                        break;
                    case SEW_E32:
                        load_store_test<uint32_t>(sew, lmul, lambda , 0, vlen);
                        break;
                    case SEW_E64:
                        load_store_test<uint64_t>(sew, lmul, lambda , 0, vlen);
                        break;
                    default:
                        break;
                    }
                    for (int signed_A = 0; signed_A < 2; signed_A++)
                    {
                        for (int signed_B = 0; signed_B < 2; signed_B++)
                        {
                            std::printf("SEW: %u, LMUL: %u, LAMBDA: %u, VLEN: %u, A %s, B %s\n", sew_val, 1U << lmul,
                                        lambda_val, vlen, signed_A ? "signed" : "unsigned",
                                        signed_B ? "signed" : "unsigned");
                            switch (sew_val)
                            {
                            case 8:
                                seq_increase_test<uint8_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A, signed_B);
                                break;
                            case 16:
                                seq_increase_test<uint16_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A, signed_B);
                                seq_increase_test_double<uint8_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A,
                                                                  signed_B);
                                break;
                            case 32:
                                seq_increase_test<uint32_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A, signed_B);
                                seq_increase_test_double<uint16_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A,
                                                                   signed_B);
                                seq_increase_test_quad<uint8_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A, signed_B);
                                break;
                            case 64:
                                seq_increase_test<uint64_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A, signed_B);
                                seq_increase_test_double<uint32_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A,
                                                                   signed_B);
                                seq_increase_test_quad<uint16_t>(sew, lmul, lambda, 16, 0, 8, vlen, signed_A, signed_B);
                                break;
                            default:
                                break;
                            }
                        }
                    }
                }
            }
        }
        vlen <<= 1;
    }
}