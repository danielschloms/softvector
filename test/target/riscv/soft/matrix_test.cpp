#include "softvector.h"
#include <array>
#include <cstdint>
#include <cstdio>

#define LMUL_M1 0b000
#define LMUL_M2 0b001
#define LMUL_M4 0b010
#define LMUL_M8 0b011

#define SEW_OFFSET 3

#define SEW_E8 0b000
#define SEW_E16 0b001
#define SEW_E32 0b010
#define SEW_E64 0b011

#define LAMBDA_1 0b001
#define LAMBDA_2 0b010
#define LAMBDA_4 0b011
#define LAMBDA_8 0b100
#define LAMBDA_16 0b101
#define LAMBDA_32 0b110
#define LAMBDA_64 0b111

struct MatrixVtype
{
    unsigned lmul = 0;
    unsigned sew = 0;
    unsigned lambda = 0;
    bool altfmt_A = false;
    bool altfmt_B = false;
    bool bs = false;
};

MatrixVtype decode_matrix_vtype(uint32_t vtype)
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

template <typename T>
void print_v_matrix_trans(T *const vector_elements, unsigned const lambda, unsigned const vlen, unsigned v_register,
                          unsigned widening, unsigned lmul)
{
    auto const sew = sizeof(T) * 8;
    auto const elements_per_register = vlen / sew;
    auto const v_base = v_register * elements_per_register;

    std::printf("v%u: ", v_register);

    auto const cols = lambda * lmul * widening;
    auto const rows = (vlen / sew) / lambda;
    std::printf("%u cols, %lu rows\n", cols, rows);
    for (size_t col = 0; col < cols; ++col)
    {
        for (size_t row = 0; row < rows; ++row)
        {
            auto const v_offset = (col / lambda) * elements_per_register;
            auto const v_element = (row * lambda) + (col % lambda);
            std::printf("| %-3u ", vector_elements[v_base + v_offset + v_element]);
        }
        std::printf("|\n\n");
    }
}

template <typename T>
void print_v_matrix(T *const vector_elements, unsigned const lambda, unsigned const vlen, unsigned v_register,
                    unsigned widening, unsigned lmul)
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
            auto const v_offset = (col / lambda) * elements_per_register;
            auto const v_element = (row * lambda) + (col % lambda);
            std::printf("| %-3u ", vector_elements[v_base + v_offset + v_element]);
        }
        std::printf("|\n\n");
    }
}

constexpr uint32_t encode_matrix_vtype(unsigned sew, unsigned lmul, unsigned lambda, bool altfmt_A, bool altfmt_B,
                                       bool bs)
{
    // Ignore VILL
    return (sew << SEW_OFFSET) | (lmul) | (lambda << 28) | (altfmt_A << 27) | (altfmt_B << 26) | (bs << 25);
}

int main()
{
    // Example
    // VLEN 128, SEW = 8, unsigned, non-widening, Lambda = 4, LMUL = 1
    // -> MUL_C = (VLEN / SEW) / Lambda^2 = 1
    // -> Multiply 2 4x4 matrices into a 4x4 matrix
    // Let vd = 0, vs1 = 1, vs2 = 2
    auto vd = 0;
    auto vs1 = 1;
    auto vs2 = 2;
    static constexpr auto vlen = 128;
    static constexpr auto vlen_bytes = vlen >> 3;
    alignas(64) auto vector_field = std::array<uint8_t, vlen_bytes>{ 0 };
    for (auto &&elm : vector_field)
    {
        elm = 0;
    }
    auto const vtype = encode_matrix_vtype(SEW_E8, LMUL_M1, LAMBDA_4, false, false, false);
    auto const decoded_vtype = decode_matrix_vtype(vtype);

    vid_v(vector_field.data(), static_cast<uint16_t>(vtype), 1, vs1, 0, vlen, vlen_bytes);
    vid_v(vector_field.data(), static_cast<uint16_t>(vtype), 1, vs2, 0, vlen, vlen_bytes);

    print_v_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vs1, 1, decoded_vtype.lmul);
    print_v_matrix_trans(vector_field.data(), decoded_vtype.lambda, vlen, vs2, 1, decoded_vtype.lmul);

    vmmacc_vv(vector_field.data(), vtype, vd, vs1, vs2, 0, vlen);
    print_v_matrix(vector_field.data(), decoded_vtype.lambda, vlen, vd, 1, decoded_vtype.lmul);
}