// SoftVector micro-benchmark harness.
//
// These benchmarks drive the public C API exactly the way a RISC-V simulator
// drives SoftVector: they hold a flat vector register file and repeatedly issue
// RVV instructions over it. The goal is to expose the cost of the per-element
// dispatch kernels in the non-legacy hot path (softvector.cpp / softfloat-ext).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>

#include "softvector.h"
#include "arithmetic/softfloat-extension.hpp"

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config
{
    uint32_t vlen_bits = 128;   // Vector register length in bits
    uint32_t reps_per_case = 100000;
    bool smoke = false;         // short run (for perf sampling)
    std::string filter;         // substring filter on case name
};

static Config g_cfg;

// ---------------------------------------------------------------------------
// Vector register file
// ---------------------------------------------------------------------------
struct VecRegFile
{
    uint32_t vlen_bits;
    std::vector<uint8_t> mem;

    explicit VecRegFile(uint32_t vlen_bits_, bool fill = true)
        : vlen_bits(vlen_bits_), mem(32 * vlen_bits_ / 8, 0)
    {
        if (fill)
        {
            uint64_t s = 0x9E3779B97F4A7C15ull;
            for (size_t i = 0; i < mem.size(); ++i)
            {
                s = s * 0x5851F42D4C957F2Dull + 0x12345;
                mem[i] = static_cast<uint8_t>(s >> 56);
            }
        }
    }

    uint8_t *data() { return mem.data(); }
};

// Encode a vtype for a given SEW code (0..3 -> 8/16/32/64) and LMUL=1.
static inline uint16_t vtype_for_sew(unsigned sew_code)
{
    // SEW field = bits[4:3], LMUL field = bits[2:0] (1 for LMUL=1)
    return static_cast<uint16_t>((sew_code & 0x3) << 3);
}

// element count consumed per instruction for a given sew (for the object we set)
static inline uint32_t vl_for(uint32_t vlen_bits, unsigned sew_bits)
{
    return vlen_bits / sew_bits;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
struct Result
{
    std::string name;
    double ns_per_op;
    double gops;
    double ns_per_elem;
    uint64_t checksum;
};

static inline uint64_t checksum_accum(uint64_t acc, uint8_t *data, size_t bytes)
{
    // simple folding checksum over the register file
    uint64_t *p = reinterpret_cast<uint64_t *>(data);
    size_t words = bytes / 8;
    for (size_t i = 0; i < words; ++i) acc ^= (p[i] * 0x9E3779B97F4A7C15ull) + (acc << 6) + (acc >> 2);
    for (size_t i = words * 8; i < bytes; ++i) acc ^= static_cast<uint64_t>(data[i]);
    return acc;
}

template <typename Fn>
static Result bench(const std::string &name, Fn &&fn, uint8_t *data, size_t bytes,
                    uint32_t elems_per_op)
{
    // warm-up
    fn();
    fn();

    uint64_t checksum = 0;
    const uint32_t reps = g_cfg.smoke ? std::min<uint32_t>(g_cfg.reps_per_case, 2000)
                                      : g_cfg.reps_per_case;

    auto t0 = Clock::now();
    for (uint32_t i = 0; i < reps; ++i)
    {
        fn();
    }
    auto t1 = Clock::now();
    checksum = checksum_accum(checksum, data, bytes);

    Result r;
    r.name = name;
    double total_ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    r.ns_per_op = total_ns / static_cast<double>(reps);
    r.gops = 1000.0 / r.ns_per_op; // Gop/s = 1e9 ns / ns_per_op / 1e9 = 1 / (ns_per_op) * 1e3
    r.ns_per_elem = elems_per_op ? (r.ns_per_op / static_cast<double>(elems_per_op)) : 0.0;
    r.checksum = checksum;
    return r;
}

// ---------------------------------------------------------------------------
// Case registry
// ---------------------------------------------------------------------------
struct Case
{
    std::string name;
    std::function<Result()> run;
};

static std::vector<Case> g_cases;
static uint8_t *g_field = nullptr;
static size_t g_field_bytes = 0;
static uint32_t g_vlen = 0;
static uint32_t g_vl_8 = 0, g_vl_16 = 0, g_vl_32 = 0, g_vl_64 = 0;

// Register index helpers: keep src/dst distinct so we can reuse the file.
static constexpr uint8_t VD = 1;
static constexpr uint8_t VS1 = 2;
static constexpr uint8_t VS2 = 3;
static constexpr uint8_t VD_F = 4;
static constexpr uint8_t VS1_F = 5;
static constexpr uint8_t VS2_F = 6;

// --- Integer VV -----------------------------------------------------------
#define DEF_INT_VV(NAME, CALL)                                                        \
    do                                                                                 \
    {                                                                                  \
        uint16_t vt8 = vtype_for_sew(0), vt16 = vtype_for_sew(1), vt32 = vtype_for_sew(2), vt64 = vtype_for_sew(3); \
        g_cases.push_back({#NAME "_sew8", [=]() -> Result {                           \
            const uint32_t vl = g_vl_8;                                               \
            auto f = [&]() { CALL(g_field, vt8, 1, VD, VS1, VS2, 0, g_vlen, vl); };   \
            return bench(#NAME "_sew8", f, g_field, g_field_bytes, vl); } });        \
        g_cases.push_back({#NAME "_sew16", [=]() -> Result {                          \
            const uint32_t vl = g_vl_16;                                              \
            auto f = [&]() { CALL(g_field, vt16, 1, VD, VS1, VS2, 0, g_vlen, vl); };  \
            return bench(#NAME "_sew16", f, g_field, g_field_bytes, vl); } });        \
        g_cases.push_back({#NAME "_sew32", [=]() -> Result {                          \
            const uint32_t vl = g_vl_32;                                              \
            auto f = [&]() { CALL(g_field, vt32, 1, VD, VS1, VS2, 0, g_vlen, vl); };  \
            return bench(#NAME "_sew32", f, g_field, g_field_bytes, vl); } });        \
        g_cases.push_back({#NAME "_sew64", [=]() -> Result {                          \
            const uint32_t vl = g_vl_64;                                              \
            auto f = [&]() { CALL(g_field, vt64, 1, VD, VS1, VS2, 0, g_vlen, vl); };  \
            return bench(#NAME "_sew64", f, g_field, g_field_bytes, vl); } });        \
    } while (0)

// --- Masked integer VV ----------------------------------------------------
#define DEF_INT_VV_MASKED(NAME, CALL)                                                 \
    do                                                                                 \
    {                                                                                  \
        uint16_t vt32 = vtype_for_sew(2);                                             \
        g_cases.push_back({#NAME "_sew32_masked", [=]() -> Result {                   \
            const uint32_t vl = g_vl_32;                                              \
            auto f = [&]() { CALL(g_field, vt32, 0, VD, VS1, VS2, 0, g_vlen, vl); };  \
            return bench(#NAME "_sew32_masked", f, g_field, g_field_bytes, vl); } });\
    } while (0)

// --- Float VV -------------------------------------------------------------
#define DEF_FLOAT_VV(NAME, CALL)                                                      \
    do                                                                                 \
    {                                                                                  \
        uint16_t vt32 = vtype_for_sew(2);                                             \
        g_cases.push_back({#NAME "_sew32", [=]() -> Result {                          \
            const uint32_t vl = g_vl_32;                                              \
            auto f = [&]() { CALL(g_field, vt32, 1, VD_F, VS1_F, VS2_F, 0, g_vlen, vl, 0); }; \
            return bench(#NAME "_sew32", f, g_field, g_field_bytes, vl); } });        \
        uint16_t vt64 = vtype_for_sew(3);                                             \
        g_cases.push_back({#NAME "_sew64", [=]() -> Result {                          \
            const uint32_t vl = g_vl_64;                                              \
            auto f = [&]() { CALL(g_field, vt64, 1, VD_F, VS1_F, VS2_F, 0, g_vlen, vl, 0); }; \
            return bench(#NAME "_sew64", f, g_field, g_field_bytes, vl); } });        \
    } while (0)

static void register_cases()
{
    DEF_INT_VV(vadd_vv, vadd_vv);
    DEF_INT_VV(vsub_vv, vsub_vv);
    DEF_INT_VV(vmul_vv, vmul_vv);
    DEF_INT_VV(vdiv_vv, vdiv_vv);
    DEF_INT_VV(vand_vv, vand_vv);
    DEF_INT_VV(vor_vv, vor_vv);
    DEF_INT_VV(vxor_vv, vxor_vv);
    DEF_INT_VV(vmaxu_vv, vmaxu_vv);
    DEF_INT_VV(vminu_vv, vminu_vv);
    DEF_INT_VV(vsll_vv, vsll_vv);
    DEF_INT_VV(vredsum_vs, vredsum_vs);
    DEF_INT_VV_MASKED(vxor_vv, vxor_vv);

    // Float single-width VV (rounding mode 0 = rnu)
    DEF_FLOAT_VV(vfadd_vv, vfadd_vv);
    DEF_FLOAT_VV(vfsub_vv, vfsub_vv);
    DEF_FLOAT_VV(vfmul_vv, vfmul_vv);
    DEF_FLOAT_VV(vfdiv_vv, vfdiv_vv);
    DEF_FLOAT_VV(vfmin_vv, vfmin_vv);
    DEF_FLOAT_VV(vfsgnj_vv, vfsgnj_vv);

    // Unary / convert ops (SEW 32)
    {

        uint16_t vt32 = vtype_for_sew(2);
        g_cases.push_back({"vfsqrt_v_sew32", [=]() -> Result {
            const uint32_t vl = g_vl_32;
            auto f = [&]() { vfsqrt_v(g_field, vt32, 1, VD_F, VS2_F, 0, g_vlen, vl, 0); };
            return bench("vfsqrt_v_sew32", f, g_field, g_field_bytes, vl);
        }});
        g_cases.push_back({"vfcvt_x_f_v_sew32", [=]() -> Result {
            const uint32_t vl = g_vl_32;
            auto f = [&]() { vfcvt_x_f_v(g_field, vt32, 1, VD, VS2_F, 0, g_vlen, vl, 0); };
            return bench("vfcvt_x_f_v_sew32", f, g_field, g_field_bytes, vl);
        }});
        g_cases.push_back({"vfcvt_f_x_v_sew32", [=]() -> Result {
            const uint32_t vl = g_vl_32;
            auto f = [&]() { vfcvt_f_x_v(g_field, vt32, 1, VD_F, VS2, 0, g_vlen, vl, 0); };
            return bench("vfcvt_f_x_v_sew32", f, g_field, g_field_bytes, vl);
        }});
        g_cases.push_back({"vfwadd_vv_sew32", [=]() -> Result {
            const uint32_t vl = g_vl_32;
            auto f = [&]() { vfwadd_vv(g_field, vt32, 1, VD_F, VS1_F, VS2_F, 0, g_vlen, vl, 0); };
            return bench("vfwadd_vv_sew32", f, g_field, g_field_bytes, vl);
        }});
    }

    // Mixed workload: realistic interleaving of int/fp/convert
    g_cases.push_back({"mixed_integer_fp", [=]() -> Result {
        const uint32_t vl = g_vl_32;
        uint16_t vt32 = vtype_for_sew(2);
        auto f = [&]() {
            vadd_vv(g_field, vt32, 1, VD, VS1, VS2, 0, g_vlen, vl);
            vmul_vv(g_field, vt32, 1, VD, VS1, VS2, 0, g_vlen, vl);
            vfadd_vv(g_field, vt32, 1, VD_F, VS1_F, VS2_F, 0, g_vlen, vl, 0);
            vfmul_vv(g_field, vt32, 1, VD_F, VS1_F, VS2_F, 0, g_vlen, vl, 0);
            vfcvt_x_f_v(g_field, vt32, 1, VD, VS2_F, 0, g_vlen, vl, 0);
        };
        // count 5 ops per invocation for throughput density
        Result r = bench("mixed_integer_fp", f, g_field, g_field_bytes, vl * 5);
        r.ns_per_op /= 5.0;
        r.gops = 1000.0 / r.ns_per_op;
        r.ns_per_elem = r.ns_per_op / static_cast<double>(vl);
        return r;
    }});
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
static void print_header()
{
    std::printf("%-26s %12s %12s %12s %12s\n", "case", "ns/op", "Gop/s", "ns/elem", "checksum");
    std::printf("%-26s %12s %12s %12s %12s\n", "----", "-----", "-----", "-------", "--------");
}

static void print_result(const Result &r)
{
    std::printf("%-26s %12.2f %12.3f %12.3f %12llu\n", r.name.c_str(), r.ns_per_op, r.gops, r.ns_per_elem,
                static_cast<unsigned long long>(r.checksum));
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--smoke") { g_cfg.smoke = true; }
        else if (a == "--filter" && i + 1 < argc) { g_cfg.filter = argv[++i]; }
        else if (a == "--vlen" && i + 1 < argc) { g_cfg.vlen_bits = static_cast<uint32_t>(std::stoul(argv[++i])); }
        else if (a == "--reps" && i + 1 < argc) { g_cfg.reps_per_case = static_cast<uint32_t>(std::stoul(argv[++i])); }
        else if (a == "--help")
        {
            std::printf("Usage: %s [--smoke] [--filter sub] [--vlen bits] [--reps N]\n", argv[0]);
            return 0;
        }
        else { std::fprintf(stderr, "unknown arg '%s'\n", a.c_str()); return 1; }
    }

    VecRegFile vrf(g_cfg.vlen_bits);
    g_field = vrf.data();
    g_field_bytes = vrf.mem.size();
    g_vlen = g_cfg.vlen_bits;
    g_vl_8 = (g_cfg.vlen_bits / 8) ? (g_cfg.vlen_bits / 8) : 1;
    g_vl_16 = (g_cfg.vlen_bits / 16) ? (g_cfg.vlen_bits / 16) : 1;
    g_vl_32 = (g_cfg.vlen_bits / 32) ? (g_cfg.vlen_bits / 32) : 1;
    g_vl_64 = (g_cfg.vlen_bits / 64) ? (g_cfg.vlen_bits / 64) : 1;

    register_cases();

    print_header();
    for (auto &c : g_cases)
    {
        if (!g_cfg.filter.empty() && c.name.find(g_cfg.filter) == std::string::npos) continue;
        Result r;
        try { r = c.run(); }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "case %s failed: %s\n", c.name.c_str(), e.what());
            continue;
        }
        print_result(r);
    }
    return 0;
}
