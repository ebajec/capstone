// dct8x8.hpp — small header-only 8x8 DCT-II / DCT-III (C++17)
//
// Orthonormal 2D DCT on 8x8 blocks (same scaling JPEG uses):
//
//   F(u,v) = c(u) c(v) Σx Σy f(x,y) cos((2x+1)uπ/16) cos((2y+1)vπ/16)
//   c(0) = sqrt(1/8), c(k>0) = 1/2
//
// forward() then inverse() is the identity up to float rounding, and the
// transform preserves energy (Parseval), so you can quantize directly.
//
// Layout: row-major, element (row y, col x) at p[y * stride + x].
// In-place calls (in == out) are fine.
//
//   #include "dct8x8.hpp"
//   float blk[64] = {...};
//   dct8::forward(blk, blk);     // spatial -> frequency
//   dct8::inverse(blk, blk);     // frequency -> spatial
//
//   // 8-bit image helpers (level shift by 128, round + clamp on the way back)
//   dct8::forward_u8(img + y*w + x, w, coeffs);
//   dct8::inverse_u8(coeffs, img + y*w + x, w);
//
// Uses a fast butterfly factorization (see detail::fdct1d): 272 multiplies
// per 2D block instead of 1024.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace dct8 {

namespace detail {

// Fast 1D 8-point DCT, after Loeffler, Ligtenberg & Moschytz (the "LL&M"
// factorization libjpeg's jfdctint/jidctint use), with constants rescaled so
// the output is orthonormal. 17 multiplies and 29 adds per 1D transform,
// vs. 64 MACs for the direct matrix product.
//
// With ck = cos(kπ/16):
template <class T>
struct K {
    static constexpr T dc     = T(0.35355339059327373);   // 1/(2√2)
    static constexpr T r6     = T(0.19134171618254492);   // c6/2
    static constexpr T r2m6   = T(0.27059805007309845);   // (c2-c6)/2
    static constexpr T r2p6   = T(0.65328148243818829);   // (c2+c6)/2
    static constexpr T o3     = T(0.41573480615127262);   // c3/2
    static constexpr T o_in0  = T(0.53079716883502259);   // ( c1+c3-c5-c7)/2
    static constexpr T o_in1  = T(1.0863674018546248);    // ( c1+c3+c5-c7)/2
    static constexpr T o_in2  = T(0.72588749085115101);   // ( c1+c3-c5+c7)/2
    static constexpr T o_in3  = T(0.10558212145139438);   // (-c1+c3+c5-c7)/2
    static constexpr T o_z1   = T(-0.31818964514320847);  // ( c7-c3)/2
    static constexpr T o_z2   = T(-0.90612744635288789);  // (-c1-c3)/2
    static constexpr T o_z3   = T(-0.69351992266107376);  // (-c3-c5)/2
    static constexpr T o_z4   = T(-0.13794968964147147);  // ( c5-c3)/2
};

// Odd half. The 4x4 odd-frequency matrix is its own transpose under this
// indexing, so the same graph serves both directions:
//   forward: in = (x0-x7, x1-x6, x2-x5, x3-x4)  ->  out = (X1, X3, X5, X7)
//   inverse: in = (X1, X3, X5, X7)               ->  out = odd parts of (x0, x1, x2, x3)
template <class T>
inline void odd4(T i0, T i1, T i2, T i3, T& o0, T& o1, T& o2, T& o3) {
    using k = K<T>;
    T z1 = i3 + i0, z2 = i2 + i1, z3 = i3 + i1, z4 = i2 + i0;
    const T z5 = (z3 + z4) * k::o3;
    z1 *= k::o_z1;
    z2 *= k::o_z2;
    z3 = z3 * k::o_z3 + z5;
    z4 = z4 * k::o_z4 + z5;
    o0 = i0 * k::o_in0 + z1 + z4;
    o1 = i1 * k::o_in1 + z2 + z3;
    o2 = i2 * k::o_in2 + z2 + z4;
    o3 = i3 * k::o_in3 + z1 + z3;
}

// 1D DCT-II on 8 elements spaced `is` apart, written `os` apart.
template <class T>
inline void fdct1d(const T* in, std::ptrdiff_t is, T* out, std::ptrdiff_t os) {
    using k = K<T>;
    const T x0 = in[0 * is], x1 = in[1 * is], x2 = in[2 * is], x3 = in[3 * is];
    const T x4 = in[4 * is], x5 = in[5 * is], x6 = in[6 * is], x7 = in[7 * is];

    // Even half: 4-point DCT of the symmetric sums.
    const T s0 = x0 + x7, s1 = x1 + x6, s2 = x2 + x5, s3 = x3 + x4;
    const T a0 = s0 + s3, a1 = s1 + s2, b0 = s0 - s3, b1 = s1 - s2;
    const T z = (b0 + b1) * k::r6;
    out[0 * os] = (a0 + a1) * k::dc;
    out[4 * os] = (a0 - a1) * k::dc;
    out[2 * os] = z + b0 * k::r2m6;
    out[6 * os] = z - b1 * k::r2p6;

    odd4(x0 - x7, x1 - x6, x2 - x5, x3 - x4,
         out[1 * os], out[3 * os], out[5 * os], out[7 * os]);
}

// 1D DCT-III (inverse of fdct1d), same stride convention.
template <class T>
inline void idct1d(const T* in, std::ptrdiff_t is, T* out, std::ptrdiff_t os) {
    using k = K<T>;
    const T X0 = in[0 * is], X2 = in[2 * is], X4 = in[4 * is], X6 = in[6 * is];

    // Even half.
    const T p = (X0 + X4) * k::dc, q = (X0 - X4) * k::dc;
    const T z = (X2 + X6) * k::r6;
    const T r0 = z + X2 * k::r2m6, r1 = z - X6 * k::r2p6;
    const T e0 = p + r0, e3 = p - r0, e1 = q + r1, e2 = q - r1;

    T o0, o1, o2, o3;
    odd4(in[1 * is], in[3 * is], in[5 * is], in[7 * is], o0, o1, o2, o3);

    out[0 * os] = e0 + o0;  out[7 * os] = e0 - o0;
    out[1 * os] = e1 + o1;  out[6 * os] = e1 - o1;
    out[2 * os] = e2 + o2;  out[5 * os] = e2 - o2;
    out[3 * os] = e3 + o3;  out[4 * os] = e3 - o3;
}

// Separable 2D transform: 1D pass over rows into tmp, then over columns.
// Reading all of `in` before writing `out` keeps in-place calls safe.
template <class T, bool Inverse>
inline void transform(const T* in, std::ptrdiff_t in_stride, T* out, std::ptrdiff_t out_stride) {
    T tmp[64];
    for (int y = 0; y < 8; ++y) {
        if (Inverse) idct1d(in + y * in_stride, 1, tmp + y * 8, 1);
        else         fdct1d(in + y * in_stride, 1, tmp + y * 8, 1);
    }
    for (int x = 0; x < 8; ++x) {
        if (Inverse) idct1d(tmp + x, 8, out + x, out_stride);
        else         fdct1d(tmp + x, 8, out + x, out_stride);
    }
}

}  // namespace detail

// ---- Core API (float or double) -------------------------------------------

// Forward 2D DCT-II: spatial block -> coefficients. out[0] is DC.
template <class T>
inline void forward(const T* in, T* out, std::ptrdiff_t in_stride = 8, std::ptrdiff_t out_stride = 8) {
    static_assert(std::is_floating_point_v<T>, "dct8: use float or double");
    detail::transform<T, false>(in, in_stride, out, out_stride);
}

// Inverse 2D DCT (DCT-III): coefficients -> spatial block.
template <class T>
inline void inverse(const T* in, T* out, std::ptrdiff_t in_stride = 8, std::ptrdiff_t out_stride = 8) {
    static_assert(std::is_floating_point_v<T>, "dct8: use float or double");
    detail::transform<T, true>(in, in_stride, out, out_stride);
}

template <class T>
using Block = std::array<T, 64>;

template <class T>
inline Block<T> forward(const Block<T>& in) { Block<T> o; forward(in.data(), o.data()); return o; }

template <class T>
inline Block<T> inverse(const Block<T>& in) { Block<T> o; inverse(in.data(), o.data()); return o; }

// ---- 8-bit image helpers ---------------------------------------------------

// Reads an 8x8 tile from an 8-bit image (stride in bytes), subtracts 128,
// and writes 64 coefficients.
template <class T = float>
inline void forward_u8(const std::uint8_t* src, std::ptrdiff_t stride, T* coeffs) {
    T blk[64];
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
			blk[y * 8 + x] = static_cast<T>(src[y * stride + x]) - T(128);
    forward(blk, coeffs);
}

// Inverse-transforms 64 coefficients, adds 128, rounds, clamps to [0,255].
template <class T = float>
inline void inverse_u8(const T* coeffs, std::uint8_t* dst, std::ptrdiff_t stride) {
    T blk[64];
    inverse(coeffs, blk);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            long v = std::lround(blk[y * 8 + x] + T(128));
            dst[y * stride + x] = static_cast<std::uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
        }
}

// ---- Zigzag order ----------------------------------------------------------

// zigzag[i] = row-major index of the i-th coefficient in JPEG zigzag order.
inline constexpr std::array<std::uint8_t, 64> zigzag = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
};

}  // namespace dct8
