#include "uniform_quantizer.h"
#include "dct8.hpp"

#include <algorithm>

UniformQuantizer::UniformQuantizer(const uint8_t *in_bit_table, float in_range) : IBlockCodec() 
{
	range = in_range;
	memcpy(bit_table, in_bit_table, sizeof(bit_table));

	for (int i = 0; i < 64; ++i) {
		if (bit_table[i]) {
			scales[i] = float(1 << (bit_table[i] - 1)) / range;
		} else {
			scales[i] = 0;
		}

		// With an orthonormal 8x8 DCT, there is a factor of 8 in front (sqrt(64))
		scales[i] *= 0.125;
	}
}

uint32_t UniformQuantizer::bits_per_block() const
{
	uint32_t bits = 0;
	for (uint8_t b : bit_table) {
		bits += b;
	}
	return bits;
}

// @return the number of bits output by the quantizer
size_t UniformQuantizer::encode(
	const float *in_values,
	uint8_t *out_bytes,
	size_t start_bit
) const
{
	size_t idx = start_bit;
	for (uint32_t i = 0; i < 64; ++i) {
		const uint32_t k = dct8::zigzag[i];
		uint32_t n_bits = bit_table[k];
		if (!n_bits)
			continue;

		long N = (1 << (n_bits - 1));

		uint32_t word = static_cast<uint32_t>(
			std::clamp(lroundf(in_values[k] * scales[k]), -N, N - 1) + N
		);

		for (uint32_t j = 0; j < n_bits; ++j) {
			out_bytes[idx >> 3] |= ((word >> j) & 0x1) << (idx & 0x7);
			++idx;
		}
	}

	return idx - start_bit;
}

size_t UniformQuantizer::decode(
	const uint8_t *in_bytes,
	size_t start_bit,
	float *out_values
) const
{
	size_t idx = start_bit;
	for (uint32_t i = 0; i < 64; ++i) {
		const uint32_t k = dct8::zigzag[i];
		uint32_t n_bits = bit_table[k];
		if (!n_bits) {
			out_values[k] = 0;
			continue;
		}

		uint32_t word = 0;

		long N = (1 << (n_bits - 1));

		for (uint32_t j = 0; j < n_bits; ++j) {
			word |= uint32_t((in_bytes[idx >> 3] >> (idx & 0x7)) & 0x1) << j;
			++idx;
		}

		out_values[k] = static_cast<float>((long)word - N) / scales[k];
	}

	return idx - start_bit;
}
