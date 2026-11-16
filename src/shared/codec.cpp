#include "codec.h"
#include "dct8.hpp"

#include <cstring>
#include <algorithm>

// @return compressed size (in bytes) of an image with the specified dimensions
size_t DCTImageCodec::get_compressed_size(uint32_t w, uint32_t h, uint32_t ch) const
{
	const uint32_t total_blks = ((w + 7) >> 3) * ((h + 7) >> 3) * ch; 
	const size_t total_bytes = 
		(quantizer->bits_per_blk() * (size_t)total_blks + 7)/8;

	return total_bytes;
}

void DCTImageCodec::encode(uint32_t w, uint32_t h, uint32_t ch,
			const uint8_t *in_bytes, uint8_t *out_bytes) const
{
	float coeffs[64];
	float blk[64];

	size_t bit_idx = 0;

	for (uint32_t by = 0; by < h; by += 8) {
		for (uint32_t bx = 0; bx < w; bx += 8) {
			// go over channels within the block

			for (uint32_t c = 0; c < ch; ++c) {

				// strided reads for each channel; repeat the edge pixels
				for (uint32_t iy = 0; iy < 8; ++iy) {
					const size_t src_y = std::min(by + iy, h - 1);
					for (uint32_t ix = 0; ix < 8; ++ix) {
						const size_t src_x = std::min(bx + ix, w - 1);
						blk[iy * 8 + ix] = dct8::unpack_u8(in_bytes[(src_y * w + src_x) * ch + c]);
					}
				}
				dct8::forward(blk, coeffs);

				bit_idx += quantizer->encode_blk(coeffs, out_bytes, bit_idx);
			}
		}
	}
}

void DCTImageCodec::decode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const
{
	float coeffs[64];
	float blk[64];

	size_t bit_idx = 0;
	for (uint32_t by = 0; by < h; by += 8) {
		for (uint32_t bx = 0; bx < w; bx += 8) {
			// Crop the padding
			const uint32_t lw = std::min(8u, w - bx);
			const uint32_t lh = std::min(8u, h - by);

			// go over all channels within the block
			for (uint32_t c = 0; c < ch; ++c) {
				bit_idx += quantizer->decode_blk(in_bytes, bit_idx, coeffs);

				dct8::inverse(coeffs, blk);

				for (uint32_t iy = 0; iy < lh; ++iy) {
					const size_t dst_y = by + iy;
					for (uint32_t ix = 0; ix < lw; ++ix) {
						const size_t dst_x = bx + ix;

						out_bytes[(dst_y * w + dst_x) * ch + c] = dct8::pack_u8(blk[iy * 8 + ix]);
					}
				}
			}
		}
	}
}

UniformDCTQuantizer::UniformDCTQuantizer(const uint8_t *in_bit_table,
										 const float *in_ranges) :
	IDCTBlockCodec()
{
	memcpy(bit_table, in_bit_table, sizeof(bit_table));
	memcpy(ranges, in_ranges ? in_ranges : default_ranges, sizeof(ranges));

	for (int i = 0; i < 64; ++i) {
		if (bit_table[i]) {
			scales[i] = float(1 << (bit_table[i] - 1)) / ranges[i];
		} else {
			scales[i] = 0;
		}
	}
}

uint32_t UniformDCTQuantizer::bits_per_blk() const
{
	uint32_t bits = 0;
	for (uint8_t b : bit_table) {
		bits += b;
	}
	return bits;
}

// @return the number of bits output by the quantizer
size_t UniformDCTQuantizer::encode_blk(
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

		idx = write_bits(word, n_bits, out_bytes, idx);
	}

	return idx - start_bit;
}

size_t UniformDCTQuantizer::decode_blk(
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
		idx = read_bits(word, n_bits, in_bytes, idx);

		long N = (1 << (n_bits - 1));
		out_values[k] = static_cast<float>((long)word - N) / scales[k];
	}

	return idx - start_bit;
}

