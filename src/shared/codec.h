#ifndef CODEC_H
#define CODEC_H

#include <cstddef>
#include <cstdint>

struct IBlockCodec
{
	virtual uint32_t bits_per_block() const = 0;
	virtual size_t encode(
		const float *in_values,
		uint8_t *out_bytes,
		size_t start_bit
	) const = 0;

	virtual size_t decode(
		const uint8_t *in_bytes,
		size_t start_bit,
		float *out_values
	) const = 0;
};

#endif
