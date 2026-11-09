#ifndef UNIFORM_QUANTIZER_H
#define UNIFORM_QUANTIZER_H

#include "codec.h"

#include <cstdint>
#include <cstring>

struct UniformQuantizer : public IBlockCodec
{
	static constexpr uint32_t max_bits = 16;

	// Indexed by row-major coefficient index (see BitTables). Coefficients
	// are still written to the stream in zigzag order.
	uint8_t bit_table[64];
	float scales[64];
	float range;
	
	UniformQuantizer(const uint8_t *in_bit_table, float in_range);

	virtual size_t encode(
		const float *in_values,
		uint8_t *out_bytes,
		size_t start_bit
	) const override;

	virtual size_t decode(
		const uint8_t *in_bytes,
		size_t start_bit,
		float *out_values
	) const override;

	virtual uint32_t bits_per_block() const override;
};


#endif
