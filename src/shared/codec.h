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

// Bits per coefficient, row-major: entry [v*8 + u] is coefficient (u, v),
// same layout as the bit grid in the UI.
struct BitTables {
	static constexpr uint8_t dc_only[64] = {
		8, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	};

	static constexpr uint8_t full[64] = {
		8, 8, 8, 8, 8, 8, 8, 8,
		8, 8, 8, 8, 8, 8, 8, 8,
		8, 8, 8, 8, 8, 8, 8, 8,
		8, 8, 8, 8, 8, 8, 8, 8,
		8, 8, 8, 8, 8, 8, 8, 8,
		8, 8, 8, 8, 8, 8, 8, 8,
		8, 8, 8, 8, 8, 8, 8, 8,
		8, 8, 8, 8, 8, 8, 8, 8,
	};
};


#endif
