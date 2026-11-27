#ifndef CODEC_H
#define CODEC_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// Fixed length image compression interface
struct IBlockImageCodec
{
	virtual ~IBlockImageCodec() {}
	// @return compressed size (in bytes) of an image with the specified dimensions
	virtual size_t get_compressed_size(uint32_t w, uint32_t h, uint32_t ch) const = 0;

	virtual void encode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const = 0;
	virtual void decode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const = 0;

	virtual const char *get_name() const {return "[unnamed]";}
};

struct IDCTBlockCodec
{
	virtual uint32_t bits_per_blk() const = 0;

	virtual ~IDCTBlockCodec() {};

	// @brief encode an 8x8 block of DCT coefficients in zigzag order
	virtual size_t encode_blk(
		const float *in_values,
		uint8_t *out_bytes,
		size_t start_bit
	) const = 0;

	// @brief decode an 8x8 block of DCT coefficients in zigzag order
	virtual size_t decode_blk(
		const uint8_t *in_bytes,
		size_t start_bit,
		float *out_values
	) const = 0;

	virtual const char* get_name() const {return "[unnamed]";}
};

// Bits per coefficient, row-major starting from 0 at the top left
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

	// 16 bits/blk
	// 0.25 bits per grayscale pixel.
	static constexpr uint8_t bpp_0_25[64] = {
		8, 3, 1, 0, 0, 0, 0, 0,
		3, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	};

	// 32 bits/blk
	// 0.5 bits per grayscale pixel.
	static constexpr uint8_t bpp_0_5[64] = {
		8, 5, 3, 1, 0, 0, 0, 0,
		5, 4, 1, 0, 0, 0, 0, 0,
		3, 1, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	};

	// 64 bits/blk
	// 1 bit per grayscale pixel.
	static constexpr uint8_t bpp_1[64] = {
		8, 6, 5, 3, 2, 1, 0, 0,
		6, 5, 3, 2, 1, 0, 0, 0,
		5, 3, 2, 1, 0, 0, 0, 0,
		3, 2, 1, 1, 0, 0, 0, 0,
		2, 1, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	};

	// 128 bits/blk
	// 2 bit per grayscale pixel.
	static constexpr uint8_t bpp_2[64] = {
		8, 7, 6, 5, 4, 3, 2, 0,
		7, 6, 5, 4, 3, 2, 1, 0,
		6, 5, 4, 3, 3, 2, 0, 0,
		5, 4, 3, 3, 2, 1, 0, 0,
		4, 3, 3, 2, 1, 0, 0, 0,
		3, 2, 2, 1, 0, 0, 0, 0,
		2, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	};

	// 256 bits/blk
	// 4 bit per grayscale pixel.
	static constexpr uint8_t bpp_4[64] = {
		8, 8, 7, 6, 6, 5, 4, 4,
		8, 7, 6, 6, 5, 4, 4, 3,
		7, 6, 6, 5, 4, 4, 3, 3,
		6, 6, 5, 4, 4, 3, 3, 2,
		6, 5, 4, 4, 4, 3, 2, 2,
		5, 4, 4, 3, 3, 2, 2, 2,
		4, 4, 3, 3, 2, 2, 2, 1,
		4, 3, 3, 2, 2, 2, 1, 1,
	};

	// 512 bits/blk
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

// @return offset + nbits
static constexpr size_t write_bits(
	uint32_t word, int nbits, uint8_t *bytes, size_t offset)
{
	size_t first = offset >> 3;
	size_t nbytes = ((offset & 0x7) + nbits + 7) >> 3;

	bytes[first] &= ((1 << (offset & 0x7)) - 1);
	for (int k = 1; k < nbytes; ++k)
		bytes[first + k] = 0;

	for (uint32_t j = 0; j < nbits; ++j) {
		bytes[offset >> 3] |= ((word >> j) & 0x1) << (offset & 0x7);
		++offset;
	}
	return offset;
}

// @return offset + nbits
static constexpr size_t read_bits( uint32_t& word, int nbits, const uint8_t *bytes, size_t offset)
{
	word = 0;
	for (uint32_t j = 0; j < nbits; ++j) {
		word |= uint32_t((bytes[offset >> 3] >> (offset & 0x7)) & 0x1) << j;
		++offset;
	}
	return offset;
}

struct DCTImageCodec : public IBlockImageCodec
{
	std::unique_ptr<IDCTBlockCodec> quantizer;

	DCTImageCodec(IDCTBlockCodec *in_quantizer) : quantizer(in_quantizer) {}

	template<typename DCTBlockCodecT, typename... Args>
	static DCTImageCodec make(Args&&... args) {
		return DCTImageCodec(new DCTBlockCodecT(std::forward<Args>(args)...));
	}

	// @return compressed size (in bytes) of an image with the specified dimensions
	virtual size_t get_compressed_size(uint32_t w, uint32_t h, uint32_t ch) const override;

	virtual void encode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const override;
	virtual void decode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const override;
	virtual const char * get_name() const final {return quantizer->get_name();}
};

struct UniformDCTQuantizer : public IDCTBlockCodec
{
	static constexpr uint32_t max_bits = 16;

	std::string name;

	// Indexed by row-major coefficient index (see BitTables). Coefficients
	// are still written to the stream in zigzag order.
	uint8_t bit_table[64];
	// Precomputed scaling for DCT coefficients before rounding.
	float scales[64];

	// Clamping ranges for pre-scaled coefficients, including a factor of 8
	// for the DCT
	float ranges[64];

	// Rough guess. Includes scaling from the DCT.  
	static constexpr float default_ranges[64] = {
		1024, 320, 160, 107,  80,  64,  53,  46,
		 320, 226, 143, 101,  78,  63,  53,  45,
		 160, 143, 113,  89,  72,  59,  51,  44,
		 107, 101,  89,  75,  64,  55,  48,  42,
		  80,  78,  72,  64,  57,  50,  44,  40,
		  64,  63,  59,  55,  50,  45,  41,  37,
		  53,  53,  51,  48,  44,  41,  38,  35,
		  46,  45,  44,  42,  40,  37,  35,  32,
	};

	// @param in_ranges optional per-coefficient range table (row-major);
	// nullptr uses default_ranges
	UniformDCTQuantizer(const uint8_t *in_bit_table,
					 const float *in_ranges = nullptr);

	virtual size_t encode_blk(
		const float *in_values,
		uint8_t *out_bytes,
		size_t start_bit
	) const final;

	virtual size_t decode_blk(
		const uint8_t *in_bytes,
		size_t start_bit,
		float *out_values
	) const final;

	virtual uint32_t bits_per_blk() const final;

	virtual const char* get_name() const final {return name.c_str();}
};

#endif
