#ifndef COVQ_IMAGE_CODEC_H
#define COVQ_IMAGE_CODEC_H

#include "codec.h"
#include "covq.h"

#include <cstddef>
#include <cstdint>

// We'll only consider rgba in the most extreme case
static constexpr int MAX_CHANNELS = 4;

struct Image
{
	static void c_deleter(void *p)
	{
		free(p);
	}

	static void null_deleter(void *p)
	{
	}

	typedef void(*FreeFn)(void *);
	typedef std::unique_ptr<uint8_t[], FreeFn> BytesPtr;

	BytesPtr bytes {nullptr, &null_deleter};
	uint32_t w;
	uint32_t h;
	uint32_t ch;

	constexpr size_t nbytes() const {return w * h * ch;}
};

// @breif vector quantization on each DCT bin when multiple channels are present
struct COVQDCTCodec : IBlockImageCodec
{
	std::string name;
	std::unique_ptr<COVQCodec> quantizers[64];
	int dim;

	// @return compressed size (in bytes) of an image with the specified dimensions
	virtual size_t get_compressed_size(uint32_t w, uint32_t h, uint32_t ch) const override;

	virtual void encode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const final;
	virtual void decode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const final;
	virtual const char* get_name() const final {return name.c_str();}
};

struct COVQDCTCodecTrainer
{
	std::unique_ptr<COVQTrainingContext> ctx[64];
	// MSE per pixel per channel (0..255 scale), including uncoded bins
	double expected_distortion;
	// summed energy of the bins with no quantizer; they always decode to 0
	double uncoded_distortion;
	uint32_t dim;
	uint64_t epoch;

	// @param dim - number of image channels to handle
	// @param matrix_generator - callable object for generating the channel
	// transition matrix with k bits
	// @param bit_table - bit allocation table
	static COVQDCTCodecTrainer *create(
		const Eigen::MatrixXd X[64],
		const uint8_t bit_table[64],
		const std::function<Eigen::MatrixXd(int k/*bits*/)> matrix_generator
	);

	void train(const Eigen::MatrixXd X[64]);
	std::unique_ptr<COVQDCTCodec> create_codec(const char *name);

	static int generate_data(
		int dim,
		uint32_t count,
		const Image *images,
		Eigen::MatrixXd X[64]
	);
};

// Vector quantization on pixels
struct COVQTestImageCodec : IBlockImageCodec
{
	std::unique_ptr<COVQCodec> quantizer;

	COVQTestImageCodec(COVQCodec *in_quantizer) : quantizer(in_quantizer) {
	}

	// @return compressed size (in bytes) of an image with the specified dimensions
	virtual size_t get_compressed_size(uint32_t w, uint32_t h, uint32_t ch) const override;

	virtual void encode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const override;
	virtual void decode(uint32_t w, uint32_t h, uint32_t ch,
					 const uint8_t *in_bytes, uint8_t *out_bytes) const override;
};
#endif //COVQ_IMAGE_CODEC_H
