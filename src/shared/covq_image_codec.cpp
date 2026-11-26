#include "covq_image_codec.h"
#include "dct8.hpp"
#include "timers.h"

#include <cassert>
#include <format>

// @brief map an input vector to an index to send over the channel
static constexpr uint32_t covq_encode(const COVQCodec *codec, const double *in)
{
	const int partitions = codec->encoder.E.cols();
	const int dim = codec->dim;

	const Eigen::MatrixXd &E = codec->encoder.E;
	const Eigen::VectorXd &v = codec->encoder.v;

	uint32_t argmin = 0;
	double d_min = DBL_MAX;

	for (uint32_t i = 0; i < partitions; ++i) {
		double Ex_i = 0.0;
		for (uint32_t j = 0; j < dim; ++j)
			Ex_i += E(j, i) * in[j];
		double d = v(i) - 2.0 * Ex_i;

		if (d < d_min) {
			argmin = i;
			d_min = d;
		}
	}

	return codec->encoder.index_map[argmin];
}

// @brief map a channel output to a reconstruction vector.
static constexpr void covq_decode(const COVQCodec *codec, uint32_t in, double *out)
{
	const double *src = codec->codebook.col(in).data(); 
	std::copy(src, src + codec->dim, out);
}


//------------------------------------------------------------------------------
// DCT based codec

// @return compressed size (in bytes) of an image with the specified dimensions
size_t COVQDCTCodec::get_compressed_size(uint32_t w, uint32_t h, uint32_t ch) const
{
	size_t blk_size = 0;
	for (int i = 0; i < 64; ++i)
		if (quantizers[i])
			blk_size += quantizers[i]->word_length;

	// word lengths are in bits; round the total up to whole bytes
	const size_t total_blks = ((w + 7) >> 3) * ((h + 7) >> 3);
	return (blk_size * total_blks + 7) / 8;
}

void COVQDCTCodec::encode(uint32_t w, uint32_t h, uint32_t ch,
				 const uint8_t *in_bytes, uint8_t *out_bytes) const
{
	float coeffs[MAX_CHANNELS][64];
	float blk[64];

	size_t bit_idx = 0;

	uint8_t lengths[64];
	for (int i = 0; i < 64; ++i) {
		if (quantizers[i])
	  		lengths[i] = quantizers[i]->word_length;
	}

	for (uint32_t by = 0; by < h; by += 8) {
		for (uint32_t bx = 0; bx < w; bx += 8) {
			// go over channels within the block

			for (uint32_t c = 0; c < dim; ++c) {

				// strided reads for each channel; repeat the edge pixels
				for (uint32_t iy = 0; iy < 8; ++iy) {
					const size_t src_y = std::min(by + iy, h - 1);
					for (uint32_t ix = 0; ix < 8; ++ix) {
						const size_t src_x = std::min(bx + ix, w - 1);
						blk[iy * 8 + ix] = dct8::unpack_u8(in_bytes[(src_y * w + src_x) * ch + c]);
					}
				}
				dct8::forward(blk, coeffs[c]);
			}
			for (int i = 0; i < 64; ++i) {
				if (!quantizers[i])
					continue;

				double X[MAX_CHANNELS] {};
				for (uint32_t c = 0; c < dim; ++c)
					X[c] = coeffs[c][i];

				uint32_t w = covq_encode(quantizers[i].get(), X);
				bit_idx = write_bits(w, lengths[i], out_bytes, bit_idx);
			}
		}
	}
}

void COVQDCTCodec::decode(uint32_t w, uint32_t h, uint32_t ch,
				 const uint8_t *in_bytes, uint8_t *out_bytes) const
{
	uint8_t lengths[64];
	for (int i = 0; i < 64; ++i) {
		if (quantizers[i])
	  		lengths[i] = quantizers[i]->word_length;
	}

	// bins without a quantizer are never written below, so they must start at 0
	float coeffs[MAX_CHANNELS][64] {};
	float blk[64];

	size_t bit_idx = 0;
	for (uint32_t by = 0; by < h; by += 8) {
		for (uint32_t bx = 0; bx < w; bx += 8) {
			// Crop the padding
			const uint32_t lw = std::min(8u, w - bx);
			const uint32_t lh = std::min(8u, h - by);

			for (int i = 0; i < 64; ++i) {
				if (!quantizers[i])
					continue;

				uint32_t w;
				bit_idx = read_bits(w, lengths[i], in_bytes, bit_idx); 
				double Y[MAX_CHANNELS] {};
				covq_decode(quantizers[i].get(), w, Y);

				for (int c = 0; c < dim; ++c) {
					coeffs[c][i] = Y[c];
				}
			}

			// go over all channels within the block
			for (uint32_t c = 0; c < dim; ++c) {
				dct8::inverse(coeffs[c], blk);

				for (uint32_t iy = 0; iy < lh; ++iy) {
					const size_t dst_y = by + iy;
					for (uint32_t ix = 0; ix < lw; ++ix) {
						const size_t dst_x = bx + ix;

						out_bytes[(dst_y * w + dst_x) * ch + c] = dct8::pack_u8(blk[iy * 8 + ix]);
					}
				}
			}

			// channels beyond what's compressed set to 1. 
			for (uint32_t c = dim; c < ch; ++c) {
				for (uint32_t iy = 0; iy < lh; ++iy) {
					const size_t dst_y = by + iy;
					for (uint32_t ix = 0; ix < lw; ++ix) {
						const size_t dst_x = bx + ix;
						out_bytes[(dst_y * w + dst_x) * ch + c] = UINT8_MAX;
					}
				}
			}
		}
	}
}

//------------------------------------------------------------------------------
// Codec training

COVQDCTCodecTrainer *COVQDCTCodecTrainer::create(
	const Eigen::MatrixXd X[64],
	const uint8_t bit_table[64],
	const std::function<Eigen::MatrixXd(int k/*bits*/)> matrix_generator
)
{
	COVQDCTCodecTrainer *trainer = new COVQDCTCodecTrainer{};
	trainer->dim = X[0].rows();

	for (int i = 0; i < 64; ++i) {
		if (!bit_table[i]) {
			trainer->uncoded_distortion += X[i].colwise().squaredNorm().mean();
			continue;
		}

		Eigen::MatrixXd P = matrix_generator(bit_table[i]);
		trainer->ctx[i].reset(COVQTrainingContext::create(X[i], P));
	}

	return trainer;
}

void COVQDCTCodecTrainer::train(const Eigen::MatrixXd X[64])
{
	TIMER_BEGIN(Training);
	// per-block pixel error is the sum of the bin errors by parseval theorem
	double sum = uncoded_distortion;
	for (int i = 0; i < 64; ++i) {
		if (ctx[i]) {
			ctx[i]->train(X[i]);
			sum += ctx[i]->expected_distortion;
		}
	}
	TIMER_END(Training);

	expected_distortion = sum / (64.0 * dim);
	++epoch;
}

std::unique_ptr<COVQDCTCodec> COVQDCTCodecTrainer::create_codec(const char *name)
{
	uint32_t bits_per_blk = 0;

	COVQDCTCodec *codec = new COVQDCTCodec{};
	for (int i = 0; i < 64; ++i) {
		if (ctx[i]) {
			codec->quantizers[i].reset(ctx[i]->create_codec());
			codec->dim = ctx[i]->dim;
			bits_per_blk += ctx[i]->word_length;
		}
	}

	codec->name = std::format("{}_bpp{:.2f}_epoch{}", name, (float)bits_per_blk/64.f, epoch); 

	return std::unique_ptr<COVQDCTCodec>(codec);
}

int COVQDCTCodecTrainer::generate_data(
	int dim,
	uint32_t count,
	const Image *images,
	Eigen::MatrixXd X[64]
)
{
	// When counting and processing 8x8 blocks for training, we omit the edges.

	uint64_t total_blks = 0;
	for (uint32_t i = 0; i < count; ++i) {
		const Image &image = images[i];
		assert(image.ch >= dim);

		total_blks += (image.w >> 3) * (image.h >> 3);
	}

	for (int i = 0; i < 64; ++i) {
		X[i] = Eigen::MatrixXd(dim, total_blks); 
	}

	float blk[64];
	float coeffs[64];

	uint64_t blk_idx = 0;

	for (uint32_t i = 0; i < count; ++i) {
		const Image &image = images[i];
	  	const uint8_t *bytes = image.bytes.get();
	  	const int ch = image.ch;
	  	const int w = image.w;
	  	const int h = image.h;

		if (ch < dim) {
			fprintf(stderr, "Cannot generate data of dimension %d from image with %d channels", dim, ch);
			return -1;
		}

		for (uint32_t by = 0; by + 8< image.h; by += 8) {
			for (uint32_t bx = 0; bx + 8 < image.w; bx += 8) {
				for (int c = 0; c < dim; ++c) {
					for (int y = 0; y < 8; ++y) {
						for (int x = 0; x < 8; ++x) {
							uint8_t src = bytes[c + ch * ((by + y) * w + (bx + x))];
							blk[y * 8 + x] = dct8::unpack_u8(src);  
						}
					}

					dct8::forward(blk, coeffs);

					// TODO: Profile this approach to writing out bins
					for (int y = 0; y < 8; ++y) {
						for (int x = 0; x < 8; ++x) {
							X[y * 8 + x](c, blk_idx) = coeffs[y * 8 + x]; 
		  				}
					}
				}
	  			++blk_idx;
			}
		}
	}

	assert(blk_idx <= total_blks);
	return 0;
}

//------------------------------------------------------------------------------
// Test codec (vq on pixels)

// @return compressed size (in bytes) of an image with the specified dimensions
size_t COVQTestImageCodec::get_compressed_size(uint32_t w, uint32_t h, uint32_t ch) const
{
	return w * h * ((quantizer->word_length + 7) >> 3);
}

void COVQTestImageCodec::encode(uint32_t w, uint32_t h, uint32_t ch,
				 const uint8_t *in_bytes, uint8_t *out_bytes) const
{
	assert(ch >= quantizer->dim);

	const int word_len = quantizer->word_length;

	size_t bit_offset = 0;

	double X[MAX_CHANNELS] {};
	for (uint32_t i = 0; i < ch * w * h; i += ch) {
		const uint8_t *pix = &in_bytes[i];

		for (int j = 0; j < ch; ++j)
			X[j] = (double)pix[j]/255.0;

		uint32_t w = covq_encode(quantizer.get(), X);
		bit_offset = write_bits(w, word_len, out_bytes, bit_offset);
	}
}
void COVQTestImageCodec::decode(uint32_t w, uint32_t h, uint32_t ch,
				 const uint8_t *in_bytes, uint8_t *out_bytes) const
{
	assert(ch >= quantizer->dim);

	const int word_len = quantizer->word_length;

	size_t bit_offset = 0;

	// this is a hack for rgba8.
	double Y[MAX_CHANNELS] {UINT8_MAX};

	for (uint32_t i = 0; i < ch * w * h; i += ch) {
		uint32_t w = 0;
		bit_offset = read_bits(w, word_len, in_bytes, bit_offset);

		covq_decode(quantizer.get(), w, Y);

		uint8_t *pix = &out_bytes[i];

		for (int j = 0; j < ch; ++j)
			pix[j] = (uint8_t)(std::clamp(Y[j], 0.0, 1.0) * 255.0);

		//hack
		pix[3] = UINT8_MAX;
	}
}

