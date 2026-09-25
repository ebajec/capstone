#include "covq.h"
#include "stb_image.h"

#include <Eigen/Eigen>

#include <cstdio>
#include <span>
#include <bit>
#include <thread>
#include <algorithm>

void c_deleter(void *p)
{
	free(p);
}

enum ImageFormat
{
	IMAGE_FORMAT_RGB8
};

struct Image
{
	std::unique_ptr<uint8_t[], decltype(&c_deleter)> bytes {nullptr, &c_deleter};
	uint32_t w;
	uint32_t h;
	ImageFormat format;

};

Eigen::MatrixXd bsc_matrix(uint32_t N, double eps)
{
	Eigen::MatrixXd P (N, N);

	uint32_t k = std::bit_width(N);

	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < N; ++j) {
			uint32_t d = i ^ j;
			uint32_t ones = std::countl_one(d);

			double p_j_i = 1.0; 

			for (uint32_t i = 0; i < ones; ++i)
				p_j_i *= eps;
			for (uint32_t i = 0; i < k - ones; ++i)
				p_j_i *= (1.0 - eps);

			P(i, j) = p_j_i;
		}
	}

	return P;
}

int load_image(const char *path, Image *out)
{
	int w, h, ch;
	stbi_uc *bytes = stbi_load(path, &w, &h, &ch, STBI_rgb);  

	if (bytes) {
		*out = Image{
			.bytes = {bytes, &c_deleter},
			.w = (uint32_t)w,
			.h = (uint32_t)h,
			.format = IMAGE_FORMAT_RGB8,
		};
		return 0;
	}

	return -1;
}

Eigen::MatrixXd images_to_matrix(std::span<const Image> images)
{
	constexpr int dim = 3;
	size_t num_pix = 0;

	for (const Image &img : images) {
		num_pix += img.w * img.h;
		assert(img.format == IMAGE_FORMAT_RGB8);
	}

	Eigen::MatrixXd X (dim, num_pix);

	size_t offset = 0;
	for (const Image &img : images) {
		uint8_t *ptr = img.bytes.get();

		for (uint32_t i = 0; i < img.h; ++i) {
			for (uint32_t j = 0; j < img.w; ++j) {

				// uses 8x more memory than what's needed...
				//
				// TODO: training can take a uint8 matrix? 
				X.col(offset + i * img.w + j) =	Eigen::Vector3d(
					double(ptr[0])/255.0, 
					double(ptr[1])/255.0,
					double(ptr[2])/255.0
				); 

				ptr += 3;
			}
		}

		offset += img.w * img.h;
	}

	return X;
}

int main(int argc, char *argv[])
{
	uint32_t N = (1 << 8);
	uint32_t jobs = 1;

	std::vector<Image> images;

	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "-j") && i + 1 < argc) {
			jobs = std::min(
				(uint32_t)strtol(argv[++i], NULL, 10),
				std::thread::hardware_concurrency()
			);
		}
		else if (!strcmp(argv[i], "-N") && i + 1 < argc) {
			N = (uint32_t)strtol(argv[++i], NULL, 10);
		}
		else {
			Image img;
			int result = load_image(argv[i], &img);

			if (result == 0) {
				images.push_back(std::move(img));
			}
		}
	}

	if (images.empty()) {
		printf("No images passed\n");
		return 0;
	}

	Eigen::setNbThreads(jobs);

	constexpr double eps = 0.00;

	Eigen::MatrixXd X = images_to_matrix(images);
	Eigen::MatrixXd P = bsc_matrix(N, eps);

	COVQBook *book = covq_train(X, P, 10);

	return 0;
}

