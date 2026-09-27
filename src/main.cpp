#include "covq.h"
#include "stb_image.h"

#include <ev2/editor.h>
#include <ev2/viewport.h>
#include <ev2/motion_camera.h>
#include <ev2/pipeline.h>
#include <ev2/utils/log.h>

#include <Eigen/Dense>

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

	uint32_t k = std::bit_width(N) - 1;

	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < N; ++j) {
			uint32_t d = i ^ j;
			uint32_t ones = std::popcount(d);

			double p_j_i = 1.0; 

			for (uint32_t l = 0; l < ones; ++l)
				p_j_i *= eps;
			for (uint32_t l = 0; l < k - ones; ++l)
				p_j_i *= (1.0 - eps);

			P(i, j) = p_j_i;

			assert(p_j_i == 0 || i == j);
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
	} else {
		fprintf(stderr, "Failed to load image: %s\n", path);
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

	size_t idx = 0;
	for (const Image &img : images) {
		uint8_t *ptr = img.bytes.get();

		size_t size = img.w * img.h;

		for (; idx < size; ++idx) {
			// uses 8x more memory than what's needed...
			//
			// TODO: training can take a uint8 matrix? 
			X.col(idx) = Eigen::Vector3d(
				double(ptr[0])/255.0, 
				double(ptr[1])/255.0,
				double(ptr[2])/255.0
			); 

			ptr += 3;
		}
	}

	return X;
}

void eigen_to_rgba8(const Eigen::MatrixXd &data, uint32_t *out)
{
	const std::decay_t<decltype(data)>::Scalar *p = data.data();
	size_t base = 0;
	for (size_t i = 0; i < data.size(); i += 3) {
		uint32_t rgba = 0x0;
		for (int j = 0; j < 3; ++j) {
			uint32_t value = static_cast<uint32_t>(std::clamp(p[i + j] * 255.0, 0.0, 255.0));
			rgba |= value << (j << 3);
		}
		out[base++] = rgba;
	}
}

struct Visualizer
{
	std::unique_ptr<ev2::Viewport> viewport;
	std::shared_ptr<ev2::MotionCamera> camera;

	std::unique_ptr<COVQTrainingDump> dump;

	ev2::BufferID points_gpu {};
	ev2::BufferID mapping_gpu {};
	ev2::BufferID codepoints_gpu {};

	uint32_t num_points = 0;
	uint32_t dump_generation = 0;

	std::atomic_uint32_t iteration = 0;

	std::vector<uint32_t> mapping;
	Eigen::MatrixXd codepoints;

	struct PushConstants {
		VkDeviceAddress points;
		VkDeviceAddress mapping;
		VkDeviceAddress codepoints;
		uint32_t count;
		float w = 0.01;
		int stride = 0;
		alignas(8) glm::ivec2 size;
	} pc = {};

	ev2::GfxContext *ctx;

	std::atomic_bool should_close = false;

	Visualizer();
	int update();

	int set_data(const Eigen::MatrixXd &data);

	int set_codepoints(const Eigen::MatrixXd &codepoints);
};

Visualizer::Visualizer()
{
	ctx = ev2::editor::ctx();

	dump = std::make_unique<COVQTrainingDump>();
	viewport.reset(new ev2::Viewport("Visualization", 0, 0, 500, 500,
		ev2::RENDER_TARGET_CREATE_DEPTH_BIT | ev2::RENDER_TARGET_CREATE_COLOR_BIT));
	camera.reset(new ev2::MotionCamera(glm::dvec3(0,0,0), glm::dvec3(2,2,2), glm::dvec3(0,0,1)));

	viewport->set_camera(camera);
	viewport->set_closable(false);
}

void gpu_upload_single(ev2::BufferID buf, void *data, size_t size)
{
	ev2::GfxContext *ctx = ev2::editor::ctx();
	ev2::UploadContext uc = ev2::begin_upload(ctx, size, alignof(uint32_t));
	memcpy(uc.ptr, data, size); 
	ev2::BufferUpload up = {.size = size};
	ev2::commit_buffer_uploads(ctx, uc, buf, &up, 1);
	ev2::flush_uploads(ctx);
}

int Visualizer::set_codepoints(const Eigen::MatrixXd &codepoints)
{
	int N = codepoints.cols();
	size_t bytes = N * sizeof(uint32_t); 
	assert(codepoints.rows() == 3);

	if (codepoints_gpu.is_valid())
		ev2::destroy_buffer(ctx, codepoints_gpu);
	codepoints_gpu = ev2::create_buffer(ctx, bytes, ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);

	ev2::UploadContext uc = ev2::begin_upload(ctx, bytes, alignof(glm::vec3));
	eigen_to_rgba8(codepoints, reinterpret_cast<uint32_t*>(uc.ptr));
	ev2::BufferUpload up = {
		.size = bytes,
	};
	ev2::commit_buffer_uploads(ctx, uc, codepoints_gpu, &up, 1);
	ev2::flush_uploads(ctx);

	pc.codepoints = ev2::get_buffer_device_address(ctx, codepoints_gpu);

	return 0;
}

int Visualizer::update()
{
	int status = ev2::SUCCESS;
	if (viewport->imgui(nullptr) < ev2::SUCCESS)
		return status;

	if (dump && dump->generation > dump_generation) {
		{
			std::unique_lock<std::mutex> lock(dump->sync);
			mapping = std::move(dump->mapping);
			codepoints = std::move(dump->codepoints);
			dump_generation = dump->generation;
		}
		assert(!mapping.empty());
		gpu_upload_single(mapping_gpu, mapping.data(), mapping.size() * sizeof(uint32_t));
		set_codepoints(codepoints);
	}

	ImGui::Begin("Editor");
	ImGui::SliderFloat("Point Size", &pc.w, 0, 0.1);
	ImGui::SliderInt("Stride", &pc.stride, 0, 1000);
	if (ImGui::Button("Iteration")) {
		++iteration;
		iteration.notify_all();
	}
	ImGui::End();

	if (!points_gpu.is_valid())
		return status;

	ev2::GfxPassInfo pass_info = {
		.target = viewport->get_target(),
		.view = viewport->get_view(),
		.clear_color = true,
		.clear_depth = true,
		.name = "Visualization",
	};

	ev2::GfxPipelineID pipeline = ev2::load_graphics_pipeline(ctx, "cap://pipeline/training_viz.yaml");

	ev2::PassID pass = ev2::begin_gfx_pass(ctx, &pass_info);
	ev2::cmd_use_buffer(pass, points_gpu, ev2::USAGE_STORAGE_READ_GRAPHICS);
	ev2::cmd_use_buffer(pass, mapping_gpu, ev2::USAGE_STORAGE_READ_GRAPHICS);
	ev2::cmd_bind_gfx_pipeline(pass, pipeline);
	ev2::cmd_push_constant(pass, pipeline, 0, sizeof(pc), &pc);
	ev2::cmd_custom(pass, [count = pc.stride > 0 ? pc.count/pc.stride : pc.count](VkCommandBuffer cmds) {
		vkCmdDraw(cmds, 4, count, 0, 0);
	});
	ev2::end_pass(ctx, pass);

	return status;
}

int Visualizer::set_data(const Eigen::MatrixXd &data)
{
	Eigen::Index dim = data.rows();
	Eigen::Index count = data.cols();

	assert(dim == 3);

	size_t points_size = count * sizeof(uint32_t);

	if (points_gpu.is_valid())
		ev2::destroy_buffer(ctx, points_gpu);
	points_gpu = ev2::create_buffer(ctx, points_size,
		ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);

	if (mapping_gpu.is_valid())
		ev2::destroy_buffer(ctx, mapping_gpu);
	mapping_gpu = ev2::create_buffer(ctx, count * sizeof(uint32_t),
		ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);

	ev2::UploadContext uc = ev2::begin_upload(ctx, points_size, alignof(glm::vec3));
	eigen_to_rgba8(data, reinterpret_cast<uint32_t*>(uc.ptr));
	ev2::BufferUpload up = {
		.size = points_size,
	};
	ev2::commit_buffer_uploads(ctx, uc, points_gpu, &up, 1);
	ev2::flush_uploads(ctx);

	num_points = count;

	pc.count = count;
	pc.points = ev2::get_buffer_device_address(ctx, points_gpu);
	pc.mapping = ev2::get_buffer_device_address(ctx, mapping_gpu);

	return 0;
}

static inline int add_project_mounts(ev2::GfxContext *ctx)
{
	struct Mount
	{
		const char *name;
		const char *path;
	};

	int failures = 0;
#ifdef EV2_PROJECT_MOUNTS
	static constexpr Mount mounts[] = { EV2_PROJECT_MOUNTS };

	for (const Mount &mount : mounts) {
		if (ev2::add_mount(ctx, mount.name, mount.path) != ev2::SUCCESS) {
			log_warn("Failed to mount %s:// at %s", mount.name, mount.path);
			++failures;
		}
	}
#endif
	return failures;
}

int main(int argc, char *argv[])
{
	int status = 0;
	if ((status = ev2::editor::init(argc, argv, "test")); status != ev2::SUCCESS) {
		return status;
	}
	add_project_mounts(ev2::editor::ctx());

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
	printf("Using %d threads for Eigen\n", jobs);

	std::unique_ptr<Visualizer> visualizer = std::make_unique<Visualizer>();

	visualizer->pc.size = glm::ivec2(
		images[0].w,
		images[0].h
	);

	constexpr double eps = 0.00;

	Eigen::MatrixXd X = images_to_matrix(images);
	Eigen::MatrixXd P = bsc_matrix(N, eps);

	COVQTrainingContext *ctx = covq_init(X, P);

	TIMER_BEGIN(Uploading);
	visualizer->set_data(X);
	TIMER_END(Uploading);

	std::thread training_thread([&X, &P, ctx, &visualizer](){

		COVQTrainingDump *dump = visualizer->dump.get();

		uint32_t its = 0;

		while (!visualizer->should_close) {
			uint32_t value = visualizer->iteration.load(); 
			if (its++ < value) {
				covq_training_it(*ctx, X, P, 10, dump);
			} else if (value <= its) {
				visualizer->iteration.wait(1 + its);
			}
		}
	});

	for (;;) {
		if (status = ev2::editor::begin_frame(); status != ev2::SUCCESS)
			break;

		if (status = visualizer->update(); status < 0)
			break;

		if (status = ev2::editor::end_frame(); status != ev2::SUCCESS)
			break;
	}

	visualizer->should_close = true;

	training_thread.join();

	return status;
}

