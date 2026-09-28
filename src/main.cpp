#include "covq.h"
#include "stb_image.h"

#include <ev2/editor.h>
#include <ev2/viewport.h>
#include <ev2/motion_camera.h>
#include <ev2/pipeline.h>
#include <ev2/image_viewer.h>
#include <ev2/utils/log.h>

#include <Eigen/Dense>

#include <glob.h>

#include <cstdio>
#include <cstring>
#include <span>
#include <bit>
#include <thread>
#include <algorithm>
#include <condition_variable>

void c_deleter(void *p)
{
	free(p);
}

enum ImageFormat
{
	IMAGE_FORMAT_RGBA8
};

struct Image
{
	std::string path;
	std::unique_ptr<uint8_t[], decltype(&c_deleter)> bytes {nullptr, &c_deleter};
	uint32_t w;
	uint32_t h;
	ImageFormat format;

	constexpr size_t nbytes() const {return w * h * sizeof(uint32_t);}
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
		}
	}

	return P;
}

int load_image(const char *path, Image *out)
{
	int w, h, ch;
	stbi_uc *bytes = stbi_load(path, &w, &h, &ch, STBI_rgb_alpha);  

	if (bytes) {
		*out = Image{
			.path = path,
			.bytes = {bytes, &c_deleter},
			.w = (uint32_t)w,
			.h = (uint32_t)h,
			.format = IMAGE_FORMAT_RGBA8,
		};
		printf("Loaded image: %s\n", path);
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

		size_t end = idx + size;

		for (; idx < end; ++idx) {
			// uses 8x more memory than what's needed...
			//
			// TODO: training can take a uint8 matrix? 
			X.col(idx) = Eigen::Vector3d(
				double(ptr[0])/255.0, 
				double(ptr[1])/255.0,
				double(ptr[2])/255.0
			); 

			ptr += 4;
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

struct COVQParams
{
	int N;
	double eps;
};

struct Visualizer;

struct COVQTrainingApp
{
	std::vector<Image> images;

	uint32_t num_points = 0;
	uint32_t dump_generation = 0;

	std::atomic_uint32_t completed_its = 0;
	std::atomic_uint32_t desired_its = 0;
	std::atomic_bool run_continuously = false;

	COVQParams quant;

	std::unique_ptr<COVQTrainingContext> trainer;

	Eigen::MatrixXd X;
	Eigen::MatrixXd P;

	std::mutex sync;
	std::thread training_thread;

	std::vector<uint32_t> mapping;
	Eigen::MatrixXd codepoints;

	enum PushConstantFlags {
		USE_HASH_COLORS_BIT = 0x1
	};

	std::unique_ptr<ev2::Viewport> viewport;
	std::shared_ptr<ev2::MotionCamera> camera;
	std::unique_ptr<COVQTrainingDump> dump;

	ev2::BufferID points_gpu {};
	ev2::BufferID mapping_gpu {};
	ev2::BufferID codepoints_gpu {};
	ev2::BufferID weights_gpu {};

	ev2::ImageID disp_image {};
	std::weak_ptr<ev2::ImageViewer> image_viewer {};

	struct Params {
		const Image *selected_image = nullptr;
		bool use_quantized = false;
		bool colorize_cells = false;
		float point_density = 1.f;
		float eps_slider = 0.05;
	} params;

	struct alignas(8) PushConstants {
		VkDeviceAddress points;
		VkDeviceAddress mapping;
		VkDeviceAddress codepoints;
		uint32_t count;
		float w = 0.004;
		int stride = 0;
		uint32_t flags;

		// index into the mapping buffer for the displayed image
		uint32_t image_start;
		uint32_t image_end;

		glm::ivec2 size;
	} pc = {};

	ev2::GfxContext *ctx;

	std::atomic_bool should_close = false;

	COVQTrainingApp();
	~COVQTrainingApp();
	int initialize(const COVQParams &in_params, std::vector<Image> && in_images);
	int update();

	int init_gpu_resources_for_data(const Eigen::MatrixXd &data);
	int init_gpu_resources_for_training(const COVQParams &covq);

	int set_codepoints(const Eigen::MatrixXd &codepoints);

	int view_image(const Image &image);

	int reset_quantizer();

	void render_points();
	void render_image();
};

struct Visualizer
{
	COVQTrainingApp *app;

};

COVQTrainingApp::COVQTrainingApp()
{
	ctx = ev2::editor::ctx();

	dump = std::make_unique<COVQTrainingDump>();
	viewport.reset(new ev2::Viewport("Visualization", 0, 0, 500, 500,
		ev2::RENDER_TARGET_CREATE_DEPTH_BIT | ev2::RENDER_TARGET_CREATE_COLOR_BIT));
	camera.reset(new ev2::MotionCamera(glm::dvec3(0.5,0.5,0.5), glm::dvec3(-0.5,-0.5,1), glm::dvec3(0,0,1)));

	viewport->set_camera(camera);
	viewport->set_closable(false);
}

COVQTrainingApp::~COVQTrainingApp()
{
	should_close = true;
	++desired_its;
	desired_its.notify_all();

	if (training_thread.joinable())
		training_thread.join();
}

int COVQTrainingApp::reset_quantizer()
{
	P = bsc_matrix(quant.N, quant.eps);
	{
		std::unique_lock<std::mutex> lock(sync);
		trainer.reset(covq_init(X, P));
	}

	init_gpu_resources_for_training(quant);

	desired_its.store(0);
	completed_its.store(0);

	desired_its.notify_one();

	return 0;
}

int COVQTrainingApp::initialize(const COVQParams &in_params, std::vector<Image> && in_images)
{
	quant = in_params;
	images = std::move(in_images);

	params.eps_slider = quant.eps;

	X = images_to_matrix(images);

	TIMER_BEGIN(Uploading);
	init_gpu_resources_for_data(X);
	TIMER_END(Uploading);

	reset_quantizer();

	training_thread = std::thread([this](){
		while (!should_close) {
			uint32_t completed = completed_its.load(); 
			uint32_t desired = desired_its.load(); 

			if (completed <= desired) {
				std::unique_lock<std::mutex> lock(sync);
				uint32_t it = completed_its++;
				printf("Iteration: %d\n", it);
				covq_training_it(*trainer, X, dump.get());
			} else{
				printf("Next iteration: %d\n", desired + 1);
				desired_its.wait(desired, std::memory_order_seq_cst);
			}
		}
	});

	return 0;
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

void gpu_buffer_zero(ev2::BufferID buf, size_t size)
{
	ev2::GfxContext *ctx = ev2::editor::ctx();
	ev2::UploadContext uc = ev2::begin_upload(ctx, size, alignof(uint32_t));
	memset(uc.ptr, 0x0, size); 
	ev2::BufferUpload up = {.size = size};
	ev2::commit_buffer_uploads(ctx, uc, buf, &up, 1);
	ev2::flush_uploads(ctx);
}

int COVQTrainingApp::view_image(const Image &in_image)
{
	size_t bytes = in_image.nbytes();

	if (disp_image.is_valid()) {
		ev2::destroy_image(ctx, disp_image);
	}
	disp_image = ev2::create_image(ctx, in_image.w, in_image.h, 1, ev2::IMAGE_FORMAT_RGBA8,
		ev2::IMAGE_USAGE_SAMPLED_BIT);

	ev2::UploadContext uc = ev2::begin_upload(ctx, bytes, alignof(uint32_t));
	memcpy(uc.ptr, in_image.bytes.get(), bytes); 
	ev2::ImageUpload up = {
		.w = in_image.w,
		.h = in_image.h,
		.d = 1,
	};
	ev2::commit_image_uploads(ctx, uc, disp_image, &up, 1);
	ev2::flush_uploads(ctx);

	if (std::shared_ptr<ev2::ImageViewer> viewer = image_viewer.lock()) {
		ev2::editor::close_image_viewer(viewer);
	}

	image_viewer = ev2::editor::open_image_viewer(disp_image,
		"Test Image", "cap://pipeline/test_image.yaml", false);

	size_t offset = 0;
	for (const Image &image : images) {
		if (image.path == in_image.path)
			break;

		offset += image.w * image.h;
	}

	pc.image_start = offset;
	pc.image_end = offset + in_image.w * in_image.h;

	pc.size = glm::ivec2(
		in_image.w,
		in_image.h
	);
	return 0;
}

int COVQTrainingApp::set_codepoints(const Eigen::MatrixXd &codepoints)
{
	int N = codepoints.cols();
	size_t bytes = N * sizeof(uint32_t); 
	assert(codepoints.rows() == 3);

	ev2::UploadContext uc = ev2::begin_upload(ctx, bytes, alignof(glm::vec3));
	eigen_to_rgba8(codepoints, reinterpret_cast<uint32_t*>(uc.ptr));
	ev2::BufferUpload up = {
		.size = bytes,
	};
	ev2::commit_buffer_uploads(ctx, uc, codepoints_gpu, &up, 1);
	ev2::flush_uploads(ctx);

	return 0;
}

void COVQTrainingApp::render_points()
{
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
}

void COVQTrainingApp::render_image()
{
	std::shared_ptr<ev2::ImageViewer> viewer = image_viewer.lock();
	if (!viewer || !mapping_gpu.is_valid() || !codepoints_gpu.is_valid())
		return;

	ev2::GfxPassInfo pass_info = {
		.target = viewer->get_target(),
		.view = viewer->get_view(),
		.clear_color = true,
		.name = viewer->get_name(),
	};

	ev2::GfxPipelineID test_pipeline =
		ev2::load_graphics_pipeline(ctx, "cap://pipeline/test_image.yaml");

	ev2::PassID pass = ev2::begin_gfx_pass(ctx, &pass_info);
	ev2::cmd_use_buffer(pass, mapping_gpu, ev2::USAGE_STORAGE_READ_GRAPHICS);
	ev2::cmd_use_buffer(pass, codepoints_gpu, ev2::USAGE_STORAGE_READ_GRAPHICS);

	if (viewer->get_pipeline() == test_pipeline) {
		ev2::cmd_push_constant(pass, test_pipeline, 0, sizeof(pc), &pc);
	}
	viewer->record_draw(pass);

	ev2::end_pass(ctx, pass);
}

int COVQTrainingApp::update()
{
	int status = ev2::SUCCESS;
	if (viewport->imgui(nullptr) < ev2::SUCCESS)
		return status;

	if (image_viewer.expired()) {
		params.selected_image = nullptr;
	}

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

	ImGui::SliderFloat("-log(\%) Point Density", &params.point_density, 0.f, 4.f);

	double min_density = std::max(1.0/double(num_points), 1e-9);

	pc.stride = pow(10, params.point_density);

	if (ImGui::RadioButton("Run Continuosly", run_continuously)) {
		run_continuously = !run_continuously;
	}

	if (ImGui::RadioButton("Quantized colors", params.use_quantized)) {
		params.use_quantized = !params.use_quantized;
	}

	if (ImGui::RadioButton("Colorize cells", params.colorize_cells)) {
		params.colorize_cells = !params.colorize_cells;
	}

	if (run_continuously && desired_its <= completed_its) {
		++desired_its;
		desired_its.notify_one();
	}

	if (ImGui::SliderFloat("Epsilon", &params.eps_slider, 0.f, 1.f)) {
		quant.eps = params.eps_slider;
	}

	if (ImGui::Button("Reset quantizer")) {
		reset_quantizer();
	}

	char label[100];
	snprintf(label, sizeof(label), "Iteration: %d", desired_its.load()); 

	if (ImGui::Button(label)) {
		++desired_its;
		desired_its.notify_one();
	}

	const Image* new_image = params.selected_image;

	if (ImGui::CollapsingHeader("Image Picker")) {
		ImGui::Indent();
		uint32_t idx = 0;
		for (const Image &image : images) {
			if (ImGui::Selectable(image.path.c_str(), params.selected_image == new_image)) {
				new_image = &image;
			}
			++idx;
		}
		ImGui::Unindent();
	}

	if (new_image != params.selected_image) {
		view_image(*new_image);
		params.selected_image = new_image;
	}

	ImGui::End();

	if (!points_gpu.is_valid())
		return status;

	pc.flags = 
		USE_HASH_COLORS_BIT * params.use_quantized;

	render_points();
	render_image();

	return status;
}

int COVQTrainingApp::init_gpu_resources_for_training(const COVQParams &covq)
{
	if (codepoints_gpu.is_valid())
		ev2::destroy_buffer(ctx, codepoints_gpu);
	codepoints_gpu = ev2::create_buffer(ctx, covq.N * sizeof(uint32_t),
		ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);

	gpu_buffer_zero(codepoints_gpu, covq.N * sizeof(uint32_t));
	pc.codepoints = ev2::get_buffer_device_address(ctx, codepoints_gpu);

	return 0;
}

int COVQTrainingApp::init_gpu_resources_for_data(const Eigen::MatrixXd &data)
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
	mapping_gpu = ev2::create_buffer(ctx, points_size,
		ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);

	ev2::UploadContext uc = ev2::begin_upload(ctx, points_size, alignof(uint32_t));
	eigen_to_rgba8(data, reinterpret_cast<uint32_t*>(uc.ptr));
	ev2::BufferUpload up = {
		.size = points_size,
	};
	ev2::commit_buffer_uploads(ctx, uc, points_gpu, &up, 1);
	ev2::flush_uploads(ctx);

	gpu_buffer_zero(mapping_gpu, points_size);

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
	if ((status = ev2::editor::init(argc, argv, "test", 1200, 800)); status != ev2::SUCCESS) {
		return status;
	}
	add_project_mounts(ev2::editor::ctx());

	uint32_t jobs = 1;

	COVQParams params = {
		.N = (1 << 8),
		.eps = 0.05
	};

	std::vector<Image> images;

	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "-j") && i + 1 < argc) {
			jobs = std::min(
				(uint32_t)strtol(argv[++i], NULL, 10),
				std::thread::hardware_concurrency()
			);
		}
		else if (!strcmp(argv[i], "-N") && i + 1 < argc) {
			params.N = (uint32_t)strtol(argv[++i], NULL, 10);
		}
		else if (!strcmp(argv[i], "-eps") && i + 1 < argc) {
			params.eps = strtod(argv[++i], NULL);
		}
		else {
			const char *path = argv[i];
			glob_t g;
			status = glob(path, 0, NULL, &g);

			if (status == GLOB_NOMATCH) {
			} else if (status != 0) {
				fprintf(stderr, "Failed to glob %s", path);
			}  else {
				for (size_t j = 0; j < g.gl_pathc; ++j) {
					Image img;
					int result = load_image(g.gl_pathv[j], &img);

					if (result == 0) {
						images.push_back(std::move(img));
					}
				}
			}
		}
	}

	if (images.empty()) {
		printf("No images passed\n");
		return 0;
	}

	Eigen::setNbThreads(jobs);
	printf("Using %d threads for Eigen\n", jobs);

	std::unique_ptr<COVQTrainingApp> app = std::make_unique<COVQTrainingApp>();

	app->initialize(params, std::move(images));

	for (;;) {
		if (status = ev2::editor::begin_frame(); status != ev2::SUCCESS)
			break;

		if (status = app->update(); status < 0)
			break;

		if (status = ev2::editor::end_frame(); status != ev2::SUCCESS)
			break;
	}

	app.reset(nullptr);

	ev2::editor::shutdown();

	return status;
}

