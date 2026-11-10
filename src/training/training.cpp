#include "covq.h"
#include "stb_image.h"

#include <ev2/editor.h>
#include <ev2/viewport.h>
#include <ev2/motion_camera.h>
#include <ev2/core.h>
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

void c_deleter(void *p)
{
	free(p);
}

struct Image
{
	std::string path;
	std::unique_ptr<uint8_t[], decltype(&c_deleter)> bytes {nullptr, &c_deleter};
	uint32_t w;
	uint32_t h;
	uint32_t ch;

	constexpr size_t nbytes() const {return w * h * ch;}
};

// @param k - number of codeword bits
Eigen::MatrixXd bsc_matrix(uint32_t k, double eps)
{
	uint32_t N = 1 << k;
	Eigen::MatrixXd P (N, N);

	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < N; ++j) {
			uint32_t d = i ^ j;
			uint32_t nb_diff = std::popcount(d);

			double p_j_i = 1.0; 

			for (uint32_t l = 0; l < nb_diff; ++l)
				p_j_i *= eps;
			for (uint32_t l = 0; l < k - nb_diff; ++l)
				p_j_i *= (1.0 - eps);

			P(i, j) = p_j_i;
		}
	}

	return P;
}

Eigen::MatrixXd nec_matrix(int k, double eps1, double eps2)
{
	uint32_t N = 1;
	for (int i = 0; i < k; ++i)
		N *= 3;

	uint32_t M = (1 << k);
	Eigen::MatrixXd P (M, N);
	Eigen::VectorXd test (M);

	for (uint32_t i = 0; i < M; ++i) {
		uint32_t ones = 0, twos = 0;

		double sum = 0;
		for (uint32_t j = 0; j < N; ++j) {
			int nb_diff = std::popcount(ones ^ (i & (~twos)));
			int nb_err = std::popcount(twos);

			uint32_t b = (twos + 1) & (~twos);
			twos &= twos + 1;
			twos |= ones & b;
			ones ^= b;

			double p_j_i = 1.0; 

			for (int l = 0; l < nb_diff; ++l)
				p_j_i *= eps1;
			for (int l = 0; l < nb_err; ++l)
				p_j_i *= eps2;
			for (int l = 0; l < k - nb_diff - nb_err; ++l)
				p_j_i *= (1.0 - eps1 - eps2);

			P(i, j) = p_j_i;
			sum += p_j_i;
		}
		test(i) = sum;
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
			.ch = 4,
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
		assert(img.ch == 4);
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

static void gpu_upload_single(ev2::BufferID buf, void *data, size_t size)
{
	ev2::GfxContext *ctx = ev2::editor::ctx();
	ev2::UploadContext uc = ev2::begin_upload(ctx, size, alignof(uint32_t));
	memcpy(uc.ptr, data, size); 
	ev2::BufferUpload up = {.size = size};
	ev2::commit_buffer_uploads(ctx, uc, buf, &up, 1);
	ev2::flush_uploads(ctx);
}

static void gpu_buffer_zero(ev2::BufferID buf, size_t size)
{
	ev2::GfxContext *ctx = ev2::editor::ctx();
	ev2::UploadContext uc = ev2::begin_upload(ctx, size, alignof(uint32_t));
	memset(uc.ptr, 0x0, size); 
	ev2::BufferUpload up = {.size = size};
	ev2::commit_buffer_uploads(ctx, uc, buf, &up, 1);
	ev2::flush_uploads(ctx);
}

struct COVQParams
{
	int nb_bits;
	double eps;
};

Eigen::MatrixXd transition_matrix(const COVQParams &params)
{
	return nec_matrix(params.nb_bits, params.eps, 0.5*params.eps);
}

// Renders the training dataset as a point cloud in RGB space, colored by the
// partition each point is currently encoded tos.
struct COVQTrainingVisualization
{
	// NOTE: must match point_viz.glsl
	enum ColorMode : uint32_t {
		// use point's value as the color
		COLOR_MODE_VALUE = 0,
		// a random color per partition cell
		COLOR_MODE_CELL_HASH = 1,
		// the reconstruction vector for the point's codeword
		COLOR_MODE_CODEPOINT = 2,
		COLOR_MODE_COUNT
	};

	// NOTE: must match point_viz.glsl
	enum FlagBits : uint32_t {
		JITTER_BIT = 0x1,
		SNAP_TO_CODEPOINT_BIT = 0x2,
	};

	// NOTE: must match point_viz.glsl
	struct alignas(8) PushConstants {
		VkDeviceAddress points;
		VkDeviceAddress codewords;
		VkDeviceAddress codepoints;
		uint32_t count;
		uint32_t stride;
		uint32_t num_codepoints;
		uint32_t color_mode;
		uint32_t flags;
		float w;
	};
	static_assert(sizeof(PushConstants) == 48);

	struct Options {
		float point_size = 0.004f;
		// log10 of the stride used to subsample the dataset
		float log_stride = 1.f;
		ColorMode color_mode = COLOR_MODE_CELL_HASH;
		// spread points over their 8 bit quantization bin
		bool jitter = true;
		// draw each point at its reconstruction vector
		bool snap_to_codepoint = false;
	} opts;

	ev2::GfxContext *gfx = nullptr;

	uint32_t num_points = 0;
	uint32_t num_codepoints = 0;

	struct GpuData {
		// dataset values (num_points)
		ev2::BufferID points;
		// channel output index each point is encoded to (num_points)
		ev2::BufferID codewords;
		// reconstruction vector for each channel output (num_codepoints)
		ev2::BufferID codepoints;
	} rd;

	std::unique_ptr<ev2::Viewport> viewport;
	std::shared_ptr<ev2::MotionCamera> camera;

	static constexpr uint64_t INVALID_EPOCH = UINT64_MAX;
	// training epoch that the GPU data was last synced with
	uint64_t synced_epoch = INVALID_EPOCH;

	// @param data - (3, count) dataset the trainer is run on
	static COVQTrainingVisualization *create(const Eigen::MatrixXd &data);
	~COVQTrainingVisualization();

	// force a re-sync on the next update, e.g. after the trainer was replaced
	void invalidate() { synced_epoch = INVALID_EPOCH; }

	// sync with the trainer if it has progressed, and render. Call once per frame.
	void update(COVQTrainingContext &trainer);

	// draw the display options into the current ImGui window
	void draw_gui();

	void sync(COVQTrainingContext &trainer);
	void render();

	uint32_t stride() const;
	uint32_t num_drawn_points() const;
};

COVQTrainingVisualization *COVQTrainingVisualization::create(const Eigen::MatrixXd &data)
{
	assert(data.rows() == 3);

	TIMER_BEGIN(VisualizationCreate);

	COVQTrainingVisualization *viz = new COVQTrainingVisualization{};
	ev2::GfxContext *gfx = viz->gfx = ev2::editor::ctx();

	viz->viewport.reset(new ev2::Viewport("Visualization", 0, 0, 500, 500,
		ev2::RENDER_TARGET_CREATE_DEPTH_BIT | ev2::RENDER_TARGET_CREATE_COLOR_BIT));
	viz->camera.reset(new ev2::MotionCamera(glm::dvec3(0.5,0.5,0.5), glm::dvec3(-0.5,-0.5,1), glm::dvec3(0,0,1)));

	viz->viewport->set_camera(viz->camera);
	viz->viewport->set_closable(false);

	viz->num_points = data.cols();
	size_t points_size = viz->num_points * sizeof(uint32_t);

	viz->rd.points = ev2::create_buffer(gfx, points_size,
		ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);
	viz->rd.codewords = ev2::create_buffer(gfx, points_size,
		ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);

	ev2::UploadContext uc = ev2::begin_upload(gfx, points_size, alignof(uint32_t));
	eigen_to_rgba8(data, reinterpret_cast<uint32_t*>(uc.ptr));
	ev2::BufferUpload up = {.size = points_size};
	ev2::commit_buffer_uploads(gfx, uc, viz->rd.points, &up, 1);
	ev2::flush_uploads(gfx);

	gpu_buffer_zero(viz->rd.codewords, points_size);

	TIMER_END(VisualizationCreate);

	return viz;
}

COVQTrainingVisualization::~COVQTrainingVisualization()
{
	for (ev2::BufferID buf : {rd.points, rd.codewords, rd.codepoints}) {
		if (buf.is_valid())
			ev2::destroy_buffer(gfx, buf);
	}
}

uint32_t COVQTrainingVisualization::stride() const
{
	return std::max(1L, std::lround(std::pow(10.0, opts.log_stride)));
}

uint32_t COVQTrainingVisualization::num_drawn_points() const
{
	uint32_t s = stride();
	return (num_points + s - 1) / s;
}

void COVQTrainingVisualization::sync(COVQTrainingContext &trainer)
{
	if (trainer.epoch.load() == synced_epoch)
		return;

	// Don't stall the frame waiting for a training iteration to finish,
	// just pick up the results on a later frame.
	std::unique_lock<std::mutex> lock(trainer.mutex, std::try_to_lock);
	if (!lock.owns_lock())
		return;

	assert(trainer.mapping.size() == num_points);

	if (num_codepoints != trainer.N || !rd.codepoints.is_valid()) {
		num_codepoints = trainer.N;

		if (rd.codepoints.is_valid())
			ev2::destroy_buffer(gfx, rd.codepoints);

		rd.codepoints = ev2::create_buffer(gfx, num_codepoints * sizeof(uint32_t),
			ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}

	size_t codewords_size = num_points * sizeof(uint32_t);
	ev2::UploadContext codewords_uc = ev2::begin_upload(gfx, codewords_size, alignof(uint32_t));
	uint32_t *codewords = reinterpret_cast<uint32_t*>(codewords_uc.ptr);
	for (uint32_t i = 0; i < num_points; ++i)
		codewords[i] = covq_index_mapping(trainer, trainer.mapping[i]);

	ev2::BufferUpload codewords_up = {.size = codewords_size};
	ev2::commit_buffer_uploads(gfx, codewords_uc, rd.codewords, &codewords_up, 1);

	size_t codepoints_size = num_codepoints * sizeof(uint32_t);
	ev2::UploadContext codepoints_uc = ev2::begin_upload(gfx, codepoints_size, alignof(uint32_t));

	// Y is empty until the first training iteration has run
	if (trainer.Y.cols() == num_codepoints) {
		assert(trainer.Y.rows() == 3);
		eigen_to_rgba8(trainer.Y, reinterpret_cast<uint32_t*>(codepoints_uc.ptr));
	} else {
		memset(codepoints_uc.ptr, 0x0, codepoints_size);
	}

	ev2::BufferUpload codepoints_up = {.size = codepoints_size};
	ev2::commit_buffer_uploads(gfx, codepoints_uc, rd.codepoints, &codepoints_up, 1);

	synced_epoch = trainer.epoch.load();
	lock.unlock();

	ev2::flush_uploads(gfx);
}

void COVQTrainingVisualization::render()
{
	if (viewport->update(nullptr) < ev2::SUCCESS || !num_points)
		return;

	uint32_t flags = 0;
	if (opts.jitter)
		flags |= JITTER_BIT;
	if (opts.snap_to_codepoint)
		flags |= SNAP_TO_CODEPOINT_BIT;

	bool has_codepoints = rd.codepoints.is_valid();

	PushConstants pc = {
		.points = ev2::get_buffer_device_address(gfx, rd.points),
		.codewords = ev2::get_buffer_device_address(gfx, rd.codewords),
		.codepoints = has_codepoints ?
			ev2::get_buffer_device_address(gfx, rd.codepoints) : 0,
		.count = num_points,
		.stride = stride(),
		.num_codepoints = has_codepoints ? num_codepoints : 0,
		.color_mode = opts.color_mode,
		.flags = flags,
		.w = opts.point_size,
	};

	ev2::GfxPassInfo pass_info = {
		.target = viewport->get_target(),
		.view = viewport->get_view(),
		.clear_color = true,
		.clear_depth = true,
		.name = "Visualization",
	};

	ev2::GfxPipelineID pipeline = ev2::load_graphics_pipeline(gfx, "cap://pipeline/training_viz.yaml");

	ev2::PassID pass = ev2::begin_gfx_pass(gfx, &pass_info);
	ev2::cmd_use_buffer(pass, rd.points, ev2::USAGE_STORAGE_READ_GRAPHICS);
	ev2::cmd_use_buffer(pass, rd.codewords, ev2::USAGE_STORAGE_READ_GRAPHICS);
	if (has_codepoints)
		ev2::cmd_use_buffer(pass, rd.codepoints, ev2::USAGE_STORAGE_READ_GRAPHICS);
	ev2::cmd_bind_gfx_pipeline(pass, pipeline);
	ev2::cmd_push_constant(pass, pipeline, 0, sizeof(pc), &pc);
	ev2::cmd_custom(pass, [count = num_drawn_points()](VkCommandBuffer cmds) {
		vkCmdDraw(cmds, 4, count, 0, 0);
	});
	ev2::end_pass(gfx, pass);
}

void COVQTrainingVisualization::update(COVQTrainingContext &trainer)
{
	sync(trainer);
	render();
}

void COVQTrainingVisualization::draw_gui()
{
	// indexed by ColorMode
	static constexpr const char *color_mode_names[] = {
		"Value",
		"Cell hash",
		"Codepoint",
	};
	static_assert(std::size(color_mode_names) == COLOR_MODE_COUNT);

	ImGui::PushID("COVQTrainingVisualization");

	ImGui::SeparatorText("Points");

	ImGui::SliderFloat("Point size", &opts.point_size, 0.0005f, 0.05f, "%.4f",
		ImGuiSliderFlags_Logarithmic);

	ImGui::SliderFloat("Subsampling", &opts.log_stride, 0.f, 4.f, "10^%.2f");
	ImGui::SetItemTooltip("Draw every n-th point of the dataset");
	ImGui::TextDisabled("Drawing %u / %u points", num_drawn_points(), num_points);

	ImGui::Checkbox("Jitter", &opts.jitter);
	ImGui::SetItemTooltip("Spread points over their 8 bit quantization bin");

	ImGui::SeparatorText("Partitions");

	int color_mode = opts.color_mode;
	if (ImGui::Combo("Color by", &color_mode, color_mode_names, COLOR_MODE_COUNT))
		opts.color_mode = static_cast<ColorMode>(color_mode);

	ImGui::Checkbox("Snap to codepoint", &opts.snap_to_codepoint);
	ImGui::SetItemTooltip("Draw each point at the reconstruction vector of its codeword");

	if (synced_epoch == INVALID_EPOCH)
		ImGui::TextDisabled("Codepoints: %u (not synced)", num_codepoints);
	else
		ImGui::TextDisabled("Codepoints: %u (epoch %lu)", num_codepoints, synced_epoch);

	ImGui::PopID();
}

struct COVQTrainingApp
{
	std::vector<Image> images;

	//--------------------------------------------------------------------------
	// Training

	COVQParams covq_params;

	// (3, count) dataset built from all of the images
	Eigen::MatrixXd X;
	// channel matrix used for the next training iteration; guarded by `sync`
	Eigen::MatrixXd P;

	// replaced under `training_sync`
	std::unique_ptr<COVQTrainingContext> trainer;
	// guarded by `codec_sync`
	std::unique_ptr<COVQCodec> codec;
	// incremented every time `codec` is replaced
	std::atomic_uint32_t epoch = 0;

	std::atomic_uint32_t completed_its = 0;
	std::atomic_uint32_t desired_its = 0;
	bool run_continuously = false;

	std::mutex training_sync;
	std::mutex sync;
	std::mutex codec_sync;
	std::thread training_thread;

	std::atomic_bool should_close = false;

	//--------------------------------------------------------------------------
	// Visualization

	// null when the visualization is disabled
	std::unique_ptr<COVQTrainingVisualization> training_viz;

	//--------------------------------------------------------------------------
	// Image display (WIP)

	enum PushConstantFlags {
		USE_HASH_COLORS_BIT = 0x1
	};

	ev2::ImageID disp_image {};
	std::weak_ptr<ev2::ImageViewer> image_viewer {};

	struct Params {
		const Image *selected_image = nullptr;
		bool use_quantized = false;
		float eps_slider = 0.05;
	} params;

	// must match map_image.frag
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

	COVQTrainingApp();
	~COVQTrainingApp();
	int initialize(const COVQParams &in_params, std::vector<Image> && in_images);
	int update();

	int reset_quantizer();
	void set_visualization_enabled(bool enabled);

	void draw_training_gui();
	void draw_image_gui();

	int view_image(const Image &image);
	void render_image();
};

COVQTrainingApp::COVQTrainingApp()
{
	ctx = ev2::editor::ctx();
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
	{
		std::unique_lock<std::mutex> lock(sync);
		P = transition_matrix(covq_params);
	}

	{
		std::unique_lock<std::mutex> lock(training_sync);
		trainer.reset(covq_init(X, P));
	}

	// new trainer starts back at epoch 0
	if (training_viz)
		training_viz->invalidate();

	desired_its.store(0);
	completed_its.store(0);

	desired_its.notify_one();

	return 0;
}

void COVQTrainingApp::set_visualization_enabled(bool enabled)
{
	if (enabled == (training_viz != nullptr))
		return;

	if (enabled)
		training_viz.reset(COVQTrainingVisualization::create(X));
	else
		training_viz.reset();
}

int COVQTrainingApp::initialize(const COVQParams &in_params, std::vector<Image> && in_images)
{
	covq_params = in_params;
	images = std::move(in_images);

	params.eps_slider = covq_params.eps;

	X = images_to_matrix(images);

	reset_quantizer();

	set_visualization_enabled(true);

	training_thread = std::thread([this](){
		while (!should_close) {
			uint32_t completed = completed_its.load(); 
			uint32_t desired = desired_its.load(); 

			if (completed < desired) {
				std::unique_lock<std::mutex> lock(training_sync);
				{
					std::unique_lock<std::mutex> lock(sync);
					trainer->P = this->P;
				}

				uint32_t it = completed_its++;
				printf("Iteration: %d\n", it);
				covq_training_it(*trainer, X);

				{
					std::unique_lock<std::mutex> codec_lock(codec_sync);
					this->codec.reset(covq_create_codec(*trainer));
					++epoch;
				}
			} else{
				printf("Next iteration: %d\n", desired + 1);
				desired_its.wait(desired, std::memory_order_seq_cst);
			}
		}
	});

	return 0;
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

void COVQTrainingApp::render_image()
{
	return;
	// disabled for now

	//std::shared_ptr<ev2::ImageViewer> viewer = image_viewer.lock();
	//if (!viewer || !mapping_gpu.is_valid() || !codepoints_gpu.is_valid())
	//	return;

	//ev2::GfxPassInfo pass_info = {
	//	.target = viewer->get_target(),
	//	.view = viewer->get_view(),
	//	.clear_color = true,
	//	.name = viewer->get_name(),
	//};

	//ev2::GfxPipelineID test_pipeline =
	//	ev2::load_graphics_pipeline(ctx, "cap://pipeline/test_image.yaml");

	//ev2::PassID pass = ev2::begin_gfx_pass(ctx, &pass_info);
	//ev2::cmd_use_buffer(pass, mapping_gpu, ev2::USAGE_STORAGE_READ_GRAPHICS);
	//ev2::cmd_use_buffer(pass, codepoints_gpu, ev2::USAGE_STORAGE_READ_GRAPHICS);

	//if (viewer->get_pipeline() == test_pipeline) {
	//	ev2::cmd_push_constant(pass, test_pipeline, 0, sizeof(pc), &pc);
	//}
	//viewer->record_draw(pass);

	//ev2::end_pass(ctx, pass);
}

static void encode_image(const COVQCodec *codec, const Image &img, uint8_t *out)
{

}

void COVQTrainingApp::draw_training_gui()
{
	ImGui::Checkbox("Run continuously", &run_continuously);

	ImGui::SliderFloat("Epsilon", &params.eps_slider, 0.f, 1.f);

	if (ImGui::Button("Reset quantizer")) {
		reset_quantizer();
	}

	ImGui::SameLine();

	char label[100];
	snprintf(label, sizeof(label), "Iteration: %d", desired_its.load()); 

	if (ImGui::Button(label)) {
		++desired_its;
		desired_its.notify_one();
	}
}

void COVQTrainingApp::draw_image_gui()
{
	if (ImGui::RadioButton("Quantized colors", params.use_quantized)) {
		params.use_quantized = !params.use_quantized;
	}

	const Image* new_image = params.selected_image;

	for (const Image &image : images) {
		if (ImGui::Selectable(image.path.c_str(), params.selected_image == &image)) {
			new_image = &image;
		}
	}

	if (new_image != params.selected_image) {
		view_image(*new_image);
		params.selected_image = new_image;
	}
}

int COVQTrainingApp::update()
{
	int status = ev2::SUCCESS;

	if (image_viewer.expired()) {
		params.selected_image = nullptr;
	}

	ImGui::Begin("Editor");

	if (ImGui::CollapsingHeader("Training", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Indent();
		draw_training_gui();
		ImGui::Unindent();
	}

	if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Indent();

		bool viz_enabled = training_viz != nullptr;
		if (ImGui::Checkbox("Enabled", &viz_enabled))
			set_visualization_enabled(viz_enabled);
		ImGui::SetItemTooltip("Disabling frees all GPU resources used by the visualization");

		if (training_viz)
			training_viz->draw_gui();

		ImGui::Unindent();
	}

	if (ImGui::CollapsingHeader("Image Picker")) {
		ImGui::Indent();
		draw_image_gui();
		ImGui::Unindent();
	}

	ImGui::End();

	if (run_continuously && desired_its <= completed_its) {
		++desired_its;
		desired_its.notify_one();
	}

	if (params.eps_slider != covq_params.eps) {
		std::unique_lock<std::mutex> lock(sync);
		covq_params.eps = params.eps_slider;
		P = transition_matrix(covq_params);
	}

	if (training_viz)
		training_viz->update(*trainer);

	pc.flags = 
		USE_HASH_COLORS_BIT * params.use_quantized;

	render_image();

	return status;
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
	if ((status = ev2::editor::init(argc, argv, "test", 1800, 800)); status != ev2::SUCCESS) {
		return status;
	}
	add_project_mounts(ev2::editor::ctx());

	uint32_t jobs = 1;

	COVQParams params = {
		.nb_bits = 8,
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
		else if (!strcmp(argv[i], "-bits") && i + 1 < argc) {
			params.nb_bits = (uint32_t)strtol(argv[++i], NULL, 10);
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

