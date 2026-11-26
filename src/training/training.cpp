#include "covq.h"
#include "stb_image.h"
#include "codec.h"
#include "covq_image_codec.h"
#include "timers.h"

#include <ev2/editor.h>
#include <ev2/viewport.h>
#include <ev2/motion_camera.h>
#include <ev2/core.h>
#include <ev2/image_viewer.h>
#include <ev2/utils/log.h>

#include <imgui_internal.h>

#include <Eigen/Dense>

#include <glob.h>

#include <cstdio>
#include <cstring>
#include <cstddef>
#include <span>
#include <bit>
#include <thread>
#include <algorithm>
#include <memory>

#define USE_COVQ_DCT_CODEC

static ev2::ImageFormat channels_to_format(int ch)
{
	switch (ch) {
	   case STBI_grey:
	   	return ev2::IMAGE_FORMAT_R8_UNORM;
	   default:
	   	return ev2::IMAGE_FORMAT_RGBA8;
	}
}

// @param k - number of codeword bits
static Eigen::MatrixXd bsc_matrix(uint32_t k, double eps)
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

static Eigen::MatrixXd nec_matrix(int k, double eps1, double eps2)
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

static int load_image(const char *path, Image *out, int out_ch)
{
	int w, h, ch;
	stbi_uc *bytes = stbi_load(path, &w, &h, &ch, out_ch);  

	if (bytes) {
		*out = Image{
			.bytes = {bytes, &Image::c_deleter},
			.w = (uint32_t)w,
			.h = (uint32_t)h,
			.ch = (uint32_t)out_ch,
		};
		printf("Loaded image: %s\n", path);
		return 0;
	} else {
		fprintf(stderr, "Failed to load image: %s\n", path);
	}

	return -1;
}

static Eigen::MatrixXd images_to_matrix(std::span<const Image> images)
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

// @brief pack (ch, n) data into rgba8 (first 4 rows), mapping v -> v * scale + offset
// before clamping to [0, 1]
static void eigen_to_rgba8(const Eigen::MatrixXd &data, uint32_t *out,
	double scale = 1.0, double offset = 0.0)
{
	const std::decay_t<decltype(data)>::Scalar *p = data.data();
	const Eigen::Index rows = data.rows();
	const int ch = (int)std::min<Eigen::Index>(rows, 4);
	size_t base = 0;
	for (Eigen::Index i = 0; i < data.size(); i += rows) {
		uint32_t rgba = 0x0;
		for (int j = 0; j < ch; ++j) {
			const double v = p[i + j] * scale + offset;
			uint32_t value = static_cast<uint32_t>(std::clamp(v * 255.0, 0.0, 255.0));
			rgba |= value << (j << 3);
		}
		out[base++] = rgba;
	}
}

// @brief pack (dim, n) data into tightly packed unorm16 (dim values per column),
// mapping v -> v * scale + offset before clamping to [0, 1]
static void eigen_to_unorm16(const Eigen::MatrixXd &data, uint16_t *out,
	double scale = 1.0, double offset = 0.0)
{
	const std::decay_t<decltype(data)>::Scalar *p = data.data();
	for (Eigen::Index i = 0; i < data.size(); ++i) {
		const double v = std::clamp(p[i] * scale + offset, 0.0, 1.0);
		out[i] = static_cast<uint16_t>(std::lround(v * (double)UINT16_MAX));
	}
}

// @brief size of a unorm16 buffer holding `count` values, padded so the shader
// can read it as an array of uint
static size_t unorm16_buffer_size(size_t count)
{
	return ((count + 1) / 2) * sizeof(uint32_t);
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
	int dim;
	int nb_bits;
	double eps;
};

static Eigen::MatrixXd transition_matrix(const COVQParams &params)
{
	return bsc_matrix(params.nb_bits, params.eps);
	//return nec_matrix(params.nb_bits, params.eps, 0.5*params.eps);
}

struct PCAPlotter
{
	struct Distribution {
		float mu;
		float var;
		double min;
		double max;
		std::vector<float> data; 

		std::vector<double> xs;
		std::vector<double> pdf;
		double pdf_max = 0;

		void set_nbins(uint32_t nbins);
		void plot(const char *name, int nbins);
	};

	// distributions along each princial component
	std::vector<Distribution> dists;
	int hist_bin_count = 256;

	PCAPlotter(const Eigen::MatrixXd &data);
	void plot(const char *name);
};

// Renders the training dataset as a point cloud, colored by the partition each
// point is currently encoded to. Points of dimension `dim` are mapped into the
// display cube with `basis` (only the first 3 components are shown).
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
	struct alignas(16) PushConstants {
		VkDeviceAddress points;
		VkDeviceAddress codewords;
		VkDeviceAddress codepoints;
		uint32_t count;
		uint32_t stride;
		uint32_t num_codepoints;
		uint32_t color_mode;
		uint32_t flags;
		uint32_t dim;
		float w;
		// std430 mat3: 3 columns padded to vec4
		alignas(16) glm::mat3x4 basis;
	};
	static_assert(offsetof(PushConstants, basis) == 64);
	static_assert(sizeof(PushConstants) == 112);

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

	// affine map from dataset values to the [0, 1] display cube:
	// v_display = v * scale + offset.
	double scale = 1.0;
	double offset = 0.0;

	uint32_t dim = 0;
	// column i is the display-space direction of component i (i < min(dim, 3))
	glm::mat3 basis = glm::mat3(1.f);

	uint32_t num_points = 0;
	uint32_t num_codepoints = 0;
	uint32_t num_possible_codepoints = 0;

	std::atomic_bool sync_failed = true;

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

	std::unique_ptr<PCAPlotter> pca_plotter;

	static constexpr uint64_t INVALID_EPOCH = UINT64_MAX;
	// training epoch that the GPU data was last synced with
	uint64_t synced_epoch = INVALID_EPOCH;

	// @param data - (dim, count) dataset the trainer is run on
	// @param scale, offset - display mapping, v -> v * scale + offset
	static COVQTrainingVisualization *create(const Eigen::MatrixXd &data,
		double scale = 1.0, double offset = 0.0);
	~COVQTrainingVisualization();

	ev2::BufferID get_codepoints_buffer() {return rd.codepoints;}

	// force a re-sync on the next update, e.g. after the trainer was replaced
	void invalidate() { synced_epoch = INVALID_EPOCH; }

	// draw the display options into the current ImGui window
	void draw_gui();

	void sync(COVQTrainingContext &trainer, std::mutex *mutex);
	void render();

	uint32_t stride() const;
	uint32_t num_drawn_points() const;
};

COVQTrainingVisualization *COVQTrainingVisualization::create(const Eigen::MatrixXd &data,
	double scale, double offset)
{
	TIMER_BEGIN(VisualizationCreate);

	COVQTrainingVisualization *viz = new COVQTrainingVisualization{};
	ev2::GfxContext *gfx = viz->gfx = ev2::editor::ctx();
	viz->scale = scale;
	viz->offset = offset;

	viz->viewport.reset(new ev2::Viewport("Visualization", 0, 0, 500, 500,
		ev2::RENDER_TARGET_CREATE_DEPTH_BIT | ev2::RENDER_TARGET_CREATE_COLOR_BIT));
	viz->camera.reset(new ev2::MotionCamera(glm::dvec3(0.5,0.5,0.5), glm::dvec3(-0.5,-0.5,1), glm::dvec3(0,0,1)));

	viz->viewport->set_camera(viz->camera);
	viz->viewport->set_closable(false);

	const int dims = data.rows();
	const int num_points = data.cols(); 
	assert(dims > 0);

	viz->dim = dims;
	viz->num_points = num_points;

	// grayscale data goes along the cube diagonal
	if (dims == 1)
		viz->basis = glm::mat3(glm::vec3(1.f), glm::vec3(0.f), glm::vec3(0.f));
	else
		viz->basis = glm::mat3(1.f);

	size_t points_size = unorm16_buffer_size((size_t)num_points * dims);
	size_t codewords_size = (size_t)num_points * sizeof(uint32_t);

	viz->rd.points = ev2::create_buffer(gfx, points_size,
		ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);
	viz->rd.codewords = ev2::create_buffer(gfx, codewords_size,
		ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);

	ev2::UploadContext uc = ev2::begin_upload(gfx, points_size, alignof(uint32_t));
	memset(uc.ptr, 0x0, points_size);
	eigen_to_unorm16(data, reinterpret_cast<uint16_t*>(uc.ptr), scale, offset);

	ev2::BufferUpload up = {.size = points_size};
	ev2::commit_buffer_uploads(gfx, uc, viz->rd.points, &up, 1);
	ev2::flush_uploads(gfx);

	gpu_buffer_zero(viz->rd.codewords, codewords_size);

	viz->pca_plotter = std::make_unique<PCAPlotter>(data);

	TIMER_END(VisualizationCreate);

	return viz;
}

PCAPlotter::PCAPlotter(const Eigen::MatrixXd &data)
{
	const int dims = data.rows(); 
	const int num_points = data.cols();

	Eigen::VectorXd mu = data.rowwise().mean();
	Eigen::MatrixXd Xc = data.colwise() - mu;
	Eigen::MatrixXd K = Xc * Xc.transpose();

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
	Eigen::MatrixXd T = es.eigenvectors().rowwise().reverse();

	for (int i = 0; i < dims; ++i) {
		dists.emplace_back(Distribution{
			.mu = (float)mu(i, 0),
			.var = (float)K(i, i),
			.data  = std::vector<float>(num_points)
		});
	}

	Xc = T.transpose() * Xc;

	Eigen::VectorXd max = Eigen::VectorXd::Constant(dims, DBL_MIN);
	Eigen::VectorXd min = Eigen::VectorXd::Constant(dims, DBL_MAX);

	for (int i = 0; i < num_points; ++i) {
		Eigen::Ref<Eigen::VectorXd> col = Xc.col(i);
		for (int j = 0; j < dims; ++j) {
	  		float x = col(j);
			dists[j].data[i] = x; 
		}
		min = min.cwiseMin(col);
		max = max.cwiseMax(col);
	}
	for (int j = 0; j < dims; ++j) {
		dists[j].min = min(j);
		dists[j].max = max(j);
	}
}

void PCAPlotter::Distribution::set_nbins(uint32_t nbins)
{
	pdf.clear();
	pdf.resize(nbins, 0);

	pdf_max = 0;

	double w = (max - min) / nbins;

	xs = std::vector<double>(nbins);
	for (uint32_t i = 0; i < nbins; ++i) {
		xs[i] = min + (i + 0.5) * w;
	}

	const double px = 1.0/(w*(double)data.size());

	for (size_t i = 0; i < data.size(); ++i) {
		uint32_t k = std::clamp(static_cast<uint32_t>(
			(float)nbins * (data[i] - min) / (max - min)
		), 0U, (uint32_t)(pdf.size() - 1));
		pdf[k] += px;

		pdf_max = std::max(pdf[k], pdf_max);
	}
}

void PCAPlotter::plot(const char *name)
{
	if (ImGui::SliderInt("Histogram Bins", &hist_bin_count, 1, 256)) {
		hist_bin_count = std::max(hist_bin_count, 1); 
	}

	if (ImPlot::BeginPlot(name, ImVec2(-1,0))) {
		double min = DBL_MAX, max = DBL_MIN, pdf_max = DBL_MIN;
		for (int i = 0; i < dists.size(); ++i) {
			min = std::min(dists[i].min, min);
			max = std::max(dists[i].max, max);
			pdf_max = std::max(dists[i].pdf_max, pdf_max);
		}

		ImPlot::SetupAxes(nullptr, nullptr,
                      ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

		for (int i = 0; i < dists.size(); ++i) {
			char buf[100];
			snprintf(buf, sizeof(buf), "Axis%i", i);
			dists[i].plot(buf, hist_bin_count);
		}
		ImPlot::EndPlot();
	}
}
void PCAPlotter::Distribution::plot(const char *name, int nbins)
{
	if (nbins != pdf.size())
		set_nbins(nbins);

	ImPlotSpec spec;
	spec.Offset = (int)0;
	spec.Flags = ImPlotCond_Always;

	int count = (int)pdf.size();

	ImPlot::PlotBars(name, xs.data(), pdf.data(), count, (max - min)/count);
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

void COVQTrainingVisualization::sync(COVQTrainingContext &trainer, std::mutex *mutex)
{
	if (trainer.epoch.load() == synced_epoch)
		return;

	// Don't stall the frame waiting for a training iteration to finish,
	// just pick up the results on a later frame.
	std::unique_lock<std::mutex> lock;

	if (mutex) {
		lock = std::unique_lock<std::mutex>(*mutex, std::try_to_lock); 

		if (!lock.owns_lock()) {
			sync_failed = true;
			return;
		}
	}

	sync_failed = false;

	assert(trainer.mapping.size() == num_points);
		
	num_possible_codepoints = trainer.num_possible_codepoints;

	if (num_codepoints != trainer.N || !rd.codepoints.is_valid()) {
		num_codepoints = trainer.N;

		if (rd.codepoints.is_valid())
			ev2::destroy_buffer(gfx, rd.codepoints);

		rd.codepoints = ev2::create_buffer(gfx, unorm16_buffer_size((size_t)num_codepoints * dim),
			ev2::BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}

	size_t codewords_size = num_points * sizeof(uint32_t);
	ev2::UploadContext codewords_uc = ev2::begin_upload(gfx, codewords_size, alignof(uint32_t));
	uint32_t *codewords = reinterpret_cast<uint32_t*>(codewords_uc.ptr);
	for (uint32_t i = 0; i < num_points; ++i)
		codewords[i] = covq_index_mapping(trainer, trainer.mapping[i]);

	ev2::BufferUpload codewords_up = {.size = codewords_size};
	ev2::commit_buffer_uploads(gfx, codewords_uc, rd.codewords, &codewords_up, 1);

	size_t codepoints_size = unorm16_buffer_size((size_t)num_codepoints * dim);
	ev2::UploadContext codepoints_uc = ev2::begin_upload(gfx, codepoints_size, alignof(uint32_t));
	memset(codepoints_uc.ptr, 0x0, codepoints_size);

	// Y is empty until the first training iteration has run
	if (trainer.Y.cols() == num_codepoints) {
		assert(trainer.Y.rows() == dim);
		eigen_to_unorm16(trainer.Y, reinterpret_cast<uint16_t*>(codepoints_uc.ptr), scale, offset);
	}

	ev2::BufferUpload codepoints_up = {.size = codepoints_size};
	ev2::commit_buffer_uploads(gfx, codepoints_uc, rd.codepoints, &codepoints_up, 1);

	synced_epoch = trainer.epoch.load();

	if (mutex)
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
		.dim = dim,
		.w = opts.point_size,
		.basis = glm::mat3x4(basis),
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

	pca_plotter->plot("Principal Components");

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
		ImGui::TextDisabled("Codepoints: %u (%u possible, not synced)",
			num_codepoints, num_possible_codepoints);
	else
		ImGui::TextDisabled("Codepoints: %u (%u possible, epoch %llu)", 
			num_codepoints, num_possible_codepoints, (unsigned long long)synced_epoch);

	ImGui::PopID();
}

//------------------------------------------------------------------------------
// Channel
//------------------------------------------------------------------------------

struct IChannel
{
	virtual size_t num_out_bytes(size_t nbytes_in) const = 0;
	virtual void transmit(unsigned seed, size_t nbytes_in, const uint8_t *in_bytes, uint8_t *out_bytes) const = 0;
};

struct BinSymChannel final : public IChannel
{
	double eps;

	BinSymChannel(double eps_in) : IChannel(), eps(eps_in) {
	}

	virtual size_t num_out_bytes(size_t nbytes_in) const final
	{
		return nbytes_in;
	}

	virtual void transmit(unsigned seed, size_t nbytes_in, const uint8_t *in_bytes, uint8_t *out_bytes) const final
	{
		srand(seed);

		const int r_thres = (int)round((double)RAND_MAX * eps);
		for (size_t i = 0; i < nbytes_in; ++i) {

			uint8_t Z = 0;

			for (int b = 0; b < 8; ++b) {
				if (rand() < r_thres)
					Z |= 1 << b;
			}

			out_bytes[i] = in_bytes[i] ^ Z;
		}
	}
};

//------------------------------------------------------------------------------
// Test Image
//------------------------------------------------------------------------------

struct RateOption
{
	const char *name;
	const uint8_t *bit_table;
};

static constexpr RateOption g_rate_options[] = {
	{"DC",       BitTables::dc_only},
	{"Full",     BitTables::full},
	{"0.25 bpp", BitTables::bpp_0_25},
	{"0.5 bpp",  BitTables::bpp_0_5},
	{"1 bpp",    BitTables::bpp_1},
	{"2 bpp",    BitTables::bpp_2},
	{"4 bpp",    BitTables::bpp_4},
};
static constexpr int RATE_COUNT = (int)std::size(g_rate_options);
// 0.5 bpp
static constexpr int DEFAULT_RATE = 3;

static bool rate_combo(const char *label, int *rate)
{
	return ImGui::Combo(label, rate, [](void *, int i) {
		return g_rate_options[i].name;
	}, nullptr, RATE_COUNT);
}

// @brief a square button with a check mark drawn in it
static bool checkmark_button(const char *id)
{
	const float sz = ImGui::GetFrameHeight();
	bool pressed = ImGui::Button(id, ImVec2(sz, sz));

	const float pad = 0.2f * sz;
	ImVec2 min = ImGui::GetItemRectMin();
	ImGui::RenderCheckMark(ImGui::GetWindowDrawList(), ImVec2(min.x + pad, min.y + pad),
		ImGui::GetColorU32(ImGuiCol_CheckMark), sz - 2.f * pad);

	return pressed;
}

static const char *path_basename(const std::string &path)
{
	size_t pos = path.find_last_of("\\/");
	return path.c_str() + (pos == std::string::npos ? 0 : pos + 1);
}

struct TestInstanceParams
{
	std::span<const std::shared_ptr<Image>> images;
	std::span<const std::string> image_paths;
	// latest codec produced by training; null until the first iteration has run
	std::shared_ptr<IBlockImageCodec> training_codec;
	const IChannel &channel;
};

enum TestCodecType : int
{
	TEST_CODEC_UNIFORM_DCT,
	// snapshot of the training codec at the time it was applied
	TEST_CODEC_TRAINING,
	TEST_CODEC_COUNT
};

static constexpr const char *TEST_CODEC_NAMES[] = {
	"Uniform DCT",
	"Training codec",
};
static_assert(std::size(TEST_CODEC_NAMES) == TEST_CODEC_COUNT);

struct TestCodecParams
{
	int type = TEST_CODEC_UNIFORM_DCT;
	int rate = DEFAULT_RATE;
};

static std::atomic<long> g_inst_id;

// An image sent through a codec and the channel.
struct TestInstance
{
	long id;
	std::string name;

	// index into TestInstanceParams::images, -1 for none
	int image_idx = 0;

	// parameters `codec` was created with
	TestCodecParams codec_params;
	// parameters being edited in the codec options popup
	TestCodecParams pending_codec_params;
	std::shared_ptr<IBlockImageCodec> codec;

	std::unique_ptr<uint8_t[]> compressed_bytes;
	Image out;

	double disp_mse = 0.0;

	ev2::ImageID gpu_img;
	std::weak_ptr<ev2::ImageViewer> viewer;

	struct Opts {
		bool display = true;
	} opts;

	// @param in_name - header title, "Test <id>" if empty
	TestInstance(std::string in_name = {}, TestCodecParams in_codec_params = {}) :
		name(std::move(in_name)),
		codec_params(in_codec_params),
		pending_codec_params(in_codec_params)
	{
		id = ++g_inst_id;

		if (name.empty()) {
			char buf[64];
			snprintf(buf, sizeof(buf), "Test %ld", id);
			name = buf;
		}

		// a training codec is picked up in update() once one exists
		if (codec_params.type == TEST_CODEC_UNIFORM_DCT)
			codec = create_uniform_dct_codec(codec_params.rate);
	}

	~TestInstance() {
		ev2::GfxContext *ctx = ev2::editor::ctx();
		if (gpu_img.is_valid())
			ev2::destroy_image(ctx, gpu_img);
		if (auto ptr = viewer.lock()) {
			ev2::editor::close_image_viewer(ptr);
		}
	}

	static std::shared_ptr<IBlockImageCodec> create_uniform_dct_codec(int rate)
	{
		return std::make_shared<DCTImageCodec>(
			new UniformDCTQuantizer(g_rate_options[rate].bit_table));
	}

	// @brief replace `codec` using `pending_codec_params`
	// @return false if the requested codec is not available
	bool reset_codec(const TestInstanceParams &params);

	void encode(const Image &img);
	void transmit_and_decode(const IChannel &channel);
	void display();

	void draw_codec_options(const TestInstanceParams &params);

	// @return false if closed from its header
	bool update(const TestInstanceParams &params);
};

static double compute_mse(const Image &A, const Image &B)
{
	if (A.w != B.w || A.h != B.h || A.ch != B.ch) {
		fprintf(stderr, "Images must have same size\n");
		return 0;
	}

	size_t count = (size_t)A.w * (size_t)A.h * (size_t)A.ch; 
	const uint8_t *X = A.bytes.get();
	const uint8_t *Y = B.bytes.get();

	double acc = 0.0;

	for (size_t i = 0; i < count; ++i) {
		double d = ((double)Y[i] - (double)X[i]) / 255.0;
		acc += d * d;
	}

	return acc / (double)(count);
}

bool TestInstance::reset_codec(const TestInstanceParams &params)
{
	std::shared_ptr<IBlockImageCodec> new_codec;

	switch (pending_codec_params.type) {
	case TEST_CODEC_UNIFORM_DCT:
		new_codec = create_uniform_dct_codec(pending_codec_params.rate);
		break;
	case TEST_CODEC_TRAINING:
		new_codec = params.training_codec;
		break;
	default:
		break;
	}

	if (!new_codec) {
		log_warn("No codec available");
		return false;
	}

	codec = std::move(new_codec);
	codec_params = pending_codec_params;

	return true;
}

void TestInstance::draw_codec_options(const TestInstanceParams &params)
{
	// start from the active codec each time the popup opens
	if (ImGui::IsWindowAppearing())
		pending_codec_params = codec_params;

	ImGui::SetNextItemWidth(160.f);
	ImGui::Combo("Codec", &pending_codec_params.type,
		TEST_CODEC_NAMES, TEST_CODEC_COUNT);

	const bool training_unavailable =
		pending_codec_params.type == TEST_CODEC_TRAINING && !params.training_codec;

	if (pending_codec_params.type == TEST_CODEC_UNIFORM_DCT) {
		ImGui::SetNextItemWidth(160.f);
		rate_combo("Rate", &pending_codec_params.rate);
	} else if (training_unavailable) {
		ImGui::TextDisabled("No training codec yet");
	}

	ImGui::Separator();

	ImGui::BeginDisabled(training_unavailable);
	if (checkmark_button("##apply")) {
		if (reset_codec(params))
			ImGui::CloseCurrentPopup();
	}
	ImGui::EndDisabled();
	ImGui::SetItemTooltip("Reset the codec with these options");
}

bool TestInstance::update(const TestInstanceParams &params)
{
	ImGui::PushID((int)id);

	if (image_idx >= (int)params.images.size())
		image_idx = -1;

	// nothing has been applied yet, so take the first training codec available
	if (!codec && codec_params.type == TEST_CODEC_TRAINING)
		codec = params.training_codec;

	bool keep_open = true;
	if (ImGui::CollapsingHeader(name.c_str(), &keep_open)) {
		ImGui::Indent();

		const char *preview = image_idx >= 0 ?
			path_basename(params.image_paths[image_idx]) : "(none)";

		if (ImGui::BeginCombo("Image", preview)) {
			for (int i = 0; i < (int)params.images.size(); ++i) {
				const bool selected = i == image_idx;
				if (ImGui::Selectable(path_basename(params.image_paths[i]), selected))
					image_idx = i;
				if (selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		ImGui::Button("Codec options");
		if (ImGui::BeginPopupContextItem("codec_options", ImGuiPopupFlags_MouseButtonLeft)) {
			draw_codec_options(params);
			ImGui::EndPopup();
		}

		ImGui::SameLine();
		if (codec_params.type == TEST_CODEC_UNIFORM_DCT)
			ImGui::TextDisabled("%s", codec ? codec->get_name() : "[none]");
		else
			ImGui::TextDisabled("%s%s", codec ? codec->get_name() : "[none]",
				codec ? "" : " (waiting for training)");

		ImGui::Checkbox("Display", &opts.display);

		ImGui::BeginDisabled(!codec || image_idx < 0);
		if (ImGui::Button("Compress and transmit")) {
			encode(*params.images[image_idx]);
			transmit_and_decode(params.channel);
			disp_mse = compute_mse(out, *params.images[image_idx]);
			display();
		}
		ImGui::EndDisabled();
		ImGui::Text("PSNR: %.3f dB", -10.0 * log10(disp_mse));

		ImGui::Unindent();
	}

	ImGui::PopID();

	return keep_open;
}

void TestInstance::display()
{
	if (!opts.display || !out.bytes)
		return;

	if (auto ptr = viewer.lock()) {
		ev2::editor::close_image_viewer(ptr);
	}

	ev2::GfxContext *ctx = ev2::editor::ctx();

	if (gpu_img.is_valid())
		ev2::destroy_image(ctx, gpu_img);

	ev2::ImageFormat fmt = channels_to_format(out.ch); 

	gpu_img = ev2::create_image(ctx, out.w, out.h, 1,
		fmt, ev2::IMAGE_USAGE_SAMPLED_BIT);

	ev2::UploadContext uc = ev2::begin_upload(ctx, out.nbytes(), alignof(uint32_t));
	memcpy(uc.ptr, out.bytes.get(), out.nbytes());
	ev2::ImageUpload up = {
		.w = out.w,
		.h = out.h,
		.d = 1,
	};

	ev2::commit_image_uploads(ctx, uc, gpu_img, &up, 1);
	ev2::flush_uploads(ctx);

	viewer = ev2::editor::open_image_viewer(gpu_img, name.c_str(), nullptr);
}

void TestInstance::encode(const Image &img)
{
	if (!codec)
		return;

	size_t nb_bytes_in = codec->get_compressed_size(img.w, img.h, img.ch);
	compressed_bytes.reset(new uint8_t[nb_bytes_in]);
	codec->encode(img.w, img.h, img.ch, img.bytes.get(), compressed_bytes.get());

	out = {
		.w = img.w,
		.h = img.h,
		.ch = img.ch,
	};
}

void TestInstance::transmit_and_decode(const IChannel &channel)
{
	if (!compressed_bytes || !codec)
		return;

	out.bytes = Image::BytesPtr{new uint8_t[out.nbytes()], &operator delete[]};
	size_t nb_bytes_in = codec->get_compressed_size(out.w, out.h, out.ch);
	size_t nb_bytes_out = channel.num_out_bytes(nb_bytes_in); 

	std::unique_ptr<uint8_t[]> received(new uint8_t[nb_bytes_out]);
	channel.transmit(256, nb_bytes_in, compressed_bytes.get(), received.get());

	codec->decode(out.w, out.h, out.ch, received.get(), out.bytes.get());
}

//------------------------------------------------------------------------------
// The app
//------------------------------------------------------------------------------

struct COVQTrainingApp
{
	ev2::GfxContext *ctx;

	std::vector<std::string> image_paths;

	// shared pointers so it's easy to have a bunch of things
	// referencing the same image if needed
	std::vector<std::shared_ptr<Image>> images;

	std::vector<std::shared_ptr<TestInstance>> test_instances;

	//--------------------------------------------------------------------------
	// Training

	COVQParams covq_params;

	// (3, count) dataset built from all of the images
#ifdef USE_COVQ_DCT_CODEC
	Eigen::MatrixXd X[64];
	std::function<Eigen::MatrixXd(uint8_t k)> matrix_generator;
#else
	// channel matrix used for the next training iteration; guarded by `sync`
	Eigen::MatrixXd P;
	Eigen::MatrixXd X;
#endif

	// replaced under `training_sync`
	struct {
		double dmax = 0;
		std::vector<double> distortion_history;

		// test results
		struct {
			std::vector<std::string> labels;
			std::vector<double> eps;
			// labels.size() * eps.size() 
			std::vector<double> psnr;
			// fit the plot axes to the results on the next draw
			bool refit = false;
		} test;
	} stats;

#ifdef USE_COVQ_DCT_CODEC
	std::unique_ptr<COVQDCTCodecTrainer> trainer;
#else
	std::unique_ptr<COVQTrainingContext> trainer;
#endif


	// incremented every time `codec` is replaced
	std::atomic_uint32_t epoch = 0;

	std::atomic_uint32_t completed_its = 0;
	std::atomic_uint32_t desired_its = 0;
	bool run_continuously = false;

	std::mutex training_sync;
	std::mutex sync;
	std::thread training_thread;

	std::atomic_bool should_close = false;

	//--------------------------------------------------------------------------
	// Visualization

	// null when the visualization is disabled
	std::unique_ptr<COVQTrainingVisualization> training_viz;
	// Access through std::atomic_load/store (for older libc++ compatibility)
	std::shared_ptr<IBlockImageCodec> avail_codec;

	std::shared_ptr<IChannel> channel;

	// bits per DCT coefficient used by the next reset of the trainer
	uint8_t bit_table[64];

	struct Options {
		float eps_slider = 0.05;
		float noise_eps = 0.f;

		// number of previous runs to plot data for.
		int plot_history = 10;

		int selected_viz_bin = 0;
		// index into k_rate_options for the training codec
		int training_rate = DEFAULT_RATE;

		float max_test_eps = 0.25;
		int num_test_samples = 8;
		int test_seed = 0;
	} opts;


	COVQTrainingApp();
	~COVQTrainingApp();
	int initialize(const COVQParams &in_params, std::vector<Image> &&in_images, std::vector<std::string> &&in_image_paths);
	int update();

	int reset_quantizer();
	void set_visualization_enabled(bool enabled);

	void training_thread_func();

	void draw_training_gui();
	void draw_test_gui();
	void run_tests();
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
#ifndef USE_COVQ_DCT_CODEC
		P = transition_matrix(covq_params);
#endif
		stats.distortion_history.clear();
		stats.dmax = 0;
	}

	{
		std::unique_lock<std::mutex> lock(training_sync);
#ifdef USE_COVQ_DCT_CODEC
		trainer.reset(COVQDCTCodecTrainer::create(X, bit_table, matrix_generator));
#else
		trainer.reset(COVQTrainingContext::create(X, P));
#endif
		// so the apple folks can be compatible without a big fat annoying mutex
		std::atomic_store(&avail_codec, std::shared_ptr<IBlockImageCodec>{});
		desired_its.store(0);
		completed_its.store(0);
	}

	// new trainer starts back at epoch 0
	if (training_viz)
		training_viz->invalidate();

	desired_its.notify_one();

	return 0;
}

#ifdef USE_COVQ_DCT_CODEC
// @brief create a visualization of RGB values in a DCT bin
static COVQTrainingVisualization *create_dct_bin_visualization(
	int bin, const Eigen::MatrixXd &data)
{
	double scale, offset;

	// DC goes to [0,1]
	if (bin == 0) {
		scale = 1.0 / (8.0 * 255.0);
		offset = 128.0 / 255.0;
	} else {
		std::vector<double> mags(data.size());
		for (Eigen::Index i = 0; i < data.size(); ++i)
			mags[i] = std::abs(data.data()[i]);

		// Rescale the points so that ~99% stay within a unit [0,1] cube
		double r = 0.0;
		if (!mags.empty()) {
			auto nth = mags.begin() + (mags.size() - 1) * 99 / 100;
			std::nth_element(mags.begin(), nth, mags.end());
			r = *nth;
		}

		scale = r > 0.0 ? 0.5 / r : 1.0;
		offset = 0.5;
	}

	return COVQTrainingVisualization::create(data, scale, offset);
}
#endif

void COVQTrainingApp::set_visualization_enabled(bool enabled)
{
	if (enabled == (training_viz != nullptr))
		return;

#ifdef USE_COVQ_DCT_CODEC
	if (enabled) {
		training_viz.reset(create_dct_bin_visualization(opts.selected_viz_bin, X[opts.selected_viz_bin]));
	} else {
		training_viz.reset();
	}
#else
	if (enabled)
		training_viz.reset(COVQTrainingVisualization::create(X));
	else
		training_viz.reset();
#endif
}

void COVQTrainingApp::training_thread_func()
{
	while (!should_close) {
		uint32_t completed = completed_its.load(); 
		uint32_t desired = desired_its.load(); 

		if (completed < desired) {
			std::unique_lock<std::mutex> lock(training_sync);

			if((training_viz && training_viz->sync_failed) || 
				completed_its.load() >= desired_its.load())
				continue;

			{
				std::unique_lock<std::mutex> lock(sync);
#ifdef USE_COVQ_DCT_CODEC
				for (int i = 0; i < 64; ++i) {
					if (trainer->ctx[i])
						trainer->ctx[i]->P = matrix_generator(trainer->ctx[i]->word_length);
				}
#else
				trainer->P = this->P;
#endif
			}

			uint32_t it = completed_its++;
			printf("Iteration: %d\n", it);
			trainer->train(X);

			{
				std::unique_lock<std::mutex> lock(sync);
				double d = trainer->expected_distortion;
				stats.distortion_history.push_back(d);
				stats.dmax = std::max(stats.dmax, d);
			}

			{
#ifdef USE_COVQ_DCT_CODEC
				std::string name = std::format("COVQ_eps{:.2f}", covq_params.eps);
				std::shared_ptr<IBlockImageCodec> codec(trainer->create_codec(name.c_str()));
#else
				std::shared_ptr<IBlockImageCodec> codec = std::make_shared<COVQTestImageCodec>(
					trainer->create_codec()
				);
#endif
				std::atomic_store(&avail_codec, codec);
				++epoch;
			}
		} else{
			printf("Next iteration: %d\n", desired);
			desired_its.wait(desired, std::memory_order_seq_cst);
		}
	}
}

int COVQTrainingApp::initialize(const COVQParams &in_params,
	std::vector<Image> &&in_images, std::vector<std::string> &&in_image_paths)
{
	covq_params = in_params;
	image_paths = std::move(in_image_paths);

	opts.eps_slider = covq_params.eps;

	memcpy(bit_table, g_rate_options[opts.training_rate].bit_table, 64);

#ifdef USE_COVQ_DCT_CODEC
	channel = std::make_shared<BinSymChannel>(covq_params.eps);
	matrix_generator = [eps = covq_params.eps](uint8_t k) {
		return bsc_matrix(k, eps);
	};
	int result = COVQDCTCodecTrainer::generate_data(in_params.dim, 
												 (uint32_t)in_images.size(), in_images.data(), X);

	if (result < 0)
		return result;
#else
	X = images_to_matrix(images);
#endif

	images.reserve(in_images.size());
	for (Image & img : in_images) {
		images.push_back(std::make_shared<Image>(std::move(img)));
	}

	reset_quantizer();

	set_visualization_enabled(true);

	test_instances.push_back(std::make_unique<TestInstance>("Baseline",
		TestCodecParams{.type = TEST_CODEC_UNIFORM_DCT}));
	test_instances.push_back(std::make_unique<TestInstance>("Training",
		TestCodecParams{.type = TEST_CODEC_TRAINING}));

	training_thread = std::thread(&COVQTrainingApp::training_thread_func, this);

	return 0;
}


void COVQTrainingApp::draw_training_gui()
{
	if (ImPlot::BeginPlot("Expected Distortion",ImVec2(-1,200))) {
		ImPlotSpec spec;
		spec.Offset = (int)0;

		int count = (int)stats.distortion_history.size();
		int num = opts.plot_history ? std::min(opts.plot_history, count) : count;

		ImPlot::SetupAxes("iteration", "E[D]", 0,
                  ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
		ImPlot::SetupAxisLimits(ImAxis_X1, std::max(count - num, 0),
                        std::max(count - 1, 1), ImPlotCond_Always);

		ImPlot::PlotLine("E(D(X,Y))", 
			stats.distortion_history.data(), (int)stats.distortion_history.size(),1, 0, spec);
		ImPlot::EndPlot();
	}

	if (ImGui::SliderInt("Plot history", &opts.plot_history, 0, 100, "%d runs")) {
		opts.plot_history = std::max(opts.plot_history, 0);
	}

	ImGui::Checkbox("Run continuously", &run_continuously);

	if (rate_combo("Rate", &opts.training_rate))
		memcpy(bit_table, g_rate_options[opts.training_rate].bit_table, 64);

	ImGui::SliderFloat("Epsilon (Training)", &opts.eps_slider, 0.f, 1.f);
	ImGui::SliderFloat("Epsilon (Noise)", &opts.noise_eps, 0.f, 1.f);

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

void COVQTrainingApp::run_tests()
{
	stats.test.labels.clear();
	stats.test.psnr.clear();
	stats.test.eps.resize(opts.num_test_samples);

	const double d_eps = opts.max_test_eps / (double)(opts.num_test_samples - 1);

	for (int i = 0; i < opts.num_test_samples; ++i)
		stats.test.eps[i] = (double)i * d_eps;

	std::vector<uint8_t> encoded;
	std::vector<uint8_t> received;
	std::vector<uint8_t> decoded;

	for (const std::shared_ptr<TestInstance> &inst : test_instances) {
		if (!inst || !inst->codec) {
			log_warn("Instance %d has no codec", inst->name.c_str());
			continue;
		}

		const Image &img = *images[inst->image_idx];
		const size_t nb_bytes_in = inst->codec->get_compressed_size(img.w, img.h, img.ch);

		encoded.resize(nb_bytes_in);
		decoded.resize(img.nbytes());

		Image dec_img = {
			.bytes = Image::BytesPtr{decoded.data(), Image::null_deleter},
			.w = img.w,
			.h = img.h,
			.ch = img.ch,
		};

		inst->codec->encode(img.w, img.h, img.ch, img.bytes.get(), encoded.data());

		for (int i = 0; i < opts.num_test_samples; ++i) {
			double eps = (double)i * d_eps;
			BinSymChannel ch = BinSymChannel(eps);

			const size_t nb_bytes_out = ch.num_out_bytes(nb_bytes_in); 
			received.resize(nb_bytes_out);

			ch.transmit((unsigned)opts.test_seed, nb_bytes_in, encoded.data(), received.data());

			inst->codec->decode(img.w, img.h, img.ch, received.data(), decoded.data());

			double mse = compute_mse(dec_img, img);
			stats.test.psnr.push_back(-10.0 * log10(mse));
		}

		stats.test.labels.push_back(inst->codec->get_name());
	}

	stats.test.refit = true;
}

void COVQTrainingApp::draw_test_gui()
{
	ImGui::SeparatorText("Parameters");

	ImGui::SliderInt("Epsilon Samples", &opts.num_test_samples, 1, 32);
	ImGui::SliderFloat("Max Epsilon", &opts.max_test_eps, 0, 0.5);
	ImGui::InputInt("Seed", &opts.test_seed);

	if (ImGui::Button("Run tests")) {
		run_tests();
	}

	ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.05f, 0.15f));

	// fit once per run so the plot can still be panned and zoomed
	if (stats.test.refit) {
		ImPlot::SetNextAxesToFit();
		stats.test.refit = false;
	}

	if (ImPlot::BeginPlot("PSNR vs eps",ImVec2(-1,0))) {
		ImPlot::SetupAxes("eps", "PSNR (dB)");

		// PSNR falls off with eps, so the top right is empty
		ImPlot::SetupLegend(ImPlotLocation_NorthEast);

		for (size_t i = 0; i < stats.test.labels.size(); ++i) {
			int count = stats.test.eps.size();

			ImPlot::PlotLine(stats.test.labels[i].c_str(),
				stats.test.eps.data(), &stats.test.psnr[i * count], count);
		}
		ImPlot::EndPlot();
	}

	ImPlot::PopStyleVar();

	ImGui::SeparatorText("Instances");

	TestInstanceParams params = {
		.images = images,
		.image_paths = image_paths,
		.training_codec = std::atomic_load(&avail_codec),
		.channel = *channel,
	};

	// instances closed from their header are destroyed after the loop
	std::erase_if(test_instances, [&params](const std::shared_ptr<TestInstance> &inst) {
		return !inst->update(params);
	});

	if (ImGui::Button("Add test instance"))
		test_instances.push_back(std::make_unique<TestInstance>());
}

int COVQTrainingApp::update()
{
	int status = ev2::SUCCESS;

	ImGui::Begin("Editor");

	if (ImGui::CollapsingHeader("Training", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Indent();
		draw_training_gui();
		ImGui::Unindent();
	}

	channel = std::make_shared<BinSymChannel>(opts.noise_eps);

	if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Indent();

		bool viz_enabled = training_viz != nullptr;
		if (ImGui::Checkbox("Enabled", &viz_enabled))
			set_visualization_enabled(viz_enabled);
		ImGui::SetItemTooltip("Disabling frees all GPU resources used by the visualization");

#ifdef USE_COVQ_DCT_CODEC
		int selected_viz_bin = opts.selected_viz_bin;
		if (ImGui::SliderInt("Selected bin", &selected_viz_bin, 0, 63)) {
			if (viz_enabled && trainer->ctx[selected_viz_bin]) {
				opts.selected_viz_bin = std::clamp(selected_viz_bin, 0, 63);
				COVQTrainingVisualization::Options old_viz_options = training_viz->opts;
				training_viz.reset(create_dct_bin_visualization(selected_viz_bin, X[selected_viz_bin]));
				training_viz->opts = old_viz_options;
			}
		}
#endif

		if (training_viz)
			training_viz->draw_gui();

		ImGui::Unindent();
	}

	if (ImGui::CollapsingHeader("Test Instances")) {
		ImGui::Indent();
		draw_test_gui();
		ImGui::Unindent();
	}

	ImGui::End();

	if (run_continuously && desired_its <= completed_its) {
		++desired_its;
		desired_its.notify_one();
	}

	if (opts.eps_slider != covq_params.eps) {
		std::unique_lock<std::mutex> lock(sync);
		covq_params.eps = opts.eps_slider;
#ifdef USE_COVQ_DCT_CODEC
		matrix_generator = [eps = covq_params.eps](uint8_t k) {
			return bsc_matrix(k, eps);
		};
#else
		P = transition_matrix(covq_params);
#endif

	}

#ifdef USE_COVQ_DCT_CODEC
	if (training_viz && trainer->ctx[opts.selected_viz_bin]) {
		training_viz->sync(*trainer->ctx[opts.selected_viz_bin], &training_sync);
		training_viz->render();
	}
#else
	if (training_viz) {
		training_viz->sync(*trainer, &training_sync);
		training_viz->render();
	}
#endif

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
	stbi_set_flip_vertically_on_load_thread(true);

	int status = 0;

	COVQParams params = {
		.dim = 3,
		.nb_bits = 8,
		.eps = 0.05
	};
	uint32_t jobs = 1;

	std::vector<std::string> image_paths;
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
					const char *path = g.gl_pathv[j];
					image_paths.emplace_back(path);
				}
			}
		}
	}

	if (image_paths.empty()) {
		printf("No images passed\n");
		return 0;
	}

	int channels = params.dim == 1 ? STBI_grey : STBI_rgb_alpha;

	// only keep the paths of images that loaded, so they index the same as `images`
	std::vector<std::string> loaded_paths;
	for (std::string &path : image_paths) {
		Image img;
		int result = load_image(path.c_str(), &img, channels);

		if (result == 0) {
			images.emplace_back(std::move(img));
			loaded_paths.emplace_back(std::move(path));
		} else {
			fprintf(stderr, "Failed to load image: %s\n", path.c_str());
		}
	}
	image_paths = std::move(loaded_paths);

	if (images.empty()) { 
		printf("No images loaded\n");
		return 0;
	}

	Eigen::setNbThreads(jobs);
	printf("Using %d threads for Eigen\n", jobs);

	if ((status = ev2::editor::init(argc, argv, "test", 1800, 800)); status != ev2::SUCCESS) {
		return status;
	}
	add_project_mounts(ev2::editor::ctx());
	std::unique_ptr<COVQTrainingApp> app = std::make_unique<COVQTrainingApp>();
	app->initialize(params, std::move(images), std::move(image_paths));

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

