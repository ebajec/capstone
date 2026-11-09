#include <cstdint>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstring>
#include <cassert>
#include <cmath>

#include <ev2/editor.h>
#include <ev2/viewport.h>
#include <ev2/utils/log.h>
#include <ev2/core.h>

#include "stb_image.h"

#include "dct8.hpp"

#include "uniform_quantizer.h"

struct Image
{
	int w, h, ch;
	std::unique_ptr<uint8_t[]> bytes;
};

struct CompressedImage
{
	uint32_t w, h;
	std::unique_ptr<uint8_t[]> bytes;
};

static void compress_image(
	const IBlockCodec &codec,
	const Image &in,
	CompressedImage &out)
{
	const uint32_t w = in.w;
	const uint32_t h = in.h;

	const uint8_t *in_bytes = in.bytes.get();

	const uint32_t total_blks = ((w + 7) >> 3) * ((h + 7) >> 3); 
	const size_t total_bytes = 
		(codec.bits_per_block() * (size_t)total_blks + 7)/8;

	float coeffs[64];

	uint8_t * out_bytes = new uint8_t[total_bytes] {};

	uint8_t blk[64];

	size_t bit_idx = 0;
	for (uint32_t by = 0; by < h; by += 8) {
		for (uint32_t bx = 0; bx < w; bx += 8) {

			for (uint32_t y = 0; y < 8; ++y) {
				const uint32_t sy = std::min(by + y, h - 1);
				for (uint32_t x = 0; x < 8; ++x) {
					const uint32_t sx = std::min(bx + x, w - 1);
					blk[y * 8 + x] = in_bytes[w * sy + sx];
				}
			}
			dct8::forward_u8(blk, 8, coeffs);

			bit_idx += codec.encode(coeffs, out_bytes, bit_idx);
		}
	}
	
	out.bytes.reset(out_bytes);
	out.w = w;
	out.h = h;
}

static void uncompress_image(
	const IBlockCodec &codec,
	const CompressedImage &in,
	Image &out)
{
	const uint32_t w = in.w;
	const uint32_t h = in.h;

	const uint8_t *in_bytes = in.bytes.get();

	uint8_t * out_bytes = new uint8_t[w * h];

	float coeffs[64];
	uint8_t blk[64];

	size_t bit_idx = 0;
	for (uint32_t by = 0; by < h; by += 8) {
		for (uint32_t bx = 0; bx < w; bx += 8) {
			bit_idx += codec.decode(in_bytes, bit_idx, coeffs);

			dct8::inverse_u8(coeffs, blk, 8);

			// Crop the padding
			const uint32_t bw = std::min(8u, w - bx);
			const uint32_t bh = std::min(8u, h - by);
			for (uint32_t y = 0; y < bh; ++y)
				memcpy(out_bytes + w * (by + y) + bx, blk + y * 8, bw);
		}
	}

	out.w = w;
	out.h = h;
	out.bytes.reset(out_bytes);
	out.ch = STBI_grey;
}

// RGB -> YCbCr, full-range BT.601 (JPEG/JFIF). Cb and Cr are offset by 128.
static void rgb8_to_ycbcr(const uint8_t rgb[3], uint8_t ycbcr[3])
{
	const float r = rgb[0], g = rgb[1], b = rgb[2];

	const float y  =          0.299f    * r + 0.587f    * g + 0.114f    * b;
	const float cb = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f      * b;
	const float cr = 128.0f + 0.5f      * r - 0.418688f * g - 0.081312f * b;

	auto to_u8 = [](float v) {
		return static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L));
	};

	ycbcr[0] = to_u8(y);
	ycbcr[1] = to_u8(cb);
	ycbcr[2] = to_u8(cr);
}

struct App
{
	struct RenderData {
		ev2::ImageID orig {};
		ev2::ImageID decomp {};
		ev2::ImageID grid_img {};
		ev2::TextureID grid_tex {};
	} rd {};

	std::shared_ptr<ev2::ImageViewer> orig_viewer;
	std::shared_ptr<ev2::ImageViewer> decomp_viewer;

	std::unique_ptr<ev2::Viewport> grid_viewport;

	CompressedImage comp_bits;
	Image orig_img;
	Image decomp_img;
	
	uint8_t bit_grid[8][8] {};

	bool prev_left = false;
	bool prev_right = false;

	ev2::GfxContext *ctx;

	App() :ctx(ev2::editor::ctx())
	{
		grid_viewport.reset(new ev2::Viewport("Grid", 0, 0, 200, 200));

		memcpy(bit_grid, BitTables::dc_only, 64);

		rd.grid_img = ev2::create_image(ctx, 8, 8, 1, ev2::IMAGE_FORMAT_R8_UNORM, ev2::IMAGE_USAGE_SAMPLED_BIT);
		rd.grid_tex = ev2::create_texture(ctx, rd.grid_img, ev2::FILTER_NEAREST);

		update_grid();
	}

	void update_grid() {
		ev2::UploadContext uc = ev2::begin_upload(ctx, 64, 1);
		memcpy(uc.ptr, bit_grid, 64);
		ev2::ImageUpload up = {.w = 8, .h = 8, .d = 1};
		ev2::commit_image_uploads(ctx, uc, rd.grid_img, &up, 1);
		ev2::flush_uploads(ctx);
	}

	void recompress_image()
	{
		std::unique_ptr<UniformQuantizer> codec = std::make_unique<UniformQuantizer>(
			&bit_grid[0][0], 128.f); 
		compress_image(*codec, orig_img, comp_bits);
		uncompress_image(*codec, comp_bits, decomp_img);
		
		size_t nbytes = orig_img.w * orig_img.h * orig_img.ch;

		ev2::UploadContext uc = ev2::begin_upload(ctx, nbytes, alignof(uint32_t));
		ev2::ImageUpload up{
			.w = (uint32_t)decomp_img.w,
			.h = (uint32_t)decomp_img.h,
		};
		memcpy(uc.ptr, decomp_img.bytes.get(), nbytes);
		ev2::commit_image_uploads(ctx, uc, rd.decomp, &up, 1);
	}

	int set_image(Image &&img) {

		orig_img = std::move(img);

		if (rd.orig.is_valid()) {
			ev2::editor::close_image_viewer(orig_viewer);
			ev2::destroy_image(ctx, rd.orig);
		}
		if (rd.decomp.is_valid()) {
			ev2::editor::close_image_viewer(decomp_viewer);
			ev2::destroy_image(ctx, rd.decomp);
		}
		
		rd.orig = ev2::create_image(ctx, img.w, img.h, 1,
			ev2::IMAGE_FORMAT_R8_UNORM, ev2::IMAGE_USAGE_SAMPLED_BIT);
		rd.decomp = ev2::create_image(ctx, img.w, img.h, 1,
			ev2::IMAGE_FORMAT_R8_UNORM, ev2::IMAGE_USAGE_SAMPLED_BIT);

		size_t nbytes = img.w * img.h * img.ch;

		ev2::UploadContext uc = ev2::begin_upload(ctx, nbytes, alignof(uint32_t));
		memcpy(uc.ptr, orig_img.bytes.get(), nbytes);
		ev2::ImageUpload up{
			.w = (uint32_t)orig_img.w,
			.h = (uint32_t)orig_img.h,
		};
		ev2::commit_image_uploads(ctx, uc, rd.orig, &up, 1);

		recompress_image();

		ev2::flush_uploads(ctx);

		orig_viewer = ev2::editor::open_image_viewer(rd.orig, "Original Image", "core://pipeline/grayscale.yaml");
		decomp_viewer = ev2::editor::open_image_viewer(rd.decomp, "Decompressed Image", "core://pipeline/grayscale.yaml");

		return 0;
	}

	int update() {
		grid_viewport->update(nullptr);

		glm::dvec2 scale = glm::dvec2(grid_viewport->get_size()); 

		glm::ivec2 grid_pos = glm::ivec2( 
			(ev2::editor::input().mouse_pos[0] - glm::dvec2(grid_viewport->get_pos())) / 
			glm::min(scale.x, scale.y) * 8.0);

		// The input flags stay set while a button is held, so only act on the
		// frame it goes down.
		const bool left = ev2::editor::input().left_mouse_pressed;
		const bool right = ev2::editor::input().right_mouse_pressed;
		const bool left_clicked = left && !prev_left;
		const bool right_clicked = right && !prev_right;
		prev_left = left;
		prev_right = right;

		if (glm::ivec2 pos = glm::clamp(grid_pos, glm::ivec2(0), glm::ivec2(7));
			pos == grid_pos && grid_viewport->is_hovered()) {
			uint8_t &bits = bit_grid[grid_pos.y][grid_pos.x];
			bool updated = false;
			if (left_clicked && bits < UniformQuantizer::max_bits) {
				bits = std::clamp(bits + 1, 0, 8);
				updated = true;
			} else if (right_clicked && bits > 0) {
				--bits;
				updated = true;
			}
			if (updated) {
				update_grid();
				recompress_image();
				ev2::flush_uploads(ctx);
			}
		}

		ev2::GfxPassInfo pass_info = {
			.target = grid_viewport->get_target(),
			.view = grid_viewport->get_view(),
			.name = "Bit Grid",
		};

		ev2::GfxPipelineID pipeline = ev2::load_graphics_pipeline(ctx, "cap://pipeline/grid.yaml");

		ev2::PassID pass = ev2::begin_gfx_pass(ctx, &pass_info);
		ev2::cmd_bind_gfx_pipeline(pass, pipeline);

		struct {
			uint32_t tex;
		} pc = {
			.tex = ev2::get_bindless_handle(ctx, rd.grid_tex)
		};
		ev2::cmd_push_constant(pass, pipeline, 0, sizeof(pc), &pc);

		ev2::cmd_custom(pass, [](VkCommandBuffer cmds){
			vkCmdDraw(cmds, 4, 1, 0, 0);
		});
		ev2::end_pass(ctx, pass);

		return 0;
	}
};

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
	if (ev2::editor::init(argc, argv, "transform coding", 1600, 800) < ev2::SUCCESS) {
		return EXIT_FAILURE;
	}
	add_project_mounts(ev2::editor::ctx());

	Image image {};
	for (int i = 1; i < argc; ++i) {
		if (true) {
			int desired_ch = STBI_grey;
			int ch;
			stbi_uc *bytes = stbi_load(argv[i], &image.w, &image.h, &ch, desired_ch);
			size_t nbytes = image.w * image.h * desired_ch;
			image.ch = desired_ch;
			image.bytes.reset(new uint8_t[nbytes]);
			memcpy(image.bytes.get(), bytes, nbytes);
			free(bytes);

			printf("Loaded image %s\n", argv[i]);
			break;
	  	}
	}

	if (!image.bytes) {
		printf("No image loaded\n");
		return EXIT_SUCCESS;
	}

	std::unique_ptr<App> app = std::make_unique<App>();

	app->set_image(std::move(image));

	for (;;) {
		if (ev2::editor::begin_frame() != ev2::SUCCESS)
			break;

		app->update();

		if (ev2::editor::end_frame() != ev2::SUCCESS)
			break;
	}
}
