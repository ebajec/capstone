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
#include "codec.h"

struct Image
{
	int w, h, ch;
	std::unique_ptr<uint8_t[]> bytes;
};

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

	Image orig_img;
	Image decomp_img;

	int sel_grid = 0;
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
		
		DCTImageCodec codec = DCTImageCodec::make<UniformDCTQuantizer>(&bit_grid[0][0]);

		size_t nb_comp_bytes = codec.get_compressed_size(orig_img.w, orig_img.h, orig_img.ch);
		std::unique_ptr<uint8_t[]> comp_bytes (new uint8_t[nb_comp_bytes]);

		codec.encode(orig_img.w, orig_img.h, orig_img.ch, orig_img.bytes.get(), comp_bytes.get());
		codec.decode(orig_img.w, orig_img.h, orig_img.ch, comp_bytes.get(), decomp_img.bytes.get());

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
			orig_viewer.reset();
			ev2::destroy_image(ctx, rd.orig);
		}
		if (rd.decomp.is_valid()) {
			decomp_viewer.reset();
			ev2::destroy_image(ctx, rd.decomp);
		}

		decomp_img = Image {
			.w = img.w,
			.h = img.h,
			.ch = img.ch,
			.bytes = std::make_unique<uint8_t[]>(img.w * img.h * img.ch),
		};

		const char *pipeline_path = nullptr;
		ev2::ImageFormat fmt;
		switch (img.ch) {
			case STBI_grey:
				fmt = ev2::IMAGE_FORMAT_R8_UNORM;
				pipeline_path = "core://pipeline/grayscale.yaml";
				break;
			case STBI_grey_alpha:
				abort();
				break;
			case STBI_rgb:
				abort();
				break;
			case STBI_rgb_alpha:
				fmt = ev2::IMAGE_FORMAT_RGBA8;
				pipeline_path = "core://pipeline/screen_quad.yaml";
				break;
			default:
				abort();
		}
		
		rd.orig = ev2::create_image(ctx, img.w, img.h, 1,
			fmt, ev2::IMAGE_USAGE_SAMPLED_BIT);
		rd.decomp = ev2::create_image(ctx, img.w, img.h, 1,
			fmt, ev2::IMAGE_USAGE_SAMPLED_BIT);

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

		orig_viewer = ev2::editor::open_image_viewer(rd.orig, "Original Image", pipeline_path);
		decomp_viewer = ev2::editor::open_image_viewer(rd.decomp, "Decompressed Image", pipeline_path);

		return 0;
	}

	int update()
	{
		ImGui::Begin("Editor");
		if (ImGui::Combo("Quality", &sel_grid, "DC\0Full\0""0.25 bpp\0""0.5 bpp\0""1 bpp\0""2 bpp\0""4 bpp")) {
			switch(sel_grid) {
				case 0:
					memcpy(bit_grid, BitTables::dc_only, 64); break;
				case 1:
					memcpy(bit_grid, BitTables::full, 64); break;
				case 2:
					memcpy(bit_grid, BitTables::bpp_0_25, 64); break;
				case 3:
					memcpy(bit_grid, BitTables::bpp_0_5, 64); break;
				case 4:
					memcpy(bit_grid, BitTables::bpp_1, 64); break;
				case 5:
					memcpy(bit_grid, BitTables::bpp_2, 64); break;
				case 6:
					memcpy(bit_grid, BitTables::bpp_4, 64); break;
			}
			update_grid();
			recompress_image();
		}
		ImGui::End();

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
			if (left_clicked && bits < UniformDCTQuantizer::max_bits) {
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
			int desired_ch = STBI_rgb_alpha;
			int ch;
			stbi_uc *bytes = stbi_load(argv[i], &image.w, &image.h, &ch, desired_ch);

			if (!bytes) {
				printf("Failed to load image %s\n", argv[i]);
				return EXIT_FAILURE;
			}

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
