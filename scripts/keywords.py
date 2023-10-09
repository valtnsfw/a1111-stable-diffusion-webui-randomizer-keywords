import re
from modules import scripts, script_callbacks, extra_networks, shared, sd_models, sd_samplers, processing, rng


operations = {
    "txt2img": processing.StableDiffusionProcessingTxt2Img,
    "img2img": processing.StableDiffusionProcessingImg2Img,
}
needs_hr_recalc = False
first_batch_entity_params = {}
processed_xyz_plot_images = 0


def is_debug():
    return shared.opts.data.get("randomizer_keywords_debug", False)


def recalc_hires_fix(p):
    def print_params(p):
        print(f"- width: {p.width}")
        print(f"- height: {p.height}")
        print(f"- hr_upscaler: {p.hr_upscaler}")
        print(f"- hr_second_pass_steps: {p.hr_second_pass_steps}")
        print(f"- hr_scale: {p.hr_scale}")
        print(f"- hr_resize_x: {p.hr_resize_x}")
        print(f"- hr_resize_y: {p.hr_resize_y}")
        print(f"- hr_upscale_to_x: {p.hr_upscale_to_x}")
        print(f"- hr_upscale_to_y: {p.hr_upscale_to_y}")

    if isinstance(p, processing.StableDiffusionProcessingTxt2Img):
        if is_debug():
            print("[RandomizerKeywords] Recalculating Hires. fix")
            print("Before:")
            print_params(p)

        for param in ["Hires upscale", "Hires resize", "Hires steps", "Hires upscaler"]:
            p.extra_generation_params.pop(param, None)

        # Don't want code duplication
        p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        p.rng = rng.ImageRNG((processing.opt_C, p.height // processing.opt_f, p.width // processing.opt_f), p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
        
        if is_debug():
            print("====================")
            print("After:")
            print_params(p)


def get_xyz_plot_axe_param_names() -> set[int]:
    xyz_plot_axe_param_names = set()
    for axe in {'x', 'y', 'z'}:
        xyz_plot_axe_param_names.add(f"xyz_plot_{axe}")
    return xyz_plot_axe_param_names


def is_xyz_plot() -> bool:
    for xyz_plot_axe_param_name in get_xyz_plot_axe_param_names():
        xyz_plot_axe = getattr(shared.state, xyz_plot_axe_param_name, None)
        if xyz_plot_axe is not None and len(xyz_plot_axe.values) > 0:
            return True
    return False


def xyz_plot_grid_size() -> int:
    size = 1
    for xyz_plot_axe_param_name in get_xyz_plot_axe_param_names():
        xyz_plot_axe = getattr(shared.state, xyz_plot_axe_param_name, None)
        if xyz_plot_axe is not None:
            size *= max(1, len(xyz_plot_axe.values))
    return size


class RandomizerKeywordSamplerParam(extra_networks.ExtraNetwork):
    def __init__(self, param_name, param_type, value_min=0, value_max=None, op_type=None, validate_cb=None, adjust_cb=None):
        super().__init__(param_name)
        self.param_type = param_type
        self.value_min = value_min
        self.value_max = value_max
        self.op_type = op_type
        self.validate_cb = validate_cb
        self.adjust_cb = adjust_cb

    def activate(self, p, params_list):
        if not params_list:
            return

        if self.op_type:
            ty = operations[self.op_type]
            if not isinstance(p, ty):
                return

        if self.name in first_batch_entity_params:
            value = first_batch_entity_params[self.name]
        else:
            value = params_list[0].items[0]
            value = self.param_type(value)

            first_batch_entity_params[self.name] = value

        if self.adjust_cb:
            value = self.adjust_cb(value, p)

        if isinstance(value, int) or isinstance(value, float):
            if self.value_min:
                value = max(value, self.value_min)
            if self.value_max:
                value = min(value, self.value_max)

        if self.validate_cb:
            error = self.validate_cb(value, p)
            if error:
                raise RuntimeError(f"Validation for '{self.name}' keyword failed: {error}")

        if is_debug():
            print(f"[RandomizerKeywords] Set SAMPLER option: {self.name} -> {value}")

        setattr(p, self.name, value)

        global needs_hr_recalc
        if self.name == "width" or self.name == "height" or self.name.startswith("hr_"):
            needs_hr_recalc = True

    def deactivate(self, p):
        pass


def validate_sampler_name(x, p):
    if isinstance(p, processing.StableDiffusionProcessingImg2Img):
        choices = sd_samplers.samplers_for_img2img
    else:
        choices = sd_samplers.samplers

    names = set(x.name for x in choices)

    if x not in names:
        return f"Invalid sampler '{x}'"
    return None


class RandomizerKeywordCheckpoint(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__("checkpoint")
        self.original_checkpoint_info = None

    def activate(self, p, params_list):
        global processed_xyz_plot_images

        if not params_list:
            return

        if self.original_checkpoint_info is None:
            self.original_checkpoint_info = shared.sd_model.sd_checkpoint_info

        params = params_list[0]
        assert len(params.items) > 0, "Must provide checkpoint name"

        name = params.items[0]
        info = sd_models.get_closet_checkpoint_match(name)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {name}")
          
        if processed_xyz_plot_images == 0:
            sd_models.reload_model_weights(shared.sd_model, info)
            if is_debug():
                print(f"[RandomizerKeywords] Set CHECKPOINT: {info.name}")

    def deactivate(self, p):
        if self.original_checkpoint_info is not None:
            # Disable for speedup
            # if is_debug():
                # print(f"[RandomizerKeywords] Reset CHECKPOINT: {self.original_checkpoint_info.name}")

            # sd_models.reload_model_weights(shared.sd_model, self.original_checkpoint_info)
            self.original_checkpoint_info = None


class Script(scripts.Script):
    def title(self):
        return "Randomizer Keywords"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def setup(self, p, *args, **kwargs):
        global first_batch_entity_params
        first_batch_entity_params = {}

        for xyz_plot_axe_param_name in get_xyz_plot_axe_param_names():
            if hasattr(shared.state, xyz_plot_axe_param_name):
                delattr(shared.state, xyz_plot_axe_param_name)
                
    def after_extra_networks_activate(self, p, *args, **kwargs):
        global all_params
        cleaned_prompts = []
        for prompt in p.all_prompts:
            for param in all_params:
                prompt = re.sub(f"<{param.name}:.*>", '', prompt)
            prompt = re.sub(r'\n+', '\n', prompt).strip()

            cleaned_prompts.append(prompt)
        
        p.all_prompts = cleaned_prompts
        
    def process_batch(self, p, *args, **kwargs):
        global needs_hr_recalc
        if needs_hr_recalc:
            recalc_hires_fix(p)

        needs_hr_recalc = False

    def postprocess_batch(self, p, *args, **kwargs):
        global first_batch_entity_params
        global processed_xyz_plot_images
        
        if is_xyz_plot():
            processed_xyz_plot_images += 1

        if not is_xyz_plot() or processed_xyz_plot_images == xyz_plot_grid_size():
            processed_xyz_plot_images = 0
            first_batch_entity_params = {}


# Sampler parameters that can be controlled. They are parameters in the Processing class.
sampler_params = [
    RandomizerKeywordSamplerParam("cfg_scale", float, 1),
    RandomizerKeywordSamplerParam("seed", int, -1),
    RandomizerKeywordSamplerParam("subseed", int, -1),
    RandomizerKeywordSamplerParam("subseed_strength", float, 0),
    RandomizerKeywordSamplerParam("sampler_name", str, validate_cb=validate_sampler_name),
    RandomizerKeywordSamplerParam("steps", int, 1),
    RandomizerKeywordSamplerParam("width", int, 64, adjust_cb=lambda x, p: x - (x % 8)),
    RandomizerKeywordSamplerParam("height", int, 64, adjust_cb=lambda x, p: x - (x % 8)),
    RandomizerKeywordSamplerParam("tiling", bool),
    RandomizerKeywordSamplerParam("restore_faces", bool),
    RandomizerKeywordSamplerParam("s_churn", float),
    RandomizerKeywordSamplerParam("s_tmin", float),
    RandomizerKeywordSamplerParam("s_tmax", float),
    RandomizerKeywordSamplerParam("s_noise", float),
    RandomizerKeywordSamplerParam("eta", float, 0),
    RandomizerKeywordSamplerParam("ddim_discretize", str),
    RandomizerKeywordSamplerParam("denoising_strength", float),

    # txt2img
    RandomizerKeywordSamplerParam("hr_scale", float, 1, op_type="txt2img"),
    RandomizerKeywordSamplerParam("hr_upscaler", str, op_type="txt2img"),
    RandomizerKeywordSamplerParam("hr_second_pass_steps", int, 1, op_type="txt2img"),
    RandomizerKeywordSamplerParam("hr_resize_x", int, 64, adjust_cb=lambda x, p: x - (x % 8), op_type="txt2img"),
    RandomizerKeywordSamplerParam("hr_resize_y", int, 64, adjust_cb=lambda x, p: x - (x % 8), op_type="txt2img"),

    # img2img
    RandomizerKeywordSamplerParam("mask_blur", float, op_type="img2img"),
    RandomizerKeywordSamplerParam("inpainting_mask_weight", float, op_type="img2img"),
]


other_params = [
    RandomizerKeywordCheckpoint(),
]

all_params = sampler_params + other_params

def on_app_started(demo, app):
    global all_params

    print(f"[RandomizerKeywords] Supported keywords: {', '.join([p.name for p in all_params])}")

    for param in all_params:
        extra_networks.register_extra_network(param)


def on_ui_settings():
    section = ('randomizer_keywords', "Randomizer Keywords")
    shared.opts.add_option("randomizer_keywords_debug", shared.OptionInfo(False, "Print debug messages", section=section))


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_settings(on_ui_settings)
