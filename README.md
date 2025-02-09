# Randomizer Keywords

This extension for stable-diffusion-webui adds some keywords using the extra networks syntax to allow randomization of parameters when combined with the [Dynamic Prompts](https://github.com/adieyal/sd-dynamic-prompts/tree/main/sd_dynamic_prompts) extension.

## Example

When used with Dynamic Prompts, this prompt will pick from a random checkpoint each batch:

```
{<checkpoint:animefull-latest>|<checkpoint:wd15-beta1-fp32>}
```

And you can use random choices inside keywords too:

```
<width:{512|768}>, <height:{512|768}>
```

You can also assemble a wildcard list containing text like model names to choose from, and deploy it inside the keywords:

```
<addnet_model_1:__artist_loras__>
```

## List of Keywords

This extension adds the following special keywords to be used in prompts:

- `<checkpoint:animefull-latest.ckpt>` - SD Checkpoint
- `<cfg_scale:7>` - CFG Scale
- `<seed:1>` - Seed
- `<subseed:1>` - Subseed
- `<subseed_strength:1>` - Subseed Strength
- `<sampler_name:Euler a>` - Sampler Name
- `<scheduler:Automatic>` - Scheduler Name
- `<steps:20>` - Sampling Steps
- `<width:512>` - Width
- `<height:512>` - Height
- `<tiling:true>` - Tiling
- `<restore_faces:true>` - Restore Faces
- `<s_churn:true>` - Sigma Churn
- `<s_tmin:true>` - Sigma Min
- `<s_tmax:true>` - Sigma Max
- `<s_noise:true>` - Sigma Noise
- `<eta:512>` - Eta
- `<ddim_discretize:quad>` - DDIM Discretize
- `<denoising_strength:0.7>` - Denoising Strength
- `<hr_upscaler:Latent>` - Hires. Fix Upscaler (txt2img)
- `<hr_second_pass_steps:10>` - Hires. Fix Steps (txt2img)
- `<mask_blur:2>` - Mask Blur (img2img)
- `<inpainting_mask_weight:2>` - Inpainting Mask Weight (img2img)

**NOTE**: These keywords will be applied *per batch*, not per prompt.

## FORK

I'm not a Python expert but I fixed some issues as best as I could.

1. `width` and `height` keywords are working again.
2. 1. Keywords are updated and applied once per batch. They were applied per batch, but updated per prompt before.
2. 2. Keywords are updated and applied once per `XYZ Plot` despite `Batch Count`. Use `Infinite Generation` instead of setting `Batch Count` therefore keywords will be updated and applied per generation.
3. Keywords are removed from the prompt once they are applied. They used to stay in the prompt before, distracting you.
