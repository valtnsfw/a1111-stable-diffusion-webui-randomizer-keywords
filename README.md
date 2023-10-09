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

**NOTE**: These keywords will be applied *per-batch*, not per-prompt.

## FORK

I'm not a Python expert but I fixed some issues as best as I could.

1. `width` and `height` keywords are working again.
2. Image metadata parameters are updated once per batch.
Imagine you have a prompt `<cfg_scale:{5|10}>` with `Batch Size` equal `2`. In this case, your first image may have `<cfg_scale:5>` both in the prompt and in the image metadata. While the second image may have `<cfg_scale:10>` in there. But the actually applied value is from the first image - `5`. So, I've fixed it and now you'll see `<cfg_scale:10>` in the prompt, but `CFG Scale: 5` in the metadata. Metadata parameters are the correct one. Just clear the prompt keywords if you want to reproduce an image.
3. If you generate a several batches of images with keywords, every batch will have its own parameters applied. But if you add XYZ Plot to that generation, keywords for the whole grid will be constant. Therefore I highly recommend you using `Infinite Generation` instead of setting `Batch Count` if you want keywords to apply per batch.
