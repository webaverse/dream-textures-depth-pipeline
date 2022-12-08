###
### Based on: https://gist.github.com/carson-katri/f51532b9d5162928d5cacbaee081a799
###

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from typing import Union, List, Optional, Callable
import numpy as np
import diffusers
import PIL.Image
import torch

from flask import Flask, Response, request
import io


app = Flask(__name__)


# This pipeline is mostly copied from StableDiffusionInpaintPipeline and StableDiffusionImg2ImgPipeline.
def prepare_depth(depth):
	if isinstance(depth, PIL.Image.Image):
		depth = np.array(depth.convert('L'))
		depth = depth.astype(np.float32) / 255.0
	depth = depth[None, None]
	depth[depth < 0.5] = 0
	depth[depth >= 0.5] = 1
	depth = torch.from_numpy(depth)
	return depth

class GeneratorPipeline(diffusers.StableDiffusionInpaintPipeline):
	def prepare_depth_latents(
		self, depth, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
	):
		# resize the mask to latents shape as we concatenate the mask to the latents
		# we do that before converting to dtype to avoid breaking in case we're using cpu_offload
		# and half precision
		depth = torch.nn.functional.interpolate(
			depth, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
		)
		depth = depth.to(device=device, dtype=dtype)

		# duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
		depth = depth.repeat(batch_size, 1, 1, 1)
		depth = torch.cat([depth] * 2) if do_classifier_free_guidance else depth
		return depth

	#def prepare_img2img_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, image=None, timestep=None):
	def prepare_img2img_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
		image = image.to(device=device, dtype=dtype)
		init_latent_dist = self.vae.encode(image).latent_dist
		init_latents = init_latent_dist.sample(generator=generator)
		init_latents = 0.18215 * init_latents

		if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
			# expand init_latents for batch_size
			deprecation_message = (
				f'You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial'
				' images (`image`). Initial images are now duplicating to match the number of text prompts. Note'
				' that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update'
				' your script to pass as many initial images as text prompts to suppress this warning.'
			)
			deprecate('len(prompt) != len(image)', '1.0.0', deprecation_message, standard_warn=False)
			additional_image_per_prompt = batch_size // init_latents.shape[0]
			init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
		elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
			raise ValueError(
				f'Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts.'
			)
		else:
			init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

		# add noise to latents using the timesteps
		noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)

		# get latents
		init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
		latents = init_latents

		return latents

	def get_timesteps(self, num_inference_steps, strength, device):
		# get the original timestep using init_timestep
		offset = self.scheduler.config.get('steps_offset', 0)
		init_timestep = int(num_inference_steps * strength) + offset
		init_timestep = min(init_timestep, num_inference_steps)

		t_start = max(num_inference_steps - init_timestep + offset, 0)
		timesteps = self.scheduler.timesteps[t_start:]

		return timesteps, num_inference_steps - t_start

	@torch.no_grad()
	def __call__(
		self,
		prompt: Union[str, List[str]],
		depth_image: Union[torch.FloatTensor, PIL.Image.Image],
		image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
		strength: float = 0.8,
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: int = 50,
		guidance_scale: float = 7.5,
		negative_prompt: Optional[Union[str, List[str]]] = None,
		num_images_per_prompt: Optional[int] = 1,
		eta: float = 0.0,
		generator: Optional[torch.Generator] = None,
		latents: Optional[torch.FloatTensor] = None,
		output_type: Optional[str] = 'pil',
		return_dict: bool = True,
		callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
		callback_steps: Optional[int] = 1,
	):
		# 0. Default height and width to unet
		height = height or self.unet.config.sample_size * self.vae_scale_factor
		width = width or self.unet.config.sample_size * self.vae_scale_factor

		# 1. Check inputs
		self.check_inputs(prompt, height, width, callback_steps)

		# 2. Define call parameters
		batch_size = 1 if isinstance(prompt, str) else len(prompt)
		device = self._execution_device
		# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
		# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
		# corresponds to doing no classifier free guidance.
		do_classifier_free_guidance = guidance_scale > 1.0

		# 3. Encode input prompt
		text_embeddings = self._encode_prompt(
			prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
		)

		# 4. Prepare the depth image
		depth = prepare_depth(depth_image)

		if image is not None and isinstance(image, PIL.Image.Image):
			image = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess(image)

		# 5. set timesteps
		self.scheduler.set_timesteps(num_inference_steps, device=device)
		timesteps = self.scheduler.timesteps
		if image is not None:
			timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

		# 6. Prepare latent variables
		num_channels_latents = self.vae.config.latent_channels
		if image is not None:
			latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
			latents = self.prepare_img2img_latents(
				image,
				latent_timestep,
				batch_size,
				num_images_per_prompt,
				text_embeddings.dtype,
				device,
				generator
			)
		else:
			latents = self.prepare_latents(
				batch_size * num_images_per_prompt,
				num_channels_latents,
				height,
				width,
				text_embeddings.dtype,
				device,
				generator,
				latents,
			)

		# 7. Prepare mask latent variables
		depth = self.prepare_depth_latents(
			depth,
			batch_size * num_images_per_prompt,
			height,
			width,
			text_embeddings.dtype,
			device,
			generator,
			do_classifier_free_guidance,
		)

		# 8. Check that sizes of mask, masked image and latents match
		num_channels_depth = depth.shape[1]
		if num_channels_latents + num_channels_depth != self.unet.config.in_channels:
			raise ValueError(
				f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects'
				f' {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +'
				f' `num_channels_mask`: {num_channels_depth}'
				f' = {num_channels_latents+num_channels_depth}. Please verify the config of'
				' `pipeline.unet` or your `mask_image` or `image` input.'
			)

		# 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

		# 10. Denoising loop
		num_warmup_steps = len(timesteps) - num_inference_steps * 0 #self.scheduler.order
		# with self.progress_bar(total=num_inference_steps) as progress_bar:
		for i, t in enumerate(timesteps):
			# expand the latents if we are doing classifier free guidance
			latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

			# concat latents, mask, masked_image_latents in the channel dimension
			latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
			latent_model_input = torch.cat([latent_model_input, depth], dim=1)

			# predict the noise residual
			noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

			# perform guidance
			if do_classifier_free_guidance:
				noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
				noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

			# compute the previous noisy sample x_t -> x_t-1
			latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

			# call the callback, if provided
			if (i + 1) > num_warmup_steps and (i + 1): # % self.scheduler.order == 0:
				# progress_bar.update()
				if callback is not None and i % callback_steps == 0:
					callback(i, t, latents)

		# 11. Post-processing
		image = self.decode_latents(latents)

		# 12. Run safety checker
		image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

		# 13. Convert to PIL
		if output_type == 'pil':
			image = self.numpy_to_pil(image)

		if not return_dict:
			return (image, has_nsfw_concept)

		return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


@app.route('/depth/predict', methods=['OPTIONS', 'POST'])
def predict():
	if (request.method == 'OPTIONS'):
		print('got options 1')
		response = Response()
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		print('got options 2')
		return response

	# color_image = request.files['color']
	# image = PIL.Image.open(color_image) if color_image is not None else None
	color_req = request.files['color']
	image = np.asarray(PIL.Image.open(color_req) if color_req is not None else None)
	depth_req = request.files['depth']
	depth = np.asarray(PIL.Image.open(depth_req))
	# try:
	# 	depth_image = request.files['depth']
	# 	# depth = PIL.Image.open(depth_image)
	# 	depth = np.asarray(PIL.Image.open(depth_image)) # convert to ndarray
	# except:
	# 	depth = request.form['depth']

	prompt = request.args.get('prompt')
	negative_prompt = request.args.get('negative_prompt')

	pipe = GeneratorPipeline.from_pretrained(CONVERTED_MODEL_PATH)
	pipe = pipe.to(DEVICE)
	pipe.enable_attention_slicing()

	# image = np.asarray(PIL.Image.open(color_image) if color_image is not None else None) # convert to ndarray
	# depth = np.asarray(PIL.Image.open(depth_image)) # convert to ndarray

	strength = float(request.args.get('strength'))
	steps = int(request.args.get('steps'))
	seed = int(request.args.get('seed'))
	cfg_scale = float(request.args.get('cfg_scale'))
	use_negative_prompt = False

	# RNG
	generator = torch.Generator(DEVICE)
	if seed is None:
		seed = random.randrange(0, np.iinfo(np.uint32).max)
	generator = generator.manual_seed(seed)

	# Inference
	rounded_size = (
		int(8 * (depth.shape[1] // 8)),
		int(8 * (depth.shape[0] // 8)),
	)
	depth_image = PIL.ImageOps.flip(PIL.Image.fromarray(np.uint8(depth * 255), 'L')).resize(rounded_size)
	init_image = None if image is None else (PIL.Image.open(image) if isinstance(image, str) else PIL.Image.fromarray(image.astype(np.uint8))).convert('RGB').resize(rounded_size)
	output = pipe(
		prompt=prompt,
		depth_image=depth_image,
		# image=image.resize(depth.size),
		image=init_image,
		strength=strength,
		# width=depth.size[0],
		# height=depth.size[1],
		width=rounded_size[0],
		height=rounded_size[1],
		num_inference_steps=steps,
		guidance_scale=cfg_scale,
		negative_prompt=negative_prompt if use_negative_prompt else None,
		num_images_per_prompt=1,
		eta=0.0,
		generator=generator,
		latents=None,
	)

	# pipeline: Pipeline,
	# model = ''
	# scheduler = Scheduler
	# optimizations = Optimizations()
	# depth = NDArray | str,
	# image = NDArray | str | None,
	# strength = 0.1
	# steps = 1
	# seed = 1
	# cfg_scale = 0.5
	# use_negative_prompt = False
	# negative_prompt = 'test'
	# step_preview_mode = 'Fast' # not needed
	# # **kwargs

	# # RNG
	# generator = torch.Generator(DEVICE)
	# if seed is None:
	# 	seed = random.randrange(0, np.iinfo(np.uint32).max)
	# generator = generator.manual_seed(seed)

	# # Inference
	# image_arr = np.asarray(PIL.Image.open(COLOR_IMAGE) if COLOR_IMAGE is not None else None)
	# depth_arr = np.asarray(PIL.Image.open(DEPTH_IMAGE))
	# rounded_size = (
	# 	int(8 * (depth_arr.shape[1] // 8)),
	# 	int(8 * (depth_arr.shape[0] // 8)),
	# )
	# depth_image = PIL.ImageOps.flip(PIL.Image.fromarray(np.uint8(depth_arr * 255), 'L')).resize(rounded_size)
	# init_image = None if image_arr is None else (PIL.Image.open(image_arr) if isinstance(image_arr, str) else PIL.Image.fromarray(image_arr.astype(np.uint8))).convert('RGB').resize(rounded_size)
	# output = pipe( $# yield from pipe(
	# 	prompt=PROMPT,
	# 	depth_image=depth,
	# 	image=image,
	# 	strength=strength,
	# 	width=rounded_size[0],
	# 	height=rounded_size[1],
	# 	num_inference_steps=steps,
	# 	guidance_scale=cfg_scale,
	# 	negative_prompt=negative_prompt if use_negative_prompt else None,
	# 	num_images_per_prompt=1,
	# 	eta=0.0,
	# 	generator=generator,
	# 	latents=None,
	# 	output_type='pil',
	# 	return_dict=True,
	# 	callback=None,
	# 	callback_steps=1,
	# 	step_preview_mode=step_preview_mode
	# )

	print(output)

	output.images[0].save('test.png')

	img_byte_arr = io.BytesIO()
	output.images[0].save(img_byte_arr, format='PNG')
	img_byte_arr = img_byte_arr.getvalue()
	response = Response(img_byte_arr, headers={'Content-Type':'image/png'})
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Headers'] = '*'
	response.headers['Access-Control-Allow-Methods'] = '*'
	response.headers['Access-Control-Expose-Headers'] = '*'
	response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
	response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
	response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
	return response


if __name__ == '__main__':
	CONVERTED_MODEL_PATH = 'carsonkatri/stable-diffusion-2-depth-diffusers'
	DEVICE = 'cuda'

	# INIT_IMAGE = 'res.png'
	# DEPTH_IMAGE = 'depth.png'

	app.run(host='0.0.0.0', port=8081, threaded=True, debug=False)
