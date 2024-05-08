import torch
import numpy as np
from tqdm import tqdm
import model_converter

from ddpm import DDPMSampler
from clip import CLIP
from vae import VAE_Encoder
from vae import VAE_Decoder
from diffusion import Diffusion

width = 512
height = 512
latents_width = width//8
latents_height = height//8



def preload_models(ckpt_path):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device='cpu')
    encoder = VAE_Encoder()
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    
    decoder = VAE_Decoder()
    decoder.load_state_dict(state_dict['decoder'], strict=True)
    
    diffusion = Diffusion()
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)
    
    clip = CLIP()
    clip.load_state_dict(state_dict['clip'], strict=True)
    
    return {'clip':clip, 'encoder':encoder, 'decoder':decoder, 'diffusion':diffusion}



# image normalization
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
    return x



# use sine and cosine time embedding
def get_time_embedding(timestep):
    # (160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160) / 160)
    # (1, 160)
    x = torch.tensor([timestep])[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

 

@torch.no_grad
def generate(prompt, uncond_prompt, input_image=None, strength=0.8, do_cfg=True, 
             cfg_scale=7.5, sampler_name ='ddpm', n_inference=50, models={}, 
             seed=None, device=None, idle_device=None, tokenizer=None):
    if not (0 < strength <= 1):
        raise ValueError('stength must be between [0,1]')
    
    if idle_device:
        to_idle = lambda x:x.to(idle_device)
    else:
        to_idle = lambda x:x
    
    
    generator = torch.Generator(device=device)
    if seed is None:
        generate.seed()
    else:
        generator.manual_seed(seed)
    
    
    clip = models['clip']
    clip = clip.to(device)
    
    # classifier free guidence
    if do_cfg:
        # convert prompt into tokens
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
        # (batch, seq_len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # (batch, seq_len) -> (batch, seq_len, dim)
        cond_context = clip(cond_tokens)
        
        # convert prompt into tokens
        uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        # (batch, seq_len) -> (batch, seq_len, dim)
        uncond_context = clip(uncond_tokens)
        
        # concatenate prompt
        # (2, seq_len, dim)
        context = torch.cat([cond_context, uncond_context])
        
    else:
        # convert prompt into tokens
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
        # (batch, seq_len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # (1, seq_len, dim)
        context = clip(cond_tokens)
    
    to_idle(clip)
    
    if sampler_name == 'ddpm':
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(n_inference)
        
    else:
        raise ValueError(f'Unknown Sampler {sampler_name}')
    
    latents_shape = (1, 4, latents_height, latents_width)
    
    # if input_image is provider, do Image-to-Image task
    if input_image:
        encoder = models['encoder']
        encoder.to(device)
        
        input_image_tensor = input_image.resize((width, height))
        input_image_tensor = np.array(input_image_tensor)
        # (height, width, channel)
        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
        
        input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
        # (height, width, channel) -> (batch, height, width, channel)
        input_image_tensor = input_image_tensor.unsqueeze(0) 
        # (batch, height, width, channel) -> (batch, channel, height, width)
        input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
        
        # sample noise
        encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
        
        # run image through the VAE decoder
        latents = encoder(input_image_tensor, encoder_noise)
        
        sampler.set_strength(strength=strength)
        
        latents = sampler.add_noise(latents, sampler.timesteps[0])
        
        to_idle(encoder)
        
    # text-to-image task, start with random noise N(0, I)
    else:
        latents = torch.randn(latents_shape, generator=generator, device=device)
    
    # load the diffusion model
    diffusion = models['diffusion']
    diffusion.to(device)
    
    # denoising
    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        # (1, 320)
        time_embedding = get_time_embedding(timestep).to(device)
        
        # (batch, 4, latent_height, latent_width)
        model_input = latents
           
        if do_cfg:
            # (batch, 4, latent_height, latent_width) -> (2, batch, 4, latent_height, latent_width)
            model_input = model_input.repeat(2, 1, 1, 1)
            
        # predicted noise by UNet
        model_output = diffusion(model_input, context, time_embedding)
        
        if do_cfg:
            output_cond, output_uncond = model_output.chunk(2)
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
        
        # remove the predicted noise
        latents = sampler.remove_noise(timestep, latents, model_output)
        
    to_idle(diffusion)
    
    # load the decoder
    decoder = models['decoder']
    decoder.to(device)
    
    images = decoder(latents)
    to_idle(decoder)
    
    images = rescale(images, (-1, 1), (0,255), clamp=True)
    # (batch, channel, height, width) -> (batch, height, width, channel)
    images = images.permute(0, 2, 3, 1)
    images = images.to('cpu', torch.uint8).numpy()
    return images[0]
            