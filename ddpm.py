# Linear Sampler

import torch
import numpy as np

class DDPMSampler:
    
    def __init__(self, generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0) # cummulative product
        self.one = torch.tensor(1.0)
        
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(num_training_steps)[::-1].copy())
    
    
    def set_inference_timesteps(self, num_inference=50):
        self.num_inference = num_inference
        step_ratio = self.num_training_steps // self.num_inference
        timesteps = (np.arange(num_inference) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    
    
    def _get_previous_timestep(self, timestep):
        prev_t = timestep - self.num_training_steps // self.num_inference
        return prev_t
    
    
    
    def _get_variance(self, timestep):
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        
        variance = (1-alpha_prod_t_prev) / (1-alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        
        return variance
    
    
    # Set how much noise to add to the input image. 
    def set_strength(self, strength=1):
        start_step = self.num_inference - int(self.num_inference * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
    
    
    # forward process 
    def add_noise(self, original_samples, timesteps):
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    
    # reverse process
    def remove_noise(self, timestep, latents, model_output):
        t = timestep
        prev_t = self._get_previous_timestep(t)
        
        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # compute the pred original sample
        pred_original_sample = (latents - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        
        # compute the coefficients for pred_orginal_sample and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
        
        # compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        
        variance = 0
        if t > 0:
            device = model_output.device
            
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = self._get_variance(t)**0.5 * noise
            
        # N(0, 1) -> N(mu, sigma^2)
        # x = mu + sigma*Z where Z ~ N(0,1)
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
        