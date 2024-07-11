import torch
from PIL import Image
from transformers import CLIPTokenizer
import matplotlib.pyplot as plt

from pipeline import generate, preload_models


def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    
    
def main():
    device = compute_device()
    
    pretrained_path = '../pretrained_models/Stable Diffusion/'
    # tokenizer
    tokenizer = CLIPTokenizer(pretrained_path+'tokenizer_vocab.json', merges_file=pretrained_path+'tokenizer_merges.txt')
    # load the pretrained model
    models = preload_models(pretrained_path+'v1-5-pruned-emaonly.ckpt')
    
    # text to image
    #prompt = 'A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.'
    prompt = 'change the car to white'
    uncond_prompt = ''
    do_cfg = True
    cfg_scale = 8
    
    # image to image
    input_image = None
    
    image_path = '../Datasets/flickr30k/flickr30k_images/5791244.jpg'
    input_image = Image.open(image_path)
    
    
    strength = 0.9
    
    sampler = 'ddpm'
    num_inference_steps = 50
    seed = 42
    
    
    # generate
    output_image = generate(
        prompt = prompt,
        uncond_prompt = uncond_prompt,
        input_image = input_image,
        strength = strength,
        do_cfg = do_cfg,
        cfg_scale = cfg_scale,
        sampler_name = sampler,
        n_inference = num_inference_steps,
        seed = seed,
        models = models,
        device = device,
        idle_device = 'cpu',
        tokenizer = tokenizer
    )
    
    

    # Plot the image
    plt.imshow(output_image)
    plt.axis('off')  # Turn off axis
    plt.show()

if __name__ == '__main__':
    main()
