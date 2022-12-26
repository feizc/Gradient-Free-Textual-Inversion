import os
import torch
import argparse 
from torch import autocast

import PIL
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    
    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
    
    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds



def main():
    model_path = './ckpt' 
    learned_embeds_path = './save/learned_embeds.bin' 
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(model_path, 'tokenizer')
    )
    text_encoder = CLIPTextModel.from_pretrained(
        os.path.join(model_path, 'text_encoder')
    )
    load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    ).to(device) 

    prompt = 'city under the sun, painting, in a style of <sks>' 

    num_samples = 2 #@param {type:"number"}
    num_rows = 2 #@param {type:"number"}

    all_images = [] 
    for _ in range(num_rows):
        with autocast("cuda"):
            images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images
        all_images.extend(images)

    grid = image_grid(all_images, num_samples, num_rows) 
    grid.save('./1.png')




if __name__ == '__main__': 
    main()
