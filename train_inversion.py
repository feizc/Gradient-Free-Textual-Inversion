from turtle import forward
import cma 
import argparse 
import torch 
import os 
import numpy as np 
import copy 

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import torch.nn.functional as F 
from utils import TextualInversionDataset 
from tqdm import tqdm


class PipelineForward:
    def __init__(self, model_path, args, init_text_inversion=None, ):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(model_path, 'tokenizer')
        ) 
        self.text_encoder =  CLIPTextModel.from_pretrained(
            os.path.join(model_path, 'text_encoder')
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        ).to(args.device) 

        self.linear = torch.nn.Linear(args.intrinsic_dim, args.model_dim, bias=False).to(args.device) 

        if args.random_proj == 'normal': 
            embedding = self.text_encoder.get_input_embeddings().weight.clone().cpu()
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(args.intrinsic_dim) * args.sigma)
            # temp = intrinsic_dim - std_hat * std_hat
            # mu = mu_hat / temp
            # std = std_hat / np.sqrt(temp)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, mu, std) 
        
        # Add the placeholder token in tokenizer
        num_added_tokens = self.tokenizer.add_tokens(args.placeholder_token)  
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        # Convert the initializer_token, placeholder_token to ids
        token_ids = self.tokenizer.encode(args.initializer_token, add_special_tokens=False) 

        initializer_token_id = token_ids[0]
        placeholder_token_id = self.tokenizer.convert_tokens_to_ids(args.placeholder_token) 
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id] 

        print('convert text inversion: ', args.placeholder_token, 'in id: ', str(placeholder_token_id)) 
        self.placeholder_token_id = placeholder_token_id 
        self.placeholder_token = args.placeholder_token
        self.init_prompt = token_embeds[initializer_token_id].to(args.device) 
        self.num_call = 0 

        train_dataset = TextualInversionDataset(
            data_root=args.train_data_dir,
            tokenizer=self.tokenizer,
            size=args.resolution,
            placeholder_token=args.placeholder_token,
            repeats=args.repeats*10,
            learnable_property=args.learnable_property,
            center_crop=args.center_crop,
            set="train",
        )
        self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.repeats, shuffle=True)
        self.batch_size = args.repeats
        self.device = args.device
        print('load data length: ', len(self.dataloader))

        self.best_inversion = None 
        self.best_loss = 1000


    def eval(self, prompt_embedding): 
        self.num_call += 1 
        pe_list = []
        if isinstance(prompt_embedding, list):  # multiple queries
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32).to(self.device)  # z 
                with torch.no_grad():
                    z = self.linear(z)  # Az 
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(z)

        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32).to(self.device)  # z
            with torch.no_grad():
                prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            pe_list.append(prompt_embedding)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        
        loss_list = [] 
        print('begin to calculate loss') 
        for pe in tqdm(pe_list): 
            token_embeds = self.text_encoder.get_input_embeddings().weight.data 
            pe.to(self.text_encoder.get_input_embeddings().weight.dtype)
            token_embeds[self.placeholder_token_id] = pe 
            loss = calculate_mse_loss(self.pipe, self.dataloader, self.device) 
            if loss < self.best_loss: 
                print('update loss: ', loss)
                self.best_loss = loss
                self.best_inversion = pe
            loss_list.append(loss)

        return loss_list 


    def save(self, output_path): 
        learned_embeds_dict = {self.placeholder_token: self.best_inversion.detach().cpu()}
        torch.save(learned_embeds_dict, os.path.join(output_path, "learned_embeds.bin"))



def calculate_mse_loss(image_generator, dataloader, device): 
    # print(image_generator.text_encoder.get_input_embeddings().weight.data[49408]) 
    noise_scheduler = DDPMScheduler.from_config('./ckpt/scheduler') 
    loss_cum = .0 
    with torch.no_grad(): 
        for batch in dataloader: 
            # Convert images to latent space
            latents = image_generator.vae.encode(batch["pixel_values"].to(device)).latent_dist.sample().detach()
            latents = latents * 0.18215 

            # Sample noise that we'll add to the latents
            noise = torch.randn(latents.shape).to(latents.device)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()


            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = image_generator.text_encoder(batch["input_ids"].to(device))[0]
            
            # Predict the noise residual
            noise_pred = image_generator.unet(noisy_latents, timesteps, encoder_hidden_states).sample 

            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean() 
            loss_cum += loss.item() 

    return loss_cum / len(dataloader) 





def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsic_dim", default=500, type=int)
    parser.add_argument("--k_shot", default=16, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--budget", default=8000, type=int)
    parser.add_argument("--popsize", default=20, type=int) # number of candidates 
    parser.add_argument("--bound", default=0, type=int)
    parser.add_argument("--sigma", default=1, type=float)
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--print_every", default=50, type=int)
    parser.add_argument("--eval_every", default=100, type=int)
    parser.add_argument("--alg", default='CMA', type=str)
    parser.add_argument("--random_proj", default='normal', type=str)
    parser.add_argument("--model_dim", default=768, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--loss_type", default='ce', type=str)
    parser.add_argument("--cat_or_add", default='add', type=str)
    parser.add_argument("--device", default= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--parallel", default=False, type=bool, help='Whether to allow parallel evaluation')
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default='<sks>',
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", 
        type=str, 
        default='painting', 
        help="A token to use as initializer word."
    )
    parser.add_argument(
        "--inference_framework",
        default='pt',
        type=str,
        help='''Which inference framework to use. 
            Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
    )
    parser.add_argument(
        "--onnx_model_path",
        default=None,
        type=str,
        help='Path to your onnx model.'
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='./data',
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--learnable_property", 
        type=str, 
        default="style", 
        help="Choose between 'object' and 'style'"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    ) 
    parser.add_argument("--repeats", type=int, default=5, help="How many times to repeat the training data.")

    args = parser.parse_args() 
    
    cma_opts = {
        'seed': args.seed,
        'popsize': args.popsize,
        'maxiter': args.budget if args.parallel else args.budget // args.popsize,
        'verbose': -1,
    }

    if args.bound > 0:
        cma_opts['bounds'] = [-1 * args.bound, 1 * args.bound] 

    pipeline_forward = PipelineForward(model_path='./ckpt', args=args)

    es = cma.CMAEvolutionStrategy(args.intrinsic_dim * [0], args.sigma, inopts=cma_opts) 
    while not es.stop(): 
        solutions = es.ask() # (popsize, intrinsic_dim)
        fitnesses = pipeline_forward.eval(solutions) 
        es.tell(solutions, fitnesses) 
    
    pipeline_forward.save('./save')


if __name__ == "__main__":
    main()
