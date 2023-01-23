"""
Plot the textual embedding and projection w_p Q distribution in stable diffusion.
"""

import os 
import argparse 

from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model_path", default='./ckpt', type=str) 
    parser.add_argument("--model_dim", default=768, type=int) 
    parser.add_argument("--lamda", default=5, type=int) 
    args = parser.parse_args() 

    
    text_encoder = CLIPTextModel.from_pretrained(
        os.path.join(args.model_path, "text_encoder")
    )
    embedding = text_encoder.get_input_embeddings().weight.clone().cpu() 
    print(embedding.size())
    embedding = embedding.detach().cpu().numpy() 
    mu_hat = np.mean(embedding.reshape(-1))
    std_hat = np.std(embedding.reshape(-1))
    print(mu_hat, std_hat)
    number = embedding.reshape(-1).shape[0]
    normal = np.random.normal(loc=0, scale=1 / args.model_dim * args.lamda, size = number)
    sampling = np.random.normal(loc=0, scale=std_hat * args.lamda, size = number) 
    
    
    
    pca = PCA(n_components=args.model_dim) 
    pca.fit(embedding) 
    pca = pca.components_.reshape(-1)
    
    # initialize the Q with norm(0, 0.5)
    cma = np.random.normal(loc=0, scale=0.5, size = number)
    
    # projection distribution with W_p Q
    normal = cma * normal 
    sampling = cma * sampling 

    cma_pca = np.random.normal(loc=0, scale=0.5, size = pca.shape[0]) 
    pca = cma_pca * pca


    kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
    embedding = embedding.reshape(-1)

    plt.hist(embedding, **kwargs, color='g', label='Textual Embedding') 
    plt.hist(normal, **kwargs, color='r', label='Random Norm') 
    plt.hist(pca, **kwargs, color='black', label='PCA') 
    plt.hist(sampling, **kwargs, color='b', label='Prior Norm') 
    
    plt.gca().set(ylabel='Frequency')
    plt.xlim(-0.1,0.1)
    plt.legend()
    plt.show()


if __name__ == '__main__': 
    main()