# Gradient-Free Textual Inversion 

Gradient-Free Textual Inversion for Personalized Text-to-Image Generation. 
We introduce to use evolution strategy from [OpenAI](https://openai.com/blog/evolution-strategies/) without gradient to optimize the text embeddings. 
Our implementation is totally compatible with [diffusers](https://github.com/huggingface/diffusers) and stable diffusion model.

## What does this repo do? 

Current personalized text-to-image approaches, which learn to bind a unique identifier with specific subjects or styles in a few given images, usually incorporate a special word and tune its embedding parameters through gradient descent. 
It is natural to question that can we optimize the textual inversions by only accessing the inference of models?  As only requiring the forward computation to determine the textual inversion retains the benefits of efficient computation and safe deployment. 
Hereto, we introduce a gradient-free framework to optimize the continuous textual inversion in personalized text-to-image generation. 




