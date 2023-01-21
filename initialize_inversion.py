"""
    automatically initialize the textual inversion with CLIP and no-parameter cross-attention
"""

import torch 
import os 
import argparse 

from PIL import Image 
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, CLIPTextModel 
from utils import imagenet_template, automatic_subjective_classnames


def embedding_generate(model, tokenizer, text_encoder, classnames, templates, device): 
    """
        pre-caculate the template sentence, token embeddings
    """
    with torch.no_grad(): 
        sentence_weights = [] 
        token_weights = [] 
        token_embedding_table = text_encoder.get_input_embeddings().weight.data
        for classname in classnames: 
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")  # tokenize
            texts = texts['input_ids'].to(device)
            class_embeddings = model.get_text_features(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            sentence_weights.append(class_embedding) 

            token_ids = tokenizer.encode(classname,add_special_tokens=False) 
            token_embedding_list = [] 
            for token_id in token_ids: 
                token_embedding_list.append(token_embedding_table[token_id])
            token_weights.append(torch.mean(torch.stack(token_embedding_list), dim=0))
            
        sentence_weights = torch.stack(sentence_weights, dim=1).to(device) 
        token_weights = torch.stack(token_weights, dim=0).to(device) 
    return sentence_weights, token_weights



def image_condition_embed_initialize(image_feature_list, sentence_embeddings, token_embeddings): 
    """
        no-parameter cross-attention: query: image, key: sentence, value: token 
    """
    inversion_emb_list = []
    for image_features in image_feature_list: 
        cross_attention = image_features @ sentence_embeddings 
        attention_probs = F.softmax(cross_attention, dim=-1) 
        inversion_emb = torch.matmul(attention_probs, token_embeddings)
        inversion_emb_list.append(inversion_emb)

    final_inversion = torch.mean(torch.stack(inversion_emb_list), dim=0)
    final_inversion = final_inversion / final_inversion.norm() 
    return final_inversion 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_path", default='./save', type=str)
    parser.add_argument("--data_path", default='./cat', type=str)
    args = parser.parse_args() 

    save_path = args.save_path
    data_path = args.data_path 
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained('./clip')
    model =  CLIPModel.from_pretrained('./clip') 
    text_encoder = CLIPTextModel.from_pretrained('./clip')
    processor = CLIPProcessor.from_pretrained('./clip') 

    sentence_embeddings, token_embeddings = embedding_generate(model, 
                                                                tokenizer, 
                                                                text_encoder,
                                                                automatic_subjective_classnames, 
                                                                imagenet_template, 
                                                                device)
    print('sentence embedding size: ', sentence_embeddings.size(), ' token embedding size: ', token_embeddings.size())

    image_feature_list = [] 
    name_list = os.listdir(data_path) 
    for name in name_list:
        image_path = os.path.join(data_path, name) 
        image = Image.open(image_path) 
        inputs = processor(images=image, return_tensors="pt") 
        image_features = model.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1)
        image_feature_list.append(image_features) 
    print('image size: ', len(image_feature_list)) 

    inversion_emb = image_condition_embed_initialize(image_feature_list, sentence_embeddings, token_embeddings)
    
    inversion_emb_dict = {"initialize": inversion_emb.detach().cpu()} 
    torch.save(inversion_emb_dict, os.path.join(save_path, 'initialize_emb.bin')) 



if __name__ == "__main__":
    main() 
