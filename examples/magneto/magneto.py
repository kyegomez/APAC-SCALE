import torch 
from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.component.embedding import PositionalEmbedding
from transformers import T5Tokenizer, CLIPProcessor, CLIPModel,
from PIL import Image
from torch.nn import Embedding, Module
import bitsandbytes as bnb



class MagnetoTokenizer:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion-2B-s32B-b82k")

        self.tokenize = T5Tokenizer.from_pretrained(
            "t5-large",
            additional_special_tokens=["<image>", "</image>"],
            extra_ids = 0,
            model_max_length = 1984
        )
        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])

    def tokenize_texts(self, texts):
        texts = self.tokenize(texts, return_tensors="pt", padding=True, truncation=True).input_ids
        #add image tokens to text 
        image_tokens = torch.tensor([[self.im_idx, self.im_end_idx]] * texts.shape[0])
        return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts
    
    def tokenize_images(self, images):
        return self.processor(images=images, return_tensors="pt").pixel_values
    
    def tokenize(self, sample):
        text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
        attention_mask = text_tokens != self.tokenizer.pad_token_id
        dummy_image_features = torch.ones((text_tokens.shape[0], 64))
        attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
        return {
            'text_tokens': text_tokens,
            'images': self.tokenize_images(sample["image"]),
            'labels': only_text_tokens,
            'attention_mask': attention_mask
        }
    

class Magneto(Module):
    def __init__(self):
        self.clip_model