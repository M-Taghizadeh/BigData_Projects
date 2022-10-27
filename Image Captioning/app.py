import torch 
import re 
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 
from PIL import Image
import matplotlib.pyplot as plt
import os


device='cpu'
encoder_checkpoint = "vlab/apps/image_processing/image_captioning/pretrained"
decoder_checkpoint = "vlab/apps/image_processing/image_captioning/pretrained"
model_checkpoint = "vlab/apps/image_processing/image_captioning/pretrained"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

# "nlpconnect/vit-gpt2-image-captioning"
# feature_extractor.save_pretrained('./pretrained/')
# tokenizer.save_pretrained('./pretrained/')
# model.save_pretrained('./pretrained/')


def get_image_captioning(img_path, max_length=64, num_beams=4):

    image = Image.open(img_path)
    image = image.convert('RGB')
    image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
    caption_ids = model.generate(image, max_length = max_length)[0]
    caption_text = clean_text(tokenizer.decode(caption_ids))

    return caption_text 
