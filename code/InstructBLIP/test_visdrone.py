from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import os
import csv
import json


model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = model.to(device)
# url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# prompt = "What is unusual about this image?"

csv_file=r'output_visdrone/instructblip-vicuna-7b_visdrone.csv'

folder_dir = "/raid/ai20resch11003/VisDrone2019-DET-val/images/" 
prompt = "Describe the image in less than 30 words."   ##"Describe the image briefly."

out_caps_list=[]
for img in os.listdir(folder_dir):
    image = Image.open(folder_dir+img).convert('RGB')
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)


    outputs = model.generate(
        **inputs,
        do_sample=True, ##False,
        num_beams=5,
        max_length=1024,  ##256,  ##1024
        min_length=1,
        top_p=1.0, ##0.9, ##1.0
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=0.2 #1,  ## 0.2
    )
    
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    #image_id=int(os.path.splitext(os.path.basename(img))[0])
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img, generated_text])

    #print(img, generated_text)
    caps_out = dict(img_name=img, caption=generated_text) 
    out_caps_list.append(caps_out) 


with open('output_visdrone/instructblip-vicuna-7b_visdrone.json', "a") as outf:
	json.dump(out_caps_list, outf)

