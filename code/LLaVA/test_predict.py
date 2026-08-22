from predict import Predictor  # Replace 'your_module' with the actual module name
import os
import json
import csv
import torch
torch.cuda.empty_cache()

# Create an instance of the Predictor class
debolena = Predictor()
#debolena.setup()

## trial with one image:
# # Specify the input parameters for the predict method
# image_path = "http://images.cocodataset.org/val2017/000000039769.jpg"
# prompt = "Describe the image."
# top_p = 0.8  # Specify the desired value
# temperature = 0.5  # Specify the desired value
# max_tokens = 100  # Specify the desired value

# out = debolena.predict(
#     image=image_path,
#     prompt=prompt,
#     top_p=top_p,
#     temperature=temperature,
#     max_tokens=max_tokens)

# sentence = ""
# for generated_text in out:
#     sentence += str(generated_text)
# print(sentence)

## with all images from a dir:
# Specify the input parameters for the predict method
prompt = "Describe the image in less than 30 words." #"Describe the image briefly."
top_p = 1.0  # Specify the desired value  ##default values given in predict.py
temperature = 0.2  # Specify the desired value  ##default values given in predict.py
max_tokens = 1024  # Specify the desired value  ##default values given in predict.py

csv_file=r'llava_output/llava-v1-5-13b_30words.csv'
out_caps_list=[]

folder_dir = "/raid/ai20resch11003/drone_data_IITH/images/"  ##"/raid/ai20resch11003/COCO/val2017/" 
for img in os.listdir(folder_dir):
    out_cap = debolena.predict(
    image=folder_dir+img,
    prompt=prompt,
    top_p=top_p,
    temperature=temperature,
    max_tokens=max_tokens)

    caption = ""
    for generated_text in out_cap:
        caption += str(generated_text)
    #print(caption)

    image_id=int(os.path.splitext(os.path.basename(img))[0])

    # print(type(image_id))
    # print(image_id)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_id, img, caption])

    caps_out = dict(image_id=image_id, img_name=img, caption=caption)
    out_caps_list.append(caps_out) 


with open('llava_output/llava-v1-5-13b_30words.json', "a") as outf:
	json.dump(out_caps_list, outf)

    
