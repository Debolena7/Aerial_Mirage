from predict import Predictor  
import os
import json
import csv
import torch
torch.cuda.empty_cache()

# Create an instance of the Predictor class
debolena = Predictor()
#debolena.setup()


cat2label = {0:'ignored regions', 1:'pedestrian', 2:'people', 3:'bicycle', 4:'car', 5:'van', 6:'truck', 7:'tricycle', 8:'awning-tricycle', 9:'bus', 10:'motor', 11:'others'}


ann_dir = "/raid/ai20resch11003/VisDrone2019-DET-val/annotations/"
def create_prompt(img):
    img_prefix = os.path.splitext(os.path.basename(img))[0]
    ann_file = '{}.txt'.format(img_prefix)
    f = open(ann_dir+ann_file, 'r')
    file_contents = f.read().splitlines() #f.readlines()

    cat_ids = []
    for ann in file_contents:
        list_ann = [int(num) for num in ann.split(',')]
        #print("list ann: ", list_ann)
        cat_ids.append(list_ann[5]) ## taking index 5 for category id

    # print("cat_ids list: ", cat_ids)
    # print("len cat ids: ",len(cat_ids))
    # print("set(cat_ids): ", set(cat_ids))
    class_names=[]
    for i in set(cat_ids):
        class_names.append(cat2label[i])
    #print("obj list: ", class_names)
    prompt = 'The objects present in the image are '+str(class_names)+'. Describe the image briefly.'
    return prompt, class_names
  



##----------------------- TAKING IMAGES IN LOOP AND GENERATING DESCRIPTION OF EACH #-----------------------------------------------

top_p = 1.0  # Specify the desired value  ##default values given in predict.py
temperature = 0.2  # Specify the desired value  ##default values given in predict.py
max_tokens = 1024  # Specify the desired value  ##default values given in predict.py

csv_file=r'llava_visdrone/llava-v1-5-13b_briefly_obj_prompts.csv'
out_caps_list=[]

folder_dir = "/raid/ai20resch11003/VisDrone2019-DET-val/images/"  #"/raid/ai20resch11003/drone_data_IITH/images/" #"/raid/ai20resch11003/COCO/val2017/" ##
for img in os.listdir(folder_dir):
    obj_prompt, obj_list =  create_prompt(img)

    out_cap = debolena.predict(
    image=folder_dir+img,
    prompt=obj_prompt,
    top_p=top_p,
    temperature=temperature,
    max_tokens=max_tokens)

    #print("out cap: ", out_cap ) #<generator object Predictor.predict at 0x7f2a14c59380>
    caption = ""
    for generated_text in out_cap:
        caption += str(generated_text)
    #print("generated text: ", caption)

    #image_id=int(os.path.splitext(os.path.basename(img))[0])
    # print(type(image_id))
    # print(image_id)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img, obj_list, caption])  ##image_id

    caps_out = dict(img_name=img, class_names=obj_list, caption=caption)  ##image_id=image_id,
    out_caps_list.append(caps_out) 


with open('llava_visdrone/llava-v1-5-13b_briefly_obj_prompts.json', "a") as outf:
	json.dump(out_caps_list, outf)

    


