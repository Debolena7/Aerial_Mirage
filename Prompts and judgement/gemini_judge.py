import pathlib
import textwrap
from pathlib import Path

import PIL.Image
import os
import json
import csv
import pandas as pd

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

genai.configure(api_key="API KEY")


safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

model_pro = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
model_pro_vision = genai.GenerativeModel('gemini-pro-vision', safety_settings=safety_settings)



#list_ids = [41872,578922,102820,514376,476704,267191,186042,415727,491497,183716,48555,48153,127955,724,110359,298396,410221,35963,223182]
#print(len(list_ids))

df = pd.read_excel('D:\LLaVA__\prompts\evaluation.xlsx') #, index_col=0)  

## taking first 20 image_ids
list_ids = df['img_id'][171:].to_list() #[:20]
list_names = df['img_names'][171:].to_list() #[:20]
# print(list_ids)
# exit(0)


def ReadFile_ModifyPrompt(filename):
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read the entire file content as a list
        file_contents = file.readlines()
        
        #print(file_contents)
        # print(type(file_contents))

        new1 = 'You are given an image as reference. Any misalignment or disagreement between the visual content of the image and textual content in "Candidate description" is considered as hallucination. '
        file_contents.insert(1, new1)

        join_str = ''.join(file_contents)
        #print("joined str:\n", join_str)

        return join_str
    


def ReadFile(filename):
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read the entire file content
        file_contents = file.read()
        #print(file_contents)
        return file_contents
    
def WriteFile(i, Mesage):
    filename = "D:\LLaVA__\prompts/Gemini_judge/"+str(i)+"_Gemini.txt"
    with open(filename, 'a') as file:
        # Append some text to the file
        file.write(Mesage)

def GetJudgement_Gemini_pro(prompt):

    response_pro = model_pro.generate_content(prompt)
    response_pro.resolve()
    #print("gemini-pro response: ", response_pro.text)
    return response_pro.text

def GetJudgement_Gemini_pro_vision(prompt, img):
    
    response_pro_vision = model_pro_vision.generate_content(contents = [prompt, img])
    response_pro_vision.resolve()
    return response_pro_vision.text




img_folder = "D:\COCO/val2017/"
for i in range(len(list_ids)):
    prompt_file = "D:/PROMPTS_hal_eval/"+str(list_ids[i])+".txt"

    prompt = ReadFile(prompt_file)
    response_pro = GetJudgement_Gemini_pro(prompt)
    text = "Gemini pro Response:\n"
    WriteFile(list_ids[i],text)
    WriteFile(list_ids[i],response_pro)
    text = "\n\n--------------------------------------------------------------------------------------------------------\n\n"
    WriteFile(list_ids[i],text)


    ##Gemini Pro-Vision:
    prompt_modified = ReadFile_ModifyPrompt(prompt_file)
    img = PIL.Image.open(img_folder+list_names[i]) ## read image  
    
    response_pro_vision = GetJudgement_Gemini_pro_vision(prompt_modified, img) 

    text = "Gemini pro-vision Response:\n"
    WriteFile(list_ids[i],text)
    WriteFile(list_ids[i],response_pro_vision)
