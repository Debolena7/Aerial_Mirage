import openai
import os
import asyncio
import tiktoken

openai.api_key = "API KEY"

list_ids = [41872,578922,102820,514376,476704,267191,186042,415727,491497,183716,48555,48153,127955,724,110359,298396,410221,35963,223182,302107]


def ReadFile(filename):
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read the entire file content
        file_contents = file.read()
        # Print the content
        return file_contents
    
def WriteFile(i, Mesage):
    filename = "Generated_text/"+str(i)+"_GPT.txt"
    with open(filename, 'a') as file:
        # Append some text to the file
        file.write(Mesage)

def GetMessage_GPT(prompt):
    response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
              ]
    )
    #print(response)
    # print("GPT_3_trubo Response:\n")
    # print(response.choices[0].message.content)
    GPT_3_turbo_response = response.choices[0].message.content

    response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": prompt}
              ]
    )
    # print("GPT_4_turbo Response:\n")
    GPT_4_response = response.choices[0].message.content
    # print(response.choices[0].message.content)
    return GPT_3_turbo_response, GPT_4_response


for i in range(len(list_ids)):
    str_text = "PROMPTS_COCO_val/"+str(list_ids[i])+".txt"
    prompt = ReadFile(str_text)
    #print(prompt)    
    GPT_3_turbo_response, GPT_4_response = GetMessage_GPT(prompt)
    text = "GPT-3.5_Turbo Response:\n"
    WriteFile(list_ids[i],text)
    WriteFile(list_ids[i],GPT_3_turbo_response)
    text = "\n\n--------------------------------------------------------------------------------------------------------\n\n"
    WriteFile(list_ids[i],text)
    text = "GPT-4 Response:\n"
    WriteFile(list_ids[i],text)
    WriteFile(list_ids[i],GPT_4_response)
    # break

