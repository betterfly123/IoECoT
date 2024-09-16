import os
import json
import time
import random
import requests
from utils import *
from openai import OpenAI



client = OpenAI(api_key='', base_url='')

instruction = "Complete the emotion recognition task by recognizing the emotion of the last utterance using the given dialogue history. The conversation history is :"

step_direct_f = 'Please give the emotion label of the last utterance can only be chosen from '
step_direct_l = ' and do not give the explanation.'

step_cot = 'Let’s think step by step'

step_personality_f = "Give the most accurate one-sentence short description of "
step_personality_m = "'s personality in the context of "
step_personality_l = "'s utterances in the history of the dialogue"

per_instruction_p = "Recognizing the emotions of utterance ("
per_instruction_l = ") based on the personality of "

per_step_predict_f = 'Please give the emotion label of the utterance ( '
per_step_predict_m = ' ) can only be chosen from '
per_step_predict_l = ' and do not give the explanation.'

step_struct = 'Please judge the sentiment of each utterance in the dialog history, noting that you can only choose from [neutral, negative, positive], reduce the judgment criteria for negative emotions. Output the utterance and the corresponding sentiment.'

def run_erc_direct(data, ground_truth, emo_prompt, model_name, emo_list, default_emo):
    preds = []
    num_id = 0
    for dialog in data:
        step_direct = step_direct_f + emo_prompt + step_direct_l
        input_direct = instruction + dialog + ' ' + step_direct
        response_direct = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_direct,
                }
            ],
            model=model_name,
            temperature=0
        )
        if response_direct.id == None:
            preds.append(default_emo)
            num_id = num_id + 1
            continue
        output_direct = response_direct.choices[0].message.content
        ans = output_direct.lower()
        pre_output = -1
        for i in range(len(emo_list)):
            if ans.find(emo_list[i]) != -1:
                pre_output = i
                break
            else:
                pre_output = -1
        if pre_output != -1:
            preds.append(pre_output)
        else:
            preds.append(default_emo)
        num_id = num_id + 1
    return preds

def run_erc_ioecot(data, utterance, ground_truth, emo_prompt, model_name, emo_list, speaker, default_emo):
    preds = []
    num_id = 0
    for dialog in data:
        step_personality = step_personality_f + speaker[num_id] + step_personality_m + speaker[
            num_id] + step_personality_l
        input_personality = "Dialog history: " + dialog + " " + step_personality
        response_personality = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_personality,
                }
            ],
            model=model_name,
            temperature=0
        )
        if response_personality.id == None:
            preds.append(default_emo)
            num_id = num_id + 1
            continue
        output_personality = response_personality.choices[0].message.content

        input_struct = "Dialog history: " + '\n' + dialog + step_struct
        response_struct = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_struct,
                }
            ],
            model=model_name,
            temperature=0
        )
        if response_struct.id == None:
            preds.append(default_emo)
            num_id = num_id + 1
            continue
        output_struct = response_struct.choices[0].message.content
        emochain = create_emochain(output_struct, speaker[num_id])

        per_instruction = per_instruction_p + utterance[num_id] + per_instruction_l + speaker[num_id] + "."
        step_cot = 'Let’s think step by step'
       input_cot = "Dialog history: " + dialog + ' ' + speaker[num_id] + "'s Personlity: "+ output_personality + " " + per_instruction + " " + step_cot
        response_cot = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_cot,

                }
            ],
            model=model_name,
            temperature=0
        )
        if response_cot.id == None:
            preds.append(default_emo)
            num_id = num_id + 1
            continue
        output_cot = response_cot.choices[0].message.content

        step_predict = per_step_predict_f + utterance[num_id] + per_step_predict_m + emo_prompt + per_step_predict_l
        input_predict = "Dialog history:\n" + dialog + ' ' + speaker[num_id] + "'s Personlity: " + output_personality \
                        + '\n' + "Emotional orientations of dialogue history:\n" + emochain + output_cot + '\n' \
                        + step_predict

        response_direct = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_predict,
                }
            ],
            model=model_name,
            temperature=0
        )
        if response_direct.id == None:
            preds.append(default_emo)
            num_id = num_id + 1
            continue
        output_direct = response_direct.choices[0].message.content
        ans = output_direct.lower()
        pre_output = -1
        for i in range(len(emo_list)):
            if ans.find(emo_list[i]) != -1:
                pre_output = i
                break
            else:
                pre_output = -1
        if pre_output != -1:
            preds.append(pre_output)
        else:
            preds.append(default_emo)
        print("predict: ", pre_output)
        print('Label: ', ground_truth[num_id])
        num_id = num_id + 1
    return preds

