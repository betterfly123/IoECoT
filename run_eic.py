import os
import json
import time
import random
import requests
from utils import *
from openai import OpenAI

client = OpenAI(api_key='', base_url='')


instruction = "Complete the emotion inference task by predicting the emotion of the next utterance using the given conversation history,note that the emotion of the next utterance is unknown. The conversation history is :"

step_direct_f = 'Please give the emotion label of the next utterance can only be chosen from '
step_direct_l = ' and do not give the explanation.'

step_cot = 'Let’s think step by step'

step_personality_f = "Give the most accurate one-sentence short description of "
step_personality_m = "'s personality in the context of "
step_personality_l = "'s utterances in the history of the dialogue"

per_instruction_f = "Complete an emotion inference task to predict the emotion of speaker "
per_instruction_m = " in the next utterance of the dialog, make inferences based on the given history of the dialog and on "
per_instruction_l = "'s personality"

per_step_predict_f = 'Please give the emotion label of the next utterance ( '
per_step_predict_m = ' ) can only be chosen from '
per_step_predict_l = ' and do not give the explanation.'

step_struct = 'Please judge the sentiment of each utterance in the dialog history, noting that you can only choose from [neutral, negative, positive], reduce the judgment criteria for negative emotions. Output the utterance and the corresponding sentiment.'

def run_direct(data, ground_truth, emo_prompt, model_name, emo_list, default_emo):
    preds = []
    num_id = 0
    for dialog in data:
        step_direct = step_direct_f + emo_prompt + step_direct_l
        input_direct = instruction + dialog + ' ' + step_direct
        print("input_direct: ", input_direct)
        response_direct = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_direct,
                }
            ],
            model=model_name,
        )
        output_direct = response_direct.choices[0].message.content
        print("output_direct: ", output_direct)
        print("**********************")
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
        print("----------------------")
        time.sleep(2)
    return preds


def run_ioecot(data, ground_truth, emo_prompt, model_name, emo_list, speaker, default_emo):
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
        )
        output_personality = response_personality.choices[0].message.content
        print('output_personality: ', output_personality)
        print("**********************")

        input_struct = "Dialog history: " + '\n' + dialog + step_struct
        print("input_struct:", input_struct)
        response_struct = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_struct,
                }
            ],
            model=model_name,
        )

        output_struct = response_struct.choices[0].message.content
        print('output_struct: ', output_struct)
        emochain = create_emochain(output_struct, speaker[num_id])

        per_instruction = per_instruction_f + speaker[num_id] + per_instruction_m + speaker[num_id] + per_instruction_l
        step_cot = 'Let’s think step by step'
         input_cot = per_instruction + ' ' + "Dialog history: " + dialog + ' ' + speaker[num_id] + "'s Personlity: " + output_personality + " " + emochain + " " + step_cot
        print("input_cot: ", input_cot)
        response_cot = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_cot,
                }
            ],
            model=model_name,
        )
        output_cot = response_cot.choices[0].message.content
        print('output_cot: ', output_cot)

        step_predict = per_step_predict_f + speaker[num_id] + per_step_predict_m + emo_prompt + per_step_predict_l

        input_predict = "Dialog history:\n" + dialog + ' ' + output_cot + step_predict

        print("input_predict: ", input_predict)
        response_direct = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_predict,
                }
            ],
            model=model_name,
        )
        output_direct = response_direct.choices[0].message.content
        print("output_direct: ", output_direct)
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
        time.sleep(2)
    return preds

