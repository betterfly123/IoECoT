import torch
from utils import *
from transformers import AutoTokenizer, AutoModel


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

def get_model():
    device = torch.device()
    tokenizer = AutoTokenizer.from_pretrained("", trust_remote_code=True)
    model = AutoModel.from_pretrained("", trust_remote_code=True).to(device)
    model = model.eval()
    return model, tokenizer


def run_erc_direct_lm(data, ground_truth, emo_prompt, model_name, emo_list, default_emo):
    model, tokenizer = get_model()
    preds = []
    num_id = 0
    for dialog in data:
        step_direct = step_direct_f + emo_prompt + step_direct_l
        input_direct = instruction + dialog + ' ' + step_direct
        output_direct, history = model.chat(tokenizer, input_direct, history=[], temperature=0.01)
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

def run_erc_ioecot_lm(data, utterance, ground_truth, emo_prompt, model_name, emo_list, speaker, default_emo):
    model, tokenizer = get_model()
    preds = []
    num_id = 0
    for dialog in data:
        step_personality = step_personality_f + speaker[num_id] + step_personality_m + speaker[
            num_id] + step_personality_l
        input_personality = "Dialog history: " + dialog + " " + step_personality
        output_personality, history = model.chat(tokenizer, input_personality, history=[], temperature=0.01)


        input_struct = "Dialog history: " + '\n' + dialog + step_struct
        output_struct, history = model.chat(tokenizer, input_struct, history=[], temperature=0.01, max_length=1000)
        emochain = create_emochain(output_struct, speaker[num_id])


        per_instruction = per_instruction_p + utterance[num_id] + per_instruction_l + speaker[num_id] + "."
        step_cot = 'Let’s think step by step'
       input_cot = "Dialog history: " + dialog + ' ' + speaker[num_id] + "'s Personlity: " \
                    + output_personality + " " + per_instruction + " " + step_cot
       output_cot, history = model.chat(tokenizer, input_cot, history=[], temperature=0.01)
        print('output_cot: ', output_cot)
        print("**********************")

        step_predict = per_step_predict_f + utterance[num_id] + per_step_predict_m + emo_prompt + per_step_predict_l
        input_predict = "Dialog history:\n" + dialog + ' ' + speaker[num_id] + "'s Personlity: " + output_personality \
                        + '\n' + "Emotional orientations of dialogue history:\n" + emochain + output_cot + '\n'\
                        + step_predict
        output_direct, history = model.chat(tokenizer, input_predict, history=[], temperature=0.01)
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




