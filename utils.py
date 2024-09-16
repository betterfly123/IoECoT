import json

sentiment = ['neutral', 'negative', 'positive']
MELD_path_dict = {"ERC": "MELD_ERC.jsonl", "EIC": "MELD_new.jsonl"}
EmoryNLP_path_dict = {"ERC": "EmoryNLP_ERC.jsonl", "EIC": "EmoryNLP.jsonl"}
DailyDialog_path_dict = {"ERC": "DailyDialog_ERC.jsonl", "EIC": "DailyDialog.jsonl"}
IEMOCAP_path_dict = {"ERC": "IEMOCAP_ERC.jsonl", "EIC": "iemocap.jsonl"}

def data_contrust(data_path, emotion2label, data_name, method_type, model_name):
    with open(data_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    ground_truth = []
    speaker_list = []
    context = []
    speaker = []
    idx = 0
    for i in range(len(content)):
        speaker_now = content[i]['speaker_list']
        dialog = content[i]['input']
        if model_name == 'chatglm' and data_name == 'IEMOCAP':
            len_id = len(content[i]['label'])
            labels = content[i]['label']
            labels_test = labels
            while labels_test[-1] == 'fearful' or labels_test[-1] == 'surprised' or labels_test[-1] == 'disgusted' or \
                    labels_test[-1] == 'other':
                len_id = len_id - 1
                labels_test = labels[:len_id]
            speaker_now = speaker_now[:len_id]
            dialog = dialog[:len_id]
            if method_type == 'ioecot':
                dialog_content = speaker_now[len(dialog) - 6] + ": " + dialog[len(dialog) - 6] + '\n'
                for j in range(len(dialog) - 5, len(dialog)):
                    dialog_content = dialog_content + speaker_now[j] + ": " + dialog[j] + '\n'
            else:
                dialog_content = speaker_now[0] + ": " + dialog[0] + '\n'
                for j in range(1, len(dialog)):
                    dialog_content = dialog_content + speaker_now[j] + ": " + dialog[j] + '\n'
            context.append(dialog_content)
            speaker.append(speaker_now[-1])
            ground_truth.append(emotion2label[labels_test])
            speaker_list.append(content[i]['speaker_list'])
        else:
            if data_name == 'IEMOCAP' and method_type == 'ioecot':
                dialog_content = speaker_list[i][len(dialog) - 11] + ": " + dialog[len(dialog) - 11] + '\n'
                for j in range(len(dialog) - 10, len(dialog) - 1):
                    dialog_content = dialog_content + speaker_list[idx][j] + ": " + dialog[j] + '\n'
            else:
                dialog_content = speaker_now[0] + ": " + dialog[0] + '\n'
                for j in range(1, len(dialog)-1):
                  dialog_content = dialog_content + speaker_now[j] + ": " + dialog[j] + '\n'
            context.append(dialog_content)
            speaker.append(content[i]['speaker'])
            ground_truth.append(emotion2label[content[i]['label']])
            speaker_list.append(content[i]['speaker_list'])
        idx = idx + 1
    return context, speaker, speaker_list, ground_truth

def ERC_data_contrust(data_path, emotion2label, data_name, method_type):
    with open(data_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    ground_truth = []
    speaker_list = []
    context = []
    speaker = []
    utterance = []
    for i in range(len(content)):
        speaker_now = content[i]['speaker_list']
        dialog = content[i]['input']
        if data_name == 'IEMOCAP':
            len_id = len(content[i]['label'])
            labels = content[i]['label']
            labels_test = labels
            while labels_test[-1] == 'frustrated' or labels_test[-1] == 'surprised' or labels_test[-1] == 'disgusted' or labels_test[-1] == 'other':
                len_id = len_id - 1
                labels_test = labels[:len_id]
            speaker_now = speaker_now[:len_id]
            dialog = dialog[:len_id]
            if method_type == 'ioecot':
                dialog_content = speaker_now[len(dialog) - 6] + ": " + dialog[len(dialog) - 6] + '\n'
                for j in range(len(dialog) - 5, len(dialog)):
                    dialog_content = dialog_content + speaker_now[j] + ": " + dialog[j] + '\n'
            else:
                dialog_content = speaker_now[0] + ": " + dialog[0] + '\n'
                for j in range(1, len(dialog)):
                    dialog_content = dialog_content + speaker_now[j] + ": " + dialog[j] + '\n'
            context.append(dialog_content)
            speaker.append(speaker_now[-1])
            ground_truth.append(emotion2label[labels_test[-1]])
            utterance.append(dialog[-1])
            speaker_list.append(content[i]['speaker_list'])
        else:
            dialog_content = speaker_now[0] + ": " + dialog[0] + '\n'
            for j in range(1, len(dialog)):
              dialog_content = dialog_content + speaker_now[j] + ": " + dialog[j] + '\n'

            context.append(dialog_content)
            speaker.append(content[i]['speaker_list'][-1])
            ground_truth.append(emotion2label[content[i]['label'][-1]])
            utterance.append(dialog[-1])
            speaker_list.append(content[i]['speaker_list'])
    return context, utterance, speaker, speaker_list, ground_truth

def data_select(data_name, task_type):
    if data_name == "MELD":
        datasetpath = MELD_path_dict[task_type]
        emotion2label = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
        emo_list = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
        emo_prompt = "[neutral, surprise, fear, sadness, joy, disgust, anger]"
    elif data_name == "EmoryNLP":
        datasetpath = EmoryNLP_path_dict[task_type]
        emotion2label = {'joyful': 0, 'mad': 1, 'peaceful': 2, 'neutral': 3, 'sad': 4, 'powerful': 5, 'scared': 6}
        emo_list = ['joyful', 'mad', 'peaceful', 'neutral', 'sad', 'powerful', 'scared']
        emo_prompt = "[joyful, mad, peaceful, neutral, sad, powerful, scared]"
    elif data_name == "DailyDialog":
        datasetpath = DailyDialog_path_dict[task_type]
        emotion2label = {'other emotion': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5,
                         'surprise': 6}
        emo_list = ['other emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        emo_prompt = "[other emotion, anger, disgust, fear, happiness, sadness, surprise]"
    else:
        datasetpath = IEMOCAP_path_dict[task_type]
        emotion2label = {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3, 'frustrated': 4, 'excited': 5, 'fearful': 6,
                         'surprised': 7, 'disgusted': 8, 'other': 9}
        # emo_list = ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'excited', 'fearful', 'surprised', 'disgusted',
        #             'other']
        emo_list = ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'excited']
        # emo_prompt = "[angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted, other]"
        emo_prompt = "[angry, happy, sad, neutral, frustrated, excited]"
    return datasetpath, emotion2label, emo_list, emo_prompt

def info_select(emo_utterance, speaker):
    person = emo_utterance.split(":")[0]
    emotion_pre = emo_utterance.lower()
    sentiment_ans = ''
    for sen in sentiment:
      if emotion_pre.find(sen) != -1:
        sentiment_ans = sen
        break
    info = person + ": " + sentiment_ans
    return info

def create_emochain(output_struct, speaker):
    struct_list = output_struct.splitlines()
    emo_chain = 'The emotion of the dialogue history is:' + '\n'
    for i in range(len(struct_list)):
      info = info_select(struct_list[i], speaker)
      emo_chain = emo_chain + info + '\n'
    return emo_chain

