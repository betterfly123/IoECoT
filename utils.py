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

    for i in range(len(content)):
        speaker_now = content[i]['speaker_list']
        dialog = content[i]['input']

        if model_name == 'chatglm' and data_name == 'IEMOCAP':
            labels = content[i]['label']
            len_id = len(labels)

            labels_test = labels
            while labels_test and labels_test[-1] in {'fearful', 'surprised', 'disgusted', 'other'}:
                len_id -= 1
                labels_test = labels[:len_id]

            speaker_now = speaker_now[:len_id]
            dialog = dialog[:len_id]

            if method_type == 'ioecot':
                start = max(0, len(dialog) - 6)
                dialog_content = speaker_now[start] + ": " + dialog[start] + "\n"
                for j in range(start + 1, len(dialog)):
                    dialog_content += speaker_now[j] + ": " + dialog[j] + "\n"
            else:
                dialog_content = speaker_now[0] + ": " + dialog[0] + "\n"
                for j in range(1, len(dialog)):
                    dialog_content += speaker_now[j] + ": " + dialog[j] + "\n"

            context.append(dialog_content)
            speaker.append(speaker_now[-1])
            ground_truth.append(emotion2label[labels_test[-1]])
            speaker_list.append(content[i]['speaker_list'])
            continue


        if data_name == 'IEMOCAP' and method_type == 'ioecot':

            start = max(0, len(dialog) - 11)
            dialog_content = speaker_now[start] + ": " + dialog[start] + "\n"
            for j in range(start + 1, len(dialog) - 1): 
                dialog_content += speaker_now[j] + ": " + dialog[j] + "\n"
        else:
            dialog_content = speaker_now[0] + ": " + dialog[0] + "\n"
            for j in range(1, len(dialog) - 1): 
                dialog_content += speaker_now[j] + ": " + dialog[j] + "\n"

        context.append(dialog_content)
        speaker.append(content[i]['speaker']) 
        ground_truth.append(emotion2label[content[i]['label']])
        speaker_list.append(content[i]['speaker_list'])

    return context, speaker, speaker_list, ground_truth
