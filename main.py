import argparse
from run_eic import *
from run_erc import *
from utils import *
from local_model import *
from sklearn.metrics import classification_report


def run_cot(data_name, data_path, emotion2label, emo_list, emo_prompt, method_type, model_name, default_emo):
    preds = []
    data, speaker, speaker_list, ground_truth = data_contrust(data_path, emotion2label, data_name, method_type, model_name)
    if model_name == 'chatglm':
        if method_type == 'direct':
            preds = run_erc_direct_lm(data, ground_truth, emo_prompt, model_name, emo_list, default_emo)
        else:
            preds = run_erc_ioecot_lm(data, utterance, ground_truth, emo_prompt, model_name, emo_list, speaker, default_emo)
    else:
        if method_type == 'direct':
            preds = run_direct(data, ground_truth, emo_prompt, model_name, emo_list, default_emo)
        else:
            preds = run_ioecot(data, ground_truth, emo_prompt, model_name, emo_list, speaker, default_emo)
    print(classification_report(ground_truth, preds, digits=4))

def run_ERC_cot(data_name, data_path, emotion2label, emo_list, emo_prompt, method_type, model_name, default_emo):

    data, utterance, speaker, speaker_list, ground_truth = ERC_data_contrust(data_path, emotion2label, data_name, method_type)
    if model_name == 'chatglm':
        if method_type == 'direct':
            preds = run_erc_direct_lm(data, ground_truth, emo_prompt, model_name, emo_list, default_emo)
       else:
            preds = run_erc_ioecot_lm(data, utterance, ground_truth, emo_prompt, model_name, emo_list, speaker, default_emo)
    else:
        if method_type == 'direct':
            preds = run_erc_direct(data, ground_truth, emo_prompt, model_name, emo_list, default_emo)
       else:
            preds = run_erc_ioecot(data, utterance, ground_truth, emo_prompt, model_name, emo_list, speaker, default_emo)
    print(classification_report(ground_truth, preds, digits=4))


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True)
    parser.add_argument('--data_name', default=None, type=str, required=True)
    parser.add_argument('--data_path', default=None, type=str, required=True)
    parser.add_argument('--task_type', default=None, type=str, required=True)
    parser.add_argument('--method_type', default=None, type=str, required=True)
    parser.add_argument('--default_emo', default=None, type=int, required=True)
    args = parser.parse_args()

    datasetpath, emotion2label, emo_list, emo_prompt = data_select(args.data_name, args.task_type)
    data_path = args.data_path + args.task_type + '/' + datasetpath
    if args.task_type == 'EIC':
        run_cot(args.data_name, data_path, emotion2label, emo_list, emo_prompt, args.method_type, args.model_name,
                args.default_emo)
    else:
        run_ERC_cot(args.data_name, data_path, emotion2label, emo_list, emo_prompt, args.method_type, args.model_name,
                args.default_emo)

