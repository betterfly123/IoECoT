# main.py
import argparse
import os
from sklearn.metrics import classification_report

from utils import data_select, data_contrust, ERC_data_contrust
from llm_clients import OpenAIChatClient, ChatGLMClient
from runner import EmotionRunner, RunConfig


def build_client(model_name: str, local_model_path: str = ""):
    if model_name == "chatglm":
        if not local_model_path:
            raise ValueError("model_name=chatglm needs --local_model_path")
        return ChatGLMClient(model_path=local_model_path)
    return OpenAIChatClient(model=model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True) 
    parser.add_argument("--local_model_path", type=str, default="")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True) 
    parser.add_argument("--task_type", type=str, required=True)  
    parser.add_argument("--method_type", type=str, required=True)
    parser.add_argument("--default_emo", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    datasetpath, emotion2label, emo_list, emo_prompt = data_select(args.data_name, args.task_type)

    data_path = os.path.join(args.data_path, args.task_type, datasetpath)

    client = build_client(args.model_name, args.local_model_path)

    if args.task_type == "EIC":
        dialogs, speakers, speaker_list, ground_truth = data_contrust(
            data_path, emotion2label, args.data_name, args.method_type, args.model_name
        )
        utterances = None
    else:
        dialogs, utterances, speakers, speaker_list, ground_truth = ERC_data_contrust(
            data_path, emotion2label, args.data_name, args.method_type
        )

    cfg = RunConfig(
        task=args.task_type,
        method=args.method_type,
        emo_prompt=emo_prompt,
        emo_list=emo_list,
        default_emo=args.default_emo,
        temperature=0.0,
        max_tokens=None,
        verbose=args.verbose,
    )

    runner = EmotionRunner(client, cfg)
    preds = runner.run(dialogs, speakers=speakers if args.method_type != "direct" else None, utterances=utterances)

    print(classification_report(ground_truth, preds, digits=4))


if __name__ == "__main__":
    main()
