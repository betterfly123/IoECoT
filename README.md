
# IoECoT: Enhancing Zero-Shot Emotion Perception in Conversation (DASFAA 2025)



## 🔥 Overview

An ideal emotional dialogue system should be able to perceive emotions in conversations across unseen scenarios, so as to better extract user needs and guide response generation. However, due to the lack of sufficient training data in new scenarios, improving the zero-shot emotion perception capability of dialogue models remains a major challenge. In addition, existing studies mostly focus on a single task, which fails to provide a comprehensive assessment of a model’s emotion perception ability.

In this work, we propose **Internal-to-External Chain-of-Thought (IoECoT)** for **emotion perception in conversation**. IoECoT first extracts **personality information** of the target user from dialogue history as *internal factors*, while deriving **utterance polarity** as *external factors*. Guided by the internal factors, the model performs emotion perception based on the external factors. We evaluate IoECoT on both **Emotion Recognition in Conversation (ERC)** and **Emotion Inference in Conversation (EIC)**, enabling a more comprehensive assessment of emotion perception capabilities. Extensive experiments on multiple large language models and four benchmark datasets show that IoECoT consistently outperforms strong baselines, demonstrating its effectiveness in enhancing LLMs’ zero-shot emotion perception performance.








## 🛠️ Installation

### 1) Create environment

```bash
conda create -n ioecot python=3.10 -y
conda activate ioecot
```

### 2) Install dependencies

```bash
pip install -U torch transformers scikit-learn openai
```

---

## 🔑 LLM Backends

This repo supports two inference backends:

### (A) API-based LLMs


### (B) Local ChatGLM



---

## 🚀 Quick Start

### Run

```bash
python main.py \
  --model_name 'model_name' \
  --data_name 'MELD' \
  --data_path '/path/to/data_root/' \
  --task_type 'ERC' \
  --method_type 'ioecot' \
  --default_emo 5
```

## 📌 Arguments

| Argument        | Description                                                |
| --------------- | ---------------------------------------------------------- |
| `--model_name`  | LLM name or API model id             |
| `--data_name`   | Dataset name: `MELD`, `EmoryNLP`, `DailyDialog`, `IEMOCAP` |
| `--data_path`   | Root path           |
| `--task_type`   | `ERC` or `EIC`                                             |
| `--method_type` | `direct` or `ioecot`                                       |
| `--default_emo` | Default label index if parsing fails                       |

---

## 📁 Project Structure

```
.
├── main.py          # entry point
├── run_eic.py       # API-based EIC inference (direct / ioecot)
├── run_erc.py       # API-based ERC inference (direct / ioecot)
├── local_model.py   # local ChatGLM inference
├── utils.py         # dataset loading + emochain creation
└── run.sh           # example script
```

---



## 🧷 Citation

If you find this work useful, please cite:

```bibtex
@inbook{Xu_2026,
  title={Enhancing Zero-Shot Emotion Perception in Conversation Through the Internal-to-External Chain-of-Thought},
  ISBN={9789819538300},
  ISSN={1611-3349},
  url={http://dx.doi.org/10.1007/978-981-95-3830-0_14},
  DOI={10.1007/978-981-95-3830-0_14},
  booktitle={Database Systems for Advanced Applications},
  publisher={Springer Nature Singapore},
  author={Xu, Xingle and Feng, Shi and Wang, Daling and Zhang, Yifei and Yang, Xiaocui},
  year={2026},
  pages={209–224}
}
```



