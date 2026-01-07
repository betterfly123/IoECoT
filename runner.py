# runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal
import re

from prompts import (
    STEP_DIRECT_F, STEP_DIRECT_L, STEP_COT,
    STEP_PERSONALITY_F, STEP_PERSONALITY_M, STEP_PERSONALITY_L,
    STEP_STRUCT,
    ERC_INSTRUCTION, ERC_DIRECT_TARGET, ERC_PER_INSTR_P, ERC_PER_INSTR_L,
    ERC_STEP_PRED_F, ERC_STEP_PRED_M, ERC_STEP_PRED_L,
    EIC_INSTRUCTION, EIC_DIRECT_TARGET, EIC_PER_INSTR_F, EIC_PER_INSTR_M, EIC_PER_INSTR_L,
    EIC_STEP_PRED_F, EIC_STEP_PRED_M, EIC_STEP_PRED_L,
)
from utils import create_emochain
from llm_clients import BaseChatClient


TaskType = Literal["ERC", "EIC"]
MethodType = Literal["direct", "ioecot"]


def extract_emotion_index(text: str, emo_list: List[str], default_emo: int) -> int:

    t = (text or "").lower()


    order = sorted(list(enumerate(emo_list)), key=lambda x: len(x[1]), reverse=True)

    for idx, label in order:
        pat = r"\b" + re.escape(label.lower()) + r"\b"
        if re.search(pat, t):
            return idx


    for idx, label in order:
        if label.lower() in t:
            return idx

    return default_emo


@dataclass(frozen=True)
class RunConfig:
    task: TaskType
    method: MethodType
    emo_prompt: str
    emo_list: List[str]
    default_emo: int
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    verbose: bool = False


class EmotionRunner:
    def __init__(self, client: BaseChatClient, cfg: RunConfig):
        self.client = client
        self.cfg = cfg

    # ---------- prompt builders ----------
    def _direct_prompt(self, dialog: str) -> str:
        if self.cfg.task == "ERC":
            instruction = ERC_INSTRUCTION
            target = ERC_DIRECT_TARGET
        else:
            instruction = EIC_INSTRUCTION
            target = EIC_DIRECT_TARGET

        step_direct = f"{STEP_DIRECT_F}{target}" + STEP_DIRECT_L.format(emo_prompt=self.cfg.emo_prompt)
        return instruction + dialog + " " + step_direct

    def _personality_prompt(self, dialog: str, speaker: str) -> str:
        step_personality = (
            STEP_PERSONALITY_F + speaker + STEP_PERSONALITY_M + speaker + STEP_PERSONALITY_L
        )
        return "Dialog history: " + dialog + " " + step_personality

    def _struct_prompt(self, dialog: str) -> str:
        return "Dialog history:\n" + dialog + STEP_STRUCT

    def _cot_prompt(self, dialog: str, speaker: str, personality: str, emochain: str, utterance: Optional[str]) -> str:
        if self.cfg.task == "ERC":
            assert utterance is not None
            per_instruction = ERC_PER_INSTR_P + utterance + ERC_PER_INSTR_L + speaker + "."

            return (
                "Dialog history: " + dialog + " "
                + f"{speaker}'s Personality: " + personality + " "
                + per_instruction + " " + STEP_COT
            )


        per_instruction = EIC_PER_INSTR_F + speaker + EIC_PER_INSTR_M + speaker + EIC_PER_INSTR_L
        return (
            per_instruction + " "
            + "Dialog history: " + dialog + " "
            + f"{speaker}'s Personality: " + personality + " "
            + emochain + " "
            + STEP_COT
        )

    def _predict_prompt(self, dialog: str, speaker: str, personality: str, emochain: str, cot: str, utterance: Optional[str]) -> str:
        if self.cfg.task == "ERC":
            assert utterance is not None
            step_predict = (
                ERC_STEP_PRED_F + utterance + ERC_STEP_PRED_M + self.cfg.emo_prompt + ERC_STEP_PRED_L
            )
            return (
                "Dialog history:\n" + dialog + " "
                + f"{speaker}'s Personality: " + personality + "\n"
                + "Emotional orientations of dialogue history:\n" + emochain
                + cot + "\n"
                + step_predict
            )


        step_predict = (
            EIC_STEP_PRED_F + speaker + EIC_STEP_PRED_M + self.cfg.emo_prompt + EIC_STEP_PRED_L
        )
        return "Dialog history:\n" + dialog + " " + cot + " " + step_predict

    # ---------- runners ----------
    def run_direct(self, dialogs: List[str]) -> List[int]:
        preds: List[int] = []
        for i, dialog in enumerate(dialogs):
            prompt = self._direct_prompt(dialog)
            out = self.client.chat(prompt, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
            pred = extract_emotion_index(out, self.cfg.emo_list, self.cfg.default_emo)
            preds.append(pred)

            if self.cfg.verbose:
                print(f"[{i}] prompt:\n{prompt}\n---\nout:\n{out}\n=> pred={pred}\n{'='*40}")
        return preds

    def run_ioecot(self, dialogs: List[str], speakers: List[str], utterances: Optional[List[str]] = None) -> List[int]:
        preds: List[int] = []
        for i, dialog in enumerate(dialogs):
            speaker = speakers[i]
            utter = utterances[i] if utterances is not None else None

            # 1) personality
            p_prompt = self._personality_prompt(dialog, speaker)
            personality = self.client.chat(p_prompt, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)

            # 2) struct -> emochain
            s_prompt = self._struct_prompt(dialog)
            struct_out = self.client.chat(s_prompt, temperature=self.cfg.temperature, max_tokens=1000)
            emochain = create_emochain(struct_out, speaker)

            # 3) CoT
            c_prompt = self._cot_prompt(dialog, speaker, personality, emochain, utter)
            cot_out = self.client.chat(c_prompt, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)

            # 4) predict
            pred_prompt = self._predict_prompt(dialog, speaker, personality, emochain, cot_out, utter)
            pred_out = self.client.chat(pred_prompt, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)

            pred = extract_emotion_index(pred_out, self.cfg.emo_list, self.cfg.default_emo)
            preds.append(pred)

            if self.cfg.verbose:
                print(f"[{i}] personality:\n{personality}\n---")
                print(f"struct_out:\n{struct_out}\n---")
                print(f"cot_out:\n{cot_out}\n---")
                print(f"pred_out:\n{pred_out}\n=> pred={pred}\n{'='*60}")

        return preds

    def run(self, dialogs: List[str], speakers: Optional[List[str]] = None, utterances: Optional[List[str]] = None) -> List[int]:
        if self.cfg.method == "direct":
            return self.run_direct(dialogs)
        assert speakers is not None, "ioecot needs speakers"
        return self.run_ioecot(dialogs, speakers, utterances)
