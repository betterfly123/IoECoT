# prompts.py

# ===== general =====
STEP_DIRECT_F = "Please give the emotion label of "
STEP_DIRECT_L = " can only be chosen from {emo_prompt} and do not give the explanation."
STEP_COT = "Let’s think step by step"

STEP_PERSONALITY_F = "Give the most accurate one-sentence short description of "
STEP_PERSONALITY_M = "'s personality in the context of "
STEP_PERSONALITY_L = "'s utterances in the history of the dialogue"

STEP_STRUCT = (
    "Please judge the sentiment of each utterance in the dialog history, "
    "noting that you can only choose from [neutral, negative, positive], "
    "reduce the judgment criteria for negative emotions. "
    "Output the utterance and the corresponding sentiment."
)

# ===== ERC =====
ERC_INSTRUCTION = (
    "Complete the emotion recognition task by recognizing the emotion of the last utterance "
    "using the given dialogue history. The conversation history is :"
)
ERC_DIRECT_TARGET = "the last utterance"

ERC_PER_INSTR_P = "Recognizing the emotions of utterance ("
ERC_PER_INSTR_L = ") based on the personality of "

ERC_STEP_PRED_F = "Please give the emotion label of the utterance ( "
ERC_STEP_PRED_M = " ) can only be chosen from "
ERC_STEP_PRED_L = " and do not give the explanation."

# ===== EIC =====
EIC_INSTRUCTION = (
    "Complete the emotion inference task by predicting the emotion of the next utterance "
    "using the given conversation history, note that the emotion of the next utterance is unknown. "
    "The conversation history is :"
)
EIC_DIRECT_TARGET = "the next utterance"

EIC_PER_INSTR_F = "Complete an emotion inference task to predict the emotion of speaker "
EIC_PER_INSTR_M = " in the next utterance of the dialog, make inferences based on the given history of the dialog and on "
EIC_PER_INSTR_L = "'s personality"

EIC_STEP_PRED_F = "Please give the emotion label of the next utterance ( "
EIC_STEP_PRED_M = " ) can only be chosen from "
EIC_STEP_PRED_L = " and do not give the explanation."
