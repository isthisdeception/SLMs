"""
Prompt templates for Chain-of-Thought reasoning, GSM8K, and ARC-Challenge evaluations.
"""

from typing import List, Dict, Any

COT_SYSTEM_PROMPT = "You are a helpful assistant that thinks step-by-step before answering."

def format_cot_prompt(question: str, model_type: str = "qwen") -> str:
    """
    Formats a user question into the ChatML format, stopping right before the assistant response starts.
    """
    if model_type.lower() == "qwen":
        prompt = (
            f"<|im_start|>system\n{COT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:  # Llama style
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{COT_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return prompt


def format_gsm8k_prompt(question: str, n_shot: int = 0) -> str:
    """
    Formats a GSM8K math reasoning prompt with optional zero-shot or few-shot examples.
    """
    few_shot_prefix = ""
    if n_shot > 0:
        few_shot_prefix = (
            "Question: Natalia sold clips to 48 of her friends in April, and then half as many in May. How many clips did Natalia sell altogether in April and May?\n"
            "Answer: Natalia sold 48 / 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips. The final answer is 72.\n\n"
            "Question: Weng earns $12 an hour for tutoring. If she tutors for 4 hours a day, 5 days a week, how much does she earn in 4 weeks?\n"
            "Answer: Weng earns $12 * 4 = $48 a day. In 5 days she earns $48 * 5 = $240. In 4 weeks she earns $240 * 4 = $960. The final answer is 960.\n\n"
        )
    
    prompt = (
        f"<|im_start|>system\n{COT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{few_shot_prefix}Question: {question}\n"
        f"Please reason step-by-step and write your final answer at the end like 'The final answer is [ANSWER]'.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def format_arc_prompt(question: str, choices: Dict[str, List[Any]]) -> str:
    """
    Formats an ARC science reasoning question with multiple choice options.
    """
    options_text = ""
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    
    for label, text in zip(labels, texts):
        options_text += f"({label}) {text}\n"

    prompt = (
        f"<|im_start|>system\n{COT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {question}\nOptions:\n{options_text}"
        f"Select the correct option (e.g. A, B, C, D). Reason step-by-step and state 'The final answer is (X)'.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt
