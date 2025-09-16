
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)
import re
from tqdm import tqdm
import random
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info
import util

def load_model(model_id, device='auto'):
    model_name = model_id.split('/')[1]
    if model_name.startswith('gemma-3') and not model_name.startswith('gemma-3-1b'):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            attn_implementation='flash_attention_2', 
            device_map=device,
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(model_id)
        tokenizer.tokenizer.padding_side = "left"
    elif model_name.startswith('Qwen2.5-VL'):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            attn_implementation='flash_attention_2', 
            device_map=device,
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(model_id)
        tokenizer.tokenizer.padding_side = "left"
    elif model_name.startswith('Phi-4-multimodal'):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            # torch_dtype=torch.bfloat16, 
            # attn_implementation='flash_attention_2', 
            device_map=device,
            trust_remote_code=True
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.tokenizer.padding_side = "left"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            attn_implementation='flash_attention_2',
            trust_remote_code=True).eval()
        # model = torch.compile(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side  = 'left'

    if not (model_name.startswith('gemma-3') or model_name.startswith('Qwen2.5-VL') or model_name.startswith('Phi-4-multimodal')):
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            # We can set pad_token as the eos_token or add a new one
            tokenizer.pad_token = tokenizer.eos_token
    # model.to(device)
    return model, tokenizer
def generate_output(model, tokenizer, prompts, batch_size=8, max_new_tokens=1024, model_name="default"):
    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_prompts in tqdm(util.batchify(prompts, batch_size), total=total_batches):
        # Tokenize the batch depending on model type
        if model_name.startswith("gemma-3"):
            # Wrap plain strings into chat-format for Gemma
            messages = [[{"role": "user", "content": p}] for p in batch_prompts]
            input_tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            ).to(model.device)

        elif model_name.startswith("Qwen2.5-VL"):
            # Wrap plain strings into chat-format for Qwen2.5-VL
            messages = [[{"role": "user", "content": p}] for p in batch_prompts]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            input_tokens = tokenizer(
                text=text,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

        else:
            # Default: plain text batching
            input_tokens = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

        input_ids = input_tokens["input_ids"]

        output_ids = model.generate(
            **input_tokens,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            # NOTE: removed "max_length=max_new_tokens" because it can truncate inputs
        )

        # Extract only the generated tokens (excluding the input prompts)
        generated_tokens = [
            output[ids.shape[-1]:] for ids, output in zip(input_ids, output_ids["sequences"])
        ]

        # Decode and clean
        responses = [
            clean_generated_text(tokenizer.decode(generated_tokens[i], skip_special_tokens=True))
            for i in range(len(generated_tokens))
        ]
        all_outputs.extend(responses)

    return all_outputs



def batchify(lst, batch_size):
    """Yield successive batches from list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def clean_generated_text(text):
    """Cleans generated text by removing unwanted prefixes like 'Assistant:', '\n', or leading spaces."""
    # if model_name.startswith("DeepSeek"):
    # text = extract_text_after_think(text)
    text = text.strip()  # Remove leading/trailing whitespace or newlines
    text = re.sub(r"^(assistant\n|Assistant:|AI:|Bot:|Response:|Reply:|.:)\s*", "", text, flags=re.IGNORECASE)  # Remove AI labels if present
    text = text.strip()  # Remove leading/trailing whitespace or newlines
    return text