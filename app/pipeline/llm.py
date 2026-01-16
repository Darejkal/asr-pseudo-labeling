import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL=None
TOKENIZER=None
def get_offline_model():
    global MODEL,TOKENIZER
    print("Loading Entity Extraction model (Qwen3-8B)...")
    model_name = "Qwen/Qwen3-8B"
    if not MODEL:
        MODEL= AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            revision="b968826d9c46dd6066d109eabc6255188de91218"
        )
    if not TOKENIZER:
        TOKENIZER = AutoTokenizer.from_pretrained(
            model_name,
            revision="b968826d9c46dd6066d109eabc6255188de91218"
        )
    return MODEL,TOKENIZER
def generate_with_thinking(
    prompt: str, 
    model, 
    tokenizer, 
    max_new_tokens: int = 512, 
    enable_thinking=False,
    thinking_token_id: int = 151668,
):
    """
    Generates a response from the model and separates the 'thinking' process 
    from the final content.
    
    Args:
        model: The loaded HuggingFace model.
        tokenizer: The loaded tokenizer.
        prompt (str): The user query.
        max_new_tokens (int): Max tokens to generate.
        thinking_token_id (int): The ID for the </think> token (Default 151668 for Qwen).
    
    Returns:
        tuple: (thinking_content, final_content)
    """
    
    # 1. Prepare the messages and formatting
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  # Specific to Qwen reasoning models
    )
    
    # 2. Tokenize and move to device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 3. Generate
    # We use no_grad to ensure we don't store gradients during inference (saves memory)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

    # 4. Extract only the newly generated tokens (remove input prompt)
    input_length = len(model_inputs.input_ids[0])
    output_ids = generated_ids[0][input_length:].tolist()

    # 5. Parse the output to separate thinking from content
    try:
        # Find the index of the </think> token (151668)
        # We search from the end to find the split point
        reversed_ids = output_ids[::-1]
        split_index = len(output_ids) - reversed_ids.index(thinking_token_id)
    except ValueError:
        # If </think> is not found, treat everything as content
        split_index = 0

    # 6. Decode parts
    thinking_content = tokenizer.decode(output_ids[:split_index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[split_index:], skip_special_tokens=True).strip()
    return thinking_content,content

import openai
import os

def generate_with_openai(
    prompt: str,
    model=None,          # Accepted for compatibility (unused)
    tokenizer=None,      # Accepted for compatibility (unused)
    max_new_tokens: int = 512,
    enable_thinking=False, # Accepted for compatibility
    thinking_token_id: int = None, # Accepted for compatibility (unused)
):
    """
    Generates a response using an OpenAI-compatible API (e.g., vLLM, Llama.cpp)
    and attempts to separate 'thinking' process from final content using string parsing.
    
    Returns:
        tuple: (thinking_content, final_content)
    """
    
    # 1. Setup Client (Environment variables are best practice)
    llm_service_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8080/v1")
    # specific API key usually not needed for local llama.cpp/vllm
    client = openai.OpenAI(base_url=llm_service_url, api_key=os.getenv("OPENAI_API_KEY", "dummy"))

    # 2. Call API
    try:
        response = client.chat.completions.create(
            model="your-model-name", # Often ignored by local servers like llama.cpp
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.6, # Slightly higher temp often helps reasoning models flow better
        )
        full_text = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "", ""

    # 3. Parse output to separate thinking from content
    # Since we are working with text (not tokens), we look for the standard XML tags 
    # used by reasoning models (e.g., DeepSeek-R1, Qwen-QwQ).
    
    thinking_content = ""
    final_content = full_text

    # Check for the closing tag </think>
    if "</think>" in full_text:
        parts = full_text.split("</think>")
        
        # The part before the split is the thinking process
        raw_thinking = parts[0]
        
        # The part after is the final content
        final_content = parts[1].strip() if len(parts) > 1 else ""
        
        # Clean up the opening <think> tag if it exists in the thinking part
        thinking_content = raw_thinking.replace("<think>", "").strip()
        
    elif "<think>" in full_text: 
        # Edge case: Model started thinking but hit max_tokens before finishing
        parts = full_text.split("<think>")
        final_content = parts[0].strip()
        thinking_content = parts[1].strip()

    return thinking_content, final_content

from typing import Literal, Tuple, Optional, Callable,Any
from pydantic import BaseModel, ValidationError
class GenerationConfig(BaseModel):
    # Literal ensures the string must be EXACTLY "offline" or "online"
    mode: Literal["offline", "online"]
    max_new_tokens: int = 512
    # Optional fields for when we are in offline mode
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

from functools import partial
def get_generator(config: GenerationConfig) -> Callable:
    """
    Returns the generation function based on the validated Pydantic config.
    """
    kwargs=config.model_dump()
    del kwargs['mode']
    if config.mode == "offline":
        # Returns the HuggingFace transformers function
        return partial(generate_with_thinking,**kwargs)
    else:
        # Returns the OpenAI-compatible API function
        return partial(generate_with_openai,**kwargs)