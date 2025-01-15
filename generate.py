import os
import json
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from pathlib import Path
from typing import List, Optional
from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
GEN = 1
TOKEN = 1
from tokenizer import ChatFormat, Dialog, Message, Tokenizer
class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required
from model import Qwen2ForCausalLM  # 假设模型结构已定义
from config import Qwen2Config


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
class Qwen2:
    @staticmethod
    def build(
            ckpt_dir: str,
            tokenizer,
            max_seq_len: int,
            seed: int = 1,
    ) -> "Qwen2":
        assert 1 <= max_seq_len <= 32768, f"max_seq_len must be between 1 and 32768, got {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        torch.manual_seed(seed)
        checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
        assert len(checkpoints) > 0, f"No checkpoint files found in {ckpt_dir}"
        ckpt_path = checkpoints[0]
        checkpoint = load_file(ckpt_path)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        config = Qwen2Config()
        model = Qwen2ForCausalLM(config).cuda()
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        model.lm_head.weight = model.model.embed_tokens.weight
        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)
        model.load_state_dict(checkpoint, strict=False)
        print(model)
        print(f"Model loaded successfully from {ckpt_path}!")
        return Qwen2(model, tokenizer, ckpt_dir)

    import torch
    import torch.nn.functional as F
    from typing import List, Tuple

    def generate(
            self,
            tokenizer,
            prompt: str,
            max_length: int = 50,
            temperature: float = 0.7,
            top_k: int = 50,
            top_p: float = 0.9,
            device: str = "cuda",
    ):
        # Tokenize the prompt and move to the specified device
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        self.model.eval()

        # Start with the prompt
        generated_ids = input_ids

        for _ in range(max_length):
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :]  # Only the last token's logits

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, k=top_k, dim=-1)
                logits[logits < top_k_values[:, -1, None]] = float("-inf")

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                for i in range(logits.size(0)):
                    logits[i][sorted_indices[i][sorted_indices_to_remove[i]]] = float("-inf")

            # Sample from the adjusted logits
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to generated_ids
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode the generated sequence
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

    def chat_completion(
            self,
            dialogs: List[Dialog],
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_gen_len: Optional[int] = None,
            logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        if max_gen_len is None:
            max_gen_len = 2047

        prompt_tokens = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
            }
            for t in generation_tokens
        ]


    def __init__(self, model, tokenizer, ckpt_dir):
        self.model = model
        self.tokenizer = tokenizer
        if TOKEN == 1:
            self.formatter = ChatFormat(tokenizer)
        else:
            self.formatter = tokenizer
        if GEN != 1:
            self.model2 = AutoModelForCausalLM.from_pretrained(ckpt_dir).to(torch.device("cuda"))
        else:
            self.model2 = 0


