"""
Thin wrapper around a HuggingFace chat model.

Everything the project needs from the LLM lives here:
  * answer()            - greedy answer for a prompt, its probability, prompt length and time;
                          optionally m sampled answers with embeddings (for SemanticDPP)
  * yes_no_confidence() - P(Yes) at every layer / last-k tokens (for Internal Confidence)
  * kv_bytes_per_token  - how much KV-cache memory one token costs on this model
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

QA_SYSTEM = "You answer trivia questions. Reply with only the short answer, a few words at most."

# Self-assessment prompt from Paper A (Section 3.2).
IC_PROMPT = ("Respond only with 'Yes' or 'No' to indicate whether you are capable of "
             "answering the following question accurately.\n"
             "Question: {question}\nAnswer Yes or No:")
# Our extension: the same self-assessment, asked after the model has been given passages.
IC_PROMPT_WITH_CONTEXT = ("Context:\n{context}\n\n"
                          "Respond only with 'Yes' or 'No' to indicate whether you are capable of "
                          "answering the following question accurately, using the context above "
                          "and what you already know.\n"
                          "Question: {question}\nAnswer Yes or No:")


def ic_prompt(question: str, passages: Optional[List[str]] = None) -> str:
    if not passages:
        return IC_PROMPT.format(question=question)
    context = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return IC_PROMPT_WITH_CONTEXT.format(context=context, question=question)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def qa_prompt(question: str, passages: Optional[List[str]] = None) -> str:
    """User message for answering a question, optionally with retrieved passages."""
    if not passages:
        return f"Question: {question}\nAnswer:"
    context = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


class LLM:
    def __init__(self, name: str = DEFAULT_MODEL, device: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.device = device or pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()

        # Token ids of the two possible self-assessment answers.
        self.yes_id = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = self.tok.encode("No", add_special_tokens=False)[0]

    # ------------------------------------------------------------------ memory
    @property
    def kv_bytes_per_token(self) -> int:
        """KV-cache bytes for ONE token: a key and a value vector in every layer."""
        cfg = self.model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        bytes_per_number = 4 if self.model.dtype == torch.float32 else 2
        return 2 * cfg.num_hidden_layers * kv_heads * head_dim * bytes_per_number

    # ----------------------------------------------------------------- helpers
    def _encode(self, user_message: str, system: Optional[str] = None):
        messages = ([{"role": "system", "content": system}] if system else []) \
                   + [{"role": "user", "content": user_message}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)

    def _sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def free_cache(self):
        """Release cached GPU memory. On an 8 GB Mac this avoids swapping."""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ answer
    @torch.no_grad()
    def answer(self, user_message: str, n_samples: int = 0, temperature: float = 1.0,
               top_p: float = 0.9, max_new_tokens: int = 24) -> dict:
        """
        Read the prompt ONCE, then decode from its KV cache:
          * the greedy answer, and
          * optionally n_samples sampled answers (for SemanticDPP).

        Every decode starts from the same cached prompt, so sampling costs a few
        extra tokens of compute and no extra prompt memory.

        Returns a dict with
          text            greedy answer
          log_prob        sum of log-probs of the greedy tokens: how likely the model
                          itself thinks this exact answer is (comes for free)
          prompt_tokens   prompt length = KV-cache entries held for this question
          seconds         prompt pass + greedy decoding
          samples, embeddings [n_samples, hidden], sample_seconds   (if n_samples > 0)
        An answer's embedding is the last-layer hidden state averaged over its
        generated tokens (Paper B, Appendix C).
        """
        inputs = self._encode(user_message, QA_SYSTEM)
        n_prompt = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        out = self.model(**inputs, use_cache=True)
        cache, first_logits = out.past_key_values, out.logits[0, -1]
        text, log_prob, _, n_decoded = self._decode(cache, first_logits, max_new_tokens)
        self._sync()
        result = {"text": text, "log_prob": log_prob, "prompt_tokens": n_prompt,
                  "seconds": time.perf_counter() - start}

        if n_samples:
            start = time.perf_counter()
            samples, embeddings = [], []
            for _ in range(n_samples):
                cache.crop(-n_decoded)                    # drop the previous decode, keep the prompt
                sample_text, _, embedding, n_decoded = self._decode(
                    cache, first_logits, max_new_tokens, temperature=temperature, top_p=top_p)
                samples.append(sample_text)
                embeddings.append(embedding)
            self._sync()
            result.update({"samples": samples, "embeddings": torch.stack(embeddings).cpu().numpy(),
                           "sample_seconds": time.perf_counter() - start})
        return result

    def _decode(self, cache, logits, max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0):
        """
        Generate token by token from a cached prompt. temperature 0 = greedy.
        Returns (text, log-probability, mean hidden state, tokens added to the cache).
        """
        stop_ids = self.model.generation_config.eos_token_id
        stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
        tokens, states, log_prob = [], [], 0.0
        for _ in range(max_new_tokens):
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            if temperature > 0:
                token = self._nucleus_sample(logits.float() / temperature, top_p)
            else:
                token = int(log_probs.argmax())
            log_prob += log_probs[token].item()
            finished = token in stop_ids
            if finished and tokens:
                break
            if not finished:
                tokens.append(token)
            out = self.model(input_ids=torch.tensor([[token]], device=self.device),
                             past_key_values=cache, use_cache=True, output_hidden_states=True)
            logits = out.logits[0, -1]
            states.append(out.hidden_states[-1][0, -1].float())
            if finished:                                  # empty answer: keep one state and stop
                break
        text = self.tok.decode(tokens, skip_special_tokens=True).strip()
        return text, log_prob, torch.stack(states).mean(dim=0), len(states)

    @staticmethod
    def _nucleus_sample(logits: torch.Tensor, top_p: float) -> int:
        """Sample from the smallest set of tokens whose probability adds up to top_p."""
        probs = torch.softmax(logits, dim=-1).cpu()
        sorted_probs, sorted_ids = probs.sort(descending=True)
        keep = (sorted_probs.cumsum(dim=0) - sorted_probs) < top_p
        choice = torch.multinomial(sorted_probs * keep, 1)
        return int(sorted_ids[choice])

    # ------------------------------------------------- internal confidence input
    @torch.no_grad()
    def yes_no_confidence(self, question: str, passages: Optional[List[str]] = None,
                          last_k: int = 6) -> Tuple[np.ndarray, int, float]:
        """
        One forward pass over the self-assessment prompt, no generation.

        Returns (p_yes [n_layers, last_k], prompt tokens, seconds) where
        p_yes[l, n] = softmax over {Yes, No} of the unembedded hidden state at
        layer l and the n-th of the last k tokens (Paper A, Eq. 1).
        """
        inputs = self._encode(ic_prompt(question, passages))
        n_prompt = inputs["input_ids"].shape[1]
        start = time.perf_counter()
        out = self.model(**inputs, output_hidden_states=True)
        # hidden_states[0] is the embedding layer; [1:] are the transformer layers.
        hidden = torch.stack(out.hidden_states[1:], dim=0)[:, 0, -last_k:, :]   # [L, k, d]
        # Intermediate layers are read through the model's final norm ("logit lens");
        # the last layer already has it applied by HuggingFace.
        final_norm = self.model.model.norm
        hidden = torch.cat([final_norm(hidden[:-1]), hidden[-1:]], dim=0)
        unembed = self.model.get_output_embeddings().weight[[self.yes_id, self.no_id]]  # [2, d]
        logits = hidden.float() @ unembed.float().T                             # [L, k, 2]
        p_yes = torch.softmax(logits, dim=-1)[..., 0]
        self._sync()
        seconds = time.perf_counter() - start
        return p_yes.cpu().numpy(), n_prompt, seconds
