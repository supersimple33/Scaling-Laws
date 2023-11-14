import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import evaluate
from evaluate import logging

from TextGenerationWithLogits import TextGenerationWithLogits

_CITATION = """HELLO"""

_DESCRIPTION = """LOREM IPSUM"""

_KWARGS_DESCRIPTION = """DW"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )
        
    def _compute(
        self, predictions, model_id, tokenizer_id = None, batch_size: int = 16, device=None, # add_start_token = True, max_length=None
    ):
        if device is not None:
            assert device in ["gpu", "cpu", "cuda", "mps"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipe = pipeline(
            "custom", # NOTE: requires pipeline fix
            model=model_id,
            tokenizer=tokenizer_id,
            framework="pt",
            revision=None, # NOTE: WE CAN GRAB CHECKPOINTS HERE
            device=device,
            pipeline_class = TextGenerationWithLogits)
        
        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")
        
        for start_index in logging.tqdm(range(0, len(predictions), batch_size)):
            end_index = min(start_index + batch_size, len(predictions))
            
            with torch.no_grad():
                outputs = pipe(predictions[start_index:end_index], return_tensors=True)
                for output in outputs:
                    out_logits = output["logits"]
                    labels = output["labels"]
                    attn_mask = output["attention_mask"]
                    
                    shift_logits = out_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

                    perplexity_batch = torch.exp(
                        (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                        / shift_attention_mask_batch.sum(1)
                    )

                    ppls += perplexity_batch.tolist()
        
        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

    def ____compute(
        self, predictions, model_id, batch_size: int = 16, add_start_token = True, device=None, max_length=None
    ):

        if device is not None:
            assert device in ["gpu", "cpu", "cuda", "mps"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size >= 1: # NOTE: I guess we still need to set this even if batch_size is 1 to stop complaints in tokenizer
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer.batch_encode_plus(
            predictions,
            add_special_tokens=False,
            padding=batch_size > 1,
            truncation=bool(max_tokenized_len),
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
