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
            "custom", # NOTE: requires pipeline fix add: custom_tasks = {"custom": {"impl": pipeline_class, "pt": (), "tf": ()}}
            model=model_id,
            tokenizer=tokenizer_id,
            framework="pt",
            revision=None, # NOTE: WE CAN GRAB CHECKPOINTS HERE
            device=device,
            batch_size=batch_size, # MARK: testudo
            pipeline_class = TextGenerationWithLogits)
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
        else:
            raise ValueError("follow up")
        
        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")
        
        for start_index in logging.tqdm(range(0, len(predictions), batch_size)):
            end_index = min(start_index + batch_size, len(predictions))
            
            with torch.no_grad():
                outputs = pipe(predictions[start_index:end_index], return_tensors=True)
                for output in outputs: # NOTE: we could probably do all this in parallel
                    out_logits = output["logits"]
                    labels = output["labels"]
                    attn_mask = output["attention_mask"]
                    
                    shift_logits = out_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
                    
                    loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    relevant_loss = ((loss * shift_attention_mask_batch).sum(1)).to(dtype=torch.double)
                    assert not torch.isinf(relevant_loss), "relevant loss is inf"

                    perplexity_batch = torch.exp(
                        relevant_loss
                        / shift_attention_mask_batch.sum(1),
                    )
                    assert not torch.isinf(perplexity_batch), "perplexity is inf"
                    # REVIEW: we could also do median here instead of mean

                    ppls += perplexity_batch.tolist()
        
        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
