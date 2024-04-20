import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import pipeline

import evaluate
from evaluate import logging

from QuestionAnswerWithLogits import QuestionAnswerWithLogits

_CITATION = """HELLO"""

_DESCRIPTION = """LOREM IPSUM"""

_KWARGS_DESCRIPTION = """DW"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class QuestionAnswerPerplexity(evaluate.Metric):
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
        self, predictions, prompts, model_id, tokenizer_id = None, batch_size: int = 16, device=None, # add_start_token = True, max_length=None
    ):
        if device is not None:
            assert device in ["gpu", "cpu", "cuda", "mps"], "device should be either gpu or cpu or mps."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipe = pipeline(
            "custom", # NOTE: requires pipeline fix add: custom_tasks = {"custom": {"impl": pipeline_class, "pt": ("AutoModelForCausalLM",), "tf": ()}}
            model=model_id,
            tokenizer=tokenizer_id,
            framework="pt",
            revision=None, # NOTE: WE CAN GRAB CHECKPOINTS HERE
            device=device,
            batch_size=batch_size,
            pipeline_class = QuestionAnswerWithLogits
        )
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
        else:
            pass # TODO: investigate over here, however this is almost never hit
            raise ValueError("follow up")
        
        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")
        
        # MARK: Main Loop of predictions
        for start_index in logging.tqdm(range(0, len(predictions), batch_size)):
            end_index = min(start_index + batch_size, len(predictions)) # get the last problem to be calculated
            
            with torch.no_grad(): # we aren't training so we dont need gradients
                # To get the actual feed forwards combine prompts with their predictions
                strings = [prompts[i] + predictions[i] for i in range(start_index, end_index)]# + " " +
                mappings = pipe.tokenizer(
                    strings, # prompts[start_index:end_index], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    return_offsets_mapping=True, # We want to ensure the tokenization does not mix parts of the answer with the prompt
                    return_overflowing_tokens=True # There should be no overflows if so raise error
                )
                answer_starts = [-1] * len(strings)
                
                # TESTING: Check that the answer starts with a space and the question does not
                for i in range(len(strings)):
                    assert not prompts[i + start_index][-1].isspace(), "prompt should not end with a space"
                    assert predictions[i + start_index][0].isspace(), "answer should start with a space"
                    assert not predictions[i + start_index].isspace(), "answer should not be blank"
                
                # TESTING: Check no tokens were truncated (this should never happen)
                if mappings.get("overflowing_tokens", 0) != 0:
                    raise ValueError("overflowing tokens should be empty")
                
                # TESTING: Check that the tokenization did not mix   
                nonzero_attention = mappings["attention_mask"].nonzero()
                active_token_lengths = [nonzero_attention[nonzero_attention[:,0]==i][:,1].max().item() + 1 for i in nonzero_attention[:,0].unique()]
                for i in range(len(strings)):
                    token_ends = [x[1].item() for x in mappings["offset_mapping"][i][:active_token_lengths[0]]]
                    prompt_end = token_ends.index(len(prompts[i + start_index])) # Will raise error if prompt and answer are mixed (ie we did not end on the length of the prompt)
                    # Note: We don't need to subtract 1 from len because of how mappings work with mappings work
                    answer_starts[i] = prompt_end + 1 # Select the token after the one which ends the prompt
                                
                assert batch_size == 1, "batch size should be 1 for now"
                # FIXME: tokenizes twice (inefficient) and does more forward than necessary, add in pipe.postprocesss(pipe.forward())
                outputs = pipe(strings, return_tensors=True)
                for output in outputs: # NOTE: we could probably do all this in parallel
                    out_logits = output["logits"]
                    labels = output["labels"]
                    attn_mask = output["attention_mask"]
                    
                    shift_logits = out_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
                    
                    # TESTING: Ensure we have a loss for each token but the first one
                    assert (shift_logits.shape[1] == mappings["input_ids"].shape[1] - 1 
                            and torch.all(mappings['input_ids'][:,1:] == shift_labels)), "bad loss transfer"
                    
                    loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    
                    # mask_length = np.zeros(len(range(0, answer_starts[0] - 1)))
                    answer_mask = torch.Tensor(np.concatenate((np.zeros(answer_starts[0] - 1), np.ones(shift_logits.shape[1] + 1 - answer_starts[0])))) # TODO: answer_starts should not be sub 0 in future
                    
                    relevant_loss = ((loss * shift_attention_mask_batch * answer_mask).sum(1)).to(dtype=torch.double)
                    # TESTING: If we get infinite loss something went wrong
                    assert not torch.isinf(relevant_loss) and not torch.isnan(relevant_loss), "relevant loss is inf"
                    
                    # TESTING: Ensure that we have a non-zero number of answer tokens
                    assert (answer_mask * shift_attention_mask_batch).sum(1) > 0, ""

                    perplexity_batch = torch.exp(
                        relevant_loss
                        / (answer_mask * shift_attention_mask_batch).sum(1),
                    )
                    assert not torch.isinf(perplexity_batch), "perplexity is inf"
                    # REVIEW: we could also do median here instead of mean

                    ppls += perplexity_batch.tolist()
        
        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls), "median_perplexity": sorted(ppls)[len(ppls) // 2]}
