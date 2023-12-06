import numpy as np
import pickle
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

from BetterPerplexity import BetterPerplexity
from QuestionAnswerPerplexity import QAPerplexity

import timeit

PULL_C4 = False
USE_ORCA = False

perplexity = Perplexity()
print(perplexity)

# C4 Dataset: loads 5000 elements off the 42^2 seed
if PULL_C4:
	dataset = load_dataset("c4", "en", split="validation", streaming=True)
	dataset = dataset.shuffle(seed=42*42, buffer_size=dataset.dataset_size).take(5000).select_columns("text") # buffer_size and seed being set corrcetly is super important
	dataset : list[str] = np.reshape(np.array(sorted([x["text"] for x in dataset], key=len)), (-1)) # MARK: the last number is batches
	with open('c4.pkl', 'wb') as f:
		pickle.dump(dataset, f)
else:
    with open('c4.pkl', 'rb') as f:
        dataset = pickle.load(f)


model_id = "openlm-research/open_llama_3b"
tokenizer_id = None # NOTE: this should be none for almost all models

# Uncomment this code to setup llama, you have to manually set where the model lives
# tokenizer_id = LlamaTokenizer.from_pretrained("path here")
# model_id = LlamaForCausalLM.from_pretrained("path here")

model_id = "openlm-research/open_llama_3b"
tokenizer_id = None # NOTE: this should be none for almost all models

# NOTE: Batch makes no difference since most sentences are > 16 tokens
if USE_ORCA:
	result = perplexity.compute(predictions=answers, prompt=questions, model_id=model_id, tokenizer_id=tokenizer_id, device="mps", batch_size=1) #predictions=None, model_id=model, add_start_token=False, device="mps"
else:
    # cereb error line 810
	result = perplexity.compute(predictions=dataset, model_id=model_id, tokenizer_id=tokenizer_id, device="mps", batch_size=1) #predictions=None, model_id=model, add_start_token=False, device="mps"

# perplexities.append(result["mean_perplexity"]) # NOTE: we could also take into account mean perplexities
print(result)
print(result["perplexities"][len(result["perplexities"]) // 2])
print()
