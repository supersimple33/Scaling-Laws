import numpy as np
import pickle
from datasets import load_dataset
from BetterPerplexity import Perplexity

import timeit

PULL_C4 = False

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

dataset = dataset[0:1000] # bad actors at # 68 gives 73920 and 566 gives

# NOTE: Batch makes no difference in time on my laptop but it may for you
result = perplexity.compute(predictions=dataset, model_id="EleutherAI/pythia-160m", device="mps", batch_size=2) #predictions=None, model_id=model, add_start_token=False, device="mps"

# perplexities.append(result["mean_perplexity"]) # NOTE: we could also take into account mean perplexities
print(result)
print(result["perplexities"][len(result["perplexities"]) // 2])
print()
