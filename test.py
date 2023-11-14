import numpy as np
from datasets import load_dataset
from BetterPerplexity import Perplexity

perplexity = Perplexity()
print(perplexity)

# C4 Dataset
dataset = load_dataset("c4", "en", split="validation", streaming=True).shuffle(seed=42*42).take(500).select_columns("text")
dataset : list[str] = np.reshape(np.array([x["text"] for x in dataset]), (-1)) # MARK: the last number is batches

# NOTE: Add start token can be false for C4 but should be true for ORCA
# perplexities = []
# for i in range(len(dataset)):
# 	batch = dataset[i]
# 	perplexity.add_batch(predictions=batch)
result = perplexity.compute(predictions=dataset, model_id="EleutherAI/pythia-70m", device="mps", batch_size=8) #predictions=None, model_id=model, add_start_token=False, device="mps"
# perplexities.append(result["mean_perplexity"]) # NOTE: we could also take into account mean perplexities
print(result)
