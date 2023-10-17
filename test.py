import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate
from datasets import load_dataset

# Load a dataset and a transformer and then test how well the transformer performs on the dataset (there should be no training)

dataset = load_dataset("oscar", "unshuffled_deduplicated_af")["train"].shuffle(seed=42).select(range(10)).select_columns("text")

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-111M")

pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


module = evaluate.load("perplexity")

evaluator = evaluate.evaluator("text-generation") # BERT

test = evaluator.compute(pipeline, dataset, metric="perplexity")
print(test)
