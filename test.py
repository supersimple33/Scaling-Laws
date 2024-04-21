import numpy as np
import pickle
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

from BetterPerplexity import BetterPerplexity
from QuestionAnswerPerplexity import QuestionAnswerPerplexity

import timeit

# MARK: Setup

PULL_DATA = False
USE_ORCA = True

# MARK: Setup Dataset and download\f if needed

if USE_ORCA:
    perplexity = QuestionAnswerPerplexity()
    
    if PULL_DATA:
        dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        dataset = dataset.shuffle(seed=42*42, buffer_size=dataset.dataset_size).take(5000).select_columns(["system_prompt", "question", "response"])
        dataset = [((x["system_prompt"] + " Prompt: " + x["question"]).rstrip() , " " + x["response"]) for x in dataset] # TODO: sort by tokenization length for small speedup
        # REVIEW: IMPORTANT DO WE WANT THE SPACE ABOVE ????????? # Take away spaces from the right of the question using rstrip
        questions = [x[0]for x in dataset] 
        answers = [x[1] for x in dataset]
        with open('orca.pkl', 'wb') as f:
            pickle.dump(dataset, f)
    else:
        with open('orca.pkl', 'rb') as f:
            dataset = pickle.load(f)
        questions = [x[0] for x in dataset]
        answers = [x[1] for x in dataset]
else:
    # C4 Dataset: loads 5000 elements off the 42^2 seed
    perplexity = BetterPerplexity()

    if PULL_DATA:
        dataset = load_dataset("c4", "en", split="validation", streaming=True) # TODO: sort by tokenization length for small speedup
        dataset = dataset.shuffle(seed=42*42, buffer_size=dataset.dataset_size).take(5000).select_columns("text") # buffer_size and seed being set corrcetly is super important
        dataset : list[str] = np.reshape(np.array(sorted([x["text"] for x in dataset], key=len)), (-1)) # MARK: the last number is batches
        with open('c4.pkl', 'wb') as f:
            pickle.dump(dataset, f)
    else:
        with open('c4.pkl', 'rb') as f:
            dataset = pickle.load(f)


# MARK: - SETUP THE DATA        (orca 82 is single token answer)
TAKE = 5000
OFFSET = 0
dataset = dataset[OFFSET:OFFSET+TAKE]
questions = questions[OFFSET:OFFSET+TAKE]
answers = answers[OFFSET:OFFSET+TAKE]

# questions, answers = ["abc def"], [" ab"]

# questions = [x + " Answer:" for x in questions] # NOTE: Do we want to have a Answer: part to signal start of the answer

model_id = "EleutherAI/pythia-160m"
tokenizer_id = None # NOTE: this should be none for almost all models

# Uncomment this code to setup llama, you have to manually set where the model lives
# tokenizer_id = LlamaTokenizer.from_pretrained("path here")
# model_id = LlamaForCausalLM.from_pretrained("path here")

# NOTE: Batch makes no difference since most sentences are > 16 tokens
if USE_ORCA:
    result = perplexity.compute(predictions=answers, prompts=questions, model_id=model_id, tokenizer_id=tokenizer_id, device="mps", batch_size=1) #predictions=None, model_id=model, add_start_token=False, device="mps"
else:
    result = perplexity.compute(predictions=dataset, model_id=model_id, tokenizer_id=tokenizer_id, device="cpu", batch_size=1) #predictions=None, model_id=model, add_start_token=False, device="mps"

# perplexities.append(result["mean_perplexity"]) # NOTE: we could also take into account mean perplexities
print(result['perplexities'])
print("median", result["median_perplexity"])
print("mean", result["mean_perplexity"])

# MARK: Save the results from this run
with open(f"Results_{"ORCA" if USE_ORCA else "C4"}/{model_id.split("/")[-1]}.pkl", 'wb') as f:
    if TAKE != 5000:
        print("are you sure you wan to save?")
        input()
    pickle.dump(result, f)
