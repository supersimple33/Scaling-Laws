from transformers import TextGenerationPipeline

class QuestionAnswerWithLogits(TextGenerationPipeline):
    def _forward(self, model_inputs, **generate_kwargs):
        # FIXME: we don't actually need to compute the perplexities on the prompt tokens but the cost is pretty small since we're on gpu
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        # prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # BS x SL
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        
        return {"logits": output['logits'], "labels": input_ids, "attention_mask": attention_mask}
    
    def postprocess(self, model_outputs, return_type=None):
        return model_outputs
        # logits = []
        # labels = []
        # attention_mask = []
        # for output in model_outputs:
        #     logits.append(output["logits"])
        #     labels.append(output["labels"])
        #     attention_mask.append(output["attention_mask"])
        # return {"logits": logits, "labels": labels, "attention_mask": attention_mask}