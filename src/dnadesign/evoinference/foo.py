"""
Eric South
"""





from transformers import AutoConfig, AutoModelForCausalLM

model_name = 'togethercomputer/evo-1-8k-base'

model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision="1.1_fix")
model_config.use_cache = True

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=model_config,
    trust_remote_code=True,
    revision="1.1_fix"
)