import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_name = "unsloth/Qwen2.5-3B"
adapter = "./checkpoints/step-llm-qwen3b"

tokenizer = AutoTokenizer.from_pretrained(adapter)

base = AutoModelForCausalLM.from_pretrained(
    base_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

model = PeftModel.from_pretrained(base, adapter)
model.eval()

retrieved_step = """DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
ENDSEC;
END-ISO-10303-21;"""

caption = "A cylindrical bolt with a hexagonal head"

prompt = f"""You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description and relevant retrieved CAD data, generate a STEP file that accurately represents the described object.


### caption:
{caption}

### retrieved relevant step file:
{retrieved_step}

### output:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]

print("Input tokens:", input_len)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
    )

new_tokens = generated[0, input_len:]
text = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n===== GENERATED =====")
print(text)
