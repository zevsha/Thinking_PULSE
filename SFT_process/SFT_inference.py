from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
from tqdm import tqdm


base_model_name = "microsoft/Phi-4-mini-instruct"
adapter_path = "Saved_Model path"
input_path = "Test data path"         

tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda")

model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

data = []
with open(input_path, "r") as f:
    for line in f:
        data.append(json.loads(line))


output_lines = []
for i, sample in enumerate(data):
    prompt = sample["input"]
    prompt+="Recommended item is: "
    #prompt+="Choose one Recommended item from [CandidateEmb]. Strictly only one single item . Dont give multiple items "
    label = sample["output"].strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=59,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    
    predicted = decoded[len(prompt):]

    
    formatted = f"{prompt.strip()}\nAnswer: {label}\nLLM: {predicted}\n"
    with open("/storage/SFTrecc/[Sbert_ablat]luxbeauty_full.txt", "w") as f:
         #print(i)  
         output_lines.append(formatted)
         f.write("\n".join(output_lines))


