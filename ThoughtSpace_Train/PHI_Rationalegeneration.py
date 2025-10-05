import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/Phi-4-mini-instruct"
DEVICE = "auto"

input_file = "path for Metadata to get ratioanles"
output_file = "Generated rationale file path"


def build_prompt(user_data):
    history_text = user_data['history']
    answer = user_data["Answer"]

    prompt = (
        "You are an expert cognitive psychologist and recommender system analyst. "
        "Your task is to dissect a user's interaction history to understand their underlying taste profile "
        "and explain their choices with deep insight.\n\n"

        "**1. User Interaction History:** You are provided with a list of items the user has previously interacted with, "
        "including their ratings and their own words from reviews.\n"
        f"{history_text}\n\n"

        "**2. The User's Choice:** From a new set of options, the user chose the following item:\n"
        f"{answer}\n\n"

        "**3. Your Task:** Follow these steps precisely to generate your analysis:\n"
        "a. Analyze the History: infer the user's latent preferences and psychological drivers.\n"
        "b. Deconstruct the Choice: briefly consider the chosen item.\n"
        "c. Synthesize the Rationale: write a single, concise rationale (<100 words) explaining WHY this choice "
        "fits their underlying taste profile.\n\n"
        "**GIVE ME OUTPUT STRICTLY UNDER 110 WORDS\n**"

        "**4. Output Format:** Respond ONLY with a JSON object in this exact format:\n"
        "{'rationale': '…'}"
    )
    return prompt.strip()


print("Loading tokenizer & model on", DEVICE)
tokenizer_1 = AutoTokenizer.from_pretrained(model_name)
model_1 = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16
)
model_1.eval()

with open(input_file, "r") as infile:
    data = json.load(infile)

results = []
for idx, user_data in enumerate(data):
    prompt = build_prompt(user_data)
    inputs = tokenizer_1(prompt, return_tensors="pt").to(model_1.device)

    with torch.no_grad():
        output_tokens = model_1.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )

    output_text = tokenizer_1.decode(output_tokens[0], skip_special_tokens=True)
    output_text=output_text.replace(prompt," ")

    results.append({"user_id":user_data.get("user_id",idx),
                    "reason":output_text})
    with open(output_file, "w") as outfile:
        json.dump(results, outfile, indent=2)

    if idx % 50 == 0:
        print(f"Processed {idx}/{len(data)} users")


print(f"Saved rationales for {len(results)} users to {output_file}")
