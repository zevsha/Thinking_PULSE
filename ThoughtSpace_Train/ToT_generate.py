import json
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import os

model_name = "microsoft/Phi-4-mini-instruct"
DEVICE_LLM = "auto"  
DEVICE_ENC = "cuda" if torch.cuda.is_available() else "cpu"  

RATIONALE_ENCODER_PATH = "rationale_encoder.pth"
BEHAVIOR_ENCODER_PATH  = "behavior_encoder.pth"

# Tree-of-Thoughts params
TOT_MAX_DEPTH = 3
TOT_BRANCHING = 3
TOT_BEAM = 2
THOUGHT_MAX_NEW_TOKENS = 61
THOUGHT_TEMPERATURE = 1
THOUGHT_TOP_K = 51
THOUGHT_TOP_P = 0.95


print("Loading tokenizer & model on", DEVICE_LLM)
tokenizer_1 = AutoTokenizer.from_pretrained(model_name)
model_1 = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE_LLM, torch_dtype=torch.bfloat16)
model_1.eval()

# ----------------------------
# DistilBERT encoders for the thought space
# ----------------------------
class ThoughtEncoder(nn.Module):
    """Encodes text into a vector (uses [CLS] from DistilBERT)."""
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return cls_embedding

def load_trained_models(rationale_path, behavior_path, device):
    print("Loading trained models...")
    rationale_encoder = ThoughtEncoder("distilbert-base-uncased").to(device)
    behavior_encoder  = ThoughtEncoder("distilbert-base-uncased").to(device)
    rationale_encoder.load_state_dict(torch.load(rationale_path, map_location=device))
    behavior_encoder.load_state_dict(torch.load(behavior_path, map_location=device))
    rationale_encoder.eval(); behavior_encoder.eval()
    print("Models loaded successfully.")
    return rationale_encoder, behavior_encoder

# Load encoders (on DEVICE_ENC) + the DistilBERT tokenizer
rationale_encoder, behavior_encoder = load_trained_models(RATIONALE_ENCODER_PATH, BEHAVIOR_ENCODER_PATH, DEVICE_ENC)
bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ----------------------------
# Prompt & generation helpers
# ----------------------------
def build_prompt1(history, similar_users):
    return (
        "User interaction data:\n"
        f"- Items, ratings (out of 5), and reviews: {history}\n"
        f"- Similar users and their interacted items: {similar_users}\n"
        "Using all of this, analyze the user's preferences, sentiments, and priorities, "
        "and produce a concise chain of thought explaining their profile.\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"Reasoning":"…"}\n'
        "Keep the reasoning under 150 words. No extra text."
    )

def generate_thoughts(expansion_prompt,
                      model,
                      tokenizer,
                      num_return_sequences=4,
                      max_new_tokens=60,
                      temperature=0.7,
                      top_k=50,
                      top_p=0.95):
    inputs = tokenizer(expansion_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        if expansion_prompt in text:
            text = text.replace(expansion_prompt, "").strip()
        if text:
            results.append(text)
    seen, uniq = set(), []
    for t in results:
        s = t.strip()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

# ----------------------------
# Cosine-similarity scorer in *thought* space (DistilBERT)
# ----------------------------
@torch.no_grad()
def compute_answer_similarity(reason_text, history_text, answer_text,
                              rationale_encoder, behavior_encoder, tokenizer,
                              device=DEVICE_ENC, max_length=128):
    # Tokenize reason for rationale encoder
    reason_inputs = tokenizer(
        reason_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(device)

    
    behavior_text = f"[CONTEXT] History: {history_text} [CHOSEN_ITEM] {answer_text}"
    behavior_inputs = tokenizer(
        behavior_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(device)

    # Forward through encoders (they return CLS embeddings)
    reason_vec   = rationale_encoder(reason_inputs["input_ids"], reason_inputs["attention_mask"])   # [1, H]
    behavior_vec = behavior_encoder(behavior_inputs["input_ids"], behavior_inputs["attention_mask"]) # [1, H]

    
    sim = F.cosine_similarity(reason_vec, behavior_vec).item()
    return sim

def score_chain(history_text, chain_text, answer_text,
                rationale_encoder, behavior_encoder, tokenizer):
    # Higher similarity is better
    return compute_answer_similarity(chain_text, history_text, answer_text,
                                     rationale_encoder, behavior_encoder, tokenizer)

# ----------------------------
# Tree-of-Thoughts search using cosine similarity
# ----------------------------
def tree_of_thoughts_search(history, similar_users, answer, model, tokenizer,
                             max_depth=TOT_MAX_DEPTH,
                             branching=TOT_BRANCHING,
                             beam_width=TOT_BEAM):
    base_prompt = build_prompt1(history, similar_users)


    root = {'chain_list': [], 'chain_text': "", 'score': 0.0}
    beam_nodes = [root]

    for depth in range(1, max_depth + 1):
        candidates = []
        for node in beam_nodes:
            existing = node['chain_text']
            expansion_prompt = (
                base_prompt
                + "\nChain of thought so far:\n"
                + (existing if existing else "<none>")
                + "\nContinue the chain with a single concise thought (1-2 sentences). "
                  "Respond ONLY with that thought and nothing else (no JSON):\n"
            )

            thoughts = generate_thoughts(
                expansion_prompt,
                model,
                tokenizer,
                num_return_sequences=branching,
                max_new_tokens=THOUGHT_MAX_NEW_TOKENS,
                temperature=THOUGHT_TEMPERATURE,
                top_k=THOUGHT_TOP_K,
                top_p=THOUGHT_TOP_P,
            )

            for t in thoughts:
                new_chain_list = node['chain_list'] + [t.strip()]
                new_chain_text = " ".join(new_chain_list).strip()

                # Score with cosine similarity in thought space (higher is better)
                score = score_chain(
                    history_text=history,
                    chain_text=new_chain_text,
                    answer_text=answer,
                    rationale_encoder=rationale_encoder,
                    behavior_encoder=behavior_encoder,
                    tokenizer=bert_tokenizer
                )

                candidates.append({
                    'chain_list': new_chain_list,
                    'chain_text': new_chain_text,
                    'score': score
                })

        if not candidates:
            break

        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        beam_nodes = candidates[:beam_width]

    # Return the best (highest similarity)
    best = max(beam_nodes, key=lambda x: x['score'])
    return best['chain_text'], best['score']



results_path = "ToT results path"


done_users = set()
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done_users.add(rec["user_id"])
            except:
                pass
print(f"Already completed: {len(done_users)} users")


with open("Sasrec output") as f1:
    data1_list = json.load(f1)


data1 = {d["user_id"]: d for d in data1_list}


for idx, user_id in enumerate(set(data1.keys())):
    if user_id in done_users:
        continue
    d1 = data1[user_id]

    try:
        
        prompt1 = build_prompt1(
            d1["history"],
            d1["[UserRep] is a user representation. Similar Users have previously interacted with"]
        )
        

        # Run ToT using cosine similarity in the learned space
        with torch.no_grad():
            best_reason, best_score = tree_of_thoughts_search(
                d1['history'],
                d1['[UserRep] is a user representation. Similar Users have previously interacted with'],
                d1['Answer'],
                model_1,
                tokenizer_1,
                max_depth=TOT_MAX_DEPTH,
                branching=TOT_BRANCHING,
                beam_width=TOT_BEAM,
            )

        result = {
            "user_id": user_id,
            "reason1": best_reason,
            "score1": best_score,  
            "score_space": "thought_embedding_cosine",
            "method": "tree_of_thoughts",
            "depth": TOT_MAX_DEPTH,
            "branching": TOT_BRANCHING,
            "beam": TOT_BEAM,
        }

        
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"[{idx}] user_done:", user_id)

    except RuntimeError as e:
        print(f"[ERROR] User {user_id} failed with error: {e}")

    
    torch.cuda.empty_cache()




print("Done. Saved to "+str(results_path))

