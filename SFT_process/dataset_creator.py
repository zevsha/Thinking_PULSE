import random
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_partition, SeqDataset, SeqDataset_Inference
import numpy as np
import torch
import json

with open(f'Item Id to text file path', 'rb') as ft:
        text_name_dict = pickle.load(ft)


def make_candidate_text(interact_ids, candidate_num, target_item_id, target_item_title):
    neg_item_id = []
    max_attempts = 50  
    attempts = 0
    item_num=6531

    interact_ids = set(interact_ids)
    
    while len(neg_item_id) < 50 and attempts < max_attempts:
        t = np.random.randint(1, item_num + 1)
        if t not in interact_ids and t not in neg_item_id:
            neg_item_id.append(t)
        attempts += 1

    
    if len(neg_item_id) < 50:
        all_items = set(range(1, item_num + 1))
        remaining_candidates = list(all_items - interact_ids - set(neg_item_id))
        needed = 50 - len(neg_item_id)
        neg_item_id.extend(random.sample(remaining_candidates, min(len(remaining_candidates), needed)))

    random.shuffle(neg_item_id)

    candidate_ids = [target_item_id]
    candidate_text = [target_item_title + '[CandidateEmb]']

    for neg_candidate in neg_item_id[:candidate_num - 1]:
        text = find_item_text_single(neg_candidate, title_flag=True, description_flag=False)
        candidate_text.append(text + '[CandidateEmb]')
        candidate_ids.append(neg_candidate)

    random_ = np.random.permutation(len(candidate_text))
    candidate_text = np.array(candidate_text)[random_]
    candidate_ids = np.array(candidate_ids)[random_]

    return ','.join(candidate_text), candidate_ids


def make_interact_text(interact_ids, interact_max_num):
    interact_item_titles_ = find_item_text(item=interact_ids, title_flag=True, description_flag=False)
    interact_text = []
    if interact_max_num == 'all':
        for title in interact_item_titles_:
            interact_text.append(title + '[HistoryEmb]')
    else:
        for title in interact_item_titles_[-interact_max_num:]:
            interact_text.append(title + '[HistoryEmb]')
        interact_ids = interact_ids[-interact_max_num:]
        
    interact_text = ','.join(interact_text)
    return interact_text, interact_ids


def find_item_text(item, title_flag=True, description_flag=True):
    t = 'title'
    d = 'description'
    t_ = 'No Title'
    d_ = 'No Description'
    if title_flag and description_flag:
        return [f'"{text_name_dict[t].get(i,t_)}, {text_name_dict[d].get(i,d_)}"' for i in item]
    elif title_flag and not description_flag:
        return [f'"{text_name_dict[t].get(i,t_)}"' for i in item]
    elif not title_flag and description_flag:
        return [f'"{text_name_dict[d].get(i,d_)}"' for i in item]

def find_item_text_single(item, title_flag=True, description_flag=True):
    t = 'title'
    d = 'description'
    t_ = 'No Title'
    d_ = 'No Description'
    if title_flag and description_flag:
        return f'"{text_name_dict[t].get(item,t_)}, {text_name_dict[d].get(item,d_)}"'
    elif title_flag and not description_flag:
        return f'"{text_name_dict[t].get(item,t_)}"'
    elif not title_flag and description_flag:
        return f'"{text_name_dict[d].get(item,d_)}"'


dataset = data_partition('Dataset name', path=f'data txt file')
[user_train, user_valid, user_test, usernum, itemnum] = dataset
print("usernum: "+str(usernum))
print("itemnum: "+ str(itemnum))
print("train len: "+str(len(user_train)))
print("valid len: "+str(len(user_valid)))
print("test len: "+str(len(user_test)))





train_data_set=SeqDataset(user_train, usernum, itemnum,50)

    

if usernum>10000:
    users = random.sample(range(1, usernum + 1), 10000)
else:
users = range(1, usernum + 1)
user_list = []
for u in users:
    if len(user_train[u]) < 1 or len(user_test[u]) < 1: continue
    user_list.append(u)




inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, 50)



user_reasons = {}
with open("Rationales file path", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            uid = obj.get("user_id")
            if isinstance(uid, str) and uid.isdigit():
                uid = int(uid)
            if uid is not None:
                user_reasons[uid] = obj
        except json.JSONDecodeError:
            print("Skipping bad line")

#Training data
text_input, text_output, sft_data = [], [], []

for i in range(len(train_data_set)):
    u, seq, pos, neg = train_data_set[i]
    
    if user_reasons.get(u) is None:
        continue
    
    target_item_id = pos[-1] if isinstance(pos, (list, np.ndarray, torch.Tensor)) else pos
    target_item_title = find_item_text_single(target_item_id, title_flag=True, description_flag=False)

    interact_text, interact_ids = make_interact_text(seq[seq > 0], 5)
    candidate_num = 10
    candidate_text, candidate_ids = make_candidate_text(seq[seq > 0], candidate_num, target_item_id, target_item_title)
    candidate_set = set(candidate_ids)  


    filtered_interact_ids = [item for item in interact_ids if item not in candidate_set]
    filtered_interact_titles = find_item_text(
        item=filtered_interact_ids,
        title_flag=True,
        description_flag=False
    )
    
    filtered_interact = ", ".join([title + "[HistoryEmb]" for title in filtered_interact_titles])

            

    input_text = f" {u} : "
    input_text += "interacted items: " + str(filtered_interact)
    input_text += " candidate items: " + str(candidate_text)

    
    entry = user_reasons.get(u)
    ada_reason = None
    if entry:
        #if entry.get("best_prompt") == "prompt1":
        ada_reason = entry.get("reason", "")
    

    if ada_reason:
        input_text += " Let me think: " + ada_reason.strip()
    else:
        input_text += " Let me think: Based on your history..."

    input_text += ". The recommendation is "

    text_input.append(input_text)
    sft_data.append({"input": input_text, "output": target_item_title})

with open("output file train path","w") as f:
    for e in sft_data:
        f.write(json.dumps(e)+ "\n")



#Test data for Inference

text_input, text_output, sft_data = [], [], []

for i in range(len(inference_data_set)):
    u, seq, pos, neg = inference_data_set[i]
    if user_reasons.get(u) is None:
        continue
    
    target_item_id = pos[-1] if isinstance(pos, (list, np.ndarray, torch.Tensor)) else pos
    target_item_title = find_item_text_single(target_item_id, title_flag=True, description_flag=False)

    interact_text, interact_ids = make_interact_text(seq[seq > 0], 3)
    candidate_num = 5
    candidate_text, candidate_ids = make_candidate_text(seq[seq > 0], candidate_num, target_item_id, target_item_title)
    candidate_set = set(candidate_ids)  

# filter out overlaps
    filtered_interact_ids = [item for item in interact_ids if item not in candidate_set]
    
    
    filtered_interact_titles = find_item_text(
        item=filtered_interact_ids,
        title_flag=True,
        description_flag=False
    )
    filtered_interact = ", ".join([title + "[HistoryEmb]" for title in filtered_interact_titles])
    input_text = f" {u} : "
    input_text += "interacted items: " + str(filtered_interact)
    input_text += " candidate items: " + str(candidate_text)

    input_text += ". The recommendation is "

    text_input.append(input_text)
    sft_data.append({"input": input_text, "output": target_item_title})

with open("output file inference path","w") as f:
    for e in sft_data:
        f.write(json.dumps(e)+ "\n")