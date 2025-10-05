import os
import os.path
import json
import pickle
from tqdm import tqdm
import gzip
from collections import defaultdict

def parse(path):
    with open(path, 'r') as f:
        for l in tqdm(f):
            yield json.loads(l)
    
        
def preprocess(fname):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    # Modified to use .json instead of .json.gz
    file_path = f'../../data/amazon/{fname}.json'
    
    # counting interactions for each user and item
    for l in parse(file_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        countU[rev] += 1
        countP[asin] += 1
    
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    review_dict = {}
    name_dict = {'title':{}, 'description':{}, 'ratings':{}, 'reviews':{}}  # Added 'image_path' field here
    
    f = open(f'../../data/amazon/meta_{fname}.json', 'r')
    json_data = f.readlines()
    f.close()
    data_list = [json.loads(line[:-1]) for line in json_data]
    meta_dict = {}
    for l in data_list:
        meta_dict[l['asin']] = l
    
    for l in parse(file_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        
        threshold = 7
        if ('Beauty' in fname) or ('Toys' in fname):
            threshold = 1
            
        if countU[rev] < threshold or countP[asin] < threshold:
            continue
        if  countP[asin] < threshold:
            continue
        
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
            
            
            
        User[userid].append([time, itemid])
        
        if itemmap[asin] in review_dict:
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]['Reasoning'][usermap[rev]] = l['Reasoning']
            except:
                a= 0
        else:
            review_dict[itemmap[asin]] = {'review': {}, 'summary':{}}
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]['Reasoning'][usermap[rev]] = l['Reasoning']
            except:
                a = 0
        try:
            if len(meta_dict[asin]['description']) ==0:
                name_dict['description'][itemmap[asin]] = 'Empty description'
            else:
                name_dict['description'][itemmap[asin]] = meta_dict[asin]['description'][0]
            name_dict['title'][itemmap[asin]] = meta_dict[asin]['title']
            if itemmap[asin] not in name_dict['ratings']:
                name_dict['ratings'][itemmap[asin]]={}
            if itemmap[asin] not in name_dict['reviews']:
                name_dict['reviews'][itemmap[asin]]={}

            name_dict['ratings'][itemmap[asin]][userid]=l['overall']
            name_dict['reviews'][itemmap[asin]][userid]=l['reviewText']
            
        except:
            a = 0
    
    # Changed to save as regular json instead of gzipped
    with open(f'../../data/amazon/{fname}_text_name_dict.json', 'wb') as tf:
        pickle.dump(name_dict, tf)
    
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])
        
    print(usernum, itemnum)
    
    f = open(f'../../data/amazon/{fname}.txt', 'w')
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()

