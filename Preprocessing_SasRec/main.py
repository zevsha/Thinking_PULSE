import os
import time
import torch
import argparse

from model import SASRec
from data_preprocess import *
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--dataset2', required=False)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)  
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--state_dict_path', default=None, type=str)


args = parser.parse_args()

device=torch.device("cpu")

def get_all_user_embeddings(model, user_train, device):
    emb_dict = {}
    for u, seq in user_train.items():
        input_seq = seq[-args.maxlen:]
        seq_tensor = torch.LongTensor([input_seq]).to(device)
        with torch.no_grad():
            emb = model.get_user_embedding(seq_tensor).squeeze(0).cpu()
        emb_dict[u] = emb
    return emb_dict

def get_topk_similar_users(target_user, emb_dict, k=5):
    target_emb = emb_dict[target_user].unsqueeze(0)
    all_users = list(emb_dict.keys())
    all_embs = torch.stack([emb_dict[u] for u in all_users])

    sim_scores = cosine_similarity(target_emb.numpy(), all_embs.numpy())[0]
    sim_user_ids = sorted(zip(all_users, sim_scores), key=lambda x: -x[1])
    sim_user_ids = [u for u in sim_user_ids if u[0] != target_user]
    return sim_user_ids[:k]

if __name__ == '__main__':
    
    # global dataset
    preprocess(args.dataset)
    dataset = data_partition(args.dataset)
    dataset2=data_partition(args.dataset2)
    #print("Dataset path:", args.dataset)
    


    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    [user_train2, user_valid2, user_test2, usernum2, itemnum2] = dataset2
    
    #print('user num:', usernum, 'item num:', itemnum)
    #print("user_train keys:", list(user_train.keys()))
    print("len(user_train):", len(user_train))
    print("len(user_valid):", len(user_valid2))

    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # dataloader
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)       
    # model init
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    model.train()
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset2, args)
        print('test (NDCG@1: %.4f, HR@1: %.4f)' % (t_test[0], t_test[1]))
    
    text_name_dict_path = 'Item id to text file path'

    with open(text_name_dict_path, 'rb') as f:
        text_name_dict = pickle.load(f)

    
    user_train = dataset[0]
    user_valid=dataset[1]
    user_test=dataset[2]
    user_emb_dict = get_all_user_embeddings(model, user_train, device)
    similar_user_output = []

    results = []
    for u, seq in user_train.items():
        hist = seq[:-1]     
        ans=seq[-1]       
        seq_tensor = torch.LongTensor([hist]).to(args.device)
        top_sim_users = get_topk_similar_users(u, user_emb_dict, k=5)
    
        history_info = []
        for item in hist:
            title=text_name_dict['title'].get(item, 'No Title')
            review=text_name_dict['reviews'].get(item,{}).get(u,"")
            
            rating=text_name_dict['ratings'].get(item,{}).get(u,"")
            
            history_info.append({"item":title,"review":review,'rating':rating})
    
        simuser_titles=[]
        hist_items_of_simusers=[]
        
            
        for sid,sim in top_sim_users:
            p=[]
            for x in user_train[sid]:

               simuser_temp = text_name_dict['title'].get(x, 'No Title')
               p.append(simuser_temp)
            
            hist_items_temp_simusers=({"user id":sid, "interacted items": p})
            hist_items_of_simusers.append(hist_items_temp_simusers)
            

        results.append({
          'user_id': u,
          'history': history_info,
    
          'Similar Users have previously interacted with': hist_items_of_simusers,
           'Answer': text_name_dict['title'].get(ans, 'No Title')
        })
        
    out = f"{args.dataset}_sasrec_output.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved Stage‑1 Top‑{args.topk} to {out}")
    
     
#
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only: break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
    
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset2, args)
            t_valid = evaluate_valid(model, dataset2, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (NDCG@1: %.4f, HR@1: %.4f), test (NDCG@1: %.4f, HR@1: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            print(str(t_valid) + ' ' + str(t_test) + '\n')
            torch.cuda.empty_cache()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset
            fname = 'SASRec.epoch={}.lr={}.layer={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.hidden_units, args.maxlen)
            if not os.path.exists(os.path.join(folder, fname)):
                try:
                    os.makedirs(os.path.join(folder))
                except:
                    print()
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))
    
    sampler.close()
    print("Done")

