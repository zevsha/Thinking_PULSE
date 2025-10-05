import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import json
import torch.nn.functional as F
import random
# --- 1. Model Definitions ---
# We define the architecture for our encoders. Both will use a pre-trained
# Transformer model and a pooling layer to get a fixed-size vector.

class ThoughtEncoder(nn.Module):
    """
    Encoder model to create a vector representation for either a rationale or a behavior.
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        super(ThoughtEncoder, self).__init__()
        # Load the pre-trained Transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        # We will use the embedding of the [CLS] token as our sentence representation.
        # No extra layers are needed for this simple pooling strategy.

    def forward(self, input_ids, attention_mask):
        """
        Forward pass to compute the vector.
        """
    
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
       
        cls_embedding = outputs.last_hidden_state[:, 0]
       
        return cls_embedding

# --- 2. Custom Dataset ---
# This class prepares your data for training. It takes your raw text data,
# tokenizes it, and formats it for the DataLoader.

class TripletDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the anchor and positive pairs.
    The negative sample will be handled 'in-batch' during training.
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
       
        # This is the high-quality reason explaining the ground truth choice.
        # It will be used to generate the positive vector.
    
        positive_text = item['reason']
       
        # This is the user's context and their actual choice.
        # It will be used to generate the behaviour vector.
        behav_text = f"[CONTEXT] History: {item['item']} [CHOSEN_ITEM] {item['answer']}"

        # Tokenize both texts
        behav_encoding = self.tokenizer(
            behav_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
       
        positive_encoding = self.tokenizer(
            positive_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'behav_input_ids': behav_encoding['input_ids'].flatten(),
            'behav_attention_mask': behav_encoding['attention_mask'].flatten(),
            'positive_input_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
        }

# --- 3. Training Function ---
# This function encapsulates the entire training loop.


def train_encoders(rationale_encoder, behavior_encoder, dataloader, optimizer, device, epochs, num_negatives=10):
    """
    Main training loop for the dual-encoder system with multiple random negatives.
    """
    rationale_encoder.train()
    behavior_encoder.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            # Move data to device
            behav_input_ids = batch['behav_input_ids'].to(device)
            behav_attention_mask = batch['behav_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)

            # --- Forward Pass ---
            v_positive = rationale_encoder(positive_input_ids, positive_attention_mask)   # shape: (B, d)
            v_behav = behavior_encoder(behav_input_ids, behav_attention_mask) # shape: (B, d)

            batch_size, dim = v_positive.size()

            # --- Negative Sampling ---
            # Collect negatives: for each anchor, pick 10 other random anchors
            negatives = []
            for i in range(batch_size):
                # candidate indices (exclude itself)
                candidates = list(range(batch_size))
                candidates.remove(i)
                
                neg_idx = random.sample(candidates, min(num_negatives, len(candidates)))
                neg_vecs = v_positive[neg_idx]  
                negatives.append(neg_vecs)

            # Stack into tensor: shape (B, 10, d)
            v_negatives = torch.stack(negatives, dim=0)

            # --- Loss Calculation ---
            
            # InfoNCE-like loss
            loss = contrastive_loss(v_behav, v_positive, v_negatives)


            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"End of Epoch {epoch+1}. Average Loss: {avg_loss:.4f}")


def contrastive_loss(v_behav, v_positive, v_negatives, temperature=1):
    """
    v_behav: (B, d)
    v_positive: (B, d)
    v_negatives: (B, K, d)
    """
    B, d = v_positive.size()
    K = v_negatives.size(1)

    # Normalize for cosine similarity
    v_behav = F.normalize(v_behav, dim=-1)
    v_positive = F.normalize(v_positive, dim=-1)
    v_negatives = F.normalize(v_negatives, dim=-1)

    # Positive similarity: (B,)
    pos_sim = torch.sum(v_behav * v_positive, dim=-1) / temperature

    # Negative similarity: (B, K)
    neg_sim = torch.bmm(v_negatives, v_behav.unsqueeze(-1)).squeeze(-1) / temperature

    # Logits: concat pos and negs → (B, 1+K)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

    # Labels: always index 0 = positive
    labels = torch.zeros(B, dtype=torch.long, device=v_behav.device)

    # Cross entropy over (1+K) classes
    loss = F.cross_entropy(logits, labels)
    return loss

if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "distilbert-base-uncased"
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 53
    MAX_LENGTH = 128
   
    print(f"Using device: {DEVICE}")

    
    print("Creating mock data...")
    with open('file path of Metadata containing generated rationales','r') as f:

        data = json.load(f)
    
    training_data = data 

    # --- Initialization ---
    print("Initializing models and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Instantiate the two encoders
    rationale_encoder = ThoughtEncoder(MODEL_NAME).to(DEVICE)
    behavior_encoder = ThoughtEncoder(MODEL_NAME).to(DEVICE)

    # Create the dataset and dataloader
    dataset = TripletDataset(training_data, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup optimizer - it needs to know about the parameters of BOTH models
    params = list(rationale_encoder.parameters()) + list(behavior_encoder.parameters())
    optimizer = optim.AdamW(params, lr=LEARNING_RATE)


    # --- Start Training ---
    print("Starting training...")
    train_encoders(
        rationale_encoder=rationale_encoder,
        behavior_encoder=behavior_encoder,
        dataloader=dataloader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        num_negatives=10
    )
    print("Training complete.")

    
    torch.save(rationale_encoder.state_dict(), "rationale_encoder.pth")
    torch.save(behavior_encoder.state_dict(), "behavior_encoder.pth")
    print("Models saved successfully to 'rationale_encoder.pth' and 'behavior_encoder.pth'.")
