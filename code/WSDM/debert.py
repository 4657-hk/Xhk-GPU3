import pandas as pd
from sklearn.metrics import accuracy_score

# 读取 Parquet 文件
train_df = pd.read_parquet("../train.parquet")
test_df = pd.read_parquet("../test.parquet")
#%%
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification
# Load model directly
# Load model directly
from transformers import AutoModel
tokenizer = AutoTokenizer.from_pretrained("../debert-v3")
model = DebertaV2ForSequenceClassification .from_pretrained("../debert-v3",num_labels=2)
#%%
inputs = train_df['prompt'] + train_df['response_a']+train_df['response_b']
encodings = tokenizer(list(inputs),max_length=512, truncation=True, padding=True)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

labels = train_df['winner'].apply(lambda x: 0 if x == 'model_a' else 1)
# %%
import torch

input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)  # 转换为张量
attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)  # 同样转换为张量
labels_tensor = torch.tensor(labels, dtype=torch.long)  # 转换 labels 为张量
# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)
train, val = train_test_split(train_dataset, test_size=0.3, random_state=42)

train_loader = DataLoader(train, batch_size=16, shuffle=True)
val_loader = DataLoader(val, batch_size=16, shuffle=True)

# %%
import numpy as np
import torch

print(np.__version__)  # 检查 numpy 版本
print(torch.__version__)  # 检查 torch 版本
print(len(train))
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# %%
for batch in train_loader:
    print(batch)
    break
# %%
from torch.optim import Adam
from torch import nn
from tqdm import tqdm

optimizer = Adam(model.parameters(), lr=1e-5)
for epoch in range(15):
    model.train()
    total_loss = 0
    for batch_input_ids, batch_attention_mask, batch_labels in tqdm(train_loader):
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask,labels=batch_labels)
        loss = outputs.loss  # 直接从模型输出中获取损失
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}")
torch.save(model.state_dict(), f'deberta_epoch_{epoch}.pth')


true_labels = []
predicted_labels = []
model.eval()
with torch.no_grad():
    for batch_input_ids, batch_attention_mask, batch_labels in tqdm(val_loader):
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        true_labels.extend(batch_labels.cpu().numpy())
        predicted_labels.extend(predictions.cpu().numpy())
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Validation Accuracy: {accuracy:.4f}")

