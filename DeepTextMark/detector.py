from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.metrics import accuracy_score
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# Load Dataset
with open('deeptext_dataset_0930.json', 'r') as f:
    dataset = json.load(f)

train_data = [(item['sentence'], item['label']) for item in dataset['train'][:]]
val_data = [(item['sentence'], item['label']) for item in dataset['validation'][:]]

print(len(dataset['train']))
# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Function to tokenize a batch of sentences
def tokenize_batch(batch):
    return tokenizer(batch, padding='max_length', truncation=True, return_tensors="pt", max_length=512)

# BERT Model
bert = BertModel.from_pretrained('bert-base-cased')

# # Transformer Classification Block
# class TransformerClassificationBlock(nn.Module):
#     def __init__(self):
#         super(TransformerClassificationBlock, self).__init__()
#         transformer_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16)
#         self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
#         self.classifier = nn.Linear(768, 2)

#     def forward(self, x):
#         x = self.transformer_encoder(x)
#         x = self.classifier(x)
#         return x[:, 0, :]  # Taking the classification output from the [CLS] token
class TransformerClassificationBlock(nn.Module):
    def __init__(self):
        super(TransformerClassificationBlock, self).__init__()
        
        # Transformer Decoder Layer
        # transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=16)
        transformer_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16)
        # self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
        
        # Pooling (assuming average pooling)
        # self.pooling = nn.AdaptiveAvgPool1d(768)
        self.pooling = nn.AdaptiveAvgPool2d((1, 768))

        
        # Fully-connected layers
        self.fc1 = nn.Linear(768, 20)
        self.fc2 = nn.Linear(20, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Pass through the Transformer Decoder
        # Note: You might need to adjust the memory and target input as per your use-case
        # x = self.transformer_decoder(tgt=x, memory=x) 
        # pdb.set_trace()
        x = self.transformer_encoder(x) 
        
        # Pooling
        # x = self.pooling(x.transpose(1,2)).transpose(1,2)
        x = self.pooling(x)
        x = x.squeeze(1)  # Remove the sequence length dimension, resulting in (batch_size, 768)
        x = self.dropout(x)

        
        # Pass through first fully-connected layer
        x = self.fc1(x)
        x = self.dropout(x)
        
        # Pass through second fully-connected layer
        x = self.fc2(x)
        
        # Taking the output from the [CLS] token for binary classification
        return x.squeeze()


# Watermark Detector (Binary Classifier)
class WatermarkDetector(nn.Module):
    def __init__(self):
        super(WatermarkDetector, self).__init__()
        self.bert = bert
        self.classifier_block = TransformerClassificationBlock()

    def forward(self, x):
        # with torch.no_grad():
        x = self.bert(**x)[0]
        x = self.classifier_block(x)
        return x

# class WatermarkDetector(nn.Module):
#     def __init__(self, bert_model):
#         super(WatermarkDetector, self).__init__()
#         self.bert = bert_model
#         self.classifier = nn.Linear(768, 1)  # assuming BERT base model with 768 hidden size

#     def forward(self, x):
#         # Assuming x is a dictionary with keys 'input_ids', 'attention_mask', etc.
#         outputs = self.bert(**x)
#         # Take the representation of [CLS] token which is the first token in the sequence
#         cls_output = outputs[0][:, 0, :]
#         # Use the classifier layer
#         logits = self.classifier(cls_output)
#         return logits


# Initialize the model
detector = WatermarkDetector()
detector.to(device)
# Criterion and Optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()


# Freeze BERT encoder parameters
for param in detector.bert.parameters():
    param.requires_grad = False

# Optimizer with initial learning rate
optimizer = optim.Adam(detector.parameters(), lr=0.0001)
# optimizer = optim.Adam(detector.parameters(), lr=0.001)

# Tokenize and Prepare DataLoader
train_texts, train_labels = zip(*train_data)
val_texts, val_labels = zip(*val_data)

train_encodings = tokenize_batch(train_texts)
val_encodings = tokenize_batch(val_texts)

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.Tensor(train_labels).long())
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.Tensor(val_labels).long())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training Loop with Loss and Validation Accuracy
# Initial training phase with frozen BERT parameters
EPOCHS = 6
import pdb
for epoch in range(EPOCHS):
    # Training
    detector.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        labels = labels.float()
        
        outputs = detector({'input_ids': input_ids, 'attention_mask': attention_mask})
        # pdb.set_trace()
        # loss = criterion(outputs, labels)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Training Loss: {avg_train_loss}")

    # Validation
    detector.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # outputs = detector({'input_ids': input_ids, 'attention_mask': attention_mask})
            # preds = torch.argmax(outputs, dim=1)
            # all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())

            outputs = detector({'input_ids': input_ids, 'attention_mask': attention_mask})
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()  # Convert logit to probability and then to binary
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {val_accuracy}")

    # Save checkpoint
    # torch.save(detector.state_dict(), f'checkpoint_epoch_{epoch+1}_frozen.pth')

# Unfreeze BERT encoder parameters
for param in detector.bert.parameters():
    param.requires_grad = True

# Update learning rate for further training
optimizer = optim.Adam([
    {'params': detector.bert.parameters(), 'lr': 1e-6},
    {'params': detector.classifier_block.parameters(), 'lr': 1e-6}
])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# Second training phase with unfrozen BERT parameters
EPOCHS = 70
for epoch in range(EPOCHS):
    # Training
    detector.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        labels = labels.float()

        
        outputs = detector({'input_ids': input_ids, 'attention_mask': attention_mask})
        loss = criterion(outputs.squeeze(), labels.float())

        # loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Training Loss: {avg_train_loss}")

    # Validation
    detector.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.float().unsqueeze(1)

        
            # outputs = detector({'input_ids': input_ids, 'attention_mask': attention_mask})
            # loss = criterion(outputs, labels.float().unsqueeze(1))
            # preds = torch.argmax(outputs, dim=1)
            # all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())
            outputs = detector({'input_ids': input_ids, 'attention_mask': attention_mask})
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()  # Convert logit to probability and then to binary
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    val_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {val_accuracy}")

    # Save checkpoint
    if epoch%10==9:
        torch.save(detector.state_dict(), f'checkpoint_epoch_{epoch+1}_unfrozen0930.pth')
 