from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
# from tensorflow_hub import load
import tensorflow_hub as hub
import nltk
import string
import numpy as np
import gensim.downloader as api
import pdb
# Initialize Universal Sentence Encoder
# universal_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

import tensorflow as tf
universal_encoder = tf.saved_model.load("/your_encoder/")
# Preprocess sentence
def preprocess(sentence):
    words = nltk.word_tokenize(sentence)
    filtered_words = [word for word in words if word.isalnum() and word.lower() not in nltk.corpus.stopwords.words('english')]
    return filtered_words

# Generate sentence proposals
def generate_sentence_proposals(sentence, model, n):
    processed_sentence = preprocess(sentence)
    sentence_proposals = []
    
    for word in processed_sentence:
        if word not in model.key_to_index:  # Skip words that are not in the vocabulary
            continue
        word_vec = word_embedding(model, word)
        # Get similar words but exclude the original word
        similar_words = [w for w, _ in model.most_similar(positive=[word_vec], topn=n) if w != word]
        
        for similar_word in similar_words:
            new_sentence = sentence.replace(word, similar_word)
            sentence_proposals.append(new_sentence)
            
    return list(set(sentence_proposals))




# Generate embedding using Word2Vec or GloVe KeyedVectors
def word_embedding(model, word):
    return model[word]  # Removed .wv


def score_sentences(original_sentence, sentence_proposals):
    original_sentence_vec = universal_encoder([original_sentence]).numpy()
    proposal_vectors = universal_encoder(sentence_proposals).numpy()
    scores = cosine_similarity(original_sentence_vec, proposal_vectors)
    best_proposal = sentence_proposals[np.argmax(scores)]
    return best_proposal



# Initialize Word2Vec model (Assume `sentences` is your training data)
model = api.load("glove-wiki-gigaword-100")
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")

# Load the saved Word2Vec model
# model = Word2Vec.load("word2vec.model")

# Main watermarking function
def watermark_sentence(sentence, model, n):
    sentence_proposals = generate_sentence_proposals(sentence, model, n)
    # print(sentence_proposals)
    # print(sentence)
    watermarked_sentence = score_sentences(sentence, sentence_proposals)
    return watermarked_sentence

# sentence = 'This is where we discovered that there is a watermark imperceptibility vs detection accuracy trade-off.'
# watermark_sentence(sentence, model, 2)





import nltk
from nltk.corpus import brown, gutenberg
from sklearn.model_selection import train_test_split
import random
import pdb
import json
# Download resources
nltk.download('brown')
nltk.download('gutenberg')
nltk.download('punkt')

# Collect sentences from the Brown Corpus
brown_sents = [' '.join(sent) for sent in brown.sents()]

# Collect sentences from the Gutenberg Corpus
gutenberg_sents = [' '.join(sent) for sent in gutenberg.sents()]
# pdb.set_trace()
# Combine the sentences
combined_sents = [sentence for sentence in brown_sents + gutenberg_sents if len(sentence.split()) >= 5]

# Shuffle and truncate to 138,638 sentences
random.shuffle(combined_sents)
truncated_sents = combined_sents[:198698]
# truncated_sents = combined_sents[:1386]

# Splitting into 'watermarked' and 'unmarked' sets
watermarked_sents, unmarked_sents = train_test_split(truncated_sents, test_size=0.5, random_state=42)

# Apply watermarking
new_watermarked_sents = []
for sentence in watermarked_sents:
    try:
        new_watermarked_sentence = watermark_sentence(sentence, model, 4)
        if sentence!=new_watermarked_sentence:
            new_watermarked_sents.append((new_watermarked_sentence, 1))
        # print(sentence)
        # print(new_watermarked_sentence)
        # pdb.set_trace()
    except Exception as e:
        print(f"Skipping sentence due to error: {e}")

unmarked_sents = [(sentence, 0) for sentence in unmarked_sents[:len(new_watermarked_sents)]]

# Split for training and validation
watermarked_train, watermarked_val = train_test_split(new_watermarked_sents, test_size=0.25, random_state=42)
unmarked_train, unmarked_val = train_test_split(unmarked_sents, test_size=0.25, random_state=42)

# Final training and validation sets
train_sents = watermarked_train + unmarked_train
val_sents = watermarked_val + unmarked_val

# Shuffle the training and validation sets
random.shuffle(train_sents)
random.shuffle(val_sents)

# Save as JSON
dataset = {
    'train': [{"sentence": sentence, "label": label} for sentence, label in train_sents],
    'validation': [{"sentence": sentence, "label": label} for sentence, label in val_sents]
}

with open('deeptext_dataset_0930.json', 'w') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("Training set size:", len(train_sents))
print("Validation set size:", len(val_sents))
print("Dataset saved as watermarked_dataset.json")