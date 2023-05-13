import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import torch
import torch.nn.functional as F
from torch import nn
import hashlib
from scipy.stats import norm
import gensim
import pdb
from transformers import BertForMaskedLM as WoBertForMaskedLM
from wobert import WoBertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import BertForMaskedLM, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import gensim.downloader as api
import Levenshtein
import string
import spacy
import paddle
from jieba import posseg

paddle.enable_static()
import re
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r'\1\n\2', para)  
    para = re.sub('([。！？\?][”’])([^，。！？\?\n ])', r'\1\n\2', para)  
    para = re.sub('(\.{6}|\…{2})([^”’\n])', r'\1\n\2', para)  
    para = re.sub('([^。！？\?]*)([:：][^。！？\?\n]*)', r'\1\n\2', para) 
    para = re.sub('([。！？\?][”’])$', r'\1\n', para) 
    para = para.rstrip()
    return para.split("\n")

def is_subword(token: str): 
    return token.startswith('##')

def binary_encoding_function(token):
    hash_value = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)
    random_bit = hash_value % 2
    return random_bit

def is_similar(x, y, threshold=0.5):
    distance = Levenshtein.distance(x, y)
    if distance / max(len(x), len(y)) < threshold:
        return True
    return False

class watermark_model:
    def __init__(self, language, mode, tau_word, lamda):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.mode = mode
        self.tau_word = tau_word
        self.tau_sent = 0.8
        self.lamda = lamda
        self.cn_tag_black_list = set(['','x','u','j','k','zg','y','eng','uv','uj','ud','nr','nrfg','nrt','nw','nz','ns','nt','m','mq','r','w','PER','LOC','ORG'])#set(['','f','u','nr','nw','nz','m','r','p','c','w','PER','LOC','ORG'])
        self.en_tag_white_list = set(['MD', 'NN', 'NNS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS'])
        if language == 'Chinese':
            self.relatedness_tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-330M-Similarity")
            self.relatedness_model = AutoModelForSequenceClassification.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-330M-Similarity").to(self.device)
            self.tokenizer = WoBertTokenizer.from_pretrained("junnyu/wobert_chinese_plus_base")
            self.model = WoBertForMaskedLM.from_pretrained("junnyu/wobert_chinese_plus_base", output_hidden_states=True).to(self.device)
            self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format('sgns.merge.word.bz2', binary=False, unicode_errors='ignore', limit=50000)
        elif language == 'English':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True).to(self.device)
            self.relatedness_model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli').to(self.device)
            self.relatedness_tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
            self.w2v_model = api.load("glove-wiki-gigaword-100")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            self.nlp = spacy.load('en_core_web_sm')

    def cut(self,ori_text,text_len):
        if self.language == 'Chinese':
            if len(ori_text) > text_len+5:
                ori_text = ori_text[:text_len+5]
            if len(ori_text) < text_len-5:
                return 'Short'
        elif self.language == 'English':
            tokens = self.tokenizer.tokenize(ori_text)
            if len(tokens) > text_len+5: 
                ori_text = self.tokenizer.convert_tokens_to_string(tokens[:text_len+5])
            if len(tokens) < text_len-5:
                return 'Short'
            return ori_text
        else:
            print(f'Unsupported Language:{self.language}')
            raise NotImplementedError  
    
    def sent_tokenize(self,ori_text):
        if self.language == 'Chinese':
            return cut_sent(ori_text)
        elif self.language == 'English':
            return nltk.sent_tokenize(ori_text)
    
    def pos_filter(self, tokens, masked_token_index, input_text):
        if self.language == 'Chinese':
            pairs = posseg.lcut(input_text)
            pos_dict = {word: pos for word, pos in pairs}
            pos_list_input = [pos for _, pos in pairs]
            pos = pos_dict.get(tokens[masked_token_index], '')
            if pos in self.cn_tag_black_list:
                return False
            else:
                return True
        elif self.language == 'English':
            pos_tags = pos_tag(tokens)
            pos = pos_tags[masked_token_index][1]
            if pos not in self.en_tag_white_list:
                return False
            if is_subword(tokens[masked_token_index]) or is_subword(tokens[masked_token_index+1]) or (tokens[masked_token_index] in self.stop_words or tokens[masked_token_index] in string.punctuation):
                return False
            return True
    
    def filter_special_candidate(self, top_n_tokens, tokens,masked_token_index,input_text):
        if self.language == 'English':
            filtered_tokens = [tok for tok in top_n_tokens if tok not in self.stop_words and tok not in string.punctuation and pos_tag([tok])[0][1] in self.en_tag_white_list and not is_subword(tok)]
            
            lemmatized_tokens = []
            # for token in filtered_tokens:
            #     doc = self.nlp(token)
            #     lemma = doc[0].lemma_ if doc[0].lemma_ != "-PRON-" else token
            #     lemmatized_tokens.append(lemma)
            
            base_word = tokens[masked_token_index] 
            base_word_lemma = self.nlp(base_word)[0].lemma_ 
            processed_tokens = [base_word]+[tok for tok in filtered_tokens if self.nlp(tok)[0].lemma_ != base_word_lemma]
            return processed_tokens
        elif self.language == 'Chinese':
            pairs = posseg.lcut(input_text)
            pos_dict = {word: pos for word, pos in pairs}
            pos_list_input = [pos for _, pos in pairs]
            pos = pos_dict.get(tokens[masked_token_index], '')
            filtered_tokens = []
            for tok in top_n_tokens:
                watermarked_text_segtest = self.tokenizer.convert_tokens_to_string(tokens[1:masked_token_index] + [tok] + tokens[masked_token_index+1:-1])
                watermarked_text_segtest = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])', '', watermarked_text_segtest)
                pairs_tok = posseg.lcut(watermarked_text_segtest)
                pos_dict_tok = {word: pos for word, pos in pairs_tok}
                flag = pos_dict_tok.get(tok, '')
                if flag not in self.cn_tag_black_list and flag == pos: 
                    filtered_tokens.append(tok)
            processed_tokens = filtered_tokens
            return processed_tokens
    
    def global_word_sim(self,word,ori_word):
        try:
            global_score = self.w2v_model.similarity(word,ori_word)
        except KeyError:
            global_score = 0
        return global_score
    
    def context_word_sim(self,init_candidates, tokens, masked_token_index, input_text):
        original_input_tensor = self.tokenizer.encode(input_text,return_tensors='pt').to(self.device)
        batch_input_ids = [[self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens[1:masked_token_index] + [token] + tokens[masked_token_index+1:-1]+ ['[SEP]'])] for token in init_candidates]
        batch_input_tensors = torch.tensor(batch_input_ids).squeeze().to(self.device)
        batch_input_tensors = torch.cat((batch_input_tensors,original_input_tensor),dim=0)
        with torch.no_grad():
            outputs = self.model(batch_input_tensors)
            cos_sims = torch.zeros([len(init_candidates)]).to(self.device)
            num_layers = len(outputs[1])
            N = 8
            i = masked_token_index
            cos_sim_sum = 0
            for layer in range(num_layers-N,num_layers):
                ls_hidden_states = outputs[1][layer][0:len(init_candidates), i, :]
                source_hidden_state = outputs[1][layer][len(init_candidates), i, :]
                cos_sim_sum += F.cosine_similarity(source_hidden_state, ls_hidden_states, dim=1)
            cos_sim_avg = cos_sim_sum / N
            
            cos_sims += cos_sim_avg
        return cos_sims.tolist()

    def sentence_sim(self,init_candidates, tokens, masked_token_index, input_text):
        if self.language == 'Chinese':
            batch_sents = [self.tokenizer.convert_tokens_to_string(tokens[1:masked_token_index] + [token] + tokens[masked_token_index+1:-1]) for token in init_candidates]
            batch_sentences = [re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])', '', sent) for sent in batch_sents]
            roberta_inputs = [input_text + '[SEP]' + s for s in batch_sentences]
        elif self.language == 'English':
            batch_sentences = [self.tokenizer.convert_tokens_to_string(tokens[1:masked_token_index] + [token] + tokens[masked_token_index+1:-1]) for token in init_candidates]
            roberta_inputs = [input_text + '</s></s>' + s for s in batch_sentences]
        
        encoded_dict = self.relatedness_tokenizer.batch_encode_plus(
                roberta_inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt')
        # Extract input_ids and attention_masks
        input_ids = encoded_dict['input_ids'].to(self.device)
        attention_masks = encoded_dict['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.relatedness_model(input_ids=input_ids, attention_mask=attention_masks)
            logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        if self.language == 'Chinese':
            relatedness_scores = probs[:, 1].tolist()
        elif self.language == 'English':
            relatedness_scores = probs[:, 2].tolist()
        
        return relatedness_scores
            
    def candidates_gen(self,tokens,masked_token_index,input_text,topk=64, dropout_prob=0.3):
        input_ids_bert = self.tokenizer.convert_tokens_to_ids(tokens)
        if not self.pos_filter(tokens,masked_token_index,input_text):
            return []
        masked_text = self.tokenizer.convert_tokens_to_string(tokens)
        # Create a tensor of input IDs
        input_tensor = torch.tensor([input_ids_bert]).to(self.device)

        with torch.no_grad():
            embeddings = self.model.bert.embeddings(input_tensor)
        dropout = nn.Dropout2d(p=dropout_prob)
        # Get the predicted logits
        embeddings[:, masked_token_index, :] = dropout(embeddings[:, masked_token_index, :])
        with torch.no_grad():
            outputs = self.model(inputs_embeds=embeddings)

        predicted_logits = outputs[0][0][masked_token_index]

        # Set the number of top predictions to return
        n = topk
        # Get the top n predicted tokens and their probabilities
        probs = torch.nn.functional.softmax(predicted_logits, dim=-1)
        top_n_probs, top_n_indices = torch.topk(probs, n)
        top_n_tokens = self.tokenizer.convert_ids_to_tokens(top_n_indices.tolist())
        processed_tokens = self.filter_special_candidate(top_n_tokens,tokens,masked_token_index)
          
        return processed_tokens

    def filter_candidates(self, init_candidates, tokens, masked_token_index, input_text):
        context_word_similarity_scores = self.context_word_sim(init_candidates, tokens, masked_token_index, input_text)
        sentence_similarity_scores = self.sentence_sim(init_candidates, tokens, masked_token_index, input_text)
        filtered_candidates = []
        for idx, candidate in enumerate(init_candidates):
            global_word_similarity_score = self.global_word_sim(tokens[masked_token_index], candidate)
            word_similarity_score = self.lamda*context_word_similarity_scores[idx]+(1-self.lamda)*global_word_similarity_score
            if word_similarity_score >= self.tau_word and sentence_similarity_scores[idx] >= self.tau_sent:
                filtered_candidates.append((candidate, word_similarity_score))#, sentence_similarity_scores[idx]))
        return filtered_candidates
        
    def watermark_embed(self,text):
        input_text = text
        # Tokenize the input text
        tokens = self.tokenizer.tokenize(input_text) 
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        masked_tokens=tokens.copy()
        start_index = 1
        end_index = len(tokens) - 1
        for masked_token_index in range(start_index+1, end_index-1):
            # pdb.set_trace()
            binary_encoding = binary_encoding_function(tokens[masked_token_index - 1] + tokens[masked_token_index])
            if binary_encoding == 1:
                continue
            init_candidates = self.candidates_gen(tokens,masked_token_index,input_text, 32, 0.3)
            if len(init_candidates) <=1:
                continue
            enhanced_candidates = self.filter_candidates(init_candidates,tokens,masked_token_index,input_text)
            hash_top_tokens = enhanced_candidates.copy()  
            for i, tok in enumerate(enhanced_candidates):
                binary_encoding = binary_encoding_function(tokens[masked_token_index - 1] + tok[0])
                if binary_encoding != 1 or (is_similar(tok[0], tokens[masked_token_index])) or (tokens[masked_token_index - 1] in tok or tokens[masked_token_index + 1] in tok):   
                    hash_top_tokens.remove(tok)                
            hash_top_tokens.sort(key=lambda x: x[1], reverse=True)    
            if len(hash_top_tokens) > 0:
                selected_token = hash_top_tokens[0][0]
            else:
                selected_token = tokens[masked_token_index]
            
            tokens[masked_token_index] = selected_token
        watermarked_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])
        if self.language == 'Chinese':
            watermarked_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])', '', watermarked_text)
        
        return watermarked_text
        
    def embed(self, ori_text):
        sents = self.sent_tokenize(ori_text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        watermarked_text = ''
        for i in range(0, num_sents, 2):
            if i+1 < num_sents:
                sent_pair = sents[i] + sents[i+1]
            else:
                sent_pair = sents[i]
            if len(watermarked_text) == 0:
                watermarked_text = self.watermark_embed(sent_pair)
            else:
                watermarked_text = watermarked_text + self.watermark_embed(sent_pair)
        if len(self.get_encodings_fast(ori_text)) == 0:
            return ''
        return watermarked_text
    
    def get_encodings_fast(self,text):
        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        encodings = []
        for i in range(0, num_sents, 2):
            if i+1 < num_sents:
                sent_pair = sents[i] + sents[i+1]
            else:
                sent_pair = sents[i]
            tokens = self.tokenizer.tokenize(sent_pair)
            
            for index in range(1,len(tokens)-1):
                if not self.pos_filter(tokens,index,text):
                    continue
                bit = binary_encoding_function(tokens[index-1]+tokens[index])
                encodings.append(bit)
        return encodings

    def watermark_detector_fast(self, text,alpha=0.05):
        p = 0.5
        encodings = self.get_encodings_fast(text)
        n = len(encodings)
        ones = sum(encodings)
        z = (ones - p * n) / (n * p * (1 - p)) ** 0.5 
        threshold = norm.ppf(1 - alpha, loc=0, scale=1)
        p_value = norm.sf(z)
        is_watermark = z >= threshold
        return is_watermark, p_value, n, ones, z
    
    def get_encodings_precise(self, text):
        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        encodings = []
        for i in range(0, num_sents, 2):
            if i+1 < num_sents:
                sent_pair = sents[i] + sents[i+1]
            else:
                sent_pair = sents[i]

            tokens = self.tokenizer.tokenize(sent_pair) 
            
            tokens = ['[CLS]'] + tokens + ['[SEP]']

            masked_tokens=tokens.copy()

            start_index = 1
            end_index = len(tokens) - 1

            for masked_token_index in range(start_index+1, end_index-1):
                init_candidates = self.candidates_gen(tokens,masked_token_index,sent_pair, 8, 0)        
                if len(init_candidates) <=1:
                    continue
                enhanced_candidates = self.filter_candidates(init_candidates,tokens,masked_token_index,sent_pair)      
                if len(enhanced_candidates) > 1:
                    bit = binary_encoding_function(tokens[masked_token_index-1]+tokens[masked_token_index])
                    encodings.append(bit)
        return encodings

    def watermark_detector_precise(self,text,alpha=0.05):
        p = 0.5
        encodings = self.get_encodings_precise(text)
        n = len(encodings)
        ones = sum(encodings)
        if n == 0:
            z = 0 
        else:
            z = (ones - p * n) / (n * p * (1 - p)) ** 0.5
        threshold = norm.ppf(1 - alpha, loc=0, scale=1)
        p_value = norm.sf(z)
        is_watermark = z >= threshold
        return is_watermark, p_value, n, ones, z