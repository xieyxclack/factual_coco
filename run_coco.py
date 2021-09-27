import os
import argparse
import torch
import numpy as np
from fairseq.models.bart import BARTModel
import spacy


def init_vocab():

    vocab = dict()
    with open('vocab_aligned') as f:
        for line in f:
            try:
                index, _, word = line[:-1].split('\t')
                vocab[int(index)] = word
            except:
                pass
    
    # add the special tokens
    vocab[0] = '<s>'
    vocab[1] = '<pad>'
    vocab[2] = '</s>'
    vocab[3] = '<unk>'
    vocab[50264] = '<mask>'

    return vocab


class Tokenizer(object):

    def __init__(self, model='en_core_web_sm', use_gpu=True):

        super(Tokenizer, self).__init__()
        if use_gpu:
            spacy.prefer_gpu()
        self.nlp_model = spacy.load(model)

    def tokenize_and_pos(self, text):

        result = self.nlp_model(text)
        tokens = [x.text for x in result]
        pos_tags = [x.pos_ for x in result]
        return tokens, pos_tags

    def sentencizer(self, doc):

        result = self.nlp_model(doc)
        return [sent.text for sent in result.sents]


def merge_subwords(subwords, summ_scores, mask_scores):
    
    assert len(subwords) == len(summ_scores)
    assert len(mask_scores) == len(summ_scores)

    merge_word = list()
    merge_summ_score = list()
    merge_mask_score = list()
    cur_word = list()
    cur_summ_score = list()
    cur_mask_score = list()
    for subword, summ_score, mask_score in zip(subwords, summ_scores, mask_scores):
        if subword in ['', ' ', '<s>', '</s>']:
            continue

        if subword.startswith(' ') and len(cur_word) > 0:
            merge_word.append(''.join(cur_word).lstrip())
            merge_summ_score.append(cur_summ_score[0])
            merge_mask_score.append(cur_mask_score[0])
            cur_word.clear()
            cur_summ_score.clear()
            cur_mask_score.clear()

        cur_word.append(subword)
        cur_summ_score.append(summ_score)
        cur_mask_score.append(mask_score)

    if len(cur_word) != 0:
        merge_word.append(''.join(cur_word).lstrip())
        merge_summ_score.append(cur_summ_score[0])
        merge_mask_score.append(cur_mask_score[0])

    return merge_word, merge_summ_score, merge_mask_score   
    

def get_coco_score(summ_model, source_doc, masked_doc, generated_summ, masked_token_list):

    # get the scores when feed the source document and the masked source document, respectively
    summ_result, tokenized_summ = summ_model.score(generated_summ, sources=source_doc)
    mask_summ_result, _ = summ_model.score(generated_summ, sources=masked_doc)
    summ_score = summ_result[0]['positional_scores'].cpu().numpy()
    summ_score = np.exp(summ_score)
    mask_score = mask_summ_result[0]['positional_scores'].cpu().numpy()
    mask_score = np.exp(mask_score)

    # decode and merget sub-words (since BART adopts the BPE tokenizer)
    tokenized_summ = [vocab[token] for token in tokenized_summ[0].numpy()]
    merge_tokens, summ_score, mask_score = merge_subwords(tokenized_summ, summ_score, mask_score)

    scores = [summ_score[idx]-mask_score[idx] for idx in range(len(merge_tokens)) if merge_tokens[idx] in masked_token_list]

    if len(scores) > 0:
        coco_score = np.mean(scores)
    else:
        coco_score = 0.0
    return coco_score


def mask(source_doc, masked_token_list, tokenizer, MASK_TOKEN='<mask>', mask_strategy='token'):

    tokenized_doc = tokenizer.tokenize_and_pos(source_doc)[0]
    masked_token_list = [x.lower() for x in masked_token_list]
    mask_matrix = np.ones_like(tokenized_doc, dtype=np.int32)
    if mask_strategy == 'doc':
        mask_matrix = np.zeros_like(tokenized_doc, dtype=np.int32)
    elif mask_strategy == 'token':
        for idx,word in enumerate(tokenized_doc):
            if word.lower() in masked_token_list:
                mask_matrix[idx] = 0
    elif mask_strategy == 'span':
        for idx, word in enumerate(tokenized_doc):
            if word.lower() in masked_token_list:
                mask_matrix[idx] = 0
                if idx-1 >= 0:
                    mask_matrix[idx-1] = 0
                if idx-2 >= 0:
                    mask_matrix[idx-2] = 0
                if idx+1 < len(tokenized_doc):
                    mask_matrix[idx+1] = 0
                if idx+2 < len(tokenized_doc):
                    mask_matrix[idx+2] = 0
    elif mask_strategy == 'sent':
        sents = tokenizer.sentencizer(source_doc)
        mask_matrix = []
        for sent in sents:
            token_sent = tokenizer.tokenize_and_pos(sent)[0]
            token_sent = [x.lower() for x in token_sent]
            sent_mask_matrix = np.ones_like(token_sent, dtype=np.int32)
            for masked_word in masked_token_list:
                if masked_word in token_sent:
                    sent_mask_matrix = np.zeros_like(token_sent, dtype=np.int32)
                    break
            mask_matrix.append(sent_mask_matrix)
        mask_matrix = np.concatenate(mask_matrix, axis=0)

    assert len(tokenized_doc) == len(mask_matrix)
    masked_doc = np.where(mask_matrix.astype(bool), tokenized_doc, [MASK_TOKEN]*len(tokenized_doc))

    return ' '.join(masked_doc)
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./bart.large.cnn', help='path to scoring model')
    parser.add_argument('--data_path', type=str, default='data', help='path to data, which includes the source documents and summaries')
    parser.add_argument('--output_file', type=str, default='coco_score.txt', help='output file for saving the results')
    parser.add_argument('--mask', type=str, default='token', help='mask strategy (token/span/sent/doc)')
    args = parser.parse_args()

    ## load the scoring model
    summ_model = BARTModel.from_pretrained(args.model_path, checkpoint_file='model.pt')
    summ_model.cuda()
    summ_model.eval()

    ## get the vocabulary (for decoding)
    vocab = init_vocab()
    print('\t initialization done!')

    ## mask strategy
    mask_strategy = args.mask
    if mask_strategy not in ['token', 'span', 'sent', 'doc']:
        print('\t The provided mask strategy is error! The default mask strategy (i.e., the token-level mask strategy) will be used ...')
        mask_strategy = 'token'


    ## Tokenizer and pos tagging model
    tokenizer = Tokenizer()
    universal_pos_tags = ['ADJ','ADP','ADV', 'AUX', 'CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X']
    ### The universal part-of-speech (pos) categories can be refered to https://universaldependencies.org/u/pos/, which include:
    # ADJ: adjective
    # ADP: adposition
    # ADV: adverb
    # AUX: auxiliary
    # CCONJ: coordinating conjunction
    # DET: determiner
    # INTJ: interjection
    # NOUN: noun
    # NUM: numeral
    # PART: particle
    # PRON: pronoun
    # PROPN: proper noun
    # PUNCT: punctuation
    # SCONJ: subordinating conjunction
    # SYM: symbol
    # VERB: verb
    # X: other
    unimportant_pos_tags = ['PUNCT', 'SYM', 'DET', 'PART', 'CCONJ', 'SCONJ']
    important_pos_tags = [tag for tag in universal_pos_tags if tag not in unimportant_pos_tags]

    # calculate the coco scores
    coco_scores = []
    count = 0
    with open(os.path.join(args.data_path,'source.txt')) as source_file, open(os.path.join(args.data_path,'summary.txt')) as summ_file:
        for source_doc, generated_summ in zip(source_file, summ_file):
            #read file
            source_doc = source_doc.strip()
            generated_summ = generated_summ.strip()

            #counter
            count += 1
            if count % 100 == 0:
                print('Working! {:d} summaries have been finished ...'.format(count))

            ## get the masked tokens list, and generate the masked document
            summ_tokens, summ_tags = tokenizer.tokenize_and_pos(generated_summ)
            masked_token_list = [k for k,v in zip(summ_tokens, summ_tags) if v in important_pos_tags]
            masked_doc = mask(source_doc, masked_token_list, tokenizer, mask_strategy=mask_strategy)

            # get the coco score
            coco_score = get_coco_score(summ_model, source_doc, masked_doc, generated_summ, masked_token_list)
            coco_scores.append(coco_score)

    # write out the results
    with open(args.output_file,'w') as out_file:
        for coco_score in coco_scores:
            out_file.write(str(coco_score)+'\n')

    print('Done! {:d} summaries have been finished ...'.format(count))

