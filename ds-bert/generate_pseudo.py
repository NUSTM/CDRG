import logging
import math
import os
from dataclasses import dataclass, field
import torch
import copy
from typing import Optional
import numpy as np, random
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

from functools import reduce
import torch.nn.functional as F
from collections import defaultdict
import argparse
from tqdm import tqdm
# from pytorch_pretrained_bert.tokenization import BertTokenizer

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    set_seed,
)


def subwords_tokenizer(tokenizer, words, labels):
	# 为了便于对齐，只使用sub word tokenizer并同步扩展标签。
    new_words, new_labels = [], []
    for w, l in zip(words, labels):
        word_list = tokenizer.wordpiece_tokenizer.tokenize(w)
        for w_ in word_list:
            new_words.append(w_)
            new_labels.append(l)
    assert len(new_words) == len(new_labels)
    return new_words, new_labels


def decode_text(tokenizer, labels, raw_text, prediction_score, sub_aspects, pos_sub_opinions,neg_sub_opinions,neu_sub_opinions):
    '''
    全局变量 sub_aspect, sub_opinions，存放目标领域的属性词和观点词，当作生成时的词典.
    :param tokenizer
    :param labels: 用于表明当前词是属性词还是观点词
    :param raw_text: ['I', 'like', 'the', '[MASK]', '[MASK]']  # token_nums
    :param prediction_score: list(list) token_nums + 2 左右各一个特殊符号 seq_len * vocab_size
    :return: decode text
    '''
    new_text = []
    covered = set()
    for i in range(len(raw_text)):
        if i in covered:
            continue
        if raw_text[i] != '[MASK]':  # raw_text[i] != '[MASK]'
            s = tokenizer.convert_ids_to_tokens([torch.argmax(prediction_score[i + 1])])
            new_text.extend(s)
            covered.add(i)
            # all_pre_aspects.add(' '.join(s))
        else:
            start, end = i, i
            # 把连续的MASK当作一个词组
            for end in range(start, len(raw_text) + 1):
                if end == len(raw_text) or raw_text[end] != '[MASK]' or labels[end] != labels[start]:
                    break
                covered.add(end)
            if end == start:
                end = start + 1
            sub_len = end - start
            max_text, max_scores = '', -1
            if not labels[i].endswith('opinions'):  # 解码出属性词
                if sub_len not in sub_aspects:
                    # 依赖BERT生成
                    max_text = [tokenizer.convert_ids_to_tokens([torch.argmax(prediction_score[index + 1])]) for index
                                in range(start, end)]
                    max_text = [t[0] for t in max_text]  # t[0]
                    print(end - start, 'aspects')
                    assert len(max_text) == end - start, print('not in!', end - start)
                else:  
                	# 依赖于词典和BERT选词
                    top_text = [
                        (w, reduce(lambda x, y: x * y, [prediction_score[start + index + 1][ids] for index, ids in
                                                        enumerate(tokenizer.convert_tokens_to_ids(w))]).detach().cpu())
                        for w in sub_aspects[sub_len]]
                    top_text = sorted(top_text, key=lambda x: x[1], reverse=True)
                    max_text = top_text[0][0]


            else:  # 解码出观点词
                if labels[i].startswith('pos'):
                    sub_opinions=pos_sub_opinions
                elif labels[i].startswith('neg'):
                    sub_opinions=neg_sub_opinions
                elif labels[i].startswith('neu'):
                    sub_opinions=neu_sub_opinions
                if sub_len not in sub_opinions:
                    # 依赖BERT生成
                    max_text = [tokenizer.convert_ids_to_tokens([torch.argmax(prediction_score[index + 1])]) for index
                                in range(start, end)]
                    max_text = [t[0] for t in max_text]
                    print(sub_len,end - start, 'opinions')
                    assert len(max_text) == end - start, print('not in!')
                else:
                    # print("123")  
                    # 依赖于词典和BERT选词
                    top_text = [
                        (w, reduce(lambda x, y: x * y, [prediction_score[start + index + 1][ids] for index, ids in
                                                        enumerate(tokenizer.convert_tokens_to_ids(w))]).detach().cpu())
                        for w in sub_opinions[sub_len]]
                    top_text = sorted(top_text, key=lambda x: x[1], reverse=True)
                    max_text = random.choice(top_text[:20])[0]
                    max_text = top_text[0][0]
                    
            new_text.extend(max_text)
    return new_text


def max_match(text, labels, tmp_labels, aspects, pos_opinions,neg_opinions,neu_opinions):
    '''
    text: list(list(str)) 句子列表
    vocab: 词典
    基于词典的最大匹配: 找到text中所包含的来自于aspects的词
    '''
    muliti_ = 0
    for index in range(len(text)):
        line, label, tmp_label = text[index], labels[index], tmp_labels[index]
        covered = set()
        for start in range(len(line)):
            if start in covered:
                continue

            # private mask
            for end in range(len(line) - 1, start - 1, -1):
                if ' '.join(line[start: end + 1]).lower() in aspects:
                    # print(' '.join(line[start: end + 1]))
                    if end != start:
                        muliti_ += 1
                    if any([i in covered for i in range(start, end + 1)]):
                        continue
                    covered.update([i for i in range(start, end + 1)])
                    break

            if start not in covered and not line[start].startswith('##'):
                end = start
                while end + 1 < len(line):
                    if not line[end + 1].startswith('##'):
                        break
                    end += 1
                if ' '.join(line[start: end + 1]) in pos_opinions:
                    line[start: end + 1] = ['[MASK]' for i in range(start, end + 1)]
                    tmp_label[start: end + 1] = ['pos_private_opinions' for i in range(start, end + 1)]
                elif ' '.join(line[start: end + 1]) in neg_opinions:
                    line[start: end + 1] = ['[MASK]' for i in range(start, end + 1)]
                    tmp_label[start: end + 1] = ['neg_private_opinions' for i in range(start, end + 1)]
                elif ' '.join(line[start: end + 1]) in neu_opinions:
                    line[start: end + 1] = ['[MASK]' for i in range(start, end + 1)]
                    tmp_label[start: end + 1] = ['neu_private_opinions' for i in range(start, end + 1)]

        for i in covered:
            line[i] = '[MASK]'
            tmp_label[i] = 'private_features'
    return muliti_

def get_sub_vocab(tokenizer, vocab):
    new_sub_vocab = set()
    for word in vocab:
        sub_word, _ = subwords_tokenizer(tokenizer, word.split(), word.split())
        if ' '.join(sub_word) in stopwords.words('english') or len(' '.join(sub_word)) == 1:
            continue
        new_sub_vocab.add(' '.join(sub_word))
    return new_sub_vocab


def get_features(text_list, labels_list):
	# 从标注样例中抽取属性词
    aspects = set()
    for index in range(len(text_list)):
        line, label = text_list[index], labels_list[index]
        covered = set()
        for start in range(len(line)):
            if start in covered:
                continue
                # ground truth
            if label[start] != 'O':
                left, right = start, start
                while left - 1 >= 0:
                    if label[left - 1] == 'O':
                        break
                    left -= 1
                while right + 1 < len(line):
                    if label[right + 1] == 'O':
                        break
                    right += 1
                aspects.add(' '.join(line[left: right + 1]))
                covered.update([i for i in range(left, right + 1)])
    return aspects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, default='laptop', help="soruce domain")
    parser.add_argument('--target_domain', type=str, default='rest', help="target domain")
    parser.add_argument('--model_name_or_path', type=str, default="./bert_lm_models/ds-bert-e/",
                        help="./bert_lm_models/ds-bert-e/")
    parser.add_argument('--do_lower', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, default="./pseudo_output/ds-bert-e")
    parser.add_argument('--batch_size', type=int, default=40)
    args = parser.parse_args()
    source_domain = args.source_domain
    target_domain = args.target_domain
    model_name_or_path = args.model_name_or_path #+ target_domain
    print(model_name_or_path)
    print(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelWithLMHead.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        # config=config,
        cache_dir='./bert_lm_models/bert-base-uncased',
    )
    model.eval()
    model.cuda()

    # ------------------------------------------------------------
    # 读取观点词
    with open('./aspect_output/' + source_domain + '/new_opinions.txt', encoding='utf-8') as fp:
        source_opinions = get_sub_vocab(tokenizer, set(fp.read().lower().splitlines()))

    with open('./aspect_output/' + target_domain + '/new_opinions.txt', encoding='utf-8') as fp:
        target_opinions = get_sub_vocab(tokenizer, set(fp.read().lower().splitlines()))

    
    # 读取属性词
    with open('./aspect_output/' + source_domain + '/extented_aspects.txt', 'r', encoding='utf-8') as fp:
        source_features = get_sub_vocab(tokenizer, set(fp.read().lower().splitlines()))

    with open('./aspect_output/' + target_domain + '/extented_aspects.txt', 'r', encoding='utf-8') as fp:
        target_features = get_sub_vocab(tokenizer, set(fp.read().lower().splitlines()))
    
    # 读取情感观点词
    # 源领域特有的观点词
    source_pos_opinions=[]
    source_neg_opinions=[]
    source_neu_opinions=[]
    with open('./aspect_output/' + source_domain + '/opinions_sentiment.txt', encoding='utf-8') as fp:
        lines =fp.read().lower().splitlines()
        for line in lines:
            word,sentiment=line.split()
            sentiment=int(sentiment)
            if sentiment==1:
                source_pos_opinions.append(word)
            elif sentiment==-1:
                source_neg_opinions.append(word)
            else:
                source_neu_opinions.append(word)
        source_pos_opinions = get_sub_vocab(tokenizer, set(source_pos_opinions))- target_opinions-source_features
        source_neg_opinions = get_sub_vocab(tokenizer, set(source_neg_opinions))- target_opinions-source_features
        source_neu_opinions = get_sub_vocab(tokenizer, set(source_neu_opinions))- target_opinions-source_features
    # 目标领域特有的观点词
    target_pos_opinions=[]
    target_neg_opinions=[]
    target_neu_opinions=[]
    with open('./aspect_output/' + target_domain + '/opinions_sentiment.txt', encoding='utf-8') as fp:
        lines =fp.read().lower().splitlines()
        for line in lines:
            word,sentiment=line.split()
            sentiment=int(sentiment)
            if sentiment==1:
                target_pos_opinions.append(word)
            elif sentiment==-1:
                target_neg_opinions.append(word)
            else:
                target_neu_opinions.append(word)
        target_pos_opinions = get_sub_vocab(tokenizer, set(target_pos_opinions))-source_opinions-target_features
        target_neg_opinions = get_sub_vocab(tokenizer, set(target_neg_opinions))-source_opinions-target_features
        target_neu_opinions = get_sub_vocab(tokenizer, set(target_neu_opinions))-source_opinions-target_features

    
    # source_private_opinions = source_opinions - target_opinions - source_features
    # print("source_private_opinions...................................................")
    # print(len(source_private_opinions))
    # 源领域特有的属性词
    # source_private_features = source_features - target_features - target_opinions
    

    # print(source_private_features)
    # 按照属性词不同的长度存放aspect words {1： [food, dessert, ...], 2: [[indian, food], ....]}
    

    # 按照观点词的长度(sub words len)存放在字典里 用于解码
    pos_sub_opinions = defaultdict(list)
    neg_sub_opinions = defaultdict(list)
    neu_sub_opinions = defaultdict(list)
    for o in target_pos_opinions:
        o = o.split()
        pos_sub_opinions[len(o)].append(o)
    for o in target_neg_opinions:
        o = o.split()
        neg_sub_opinions[len(o)].append(o)
    for o in target_neu_opinions:
        o = o.split()
        neu_sub_opinions[len(o)].append(o)
    # 读取源领域的有标签训练数据并进行MASK
    with open('./raw_data/' + source_domain + '_train.txt', 'r', encoding='utf-8') as fp:
        lines = fp.read().splitlines()
        lines = [t.split('####')[1] for t in lines]

    text_list = [[raw.split('=')[0].lower() if args.do_lower else raw.split('=')[0] for raw in t.split()] for t in
                 lines]
    labels_list = [[raw.split('=')[-1] for raw in t.split()] for t in lines]

    raw_text_list = copy.deepcopy(text_list)
    for i in range(len(text_list)):
        text_list[i], labels_list[i] = subwords_tokenizer(tokenizer, text_list[i], labels_list[i])
    tmp_labels_list = copy.deepcopy(labels_list)

    # ground truth mask: 使用源领域的属性标签获取属性词词典
    source_private_features = get_features(text_list, labels_list) - target_features -source_opinions

    target_private_features=target_features-get_features(text_list, labels_list)-target_opinions
    sub_aspects = defaultdict(list)  # 用于解码属性词
    for w in target_private_features:
        w = w.split()
        sub_aspects[len(w)].append(w)
    

    # DP mask: 使用DP算法的抽取结果获取词典
    # source_private_features = source_features - target_features

    max_match(text_list, labels_list, tmp_labels_list, source_private_features, source_pos_opinions,source_neg_opinions
    ,source_neu_opinions)

    # 样例展示
    print('dataset size：{}'.format(len(text_list)))
    for i in range(3):
        print('raw text: ')
        print(raw_text_list[i])
        print('mask text: ')
        print(text_list[i])
        print('labels: ')
        print(labels_list[i])
        print('mask labels: ')
        print(tmp_labels_list[i])

    # 逐batch送入bert中，生成目标领域的文本。
    output_examples = []
    batch = args.batch_size

    for index in tqdm(range(0, len(text_list), batch)):
        next_batch = min(index + batch, len(text_list))
        raw_text = raw_text_list[index: next_batch]  # list(list) [['a', 'good', 'laptop', ...]]
        text = text_list[index: next_batch]
        labels = labels_list[index: next_batch]
        tmp_labels = tmp_labels_list[index: next_batch]

        max_length = 100
        input_ids = []
        attention_mask = []
        subwords_flag = []

        for tmp_text in text:
            new_tmp_text = ['[CLS]'] + tmp_text + ['[SEP]']
            new_tmp = tokenizer.convert_tokens_to_ids(new_tmp_text)
            text_len = len(new_tmp) if len(new_tmp) <= max_length else max_length
            attention_mask.append([1 for _ in range(text_len)] + [0] * (max_length - text_len))
            if len(new_tmp) < max_length:
                new_tmp.extend([0] * (max_length - len(new_tmp)))
            if len(new_tmp) > max_length:
                new_tmp = new_tmp[:max_length]
            input_ids.append(new_tmp)

        with torch.no_grad():
            input_ids = torch.tensor(input_ids).cuda()
            attention_mask = torch.tensor(attention_mask).cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        loss, prediction_scores = outputs  # scores: logits before softmax
        prediction_scores = F.softmax(prediction_scores, dim=-1)  # batch, seq_len, vocab_size

        output_text = []
        for i, mask_t in enumerate(text):
            mask_t = mask_t[:max_length - 2]
            d_text = decode_text(tokenizer, tmp_labels[i], mask_t, prediction_scores[i], sub_aspects,
                                 pos_sub_opinions,neg_sub_opinions,neu_sub_opinions)
            assert len(d_text) == len(mask_t), print(d_text, mask_t)
            output_text.append(d_text)

        for index in range(len(raw_text)):
            labels[index] = labels[index][:max_length - 2]
            assert len(output_text[index]) == len(labels[index]), print(output_text[index], labels[index])
            output_examples.append(' '.join(output_text[index]) + '####' + ' '.join(labels[index]))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, source_domain + '_' + target_domain + '.txt'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(output_examples))


if __name__ == '__main__':
    set_seed(42)
    main()

