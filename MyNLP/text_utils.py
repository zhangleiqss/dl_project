import jieba
import re
import numpy as np

def get_term_list(sentence, word_seg=False):
    if word_seg:
        seg_list = jieba.cut(sentence.strip().lower())
        return " ".join(seg_list).split()
    else:
        return list(sentence)

def update_bow(sentence, id_term_map, word_seg=False, terminator=1):
    sen_id = []
    for s in get_term_list(sentence, word_seg):
        if s not in id_term_map:
            id_term_map[s] = len(id_term_map)+1
        sen_id.append(id_term_map[s])
    sen_id.append(terminator)
    return sen_id
    
def generate_prediction_array(raw_sentence, id_term_map, maxlen=120):
    sentence_id = update_bow(weibo_filter(raw_sentence.strip()), id_term_map)
    seq_len = len(sentence_id)
    document = np.pad(np.array(sentence_id),((0),(maxlen-seq_len)), mode='constant', constant_values=0)
    return document, seq_len

def generate_prediction_ndarray(raw_sentences, id_term_map, maxlen=120):
    documents = np.empty(shape=(len(raw_sentences),maxlen))
    seq_len = np.empty(shape=(len(raw_sentences)))
    for i, raw_sentence in enumerate(raw_sentences):
        document, seqlen = generate_prediction_array(raw_sentence, id_term_map, maxlen)
        documents[i] = document
        seq_len[i] = seqlen
    return documents, seq_len               

    
def weibo_filter(sentence):
    #sentence = sentence.decode('utf8')
    #reply
    sentence = re.sub(u'\u56de\u590d@([\u4e00-\u9fa5a-zA-Z0-9_-]+)(\u0020\u7684\u8d5e)?:',' ',sentence)
    #at
    sentence = re.sub(u'@[\u4e00-\u9fa5a-zA-Z0-9_-]+', ' ', sentence)
    #link
    sentence = re.sub('([hH]ttp[s]{0,1})://[a-zA-Z0-9\.\-]+\.([a-zA-Z]{2,4})(:\d+)?(/[a-zA-Z0-9\-~!@#$%^&*+?:_/=<>.\',;]*)?',' ',sentence) 
    #topic
    sentence = re.sub('#[^#]+#', '', sentence)
    return sentence
