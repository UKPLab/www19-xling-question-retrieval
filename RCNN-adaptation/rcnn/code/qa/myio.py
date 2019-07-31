# This file has been modified. Original available at https://github.com/taolei87/rcnn

# sys.setdefaultencoding("utf-8")
import gzip
import random
import string
import sys
from collections import Counter, defaultdict

import numpy as np
import theano
from nltk import flatten, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from nn import EmbeddingLayer


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()

# KRISHNKANT EDIT
def read_corpus(path, path_to_translations = None, path_to_translatable_ids = None, path_to_generated_questions=None):
    translations = {}
    translate_count = 0
    if path_to_translations is not None:
        # INITIALIZE TOKENIZER FOR TRANSLATIONS
        # import sys
        # sys.setdefaultencoding("utf-8")
        import spacy
        nlp = spacy.load('en')
        from spacy.lang.en import English
        tokenizer = English().Defaults.create_tokenizer(nlp)
        def fun_proc(t):
            t = tokenizer(u'{}'.format(t.decode("utf-8")))
            t = ' '.join([str(i) for i in t]).lower().strip()
            return t
        # STORE TRANSLATIONS IN A DICT
        fopen = gzip.open if path_to_translations.endswith(".gz") else open
        with fopen(path_to_translations) as f:
            for l in f:
                if len(l.strip())>0:
                    i, t, b = l.strip().split('\t')
                    i, t = i.strip(), t.strip()
                    translations[i] = fun_proc(t)

        if path_to_translatable_ids is not None:
            # READ TRANSLATABLE IDS AND POP ALL THE OTHER IDS FROM TRANSLATIONS
            translatables = []
            with open(path_to_translatable_ids, 'r') as fid:
                for l in fid:
                    if len(l.strip()) > 0:
                        i = l.strip().split()[0]
                        translatables.append(i)
                translations_keys = translations.keys()
                for i in translations_keys:
                    if i not in translatables:
                        translations.pop(i)

    raw_corpus = {}

    # AR edit.
    # We add all the generated titles as additional items with id <orig-id>_qgen
    # Later we imply a truth label for the pairs (<orig-id>, <orig-id>_qgen)
    if path_to_generated_questions is not None:
        with open(path_to_generated_questions) as f:
            for l in f:
                qid, dist, q = l.strip().split('\t')
                key = '{}_qgen'.format(qid)
                assert (key not in raw_corpus)
                raw_corpus[key] = q.strip().split(), []
        print('Read {} generated questions'.format(len(raw_corpus)))

    empty_cnt = 0
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                print(id)
                empty_cnt += 1
                continue
            if id in translations:
                translate_count += 1
                title = translations[id]
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[id] = (title, body)
    say("{} empty titles ignored.\n".format(empty_cnt))
    say("{} titles translated.\n".format(translate_count))
    return raw_corpus

# ORIGINAL
# def read_corpus(path):
#     empty_cnt = 0
#     raw_corpus = {}
#     fopen = gzip.open if path.endswith(".gz") else open
#     with fopen(path) as fin:
#         for line in fin:
#             id, title, body = line.split("\t")
#             if len(title) == 0:
#                 print(id)
#                 empty_cnt += 1
#                 continue
#             title = title.strip().split()
#             body = body.strip().split()
#             raw_corpus[id] = (title, body)
#     say("{} empty titles ignored.\n".format(empty_cnt))
#     return raw_corpus


def read_generated_questions(generated_questions_path):
    result = defaultdict(lambda: list())
    if generated_questions_path:
        for post_id, distance, generated_question_text in read_tsv(generated_questions_path):
            result[post_id].append(generated_question_text.lower().strip().split(' '))
    return result

def read_tsv(file_path, is_gzip=False):
    result = []

    if is_gzip:
        f = gzip.open(file_path, 'r')
    else:
        f = open(file_path, 'r')

    try:
        for line in f:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            line = line.rstrip()
            if line:
                result.append(line.split('\t'))
    finally:
        f.close()

    return result

def create_embedding_layer(raw_corpus, n_d, embs=None, \
        cut_off=2, unk="<unk>", padding="<padding>", fix_init_embs=True, generated_questions=None):

    cnt = Counter(w for id, pair in raw_corpus.iteritems() for x in pair for w in x)
    if generated_questions is not None:
        cnt.update(w for t in generated_questions.values() for w in t)

    cnt[unk] = cut_off + 1
    cnt[padding] = cut_off + 1
    embedding_layer = EmbeddingLayer(
            n_d = n_d,
            #vocab = (w for w,c in cnt.iteritems() if c > cut_off),
            vocab = [ unk, padding ],
            embs = embs,
            fix_init_embs = fix_init_embs
        )
    return embedding_layer

def create_idf_weights(corpus_path, embedding_layer):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)

    lst = [ ]
    fopen = gzip.open if corpus_path.endswith(".gz") else open
    with fopen(corpus_path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            lst.append(title)
            lst.append(body)
    vectorizer.fit_transform(lst)

    idfs = vectorizer.idf_
    avg_idf = sum(idfs)/(len(idfs)+0.0)/4.0
    weights = np.array([ avg_idf for i in xrange(embedding_layer.n_V) ],
                    dtype = theano.config.floatX)
    vocab_map = embedding_layer.vocab_map
    for word, idf_value in zip(vectorizer.get_feature_names(), idfs):
        id = vocab_map.get(word, -1)
        if id != -1:
            weights[id] = idf_value
    return theano.shared(weights, name="word_weights")

def map_corpus(raw_corpus, embedding_layer, max_len=100, generated_questions=None):
    ids_corpus = { }
    for id, pair in raw_corpus.iteritems():
        title_ids = embedding_layer.map_to_ids(pair[0], filter_oov=True, is_title = True)
        generated_questions_ids = []
        questions_weights = [1.0]
        if generated_questions is not None:
            generated_questions_ids = [embedding_layer.map_to_ids(gq, filter_oov=True, is_title=True) for gq in
                                   generated_questions[id]]
            questions_weights += [gq_score_novelty(qg, pair[0]) for qg in generated_questions[id]]
        body_ids = embedding_layer.map_to_ids(pair[1], filter_oov=True)[:max_len]
        item = (title_ids, body_ids, generated_questions_ids, questions_weights)
        #if len(item[0]) == 0:
        #    say("empty title after mapping to IDs. Doc No.{}\n".format(id))
        #    continue
        ids_corpus[id] = item
    return ids_corpus

def read_annotations(path, K_neg=20, prune_pos_cnt=10, training_data_percent = 100):
    lst = [ ]
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
            if K_neg != -1:
                random.shuffle(neg)
                neg = neg[:K_neg]
            s = set()
            qids = [ ]
            qlabels = [ ]
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            lst.append((pid, qids, qlabels))
    LIMIT = int (len(lst) * training_data_percent * 0.01)
    lst = random.sample(lst,LIMIT)
    # lst = lst[:LIMIT]
    return lst

def create_batches(ids_corpus, data, batch_size, padding_id, perm=None, pad_left=True, include_generated_questions=False):
    # adaptation AR
    # extend the data by generated questions
    if include_generated_questions:
        orig_questions_ids = [d[0] for d in data]
        pairs = [(k, k[:-5]) for k in ids_corpus.keys() if k.endswith('_qgen') and k[:-5] in orig_questions_ids]
        # pairs of generated-question-id, source-id. These are now duplicates / paraphrases
        qgen_data = []
        for generated_question, question in pairs:
            qgen_data_point = (question, [generated_question] + np.random.choice(orig_questions_ids, size=20).tolist(), [1] + [0] * 20)
            qgen_data.append(qgen_data_point)
        print('Extended the batches with {} generated questions for training (orig: {})'.format(len(qgen_data), len(data)))
        data = data + qgen_data

    N = len(data)

    if perm is None:
        perm = range(N)
        random.seed(8)
        random.shuffle(perm)

    cnt = 0
    pid2id = {}
    titles = [ ]
    bodies = [ ]
    triples = [ ]
    batches = [ ]
    for u in xrange(N):
        i = perm[u]
        pid, qids, qlabels = data[i]
        if pid not in ids_corpus: continue
        cnt += 1
        for id in [pid] + qids:
            if id not in pid2id:
                if id not in ids_corpus: continue
                pid2id[id] = len(titles)
                t, b, gq, weights = ids_corpus[id]
                titles.append(t)
                bodies.append(b)
        pid = pid2id[pid]
        pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ]
        neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ]
        triples += [ [pid,x]+neg for x in pos ]

        if cnt == batch_size or u == N-1:
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
            triples = create_hinge_batch(triples)
            batches.append((titles, bodies, triples))
            titles = [ ]
            bodies = [ ]
            triples = [ ]
            pid2id = {}
            cnt = 0
    return batches

def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    lst = [ ]
    for pid, qids, qlabels in data:
        titles = [ ]
        bodies = [ ]
        generated_questions = [ ]
        questions_weights = [ ]
        for id in [pid]+qids:
            t, b, gq, weights = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
            generated_questions.append(gq)
            questions_weights.append(weights)
        # titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        titles, bodies, questions_count = create_multi_batch(titles, bodies, padding_id, pad_left, generated_questions)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32"), questions_weights))
    return lst

def create_one_batch(titles, bodies, padding_id, pad_left):
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    if pad_left:
        titles = np.column_stack([ np.pad(x,(max_title_len-len(x),0),'constant',
                                          constant_values=padding_id) for x in titles])
        bodies = np.column_stack([ np.pad(x,(max_body_len-len(x),0),'constant',
                                          constant_values=padding_id) for x in bodies])
    else:
        titles = np.column_stack([ np.pad(x,(0,max_title_len-len(x)),'constant',
                                          constant_values=padding_id) for x in titles])
        bodies = np.column_stack([ np.pad(x,(0,max_body_len-len(x)),'constant',
                                          constant_values=padding_id) for x in bodies])

    return titles, bodies


def create_multi_batch(titles, bodies, padding_id, pad_left, generated_questions):
    questions_count = [len(gq) + 1 for gq in generated_questions]

    titles = flatten(zip(titles, generated_questions))
    bodies = flatten([[b] * questions_count[i] for (i, b) in enumerate(bodies)])

    assert len(titles) == len(bodies)
    assert sum(questions_count) == len(titles)

    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    if pad_left:
        titles = np.column_stack([ np.pad(x,(max_title_len-len(x),0),'constant',
                                          constant_values=padding_id) for x in titles])
        bodies = np.column_stack([ np.pad(x,(max_body_len-len(x),0),'constant',
                                          constant_values=padding_id) for x in bodies])
    else:
        titles = np.column_stack([ np.pad(x,(0,max_title_len-len(x)),'constant',
                                          constant_values=padding_id) for x in titles])
        bodies = np.column_stack([ np.pad(x,(0,max_body_len-len(x)),'constant',
                                          constant_values=padding_id) for x in bodies])

    return titles, bodies, questions_count


stemmer = PorterStemmer()
sw = set(stopwords.words('english') + list(string.punctuation))


def gq_score_novelty(gq, q):
    def s(w):
        try:
            return stemmer.stem(w)
        except:
            return w

    gqt = set([s(w) for w in gq if w not in sw])
    qt = set([s(w) for w in q if w not in sw])

    if len(gqt) == 0:
        return 0.0
    else:
        g_gen_unique_toks = len(gqt - qt)
        return float(g_gen_unique_toks) / len(gqt)

def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)
    triples = np.vstack([ np.pad(x,(0,max_len-len(x)),'edge')
                        for x in triples ]).astype('int32')
    return triples
