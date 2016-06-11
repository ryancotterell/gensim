from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import Text8Corpus
import numpy as np
from scipy import linalg, mat, dot, stats
import logging

logging.basicConfig(level=logging.INFO)

def rho(vec1, vec2):
    return stats.stats.spearmanr(vec1, vec2)[0]

def read_sim(fname):
    scores = {}
    with open(fname, 'rb') as f:
        for line in f:
            _, _, eng1, eng2, score = line.strip().split("\t")
            scores[(eng1, eng2)] = float(score)
    return scores

def wordsim(scores, model):
    vec1, vec2 = [], []
    for (eng1, eng2), score in scores.items():
        if eng1 in model and eng2 in model:
            vec1.append(score)
            vec2.append(model.similarity(eng1, eng2))
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return rho(vec1, vec2)
    
def main(fname, fsim, fquestions, C):
    scores = read_sim(fsim)
    corpus = Text8Corpus(fname)
    model = Word2Vec(corpus, sg=1, hs=0, negative=5, bayes=1, samples=1, workers=4, min_count=20, C=C, iter=5)
    #model.accuracy(fquestions)
    #print np.linalg.norm(model["Russia"], 2)
    print wordsim(scores, model)
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--data', type=str, help='input text')
    p.add_argument('--sim', type=str, help='word similarity')
    p.add_argument('--questions', type=str, help='questions')
    p.add_argument('--C', type=float, help='C')
    args = p.parse_args()
    main(args.data, args.sim, args.questions, float(args.C))

