""" make a word2vec model
Hai Hu Oct 2019
"""

from gensim.test.utils import datapath
from gensim import utils
import glob, argparse, re
import gensim.models

# [Algo, Algos, algos] -> algo             d'algo, dalgo -> de algo
p7 = re.compile(r"(\b)(Algo|Algos|algos)(\b)")
p8 = re.compile(r"(\b)(d'algo|dalgo|D'algo|Dalgo)(\b)")

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, lower_case, prepro):
        self.lower_case = lower_case
        self.preprocess = lambda x : x
        if prepro == 'gensim':
            self.preprocess = utils.simple_preprocess

class ChronCorpus(MyCorpus):

    def __init__(self, lower_case, prepro):
        super().__init__(lower_case, prepro)

    def __iter__(self):
        #for fn in glob.glob("chronicles_clean/*txt"):
        fn = "chronicles_clean_all.txt.clean"
        for line in open(fn):
            # assume there's one document per line, tokens separated by whitespace
            if self.lower_case: yield self.preprocess(line.lower())
            else: yield self.preprocess(line)

class ChronCorpusNormalized(MyCorpus):

    def __init__(self, lower_case, prepro):
        super().__init__(lower_case, prepro)

    def __iter__(self):
        #for fn in glob.glob("chronicles_clean/*txt"):
        fn = "../chronicles_clean_all.txt.clean.normalized"
        for line in open(fn):
            # assume there's one document per line, tokens separated by whitespace
            if self.lower_case: yield self.preprocess(line.lower())
            else: yield self.preprocess(line)



class IMPACTCorpus(MyCorpus):

    def __init__(self, lower_case, prepro):
        super().__init__(lower_case, prepro)  

    def __iter__(self):              
        fh1 = open('IMPACT_GT_all_clean.txt')   # ('impact_GT.txt')
        fh2 = open('IMPACT_BVC_all_clean.txt')  # ('impact_BVC.txt')
        for fh in [fh1, fh2]:
            for line in fh:
                line = p7.sub(r'\1algo\3', line)  # algo
                line = p8.sub(r'\1de algo\3', line)  # algo
                if self.lower_case: yield self.preprocess(line.lower())
                else: yield self.preprocess(line)

class ModernCorpus(MyCorpus):

    def __init__(self, lower_case, prepro):
        super().__init__(lower_case, prepro)

    def __iter__(self):              
        pass

def main():
    # -------------------------------------
    # parse cmd arguments
    description = """
    Train embeddings. Author: Hai Hu
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-c', dest='corpus', choices=['chronicles', 'IMPACT', 'modern', 'chronicles.n'])
    parser.add_argument('-l', dest='lower_case', type=int, choices=[0, 1])
    parser.add_argument('-p', dest='prepro', choices=['0', 'gensim'])
    parser.add_argument('-s', dest='dim', type=int, choices=[50, 100, 200, 300])
    parser.add_argument('-w', dest='window_size', type=int, choices=[3,5,7])
    parser.add_argument('-m', dest='min_count', type=int, choices=[3,5,7,10,20])
    parser.add_argument('-g', dest='sg', type=int, choices=[0,1]) # 1 is skipgram
    parser.add_argument('-n', dest='negative', type=int, choices=[5,10,20])
    parser.add_argument('-e', dest='seed', type=int, choices=[1,2,3])
    parser.add_argument('-x', dest='expo', type=str)
    args = parser.parse_args()

    custom_name = "corpus-{}_lower-{}_prepro-{}_dim-{}_wsize-{}_mincnt-{}_sg-{}_neg-{}_expo-{}_seed-{}".format(
        args.corpus,
        args.lower_case,
        args.prepro,
        args.dim,
        args.window_size,
        args.min_count,
        args.sg,
        args.negative,
        args.expo,
        args.seed
    )

    outdir="models_new_10epoch/"

    # -------------------------------------
    # logging
    import logging
    logging.basicConfig(
        filename=outdir + custom_name + ".log",
        filemode='w',
        format='%(asctime)s : %(levelname)s : %(message)s', 
        level=logging.INFO
        )
    # -------------------------------------

    if args.corpus == "chronicles":
        sentences = ChronCorpus(args.lower_case, args.prepro)
    elif args.corpus == "IMPACT":
        sentences = IMPACTCorpus(args.lower_case, args.prepro)
    elif args.corpus == "chronicles.n":
        sentences = ChronCorpusNormalized(args.lower_case, args.prepro)
    elif args.corpus == "Modern":
        raise NotImplementedError()
        exit()

    if args.sg == 1:  # skip gram
        model = gensim.models.Word2Vec(
            sentences=sentences, 
            workers=1, 
            size=int(args.dim),
            window=int(args.window_size),
            min_count=int(args.min_count),
            sg=int(args.sg), 
            negative=int(args.negative),
            seed=int(args.seed),
            ns_exponent=float(args.expo),
            iter=10
            )
    else:  # CBOW
        model = gensim.models.Word2Vec(
            sentences=sentences, 
            workers=1, 
            size=int(args.dim),
            window=int(args.window_size),
            min_count=int(args.min_count),
            sg=int(args.sg), 
            seed=int(args.seed),
            iter=10
            )    

    # save
    model.save(outdir + custom_name + ".model")

if __name__ == "__main__":
    main()
