#!/usr/bin/env python
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer
import dill as pickle 
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return content


#class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    model = MLE(2)

    @classmethod
    def load_training_data(cls):
        
        
        return []

    @classmethod
    def load_test_data(cls, fname):
        #using base code
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        text = open("alice's_adventures.txt").read()
        #parses data and trains model
        data = list(text)

        data = [x.strip(' ') for x in data]
        data = [x.strip('\n') for x in data]
        data = [ele for ele in data if ele.strip()]

        data = list(pad_both_ends(data, n=2))

        model = MLE(2)

        train, vocab = padded_everygram_pipeline(1, data)

        model.fit(train, vocab)
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [generate_sent(model, 3, inp)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        #saves checkpoint
        with open(os.path.join('work', 'model.pkl'), 'wb') as fout:
            pickle.dump(model, fout)
        return[]

    @classmethod
    def load(cls, work_dir):
        # loads saved checkpoint
        with open(os.path.join('work', 'model.pkl'), 'rb') as fin:
            model = pickle.load(fin)
        return MyModel()


#args.test_data    args.work_dir  args.test_output


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        text = open('MonteCristo.txt', 'r', encoding='utf8', errors='ignore').read()


        data = list(text)

        data = [x.strip(' ') for x in data]
        data = [x.strip('\n') for x in data]
        data = [ele for ele in data if ele.strip()]

        data = list(pad_both_ends(data, n=2))

        model = MLE(3)

        train, vocab = padded_everygram_pipeline(1, data)

        model.fit(train, vocab)

        with open(os.path.join(args.work_dir, 'model.pkl'), 'wb') as fout:
            pickle.dump(model, fout)

        print('Training Finished')


    elif args.mode == 'test':
        with open(os.path.join(args.work_dir, 'model.pkl'), 'rb') as fin:
            model = pickle.load(fin)

        with open(args.test_data) as fin:
            line = fin.readline()
            cnt = 1
            with open(args.test_output, 'wt') as fout:
                while line:
                    string = ''.join(generate_sent(model, 3, line.strip()))
                    fout.write(string.replace(" ",""))
                    fout.write('\n')
                    line = fin.readline()
                    cnt += 1


        
        
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
