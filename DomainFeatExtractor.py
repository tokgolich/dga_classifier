#coding:utf-8
import math, sys
from collections import Counter,defaultdict
import tldextract #TLD
import numpy as np
from itertools import groupby
from sklearn.externals import joblib
#from publicsuffix import PublicSuffixList

#see if a domain is pronunceable using 2-letter Markov Chain
#https://github.com/rrenaud/Gibberish-Detector
import pickle
import gib_detect_train

hmm_prob_threshold = -120

def ave(array_):#sanity check for NaN
    if len(array_)>0:
        return array_.mean()
    else:
        return 0

def count_vowels(word):#how many a,e,i,o,u
    vowels=list('aeiou')
    return sum(vowels.count(i) for i in word.lower())

def count_digits(word):#how many digits
    digits=list('0123456789')
    return sum(digits.count(i) for i in word.lower())

def count_repeat_letter(word):#how many repeated letter
    count = Counter(i for i in word.lower() if i.isalpha()).most_common()
    cnt = 0
    for letter,ct in count:
        if ct>1:
            cnt+=1
    return cnt

def consecutive_digits(word):#how many consecutive digit
    cnt = 0
    digit_map = [int(i.isdigit()) for i in word]
    consecutive=[(k,len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i,j in consecutive if j>1 and i==1)
    return count_consecutive

def consecutive_consonant(word):#how many consecutive consonant
    cnt = 0
    #consonant = set(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'z'])
    consonant = set(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w','x', 'y', 'z'])
    digit_map = [int(i in consonant) for i in word]
    consecutive=[(k,len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i,j in consecutive if j>1 and i==1)
    return count_consecutive

def std(array_):#sanity check for NaN
    if len(array_)>0:
        return array_.std()
    else:
        return 0

def bigrams(words):
    wprev = None
    for w in words:
        if not wprev==None:
            yield (wprev, w)
        wprev = w

def trigrams(words):
    wprev1 = None
    wprev2 = None
    for w in words:
        if not (wprev1==None or wprev2==None):
            yield (wprev1,wprev2, w)
        wprev1 = wprev2
        wprev2 = w

def hmm_prob(domain, transitions):
    bigram = [''.join((i,j)) for i,j in bigrams(domain) if not i==None]
    prob = transitions[''][bigram[0]]
    for x in xrange(len(bigram)-1):
        next_step = transitions[bigram[x]][bigram[x+1]]
        prob*=next_step

    return prob

class DomainFeatExtractor:
    def __init__(self):
        model_data = pickle.load(open('data/gib_model.pki', 'rb'))
        self.model_mat = model_data['mat']
        self.threshold = model_data['thresh']

        private_tld_file = open('data/private_tld.txt', 'r')
        self.private_tld = set(f.strip() for f in private_tld_file)  # black list for private tld
        private_tld_file.close()

        n_gram_file = open('data/n_gram_rank_freq.txt', 'r')
        self.gram_rank_dict = dict()
        for i in n_gram_file:
            cat, gram, freq, rank = i.strip().split(',')
            self.gram_rank_dict[gram] = int(rank)
        n_gram_file.close()

        # load trans matrix for bigram markov model
        self.transitions = defaultdict(lambda: defaultdict(float))
        f_trans = open('data/trans_matrix.csv', 'r')
        for f in f_trans:
            key1, key2, value = f.rstrip().split('\t')  # key1 can be '' so rstrip() only
            value = float(value)
            self.transitions[key1][key2] = value
        f_trans.close()

        norm_para_file = open('data/features_norm_para.txt', 'r')
        self.norm_para = dict()
        for f in norm_para_file:
            feat, mean_, max_, min_ = f.strip().split('\t')
            self.norm_para[feat] = (float(mean_), float(max_), float(min_))

    def extractFeat(self, domain):
        strip_domain = domain.strip()
        ext = tldextract.extract(strip_domain)
        if len(ext.domain) > 4 and ext.domain[:4] == 'xn--':  # remove non-ascii domain
            return None

        main_domain = '$' + ext.domain + '$'  # add begin and end
        hmm_main_domain = '^' + domain.strip('.') + '$'  # ^ and $ of full domain name for HMM
        tld = ext.suffix
        has_private_tld = 0
        # check if it is a private tld
        if tld in self.private_tld:
            has_private_tld = 1
            tld_list = tld.split('.')  # quick hack: if private tld, use its last part of top TLD
            tld = tld_list[-1]
            main_domain = '$' + tld_list[-2] + '$'  # and overwrite the main domain
        #bigram = [''.join(i) for i in bigrams(main_domain)]  # extract the bigram
        #trigram = [''.join(i) for i in trigrams(main_domain)]  # extract the bigram
        f_len = float(len(main_domain))
        count = Counter(i for i in main_domain).most_common()  # unigram frequency
        entropy = -sum(j / f_len * (math.log(j / f_len)) for i, j in count)  # shannon entropy
        unigram_rank = np.array([self.gram_rank_dict[i] if i in self.gram_rank_dict else 0 for i in main_domain[1:-1]])
        bigram_rank = np.array([self.gram_rank_dict[''.join(i)] if ''.join(i) in self.gram_rank_dict else 0 for i in
                                bigrams(main_domain)])  # extract the bigram
        trigram_rank = np.array([self.gram_rank_dict[''.join(i)] if ''.join(i) in self.gram_rank_dict else 0 for i in
                                 trigrams(main_domain)])  # extract the bigram

        # linguistic feature: % of vowels, % of digits, % of repeated letter, % consecutive digits and % non-'aeiou'
        vowel_ratio = count_vowels(main_domain) / f_len
        digit_ratio = count_digits(main_domain) / f_len
        repeat_letter = count_repeat_letter(main_domain) / f_len
        consec_digit = consecutive_digits(main_domain) / f_len
        consec_consonant = consecutive_consonant(main_domain) / f_len

        # probability of staying in the markov transition matrix (trained by Alexa)
        hmm_prob_ = hmm_prob(hmm_main_domain, self.transitions)
        if hmm_prob_ < math.e ** hmm_prob_threshold:  # probability is too low to be non-DGA
            hmm_log_prob = -999.
        else:
            hmm_log_prob = math.log(hmm_prob_)

        #advanced linguistic feature: pronouncable domain
        gib_value = int(gib_detect_train.avg_transition_prob(main_domain.strip('$'), self.model_mat) > self.threshold)

        feat = dict()

        feat["domain"] = domain
        feat["entropy"] = entropy
        feat["len"] = f_len
        feat["norm_entropy"] = entropy/f_len
        feat["vowel_ratio"] = vowel_ratio

        feat["digit_ratio"] = digit_ratio
        feat["repeat_letter"] = repeat_letter
        feat["consec_digit"] = consec_digit
        feat["consec_consonant"] = consec_consonant
        feat["gib_value"] = gib_value
        feat["hmm_log"] = hmm_log_prob

        feat["uni_rank"] = ave(unigram_rank)
        feat["bi_rank"] = ave(bigram_rank)
        feat["tri_rank"] = ave(trigram_rank)
        feat["uni_std"] = std(unigram_rank)
        feat["bi_std"] = std(bigram_rank)
        feat["tri_std"] = std(trigram_rank)

        return feat

    def normFeat(self, feat):
        black_list = ["domain"]
        norm_feat = dict()

        for key in feat:
            if key in black_list:
                continue
            mean_, max_, min_ = self.norm_para[key]
            if feat[key] > max_:
                val = max_
            elif feat[key] < min_:
                val = min_
            else:
                val = feat[key]
            norm_feat[key] = (val-mean_)/(max_-min_)

        return norm_feat

    def testFeat(self, feat):
        feat_list = [feat["bi_rank"], feat["bi_std"], feat["consec_consonant"], feat["consec_digit"],
                     feat["digit_ratio"], feat["entropy"], feat["gib_value"], feat["hmm_log"],
                     feat["len"], feat["norm_entropy"], feat["repeat_letter"], feat["tri_rank"],
                     feat["tri_std"], feat["uni_rank"], feat["uni_std"], feat["vowel_ratio"]]

        return np.array(feat_list).reshape(1,-1)

if __name__ == '__main__':
    print "Test!"
    extractor = DomainFeatExtractor()
    feat = extractor.extractFeat(sys.argv[1])
    print feat
    norm_feat = extractor.normFeat(feat)
    print norm_feat
    test_feat = extractor.testFeat(norm_feat)
    print test_feat
    clf = joblib.load('data/dga_model.pkl')
    prob = clf.predict(test_feat)
    print prob
