"""
This script has several modes.
-b: build a model:
    Input: corpus
    Output: a model, which includes:
            description-title co-occurrence frequency dict
            title frequency dict
            description frequency dict
-c: cleanup title names
    Input: a model
    Output: a cleaned model
-m: merge several models
    Input: a directory containing several models
    Output: a merged model
-p: prune a model
    Input: a model
    Output: a pruned model
-fl: interactive mode
    Input: Type in a query
    Output: Print top k related titles
-e: evaluation mode
    Input: a model
           a evaluation data
    Output: print evaluation result

Written by Dongxu
Last edit time: 2017/3/15
"""

import argparse
import os, subprocess
import re
import pickle
import copy
import math
import sys
import trie, actrie
import collections
import heapq
import time

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-l","--load",help="loads previous model instead of building a new one and \
                                       also transforms the co-occurrence dict. \
                                       This mode lets you check both title ranking and desc \
                                       ranking", action="store_true")
group.add_argument("-fl","--fastload",help="loads previous model instead of building a new one, \
                                            this mode simulates the server's action.", \
                                      action="store_true")
group.add_argument("-b","--build",help="builds a new model.", action="store_true")
group.add_argument("-m","--merge",help="merges the model files",action="store_true")
group.add_argument("-p","--prune",help="prunes the model file",action="store_true")
group.add_argument("-e","--evaluate",help="evaluates a model with a ranking strategy", \
                                     action="store_true")
group.add_argument("-c","--clean",help="clean title names.",action="store_true")
group.add_argument("-q","--query",help="input a query description and return a full ranking \
                                        list.",action="store_true")
parser.add_argument("--model_type",help="This parameter defines the type of the new model. \
                    The type is distinguished by its window type [paragraph|window|weightedwindow]",\
                    default="weightedwindow")
parser.add_argument("--input",help="the path of the input directory that contains the corpus\
        to cope with.")
parser.add_argument("--descriptor",help="the path of the file that contains\
        descriptors",default="data/desc.txt")
parser.add_argument("--title",help="the path of the file that contains film titles. \
                                    If this parameter is not given, script will match titles \
                                    with brackets only.",default="")
parser.add_argument("--model",help="the path of the model to save or load",\
                              required=True)
parser.add_argument("--mergedir",help="the path of directory where model files were saved and \
                                       will be merged.",default="/home/dongxu/descriptor/model")
parser.add_argument("--window_size",help="window size for left and right windows",type=int, \
                                    default=10)
parser.add_argument("--smooth_factor",help="smoothing factor for window weights",type=int, \
                                      default=2)
parser.add_argument("--pruned_model", help="the path of output pruned model file", default="prune")
parser.add_argument("--cleaned_model", help="the path of output cleaned model file",default="prune")
parser.add_argument("--prune_threshold", help="the path of output pruned model file", \
                                         type=float, default=1.0)
parser.add_argument("--testset",help="testset for evaluation",default="")
parser.add_argument("--topk", help="sets top k value for ranking", type=int, default=50)
parser.add_argument("--string_match", help="string match method. Maxmatch use trie. \
                                            Allmatch use Aho-Corasick automation. [max|all]", \
                                      default="max")
parser.add_argument("--query_string",help="input description string",default="电影")
parser.add_argument("--disjunction",help="scoring 'or' logic for query mode.",action="store_true")

def load_model_decorator(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        model_file = args[1]
        string_match = args[2]
        model_dict = pickle.load(open(model_file,"rb"))
        self._title_freq_dict = collections.Counter(model_dict["title_freq_dict"])
        self._desc_freq_dict = collections.Counter(model_dict["desc_freq_dict"])
        for desc in model_dict["co_freq_dict"]:
            self._co_freq_dict[desc].update(model_dict["co_freq_dict"][desc])

        self.string_match = string_match
        if self.string_match == 'max':
            self.desc_trie = trie.Trie(self._co_freq_dict)
        elif self.string_match == 'all':
            self.desc_actrie = actrie.load(self._co_freq_dict)

        self.max_length_desc = 0
        for desc in self._co_freq_dict:
            if len(desc) > self.max_length_desc:
                self.max_length_desc = len(desc)

        self.word_popularity_dict = {}
        for line in open('data/clusters300k.txt'):
            l = line.strip().split('\t')
            title = "".join(l[0].split())
            prob = float(l[1])
            self.word_popularity_dict[title] = prob
        self.smallest_prob = -15.7357
        self.word_trie = trie.Trie(self.word_popularity_dict)

        func(*args, **kwargs)
    return wrapper

class Descriptor:
    def __init__(self):
        self._co_freq_dict = collections.defaultdict(collections.Counter)
        self._title_freq_dict = collections.Counter()
        self._desc_freq_dict = collections.Counter()
        self.pattern = re.compile("《(.*?)》") #predefined pattern
        self.noise_pattern = re.compile("&(.*?);") 
        self.noise_string = set(("<a>", "</a>", "&quot"))
        self.nonenglish_space_pattern = re.compile("(?=([^a-zA-Z \t][ \t]+?[^a-zA-Z \t]))")
        self.noise_punc = set((".", ",", "?", ":", "-", "(", ")", "。", "，", 
                                "！", "、", "：", "·", "（", "）", "《", "》", 
                                "〉", "〈", "…", "_"))
        self.number_map = {"1":"一", "2":"二", "3":"三", "4":"四", "5":"五",
                           "6":"六", "7":"七", "8":"八", "9":"九", "0":"零",
                           "Ⅰ":"一", "Ⅱ":"二", "Ⅲ":"三", "Ⅳ":"四", "X":"十"}
        self.scenarios = {'server': (self.load_model, self.rank_titles), 
                          'query':(self.load_model_normalize, self.rank_titles_full_rank)}
   
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    @property
    def co_freq_dict(self):
        return self._co_freq_dict

    @property
    def title_freq_dict(self):
        return self._title_freq_dict

    @property
    def desc_freq_dict(self):
        return self._desc_freq_dict

    def list_dir(self, rootdir): 
        """List all file paths inside a given directory (recursively)"""
        filelist = []
        for lists in os.listdir(rootdir): 
            path = os.path.join(rootdir, lists)
            if not os.path.isdir(path):
                filelist.append(path)
            else:
                filelist.extend(self.list_dir(path))
        return filelist

    def load_desc(self, desc_file, string_match = 'max'):
        """Load description list into memory. This list is required."""
        descriptors = open(desc_file).read().split('\n')
        max_length_desc = 0
        for i in range(len(descriptors)):
            descriptors[i] = "".join(descriptors[i].split())
            if len(descriptors[i]) > max_length_desc:
                max_length_desc = len(descriptors[i])
        self.max_length_desc = max_length_desc
        desc_set = set(descriptors)
        self.string_match = string_match
        if self.string_match == 'max':
            self.desc_trie = trie.Trie(desc_set)
        elif self.string_match == 'all':
            self.desc_actrie = actrie.load(desc_set)
            
        
        print(str(len(descriptors)) + " descriptors loaded.")
        self.desc_set = desc_set

    def load_title(self, title_file):
        """If we provide a title list, then load it into memory"""
        titles=open(title_file).read().split('\n')[:-1]
        for i in range(len(titles)):
            titles[i] = titles[i].split()[0]
        title_set = set(titles)
        print(str(len(titles)) + " title names loaded.")
        self.title_set = title_set

    def match_pattern_in_line(self, line):
        """match the predefined pattern in a line. 
           Return a list of tuples containing 
           the start and end+1 index of each matched string."""
        iterator = self.pattern.finditer(line)
        pattern_positions = [match.span() for match in iterator]
        return pattern_positions

    def count_freq(self):
        '''If you want to inherit this class, 
           you need to complete this function'''
        raise NotImplementedError
    
    def save_model(self, model_file):
        """Save counted frequency into a model_file using pickle"""
        model_dict = {"co_freq_dict":self._co_freq_dict,
                      "title_freq_dict":self._title_freq_dict,
                      "desc_freq_dict":self._desc_freq_dict}
        pickle.dump(model_dict, open(model_file + ".p","wb"), protocol=2)

    def merge_model(self, merge_dir, output_model_file):
        """merge different models under a directory and save the merged model."""
        print("start merging...")
        model_files = self.list_dir(merge_dir)
        print(str(len(model_files)) + " model files founded")
        print("start loading models")
        for model_file in model_files:
            model_dict = pickle.load(open(model_file,"rb"))
            for desc in model_dict["co_freq_dict"]:
                self._co_freq_dict[desc].update(model_dict["co_freq_dict"][desc])
            self._title_freq_dict.update(model_dict["title_freq_dict"])
            self._desc_freq_dict.update(model_dict["desc_freq_dict"])
            print(model_file + " merged")
            sys.stdout.flush()
        self.save_model(output_model_file)
        print("model merged")

    @load_model_decorator
    def load_model(self, model_file, string_match = 'max'):
        """Load model into memory,
           count probability of (description, "《》" ) given a title"""
        self.co_probability = {} #log probability of description given a brackted title
        self.title_probability = {} #log probability of title as a string
        self.prescores = {} #save time by doing some computation in model loading step.
        for title in self._title_freq_dict:
            title_prob = 0
            history_position = 0
            while(history_position < len(title)):
                offset = self.word_trie.maxmatch(title[history_position:])
                if offset == 0:
                    title_prob = self.smallest_prob
                    break
                else:
                    title_prob += self.word_popularity_dict[title[
                                         history_position:history_position + offset]]
                history_position += offset
            self.title_probability[title] = max(title_prob, self.smallest_prob)

        self.prob_not_appear = math.log(1.0/max([freq for 
                                   title, freq in self._title_freq_dict.items()]))
        for desc, titles in self._co_freq_dict.items():
            self.co_probability[desc] = {}
            self.prescores[desc] = {}
            for title in titles:
                self.co_probability[desc][title] = \
                math.log((self._co_freq_dict[desc][title] + 1)\
                / (self._title_freq_dict[title] + 1))
                self.prescores[desc][title] = math.log(self._title_freq_dict[title]) \
                                            - self.title_probability[title]

        for title, freq in self._title_freq_dict.items():
            self._title_freq_dict[title] = math.log(freq)
        
    @load_model_decorator
    def load_model_normalize(self, model_file, string_match = 'max'):
        """Load model into memory,
           normalize numbers in titles into chinese,
           add  normalized_title into co_freq_dict and delete original title 
           if they are different.
           count probability of (description, "《》" ) given a title"""
        self.normalize2clean = collections.defaultdict(list)
        self.clean2normalize = {}
        for title, freq in list(self._title_freq_dict.items()):
            normalized_title = self.replace_number(title)
            #print(title + '\t'+normalized_title)
            self.clean2normalize[title] = normalized_title
            self.normalize2clean[normalized_title].append(title)
            if normalized_title != title:
                self.clean2normalize[normalized_title] = normalized_title
                self._title_freq_dict[normalized_title] += freq
                del self._title_freq_dict[title]
        
        for desc, titles in list(self._co_freq_dict.items()):
            for title, freq in list(titles.items()):
                normalized_title = self.clean2normalize[title]
                if normalized_title != title:
                    self._co_freq_dict[desc][normalized_title] += freq
                    del self._co_freq_dict[desc][title]

        self.co_probability = {}
        self.title_probability = {}
        for normalized_title in self._title_freq_dict:
            title_prob = 0
            history_position = 0
            while(history_position < len(normalized_title)):
                offset = self.word_trie.maxmatch(normalized_title[history_position:])
                if offset == 0:
                    title_prob = self.smallest_prob
                    break
                else:
                    title_prob += self.word_popularity_dict[
                                         normalized_title[
                                         history_position:history_position + offset]]
                history_position += offset
            self.title_probability[normalized_title] = max(title_prob, self.smallest_prob)

        for desc, titles in self._co_freq_dict.items():
            self.co_probability[desc] = {}
            for normalized_title, freq in titles.items():
                self.co_probability[desc][normalized_title] = \
                    (self._co_freq_dict[desc][normalized_title] + 1)/\
                    (self.title_freq_dict[normalized_title] + 1)
                clean_titles = set(self.normalize2clean[normalized_title])
                if normalized_title in clean_titles:
                    clean_titles.remove(normalized_title)
                for clean_title in clean_titles:
                    self.co_probability[desc][clean_title] = \
                        self.co_probability[desc][normalized_title]
                    
        self.prob_not_appear = 1.0/max([freq for title, freq in self._title_freq_dict.items()])

    def load_testset(self, testset):
        """Load data for evaluation, where there are some descriptions 
           and their corresponding movie titles from Douban"""
        self.test_dict = pickle.load(open(testset, "rb"))
        for i, (desc, titles) in enumerate(self.test_dict.items()):
            self.test_dict[desc] = set(titles)

    def match_desc_max(self, string):
        """Forward maximum match descriptions in a string
           and return a list of matched descriptions.
           Use this function in query analysis."""
        ngram_descs = []
        string_length = len(string)
        current_index = 0
        while(current_index < string_length):
            add_index = self.desc_trie.maxmatch(string[current_index:])
            if add_index == 0:
                current_index += 1
            else:
                ngram_descs.append(string[current_index:current_index + add_index])
                current_index += add_index 
        return ngram_descs

    def match_desc_all(self, string):
        """Match all descriptions in a string
           and return a list of matched descriptions.
           Use this function in query analysis."""
        ngram_descs = []
        node = self.desc_actrie
        for ch in string:
            node = node.move(ch)
            if node is None:
                node = self.desc_actrie
            else:
                ngram_descs.extend([match_string for match_string in 
                                                 node.generate_all_suffix_nodes_values()])
        return ngram_descs

    def heap_sort_descent(self, dist_list, topk):
        return heapq.nlargest(topk, dist_list, key = lambda x:x[1])

    def rank_titles(self, ngram_descs, topk):
        """Given a list of descriptions, return top k most related titles 
           (for example movie names).
           Scoring method is the sum of title scores over different descriptions. 
           title score = log(co_freq/sqrt(title_freq)). 
           If co_freq = 0, then title score = log(1/sqrt(max_title_freq)) """
        if len(ngram_descs) == 0:
            print("描述词未出现")
            sys.stdout.flush()
            ngram_descs = ["电影"]
        time1 = time.perf_counter()

        #rank over all candidate title names.
        ngram_descs_ = [desc for desc in ngram_descs if desc in self.prescores]
        title_scores = {}
        for desc in ngram_descs_:
            title_scores.update(self.prescores[desc])
        for desc in ngram_descs_:
            prob_desc = self.co_probability[desc]
            for title in prob_desc:
                title_scores[title] += prob_desc[title] - self.prob_not_appear
        first_rank_results = dict([(k, v) for k, v in \
                                 self.heap_sort_descent(title_scores.items(), topk)])

        #rerank over topk title names.
        first_rank_results_items = list(first_rank_results.items())
        for i in range(len(first_rank_results_items)):
            title = first_rank_results_items[i][0]
            score = first_rank_results_items[i][1]
            first_rank_results_items[i] = (title, [])
            for desc in ngram_descs_:
                #first_rank_results_items[i][1].append(\
                #    self.co_probability[desc].get(title, self.prob_not_appear))
                first_rank_results_items[i][1].append(\
                    self.co_probability[desc].get(title, -10.0))
            #first_rank_results_items[i][1].extend([self._title_freq_dict[title],
            #                                -self.title_probability[title], score])
            if self._title_freq_dict[title] > 10.0:
                title_freq = 10
            else:
                title_freq = self._title_freq_dict[title]
            first_rank_results_items[i][1].extend([title_freq, 
                -self.title_probability[title], score])
        first_rank_results = dict(first_rank_results_items)
        reranking_results = dict([(title, [0]) for title in first_rank_results])
        for desc_index in range(len(ngram_descs_) + 1):
            sorted_first_rank_results_items = sorted(first_rank_results_items, \
                                                     key = lambda x:x[1][desc_index])
            history_index = 0 # number of titles that have smaller scores than current title
            previous_score = sorted_first_rank_results_items[0][1][desc_index] - 1
            title_index = 0
            while(title_index < len(sorted_first_rank_results_items)):
                current_info = sorted_first_rank_results_items[title_index]
                if current_info[1][desc_index] > previous_score:
                    #previous_prob = math.log((history_index + 1 + title_index)/2/topk)
                    previous_prob = math.log((history_index + 1)/topk)
                    for k in range(history_index, title_index):
                        reranking_results[sorted_first_rank_results_items[k][0]][0] += previous_prob
                        reranking_results[sorted_first_rank_results_items[k][0]].append(previous_prob)
                    history_index = title_index
                    previous_score = current_info[1][desc_index]
                title_index += 1
            #previous_prob = math.log((history_index + 1 + title_index)/2/topk)
            previous_prob = math.log((history_index + 1)/topk)
            for k in range(history_index, title_index):
                reranking_results[sorted_first_rank_results_items[k][0]][0] += previous_prob
                reranking_results[sorted_first_rank_results_items[k][0]].append(previous_prob)
        ranking_result = [(title, reranking_score, first_rank_results[title]) for title, reranking_score in sorted(\
                              reranking_results.items(), key = lambda x:x[1][0], reverse = True)]
        time2 = time.perf_counter()
        #print('response time:' + str(time2-time1)[:6] + ' s')
        return ranking_result 
            

    def rank_titles_full_rank(self, ngram_descs, or_logic = False):
        if len(ngram_descs) == 0:
            print("描述词未出现")
            sys.stdout.flush()
        else:
            title_scores = dict([(t, []) for t in set([title for desc in ngram_descs
                                                             if desc in self.co_probability
                                                             for title in self.co_probability[desc]])])
            ngram_descs_ = [desc for desc in ngram_descs if desc in self.co_probability]
            if or_logic == False:
                for title in title_scores:
                    for desc in ngram_descs_:
                        if title in self.co_probability[desc]:
                            title_scores[title].append(self.co_probability[desc][title])
                        else:
                            title_scores[title].append(self.prob_not_appear)
                    title_freq = self._title_freq_dict[self.clean2normalize[title]]
                    log_title_freq = math.log(title_freq)
                    title_scores[title].extend([log_title_freq,
                                        -self.title_probability[self.clean2normalize[title]]])
                    total_score = sum([math.log(score) for score in title_scores[title][:-2]] \
                                      + title_scores[title][-2:])
                    title_scores[title].append(total_score)
            else:
                for title in title_scores:
                    for desc in ngram_descs_:
                        if title in self.co_probability[desc]:
                            title_scores[title].append(self.co_probability[desc][title])
                        else:
                            title_scores[title].append(self.prob_not_appear)
                    title_freq = self._title_freq_dict[self.clean2normalize[title]]
                    log_title_freq = math.log(title_freq)
                    title_scores[title].extend([log_title_freq,  
                                        -self.title_probability[self.clean2normalize[title]]])

                    total_score = sum([math.log(sum(title_scores[title][:-2]))] \
                                      + title_scores[title][-2:])
                    title_scores[title].append(total_score)
            for title, values in sorted(title_scores.items(),
                                key=lambda x:x[1][-1], reverse=True):
                print(title + '\t' + str(values[-1]))

    def prune(self, model_file, prune_threshold = 1.0):
        """load model and prune it with a threshold"""
        model_dict = pickle.load(open(model_file,"rb"))
        self._title_freq_dict = model_dict["title_freq_dict"]
        self._desc_freq_dict = model_dict["desc_freq_dict"]
        self._co_freq_dict = model_dict["co_freq_dict"]

        title_after_prune = {}
        for i, (desc, titles) in list(enumerate(self._co_freq_dict.items())):
            for title, freq in list(titles.items()):
                if freq <= prune_threshold:
                    del self._co_freq_dict[desc][title]
                elif title not in title_after_prune:
                    title_after_prune[title]=""
            if len(self._co_freq_dict[desc]) == 0:
                del self._co_freq_dict[desc]
        
        for i, (title, freq) in list(enumerate(self._title_freq_dict.items())):
            if title not in title_after_prune:
                del self._title_freq_dict[title]

        for i, (desc, freq) in list(enumerate(self._desc_freq_dict.items())):
            if desc not in self._co_freq_dict:
                del self._desc_freq_dict[desc]

    def remove_nonenglish_space(self, string):
        string = string.strip()
        new_string = ""
        history = 0
        for match in self.nonenglish_space_pattern.finditer(string):
            start = match.start() + 1
            end = start + len(match.group(1)) - 2
            new_string += string[history:start]
            history = end
        new_string += string[history:]
        return new_string

    def remove_noise_pattern(self, string):
        string = string.strip()
        new_string = ""
        history = 0
        for match in self.noise_pattern.finditer(string):
            start, end = match.span()
            new_string += string[history:start]
            history = end
        new_string += string[history:]
        return new_string

    def remove_noise_string(self, string):
        string = string.strip()
        for s in self.noise_string:
            string = string.replace(s, "")
        return string

    def remove_noise_punc(self, string):
        string = string.strip()
        return "".join([ch for ch in string if ch not in self.noise_punc])

    def replace_number(self, string):
        new_string = ""
        for ch in string:
            if ch in self.number_map:
                new_string += self.number_map[ch]
            else:
                new_string += ch
        return new_string

    def cleanup_title(self, model_file):
        """load model and clean title names."""
        model_dict = pickle.load(open(model_file, "rb"))
        self._title_freq_dict = model_dict["title_freq_dict"]
        self._desc_freq_dict = model_dict["desc_freq_dict"]
        self._co_freq_dict = model_dict["co_freq_dict"]

        raw2clean = {}
        for i, (title, freq) in list(enumerate(self._title_freq_dict.items())):
            if title not in self._title_freq_dict:
                continue
            clean_title = self.remove_nonenglish_space(
                           self.remove_noise_punc(
                           self.remove_noise_string(
                           self.remove_noise_pattern(title)))).lower()
            raw2clean[title] = clean_title
            if clean_title != title:
                self._title_freq_dict[clean_title] += freq
                del self._title_freq_dict[title]
            if len(clean_title) > 30 or len(clean_title) == 0:
                del self._title_freq_dict[clean_title]
                
        for i, (desc, titles) in list(enumerate(self._co_freq_dict.items())):
            for j, (title, freq) in list(enumerate(titles.items())):
                if title not in self._co_freq_dict[desc]:
                    continue
                if title in raw2clean:
                    clean_title = raw2clean[title]
                else:
                    clean_title = self.remove_nonenglish_space(
                                   self.remove_noise_punc(
                                   self.remove_noise_string(
                                   self.remove_noise_pattern(title)))).lower()
                    
                if clean_title != title:
                    self._co_freq_dict[desc][clean_title] += freq
                    del self._co_freq_dict[desc][title]

                if len(clean_title) > 30 or len(clean_title) == 0:
                    del self._co_freq_dict[desc][clean_title]
                if len(self._co_freq_dict[desc]) == 0:
                    del self._co_freq_dict[desc]
        for i, (desc, freq) in list(enumerate(self._desc_freq_dict.items())):
            if desc not in self._co_freq_dict:
                del self._desc_freq_dict[desc]
    
    def evaluate(self, model_file, method = 'server', topk = 10):
        load_model = self.scenarios[method][0]
        ranking = self.scenarios[method][1]
        load_model(model_file, "max")
        print("data loaded")
        average_good_proportion = 0
        for desc, titles in self.test_dict.items():
            if desc not in self._co_freq_dict:
                continue
            result_titles = ranking([desc], topk * 5)[:topk]
            for title, _, _ in result_titles:
                if title in titles:
                    average_good_proportion += 1
                    break
                
        average_good_proportion /= float(len(self.test_dict))
        print("average top" +str(topk) + " hit rate= " + str(average_good_proportion))

class DescriptorParagraph(Descriptor):

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def _descriptor_allmatch(self, string):
        """when counting frequency, match all descriptions in a string
           and return matched descriptions in a list"""
        ngram_descs = []
        node = self.desc_actrie
        for ch in string:
            node = node.move(ch)
            if node is None:
                node = self.desc_actrie
            else:
                ngram_descs.extend([match_data for match_data in 
                                               node.generate_all_suffix_nodes_values()])
        return ngram_descs

    def _descriptor_maxmatch(self, string):
        """When counting frequency, match descriptions in a string
           using forward maximum match method."""
        ngram_descs = []
        string_length = len(string)
        current_index = 0
        while(current_index < string_length):
            add_index = self.desc_trie.maxmatch(string[current_index:])
            if add_index == 0:
                current_index += 1
            else:
                ngram_descs.append(string[current_index: current_index + add_index])
                current_index += add_index
        return ngram_descs 

    def count_freq(self, input_file, given_title = False):
        """count frequency of co-occurred (line-wise) title-description pairs, 
           frequency of titles and frequency of descriptions given a input file"""
        if self.string_match == 'max':
            descriptor_match = self._descriptor_maxmatch
        elif self.string_match == 'all':
            descriptor_match = self._descriptor_allmatch

        if input_file[-4:] == '.bz2':
            lines = subprocess.check_output(["bunzip2","-c",input_file]).\
                    decode('utf-8','replace').split('\n')
        else:
            lines = open(input_file, encoding='utf-8', errors='replace')
            
        for line_ in lines:
            line = line_.strip()
            if line[:7] == '<docno>' or line[:5] == '<url>':
                continue
            pattern_positions = self.match_pattern_in_line(line)
            
            matched_titles = []
            matched_descriptors = []
            history = 0
            for start, end in pattern_positions:
                title = line[start + 1:end - 1]
                if start < end - 2:
                    if (given_title == True and title in self.title_set) \
                        or given_title == False:
                        matched_titles.append(title)
                        matched_descriptors.extend(descriptor_match(line[history:start]))
                    history = end
            matched_descriptors.extend(descriptor_match(line[history:]))
            matched_titles = list(set(matched_titles))
            matched_descriptors = list(set(matched_descriptors))
            for title in matched_titles:
                self._title_freq_dict[title] += 1
                for desc in matched_descriptors:
                    self._co_freq_dict[desc][title] += 1
            for desc in matched_descriptors:
                self._desc_freq_dict[desc] += 1


class DescriptorWindow(Descriptor):

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def _descriptor_allmatch(self, string, start, end):
        """when counting frequency, match all descriptions in a string[start:end]
           and add the start and end index of matched descriptions into 
           self.index_desc_start and self.index_desc_end seperately."""
        node = self.desc_actrie
        current_index = start
        for ch in string[start:end]:
            node = node.move(ch)
            if node is None:
                node = self.desc_actrie
            else:
                for match_data in node.generate_all_suffix_nodes_values():
                    self.index_desc_start[current_index - len(match_data) + 1].append(match_data)
                    self.index_desc_end[current_index].append(match_data)
            current_index += 1

    def _descriptor_maxmatch(self, string, start, end):
        """when counting frequency, match descriptions in a string[start:end]
           using maximum match method and add the start and end index of 
           matched descriptions into self.index_desc_start and self.index_desc_end
           seperately."""
        current_index = start
        while(current_index < end):
            add_index = self.desc_trie.maxmatch(string[current_index:end])
            if add_index == 0:
                current_index += 1
            else:
                self.index_desc_start[current_index].\
                     append(string[current_index: current_index + add_index])
                self.index_desc_end[current_index + add_index - 1].\
                     append(string[current_index: current_index + add_index])
                current_index += add_index

    def set_window_weight(self, window_size):
        """In this class, we set window weights equally."""
        self.weight = []
        self.window_size = window_size
        for i in range(window_size):
            self.weight.append(1.0)

    def count_freq(self, input_file, given_title = False):
        """count frequency of co-occurred (character-window-wise) title-description pairs, 
           frequency of titles and frequency of descriptions given a input file"""
        if self.string_match == 'max':
            descriptor_match = self._descriptor_maxmatch
        elif self.string_match == 'all':
            descriptor_match = self._descriptor_allmatch

        if input_file[-4:] == '.bz2':
            lines = subprocess.check_output(["bunzip2","-c",input_file]).\
                    decode('utf-8','replace').split('\n')
        else:
            lines = open(input_file, encoding='utf-8', errors='replace')
            
        for line_ in lines:
            line = line_.strip()
            if line[:7] == '<docno>' or line[:5] == '<url>':
                continue
            pattern_positions = self.match_pattern_in_line(line)
            
            title_positions = [] # save start and end position of a title in a tuple list
                                 # title = line[start+1:end-1]
            self.index_desc_start = {}
                                 # save descriptors in this line by its start position index
                                 # key=index, value=list of descriptors
            self.index_desc_end = {}
                                 # save descriptors in this line by its end position index
                                 # key=index, value=list of descriptors
            for i in range(len(line)):
                self.index_desc_start[i] = []
                self.index_desc_end[i] = []
            # 
            history = 0
            for start, end in pattern_positions:
                if start < end - 2:
                    title = line[start + 1:end - 1]
                    if (given_title == True and title in self.title_set) \
                        or given_title == False:
                        #count frequency of titles
                        self._title_freq_dict[title] += 1
                        title_positions.append([start, end]) # save margins of this title

                        #save descriptors between the last title and current title.
                        descriptor_match(line, history, start)
                        history = end
                        
            descriptor_match(line, history, len(line))

            #count frequency of descs
            for index, descriptors in self.index_desc_start.items():
                for desc in descriptors:
                    self._desc_freq_dict[desc] += 1

            #count frequency of co-occured descriptors near each title
            for i, (start, end) in enumerate(title_positions):
                if i == 0:
                    left_window = (start - self.window_size 
                                   if start - self.window_size > 0 else 0, start)
                else:
                    if start - title_positions[i - 1][1] > self.window_size:
                        left_window = (start - self.window_size, start)
                    else:
                        left_window = (title_positions[i - 1][1], start)
                if i == len(title_positions) - 1:
                    right_window = (end, end + self.window_size 
                                    if end + self.window_size < len(line) else len(line))
                else:
                    if title_positions[i + 1][0] - end > self.window_size:
                        right_window = (end, end + self.window_size)
                    else:
                        right_window = (end, title_positions[i + 1][0])

                title = line[start + 1:end - 1]

                #left window
                for index in range(left_window[0], left_window[1]):
                    dist = left_window[1] - index
                    for desc in self.index_desc_start[index]:
                        dist_ = dist - len(desc)#minimum distance, start from zero
                        self._co_freq_dict[desc][title] += self.weight[dist_]
                #right window
                for index in range(right_window[0], right_window[1]):
                    dist = index - right_window[0] + 1
                    for desc in self.index_desc_end[index]:
                        dist_ = dist - len(desc)#minimum distance, start from zero
                        self._co_freq_dict[desc][title] += self.weight[dist_]


class DescriptorWeightedWindow(DescriptorWindow):
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def set_window_weight(self, window_size, sf = 2):
        """In this class, we decay window weigth 
           when distance between title and description increases. 
           sf is smooth factor. default is 2.
           Shortest distance is 1."""
        self.weight = []
        self.window_size = window_size
        self.weight = [float(sf)/(sf + 1 + i) for i in range(window_size)]


def main(): 
    args = parser.parse_args()
    if args.build:
        desc_file = args.descriptor
        title_file = args.title
        input_file = args.input
        model_file = args.model

        if args.model_type == "paragraph":
            descriptor = DescriptorParagraph()
            descriptor.load_desc(desc_file, args.string_match)
            if title_file != "":
                descriptor.load_title(title_file)
                descriptor.count_freq(input_file, given_title = True)
            else:
                descriptor.count_freq(input_file, given_title = False)
            descriptor.save_model(model_file)
        elif args.model_type == "window":
            descriptor = DescriptorWindow()
            descriptor.load_desc(desc_file, args.string_match)
            if title_file != "":
                descriptor.load_title(title_file)
                descriptor.set_window_weight(args.window_size)
                descriptor.count_freq(input_file, given_title = True)
            else:
                descriptor.set_window_weight(args.window_size)
                descriptor.count_freq(input_file, given_title = False)
            descriptor.save_model(model_file)
        elif args.model_type == "weightedwindow":
            descriptor = DescriptorWeightedWindow()
            descriptor.load_desc(desc_file, args.string_match)
            if title_file != "":
                descriptor.load_title(title_file)
                descriptor.set_window_weight(args.window_size, args.smooth_factor)
                descriptor.count_freq(input_file, given_title = True)
            else:
                descriptor.set_window_weight(args.window_size, args.smooth_factor)
                descriptor.count_freq(input_file, given_title = False)
            descriptor.save_model(model_file)

    elif args.prune:
        descriptor = Descriptor()
        descriptor.prune(args.model, prune_threshold = args.prune_threshold)
        descriptor.save_model(args.pruned_model)

    elif args.clean:
        descriptor = Descriptor()
        descriptor.cleanup_title(args.model)
        descriptor.save_model(args.cleaned_model)

    elif args.merge:
        output_model_file = args.model
        merge_dir = args.mergedir
        descriptor = Descriptor()
        descriptor.merge_model(merge_dir, output_model_file)

    elif args.evaluate:
        descriptor = Descriptor()
        descriptor.load_testset(args.testset)
        descriptor.evaluate(args.model, "server", args.topk)

    elif args.query:
        descriptor = Descriptor()
        load_model = descriptor.scenarios["query"][0]
        ranking = descriptor.scenarios["query"][1]
        if args.string_match == 'max':
            match_desc = descriptor.match_desc_max
        elif args.string_match == 'all':
            match_desc = descriptor.match_desc_all
        load_model(args.model, args.string_match)
        ngram_descs = match_desc(args.query_string)
        if args.disjunction == True:
            ranking(ngram_descs, or_logic = True)
        else:
            ranking(ngram_descs)

    elif args.fastload:
        descriptor = Descriptor()
        load_model = descriptor.scenarios["server"][0]
        ranking = descriptor.scenarios["server"][1]
        if args.string_match == 'max':
            match_desc = descriptor.match_desc_max
        elif args.string_match == 'all':
            match_desc = descriptor.match_desc_all
        load_model(args.model, args.string_match)
        title_format_pattern = re.compile('[\x00-\xff]')
        while(1):
            print("请输入描述：")
            string = input()
            ngram_descs = match_desc(string)
            titles = ranking(ngram_descs, args.topk)
            print("title".ljust(30) +\
                  "rank_prob\t" + "rank_prob(" + ")\trank_prob(".join(ngram_descs) + ")\trank_prob(freq(c,t))\t||\t" +\
                  "log(p(" + "|c,t))\tlog(p(".join(ngram_descs) + "|c,t))\tlog(freq(c,t))\t-log(P(t))")
            print("——————————————————————")
            for title, scores, origin_scores in titles:
                format_title = title + " " * (30 - len(title) * 2 + len(title_format_pattern.findall(title)))
                print(format_title + \
                      str(scores[0])[:7] + '\t' +'\t'.join([str(r)[:7] for r in scores[1:]]) + '\t||\t' + 
                      '\t'.join([str(s)[:7] for s in origin_scores[:-1]]))
            print("——————————————————————")

    elif args.load:
        model_file = args.model
        model_dict = pickle.load(open(model_file,"rb"))

        title_sqrt_dict = copy.deepcopy(model_dict["title_freq_dict"])
        for title in title_sqrt_dict:
            title_sqrt_dict[title] = math.sqrt(float(title_sqrt_dict[title]))
        desc_sqrt_dict = copy.deepcopy(model_dict["desc_freq_dict"])
        for desc in desc_sqrt_dict:
            desc_sqrt_dict[desc] = math.sqrt(float(desc_sqrt_dict[desc]))

        cofreq_devide_title = copy.deepcopy(model_dict["co_freq_dict"])# dict[desc][title]
        for desc in model_dict["co_freq_dict"]:
            for title in model_dict["co_freq_dict"][desc]:
                cofreq_devide_title[desc][title] = cofreq_devide_title[desc][title]/\
                                                   title_sqrt_dict[title]

        cofreq_devide_desc = {}# dict[title][desc]
        for desc in model_dict["co_freq_dict"]:
            for title in model_dict["co_freq_dict"][desc]:
                if title not in cofreq_devide_desc:
                    cofreq_devide_desc[title] =\
                    {desc:(model_dict["co_freq_dict"][desc][title])/desc_sqrt_dict[desc]}
                else:
                    cofreq_devide_desc[title][desc] =\
                    (model_dict["co_freq_dict"][desc][title])/desc_sqrt_dict[desc]

        while(1):
            print("输入电影名[t] Or 输入描述词[d] or 退出[exit]：[t/d/exit]")
            act = input()
            if act == 't':
                print("请输入电影名：")
                title=input()
                if title not in model_dict["title_freq_dict"]:
                    print("电影名不存在")
                    continue
                print("——————————————————————")
                print("descriptor\tco_freq/desc_freq\tco_freq\tdesc_freq")
                for k,v in sorted(cofreq_devide_desc[title].items(), key = lambda x: x[1],
                                  reverse=True)[:args.topk]:
                    print(k + "\t" + str(v) + "\t" + str(model_dict["co_freq_dict"][k][title])\
                                + "\t" + str(model_dict["desc_freq_dict"][k]))
                print("——————————————————————")
            elif act == "d":
                print("请输入描述词：")
                desc = input()
                if desc not in model_dict['desc_freq_dict']:
                    print("描述词不存在")
                    continue
                print("——————————————————————")
                print("title\tco_freq/sqrt(title_freq)\tco_freq\ttitle_freq")
                for k,v in sorted(cofreq_devide_title[desc].items(), key = lambda x: x[1],
                        reverse=True)[:args.topk]:
                    print(k + "\t" + str(v) + "\t" + str(model_dict["co_freq_dict"][desc][k]) +\
                                "\t" + str(model_dict["title_freq_dict"][k]))
            elif act == "exit":
                break

if __name__ == '__main__':
    main()
