#coding=utf-8
import argparse
import os
import re
import pickle
import copy
import math
import sys

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-l","--load",help="load previous model instead of building a new one and \
                                       also transform the co-occurrence dict", action="store_true")
group.add_argument("-fl","--fastload",help="load previous model instead of building a new one.", action="store_true")
group.add_argument("-b","--build",help="build a new model.", action="store_true")
group.add_argument("-m","--merge",help="merge the model files",action="store_true")
parser.add_argument("--model_type",help="This parameter defines the type of the new model. \
                    The type is distinguished by its window type [paragraph|window|weightedwindow]",\
                    default="window")
parser.add_argument("--input",help="the path of the input directory that contains the corpus\
        to cope with.")
parser.add_argument("--descriptor",help="the path of the file that contains\
        descriptors",default="/zfs/octp/sogout/outputs/extract_desc/desc.txt")
parser.add_argument("--title",help="the path of the file that contains film\
        titles",default="/zfs/octp/sogout/outputs/extract_desc/movie_titles_logp.txt")
parser.add_argument("--model",help="the path of the model to save or load",\
        required=True)
parser.add_argument("--mergedir",help="the path of directory where model files were saved and will\
        be merged.",default="/home/dongxu/descriptor/model")
parser.add_argument("--window_size",help="window size for left and right windows",type=int,default=10)
parser.add_argument("--smooth_factor",help="smoothing factor for window weights",type=int,default=2)


class Descriptor:

    def __init__(self):
        self.co_freq_dict = {}
        self.title_freq_dict = {}
        self.desc_freq_dict = {}
        self.pattern = re.compile("《(.*?)》".decode('utf-8'))

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

    def load_desc(self, desc_file):
        descriptors = open(desc_file).read().decode('utf-8').split('\n')
        for i in range(len(descriptors)):
            descriptors[i] = "".join(descriptors[i].split())
        desc_dict = {}#dict is faster than set or list
        for d in descriptors:
            if d not in desc_dict:
                desc_dict[d] = 1
        
        print str(len(descriptors)) + " descriptors loaded."
        self.desc_dict = desc_dict

    def load_title(self, title_file):
        titles=open(title_file).read().decode('utf-8').split('\n')[:-1]
        for i in range(len(titles)):
            titles[i] = titles[i].split()[0]
        title_dict = {}#title is faster than set or list
        for t in titles:
            if t not in title_dict:
                title_dict[t] = 1
        print str(len(titles)) + " title names loaded."
        self.title_dict = title_dict

    def match_pattern_in_line(self, line):
        iterator = self.pattern.finditer(line)
        pattern_positions = []
        for match in iterator:
            pattern_positions.append(match.span())
        return pattern_positions

    def count_freq(self):
        raise NotImplementedError
    
    def save_model(self, model_file):
        model_dict = {"co_freq_dict":self.co_freq_dict, \
                      "title_freq_dict":self.title_freq_dict, \
                      "desc_freq_dict":self.desc_freq_dict}
        pickle.dump(model_dict, open(model_file + ".p","wb"))

    def merge_model(self, merge_dir, output_model_file):
        print "start merging..."
        model_files = self.list_dir(merge_dir)
        print str(len(model_files)) + " model files founded"
        print "start loading models"
        for model_file in model_files:
            model_dict = pickle.load(open(model_file,"rb"))
            for desc in model_dict["co_freq_dict"]:
                if desc not in self.co_freq_dict:
                    self.co_freq_dict[desc] = {}
                for title in model_dict["co_freq_dict"][desc]:
                    if title not in self.co_freq_dict[desc]:
                        self.co_freq_dict[desc][title] =\
                        model_dict["co_freq_dict"][desc][title]
                    else:
                        self.co_freq_dict[desc][title] +=\
                        model_dict["co_freq_dict"][desc][title]
            for title in model_dict["title_freq_dict"]:
                if title not in self.title_freq_dict:
                    self.title_freq_dict[title] = model_dict["title_freq_dict"][title]
                else:
                    self.title_freq_dict[title] += model_dict["title_freq_dict"][title]
            for desc in model_dict["desc_freq_dict"]:
                if desc not in self.desc_freq_dict:
                    self.desc_freq_dict[desc] = model_dict["desc_freq_dict"][desc]
                else:
                    self.desc_freq_dict[desc] += model_dict["desc_freq_dict"][desc]
            print model_file + " merged"
            sys.stdout.flush()
        self.save_model(output_model_file)
        print "model merged"

    def load_model(self, model_file):
        model_dict = pickle.load(open(model_file,"rb"))
        self.title_freq_dict = model_dict["title_freq_dict"]
        self.desc_freq_dict = model_dict["desc_freq_dict"]
        self.co_freq_dict = model_dict["co_freq_dict"]

        self.title_sqrt_dict = copy.deepcopy(self.title_freq_dict)
        for title in self.title_sqrt_dict:
            self.title_sqrt_dict[title] = math.sqrt(float(self.title_sqrt_dict[title]))
        self.co_freq_devide_title = copy.deepcopy(self.co_freq_dict)
        for desc, titles in self.co_freq_devide_title.items():
            for title in titles:
                self.co_freq_devide_title[desc][title] /= self.title_sqrt_dict[title]

    def match_desc(self, string):
        ngram_descs = []
        string_length = len(string)
        current_index = 0
        while(current_index < string_length):
            if current_index + 4 <= string_length and \
               string[current_index:current_index + 4] in self.co_freq_devide_title:
                ngram_descs.append(string[current_index:current_index + 4])
                current_index += 4
            elif current_index + 3 <= string_length and \
               string[current_index:current_index + 3] in self.co_freq_devide_title:
                ngram_descs.append(string[current_index:current_index + 3])
                current_index += 3 
            elif current_index + 2 <= string_length and \
               string[current_index:current_index + 2] in self.co_freq_devide_title:
                ngram_descs.append(string[current_index:current_index + 2])
                current_index += 2
            else:
                current_index += 1
        return ngram_descs

    def rank_titles(self, ngram_descs, topk):
        result_titles = []
        if len(ngram_descs) == 0:
            print "描述词未出现"
            for k,v in sorted(self.title_freq_dict.items(), \
                        lambda x, y: cmp(x[1], y[1]), reverse=True)[:topk]:
                result_titles.append(k)

        else:
            title_scores = {}
            for desc in ngram_descs:
                if desc in self.co_freq_devide_title:
                    for title, score in self.co_freq_devide_title[desc].items():
                        if title not in title_scores:
                            title_scores[title] = score
                        else:
                            title_scores[title] += score
            
            for k,v in sorted(title_scores.items(), \
                        lambda x, y: cmp(x[1], y[1]), reverse=True)[:topk]:
                result_titles.append(k)
        return result_titles

class DescriptorParagraph(Descriptor):

    def __descriptor_allmatch(self, string):
        ngram_descs = []
        string_length = len(string)
        if string_length >= 2:
            for i in range(string_length - 1):
                if string[i:i + 2] in self.desc_dict:
                    ngram_descs.append(string[i:i + 2])
        if string_length >= 3:
            for i in range(string_length - 2):
                if string[i:i + 3] in self.desc_dict:
                    ngram_descs.append(string[i:i + 3])
        if string_length >= 4:
            for i in range(string_length - 3):
                if string[i:i + 4] in self.desc_dict:
                    ngram_descs.append(string[i:i + 4])
        return ngram_descs

    def __descriptor_maxmatch(self, string):
        ngram_descs = []
        string_length = len(string)
        current_index = 0
        while(current_index < string_length):
            if current_index + 4 <= string_length and \
               string[current_index:current_index + 4] in self.desc_dict:
                ngram_descs.append(string[current_index:current_index + 4])
                current_index += 4
            elif current_index + 3 <= string_length and \
               string[current_index:current_index + 3] in self.desc_dict:
                ngram_descs.append(string[current_index:current_index + 3])
                current_index += 3 
            elif current_index + 2 <= string_length and \
               string[current_index:current_index + 2] in self.desc_dict:
                ngram_descs.append(string[current_index:current_index + 2])
                current_index += 2
            else:
                current_index += 1
        return ngram_descs 

    def count_freq(self, input_file):
        if input_file[-4:] == '.bz2':
            lines = os.popen("bunzip2 -c " + input_file).read().split("\n")
        else:
            lines = open(input_file).read().split("\n")
            
        for line_ in lines:
            try:
                line = line_.decode('utf-8')
            except:
                print "Error utf-8 decoding"
                print line_
                continue
            if line[:7] == '<docno>' or line[:5] == '<url>':
                continue
            pattern_positions = self.match_pattern_in_line(line)
            
            matched_titles = []
            matched_descriptors = []
            history = 0
            for start, end in pattern_positions:
                title = line[start + 1:end - 1]
                if start < end - 2 and title in self.title_dict:
                    matched_titles.append(title)
                    matched_descriptors.extend(self.__descriptor_maxmatch(line[history:start]))
                    history = end
            matched_descriptors.extend(self.__descriptor_maxmatch(line[history:]))
            matched_titles = list(set(matched_titles))
            matched_descriptors = list(set(matched_descriptors))
            for title in matched_titles:
                if title not in self.title_freq_dict:
                    self.title_freq_dict[title] = 1
                else:
                    self.title_freq_dict[title] += 1

                for desc in matched_descriptors:
                    if desc not in self.co_freq_dict:
                        self.co_freq_dict[desc]={title:1}
                    else:
                        if title not in self.co_freq_dict[desc]:
                            self.co_freq_dict[desc][title] = 1
                        else:
                            self.co_freq_dict[desc][title] += 1
            for desc in matched_descriptors:
                if desc not in self.desc_freq_dict:
                    self.desc_freq_dict[desc] = 1
                else:
                    self.desc_freq_dict[desc] += 1


class DescriptorWindow(Descriptor):

    def __descriptor_allmatch(self, string, start, end):
        length = end - start
        #2gram
        if length >= 2:
            for i in range(start, end - 1):
                if string[i:i + 2] in self.desc_dict:
                    self.index_desc_start[i].append(string[i:i + 2])
                    self.index_desc_end[i + 1].append(string[i:i + 2])
        #3gram
        if length >= 3:
            for i in range(start, end - 2):
                if string[i:i + 3] in self.desc_dict:
                    self.index_desc_start[i].append(string[i:i + 3])
                    self.index_desc_end[i + 2].append(string[i:i + 3])
        #4gram
        if length >= 4:
            for i in range(start, end - 3):
                if string[i:i + 4] in self.desc_dict:
                    self.index_desc_start[i].append(string[i:i + 4])
                    self.index_desc_end[i + 3].append(string[i:i + 4])

    def __descriptor_maxmatch(self, string, start, end):
        current_index = start
        while(current_index < end):
            if current_index + 3 < end and \
               string[current_index:current_index + 4] in self.desc_dict:
                self.index_desc_start[current_index].append(string[current_index:current_index + 4])
                self.index_desc_end[current_index + 3].append(string[current_index:current_index + 4])
                current_index += 4
            elif current_index + 2 < end and \
               string[current_index:current_index + 3] in self.desc_dict:
                self.index_desc_start[current_index].append(string[current_index:current_index + 3])
                self.index_desc_end[current_index + 2].append(string[current_index:current_index + 3])
                current_index += 3
            elif current_index + 1 < end and \
               string[current_index:current_index + 2] in self.desc_dict:
                self.index_desc_start[current_index].append(string[current_index:current_index + 2])
                self.index_desc_end[current_index + 1].append(string[current_index:current_index + 2])
                current_index += 2
            else:
                current_index += 1

    def set_window_weight(self, window_size):
        self.weight = []
        self.window_size = window_size
        for i in range(window_size):
            self.weight.append(1.0)

    def count_freq(self, input_file):
        if input_file[-4:] == '.bz2':
            lines = os.popen("bunzip2 -c " + input_file).read().split("\n")
        else:
            lines = open(input_file).read().split("\n")
            
        for line_ in lines:
            try:
                line = line_.decode('utf-8')
            except:
                print "Error utf-8 decoding"
                print line_
                continue
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
                    if title in self.title_dict:
                        #count frequency of titles
                        if title not in self.title_freq_dict:
                            self.title_freq_dict[title] = 1
                        else:
                            self.title_freq_dict[title] += 1
                        title_positions.append([start, end]) # save margins of this title

                        #save descriptors between the last title and current title.
                        self.__descriptor_maxmatch(line, history, start)
                        history = end
            self.__descriptor_maxmatch(line, history, len(line))

            #count frequency of descs
            for index, descriptors in self.index_desc_start.items():
                for desc in descriptors:
                    if desc not in self.desc_freq_dict:
                        self.desc_freq_dict[desc] = 1
                    else:
                        self.desc_freq_dict[desc] += 1

            #count frequency of co-occured descriptors near each title
            for i, (start, end) in enumerate(title_positions):
                if i == 0:
                    left_window = (start - self.window_size if start - self.window_size > 0 else 0, start)
                else:
                    if start - title_positions[i - 1][1] > self.window_size:
                        left_window = (start - self.window_size, start)
                    else:
                        left_window = (title_positions[i - 1][1], start)
                if i == len(title_positions) - 1:
                    right_window = (end, end + self.window_size if end + self.window_size < len(line) else len(line))
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
                        dist = dist - len(desc)#minimum distance, start from zero
                        if desc not in self.co_freq_dict:
                            self.co_freq_dict[desc] = {title:self.weight[dist]}
                        else:
                            if title not in self.co_freq_dict[desc]:
                                self.co_freq_dict[desc][title] = self.weight[dist]
                            else:
                                self.co_freq_dict[desc][title] += self.weight[dist]
                #right window
                for index in range(right_window[0], right_window[1]):
                    dist = index - right_window[0] + 1
                    for desc in self.index_desc_end[index]:
                        dist = dist - len(desc)#minimum distance, start from zero   
                        if desc not in self.co_freq_dict:
                            self.co_freq_dict[desc] = {title:self.weight[dist]}
                        else:
                            if title not in self.co_freq_dict[desc]:
                                self.co_freq_dict[desc][title] = self.weight[dist]
                            else:
                                self.co_freq_dict[desc][title] += self.weight[dist]


class DescriptorWeightedWindow(DescriptorWindow):

    def set_window_weight(self, window_size, sf):
        self.weight = []
        self.window_size = window_size
        for i in range(window_size):
            self.weight.append(float(sf)/(sf + 1 + i))


def main(): 
    args = parser.parse_args()
    if args.build:
        desc_file = args.descriptor
        title_file = args.title
        input_file = args.input
        model_file = args.model

        if args.model_type == "paragraph":
            descriptor = DescriptorParagraph()
            descriptor.load_desc(desc_file)
            descriptor.load_title(title_file)
            descriptor.count_freq(input_file)
            descriptor.save_model(model_file)
        elif args.model_type == "window":
            descriptor = DescriptorWindow()
            descriptor.load_desc(desc_file)
            descriptor.load_title(title_file)
            descriptor.set_window_weight(args.window_size)
            descriptor.count_freq(input_file)
            descriptor.save_model(model_file)
        elif args.model_type == "weightedwindow":
            descriptor = DescriptorWeightedWindow()
            descriptor.load_desc(desc_file)
            descriptor.load_title(title_file)
            descriptor.set_window_weight(args.window_size, args.smooth_factor)
            descriptor.count_freq(input_file)
            descriptor.save_model(model_file)
            
    elif args.merge:
        output_model_file = args.model
        merge_dir = args.mergedir
        descriptor = Descriptor()
        descriptor.merge_model(merge_dir, output_model_file)

    elif args.fastload:
        descriptor = Descriptor()
        descriptor.load_model(args.model)
        
        while(1):
            print "输入描述[d] or 退出[exit]：[d/exit]"
            act = raw_input().decode('utf-8')
            if act == "d":
                print "请输入描述："
                try:
                    string = raw_input().decode('utf-8')
                except:
                    string = raw_input()

                ngram_descs = descriptor.match_desc(string)
                titles = descriptor.rank_titles(ngram_descs, 10)
                print "——————————————————————"
                for title in titles:
                    try:
                        print title.encode('utf-8')
                    except:
                        print title
            elif act == "exit":
                break
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
                cofreq_devide_title[desc][title] /= title_sqrt_dict[title]

        cofreq_devide_desc = {}# dict[title][desc]
        for desc in model_dict["co_freq_dict"]:
            for title in model_dict["co_freq_dict"][desc]:
                if title not in cofreq_devide_desc:
                    cofreq_devide_desc[title] =\
                    {desc:model_dict["co_freq_dict"][desc][title]/desc_sqrt_dict[desc]}
                else:
                    cofreq_devide_desc[title][desc] =\
                    model_dict["co_freq_dict"][desc][title]/desc_sqrt_dict[desc]

        while(1):
            print "输入电影名[t] Or 输入描述词[d] or 退出[exit]：[t/d/exit]"
            act = raw_input().decode('utf-8')
            if act == 't':
                print "请输入电影名："
                try:
                    title=raw_input().decode('utf-8')
                except:
                    title=raw_input()
                if title not in model_dict["title_freq_dict"]:
                    print "电影名不存在"
                    continue
                print "——————————————————————"
                print "descriptor\tco_freq/desc_freq\tco_freq\tdesc_freq"
                for k,v in sorted(cofreq_devide_desc[title].items(), lambda x, y: cmp(x[1],\
                   y[1]), reverse=True)[:30]:
                    try:
                        print (k + "\t" + str(v) + "\t" + str(model_dict["co_freq_dict"][k][title])\
                                + "\t" + str(model_dict["desc_freq_dict"][k])).encode('utf-8')
                    except:
                        print k + "\t" + str(v) + "\t" + str(model_dict["co_freq_dict"][k][title]) \
                                + "\t" +str(model_dict["desc_freq_dict"][k])
                print "——————————————————————"
            elif act == "d":
                print "请输入描述词："
                try:
                    desc = raw_input().decode('utf-8')
                except:
                    desc = raw_input()
                if desc not in model_dict['desc_freq_dict']:
                    print "描述词不存在"
                    continue
                print "——————————————————————"
                print "title\tco_freq/sqrt(title_freq)\tco_freq\ttitle_freq"
                for k,v in sorted(cofreq_devide_title[desc].items(), lambda x, y: cmp(x[1], y[1]),\
                        reverse=True)[:30]:
                    try:
                        print (k + "\t" + str(v) + "\t" + str(model_dict["co_freq_dict"][desc][k]) +\
                                "\t" + str(model_dict["title_freq_dict"][k])).encode('utf-8')
                    except:
                        print k + "\t" + str(v) + "\t" + str(model_dict["co_freq_dict"][desc][k]) +\
                                "\t" + str(model_dict["title_freq_dict"][k])
            elif act == "exit":
                break


if __name__ == '__main__':
    main()
