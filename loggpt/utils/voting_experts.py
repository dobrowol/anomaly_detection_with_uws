import multiprocessing as mp
import pickle
import subprocess
import numpy as np
import time
import math
import os
import logging
from tqdm import tqdm
from pathlib import Path
from utils.tree import find_node, calculate_experts_features
from utils.tree import tree_from_ngram, Tree, get_stats, standardize
from utils.tree import build_tree, update_tree
from concurrent.futures import ThreadPoolExecutor, wait

# from tree import find_node, calculate_experts_features
# from tree import tree_from_ngram, Tree, get_stats, standardize
# from tree import build_tree, update_tree

logger = logging.getLogger(__name__)

def get_num_of_sliding_windows_in_a_sentence(sentence, sliding_window_size):
    num_of_sliding_windows_in_a_sentence=len(sentence)-sliding_window_size+1
    return 1 if num_of_sliding_windows_in_a_sentence<=0 else num_of_sliding_windows_in_a_sentence

def _split_with_length(sentence, size_of_a_bucket, ngram_size):
    sentence_len = len(sentence)
    #split_data = [sentence[0:size_of_a_bucket]]
    split_data_2 = [sentence[i:i+size_of_a_bucket] for i in range(0, sentence_len, size_of_a_bucket)]
    #split_data.extend(split_data_2)
    return split_data_2


class VotingExperts:
    def __init__(self, window_size, threshold, out_directory=None, ngram_tree=None, silent=True, threads=55,
                  max_line_size=140000):
        """
        tree_depth: int. Depth of n_gram tree. Depth of 4 will create a tree of 3_grams
        window_size: int. Size of window considered by voting expert in each iteration. Should be less than (tree_depth - 1)
        threshold: int. Boundary of votes that needs to be exceeded in order to split the sentence at given index. With only entropy =2 works well.
        silent: boolean. If false, program will show current progress.
        threads: int. Number of threads working in parallel
        """
        self.out_directory=out_directory
        if out_directory is None:
            self.out_directory="./"
        self.tree_depth = window_size+1
        self.threshold = threshold
        self.silent = silent
        self.window_size = window_size
        self.ngram_tree = ngram_tree
        self.threads = threads
        self.longest_word = 0
        self.max_line_size = max_line_size
        self.tree_name = "ngram_tree"
        self.thread_names = []

    def get_ngram_tree(self):
        return self.ngram_tree

    def get_longest_word(self):
        return self.longest_word

    def set_tree_depth(self, value):
        self.tree_depth = value

    def find_frequencies(self, node, depth, nodes):
        if depth == 0:
            if node.value > 0:
                nodes.append(node.value)
        for child in node.nodes.values():
            self.find_frequencies(child, depth - 1, nodes)
    
    def count_ngrams(self, file_name):

        srilm_path = os.getenv('SRILM_PATH')
        out_file = Path(self.out_directory)/f"{Path(file_name).stem}_{self.tree_depth}gram"

        self.split_file_name = self.create_split_file(file_name)

        if Path(out_file).is_file():
            return out_file

        logger.debug(f"{srilm_path}/ngram-count -order {self.tree_depth} -text {self.split_file_name} -no-sos -no-eos -write {out_file} ")
        start_time = time.time()
        subprocess.call(f"{srilm_path}/ngram-count -order {self.tree_depth} -text {self.split_file_name} -no-sos -no-eos -write {out_file}", shell=True)
        logger.debug(f"calculating ngram of order {self.tree_depth} for file {self.split_file_name} took ----- {(time.time() - start_time)} seconds -----")
                    
        return out_file
    
    def retrieve_tree(self, a_dir):
        if self.ngram_tree is not None:
            return self.ngram_tree
        start_time = time.time()
        trees = list(Path(a_dir).glob(f"*_{self.tree_depth}gram.tree"))
        if len(trees) == 0:
            return None
        tree_file = trees[0]
        logger.debug(f"using old tree {tree_file}")
        with open(tree_file, "rb") as inp:
            self.ngram_tree = pickle.load(inp)
            logger.info("loading old ngram tree took ----- %s seconds -----" % (time.time() - start_time))
        return self.ngram_tree
    
    def fit(self, data):
        if self.max_line_size < self.tree_depth:
            logger.error("tree depth cannot be greater than a line")
            return False
        start_time = time.time()
        self.ngram_tree = None

        if self.window_size < 1 or self.tree_depth < 2:
            return False
        if self.ngram_tree is None:
            self.build_ngram_tree(data)
            self.standardize_tree()
            out_file = Path(self.out_directory)/f"{self.tree_name}_{self.tree_depth}gram.tree"
            # with open(out_file, "wb") as out:
            #     pickle.dump(self.ngram_tree, out)
            #logger.debug(f"ngram tree saved to {out_file}")
            #logger.info("-----fit took %s seconds -----" % (time.time() - start_time))
            #print("-----fit took %s seconds -----" % (time.time() - start_time))
        
        return True

    def build_ngram_tree_with_srilm(self, file_name):
        lm = self.count_ngrams(file_name)
        if self.ngram_tree is None:
            self.ngram_tree = tree_from_ngram(Tree(0), lm)
        else:
            self.ngram_tree = tree_from_ngram(self.ngram_tree, lm)
        Path(lm).unlink()

    def build_ngram_tree(self, data):
        lines = 0
        sentences = []
        # self.thread_names=[]
        for line in data.values():
            lines +=1
            sentence = []
            # self.thread_names.append(line.split(';')[0].strip())
            for word in line:
                sentence.append(word)
            sentences.append(sentence)
        if lines == 0:
            logger.error("Draining empty. See above errors.")
            return
        if not self.ngram_tree:
            self.ngram_tree = build_tree(sentences, self.tree_depth)
            #print("ngram tree is None", self.ngram_tree is None)
        else:
            update_tree(self.ngram_tree, sentences, self.tree_depth)

    def standardize_tree(self):
       
        standards = {}
        calculate_experts_features(self.ngram_tree, standard=standards)
        stats = get_stats(standards)
        logger.info("standardizing experts features")
        standardize(self.ngram_tree, stats, threads=self.threads)
        

    def transform(self, data, window=None, threshold=None):
        
        if window is not None:
            self.window_size = window
            self.tree_depth = window+1
        if threshold is not None:
            self.threshold = threshold
        
        # with ThreadPoolExecutor(max_workers=self.threads) as pool:
                    
        #     futures = [pool.submit(self.transform_file, file_name, out_filenames)
        #                 for file_name in tqdm(file_list)]
        #     print('Waiting for tasks to complete...')
        #     wait(futures)

        return self.transform_file(data) 
            

    def transform_file(self, data):

        start_time = time.time()
        
        transformed_lines = {}
        [self.transform_line( key, line, transformed_lines)
                        for key,line in data.items()]
        # with ThreadPoolExecutor(max_workers=self.threads) as pool:

        #     futures = [pool.submit(self.transform_line, key, line, transformed_lines)
        #                 for key,line in data.items()]
        #     print('Waiting for transforming tasks to complete...')
        #     wait(futures)
        # for key,value in transformed_lines.items():
        #     print(key, value)

            #self.save_results(out_filename, transformed_lines)
        

        return transformed_lines

    def transform_line(self, key, line, transformed_lines):
        
        thread_name =key
        sentence =  line
        if sentence == []:
            return          
        split_pattern = self.vote(sentence)
        fragmented_sentence = self.split(sentence, split_pattern)
        transformed_lines[thread_name] = fragmented_sentence
        return fragmented_sentence

    def vote_parallel(self, sentence):
        num_of_sliding_windows_in_a_sentence = get_num_of_sliding_windows_in_a_sentence(sentence,self.window_size)
        if num_of_sliding_windows_in_a_sentence<self.threads:
            threads_count = num_of_sliding_windows_in_a_sentence
        else:
            threads_count = self.threads 

        manager = mp.Manager()
        return_dict = manager.dict()
        split_pattern = np.zeros((len(sentence)),dtype='float64')
        self.vote_thread(0, sentence, return_dict)
        #subsets = _split(sentence, self.threads, self.tree_depth)
        #with ThreadPoolExecutor(max_workers=self.threads) as pool:
        #            for sub in tqdm(subsets):
        #                pool.submit(self.vote_thread, sub[0], sub[1], return_dict)
        #jobs = []
        
        # for sub in subsets:
        #     p = multiprocessing.Process(target=self.thread, args=(sub[0], sub[1], return_dict))
        #     jobs.append(p)
        #     p.start()
        # for proc in jobs:
        #     proc.join()

        for key,value in return_dict.items():
            end = value.shape[0]
            split_pattern[key:key+end] += value
        return split_pattern

    def vote(self, sequence):
        """
        sequence: a list of characters creating text meant to be split
            for example ['i', 't', 'w', 'a', 's', 'a' ,'c', 'o', 'l', 'd'] instead of 'it was a cold' or ['11', '12', '11', '13'] instead of '11 12 11 13'

        returns: list[str]. List of len(sequence) with votes for each possible split
        """
        fixed_length = len(sequence)
        slice_array = np.zeros(fixed_length)
        windows_count = fixed_length - self.window_size + 1
        iter_range = range(windows_count)
        if not self.silent:
            iter_range = tqdm(iter_range)
        for i in iter_range:
            subsequence = sequence[i:i + self.window_size]
            entropies = []
            frequencies = []
            prev_node = self.ngram_tree
            for j in range(0,len(subsequence)):
                # print(subsequence)
                node = prev_node.nodes[subsequence[j]]
                node_rest = find_node(subsequence[j+1:], self.ngram_tree)
                prev_node = node
                entropies.append(node.entropy)
                frequencies.append(node.frequency + node_rest.frequency)
            max_entropy = max(entropies)
            max_frequency = max(frequencies)
            entropy_slice_spot = i + entropies.index(max_entropy)
            frequency_slice_spot = i + frequencies.index(max_frequency)
            slice_array[entropy_slice_spot] += 1
            slice_array[frequency_slice_spot] += 1
        return slice_array

    def split(self, sequence, slicing_pattern):
        #TODO zmienić, żeby zwracał indeksy powyżej threshold, 
        #i na podstawie draina odbudowywał sekwencje podziału
        sen = []
        split_sequence = []
        for i in range(len(slicing_pattern)):
            sen.append(sequence[i])
            if slicing_pattern[i] > self.threshold:
                split_sequence.append(sen)
                sen = []
        if not sen == []:
            split_sequence.append(sen)
        return split_sequence

    def vote_thread(self, sub_id, sub, return_dict):
        pattern = self.vote(sub)
        return_dict[sub_id] = pattern

    def create_split_file(self, file_name):
        # print("create split file")
        split_file_name = Path(self.out_directory)/f"{Path(file_name).stem}_{self.max_line_size}_splitted.txt"
        if not Path(split_file_name).is_file():
            with open(file_name, "r") as data_file:
                for line in data_file:
                    sentence =  line.strip().split(' ') 
                    if len(sentence) < self.max_line_size:
                        return file_name
                    
                    segments = _split_with_length(sentence, self.max_line_size, self.tree_depth)
                    
                    if 'split_file' not in locals():
                        split_file = open(split_file_name, "w")
                    for seg in segments:
                        split_file.write(' '.join(seg)+'\n')
                    split_file.close()
        logger.debug("splitting file to ", split_file_name)
        return split_file_name
          
    def save_results(self, out_filename, transformed_lines):
        with open(out_filename, "wb") as out_file:
            print(f"saving transformed {len(transformed_lines.keys())} lines")
            
            pickle.dump(list(sorted(transformed_lines.values(), key=lambda item: item[0][0])), out_file)

    def fit_transform(self, data):
        """
        dataset_file: string. File contains list of *.drain files that are going to be scanned

        returns a file name with result. Each line is a list of segments of log_ids separated by whitespaces. Example ['11 11 12', '13 14', '15', '16 17 18 19']
        """
        #logger.debug(f"fit transform Entropy window {self.window_size} threshold {self.threshold}")
        if self.window_size < 1 or self.tree_depth < 2 or self.window_size >= self.tree_depth:
            return []
 
        self.fit(data)

        return self.transform(data)

if __name__ == '__main__':
    import sys
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("window", type=int)
    parser.add_argument("threshold", type=int)
    parser.add_argument("out_dir", type=str)

    args = parser.parse_args()

    ve = VotingExperts(args.window, args.threshold, out_directory=args.out_dir)
    with open(args.dataset, "r") as dataset:
            files_list = dataset.read().splitlines()
    ve.fit(files_list)