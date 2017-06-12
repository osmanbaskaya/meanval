import pandas as pd
from collections import Counter
from glob import glob
import os
from collections import defaultdict as dd


class DataPreprocessor(object):
    pass


class WMT15DataPreprocessor(DataPreprocessor):

    lang_map = {'fi': 'fin', 'ru': 'rus', 'fr': 'fre', 'cs': 'ces', 'de': 'deu'}

    def preprocess(self, csv_file):
        df = pd.read_csv(csv_file)
        df.columns = map(str.lower, df.columns)
        reference_files = glob("../data/wmt15/wmt15-submitted-data/txt/references/*.en")
        translation_files = glob("../data/wmt15/wmt15-submitted-data/txt/system-outputs/*/*-en/*-en")
        reference_d = self.create_translation_dict_from_ref(reference_files)
        system_d = self.create_translation_dict_from_mt(translation_files)
        dfn = WMT15DataPreprocessor.transform_into_one_line_one_system(df)
        dfn['reference'] = dfn.apply(lambda r: WMT15DataPreprocessor.get_transition_ref(r, reference_d), axis=1)
        dfn['transition'] = dfn.apply(lambda r: WMT15DataPreprocessor.get_transition_mt(r, system_d), axis=1)
        dfn.to_csv('wmt15.csv', sep='\t', index=False, header=False)
        print("Shape: {}".format(dfn.shape))

    @staticmethod
    def transform_into_one_line_one_system(df):
        d = dd(list)
        for i, row in df.iterrows():
            row = dict(row)
            common_keys = ['srclang', 'trglang', 'srcindex', 'segmentid', 'judgeid', 'rankingid']
            # Add common keys for the first system
            system1id = row['system1id']
            if 'tuning' not in system1id:
                list(map(lambda k: d[k].append(row[k]), common_keys))
                d["system"].append(system1id)
                d['rank'].append(row['system1rank'])
            # Add common keys for the second system
            system2id = row['system2id']
            if 'tuning' not in system2id:
                list(map(lambda k: d[k].append(row[k]), common_keys))
                d["system"].append(system2id)
                d['rank'].append(row['system2rank'])
        return pd.DataFrame(d).drop_duplicates()

    def read_translation_file(self, filename):
        lines = open(filename).read().splitlines()
        return dict(zip(range(1, len(lines) + 1), lines))

    def create_translation_dict_from_ref(self, filenames):
        d = {}
        for filename in filenames:
            fn = os.path.basename(filename)
            source_lang = fn.split('-')[1][:2]
            source_lang = WMT15DataPreprocessor.lang_map[source_lang]
            if fn == 'newsdiscussdev2015-fren-ref.en':
                continue  # we do not want dev set.
            d[source_lang] = self.read_translation_file(filename)
        return d

    def create_translation_dict_from_mt(self, filenames):
        d = dd(dict)
        for filename in filenames:
            system_name = os.path.basename(filename)
            source_lang = system_name.split('.')[-1].split('-')[0]
            source_lang = WMT15DataPreprocessor.lang_map[source_lang]
            if filename == 'newsdiscussdev2015-fren-ref.en':
                continue  # we do not want dev set.
            d[source_lang]["%s.txt" % system_name] = self.read_translation_file(filename)
        return d

    @staticmethod
    def get_transition_ref(row, d):
        srclang = row['srclang']
        segmentid = row['segmentid']
        return d[srclang][segmentid]

    @staticmethod
    def get_transition_mt(row, d):
        srclang = row['srclang']
        segmentid = row['segmentid']
        system = row['system']
        return d[srclang][system][segmentid]


class Vocabulary(object):
    def __init__(self, word_to_id, num_of_already_allocated_tokens=0):
        self.word_to_id = word_to_id
        self.num_of_already_allocated_tokens = num_of_already_allocated_tokens
        self.__vocab_size = len(word_to_id) + num_of_already_allocated_tokens

    def __iter__(self):
        for word, idx in self.word_to_id.items():
            yield word, idx

    def __getitem__(self, item):
        return self.word_to_id[item]

    @property
    def size(self):
        return self.__vocab_size

    def get(self, item, default=None):
        return self.word_to_id.get(item, default)

    @staticmethod
    def build(words, min_occurrence=1, num_already_allocated_tokens=0):
        counter = Counter(words)

        # Remove words that are observed fewer than the threshold.
        remove_list = [w for w, f in counter.items() if f < min_occurrence]
        list(map(counter.pop, remove_list))

        range_start = num_already_allocated_tokens
        range_finish = len(counter) + num_already_allocated_tokens

        word_to_id = dict(zip(counter.keys(), range(range_start, range_finish)))
        return Vocabulary(word_to_id, num_already_allocated_tokens)

