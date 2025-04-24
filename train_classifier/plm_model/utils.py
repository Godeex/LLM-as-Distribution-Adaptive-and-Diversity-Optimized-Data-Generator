import torch
from torch import nn
from torch.nn import functional as F
# import pandas as pd
# import numpy as np
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import train_test_split
# from collections import Counter, defaultdict
from nltk.corpus import stopwords
from transformers import AutoTokenizer

import re
import json
import logging
import copy
import os
from torch.utils.data import TensorDataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


logger = logging.getLogger(__name__)


def init_logger():
    # 初始化一个日志记录器（logger）
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, force_download=False)
    # tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_doc(x, word_freq):
    """
    清理文档函数
    Args:
    x -- 待清理的文档列表，每个文档是一个字符串
    word_freq -- 一个字典，键是单词，值是该单词在语料库中出现的频率

    Return:
    clean_docs -- 清理后的文档列表，每个文档是一个字符串，只包含非停用词且频率大于等于5的单词
    """
    # 获取英文停用词集合
    stop_words = set(stopwords.words('english'))
    # 初始化一个空列表，用于存储清理后的文档
    clean_docs = []
    # 获取word_freq中频率最高的50000个单词及其频率（如果word_freq中的单词少于50000个，则获取全部）
    most_commons = dict(word_freq.most_common(min(len(word_freq), 50000)))
    for doc_content in x:
        doc_words = []
        cleaned = clean_str(doc_content.strip())
        for word in cleaned.split():
            if word not in stop_words and word_freq[word] >= 5:
                if word in most_commons:
                    doc_words.append(word)
                else:
                    doc_words.append("<UNK>")
        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)
    return clean_docs


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and val examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, 
                 e1_mask=None, e2_mask=None, keys=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.keys = keys

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Processor(object):
    """
    Processor类用于处理text数据集。
    它根据任务类型设置标签数量，并提供读取数据和创建样本的方法。
    """
    def __init__(self, args):
        self.args = args
        # 根据任务类型设置标签数量（num_label）
        if self.args.task in ['agnews']:
            self.num_label = 4
        elif self.args.task in ['YelpReviewPolarity', 'imdb', 'youtube', 'amazon-polarity', 'SST-2', 'sst-2', 'elec', 'qnli', 'yelp']:
            self.num_label = 2
        elif self.args.task in ['nyt']:
            self.num_label = 26
        elif self.args.task in ['amazon']:
            self.num_label = 23
        elif self.args.task in ['reddit']:
            self.num_label = 45
        elif self.args.task in ['stackexchange']:
            self.num_label = 50
        elif self.args.task in ['arxiv']:
            self.num_label = 98
        # 创建标签的列表（relation_labels）、标签到ID的映射（label2id）和ID到标签的映射（id2label）
        self.relation_labels = [x for x in range(self.num_label)]
        self.label2id = {x:x for x in range(self.num_label)}
        self.id2label = {x:x for x in range(self.num_label)}

    def read_data(self, filename):
        """
        读取数据文件。

        Args:
            filename: 数据文件的路径。

        Yields:
            逐行读取并解析jsonl数据文件中的JSON对象。
        """
        path = filename
        with open(path, 'r') as f:
            data = f  # json.load(f)
            for x in data:
                yield json.loads(x)
        # return data

    def _create_examples(self, data, set_type):
        """
        根据读取的数据创建样本列表。

        Args:
            data: 解析后的数据列表，每个元素是一个字典。
            set_type: 数据集类型，如train、val、test等。

        Returns:
            包含InputExample对象的列表。
        """
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)  # 生成全局唯一标识符
            try:
                # 尝试获取text字段并清洗
                text_a = d["text"].strip().replace('\\n', ' ').replace('\\', ' ').strip("\n\t\"")
            except:
                # 如果不存在text字段，则尝试获取text_a字段
                text_a = d["text_a"].strip().replace('\\n', ' ').replace('\\', ' ').strip("\n\t\"")
            # 标签存储在_id字段中
            label = d["_id"]
            # 尝试获取text_b字段，如果不存在则默认为空字符串
            if 'text_b' in d:
                text_b = d["text_b"]
            else:
                text_b = ''
            # 每处理5000个样本就记录一条日志
            if i % 5000 == 0:
                logger.info(d)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_examples(self, mode, file_to_read=None):
        """
        Args:
            mode: 指定数据集的类型，如train、val、test、unlabeled、contrast等。
            file_to_read: 可选参数，指定要读取的文件路径。如果为None，则根据mode从配置信息中获取文件路径。

        """
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'val':
            file_to_read = self.args.val_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'unlabeled':
            file_to_read = self.args.unlabel_file
        elif mode == 'contrast':
            file_to_read = file_to_read

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        # if mode == 'contrast':
        #     return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read)), mode)
        # else:
        return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read)), mode)


def load_and_cache_examples(args, tokenizer, mode, size=-1, contra_name=None):
    processor = Processor(args)
    if mode in ["test"]:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len_test,
            )
        )
    elif mode in ["val"]:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len_test,
            )
        )
    elif mode in ["contrast"]:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}_{}'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len_test,
                contra_name[:-5]
            )
        )
    else:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len,
            )
        )

    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.cache_dir)
        if mode == "train":
            examples = processor.get_examples("train")
            max_seq_len = args.max_seq_len
        elif mode == "val":
            examples = processor.get_examples("val")
            max_seq_len = args.max_seq_len
        elif mode == "test":
            examples = processor.get_examples("test")
            max_seq_len = args.max_seq_len_test
        elif mode == 'contrast':
            examples = processor.get_examples("contrast", file_to_read=contra_name)
            if "sst2" in contra_name[:-5] or 'mr' in contra_name[:-5] or 'rt' in contra_name[:-5]:
                max_seq_len = 128
            elif 'amazon' in contra_name[:-5]:
                max_seq_len = 128
            elif "yelp" in contra_name[:-5] or 'imdb' in contra_name[:-5]:
                max_seq_len = 400
            else:
                max_seq_len = args.max_seq_len

        else:
            raise Exception("For mode, Only train, val, test is available")
        features = convert_examples_to_features(examples, max_seq_len, tokenizer, add_sep_token=args.add_sep_token,
                                                multi_label=True if args.multi_label else False)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
   
    # Convert to Tensors and build dataset
    if size > 0:
        import random 
        random.shuffle(features)
        features = features[:size]  # 如果指定了size，则只取打乱后的前size个特征
    else:
        size = len(features)
    # 将每个特征转换为长整型张量
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if args.multi_label:
        def generate_list(id_list, length):
            result = [0] * length  # Initialize the list with zeros
            for id in id_list:
                if id < length:
                    result[id] = 1  # Set the corresponding index to 1
            return result
        label_lst = [generate_list(f.label_id, 98) for f in features]
        all_label_ids = torch.tensor(label_lst, dtype=torch.long)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    # 为每个特征生成一个唯一的ID（这里简单地使用特征的索引）
    all_ids = torch.tensor([ _ for _,f in enumerate(features)], dtype=torch.long)
    # 使用TensorDataset构建数据集
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_ids)
    return dataset, processor.num_label, size


def load_and_cache_unlabeled_examples(args, tokenizer, mode, train_size = 100, size = -1):
    processor = Processor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.cache_dir,
        'cached_{}_{}_{}_{}_unlabel'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        )
    )

    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.cache_dir)

        assert mode == "unlabeled"
        examples = processor.get_examples("unlabeled")
        
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    if size > 0:
        features = features[:size]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ids = torch.tensor([_+train_size for _, f in enumerate(features)], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids, all_ids)

    return dataset, len(features)


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                 multi_label=False,
                                ):
    # 初始化一个空列表，用于存储转换后的特征
    features = []
    # sample_per_example = 3
    # 遍历输入示例列表
    for (ex_index, example) in enumerate(examples):
        # 每处理5000个示例，记录一次日志
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # print(example.text_a)
        # 对示例的text_a进行分词
        tokens_a = tokenizer.tokenize(example.text_a)
        # 如果text_b不为空，也对text_b进行分词
        if example.text_b != "":
            tokens_b = tokenizer.tokenize(example.text_b)
        else:
            tokens_b = ''

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        # 如果存在text_b且tokens_a太长，则截断tokens_a
        if tokens_b != '':
            if len(tokens_a) > max_seq_len - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]
        else:
            # 如果不存在text_b，则根据一定规则截断tokens_a
            cutoff_len = int((max_seq_len - special_tokens_count) * 0.7)
            if len(tokens_a) > cutoff_len:
                tokens_a = tokens_a[:cutoff_len]
        # 初始化tokens为tokens_a
        tokens = tokens_a
        # 如果存在text_b，则在tokens_a后添加SEP标记和tokens_b
        if tokens_b != '':
            tokens += [tokenizer.sep_token]
            tokens += tokens_b
        # 如果需要添加SEP标记，则在tokens后添加SEP标记
        if add_sep_token:
            sep_token = tokenizer.sep_token
            tokens += [sep_token]
        # 确保最终的tokens长度不超过max_seq_len - special_tokens_count
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
        # 初始化token_type_ids列表，所有元素都设置为sequence_a_segment_id
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        # 在tokens前添加CLS标记
        cls_token = tokenizer.cls_token
        tokens = [cls_token] + tokens
        # 在token_type_ids前添加cls_token_segment_id
        token_type_ids = [cls_token_segment_id] + token_type_ids
        # tokens[0] = "$"
        # tokens[1] = "<e2>"
        # 将tokens转换为输入ID
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        # 根据multi_label参数的值，将example.label转换为整数或列表
        label_id = int(example.label) if not multi_label else list(example.label)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if multi_label:
                logger.info(f"label: {label_id}")
            else:
                logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(
            InputFeatures(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id=label_id,
                          )
            )

    return features


