import os
import logging
from tqdm import tqdm, trange
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.sampler import SubsetRandomSampler
# from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification
import copy
import math
import os
import random 
from sklearn.metrics import f1_score
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter
from collections import Counter
from sklearn.metrics import confusion_matrix, ndcg_score, jaccard_score

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 检查是否指定了使用GPU的数量（args.n_gpu），并且系统上有可用的CUDA GPU
    if args.n_gpu > 0  and torch.cuda.is_available():
        # print('yes')
        # assert 0
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def compute_metrics_rel(key, prediction):
    # 初始化三个计数器，分别用于追踪每个关系被正确预测的次数、被猜测的次数以及实际出现的次数
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == 0 and guess == 0:
            pass
        elif gold == 0 and guess != 0:
            guessed_by_relation[guess] += 1
        elif gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        elif gold != 0 and guess != 0:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return {'p':prec_micro, 'r':recall_micro, 'f':f1_micro}


def acc_and_f1(preds, labels, average='macro'):
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='macro')
    # macro_recall = recall_score(y_true=labels, y_pred = preds, average = 'macro')
    # micro_recall = recall_score(y_true=labels, y_pred = preds, average = 'micro')
    # print(acc, macro_recall, micro_recall)

    return {
        "acc": acc,
        "f1": f1
    }


class Trainer(object):
    def __init__(self, args, train_dataset=None, val_dataset=None,
                 test_dataset=None, unlabeled=None, num_labels=10, data_size=100, n_gpu=1):
        """
        初始化Trainer类。

        参数:
        - args: 包含模型路径、缓存目录等配置的参数对象。
        - train_dataset: 训练数据集，默认为None。
        - val_dataset: 验证数据集，默认为None。
        - test_dataset: 测试数据集，默认为None。
        - unlabeled: 未标注的数据集，默认为None。
        - num_labels: 分类任务中的标签数量，默认为。
        - data_size: 数据集大小，默认为。
        - n_gpu: 使用的GPU数量，默认为1。
        """
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.unlabeled = unlabeled
        self.data_size = data_size

        self.num_labels = num_labels
        self.config_class = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.n_gpu = 1

    def reinit(self):
        """
        重新初始化模型。
        包括加载模型和初始化模型设备（GPU/CPU）。
        """
        self.load_model()
        self.init_model()

    def init_model(self):
        """
        初始化模型设备。
        根据是否有可用的GPU和n_gpu参数，决定模型是在GPU还是CPU上运行。
        如果n_gpu > 1，则使用DataParallel进行多GPU并行计算。
        """
        self.device = "cuda" if torch.cuda.is_available() and self.n_gpu > 0 else "cpu"
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
    
    def load_model(self, path = None):
        """
        加载模型。

        参数:
        - path: 模型路径。如果为None，则从原始预训练模型路径加载。
        """
        print("load Model")
        if path is None:
            logger.info("No ckpt path, load from original ckpt!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            )
        else:
            print(f"Loading from {path}!")
            logger.info(f"Loading from {path}!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            )

    def save_prediction_test(self, test_preds, test_labels):
        """
        保存测试集的预测结果和真实标签。

        参数:
        - test_preds: 测试集的预测结果。
        - test_labels: 测试集的真实标签。
        """
        output_dir = os.path.join(
            self.args.output_dir, self.args.gen_model, self.args.model_type
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 如果是多标签分类任务, 保存真实标签到 reciprocal_ranks.npy 文件
        if self.args.multi_label:
            with open(f"{output_dir}/reciprocal_ranks.npy", 'wb') as f:
                np.save(f, test_labels)
            
            with open(f"{output_dir}/precision_5_per_example.npy", 'wb') as f:
                np.save(f, test_preds)
        # 如果是单标签分类任务，保存真实标签到 test_label.npy 文件
        else:
            with open(f"{output_dir}/test_label.npy", 'wb') as f:
                np.save(f, test_labels)
            
            with open(f"{output_dir}/test_pred.npy", 'wb') as f:
                np.save(f, test_preds)

    def save_prediction(self, loss, preds, labels, test_preds, test_labels):
        """
        保存训练集的损失、预测结果、真实标签以及测试集的预测结果和真实标签。

        参数:
        - loss: 训练集的损失值。
        - preds: 训练集的预测结果。
        - labels: 训练集的真实标签。
        - test_preds: 测试集的预测结果。
        - test_labels: 测试集的真实标签。
        """
        output_dir = os.path.join(
            self.args.output_dir, self.args.gen_model
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/train_pred.npy", 'wb') as f:
            np.save(f, preds)
    
        with open(f"{output_dir}/train_loss.npy", 'wb') as f:
            np.save(f, loss)
        
        with open(f"{output_dir}/train_label.npy", 'wb') as f:
            np.save(f, labels)

        with open(f"{output_dir}/test_label.npy", 'wb') as f:
            np.save(f, test_labels)
        
        with open(f"{output_dir}/test_pred.npy", 'wb') as f:
            np.save(f, test_preds)

    def save_model(self, stage = 0):
        """
        保存模型和训练参数。

        参数:
        - stage: 当前训练阶段（迭代次数），默认为0。
        """
        output_dir = os.path.join(
            self.args.output_dir,  "checkpoint-{}".format(len(self.train_dataset)), self.args.model_type, "iter-{}".format(stage), f"seed{self.args.train_seed}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 如果模型是并行训练的（如多GPU），则保存模型的 module 属性，否则直接保存模型
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        # 保存模型到指定目录-Hugging Face Transformers 提供的方法
        model_to_save.save_pretrained(output_dir)
        # 保存训练参数到 training_args.bin 文件
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        # 记录日志，提示模型已保存
        logger.info("Saving model checkpoint to %s", output_dir)

    def evaluate(self, mode, dataset=None, global_step=-1, return_preds=False):
        """
       在指定数据集上评估模型性能。

       参数:
       - mode: 评估模式，可以是 'test'、'val'、'contra' 或 'unlabeled'。
       - dataset: 如果 mode 为 'contra'，则需要传入自定义数据集。
       - global_step: 当前训练步数，默认为 -1。
       - return_preds: 是否返回预测结果，默认为 False。

       返回:
       - 如果 return_preds 为 True，返回评估结果和预测结果。
       - 否则，仅返回评估结果。
       """
        # We use test dataset because semeval doesn't have val dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'val':
            dataset = self.val_dataset
        elif mode == 'contra':
            dataset = dataset
        elif mode == 'unlabeled':
            dataset = self.unlabeled
        else:
            raise Exception("Only val and test dataset available")

        # 创建顺序采样器和数据加载器
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # 打印评估信息
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", self.args.batch_size)

        # 初始化评估变量
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # 将模型设置为评估模式
        self.model.eval()

        # 遍历数据加载器
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # 将数据移动到指定设备（GPU或CPU）
            batch = tuple(t.to(self.device) for t in batch)
            # 禁用梯度计算
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }

                # 如果模型类型是 distilbert，则删除 token_type_ids
                if 'distilbert' in self.args.model_type:
                    del inputs['token_type_ids']

                # 如果是多标签分类任务，临时保存标签并删除 inputs 中的 labels
                if self.args.multi_label:
                    lbls = inputs["labels"]
                    del inputs["labels"]

                # 前向传播，获取模型输出
                outputs = self.model(**inputs)

                # 如果是多标签分类任务
                if self.args.multi_label:
                    # 恢复 inputs 中的 labels
                    inputs["labels"] = lbls
                    # 获取 logits 并应用 sigmoid 函数
                    logits = outputs[0]
                    logits = F.sigmoid(logits)
                    # 计算二元交叉熵损失
                    tmp_eval_loss = F.binary_cross_entropy(logits, lbls.float())
                    # print(logits.shape)   
                    # print(F.sigmoid(logits))    
                    # print(batch[3])  
                    # f1_score(y_true, y_pred, average = None) 
                    # exit()
                else:
                    # 如果是单标签分类任务，直接获取损失和 logits
                    tmp_eval_loss, logits = outputs[:2]
                # 累加损失
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            # 将预测结果和标签保存到 NumPy 数组中
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        # 计算平均损失
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss 
        }
        
        def calculate_mrr(labels, predictions):
            """
            计算平均倒数排名（Mean Reciprocal Rank, MRR）。

            参数:
            - labels: 真实标签矩阵（形状为 [样本数, 标签数]）。
            - predictions: 预测概率矩阵（形状为 [样本数, 标签数]）。

            返回:
            - mrr: 平均倒数排名。
            - reciprocal_ranks: 每个样本的倒数排名。
            """
            # 对预测概率按行排序，获取排序后的索引（从高到低）
            sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]
            # 找到每个样本的真实标签在排序后的预测中的排名
            ranks = np.argmax(labels[np.arange(len(labels))[:, None], sorted_indices], axis=1) + 1
            # 计算倒数排名
            reciprocal_ranks = 1.0 / ranks
            # 计算平均倒数排名
            mrr = np.mean(reciprocal_ranks)
            return mrr, reciprocal_ranks

        def calculate_precision_at_k(labels, predictions, k):
            """
            计算前 k 个预测的精度（Precision@K）。

            参数:
            - labels: 真实标签矩阵（形状为 [样本数, 标签数]）。
            - predictions: 预测概率矩阵（形状为 [样本数, 标签数]）。
            - k: 计算精度时考虑的前 k 个预测。

            返回:
            - precision: 平均精度。
            - precision_per_example: 每个样本的精度。
            """
            # 对预测概率按行排序，获取排序后的索引（从高到低）
            sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]
            # 获取前 k 个预测的标签
            top_k_labels = labels[np.arange(len(labels))[:, None], sorted_indices[:, :k]]
            # 计算平均精度
            precision = np.mean(top_k_labels)
            # 计算每个样本的精度
            precision_per_example = np.mean(top_k_labels , axis=1)
            return precision, precision_per_example

        if self.args.multi_label:
            # 获取预测概率
            preds_probs = preds
            # 将预测概率转换为二进制预测（大于等于 0.5 的为 1，否则为 0）
            preds_binary = np.zeros(preds.shape) 
            preds_binary[np.where(preds >= 0.5)] = 1 # binary prediction
            # 计算 NDCG@1, NDCG@3, NDCG@5
            ndcg_1 = ndcg_score(out_label_ids, preds_probs, k = 1)
            ndcg_3 = ndcg_score(out_label_ids, preds_probs, k = 3)
            ndcg_5 = ndcg_score(out_label_ids, preds_probs, k = 5)
            # 计算 F1 分数（宏平均和微平均）
            f1_macro = f1_score(out_label_ids, preds_binary, average='macro')
            f1_micro = f1_score(out_label_ids, preds_binary, average='micro')
            # 计算 Precision@1, Precision@3, Precision@5
            precision_1, precision_1_per_example = calculate_precision_at_k(out_label_ids, preds_probs, k = 1)
            precision_3, precision_3_per_example = calculate_precision_at_k(out_label_ids, preds_probs, k = 3)
            precision_5, precision_5_per_example = calculate_precision_at_k(out_label_ids, preds_probs, k = 5)
            # 计算 MRR
            mrr, reciprocal_ranks = calculate_mrr(out_label_ids, preds_probs)
            print("=========================")
            print("NDCG@1", ndcg_1, "NDCG@3", ndcg_3, "NDCG@5", ndcg_5, "F1 Macro", f1_macro, "F1 Micro", f1_micro)
            print("P@1",precision_1, "P@3",precision_3, "P@5",precision_5,  "MRR", mrr)
            print("=========================")
            if return_preds:
                 return results["loss"], f1_macro, ndcg_3, precision_5_per_example, reciprocal_ranks
            else:
                return results["loss"], f1_macro, ndcg_3
        else:
            # 将 logits 转换为概率（通过 softmax）
            preds_probs = np.exp(preds) / np.sum(np.exp(preds), axis = -1, keepdims = True)
            # 获取预测类别（概率最大的类别）
            preds = np.argmax(preds, axis=1)

            # 如果是未标注数据集，返回预测结果和概率
            if mode == 'unlabeled':
                return preds, preds_probs, out_label_ids
            # 计算评估指标（如准确率、F1 分数等）
            result = compute_metrics(preds, out_label_ids)
            result.update(result)
            logger.info("***** Eval results *****")

            # print('Accu: %.4f'%(result["acc"]))
            # 如果需要返回预测结果
            if return_preds:
                print("=================")
                print("Confusion Matrix:")
                print(confusion_matrix(out_label_ids, preds))
                print("====================")
                return results["loss"], result["acc"], result["f1"], preds_probs, out_label_ids
            else:
                return results["loss"], result["acc"], result["f1"]
    
    def train(self, n_sample = 20):
        """
        训练模型。

        参数:
        - n_sample: 样本数量，默认为 20。

        返回:
        - global_step: 总训练步数。
        - tr_loss / global_step: 平均训练损失。
        """
        ### 1. 初始化训练设置
        # 使用随机采样器创建训练数据加载器
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        # 定义不需要权重衰减的参数名称列表
        no_decay = ['bias', 'LayerNorm.weight']
        # 根据是否需要权重衰减分组模型参数
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # 初始化AdamW优化器
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # 计算总训练步数
        training_steps = max(self.args.max_steps, int(self.args.num_train_epochs) * len(train_dataloader))
        # 根据是否为多标签任务选择损失函数
        criterion = nn.CrossEntropyLoss(reduction='mean') if not self.args.multi_label else nn.BCEWithLogitsLoss()
        # 初始化学习率调度器
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(training_steps * 0.06 / self.args.gradient_accumulation_steps), num_training_steps = training_steps)

        ### 2. 打印训练信息
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size )
        logger.info("  Real batch size = %d", self.args.batch_size * self.args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", training_steps)

        ### 3. 训练循环
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        # 迭代训练
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)
        best_val = -np.float64('inf')

        for _ in train_iterator:  # 外层循环：遍历每个epoch
            global_step = 0
            tr_loss = 0.0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            local_step = 0
            for step, batch in enumerate(epoch_iterator):   # 内层循环：遍历每个batch
                self.model.train()  # 设置模型为训练模式
                # 将batch数据移动到指定设备（GPU或CPU）
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                # 如果模型类型是 distilbert，则删除 token_type_ids
                if 'distilbert' in self.args.model_type:
                    del inputs['token_type_ids']

                # 如果是多标签任务，删除 inputs 中的 labels
                if self.args.multi_label:
                    del inputs["labels"]

                # 前向传播，获取模型输出
                outputs = self.model(**inputs)

                # 根据是否为多标签任务计算损失
                if self.args.multi_label:
                    logits = outputs[0]
                    loss = criterion(input=logits, target=batch[3].float())
                else:
                    loss = outputs[0]

                # 梯度累积和反向传播
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                # 如果使用多 GPU，对损失取平均
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                # 反向传播
                loss.backward()
                tr_loss += loss.item()

                # 梯度累积步骤完成后更新模型参数
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    local_step += 1
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step() # 更新参数
                    scheduler.step() # 更新学习率
                    self.model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("iteration:%d, Loss:%.3f, best val:%.3f" % (_, tr_loss/global_step, 100*best_val)) # 更新进度条描述
                # 如果达到最大训练步数，提前结束训练
                if 0 < training_steps < global_step:
                    epoch_iterator.close()
                    break

            ### 4. 验证和测试
            # 每个epoch结束后在开发集和测试集上评估模型
            loss_val, acc_val, f1_val = self.evaluate('val', global_step)
            loss_test, acc_test, f1_test = 0, 0, 0
            loss_test, acc_test, f1_test = self.evaluate('test', global_step)
            # 更新最佳模型
            if acc_val > best_val:
                logger.info("Best model updated!")
                self.best_model = copy.deepcopy(self.model.state_dict())
                best_val = acc_val
            print(f'Val: Loss: {loss_val}, Acc: {acc_val}, F1: {f1_val}', f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {f1_test}')

        ### 5. 保存结果和模型
        result_dict = {'seed': self.args.train_seed}
        loss_test, acc_test, acc_f1, preds_probs, out_label_ids = self.evaluate('test', global_step, return_preds = True)
        result_dict['acc'] = acc_f1
        result_dict['lr'] = self.args.learning_rate
        result_dict['bsz'] = self.args.batch_size
        result_dict['model'] = self.args.gen_model
        print(f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {acc_f1}')
        self.save_prediction_test(preds_probs, out_label_ids)
        self.save_model(stage = n_sample)
        # 返回总训练步数和平均训练损失
        return global_step, tr_loss / global_step
  