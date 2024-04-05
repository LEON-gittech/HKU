IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_CLS_TOKEN = "[CLS]"
DEFAULT_MASK_TOKEN = "[MASK]"
DEFAULT_SEP_TOKEN = "[SEP]"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from accelerate import Accelerator
import pandas as pd
import os
import numpy as np
import gc
import copy
from tqdm import tqdm
import transformers
from typing import Dict, Optional, Sequence

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_special_tokens_dict(tokenizer):
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if tokenizer.cls_token is None:
        special_tokens_dict["cls_token"] = DEFAULT_CLS_TOKEN
    if tokenizer.mask_token is None:
        special_tokens_dict["mask_token"] = DEFAULT_MASK_TOKEN
    if tokenizer.sep_token is None:
        special_tokens_dict["sep_token"] = DEFAULT_SEP_TOKEN
    return special_tokens_dict

def compute_metrics(logits, labels):
    """
    计算给定 logits 和 labels 的准确率、召回率和正负例的平均置信度。
    :param logits: PyTorch 张量，表示模型的输出 logits。
    :param labels: PyTorch 张量，表示真实的标签。
    :return: 包含准确率、召回率和正负例平均置信度的元组 (accuracy, recall, pos_prob, neg_prob)。
    """
    def confusion_matrix(preds, labels):
        """
        计算混淆矩阵及其四个基本指标：真阳性、假阳性、真阴性和假阴性
        """
        tp = torch.sum((preds == 1) & (labels == 1)).float()
        fp = torch.sum((preds == 1) & (labels == 0)).float()
        tn = torch.sum((preds == 0) & (labels == 0)).float()
        fn = torch.sum((preds == 0) & (labels == 1)).float()
        return tp, fp, tn, fn

    def compute_accuracy(tp, fp, tn, fn):
        """
        计算准确率
        """
        total = tp + fp + tn + fn
        acc = (tp + tn) / total
        return acc

    def compute_recall(tp, fp, tn, fn):
        """
        计算召回率
        """
        if tp+fn:
            recall = tp / (tp + fn)
        else:
            recall = 0
        assert recall == 0 or (not torch.isnan(recall))
        return recall

    def calc_accuracy_recall(logits, labels):
        preds = torch.argmax(logits, dim=1)
        # print(f"preds:{preds}")
        # print(f"labels:{labels}")
        tp, fp, tn, fn = confusion_matrix(preds, labels)
        # print(f"tp {tp} fp{fp} tn{tn} fn{fn}")
        assert tp+fp+tn+fn==labels.shape[0]
        acc = compute_accuracy(tp, fp, tn, fn)
        recall = compute_recall(tp, fp, tn, fn)
        # print(f"acc {acc} recall {recall}")

        if not isinstance(recall, int): recall = recall.item()
        acc = acc.item()
        return acc, recall

    def calc_confidence(logits, labels):
        probs = torch.softmax(logits, dim=1)
        # print(f"probs:{probs}")
        confidence_pos = probs[:, 1][labels == 1].mean().item()
        confidence_neg = probs[:, 0][labels == 0].mean().item()
        if np.isnan(confidence_pos): confidence_pos=0
        if np.isnan(confidence_neg): confidence_neg=0
        assert not (np.isnan(confidence_pos) or np.isnan(confidence_neg))
        return confidence_pos, confidence_neg
    
    accuracy, recall = calc_accuracy_recall(logits, labels)
    confidence_pos, confidence_neg = calc_confidence(logits, labels)
    return (accuracy, recall, confidence_pos, confidence_neg)