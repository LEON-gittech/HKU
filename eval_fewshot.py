import argparse
import pprint
import os
import copy
from str2bool import str2bool
from typing import Dict, Sequence, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification

IGNORE_INDEX = -100
from tqdm import tqdm
import torch
import json

import transformers
from modeling_phi import PhiForCausalLM
from tokenization_codegen import CodeGenTokenizer
from modeling_genmc import GenMC

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_arc_problems(data_path="data/ARC-Easy-test.jsonl"):
    dataset = []
    with open(data_path, encoding="utf-8") as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            candidate_answers = " ".join([f"({label}) {text}" for text, label in zip(json_obj["choices"]["text"], json_obj["choices"]["label"])]).strip()
            for text, label in zip(json_obj["choices"]["text"], json_obj["choices"]["label"]):
                dataset.append({
                    "id": json_obj["id"],
                    "question": json_obj["question"],
                    "candidate_answers": candidate_answers,
                    "answer": text,
                    "label": label,
                    "answerKey": json_obj["answerKey"],
                })
    return dataset


def load_all_demonstrations(train_path="data/ARC-Challenge-train.jsonl"):
    demonstrations = []
    with open(train_path, encoding="utf-8") as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            demonstrations.append((json_obj["question"], json_obj["choices"]["text"], json_obj["choices"]["label"], json_obj["answerKey"]))
    print(f"load {len(demonstrations)} demonstrations from {train_path}")
    return demonstrations


def llm_embedder(llm, sentences, is_query=True):
    INSTRUCTIONS = {
        "qa": {
            "query": "Represent this query for retrieving relevant documents: ",
            "key": "Represent this document for retrieval: ",
        },
        "icl": {
            "query": "Convert this example into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },
        "chat": {
            "query": "Embed this dialogue to find useful historical dialogues: ",
            "key": "Embed this historical dialogue for retrieval: ",
        },
        "lrlm": {
            "query": "Embed this text chunk for finding useful historical chunks: ",
            "key": "Embed this historical text chunk for retrieval: ",
        },
        "tool": {
            "query": "Transform this user request for fetching helpful tool descriptions: ",
            "key": "Transform this tool description for retrieval: "
        },
        "convsearch": {
            "query": "Encode this query and context for searching relevant passages: ",
            "key": "Encode this passage for retrieval: ",
        },
    }

    instruction = INSTRUCTIONS["icl"]
    # prompt
    if is_query:
        sentences = [instruction["query"] + s for s in sentences]
    else:
        sentences = [instruction["key"] + s for s in sentences]

    # Encode
    sentence_embeddings = llm.encode(sentences)
    return sentence_embeddings

def candidate_answers_formating(texts, labels):
    candidate_answers = " ".join([f"({label}) {text}" for text, label in zip(texts, labels)]).strip()
    return candidate_answers

# task 4
def example_formating(question, answer=None, candidate_answers=None, prompt_type="v2.0"):
    if prompt_type == "v1.0":
        if answer is not None:
            prompt = f"Question: {question}\nCandidate answers: {candidate_answers}\nGold answer: {answer}"
        else:
            prompt = f"Question: {question}\nCandidate answers: {candidate_answers}\nGold answer:"
    elif prompt_type == "v2.0":
        if answer is not None:
            prompt = f"Question: {question}\nAnswer: {answer}"
        else:
            prompt = f"Question: {question}\nAnswer:"
    else:
        raise NotImplementedError
    return prompt

def generate_prompt(question, candidate_answers, prompt_type, N,
                    demonstrations, demonstration_embeddings, embedder,
                    top_k=False, top_k_reverse=False):

    indices = list(range(len(demonstrations)))
    if top_k: # task 5
        question_embeddings = llm_embedder(embedder, [question], True) # [1, n_dim] （1，384）
        similarity = question_embeddings @ demonstration_embeddings.T # [1, n_demo] （2251,384）
        indices_sorted = sorted(list(range(len(demonstrations))), key=lambda x: similarity[0][x], reverse=True)
        if top_k_reverse:
            indices = indices_sorted[:N][::-1] + indices_sorted[N:] #逆序操作
        else:
            indices = indices_sorted

    template = ""
    for idx in indices[:N]:
        demo = demonstrations[idx]
        candidate = candidate_answers_formating(demo[1], demo[2])
        gold = demo[1][demo[2].index(demo[3])]
        template += f"\n\n{example_formating(demo[0], answer=gold, candidate_answers=candidate, prompt_type=prompt_type)}"

    template += f"\n\n{example_formating(question, candidate_answers=candidate_answers, prompt_type=prompt_type)}"

    return template.strip()


def get_model(
    base_model: str = "bigcode/starcoder",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = CodeGenTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = PhiForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16
        # device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return tokenizer, model

def get_mistral(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return tokenizer, model

def remove_t5_model_prefix(key):
    # 替换前缀
    return key.replace('t5_model.', '')

def get_t5(base_model, checkpoint):
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    check_point = torch.load(checkpoint)["model_state_dict"]
    # for name, param in model.named_parameters():
    #     print(name)
    # updated_checkpoint = {remove_t5_model_prefix(k): v for k, v in check_point.items()}
    # model.load_state_dict(updated_checkpoint)
    model.load_state_dict(check_point)
    model.eval()
    return tokenizer, model

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=args.max_len,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    args
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, args) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=torch.stack(input_ids).to(device), labels=torch.stack(labels).to(device))

def mistral_preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    args
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, args) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    attention_masks = torch.ones([len(input_ids)]+list(input_ids[0].shape))
    labels = copy.deepcopy(input_ids)
    for label, source_len, attention_mask in zip(labels, sources_tokenized["input_ids_lens"], attention_masks):
        label[:source_len] = IGNORE_INDEX
        attention_mask[source_len:] = 0
    return dict(input_ids=torch.stack(input_ids).to(device), labels=torch.stack(labels).to(device), attention_mask=attention_masks.to(device))



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="/opt/tiger/HKU-DASC7606-A2/data/ARC-Easy-test.jsonl")
    parser.add_argument('--device_id', type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument('--model', type=str, default='microsoft/phi-1_5', help="")
    parser.add_argument('--embedder', type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--N', type=int, default=8, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--overwrite', type=str2bool, default=False, help="")
    parser.add_argument('--prompt_type', type=str, default="v1.0", help="")
    parser.add_argument('--top_k', type=str2bool, default=False, help="")
    parser.add_argument('--top_k_reverse', type=str2bool, default=False, help="")
    args = """--model '/opt/tiger/HKU/saved_models' --embedder "BAAI/bge-small-en-v1.5" --data_path "data/ARC-Challenge-test.jsonl" --start_index 0 --end_index 9999 --max_len 1024 --output_path "test_finetune" --overwrite False --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False""".replace("\"","").replace("\'","").split(" ")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = get_arc_problems(args.data_path)[args.start_index: args.end_index]

    num_samples = len(problems)
    tokenizer, model = get_model(base_model=args.model)
    model.to(device)
    # tokenizer, model = get_mistral(base_model=args.model)
    # tokenizer, model = get_t5(base_model=args.model, checkpoint="/opt/tiger/GenMC/outputs/arc_easy_large/lr_5e-05_seed_1_bs_8_ga_2_layer_num_1_alpha_1.0_beta_0.5/pytorch_model.bin")
    print(f"Loaded {args.model}.")

    embedder = SentenceTransformer(args.embedder, device=device)
    print(f"loaded {args.embedder}.")

    demonstrations = load_all_demonstrations(args.data_path.replace("test", "train"))
    demonstration_embeddings = llm_embedder(embedder, [d[0] for d in demonstrations], False) # ndarray: [n_demons, n_dim]， [d[0] for d in demonstrations]=questions

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        question = problems[i]["question"]
        answer = problems[i]["answer"]
        candidate_answers = problems[i]["candidate_answers"]

        source = generate_prompt(question, candidate_answers, args.prompt_type, args.N,
                                 demonstrations, demonstration_embeddings, embedder,
                                 top_k=args.top_k, top_k_reverse=args.top_k_reverse)
        if i == 0:
            print(f"prompt #{i}: {source}")

        target = " {}".format(answer)
        # encoding = mistral_preprocess([source], [target], tokenizer, args)
        encoding = preprocess([source], [target], tokenizer, args)

        with torch.no_grad():
            outputs = model(**encoding)
            log_likelihood = outputs.loss * -1

        print("Saving results to {}".format(output_file))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": problems[i]["id"],
                "log_likelihood": log_likelihood.tolist(),
                "question": question,
                "candidate_answers": candidate_answers,
                "answer": answer,
                "label": problems[i]["label"],
                "answerKey": problems[i]["answerKey"],
            }) + "\n")

import torch.nn.functional as F
def compute_loss(model, input_ids, input_mask, labels, return_outputs=False):
    hidden_states, logits, outputs = model(input_ids, attention_mask=input_mask, labels=labels)
    
    logits = logits[:,0,:]
    try:
        targets = F.one_hot(labels, num_classes=4)
    except:
        print(labels)
    
    p = torch.softmax(logits, dim=1)
    loss = F.cross_entropy(logits, targets.float(), reduction="none")
    loss = loss.mean()
    # if focalloss_reduction == "mean":
    #     loss = loss.mean()
    # elif focalloss_reduction == "sum":
    #     loss = loss.sum()

    return loss, p

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_CLS_TOKEN = "[CLS]"
DEFAULT_MASK_TOKEN = "[MASK]"
DEFAULT_SEP_TOKEN = "[SEP]"

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

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/opt/tiger/HKU-DASC7606-A2/data/ARC-Easy-test.jsonl")
    parser.add_argument('--device_id', type=str, default="0")
    parser.add_argument('--model', type=str, default='microsoft/phi-1_5', help="")
    parser.add_argument('--embedder', type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--N', type=int, default=8, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--overwrite', type=str2bool, default=False, help="")
    parser.add_argument('--prompt_type', type=str, default="v1.0", help="")
    parser.add_argument('--top_k', type=str2bool, default=False, help="")
    parser.add_argument('--top_k_reverse', type=str2bool, default=False, help="")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=10)
    args = """--model '/opt/tiger/HKU/saved_models' --embedder "BAAI/bge-small-en-v1.5" --data_path "data/ARC-Challenge-train.jsonl" --start_index 0 --end_index 9999 --max_len 1024 --output_path "test_phi2_preprocess" --overwrite False --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False""".replace("\"","").replace("\'","").split(" ")
    args = parser.parse_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = get_arc_problems(args.data_path)[args.start_index: args.end_index]
    tmp = []
    for i in range(len(problems)):
        if i%4==0:
            tmp.append(problems[i])
    problems = tmp
    data = {}
    data["data"] = []
    data["label"]=[]
    dic = {"A":0,"B":1,"C":2,"D":3}
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    prompt = "this is a question and four options marked as A,B,C,D. Choose the right option for the queston and output the answer, the answer should be one of A,B,C,D"
    for p in problems:
        data["data"].append("[CLS] "+prompt+p["question"]+p["candidate_answers"])
        if p["answerKey"] not in "ABCD": data["label"].append(int(p["answerKey"])-1)
        else: data["label"].append(dic[p["answerKey"]])
    num_samples = len(problems)
    tokenizer, model = get_model(base_model=args.model)
    special_tokens_dict = get_special_tokens_dict(tokenizer)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    MAX_LEN = 256
    data["data"] = tokenizer(data["data"], add_special_tokens = True, padding = 'max_length', truncation=True, max_length = MAX_LEN, return_attention_mask = True, return_tensors = 'pt')
    batch_size = 32
    # train_data = TensorDataset()
    print(f"Loaded {args.model}.")
    import numpy as np
    train_inputs = data["data"]["input_ids"]
    train_masks = data["data"]["attention_mask"]
    train_labels = data["label"]
    train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    import random
    model.train()
    model.to(device)
    epochs = args.epoch
    from utils import compute_metrics
    for epoch_i in range(epochs):
        # Set model to training mode
        model.train()

        # Initialize lists to store training loss and predictions for each batch
        train_loss_values = []
        train_preds = []

        # Get the current learning rate
        lr = optimizer.param_groups[0]['lr']

        # Loop through batches of the training dataset using tqdm for progress tracking
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_i+1}/{epochs}", unit="batch")
        # cnt = 0
        for batch in progress_bar:
            # cnt += 1 
            # Load batch to GPU
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Perform forward pass
            # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            # Compute loss
            # loss = outputs[0]
            loss, logits = compute_loss(model,b_input_ids,b_input_mask,b_labels)
            try:
                train_loss_values.append(loss.item())
            except:
                print(loss)

            # Perform backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            
            # if cnt == 10: break
            # Print metrics for this epoch
            print(f"batch: training loss: {loss:.4f}")

        # Calculate average epoch training loss
        train_loss = np.mean(train_loss_values)

        # Print metrics for this epoch
        print(f"Epoch {epoch_i+1}/{epochs} training loss: {train_loss:.4f}")
    
    save_directory = "./saved_models"  # 你想要保存模型的目录

    # 保存模型和分词器
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

if __name__ == '__main__':
    # train()
    main()