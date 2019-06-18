import logging
import pdb
import numpy as np
import random
import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import argparse
from tqdm import tqdm, trange
# from sklearn.metrics import precision_recall_curve, roc_curve, auc

from turing.timer import ThroughputTimer as tt
from turing.logger import Logger
from turing.utils import get_sample_writer
from turing.models import BertMultiTask
from turing.sources import PretrainingDataCreator, TokenInstance, WikiNBookCorpusPretrainingDataCreator
from turing.sources import WikiPretrainingDataCreator
from turing.dataset import PreTrainingDataset
from turing.dataset import PretrainBatch, PretrainDataType
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from turing.optimization import warmup_linear, warmup_linear_decay_exp
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from deepscale.distributed import DistributedDataParallel as DeepScaleDataParallel


def checkpoint_model(PATH, model, optimizer, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {'epoch': epoch,
                             'last_global_step': last_global_step,
                             'model_state_dict': model.network.module.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)
    torch.save(checkpoint_state_dict, PATH)
    return


def load_training_checkpoint(PATH):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = torch.load(PATH, map_location=torch.device("cpu"))
    model.network.module.load_state_dict(
        checkpoint_state_dict['model_state_dict'])
    optimizer.load_state_dict(checkpoint_state_dict['optimizer_state_dict'])
    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    del checkpoint_state_dict
    return (epoch + 1, last_global_step)


def get_effective_batch(total):
    if args.local_rank != -1:
        return total//dist.get_world_size()//args.train_batch_size//args.gradient_accumulation_steps//args.refresh_bucket_size
    else:
        return total//args.train_batch_size//args.gradient_accumulation_steps//args.refresh_bucket_size


def get_dataloader(dataset: Dataset, eval_set=False):
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
    return (x for x in DataLoader(dataset, batch_size=args.train_batch_size//2 if eval_set else args.train_batch_size, sampler=train_sampler, num_workers=config['training']['num_workers']))


def pretrain_validation(index):
    model.eval()
    dataset = PreTrainingDataset(tokenizer=args.tokenizer,
                                folder=config['validation']['path'],
                                logger=args.logger, max_seq_length=args.max_seq_length,
                                index=index, data_type=PretrainDataType.VALIDATION,
                                max_predictions_per_seq=args.max_predictions_per_seq,
                                masked_lm_prob=args.masked_lm_prob)
    data_batches = get_dataloader(dataset, eval_set=True)
    eval_loss = 0
    nb_eval_steps = 0
    for batch in tqdm(data_batches):
        batch = tuple(t.to(args.device) for t in batch)
        tmp_eval_loss = model.network(batch, log=False)
        dist.reduce(tmp_eval_loss, 0)
        # Reduce to get the loss from all the GPU's
        tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"Validation Loss for epoch {index + 1} is: {eval_loss}")
    if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
        args.summary_writer.add_scalar(f'Validation/Loss', eval_loss, index+1)
    return

def train(index):
    model.train()
    dataloaders = {}
    i = 0
    global global_step
    datalengths = []
    batchs_per_dataset = []
    batch_mapping = {}

    dataset_paths = config["data"]["datasets"]
    # Pretraining datasets

    wiki_pretrain_dataset = PreTrainingDataset(tokenizer=args.tokenizer,
                                                folder=dataset_paths['wiki_pretrain_dataset'],
                                                logger=args.logger, max_seq_length=args.max_seq_length,
                                                index=index, data_type=PretrainDataType.WIKIPEDIA,
                                                max_predictions_per_seq=args.max_predictions_per_seq,
                                                masked_lm_prob=args.masked_lm_prob)


    datalengths.append(len(wiki_pretrain_dataset))
    dataloaders[i] = get_dataloader(wiki_pretrain_dataset)
    batch_mapping[i] = PretrainBatch
    batchs_per_dataset.append(
        get_effective_batch(len(wiki_pretrain_dataset)))
    i += 1

    bc_pretrain_dataset = PreTrainingDataset(tokenizer=args.tokenizer,
                                            folder=dataset_paths['bc_pretrain_dataset'],
                                            logger=args.logger, max_seq_length=args.max_seq_length,
                                            index=index, data_type=PretrainDataType.BOOK_CORPUS,
                                            max_predictions_per_seq=args.max_predictions_per_seq,
                                            masked_lm_prob=args.masked_lm_prob)
    datalengths.append(len(bc_pretrain_dataset))
    dataloaders[i] = get_dataloader(bc_pretrain_dataset)
    batch_mapping[i] = PretrainBatch
    batchs_per_dataset.append(
        get_effective_batch(len(bc_pretrain_dataset)))
    i += 1

    total_length = sum(datalengths)

    num_batches = total_length // args.train_batch_size

    dataset_batches = []
    for i, batch_count in enumerate(batchs_per_dataset):
        dataset_batches.extend([i]*batch_count)
    # shuffle
    random.shuffle(dataset_batches)

    dataset_picker = []
    for dataset_batch_type in dataset_batches:
        dataset_picker.extend([dataset_batch_type] *
                              args.gradient_accumulation_steps * args.refresh_bucket_size)

    model.train()
    forward_timer = tt(name="Forward Timer: ")
    backward_timer = tt(name="Backward Timer: ")
    overall_timer = tt(name="Overall Timer: ")

    for step, dataset_type in enumerate(tqdm(dataset_picker)):
        try:
            batch = next(dataloaders[dataset_type])
            if args.n_gpu == 1:
                batch = tuple(t.to(args.device) for t in batch)  # Move to GPU

            if step > 1 and step % 100 == 0:
                forward_timer.print_elapsed_time()
                backward_timer.print_elapsed_time()
                overall_timer.print_elapsed_time()

            overall_timer.start()

            # Calculate forward pass
            forward_timer.start()
            loss = model.network(batch)
            forward_timer.stop()

            if args.n_gpu > 1:
                # this is to average loss for multi-gpu. In DistributedDataParallel
                # setting, we get tuple of losses form all proccesses
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Enabling DeepScale optimized Reduction
            # reduction only happens in backward if this method is called before
            # when using the deepscale distributed module
            if args.deepscale and args.local_rank != -1 and (step + 1) % args.gradient_accumulation_steps == 0:
                model.network.enable_need_reduction()
            else:
                model.network.disable_need_reduction()
            backward_timer.start()
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            backward_timer.stop()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # exit(0)
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = config["training"]["learning_rate"] * warmup_linear_decay_exp(global_step,
                                                                                                 config["training"]["decay_rate"],
                                                                                                 config["training"]["decay_step"],
                                                                                                 config["training"]["total_training_steps"],
                                                                                                 config["training"]["warmup_proportion"])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                    ##### Record the LR against global_step on tensorboard #####
                    if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
                        args.summary_writer.add_scalar(
                            f'Train/lr', lr_this_step, global_step)
                    ##### Recording  done. #####
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            overall_timer.stop()
        except StopIteration:
            continue

    # Run Validation Loss
    if args.max_seq_length == 512:
        logger.info(f"TRAIN BATCH SIZE: {args.train_batch_size}")
        pretrain_validation(index)



parser = argparse.ArgumentParser()

# Required_parameter
parser.add_argument("--config-file", "--cf",
                    help="pointer to the configuration file of the experiment", type=str, required=True)
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model checkpoints will be written.")

# Optional Params
parser.add_argument("--max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                    "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--max_predictions_per_seq", "--max_pred", default=80, type=int,
                    help="The maximum number of masked tokens in a sequence to be predicted.")
parser.add_argument("--masked_lm_prob", "--mlm_prob", default=0.15,
                    type=float, help="The masking probability for languge model.")
parser.add_argument("--train_batch_size", default=32,
                    type=int, help="Total batch size for training.")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--do_lower_case",
                    default=True,
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--deepscale',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--use_pretrain',
                    default=False,
                    action='store_true',
                    help="Whether to use Bert Pretrain Weights or not")

parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--refresh_bucket_size',
                    type=int,
                    default=1,
                    help="This param makes sure that a certain task is repeated for this time steps to \
                        optimise on the back propogation speed with APEX's DistributedDataParallel")
parser.add_argument('--load_training_checkpoint', '--load_cp',
                    type=str,
                    default=None,
                    help="This is the path to the TAR file which contains model+opt state_dict() checkpointed.")

args = parser.parse_args()

# Prepare Logger
logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
args.logger = logger
config = json.load(open(args.config_file, 'r', encoding='utf-8'))
args.config = config

print("Running Config File: ", config['name'])
# Setting the distributed variables

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    if args.fp16:
        logger.info(
            "16-bits distributed training not officially supported but seems to be working.")
        args.fp16 = True  # (see https://github.com/pytorch/pytorch/pull/13496)
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))

if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
        args.gradient_accumulation_steps))

args.train_batch_size = int(
    args.train_batch_size / args.gradient_accumulation_steps)

# Setting all the seeds so that the task is random but same accross processes
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
logger.info
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

# if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
#     raise ValueError("Output directory () already exists and is not empty.")

os.makedirs(args.output_dir, exist_ok=True)
args.saved_model_path = os.path.join(
    args.output_dir, "saved_models/", config['name'])

# Prepare Summary Writer and saved_models path
if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
    summary_writer = get_sample_writer(
        name=config['name'], base=args.output_dir)
    args.summary_writer = summary_writer
    os.makedirs(args.saved_model_path, exist_ok=True)


# set device
args.device = device
args.n_gpu = n_gpu

# Loading Tokenizer
tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
args.tokenizer = tokenizer

# Loading Model
model = BertMultiTask(args)

if args.fp16:
    model.half()
model.to(device)

if args.local_rank != -1:
    try:
        if args.deepscale:
            logger.info(
                "***** Enabling DeepScale Optimized Data Parallel *****")
            from deepscale.distributed_apex import DistributedDataParallel as DDP
        else:
            logger.info(
                "***** Using Default Apex Distributed Data Parallel *****")
            from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    torch.cuda.set_device(args.local_rank)
    model.network = DDP(model.network, delay_allreduce=False)

elif n_gpu > 1:
    model.network = nn.DataParallel(model.network)

# Prepare Optimizer
param_optimizer = list(model.network.named_parameters())
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if args.fp16:
    try:
        from apex.optimizers import FP16_Optimizer, FusedAdam
    except:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=config["training"]["learning_rate"],
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(
            optimizer, static_loss_scale=args.loss_scale)
else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config["training"]["learning_rate"],
                         warmup=config["training"]["warmup_proportion"],
                         t_total=config["training"]["total_training_steps"])

global_step = 0
start_epoch = 0
if args.load_training_checkpoint is not None:
    logger.info(
        f"Restoring previous training checkpoint from {args.load_training_checkpoint}")
    start_epoch, global_step = load_training_checkpoint(
        args.load_training_checkpoint)
    logger.info(
        f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step}")


for index in range(start_epoch, config["training"]["num_epochs"]):
    logger.info(f"Training Epoch: {index + 1}")
    train(index)

    if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
        logger.info(
            f"Saving a checkpointing of the model for epoch: {index+1}")
        model.save_bert(os.path.join(args.saved_model_path,
                                        "bert_encoder_epoch_{}.pt".format(index + 1)))
        checkpoint_model(os.path.join(args.saved_model_path,
                                        "training_state_checkpoint_{}.tar".format(index + 1)), model, optimizer, index, global_step)
