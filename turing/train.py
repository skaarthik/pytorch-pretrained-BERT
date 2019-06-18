from datetime import datetime
import logging
import pdb
from operator import itemgetter
from pathlib import Path

import glob
# import GPUtil
import numpy as np
import random
import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from shutil import copyfile
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

from azureml.core.run import Run


def copy_files(files_list):
    """
    - - - For testing with files copied onto the machines - - -

    :param files_list: List of files to copy from the blob onto the machines
    :return: Nothing but files are copied
    """
    os.makedirs('/tmp/wiki_pretrain', exist_ok=True)
    os.makedirs('/tmp/bookcorpus_pretrain', exist_ok=True)
    os.makedirs('/tmp/validation_512_only', exist_ok=True)
    src_root = config['data']['datasets']
    for file in files_list:
        if 'wiki' in file:
            src = src_root['wiki_pretrain_dataset']
            dest = 'wiki_pretrain'
        elif 'book' in file:
            src = src_root['bc_pretrain_dataset']
            dest = 'bookcorpus_pretrain'
        else:
            src = config['validation']['path']
            dest = 'validation_512_only'

        logger.info("Copying {}".format(file))
        full_src = os.path.join(src, file)
        full_dest = os.path.join('/tmp', dest, file)
        copyfile(full_src, full_dest)


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


def latest_checkpoint_file(reference_folder: str) -> str:
    """Extracts the name of the last checkpoint file

    :param reference_folder: (str) Path to the parent_folder
    :return: (str) Path to the most recent checkpoint tar file
    """

    # Extract sub-folders under the reference folder
    matching_sub_dirs = [d for d in os.listdir(reference_folder)]
    logger.info("reference_folder = {}".format(reference_folder))
    logger.info("matching_sub_dirs = {}".format(matching_sub_dirs))

    # For each of these folders, find those that correspond
    # to the proper architecture, and that contain .tar files
    candidate_files = []
    for sub_dir in matching_sub_dirs:
        logger.info("sub_dir = {}".format(sub_dir))
        logger.info("children = {}".format([d for d in os.listdir(os.path.join(reference_folder, sub_dir))]))
        for dir_path, dir_names, filenames in os.walk(os.path.join(reference_folder, sub_dir)):
            if 'saved_models' in dir_path:
                relevant_files = [f for f in filenames if f.endswith('.tar')]
                logger.info("relevant_files = {}".format(relevant_files))
                if relevant_files:
                    latest_file = max(relevant_files)  # assumes that checkpoint number is of format 000x
                    candidate_files.append((dir_path, latest_file))
                    logger.info("candidate_files = {}".format(candidate_files))

    checkpoint_file = max(candidate_files, key=itemgetter(1))
    checkpoint_path = os.path.join(checkpoint_file[0], checkpoint_file[1])

    return checkpoint_path


def get_effective_batch(total):
    if local_rank != -1:
        return total//dist.get_world_size()//args.train_batch_size//args.gradient_accumulation_steps//args.refresh_bucket_size
    else:
        return total//args.train_batch_size//args.gradient_accumulation_steps//args.refresh_bucket_size


def get_dataloader(dataset: Dataset, eval_set=False):
    if local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
    return (x for x in DataLoader(dataset, batch_size=args.train_batch_size // 2 if eval_set else args.train_batch_size,
                                  sampler=train_sampler, num_workers=config['training']['num_workers']))


def pretrain_validation(index):
    model.eval()
    dataset = PreTrainingDataset(tokenizer=args.tokenizer,
                                 folder=config['validation']['path'],
                                 # folder='/tmp/validation_512_only',
                                 logger=args.logger, max_seq_length=args.max_seq_length,
                                 index=index, data_type=PretrainDataType.VALIDATION,
                                 max_predictions_per_seq=args.max_predictions_per_seq,
                                 masked_lm_prob=args.masked_lm_prob)
    data_batches = get_dataloader(dataset, eval_set=True)
    eval_loss = 0
    nb_eval_steps = 0

    if display_progress_bar:
        data_batches = tqdm(data_batches)

    # for batch in tqdm(data_batches):
    for batch in data_batches:
        batch = tuple(t.to(args.device) for t in batch)
        tmp_eval_loss = model.network(batch, log=False)
        dist.reduce(tmp_eval_loss, 0)
        # Reduce to get the loss from all the GPU's
        tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"Validation Loss for epoch {index + 1} is: {eval_loss}")
    run.log("validation_loss", np.float(eval_loss))
    if (not no_cuda and dist.get_rank() == 0) or (no_cuda and local_rank == -1):
        args.summary_writer.add_scalar(f'Validation/Loss', eval_loss, index + 1)
    return


def train(index):
    run.log('epoch_number', np.int(index))
    model.train()
    dataloaders = {}
    i = 0
    global global_step
    datalengths = []
    batchs_per_dataset = []
    # batch_mapping = {}

    dataset_paths = config["data"]["datasets"]
    # Pretraining datasets

    wiki_pretrain_dataset = PreTrainingDataset(tokenizer=args.tokenizer,
                                               folder=dataset_paths['wiki_pretrain_dataset'],
                                               # folder='/tmp/wiki_pretrain',
                                               logger=args.logger, max_seq_length=args.max_seq_length,
                                               index=index, data_type=PretrainDataType.WIKIPEDIA,
                                               max_predictions_per_seq=args.max_predictions_per_seq,
                                               masked_lm_prob=args.masked_lm_prob)

    datalengths.append(len(wiki_pretrain_dataset))
    dataloaders[i] = get_dataloader(wiki_pretrain_dataset)
    # batch_mapping[i] = PretrainBatch
    batchs_per_dataset.append(
        get_effective_batch(len(wiki_pretrain_dataset)))
    i += 1

    bc_pretrain_dataset = PreTrainingDataset(tokenizer=args.tokenizer,
                                             folder=dataset_paths['bc_pretrain_dataset'],
                                             # folder='/tmp/bookcorpus_pretrain',
                                             logger=args.logger, max_seq_length=args.max_seq_length,
                                             index=index, data_type=PretrainDataType.BOOK_CORPUS,
                                             max_predictions_per_seq=args.max_predictions_per_seq,
                                             masked_lm_prob=args.masked_lm_prob)
    datalengths.append(len(bc_pretrain_dataset))
    dataloaders[i] = get_dataloader(bc_pretrain_dataset)
    # batch_mapping[i] = PretrainBatch
    batchs_per_dataset.append(
        get_effective_batch(len(bc_pretrain_dataset)))
    i += 1

    total_length = sum(datalengths)

    num_batches = total_length // args.train_batch_size

    dataset_batches = []
    for i, batch_count in enumerate(batchs_per_dataset):
        dataset_batches.extend([i] * batch_count)
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

    if display_progress_bar:
        dataset_picker = tqdm(dataset_picker)

    # Counter of sequences in an "epoch"
    sequences_counter = 0

    # for step, dataset_type in enumerate(tqdm(dataset_picker)):
    for step, dataset_type in enumerate(dataset_picker):
        try:
            batch = next(dataloaders[dataset_type])

            sequences_counter += len(batch)

            if args.n_gpu == 1:
                batch = tuple(t.to(args.device) for t in batch)  # Move to GPU

            if step > 1 and step % 1000 == 0:
                forward_timer.print_elapsed_time()
                backward_timer.print_elapsed_time()
                overall_timer.print_elapsed_time()
                logger.info("Current time: {}".format(datetime.utcnow()))
                logger.info("Type(batch): {}".format(type(batch)))  # This is a tuple
                logger.info("batch[0]: {}".format(batch[0]))
                logger.info("Cumulative sequences count: {}".format(sequences_counter))

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
                if step % 5000 == 0:
                    run.log("training_loss", np.float(loss))
                loss = loss / args.gradient_accumulation_steps

            # Enabling DeepScale optimized Reduction
            # reduction only happens in backward if this method is called before
            # when using the deepscale distributed module
            if deepscale and local_rank != -1 and (step + 1) % args.gradient_accumulation_steps == 0:
                model.network.enable_need_reduction()
            else:
                model.network.disable_need_reduction()
            backward_timer.start()
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            backward_timer.stop()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # exit(0)
                if fp16:
                    # modify learning rate with special warm up BERT uses
                    # if fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = \
                        config["training"]["learning_rate"] * warmup_linear_decay_exp(global_step,
                                                                                      config["training"][
                                                                                          "decay_rate"],
                                                                                      config["training"][
                                                                                          "decay_step"],
                                                                                      config["training"][
                                                                                          "total_training_steps"],
                                                                                      config["training"][
                                                                                          "warmup_proportion"])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                    ##### Record the LR against global_step on tensorboard #####
                    if (not no_cuda and dist.get_rank() == 0) or (no_cuda and local_rank == -1):
                        args.summary_writer.add_scalar(
                            f'Train/lr', lr_this_step, global_step)
                        # if step % 5000 == 0:
                        #     run.log("learning_rate", np.float(lr_this_step))
                    ##### Recording  done. #####
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            overall_timer.stop()
        except StopIteration:
            continue
    logger.info("Steps count: {}".format(step))
    logger.info("Sequences count: {}".format(sequences_counter))

    # Run Validation Loss
    if args.max_seq_length == 512:
        logger.info(f"TRAIN BATCH SIZE: {args.train_batch_size}")
        pretrain_validation(index)


def str2bool(val):
    """

    :param val: (str) String value of the boolean parameter
    :return: (bool) Boolean version of the input string
    """
    return val.lower() == "true" or val.lower() == "t" or val.lower() == "1"


print("The arguments are: " + str(sys.argv))

# print('** Environment variables are: **')
# for key in os.environ.keys():
#     print('{0}: {1}'.format(key, os.environ[key]))
# print('** End of env variables **')

parser = argparse.ArgumentParser()

# Required_parameter
parser.add_argument("--config_file", "--cf",
                    help="pointer to the configuration file of the experiment", type=str, required=True)
parser.add_argument("--files_location", default=None, type=str, required=True,
                    help="The directory in the blob storage which contains data and config files.")
# parser.add_argument("--output_dir", default=None, type=str, required=True,
#                     help="The output directory where the model checkpoints will be written.")

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
                    type=str,
                    default='False',
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
                    type=str,
                    default='True',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16',
                    type=str,
                    default='False',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--deepscale',
                    type=str,
                    default='False',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--use_pretrain',
                    type=str,
                    default='False',
                    help="Whether to use Bert Pretrain Weights or not")

parser.add_argument('--loss_scale',
                    type=float,
                    default=0,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--refresh_bucket_size',
                    type=int,
                    default=1,
                    help="This param makes sure that a certain task is repeated for this time steps to \
                        optimise on the back propogation speed with APEX's DistributedDataParallel")
parser.add_argument('--load_training_checkpoint', '--load_cp',
                    type=str,
                    default='False',
                    help="This is the path to the TAR file which contains model+opt state_dict() checkpointed.")

parser.add_argument('--display_progress_bar',
                    type=str,
                    default='False',
                    help="Whether to display progress bars in the log files or not")


args = parser.parse_args()

no_cuda = str2bool(args.no_cuda)
do_lower_case = str2bool(args.do_lower_case)
fp16 = str2bool(args.fp16)
deepscale = str2bool(args.deepscale)
use_pretrain = str2bool(args.use_pretrain)
display_progress_bar = str2bool(args.display_progress_bar)

args.no_cuda = no_cuda
args.do_lower_case = do_lower_case
args.fp16 = fp16
args.deepscale = deepscale
args.use_pretrain = use_pretrain
args.display_progress_bar = display_progress_bar
# ^^ this is needed below, in model = BertMultiTask(args)

local_rank = -1

local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

# TODO remove this hack
# Doing following assignment because BertMultiTask takes args in constructor and references local_rank from it
args.local_rank = local_rank

os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
master_node_params = os.environ['AZ_BATCH_MASTER_NODE'].split(':')
os.environ['MASTER_ADDR'] = master_node_params[0]
os.environ['MASTER_PORT'] = master_node_params[1]
print('NCCL_SOCKET_IFNAME = {}'.format(os.environ['NCCL_SOCKET_IFNAME']))
os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'

print('RANK = {}'.format(os.environ['RANK']))
print('WORLD_SIZE = {}'.format(os.environ['WORLD_SIZE']))
print('MASTER_ADDR = {}'.format(os.environ['MASTER_ADDR']))
print('MASTER_PORT = {}'.format(os.environ['MASTER_PORT']))
# print('MASTER_NODE = {}'.format(os.environ['MASTER_NODE']))
print('NCCL_SOCKET_IFNAME = {}'.format(os.environ['NCCL_SOCKET_IFNAME']))

# Prepare Logger
logger = Logger(cuda=torch.cuda.is_available() and not no_cuda)
args.logger = logger

print('local_rank is ' + str(local_rank))

# # Extact config file from blob storage
config_path = os.path.join(args.files_location, args.config_file)

config = json.load(open(config_path, 'r', encoding='utf-8'))
# Replace placeholder path prefix by path corresponding to "ds.path('data/bert_data/').as_mount()"
config['data']['datasets'] = {key: value.replace('placeholder/', args.files_location)
                              for (key, value) in config['data']['datasets'].items()}
config['validation']['path'] = config['validation']['path'].replace('placeholder/', args.files_location)
args.config = config

print("Running Config File: ", config['name'])
# Setting the distributed variables

run = Run.get_context()

if local_rank == -1 or no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available()
                                    and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    if fp16:
        logger.info(
            "16-bits distributed training not officially supported but seems to be working.")
        fp16 = True  # (see https://github.com/pytorch/pytorch/pull/13496)
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(local_rank != -1), fp16))

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

# Create an outputs/ folder in the blob storage
parent_dir = os.path.join(args.files_location, 'outputs', str(run.experiment.name))
output_dir = os.path.join(parent_dir, str(run.id))
os.makedirs(output_dir, exist_ok=True)
args.saved_model_path = os.path.join(output_dir, "saved_models", config['name'])

saved_model_path = args.saved_model_path

# Prepare Summary Writer and saved_models path
if (not no_cuda and dist.get_rank() == 0) or (no_cuda and local_rank == -1):
    summary_writer = get_sample_writer(
        name=config['name'], base=output_dir)
    args.summary_writer = summary_writer
    os.makedirs(args.saved_model_path, exist_ok=True)

# set device
args.device = device
args.n_gpu = n_gpu

# Loading Tokenizer (vocabulary from blob storage, if exists)
logger.info("*** Extracting the vocabulary ***")
tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"], cache_dir=args.files_location)
args.tokenizer = tokenizer
logger.info("Vocabulary contains {} tokens".format(len(list(tokenizer.vocab.keys()))))

# Loading Model
logger.info("*** Loading the model ***")
model = BertMultiTask(args)
logger.info("Model of type: {}".format(type(model)))

logger.info("*** Converting the input parameters ***")
if fp16:
    model.half()
model.to(device)

if local_rank != -1:
    try:
        if deepscale:
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
    torch.cuda.set_device(local_rank)
    model.network = DDP(model.network, delay_allreduce=False)

elif n_gpu > 1:
    model.network = nn.DataParallel(model.network)

# Prepare Optimizer
logger.info("*** Preparing the optimizer ***")
param_optimizer = list(model.network.named_parameters())
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

logger.info("*** Loading Apex and building the FusedAdam optimizer ***")
if fp16:
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

logger.info("*** Verifying checkpoints ***")
global_step = 0
start_epoch = 0
# if args.load_training_checkpoint is not None:
if args.load_training_checkpoint != 'False':
    # logger.info(
    #     f"Restoring previous training checkpoint from {args.load_training_checkpoint}")
    # start_epoch, global_step = load_training_checkpoint(
    #     args.load_training_checkpoint)

    # Extract folder path and filename of latest checkpoint
    latest_checkpoint_path = latest_checkpoint_file(parent_dir)

    # latest_checkpoint_path = os.path.join(args.files_location, 'outputs', '2019_05_26_064558',
    #                                       'bert-pretraining-base-500epochs',
    #                                       'bert-pretraining-base-500epochs_1558853109_3902e454',
    #                                       'saved_models', 'bing-bert-base',
    #                                       'training_state_checkpoint_202.tar')

    logger.info(f"Restoring previous training checkpoint from {latest_checkpoint_path}")
    start_epoch, global_step = load_training_checkpoint(latest_checkpoint_path)
    logger.info(
        f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step}")

# - - - For testing with files copied onto the machines - - -
# logger.info("*** Copying files locally ***")
# files_to_copy = ['bookcorpus_segmented_part_0.bin', 'bookcorpus_segmented_part_1.bin',
#                  'bookcorpus_segmented_part_2.bin', 'wikipedia_segmented_part_0.bin',
#                  'wikipedia_segmented_part_1.bin', 'wikipedia_segmented_part_2.bin',
#                  'validation_set.bin']
# copy_files(files_to_copy)

logger.info("*** Training the model ***")
for index in range(start_epoch, config["training"]["num_epochs"]):
    logger.info(f"Training Epoch: {index + 1}")
    # with torch.autograd.profiler.profile() as prof:
    #     train(index)
    # logger.info(prof)
    train(index)

    # gpu = GPUtil.getGPUs()[int(os.environ['RANK'])]
    # mpi_rank = int(os.environ['RANK'])
    # run.log('GPU memory: {}'.format(mpi_rank), gpu.memoryUsed)
    # run.log('GPU load: {}'.format(mpi_rank), gpu.load)
    # print(mpi_rank, gpu.memoryUsed)  # REMOVE THIS
    # print(mpi_rank, gpu.load)  # REMOVE THIS

    if (not no_cuda and dist.get_rank() == 0) or (no_cuda and local_rank == -1):
        logger.info(
            f"Saving a checkpointing of the model for epoch: {index + 1}")
        model.save_bert(os.path.join(args.saved_model_path,
                                     "bert_encoder_epoch_{0:04d}.pt".format(index + 1)))
        checkpoint_model(os.path.join(args.saved_model_path,
                                      "training_state_checkpoint_{0:04d}.tar".format(index + 1)),
                         model, optimizer, index, global_step)

        # - - - For testing with files copied onto the machines - - -
        # model.save_bert(os.path.join('/tmp',
        #                              "bert_encoder_epoch_{}.pt".format(index + 1)))
        # checkpoint_model(os.path.join('/tmp',
        #                               "training_state_checkpoint_{}.tar".format(index + 1)), model, optimizer, index,
        #                  global_step)