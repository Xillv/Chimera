import argparse
import logging
import os
import random
import datetime

import torch
import numpy as np
from transformers import AutoTokenizer

from dataset import Twitter_Dataset_FlanT5
from model import Chimera
from trainer_flant5 import train, evaluate
from utils import write_json

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()

    # Directory parameters
    parser.add_argument('--dataset_name', type=str, default='political_twitter', choices=['twitter2015', 'twitter2017', 'political_twitter'])
    parser.add_argument('--data_dir', type=str, default='./data', help='A Directory of the data')

    parser.add_argument('--pretrained_model_dir', type=str, default='google/flan-t5-base', help='Path to the pretrained model')
    parser.add_argument('--pretrained_model_config_dir', type=str, default='./pretrained/flan-t5-base', help='Path to the pretrained model')
    parser.add_argument('--generation_config', type=str, default='generation_config.json', help='File name of generation_config')

    parser.add_argument('--save_model_dir', type=str, default='./checkpoints/Chimera', help='Path to checkpoints')
    parser.add_argument('--seed', type=int, default=11, help='random seed for initialization')
    parser.add_argument('--cuda_id', type=str, default='0', help='Choose which GPUs to run')

    # Model parameters
    parser.add_argument("--img_hidden_size", default=768, type=int, help="Hidden size of image feature.")
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of pretrained model.')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes of ABSA.')

    # Training parameters
    parser.add_argument('--num_workers', type=int, default=0, help='#workers for data loader')
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evalating.")

    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--num_train_epochs", default=10, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="The proportion of warmup in total steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--logging_steps', type=int, default=600, help="Log every X updates steps.")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument('--eval_metric', type=str, default='avg', help='select checkpoint by which metrics')
    parser.add_argument('--multi_task', type=str, default='multi_task', choices=['multi_task','no_sr', 'no_ir', 'no_all'])
    parser.add_argument('--lamda', type=float, default=0.15, help="lamda is a hyperparameter for the main loss")
    parser.add_argument('--cap_index', type=int, default=3, help="the index of caption")

    
    parser.add_argument('--aggr_ratio', type=float, default=0.4, help='the aggr rate for visual token')
    parser.add_argument('--sparse_ratio', type=float, default=0.5, help='the sprase rate for visual token') 
    parser.add_argument('--attention_weight', type=int, default=0.7, help='the weight of attention_map for mask prediction') 
    parser.add_argument('--ratio_weight', type=float, default=2.0, help='if use detach for kt loss')
    parser.add_argument('--loss', type=str, default='vse', help='the objectve function for optimization')
    parser.add_argument('--margin', default=0.1, type=float, help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--beta', type=float, default=0.5, help="beta is a hyperparameter for the patch-token alignment")
    

    return parser.parse_args()

def check_args(args):
    '''
    eliminate conflict situations
    '''
    logger.info(vars(args))
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    args_file_path = os.path.join(args.save_model_dir, 'args.json')
    write_json(args_file_path, vars(args), mode='a')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def create_experiment_directory(base_dir):
    # Generate a directory with the current date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # Parse args
    args = parse_args()

    # Create a unique directory for each experiment based on date and time
    experiment_dir = create_experiment_directory(args.save_model_dir)
    args.save_model_dir = experiment_dir  # Update save_model_dir to use the new directory

    check_args(args)

    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_dir)   # len =32100
    tokenizer.add_tokens(
        ['<image>', '</image>', '<explain>', '</explain>', '<impression>', '</impression>', '<emotion>', '</emotion>', '<aesthetic>', '</aesthetic>'
         'qa: ', 'qea: ', 'qiea: '])
    args.tokenizer = tokenizer

    # Build Dataset
    train_dataset = Twitter_Dataset_FlanT5(args, split='train')
    dev_dataset = Twitter_Dataset_FlanT5(args, split='dev')
    test_dataset = Twitter_Dataset_FlanT5(args, split='test')

    # Build Model
    model = MDCAFlanT5Model(args)
    model.to(args.device)

    # Train Model
    _, _, all_eval_results, best_model = train(args, train_dataset, model, dev_dataset)
    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x['acc'])
        for key in sorted(best_eval_result.keys()):
            logger.info("  %s = %s", key, str(best_eval_result[key]))

    # Test
    test_results = evaluate(args, test_dataset, best_model, True)
    logger.info("***** Test Results *****")
    for key in test_results.keys():
        logger.info("  %s = %s", key, str(test_results[key]))
    readme_path = os.path.join(args.save_model_dir, 'readme.txt')
    with open(readme_path, 'a+') as writer:
        writer.write('***** Test Results *****\n')
        writer.write('acc={}, f1={}'.format(test_results['acc'], test_results['f1']))
        writer.write('\n')

if __name__ == '__main__':
    main()
