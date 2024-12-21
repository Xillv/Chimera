import logging
import os
import copy
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
import json
from dataset import collate_fn_flant5
from utils import write_json, compute_metrics

logger = logging.getLogger(__name__)


def train(args, train_dataset, model, eval_dataset):
    '''Train the model'''
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset ,sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_flant5)

    t_total = int(len(train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps)
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = get_optimizer(args, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    best_acc, best_f1 = 0.0, 0.0
    best_epoch = 0
    best_model = None
    model.zero_grad()
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        ea_losses, iea_losses, a_losses = 0, 0, 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs, senti_labels = get_input_from_batch(args, batch)
            inputs['is_eval'] = False
            a_loss, ea_loss, iea_loss = model(**inputs)

            ea_losses += ea_loss.item()
            iea_losses += iea_loss.item()
            a_losses += a_loss.item()
  
            # ablation study
            if args.multi_task == 'no_sr':
                loss =  (1-args.lamda) * iea_loss +  args.lamda * a_loss
                del ea_loss
                torch.cuda.empty_cache()
            elif args.multi_task == 'no_ir':
                loss =  (1-args.lamda) * ea_loss +  args.lamda * a_loss
                del iea_loss
                torch.cuda.empty_cache()
            elif args.multi_task == 'no_all':
                loss =  a_loss
                del ea_loss, iea_loss
                torch.cuda.empty_cache()
            else:
                loss = (1-args.lamda) / 2 * ea_loss + (1-args.lamda) / 2 * iea_loss +  args.lamda * a_loss            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            tr_loss += loss.item()

            # Log metrics
            if (epoch > 2 and args.logging_steps > 0 and global_step % args.logging_steps == 0) or global_step==t_total:
                results = evaluate(args, eval_dataset, model)
                all_eval_results.append(results)
                logger.info("Traing Loss: {}".format((tr_loss - logging_loss) / args.logging_steps))
                logging_loss = tr_loss

                if results['acc'] >= best_acc:
                    best_acc = results['acc']
                    best_f1 = results['f1']
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch + 1


        logger.info("ea_loss: {}, iea_loss: {}, a_loss: {}".format(ea_losses, iea_losses, a_losses))
     
    # Save model checkpoint
    save_model(args.save_model_dir, best_model)
    readme_path = os.path.join(args.save_model_dir, 'readme.txt')
    with open(readme_path, 'a+') as writer:
        writer.write('Save best model at {} epoch, best_acc={}, best_f1={}'.format(best_epoch, best_acc, best_f1))
        writer.write('\n')
        
    return global_step, tr_loss/global_step, all_eval_results, best_model


def evaluate(args, eval_dataset, model, is_test=False):

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn_flant5)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    a_pred_sequences, ea_pred_sequences, iea_pred_sequences = [], [], []
    out_label_ids = []
    a_results, ea_results, iea_results = {}, {}, {}
    
    for batch in tqdm(eval_dataloader):
        model.eval()
        with torch.no_grad():
            inputs, senti_labels = get_input_from_batch(args, batch)
            inputs['is_eval'] = True
            a_sequence, ea_sequence, iea_sequence = model(**inputs)

            a_pred_sequences.extend(a_sequence)
            ea_pred_sequences.extend(ea_sequence)
            iea_pred_sequences.extend(iea_sequence)
            out_label_ids.extend(senti_labels.detach().cpu().numpy())

    a_preds = parse_sequences(a_pred_sequences)
    ea_preds = parse_sequences(ea_pred_sequences)
    iea_preds = parse_sequences(iea_pred_sequences)

    a_result = compute_metrics(a_preds, out_label_ids)
    ea_result = compute_metrics(ea_preds, out_label_ids)
    iea_result = compute_metrics(iea_preds, out_label_ids)
    
    a_results.update(a_result)
    ea_results.update(ea_result)
    iea_results.update(iea_result)

    results = {'a_results': a_results, 'ea_results': ea_results, 'iea_results': iea_results}
    results['avg_results'] = {}

    # ablation study
    if args.multi_task == 'multi_task':
        results['avg_results']['acc'] = (results['a_results']['acc'] + results['ea_results']['acc'] + results['iea_results']['acc']) / 3
        results['avg_results']['f1'] = (results['a_results']['f1'] + results['ea_results']['f1'] +  results['iea_results']['f1']) / 3
    elif args.multi_task == 'no_iea':
        results['avg_results']['acc'] = (results['a_results']['acc'] + results['ea_results']['acc']) / 2
        results['avg_results']['f1'] = (results['a_results']['f1'] + results['ea_results']['f1'] ) / 2
    elif args.multi_task == 'no_ea':
        results['avg_results']['acc'] = (results['a_results']['acc'] + results['iea_results']['acc']) / 2
        results['avg_results']['f1'] = (results['a_results']['f1'] + results['iea_results']['f1'] ) / 2
    else:
        results['avg_results']['acc'] = results['a_results']['acc']
        results['avg_results']['f1'] = results['a_results']['f1']
    
    output_eval_file = os.path.join(args.save_model_dir, 'eval_results.txt')
    with open(output_eval_file, 'a+') as writer:
        if is_test:
            logger.info('***** Test results *****')
            writer.write('***** Test results *****')
        else:
            logger.info('***** Eval results *****')
            writer.write('***** Eval results *****')
        for type, result in results.items():
            logger.info("the result of %s", type)
            writer.write("#")
            writer.write("the result of %s" % (type))
            writer.write('\n')
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("  %s = %s" % (key, str(result[key])))
                writer.write('\n')
            writer.write('\n')
            
    if is_test:
        pred_data = []
        for a_p, a_s, ea_p, ea_s, iea_p, iea_s, l in zip(a_preds.tolist(), a_pred_sequences, ea_preds.tolist(), ea_pred_sequences, iea_preds.tolist(), iea_pred_sequences, out_label_ids):
            data = {}
            data['a_pred'] = a_p
            data['ea_pred'] = ea_p
            data['iea_pred'] = iea_p
            data['label'] = int(l)
            data['a_sequence'] = a_s
            data['ea_sequence'] = ea_s
            data['iea_sequence'] = iea_s
            pred_data.append(data)

        pred_file = os.path.join(args.save_model_dir, 'pred_results.json')
        with open(pred_file, 'w') as f:
            json.dump(pred_data, f, indent=4, ensure_ascii=False)
    if args.eval_metric == 'avg':
        return results['avg_results']
    elif args.eval_metric == 'a':
        return results['a_results']
    elif args.eval_metric == 'ea':
        return results['ea_results']
    else:
        return results['iea_results']
    
def get_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def get_input_from_batch(args, batch):
    inputs = {'a_input_ids': batch[0].to(args.device),
            'a_attention_mask': batch[1].to(args.device),
            'a_decoder_output_labels': batch[2].to(args.device),
            'ea_input_ids': batch[3].to(args.device),
            'ea_attention_mask': batch[4].to(args.device),
            'ea_decoder_output_labels': batch[5].to(args.device),
            'iea_input_ids': batch[6].to(args.device),
            'iea_attention_mask': batch[7].to(args.device),
            'iea_decoder_output_labels': batch[8].to(args.device),
            'image_feature': batch[9].to(args.device),
            'cap_input_ids': batch[11].to(args.device),
            'cap_attention_mask': batch[12].to(args.device),
            'imgid': batch[13].to(args.device),
            }
    sentiment_labels = batch[10].to(args.device)
    return inputs, sentiment_labels

def save_model(save_dir, model):
     # Save model checkpoint
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_model_path = os.path.join(save_dir, 'model.pth')
    torch.save(model.state_dict(), save_model_path)
    logger.info('Save best model in {}'.format(save_model_path))

def parse_sequences(pred_sequences):
    preds = []
    for seq in pred_sequences:
        seq = seq.lower().replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
        seq = seq.split('<emotion>')[-1]
        if 'negative' in seq:
            pred = 2
        elif 'positive' in seq:
            pred = 1
        else:
            pred = 0
        preds.append(pred)
    return np.array(preds)



