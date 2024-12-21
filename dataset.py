import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import read_json



class Twitter_Dataset_FlanT5(Dataset):
    def __init__(self, args, split):

        self.args = args
        self.data_path = os.path.join(args.data_dir, args.dataset_name)
        # self.captions_path = os.path.join(self.data_path, 'captions.json')
        self.images_feature_path = os.path.join(self.data_path, 'images_feature')

        # self.caption_data = self.get_captions(self.captions_path)

        if split == 'train':
            self.data_set = json.load(
                open(self.data_path + '/train_cause_merge_facial.json', 'r'))
        elif split == 'dev':
            self.data_set = json.load(
                open(self.data_path + '/dev_cause_merge_facial.json', 'r'))
        elif split == 'test':
            self.data_set = json.load(
                open(self.data_path + '/test_cause_merge_facial.json', 'r'))
        else:
            raise RuntimeError("split type is not exist!!!")

    def __len__(self):
        return len(self.data_set)

    def get_captions(self, data_path):
        data = read_json(data_path)
        return data

    def get_image_feature(self, image_id):
        image_feature = np.load(os.path.join(self.images_feature_path, image_id[:-4] + '.npz'))['embedding']
        return image_feature


    def get_input_sentence(self, sentence, aspect, caption, facial, object_aesthetic):
        # print()
        if facial == "None" and object_aesthetic == "None":

            a_input_sentence = "qa: aesthetic description: {} ".format(caption) + "sentence: {} aspect: {}".format(sentence, aspect)
            ea_input_sentence = "qea: aesthetic description: {} ".format(caption) + "sentence: {} aspect: {}.".format(sentence, aspect)
            iea_input_sentence = "qiea: aesthetic description: {} ".format(caption) + "sentence: {} aspect: {}.".format(sentence, aspect)
        elif facial != "None" and object_aesthetic == "None":
            a_input_sentence = "qa: sentence: {}".format(sentence) + "aspect facial description: {}".format(facial) + "aspect: {}".format(aspect)
            ea_input_sentence = "qea: sentence: {}".format(sentence) + "aspect facial description: {}".format(facial) + "aspect: {}".format(aspect)
            iea_input_sentence = "qiea: sentence: {}".format(sentence) + "aspect facial description: {}".format(facial) + "aspect: {}".format(aspect)  

        elif facial == "None" and object_aesthetic != "None":
            a_input_sentence = "qa: sentence: {}".format(sentence) + "aspect aesthetic description: {}".format(object_aesthetic) + "aspect: {}".format(aspect)
            ea_input_sentence = "qea: sentence: {}".format(sentence) + "aspect aesthetic description: {}".format(object_aesthetic) + "aspect: {}".format(aspect)
            iea_input_sentence = "qiea: sentence: {}".format(sentence) + "aspect aesthetic description: {}".format(object_aesthetic) + "aspect: {}".format(aspect)

        return a_input_sentence, ea_input_sentence, iea_input_sentence


    def get_output_sentence(self, label, explanation, i_explanation):
        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        sentiment = sentiment_map[str(label)]
        a_output_sentence = '<emotion>{}</emotion>'.format(sentiment)
        ea_output_sentence = '<explain>{}</explain><emotion>{}</emotion>'.format(explanation, sentiment)
        iea_output_sentence = '<impression>{}</impression><emotion>{}</emotion>'.format(i_explanation, sentiment)
        return a_output_sentence, ea_output_sentence, iea_output_sentence


    def __getitem__(self, index):
        data = self.data_set[index]
        image_id = data['image']
        sentiment_label = data['label']
        # captions_scene = self.caption_data[image_id]  
        captions = data['aesthetic']  
        image_feature= self.get_image_feature(image_id)  # np (1,196,768)
        a_input_sentence, ea_input_sentence, iea_input_sentence = self.get_input_sentence(data['sentence'], data['aspect'], captions, data['facial_caption'], data['object_aesthetic'])
        a_output_sentence, ea_output_sentence, iea_output_sentence = self.get_output_sentence(sentiment_label, data['gpt_response'], data['gpt_impression'])
        
        a_input_tokens = self.args.tokenizer.tokenize(a_input_sentence)
        a_input_ids = self.args.tokenizer.convert_tokens_to_ids(a_input_tokens)
        ea_input_tokens = self.args.tokenizer.tokenize(ea_input_sentence)
        ea_input_ids = self.args.tokenizer.convert_tokens_to_ids(ea_input_tokens)
        iea_input_tokens = self.args.tokenizer.tokenize(iea_input_sentence)
        iea_input_ids = self.args.tokenizer.convert_tokens_to_ids(iea_input_tokens)

        a_output_tokens = self.args.tokenizer.tokenize(a_output_sentence)
        a_output_ids = [self.args.tokenizer.pad_token_id] + self.args.tokenizer.convert_tokens_to_ids(a_output_tokens)
        a_output_labels = a_output_ids[1:] + [self.args.tokenizer.eos_token_id]

        ea_output_tokens = self.args.tokenizer.tokenize(ea_output_sentence)
        ea_output_ids = [self.args.tokenizer.pad_token_id] + self.args.tokenizer.convert_tokens_to_ids(ea_output_tokens)
        ea_output_labels = ea_output_ids[1:] + [self.args.tokenizer.eos_token_id]

        iea_output_tokens = self.args.tokenizer.tokenize(iea_output_sentence)
        iea_output_ids = [self.args.tokenizer.pad_token_id] + self.args.tokenizer.convert_tokens_to_ids(iea_output_tokens)
        iea_output_labels = iea_output_ids[1:] + [self.args.tokenizer.eos_token_id]

        a_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(a_input_ids)  # X </s>
        ea_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(ea_input_ids)  # X </s>
        iea_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(iea_input_ids)  # X </s>



        a_attention_mask = [1] * (len(a_input_ids))
        ea_attention_mask = [1] * (len(ea_input_ids))
        iea_attention_mask = [1] * (len(iea_input_ids))

        cap_input_tokens = self.args.tokenizer.tokenize(data['aspect'])
        cap_input_ids = self.args.tokenizer.convert_tokens_to_ids(cap_input_tokens)
        cap_attention_mask = [1] * (len(cap_input_ids))

        imgid = image_id.replace('.jpg', '')
        imgid = torch.tensor([hash(imgid)])

        return (torch.tensor(a_input_ids), torch.tensor(a_attention_mask), torch.tensor(a_output_labels),
                torch.tensor(ea_input_ids), torch.tensor(ea_attention_mask), torch.tensor(ea_output_labels),
                torch.tensor(iea_input_ids), torch.tensor(iea_attention_mask), torch.tensor(iea_output_labels),
                torch.from_numpy(image_feature), torch.tensor(sentiment_label), torch.tensor(cap_input_ids), torch.tensor(cap_attention_mask), imgid)



def collate_fn_bart(batch):
    '''
    Pad sentence a batch.
    Turn all into tensors.
    '''
    a_input_ids, a_attention_mask, cls_indexer, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature, sentiment_labels, cap_input_ids, cap_len = zip(*batch)

    a_input_ids = pad_sequence(a_input_ids, batch_first=True, padding_value=1)
    cls_indexer = torch.tensor(cls_indexer)
    ea_input_ids = pad_sequence(ea_input_ids, batch_first=True, padding_value=1)
    ea_output_labels = pad_sequence(ea_output_labels, batch_first=True, padding_value=-100)
    iea_input_ids = pad_sequence(iea_input_ids, batch_first=True, padding_value=1)
    iea_output_labels = pad_sequence(iea_output_labels, batch_first=True, padding_value=-100)
    
    image_feature = pad_sequence(image_feature, batch_first=True, padding_value=0)
     
    a_attention_mask = pad_sequence(a_attention_mask, batch_first=True, padding_value=0)
    ea_attention_mask= pad_sequence(ea_attention_mask, batch_first=True, padding_value=0)
    iea_attention_mask= pad_sequence(iea_attention_mask, batch_first=True, padding_value=0)
    
    sentiment_labels = torch.tensor(sentiment_labels)

    return a_input_ids, a_attention_mask, cls_indexer, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature, sentiment_labels


def collate_fn_flant5(batch):
    '''
    Pad sentence a batch.
    Turn all into tensors.
    '''
    a_input_ids, a_attention_mask, a_output_labels, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature, sentiment_labels, cap_input_ids, cap_attention_mask, imgid= zip(*batch)

    a_input_ids = pad_sequence(a_input_ids, batch_first=True, padding_value=0)
    a_output_labels = pad_sequence(a_output_labels, batch_first=True, padding_value=-100)
    ea_input_ids = pad_sequence(ea_input_ids, batch_first=True, padding_value=0)
    ea_output_labels = pad_sequence(ea_output_labels, batch_first=True, padding_value=-100)
    iea_input_ids = pad_sequence(iea_input_ids, batch_first=True, padding_value=0)
    iea_output_labels = pad_sequence(iea_output_labels, batch_first=True, padding_value=-100)

    image_feature = pad_sequence(image_feature, batch_first=True, padding_value=0)

    a_attention_mask = pad_sequence(a_attention_mask, batch_first=True, padding_value=0)
    ea_attention_mask = pad_sequence(ea_attention_mask, batch_first=True, padding_value=0)
    iea_attention_mask = pad_sequence(iea_attention_mask, batch_first=True, padding_value=0)

    sentiment_labels = torch.tensor(sentiment_labels)
    cap_input_ids = pad_sequence(a_input_ids, batch_first=True, padding_value=0)
    cap_attention_mask = pad_sequence(cap_attention_mask, batch_first=True, padding_value=0)
    imgid = torch.tensor(imgid)
    return a_input_ids, a_attention_mask, a_output_labels, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature, sentiment_labels, cap_input_ids, cap_attention_mask, imgid
     





