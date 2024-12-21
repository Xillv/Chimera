import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, GenerationConfig, T5ForConditionalGeneration
from lib.cross_net import CrossSparseAggrNet_v2
from torch.nn.utils.rnn import pad_sequence
from lib.loss import loss_select



        
class Chimera(nn.Module):
    def __init__(self, args) -> None:
        super(Chimera, self).__init__()
        self.args = args

        self.t5 = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_dir)
        self.t5.resize_token_embeddings(len(args.tokenizer))
        self.text_embeddings = self.t5.get_input_embeddings()

        self.img_fc = nn.Linear(args.img_hidden_size, args.hidden_size)

        self.a_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_config_dir, 'a_generation_config.json')
        self.ea_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_config_dir, 'ea_generation_config.json')
        self.iea_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_config_dir, 'iea_generation_config.json')

        self.cross_net = CrossSparseAggrNet_v2(args)
        self.criterion = loss_select(args, loss_type=args.loss)

        self.beta = args.beta

    def forward(self, a_input_ids, a_attention_mask, a_decoder_output_labels, ea_input_ids, ea_attention_mask, ea_decoder_output_labels, iea_input_ids,
                iea_attention_mask, iea_decoder_output_labels, image_feature, cap_input_ids, cap_attention_mask, imgid, is_eval=False):
        img_feat = self.img_fc(image_feature)
        cap_encoder_inputs_embeds = self.text_embeddings(cap_input_ids)
        cap_attention_mask_ones = cap_attention_mask[cap_attention_mask == 1]

        improved_sims, score_mask_all, select_tokens = self.cross_net(img_feat, cap_encoder_inputs_embeds, cap_attention_mask_ones)


        a_encoder_inputs_embeds = self.text_embeddings(a_input_ids)  # (B, L, H)

        ea_encoder_inputs_embeds = self.text_embeddings(ea_input_ids)  # (B, L, H)


        iea_encoder_inputs_embeds = self.text_embeddings(iea_input_ids)  # (B, L, H)



        if not is_eval: 
            a_t5_output = self.t5(inputs_embeds=a_encoder_inputs_embeds, attention_mask=a_attention_mask, labels=a_decoder_output_labels)
            a_loss = a_t5_output.loss

            ea_t5_output = self.t5(inputs_embeds=ea_encoder_inputs_embeds, attention_mask=ea_attention_mask, labels=ea_decoder_output_labels)
            ea_loss = ea_t5_output.loss

            iea_t5_output = self.t5(inputs_embeds=iea_encoder_inputs_embeds, attention_mask=iea_attention_mask, labels=iea_decoder_output_labels)
            iea_loss = iea_t5_output.loss

            align_loss = self.criterion(img_feat, cap_encoder_inputs_embeds, imgid, improved_sims)
            a_loss = a_loss + align_loss * self.beta

            return a_loss, ea_loss, iea_loss

        else:
            a_sequence_ids = self.t5.generate(inputs_embeds=a_encoder_inputs_embeds, attention_mask=a_attention_mask, generation_config=self.a_generation_config)
            ea_sequence_ids = self.t5.generate(inputs_embeds=ea_encoder_inputs_embeds, attention_mask=ea_attention_mask, generation_config=self.ea_generation_config)
            iea_sequence_ids = self.t5.generate(inputs_embeds=iea_encoder_inputs_embeds, attention_mask=iea_attention_mask, generation_config=self.iea_generation_config)
            a_sequence = self.args.tokenizer.batch_decode(a_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ea_sequence = self.args.tokenizer.batch_decode(ea_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            iea_sequence = self.args.tokenizer.batch_decode(iea_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            return a_sequence, ea_sequence, iea_sequence


