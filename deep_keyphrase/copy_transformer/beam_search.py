# -*- coding: UTF-8 -*-
import torch
from deep_keyphrase.dataloader import (OOV_LIST, TOKENS, EOS_WORD, TOKENS_OOV)


class TransformerBeamSearch(object):
    def __init__(self, model, beam_size, max_target_len, id2vocab, bos_idx, args):
        self.model = model
        self.beam_size = beam_size
        self.id2vocab = id2vocab
        self.max_target_len = max_target_len
        self.bos_idx = bos_idx
        self.target_hidden_size = args.target_hidden_size
        self.bidirectional = args.bidirectional
        self.input_dim = args.input_dim

    def beam_search(self, src_dict, delimiter=''):
        batch_size = len(src_dict[TOKENS])
        beam_batch_size = batch_size * self.beam_size
        encoder_output = encoder_mask = None
        prev_copy_state = None
        prev_decoder_state = torch.zeros(batch_size, )
        prev_output_tokens = torch.tensor([[self.bos_idx]] * batch_size)

        output = self.model(src_dict=src_dict,
                            prev_output_tokens=prev_output_tokens,
                            encoder_output=encoder_output,
                            encoder_mask=encoder_mask,
                            prev_decoder_state=prev_decoder_state,
                            position=0,
                            prev_copy_state=prev_copy_state)
        decoder_prob, prev_decoder_state, prev_copy_state, encoder_output, encoder_mask = output
        prev_decoder_state = self.beam_repeat(prev_decoder_state)
        prev_copy_state = self.beam_repeat(prev_copy_state)
        encoder_output = self.beam_repeat(encoder_output)
        encoder_mask = self.beam_repeat(encoder_mask)
        src_dict[TOKENS] = self.beam_repeat(src_dict[TOKENS])
        src_dict[TOKENS_OOV] = self.beam_repeat(src_dict[TOKENS_OOV])
        prev_best_probs, prev_best_index = torch.topk(decoder_prob, self.beam_size, 1)
        beam_search_best_probs = torch.abs(prev_best_probs)
        result_sequences = prev_best_index.unsqueeze(2)

        for target_idx in range(1, self.max_target_len):
            output = self.model(src_dict=src_dict,
                                prev_output_tokens=prev_output_tokens,
                                encoder_output=encoder_output,
                                encoder_mask=encoder_mask,
                                prev_decoder_state=prev_decoder_state,
                                position=target_idx,
                                prev_copy_state=prev_copy_state)
            decoder_prob, decoder_state, copy_state, encoder_output, encoder_mask = output
            accumulated_probs = beam_search_best_probs.view(beam_batch_size, -1)
            accumulated_probs = accumulated_probs.repeat(1, decoder_prob.size(1))
            accumulated_probs += torch.abs(decoder_prob)
            accumulated_probs = accumulated_probs.view(batch_size, -1)
            top_token_probs, top_token_index = torch.topk(-accumulated_probs, self.beam_size, 1)
            beam_search_best_probs = -top_token_probs

            select_idx_factor = torch.tensor(range(batch_size)) * self.beam_size
            select_idx_factor = select_idx_factor.unsqueeze(1).repeat(1, self.beam_size)
            if torch.cuda.is_available():
                select_idx_factor = select_idx_factor.cuda()
            state_select_idx = top_token_index.flatten() // decoder_prob.size(1)
            state_select_idx += select_idx_factor.flatten()

            prev_decoder_state = prev_decoder_state.index_select(0, state_select_idx)
            prev_copy_state = prev_copy_state.index_select(0, state_select_idx)
            prev_output_tokens = prev_output_tokens.index_select(0, state_select_idx)

            prev_best_index = top_token_index % decoder_prob.size(1)
            result_sequences = result_sequences.view(beam_batch_size, -1)
            result_sequences = result_sequences.index_select(0, state_select_idx)
            result_sequences = result_sequences.view(batch_size, self.beam_size, -1)

            result_sequences = torch.cat([result_sequences, prev_best_index.unsqueeze(2)], dim=2)
            prev_best_index = prev_best_index.view(beam_batch_size, -1)
            prev_output_tokens = torch.cat([prev_output_tokens, prev_best_index.unsqueeze(2)], dim=2)
        result = self.__idx2result_beam(delimiter, src_dict[OOV_LIST], result_sequences.tolist())
        return result

    def beam_repeat(self, t):
        size = list(t.size())
        size[0] *= self.beam_size
        repeat_size = [1] * len(size)
        repeat_size[1] *= self.beam_size
        t = t.unsqueese(1).repeat(repeat_size)
        t = t.reshape(size)
        return t

    def greedy_search(self, src_dict, delimiter=''):
        batch_size = len(src_dict[TOKENS])
        encoder_output = encoder_mask = None
        prev_copy_state = None
        prev_decoder_state = torch.zeros(batch_size, self.input_dim)
        prev_output_tokens = torch.tensor([[self.bos_idx]] * batch_size)
        result_matrix = None
        for target_idx in range(self.max_target_len):
            output = self.model(src_dict=src_dict,
                                prev_output_tokens=prev_output_tokens,
                                encoder_output=encoder_output,
                                encoder_mask=encoder_mask,
                                prev_decoder_state=prev_decoder_state,
                                position=target_idx,
                                prev_copy_state=prev_copy_state)
            total_prob, prev_decoder_state, prev_copy_state, encoder_output, encoder_mask = output
            prev_output_tokens = total_prob.topk(k=1, dim=1).clone()
            if result_matrix is None:
                result_matrix = prev_output_tokens
            else:
                result_matrix = torch.cat([result_matrix, prev_output_tokens], dim=1)
        result = self.__idx2result_beam(delimiter, src_dict[OOV_LIST], result_matrix.tolist())
        return result

    def __idx2result_beam(self, delimiter, oov_list, result_sequences):
        results = []
        for batch in result_sequences:
            beam_list = []
            for beam in batch:
                phrase = []
                for idx in beam:
                    if self.id2vocab[idx] == EOS_WORD:
                        break
                    if idx in self.id2vocab:
                        phrase.append(self.id2vocab[idx])
                    else:
                        phrase.append(oov_list[idx - len(self.id2vocab)])

                if delimiter is not None:
                    phrase = delimiter.join(phrase)
                if phrase not in beam_list:
                    beam_list.append(phrase)
            results.append(beam_list)
        return results
