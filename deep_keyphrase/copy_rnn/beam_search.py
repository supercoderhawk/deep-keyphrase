# -*- coding: UTF-8 -*-
import torch
from deep_keyphrase.dataloader import (TOKENS, TOKENS_LENS, TOKENS_OOV,
                                       OOV_COUNT, OOV_LIST, EOS_WORD, UNK_WORD)


class BeamSearch(object):
    def __init__(self, model, beam_size, max_target_len, id2vocab, bos_idx, args):
        self.model = model
        self.beam_size = beam_size
        self.id2vocab = id2vocab
        self.max_target_len = max_target_len
        self.bos_idx = bos_idx
        self.target_hidden_size = args.target_hidden_size

    def beam_search(self, src_dict, delimiter=None):
        """
        generate beam search result
        main idea: inference input Batch x beam size, select
        :param src_dict:
        :param delimiter:
        :return:
        """
        oov_list = src_dict[OOV_LIST]
        batch_size = len(src_dict[TOKENS])
        encoder_output_dict = None
        hidden_state = None
        beam_batch_size = self.beam_size * batch_size
        prev_output_tokens = torch.tensor([[self.bos_idx]] * batch_size, dtype=torch.int64)
        decoder_state = torch.zeros(batch_size, self.target_hidden_size)

        if torch.cuda.is_available():
            prev_output_tokens = prev_output_tokens.cuda()
            decoder_state = decoder_state.cuda()

        # get BOS output and repeat to beam size
        model_output = self.model(src_dict=src_dict,
                                  prev_output_tokens=prev_output_tokens,
                                  encoder_output_dict=encoder_output_dict,
                                  prev_decoder_state=decoder_state,
                                  prev_hidden_state=hidden_state)
        decoder_prob, encoder_output_dict, decoder_state, hidden_state = model_output
        prev_best_probs, prev_best_index = torch.topk(decoder_prob, self.beam_size, 1)
        # B*b x TH
        prev_decoder_state = decoder_state.unsqueeze(1).repeat(1, self.beam_size, 1)
        prev_decoder_state = prev_decoder_state.view(beam_batch_size, -1)
        hidden_state[0] = hidden_state[0].unsqueeze(2).repeat(1, 1, self.beam_size, 1)
        hidden_state[0] = hidden_state[0].view(-1, beam_batch_size, hidden_state[0].size(-1))
        hidden_state[1] = hidden_state[1].unsqueeze(2).repeat(1, 1, self.beam_size, 1)
        hidden_state[1] = hidden_state[1].view(-1, beam_batch_size, hidden_state[1].size(-1))
        prev_hidden_state = hidden_state

        result_sequences = prev_best_index.unsqueeze(2)
        encoder_output_dict = self.expand_encoder_output(encoder_output_dict, batch_size)
        # beam search best probability, its the opposite number of real log probability
        # B x b
        beam_search_best_probs = torch.abs(prev_best_probs)

        for k in [TOKENS, TOKENS_OOV]:
            src_dict[k] = src_dict[k].unsqueeze(1).repeat(1, self.beam_size, 1).reshape(beam_batch_size, -1)

        for k in [TOKENS_LENS, OOV_COUNT]:
            src_dict[k] = src_dict[k].unsqueeze(1).repeat(1, self.beam_size).flatten()

        for target_idx in range(1, self.max_target_len):
            model_output = self.model(src_dict=src_dict,
                                      prev_output_tokens=prev_best_index.view(-1, 1),
                                      encoder_output_dict=encoder_output_dict,
                                      prev_decoder_state=prev_decoder_state,
                                      prev_hidden_state=prev_hidden_state)
            decoder_prob, encoder_output_dict, decoder_state, prev_hidden_state = model_output
            # accumulated_probs: B x b*V
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

            prev_decoder_state = decoder_state.index_select(0, state_select_idx)
            prev_best_index = top_token_index % decoder_prob.size(1)
            prev_hidden_state[0] = prev_hidden_state[0].index_select(1, state_select_idx)
            prev_hidden_state[1] = prev_hidden_state[1].index_select(1, state_select_idx)

            result_sequences = result_sequences.view(batch_size * self.beam_size, -1)
            result_sequences = result_sequences.index_select(0, state_select_idx)
            result_sequences = result_sequences.view(batch_size, self.beam_size, -1)

            result_sequences = torch.cat([result_sequences, prev_best_index.unsqueeze(2)], dim=2)
            prev_best_index = prev_best_index.view(beam_batch_size, -1)

        if torch.cuda.is_available():
            result_sequences = result_sequences.cpu().numpy().tolist()
        else:
            result_sequences = result_sequences.numpy().tolist()
        for k in [TOKENS, TOKENS_OOV, TOKENS_LENS, OOV_COUNT]:
            src_dict[k] = src_dict[k].narrow(0, 0, batch_size)

        results = self.__idx2result_beam(delimiter, oov_list, result_sequences)
        return results

    def __idx2result_beam(self, delimiter, oov_list, result_sequences):
        results = []
        for batch in result_sequences:
            beam_list = []
            for beam in batch:
                phrase = []
                for idx in beam:
                    if self.id2vocab.get(idx) == EOS_WORD:
                        break
                    if idx in self.id2vocab:
                        phrase.append(self.id2vocab[idx])
                    else:
                        oov_idx = idx - len(self.id2vocab)
                        if oov_idx < len(oov_list):
                            phrase.append(oov_list[oov_idx])
                        else:
                            phrase.append(UNK_WORD)

                if delimiter is not None:
                    phrase = delimiter.join(phrase)
                if phrase not in beam_list:
                    beam_list.append(phrase)
            results.append(beam_list)
        return results

    def expand_encoder_output(self, encoder_output_dict, batch_size):
        beam_batch_size = batch_size * self.beam_size
        encoder_output = encoder_output_dict['encoder_output']
        encoder_mask = encoder_output_dict['encoder_padding_mask']
        encoder_hidden_state = encoder_output_dict['encoder_hidden']
        max_len = encoder_output.size(-2)
        hidden_size = encoder_hidden_state[0].size(-1)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
        encoder_output = encoder_output.reshape(beam_batch_size, max_len, -1)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(1, self.beam_size, 1)
        encoder_mask = encoder_mask.reshape(beam_batch_size, -1)
        encoder_hidden_state0 = encoder_hidden_state[0].unsqueeze(2).repeat(1, 1, self.beam_size, 1)
        encoder_hidden_state0 = encoder_hidden_state0.reshape(-1, beam_batch_size, hidden_size)
        encoder_hidden_state1 = encoder_hidden_state[1].unsqueeze(2).repeat(1, 1, self.beam_size, 1)
        encoder_hidden_state1 = encoder_hidden_state1.reshape(-1, beam_batch_size, hidden_size)
        encoder_output_dict['encoder_output'] = encoder_output
        encoder_output_dict['encoder_padding_mask'] = encoder_mask
        encoder_output_dict['encoder_hidden'] = [encoder_hidden_state0, encoder_hidden_state1]
        return encoder_output_dict

    def greedy_search(self, src_dict, delimiter=None):
        """

        :param src_dict:
        :param delimiter:
        :return:
        """
        oov_list = src_dict[OOV_LIST]
        batch_size = len(src_dict[TOKENS])
        encoder_output_dict = None
        hidden_state = None
        prev_output_tokens = [[self.bos_idx]] * batch_size
        decoder_state = torch.zeros(batch_size, self.model.decoder.target_hidden_size)
        result_seqs = None

        for target_idx in range(self.max_target_len):
            model_output = self.model(src_dict=src_dict,
                                      prev_output_tokens=prev_output_tokens,
                                      encoder_output_dict=encoder_output_dict,
                                      prev_decoder_state=decoder_state,
                                      prev_hidden_state=hidden_state)
            decoder_prob, encoder_output_dict, decoder_state, hidden_state = model_output
            best_probs, best_indices = torch.topk(decoder_prob, 1, dim=1)
            if result_seqs is None:
                result_seqs = best_indices
            else:
                result_seqs = torch.cat([result_seqs, best_indices], dim=1)
            prev_output_tokens = result_seqs[:, -1].unsqueeze(1)
        result = self.__idx2result_greedy(delimiter, oov_list, result_seqs)

        return result

    def __idx2result_greedy(self, delimiter, oov_list, result_seqs):
        result = []
        for batch in result_seqs.numpy().tolist():
            phrase = []
            for idx in batch:
                if self.id2vocab.get(idx) == EOS_WORD:
                    break
                if idx in self.id2vocab:
                    phrase.append(self.id2vocab[idx])
                else:
                    oov_idx = idx - len(self.id2vocab)
                    if oov_idx < len(oov_list):
                        phrase.append(oov_list[oov_idx])
                    else:
                        phrase.append(UNK_WORD)
            if delimiter is not None:
                phrase = delimiter.join(phrase)
            result.append(phrase)
        return result
