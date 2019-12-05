# -*- coding: UTF-8 -*-
import time
import traceback
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pysenal import write_json, get_logger
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.dataloader import KeyphraseDataLoader
from deep_keyphrase.evaluation import KeyphraseEvaluator
from deep_keyphrase.utils.constants import PAD_WORD


class BaseTrainer(object):
    def __init__(self, args, model):
        torch.manual_seed(0)
        self.args = args
        self.vocab2id = load_vocab(self.args.vocab_path, self.args.vocab_size)

        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        if args.train_parallel:
            self.model = nn.DataParallel(self.model)
        self.loss_func = nn.NLLLoss(ignore_index=self.vocab2id[PAD_WORD])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   self.args.schedule_step,
                                                   self.args.schedule_gamma)
        self.logger = get_logger('train')
        self.train_loader = KeyphraseDataLoader(self.args.train_filename,
                                                self.vocab2id,
                                                self.args.batch_size,
                                                self.args.max_src_len,
                                                self.args.max_oov_count,
                                                self.args.max_target_len,
                                                'train',
                                                pre_fetch=True,
                                                token_field=args.token_field,
                                                keyphrase_field=args.keyphrase_field)
        if self.args.train_from:
            self.dest_dir = os.path.dirname(self.args.train_from) + '/'
        else:
            timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
            self.dest_dir = os.path.join(self.args.dest_base_dir, self.args.exp_name + '-' + timemark) + '/'
            os.mkdir(self.dest_dir)

        fh = logging.FileHandler(os.path.join(self.dest_dir, args.logfile))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        self.logger.addHandler(fh)

        if not self.args.tensorboard_dir:
            tensorboard_dir = self.dest_dir + 'logs/'
        else:
            tensorboard_dir = self.args.tensorboard_dir
        self.writer = SummaryWriter(tensorboard_dir)
        self.eval_topn = (5, 10)
        self.macro_evaluator = KeyphraseEvaluator(self.eval_topn, 'macro')
        self.micro_evaluator = KeyphraseEvaluator(self.eval_topn, 'micro')
        self.best_f1 = None
        self.best_step = 0
        self.not_update_count = 0

    def parse_args(self):
        raise NotImplementedError('build_parser is not implemented')

    def train(self):
        step = 0
        is_stop = False
        if self.args.train_from:
            step = self.args.step
            self.logger.info('train from destination dir:{}'.format(self.dest_dir))
            self.logger.info('train from step {}'.format(step))
        else:
            self.logger.info('destination dir:{}'.format(self.dest_dir))
        for epoch in range(1, self.args.epochs + 1):
            for batch_idx, batch in enumerate(self.train_loader):
                self.model.train()
                try:
                    loss = self.train_batch(batch)
                except Exception as e:
                    err_stack = traceback.format_exc()
                    self.logger.error(err_stack)
                    loss = 0.0
                step += 1
                self.writer.add_scalar('loss', loss, step)
                del loss
                if step and step % self.args.save_model_step == 0:
                    torch.cuda.empty_cache()
                    self.evaluate_and_save_model(step, epoch)
                    if self.not_update_count >= self.args.early_stop_tolerance:
                        is_stop = True
                        break
            if is_stop:
                self.logger.info('best step {}'.format(self.best_step))
                break

    def train_batch(self, batch):
        raise NotImplementedError('train method is not implemented')

    def evaluate_and_save_model(self, step, epoch):
        valid_f1 = self.evaluate(step)
        if self.best_f1 is None:
            self.best_f1 = valid_f1
            self.best_step = step
        elif valid_f1 >= self.best_f1:
            self.best_f1 = valid_f1
            self.not_update_count = 0
            self.best_step = step
        else:
            self.not_update_count += 1
        exp_name = self.args.exp_name
        model_basename = self.dest_dir + '{}_epoch_{}_batch_{}'.format(exp_name, epoch, step)
        torch.save(self.model.state_dict(), model_basename + '.model')
        write_json(model_basename + '.json', vars(self.args))
        score_msg_tmpl = 'best score: step {} macro f1@{} {:.4f}'
        self.logger.info(score_msg_tmpl.format(self.best_step, self.eval_topn[-1], self.best_f1))
        self.logger.info('epoch {} step {}, model saved'.format(epoch, step))

    def evaluate(self, step):
        raise NotImplementedError('evaluate method is not implemented')

    def get_basename(self, filename):
        return os.path.splitext(os.path.basename(filename))[0]
