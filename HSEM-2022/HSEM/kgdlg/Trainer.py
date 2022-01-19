import time
import kgdlg.utils.misc_utils as misc_utils
import kgdlg.utils.print_utils as print_utils
import kgdlg.utils.time_utils as time_utils
import torch
from torch.autograd import Variable
from sklearn.cluster import KMeans
import random
import os
import sys
import math


class Statistics(object):
    """
    Train/validate loss statistics.
    """

    def __init__(self, loss=0, memory_loss=0, kld=0, cluster_loss=0, n_words=0, n_correct=0, emo_cls_loss=0,
                 emo_pred_loss=0, pred_correct=0):
        self.loss = loss
        self.memory_loss = memory_loss
        self.loss_per_sample = loss
        self.memory_loss_per_sample = memory_loss
        self.kld = kld
        self.cluster_loss = cluster_loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.emo_cls_loss_per_sample = emo_cls_loss
        self.emo_pred_loss_per_sample = emo_pred_loss
        self.pred_correct_per_sample = pred_correct
        self.pred_correct = pred_correct

    def update(self, stat):
        self.loss += stat.loss
        self.memory_loss += stat.memory_loss
        self.loss_per_sample += stat.loss_per_sample
        self.memory_loss_per_sample += stat.memory_loss_per_sample
        self.cluster_loss += stat.cluster_loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.kld += stat.kld
        self.emo_cls_loss_per_sample += stat.emo_cls_loss_per_sample
        self.emo_pred_loss_per_sample += stat.emo_pred_loss_per_sample
        self.pred_correct_per_sample += stat.pred_correct_per_sample
        self.pred_correct += stat.pred_correct

    def ppl(self):
        return misc_utils.safe_exp(self.loss / self.n_words)

    def memory_ppl(self):
        return misc_utils.safe_exp(self.memory_loss / self.n_words)

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()

        out_info = ("Epoch %2d,%5d/%5d|acc: %6.2f |ppl: %6.2f |auxppl: %6.2f |kld: %6.2f |cluloss: %6.2f |" + \
                    "rec_loss: %6.2f |pred_loss: %6.2f |emo_cls_loss: %6.2f |emo_pred_loss: %6.2f |emo_pred_acc: %6.2f |" + \
                    "%3.0f tgt'tok/s|%4.0f s'elapsed") % \
                   (epoch, batch, n_batches,
                    self.accuracy(),
                    self.ppl(),
                    self.memory_ppl(),
                    self.kld,
                    self.cluster_loss,
                    self.loss_per_sample,
                    self.memory_loss_per_sample,
                    self.emo_cls_loss_per_sample,
                    self.emo_pred_loss_per_sample,
                    self.pred_correct_per_sample,
                    self.n_words / (t + 1e-5),
                    time.time() - self.start_time)

        print(out_info)
        sys.stdout.flush()


class Trainer(object):
    def __init__(self, opt, model, train_iter,
                 train_loss, optim, lr_scheduler):

        self.opt = opt
        self.model = model
        self.train_iter = train_iter
        self.train_loss = train_loss
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.kmeans_tool = KMeans(n_clusters=opt.cluster_num, init="k-means++", \
                                  n_init=10, max_iter=opt.kmeans_max_iter, tol=0.0001, \
                                  precompute_distances="auto", verbose=0, random_state=None, \
                                  copy_x=True, n_jobs=-1, algorithm="auto")
        self.kmeans_model = None

         
        self.model.train()   

        self.global_step = 0
        self.step_epoch = 0

    def order_sequence(self, inputs, lengths):
        idx_len_pair = []
        for i, l in enumerate(lengths):
            idx_len_pair.append((i, l))
        sorted_idx_len_pair = sorted(idx_len_pair, key=lambda x: x[1], reverse=True)
        sorted_idx = []
        sorted_length = []
        for x in sorted_idx_len_pair:
            sorted_idx.append(x[0])
            sorted_length.append(x[1])

        order_inputs = inputs[:, sorted_idx]
        return order_inputs, sorted_length, sorted_idx

    def reorder_enc_hidden(self, inputs, sorted_idx):
        raw_sorted_pair = []
        for raw_idx, sorted_idx in enumerate(sorted_idx):
            raw_sorted_pair.append((raw_idx, sorted_idx))
        sorted_pair = sorted(raw_sorted_pair, key=lambda x: x[1], reverse=False)
        raw_indices = [x[0] for x in sorted_pair]
        reorder_inputs = inputs[:, raw_indices, :]
        return reorder_inputs

    def reorder_enc_seqence(self, inputs, sorted_idx):
        raw_sorted_pair = []
        for raw_idx, sorted_idx in enumerate(sorted_idx):
            raw_sorted_pair.append((raw_idx, sorted_idx))
        sorted_pair = sorted(raw_sorted_pair, key=lambda x: x[1], reverse=False)
        raw_indices = [x[0] for x in sorted_pair]
        reorder_inputs = inputs[:, raw_indices, :]
        return reorder_inputs

    def get_kl_loss_anneal_weight(self):
        cur_kl_weight = 1
        if 0 != self.opt.kl_anneal_total_step:
            if self.global_step > self.opt.kl_anneal_total_step:
                cur_kl_weight = 1   
            adjust_weight_times = 1000
            adjust_weight_per_step = self.opt.kl_anneal_total_step / adjust_weight_times
             
            cur_kl_weight = self.global_step / adjust_weight_per_step / adjust_weight_times
         

         
        if cur_kl_weight > 1:
            cur_kl_weight = 1

        if 0 == self.global_step % self.opt.print_per_step:
            print("global_step:", self.global_step, "current KL loss weight:", cur_kl_weight)

        return cur_kl_weight

    def update(self, batch):   
        self.model.zero_grad()
        src_inputs = batch.src[0]
        src_lengths = batch.src[1].tolist()

         
        tgt_inputs = batch.tgt[0][:-1]
        tgt_lengths = batch.tgt[1] - 1
        tgt_lengths = tgt_lengths.tolist()

         
         
         
         

        src_emo = batch.src_emo   
        tgt_emo = batch.tgt_emo
        outputs, attn, recog_mu, recog_logvar, prior_mu, \
        prior_logvar, P_c_given_x, gmm_loss, cluster_loss, \
        dec_outputs_memory, attn_memory, \
        pred_emo, emo_pred_loss, pred_correct \
            = self.model(src_inputs, src_lengths, tgt_inputs, tgt_lengths, src_emo, tgt_emo)   

        kl_loss_anneal_weight = self.get_kl_loss_anneal_weight()
        stats = self.train_loss.compute_train_loss(batch, outputs, \
                                                   recog_mu, recog_logvar, prior_mu, prior_logvar, \
                                                   gmm_loss, cluster_loss, dec_outputs_memory, kl_loss_anneal_weight,
                                                   emo_pred_loss, pred_correct, pred_emo)

        '''
            cluster_per_step: 100000000 # 100000000(very large): means do not cluster at step level (default: 0)
        '''
        if 0 == self.global_step % self.opt.cluster_per_step and self.opt.cluster_num > 0:
            if self.opt.debug_mode >= 2:
                time_utils.print_time("Start to do Clustering!")
            self.kmeans_model = self.kmeans_tool.fit(self.model.data_cluster.src_samples)

            if self.opt.debug_mode >= 2:
                time_utils.print_time("Clustering finished!")
            self.model.data_cluster.aggregate_cluster_centers(self.kmeans_model)

             
            if self.opt.debug_mode >= 2:
                time_utils.print_time("Postprocess after clustering finished!")

        self.optim.step()
        return stats

    def feedforward_to_get_samples_for_cluster(self, batch):   
        if random.random() > self.opt.training_data_sample_rate_for_cluster:
            return

        self.model.zero_grad()
        src_inputs = batch.src[0]
        src_lengths = batch.src[1].tolist()

         
        tgt_inputs = batch.tgt[0][:-1]
        tgt_lengths = batch.tgt[1] - 1
        tgt_lengths = tgt_lengths.tolist()

        only_get_samples_for_cluster = True
        rets = self.model(src_inputs, src_lengths, tgt_inputs, tgt_lengths, only_get_samples_for_cluster)   

    def train(self, epoch, report_func=None):   
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        if self.opt.debug_mode >= 2:
            time_utils.print_time("Start to train NN epoch:" + str(epoch))

         
        for batch in self.train_iter:
             
             
             
            self.global_step += 1
            step_batch = self.train_iter.iterations
            stats = self.update(batch)   

            report_stats.update(stats)
            total_stats.update(stats)

            if report_func is not None:
                report_stats = report_func(self.global_step,
                                           epoch, step_batch, len(self.train_iter),
                                           total_stats.start_time, self.optim.lr, report_stats)

        self.save_per_epoch(epoch, self.opt.out_dir, 1)

         
        if 0 == self.opt.train_collect_clusterdata_sync:
            for batch in self.train_iter:
                self.feedforward_to_get_samples_for_cluster(batch)
            mark_on_name = "after_collect_before_cluster"
            self.save_per_epoch(epoch, self.opt.out_dir, 1, mark_on_name)

        if self.opt.debug_mode >= 2:
            time_utils.print_time("train NN epoch:" + str(epoch) + " done.")

        if 0 == epoch % self.opt.cluster_per_epoch and self.opt.cluster_num > 0:
            if self.opt.debug_mode >= 2:
                time_utils.print_time("Start to do Clustering!")

            self.kmeans_model = self.kmeans_tool.fit(self.model.data_cluster.src_samples)
            if self.opt.debug_mode >= 2:
                time_utils.print_time("Clustering finished!")

            self.model.data_cluster.aggregate_cluster_centers(self.kmeans_model)
            self.model.data_cluster.clear_samples()
            if self.opt.debug_mode >= 2:
                time_utils.print_time("Postprocess after clustering finished!")

        self.save_per_epoch(epoch, self.opt.out_dir, 2)

        return total_stats

    def save_per_epoch(self, epoch, out_dir, save_part=0, mark_on_name=""):
        '''
            save_part: 0:all, 1:model 2:cluster
        '''

        if save_part in [0, 1]:
            f = open(os.path.join(out_dir, 'checkpoint'), 'w')
            f.write('latest_checkpoint:checkpoint_epoch%d.pkl' % (epoch))
            f.close()

            if self.opt.save_model_mode in [0, 2]:
                if "" == mark_on_name:
                    filename = os.path.join(out_dir, "checkpoint_epoch%d.pkl" % (epoch))
                else:
                    filename = os.path.join(out_dir, "checkpoint_epoch%d.pkl" % (epoch)) + "_" + mark_on_name
                self.model.save_checkpoint(epoch, self.opt, filename)
            if self.opt.save_model_mode in [1, 2]:
                if "" == mark_on_name:
                    filename = os.path.join(out_dir, "checkpoint_epoch_by_layers%d.pkl" % (epoch))
                else:
                    filename = os.path.join(out_dir, "checkpoint_epoch_by_layers%d.pkl" % (epoch)) + "_" + mark_on_name
                self.model.save_checkpoint_by_layers(epoch, self.opt, filename)
        elif save_part in [0, 2]:
            if self.opt.cluster_num > 0:
                self.model.data_cluster.save_cluster_txt(epoch + 1, False, mark_on_name)

    def epoch_step(self, epoch, out_dir):
        """ Called for each epoch to update learning rate. """
         
         
        self.save_per_epoch(epoch, out_dir)
