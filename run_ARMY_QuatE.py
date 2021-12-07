import config
from  models import *
import json
import os
import numpy as np
import random
import torch
import argparse

def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    con = config.Config()
    con.set_in_path(args.dataset)
    con.set_work_threads(8)
    con.set_train_times(15000)
    con.set_nbatches(10)
    con.set_alpha(0.1)
    con.set_bern(0)
    con.set_dimension(100)
    con.set_lmbda(0.2)
    con.set_margin(1.0)
    con.set_ent_neg_rate(10)
    con.set_rel_neg_rate(0)
    con.set_opt_method("adagrad")
    con.set_save_steps(3000)
    con.set_valid_steps(3000)
    con.set_early_stopping_patience(10)
    con.set_checkpoint_dir("")
    con.set_result_dir("")

    print('Hidden = %6d' % con.hidden_size)
    print('Margin = %6.3f' % con.margin)
    print('Optim  = %6s' % con.opt_method)
    print('Patient= %6d' % con.early_stopping_patience)
    print('NBatch = %6d' % con.nbatches)
    print('P_norm = %6d' % con.p_norm)
    print('LrDecay= %6.3f' % con.lr_decay)
    print('WDecay = %6.3f' % con.weight_decay)
    print('Lambda = %6.3f' % con.lmbda)
    print('Alpha  = %6.3f\n' % con.alpha, flush=True)

    con.set_test_link(True)
    con.set_test_triple(True)
    con.init()
    con.set_train_model(QuatE)
    con.train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='transE_argument')
	parser.add_argument('--dataset', type=str, default="./benchmarks/army_clean_all_shuffle_small/", help='dataset directory')
	args = parser.parse_args()
	main(args)

