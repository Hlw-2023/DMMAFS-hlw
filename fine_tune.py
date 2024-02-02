# -*- coding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch.utils.data import DataLoader, SequentialSampler
import torch
from tqdm import tqdm
import torchvision
from protest import get_BMF, get_BMF_H, get_BMF_M, get_BMF_F, get_BMF_R
from model import Model
from utils import Dataset, My_sampler
import os
import argparse
from utils import Dataset_SEQ_BP, Dataset_SEQ_MF, Dataset_SEQ_CC
import random
import numpy as np
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine_dataset', default='/Datasets/hlw/', type=str, help='')
    # parser.add_argument('--fine_dataset', default='F://data//vi_ir', type=str, help='')
    parser.add_argument('--load_pt', default=False, type=bool, help='')
    parser.add_argument('--weights_path', default='weights/h_bp/best_fusion.pt', type=str, help='')
    parser.add_argument('--lr', default= 1e-4, type=float, help='')
    parser.add_argument('--devices', default="1", type=str, help='')
    parser.add_argument('--device', default="cuda", type=str, help='')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--epochs', default=5, type=int, help='')
    parser.add_argument('--multiGPU', default=False, type=bool, help='')
    parser.add_argument('--GPUs', default=[0, 1], type=list, help='')
    parser.add_argument('--backends', default=False, type=bool, help='')
    parser.add_argument('--contiguousparams', default=False, type=bool, help='')
    parser.add_argument('--workers', default=0, type=int, help='')
    parser.add_argument('--train_mode', default='fine_tune', type=str, help='')
    parser.add_argument('--type', default='BP', type=str, help='')
    parser.add_argument('--speci', default='human', type=str, help='')
    return parser.parse_args()

def go_01(go, label):
    label_list = []
    for i, val in enumerate(label):
        val = ''.join(val)
        label_list.append(val)
    label_list.sort()
    chars = go
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))
    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1
    embed_x = [vocab_embed[v] for v in label_list]
    label_one_hot = np.array([vocab_one_hot[j, :] for j in embed_x])
    label_01 = np.sum(label_one_hot, axis=0)
    return label_01



if __name__ == "__main__":

    print("===============Initialization===============")
    args = parse_args()
    print(args)
    os.chdir(r'./')
    torch.backends.cudnn.benchmark = args.backends
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    print("epochs", args.epochs, "batch_size", args.batch_size)
    go_type = args.type
    specis = args.speci

    cmap_path = args.fine_dataset + 'train_c_hmfr'
    if specis == 'human':
        if go_type == 'BP':
            seq_path = args.fine_dataset + 'h_train_bp'
            seq_data = Dataset_SEQ_BP(seq_path, cmap_path, gray=True)
        if go_type == 'MF':
            seq_path = args.fine_dataset + 'h_train_mf'
            seq_data = Dataset_SEQ_MF(seq_path, cmap_path, gray=True)
        if go_type == 'CC':
            seq_path = args.fine_dataset + 'h_train_cc'
            seq_data = Dataset_SEQ_CC(seq_path, cmap_path, gray=True)
    if specis =='mouse':
        if go_type == 'BP':
            seq_path = args.fine_dataset + 'm_train_bp'
            seq_data = Dataset_SEQ_BP(seq_path, cmap_path, gray=True)
        if go_type == 'MF':
            seq_path = args.fine_dataset + 'm_train_mf'
            seq_data = Dataset_SEQ_MF(seq_path, cmap_path, gray=True)
        if go_type == 'CC':
            seq_path = args.fine_dataset + 'm_train_cc'
            seq_data = Dataset_SEQ_CC(seq_path, cmap_path, gray=True)
    if specis =='fruit fly':
        if go_type == 'BP':
            seq_path = args.fine_dataset + 'f_train_bp'
            seq_data = Dataset_SEQ_BP(seq_path, cmap_path, gray=True)
        if go_type == 'MF':
            seq_path = args.fine_dataset + 'f_train_mf'
            seq_data = Dataset_SEQ_MF(seq_path, cmap_path, gray=True)
        if go_type == 'CC':
            seq_path = args.fine_dataset + 'f_train_cc'
            seq_data = Dataset_SEQ_CC(seq_path, cmap_path, gray=True)
    if specis =='rat':
        if go_type == 'BP':
            seq_path = args.fine_dataset + 'r_train_bp'
            seq_data = Dataset_SEQ_BP(seq_path, cmap_path, gray=True)
        if go_type == 'MF':
            seq_path = args.fine_dataset + 'r_train_mf'
            seq_data = Dataset_SEQ_MF(seq_path, cmap_path, gray=True)
        if go_type == 'CC':
            seq_path = args.fine_dataset + 'r_train_cc'
            seq_data = Dataset_SEQ_CC(seq_path, cmap_path, gray=True)
    if specis =='all':
        if go_type == 'BP':
            seq_path = args.fine_dataset + 'train_s_bp'
            seq_data = Dataset_SEQ_BP(seq_path, cmap_path, gray=True)
        if go_type == 'MF':
            seq_path = args.fine_dataset + 'train_s_mf'
            seq_data = Dataset_SEQ_MF(seq_path, cmap_path, gray=True)
        if go_type == 'CC':
            seq_path = args.fine_dataset + 'train_s_cc'
            seq_data = Dataset_SEQ_CC(seq_path, cmap_path, gray=True)
    if specis =='human':
        bp, mf, cc = get_BMF_H()
    if specis =='mouse':
        bp, mf, cc = get_BMF_M()
    if specis =='fruit fly':
        bp, mf, cc = get_BMF_F()
    if specis =='rat':
        bp, mf, cc = get_BMF_R()
    if specis =='all':
        bp, mf, cc = get_BMF()

    length = int(len(seq_data))
    indices = list(range(length))



    sampler_seq = My_sampler(seq_data, indices)
    seq_loader = DataLoader(seq_data, batch_size=args.batch_size, num_workers=args.workers, sampler=sampler_seq)

    model = Model(args).to(args.device)

    print("over=======================")

    total = sum([params.nelement() for params in model.fusion.parameters()])
    print("Number of params: {%.2f M}"%(total/1e6))
    start_epoch = model.start_epoch



    loss = torch.zeros(1)


    for epoch in range(start_epoch,args.epochs):
        model.scheduler.step(loss.item())
        random.shuffle(indices)
        sampler_seq.indices = indices
        tqdms = tqdm(seq_loader, total=len(seq_loader))
        num = 0
        for seq, label, cmap in tqdms:
            if go_type == 'BP':
                label = go_01(bp, label)
                label = label.astype(np.float32)
            if go_type == 'MF':
                label = go_01(mf, label)
                label = label.astype(np.float32)
            if go_type == 'CC':
                label = go_01(cc, label)
                label = label.astype(np.float32)

            label = torch.from_numpy(label)
            label = label.unsqueeze(0)


            model.setdata_for_fine(seq, cmap, label=label)
            model.step()
            tqdms.set_description("epoch:%d %s" % (epoch, model.print))
            num += 1
            if num % 1000 == 0:
                if go_type == 'BP':
                    model.save_BP(epoch)
                if go_type == 'MF':
                    model.save_MF(epoch)
                if go_type == 'CC':
                    model.save_CC(epoch)
        if go_type == 'BP':
            model.save_BP(epoch)
        if go_type == 'MF':
            model.save_MF(epoch)
        if go_type == 'CC':
            model.save_CC(epoch)
