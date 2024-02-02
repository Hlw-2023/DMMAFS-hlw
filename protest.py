import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

import network
from utils import Dataset_SEQ_BP_t, Dataset_SEQ_MF_t, Dataset_SEQ_CC_t
from utils import My_sampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from itertools import chain
import os

import math
from go_utils import Ontology
from sklearn.metrics import auc, matthews_corrcoef,roc_auc_score,hamming_loss
device = 'cuda:1'
EPSILON = 1e-10
type = 'bp'
speci = 'h'

def get_BMF():
    df = pd.read_csv("/home/HLW/seqcmap/dataset/go_id.csv")
    df = pd.DataFrame(df, columns=['BP', 'MF', 'CC'])
    it = df.itertuples()
    bp = []
    mf = []
    cc = []
    for id, row in enumerate(it):
        bp_go = row.BP
        mf_go = row.MF
        cc_go = row.CC
        bp.append(bp_go)
        if id < 1862:
            mf.append(mf_go)
        if id < 1717:
            cc.append(cc_go)
    return bp, mf, cc
    
def get_BMF_H():
    df = pd.read_csv("/home/HLW/seqcmap/dataset/go_id_h.csv")
    df = pd.DataFrame(df, columns=['BP', 'MF', 'CC'])
    it = df.itertuples()
    bp = []
    mf = []
    cc = []
    for id, row in enumerate(it):
        bp_go = row.BP
        mf_go = row.MF
        cc_go = row.CC
        bp.append(bp_go)
        if id < 692:
            mf.append(mf_go)
        if id < 755:
            cc.append(cc_go)
    return bp, mf, cc

def get_BMF_M():
    df = pd.read_csv("/home/HLW/seqcmap/dataset/go_id_m.csv")
    df = pd.DataFrame(df, columns=['BP', 'MF', 'CC'])
    it = df.itertuples()
    bp = []
    mf = []
    cc = []
    for id, row in enumerate(it):
        bp_go = row.BP
        mf_go = row.MF
        cc_go = row.CC
        bp.append(bp_go)
        if id < 685:
            mf.append(mf_go)
        if id < 728:
            cc.append(cc_go)
    return bp, mf, cc
    
def get_BMF_F():
    df = pd.read_csv("/home/HLW/seqcmap/dataset/go_id_f.csv")
    df = pd.DataFrame(df, columns=['BP', 'MF', 'CC'])
    it = df.itertuples()
    bp = []
    mf = []
    cc = []
    for id, row in enumerate(it):
        bp_go = row.BP
        mf_go = row.MF
        cc_go = row.CC
        bp.append(bp_go)
        if id < 229:
            mf.append(mf_go)
        if id < 209:
            cc.append(cc_go)
    return bp, mf, cc
    
def get_BMF_R():
    df = pd.read_csv("/home/HLW/seqcmap/dataset/go_id_r.csv")
    df = pd.DataFrame(df, columns=['BP', 'MF', 'CC'])
    it = df.itertuples()
    bp = []
    mf = []
    cc = []
    for id, row in enumerate(it):
        bp_go = row.BP
        mf_go = row.MF
        cc_go = row.CC
        bp.append(bp_go)
        if id < 491:
            mf.append(mf_go)
        if id < 535:
            cc.append(cc_go)
    return bp, mf, cc


def get_anno(type):
    if type == 'mf':
        train_path = '/Datasets/hlw/train_s_mf'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/train_s_mf/' + i)
            label_go = npzfile['MF_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
        print(1)
    if type == 'bp':
        train_path = '/Datasets/hlw/train_s_bp'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/train_s_bp/' + i)
            label_go = npzfile['BP_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    if type == 'cc':
        train_path = '/Datasets/hlw/train_s_cc'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/train_s_cc/' + i)
            label_go = npzfile['CC_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    return anno

def get_anno_h(type):
    if type == 'mf':
        train_path = '/Datasets/hlw/h_train_mf'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/h_train_mf/' + i)
            label_go = npzfile['MF_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
        print(1)
    if type == 'bp':
        train_path = '/Datasets/hlw/h_train_bp'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/h_train_bp/' + i)
            label_go = npzfile['BP_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    if type == 'cc':
        train_path = '/Datasets/hlw/h_train_cc'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/h_train_cc/' + i)
            label_go = npzfile['CC_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    return anno

def get_anno_m(type):
    if type == 'mf':
        train_path = '/Datasets/hlw/m_train_mf'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/m_train_mf/' + i)
            label_go = npzfile['MF_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
        print(1)
    if type == 'bp':
        train_path = '/Datasets/hlw/m_train_bp'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/m_train_bp/' + i)
            label_go = npzfile['BP_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    if type == 'cc':
        train_path = '/Datasets/hlw/m_train_cc'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/m_train_cc/' + i)
            label_go = npzfile['CC_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    return anno
    
    
def get_anno_f(type):
    if type == 'mf':
        train_path = '/Datasets/hlw/f_train_mf'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/f_train_mf/' + i)
            label_go = npzfile['MF_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
        print(1)
    if type == 'bp':
        train_path = '/Datasets/hlw/f_train_bp'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/f_train_bp/' + i)
            label_go = npzfile['BP_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    if type == 'cc':
        train_path = '/Datasets/hlw/f_train_cc'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/f_train_cc/' + i)
            label_go = npzfile['CC_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    return anno
    
def get_anno_r(type):
    if type == 'mf':
        train_path = '/Datasets/hlw/r_train_mf'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/r_train_mf/' + i)
            label_go = npzfile['MF_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
        print(1)
    if type == 'bp':
        train_path = '/Datasets/hlw/r_train_bp'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/r_train_bp/' + i)
            label_go = npzfile['BP_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    if type == 'cc':
        train_path = '/Datasets/hlw/r_train_cc'
        filename = os.listdir(train_path)
        anno = []
        for i in filename:
            npzfile = np.load('/Datasets/hlw/r_train_cc/' + i)
            label_go = npzfile['CC_label']
            real_go = []
            for i, val in enumerate(label_go):
                if val != '0':
                    real_go.append(val)
            anno.append(set(real_go))
    return anno
def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = set(pred_annots[i]) - tp
        fn = set(real_annots[i]) - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns

if __name__ == "__main__":


    if type == 'bp':
        npzfile = np.load('/home/HLW/seqcmap/dataset/annos_bp.npz', allow_pickle=True)
        annotations = npzfile['annos']
    if type == 'mf':
        npzfile = np.load('/home/HLW/seqcmap/dataset/annos_mf.npz', allow_pickle=True)
        annotations = npzfile['annos']
    if type == 'cc':
        npzfile = np.load('/home/HLW/seqcmap/dataset/annos_cc.npz', allow_pickle=True)
        annotations = npzfile['annos']
    annotations = list(map(lambda x: set(x), annotations))
    if speci == 'h':
        bp, mf, cc = get_BMF_H()
    if speci == 'm':
        bp, mf, cc = get_BMF_M()
    if speci == 'f':
        bp, mf, cc = get_BMF_F()
    if speci == 'r':
        bp, mf, cc = get_BMF_R()
    if speci == 'all':
        bp, mf, cc = get_BMF()

    fuse_model = network.TransEX().to(device)
    fuse_model.mode = "fuse"
    if speci == 'h':
        if type == 'mf':
            state_dict = torch.load('/home/HLW/TransEx/weights/h_mf/best_fusion.pt')
        if type == 'bp':
            state_dict = torch.load('/home/HLW/TransEx/weights/h_bp/best_fusion.pt')
        if type == 'cc':
            state_dict = torch.load('/home/HLW/TransEx/weights/h_cc/best_fusion.pt')
    if speci == 'm':
        if type == 'mf':
            state_dict = torch.load('/home/HLW/TransEx/weights/m_mf/best_fusion.pt')
        if type == 'bp':
            state_dict = torch.load('/home/HLW/TransEx/weights/m_bp/best_fusion.pt')
        if type == 'cc':
            state_dict = torch.load('/home/HLW/TransEx/weights/m_cc/best_fusion.pt')
    if speci == 'f':
        if type == 'mf':
            state_dict = torch.load('/home/HLW/TransEx/weights/f_mf/best_fusion.pt')
        if type == 'bp':
            state_dict = torch.load('/home/HLW/TransEx/weights/f_bp/best_fusion.pt')
        if type == 'cc':
            state_dict = torch.load('/home/HLW/TransEx/weights/f_cc/best_fusion.pt')
    if speci == 'r':
        if type == 'mf':
            state_dict = torch.load('/home/HLW/TransEx/weights/r_mf/best_fusion.pt')
        if type == 'bp':
            state_dict = torch.load('/home/HLW/TransEx/weights/r_bp/best_fusion.pt')
        if type == 'cc':
            state_dict = torch.load('/home/HLW/TransEx/weights/r_cc/best_fusion.pt')
    if speci == 'all':
        if type == 'mf':
            state_dict = torch.load('/home/HLW/TransEx/weights/all_mf/best_fusion.pt')
        if type == 'bp':
            state_dict = torch.load('/home/HLW/TransEx/weights/all_bp/best_fusion.pt')
        if type == 'cc':
            state_dict = torch.load('/home/HLW/TransEx/weights/all_cc/best_fusion.pt')
    # state_dict = torch.load('weights/best_fusion.pt')
    fuse_model.load_state_dict(state_dict['weight'])
    # fuse_model.to(device)
    fuse_model.eval()
    total = sum([params.nelement() for params in fuse_model.parameters()])
    print("Number of params: {%.2f M}, model epoch : {%d}" % (total / 1e6, state_dict['epoch']))

    if speci == 'h':
        if type == 'bp':
            test_seq_path = '/home/HLW/seqcmap/dataset/h_test_bp/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_BP_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'mf':
            test_seq_path = '/home/HLW/seqcmap/dataset/h_test_mf/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_MF_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'cc':
            test_seq_path = '/home/HLW/seqcmap/dataset/h_test_cc/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_CC_t(test_seq_path, test_cmap_path, gray=True)
    if speci == 'm':
        if type == 'bp':
            test_seq_path = '/home/HLW/seqcmap/dataset/m_test_bp/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_BP_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'mf':
            test_seq_path = '/home/HLW/seqcmap/dataset/m_test_mf/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_MF_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'cc':
            test_seq_path = '/home/HLW/seqcmap/dataset/m_test_cc/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_CC_t(test_seq_path, test_cmap_path, gray=True)
    if speci == 'f':
        if type == 'bp':
            test_seq_path = '/home/HLW/seqcmap/dataset/f_test_bp/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_BP_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'mf':
            test_seq_path = '/home/HLW/seqcmap/dataset/f_test_mf/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_MF_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'cc':
            test_seq_path = '/home/HLW/seqcmap/dataset/f_test_cc/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_CC_t(test_seq_path, test_cmap_path, gray=True)
    if speci == 'r':
        if type == 'bp':
            test_seq_path = '/home/HLW/seqcmap/dataset/r_test_bp/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_BP_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'mf':
            test_seq_path = '/home/HLW/seqcmap/dataset/r_test_mf/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_MF_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'cc':
            test_seq_path = '/home/HLW/seqcmap/dataset/r_test_cc/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_CC_t(test_seq_path, test_cmap_path, gray=True)
    if speci == 'all':
        if type == 'bp':
            test_seq_path = '/home/HLW/seqcmap/dataset/test_s_bp/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_BP_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'mf':
            test_seq_path = '/home/HLW/seqcmap/dataset/test_s_mf/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_MF_t(test_seq_path, test_cmap_path, gray=True)
        if type == 'cc':
            test_seq_path = '/home/HLW/seqcmap/dataset/test_s_cc/'
            test_cmap_path = '/Datasets/hlw/train_c_hmfr/'
            seq_data = Dataset_SEQ_CC_t(test_seq_path, test_cmap_path, gray=True)



    length = int(len(seq_data))
    indices = list(range(length))

    sampler_seq = My_sampler(seq_data, indices)
    seq_loader = DataLoader(seq_data,  sampler=sampler_seq)

    sampler_seq.indices = indices
    tqdms = tqdm(seq_loader, total=len(seq_loader))
    acc_num = 0
    pre_num = 0
    rec_num = 0
    fs_num = 0
    hl_sum = 0
    pred_sum = []
    label_sum = []
    go_sum = []

    with torch.no_grad():
        for seq, label, cmap, l_go in tqdms:

            seq = seq.to(device)
            cmap = cmap.to(device)
            glo_sum = fuse_model.forward_encoder(seq, cmap)
            pred = fuse_model.forward_decoder(glo_sum)
            p_sum = pred.sum(dim=1)

            pred = pred.tolist()
            pred = list(chain.from_iterable(pred))
            label_list = []
            l_go_list = []
            for i, val in enumerate(label):
                val = ''.join(val)
                label_list.append(val)
            label_sum.append(label_list)
            pred_sum.append(pred)
            for i, val in enumerate(l_go):
                val = ''.join(val)
                l_go_list.append(val)
            go_sum.append(l_go_list)
            print(2)



    go_rels = Ontology('/home/HLW/seqcmap/dataset/go.obo', with_rels=True)
    test_annotations = np.array(label_sum)
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    fs = []
    for t in range(1, 101):
        threshold = t / 100.0
        pre_sum = []
        for i, val in enumerate(pred_sum):
            pre = []
            if type == 'bp':
                for j, v in enumerate(bp):
                    if val[j] >= threshold:
                        pre.append(v)
            if type == 'mf':
                for j, v in enumerate(mf):
                    if val[j] >= threshold:
                        pre.append(v)
            if type == 'cc':
                for j, v in enumerate(cc):
                    if val[j] >= threshold:
                        pre.append(v)
            pre_sum.append(pre)

        fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, label_sum, pre_sum)
        fs.append(fscore)
        precisions.append(prec)
        recalls.append(rec)
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            preci = prec
            reca = rec
        if smin > s:
            smin = s
    print('fs:', fs)
    print(f'threshold: {tmax}')
    print(f'Fmax: {fmax:0.3f}')
    print(f'Prec: {preci:0.3f}')
    print(f'Rec: {reca:0.3f}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')



