import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.se_resnet_attention import se_resnet50, get_lr_params
from networks.discriminator import Discriminator
from networks.resnet50off import CNN
from core.trainer import train_model
from utils.utils import get_logger
from utils.altutils import get_mscoco, get_flir
from utils.altutils import get_m3fd
from utils.altutils import setLogger
from utils.analysis import collect_feature, tsne
from sklearn.metrics import accuracy_score
import numpy as np
import logging


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_task_parameters(model, task):
    def ismember(layer_txt, tasks):
        exists = False
        for task in tasks:
            exists = exists or layer_txt.find(task) > 0
            return exists
    return sum(p.numel() for name, p in model.named_parameters() if ismember(name, task) and p.requires_grad)

def run(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'test_cls.log'))
    logger.info(args)

    # data loaders
    dataset_root = './dataset_dir/'
    source_val_loader = get_mscoco(dataset_root, args.batch_size, train=False)
    if args.tgt_cat == 'flir':
        target_train_loader, data_path = get_flir(dataset_root, args.batch_size, train=True, pseudo_label=True)
        target_val_loader = get_flir(dataset_root, args.batch_size, train=False)
    elif args.tgt_cat == 'm3fd':
        target_train_loader, data_path = get_m3fd(dataset_root, args.batch_size, train=True, pseudo_label=True)
        target_val_loader = get_m3fd(dataset_root, args.batch_size, train=False, test=True)
    else:
        raise ValueError("Target dataset {} is not defined.".format(args.tgt_cat))

    args.classInfo = {'classes': torch.unique(torch.tensor(source_val_loader.dataset.targets)),
                      'classNames': source_val_loader.dataset.classes}
    logger.info('cls training')

    # load model
    tasks = ['source', 'target']
    squeeze = True
    adapters = True
    train_norm_layers = True
    model = se_resnet50(n_classes=args.n_classes,
                        pretrained='imagenet',
                        output_stride=8,
                        tasks=tasks,
                        squeeze=squeeze,
                        adapters=adapters,
                        train_norm_layers=train_norm_layers).to(args.device)
    if os.path.isfile(args.trained):
        c = torch.load(args.trained)
        model.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.trained))
    
    # load discriminator
    discriminator = Discriminator(args=args).to(args.device)
    if os.path.isfile(args.d_trained):
        c = torch.load(args.d_trained)
        discriminator.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.d_trained))
    
    # load baselines
    source_cnn = CNN(in_channels=args.in_channels).to(args.device)
    if os.path.isfile(args.s_trained):
        c = torch.load(args.s_trained)
        source_cnn.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.s_trained))
    target_cnn = CNN(in_channels=args.in_channels).to(args.device)
    if os.path.isfile(args.t_trained):
        c = torch.load(args.t_trained)
        target_cnn.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.t_trained))

    # plot t-SNE
    if args.tsne:
        tSNE_filename = os.path.join(args.logdir, 'TSNE.pdf')
        if args.trained:
            target_feature, target_label = collect_feature(target_val_loader, model, args.device, task='target')
        elif args.t_trained:
            target_feature, target_label = collect_feature(target_val_loader, target_cnn.encoder, args.device, task=None)
        tsne.visualize_cls(target_feature, target_label, tSNE_filename)
        if args.s_trained:
            source_feature, source_label = collect_feature(target_val_loader, source_cnn.encoder, args.device, task=None)
            tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)

    # testing
    if args.trained and args.d_trained:
        testing = test(model, discriminator, target_train_loader, task='target', datapath=data_path, args=args) # generate pseudo labels
    elif args.trained:
        testing = test(model, discriminator, target_val_loader, task='target', datapath=None, args=args)
    elif args.t_trained:
        testing = test(target_cnn, discriminator, target_val_loader, task=None, datapath=None, args=args)
    best_acc = testing['avgAcc']
    best_class = testing['classAcc']
    classNames = testing['classNames']

    # log results
    bestClassWiseDict = {}
    for cls_idx, clss in enumerate(classNames):
        bestClassWiseDict[clss] = best_class[cls_idx].item()
    logger.info('Best acc.: {}'.format(best_acc))
    logger.info('Best acc. (Classwise):')
    logger.info(bestClassWiseDict)
    
    return best_acc, bestClassWiseDict

def step(model, data, label, task, args):
    data, label = data.to(args.device), label.to(args.device)
    if task:
        output, feat = model(data, task=task)
    else:
        output = model(data)
        feat = model.encoder(data)
    return output, feat

def test(model, discriminator, dataloader, task, datapath=None, args=None):
    model.eval()
    labels, probas = [], []
    if args.classInfo == None:
        classes = torch.unique(torch.tensor(dataloader.dataset.targets))
        classNames = dataloader.dataset.classes
    else:
        classes = args.classInfo['classes']
        classNames = args.classInfo['classNames']
    class_acc = torch.zeros(len(classes))
    class_len = torch.zeros(len(classes))
    acc_ev = 0
    with torch.no_grad():
        if datapath is not None:
            f = open('pseudo_labels.txt', 'w')
        for iter_i, (data, label) in enumerate(dataloader):
            bs = label.size(0)
            output, d_input = step(model, data, label, task, args)
            pred_cls = output.data.max(1)[1]
            acc_ev += pred_cls.cpu().eq(label.data).cpu().sum()
            for class_idx, class_id in enumerate(classes):
                idxes = torch.nonzero(label==class_id.to(label.device), as_tuple=False)
                class_acc[class_idx] += pred_cls[idxes].cpu().eq(label[idxes].data).cpu().sum()
                class_len[class_idx] += len(idxes)
            output = torch.softmax(output, dim=1)
            if datapath is not None:
                d_output = discriminator(d_input)
                d_pred_cls = d_output.data.max(1)[1]
                d_output = torch.softmax(d_output, dim=1)
                if args.tgt_cat == 'flir':
                    weight = [17.13749324689357, 1.6411775357632512, 3.0090590020868904]
                elif args.tgt_cat == 'm3fd':
                    weight = [57.05215419501134, 1.940010794972627, 13.228180862250262, 65.86387434554973, 2.8688711516533636, 36.14942528735632]
                f.write(datapath[int(iter_i)][0][34:] + ' ' + str(pred_cls[0].cpu().numpy()) + ' ' + str(output[0, pred_cls][0].cpu().numpy()) + ' '
                        + str(d_pred_cls[0].cpu().numpy()) + ' ' + str(d_output[0, d_pred_cls][0].cpu().numpy()) + ' ' + str(weight[label]) + '\n')
            labels.extend(label.cpu().numpy().tolist())
            probas.extend(output.cpu().numpy().tolist())
    probas = np.asarray(probas)
    preds = np.argmax(probas, axis=1)
    acc = accuracy_score(labels, preds)
    class_acc /= class_len
    avgAcc = 0.0
    for i in range(len(class_acc)):
        avgAcc += class_acc[i]
    avgAcc = avgAcc / len(class_acc)
    return {
        'acc': acc, 'avgAcc': avgAcc, 'classAcc': class_acc, 'classNames': classNames,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--d_trained', type=str, default='')
    parser.add_argument('--s_trained', type=str, default='')
    parser.add_argument('--t_trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    # test
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tsne', action="store_true")
    # misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='outputs/')
    # office dataset categories
    parser.add_argument('--src_cat', type=str, default='mscoco')
    parser.add_argument('--tgt_cat', type=str, default='flir')
    parser.add_argument('--message', type=str, default='altinel')  # to track parallel device outputs

    args, unknown = parser.parse_known_args()
    run(args)
