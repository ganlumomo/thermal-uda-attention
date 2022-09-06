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
from core.trainer import train_model
from utils.utils import get_logger
from utils.altutils import get_mscoco, get_flir, get_flir_from_list_wdomain
from utils.altutils import get_m3fd, get_m3fd_from_list_wdomain
from utils.altutils import setLogger
import logging


def get_train_params(model, part, tasks, lr):
    train_params = [{'params': get_lr_params(model, part=part, tasks=tasks), 'lr': lr},
                    {'params': get_lr_params(model, part='classifier'), 'lr': 10*lr}]
    return train_params

def run(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'train_cls.log'))
    logger.info(args)

    # data loaders
    dataset_root = './dataset_dir/'
    source_train_loader = get_mscoco(dataset_root, args.batch_size, train=True)
    source_val_loader = get_mscoco(dataset_root, args.batch_size, train=False)
    if args.tgt_cat == 'flir':
        target_train_loader = get_flir(dataset_root, args.batch_size, train=True)
        target_val_loader = get_flir(dataset_root, args.batch_size, train=False)
        target_conf_train_loader = get_flir_from_list_wdomain(dataset_root, args.batch_size, train=True)
    elif args.tgt_cat == 'm3fd':
        target_train_loader = get_m3fd(dataset_root, args.batch_size, train=True)
        target_val_loader = get_m3fd(dataset_root, args.batch_size, train=False)
        target_conf_train_loader = get_m3fd_from_list_wdomain(dataset_root, args.batch_size, train=True)
    else:
        raise ValueError("Target dataset {} is not defined.".format(args.tgt_cat))


    args.classInfo = {'classes': torch.unique(torch.tensor(source_train_loader.dataset.targets)),
                      'classNames': source_train_loader.dataset.classes}
    logger.info('cls training')

    # init model
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

    # log parameters
    config={
        'squeeze': squeeze,
        'adapters': adapters,
        'train_norm_layers': train_norm_layers
    }
    config.update({arg: getattr(args, arg) for arg in vars(args)})
    wandb = None
    if args.wandb:
        import wandb
        wandb.init(
            project='thermal-uda-cls',
            config=config
        )
    
    # classification criterion
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    # optimizer for source parameters
    optimizer = optim.Adam(
            get_train_params(model, part='backbone', tasks=['target'], lr=args.lr),
            lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    
    # optimizer for target parameters
    t_optimizer = optim.Adam(
            get_lr_params(model, part='task_specific', tasks=['target']),
            lr=args.t_lr, betas=args.betas, weight_decay=args.weight_decay)

    # init discriminator
    discriminator = Discriminator(args=args).to(args.device)
    d_criterion = nn.CrossEntropyLoss().to(args.device)
    d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=args.d_lr, betas=args.betas, weight_decay=args.weight_decay)
    
    # training
    best_acc, best_class, classNames = train_model(
        model, discriminator, criterion, d_criterion,
        optimizer, t_optimizer, d_optimizer,
        source_train_loader, target_train_loader, target_conf_train_loader, target_val_loader,
        logger, wandb, args=args)

    # log results
    bestClassWiseDict = {}
    for cls_idx, clss in enumerate(classNames):
        bestClassWiseDict[clss] = best_class[cls_idx].item()
    logger.info('Best acc.: {}'.format(best_acc))
    logger.info('Best acc. (Classwise):')
    logger.info(bestClassWiseDict)
    if wandb:
        wandb.log({'val/best_acc': best_acc})
    
    return best_acc, bestClassWiseDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    # train
    parser.add_argument('--self_train', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--t_lr', type=float, default=1e-5)
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--betas', type=float, nargs='+', default=(.5, .999))
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lam', type=float, default=0.25)
    parser.add_argument('--thr', type=float, default=0.90)
    parser.add_argument('--thr_domain', type=float, default=0.87)
    parser.add_argument('--num_val', type=int, default=6)  # number of val. within each epoch
    # misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='outputs/')
    parser.add_argument('--wandb', action="store_true")
    # dataset categories
    parser.add_argument('--src_cat', type=str, default='mscoco')
    parser.add_argument('--tgt_cat', type=str, default='flir')
    parser.add_argument('--message', type=str, default='altinel')  # to track parallel device outputs

    args, unknown = parser.parse_known_args()
    run(args)
