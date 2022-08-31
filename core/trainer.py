import os
import sys
sys.path.append(os.path.abspath('.'))
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from utils.utils import AverageMeter, save, save_d
from utils.altutils import ForeverDataIterator

def lr_poly(base_lr, iter_, max_iter, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

def adjust_learning_rate(optimizer, base_lr, iter_, max_iter):
    lr = lr_poly(base_lr, iter_, max_iter)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def train_model(
    model,
    discriminator,
    criterion,
    d_criterion,
    optimizer,
    t_optimizer,
    d_optimizer,
    source_train_loader,
    target_train_loader,
    target_test_loader,
    logger,
    wandb,
    args=None
):
    validation = validate(model, target_test_loader, criterion, task='source', args=args)
    log_source = 'Source/Acc {:.3f} '.format(validation['avgAcc'])

    try:
        best_score = None
        best_class_score = None
        for epoch_i in range(args.epochs):
            start_time = time()
            training = train(
                model, discriminator,
                source_train_loader, target_train_loader, target_test_loader,
                criterion, d_criterion, optimizer, t_optimizer, d_optimizer,
                best_score, best_class_score, epoch_i, logger, wandb, args=args
            )
            # adjust learning rate
            adjust_learning_rate(optimizer, args.lr, epoch_i, args.epochs)
            adjust_learning_rate(t_optimizer, args.t_lr, epoch_i, args.epochs)
            adjust_learning_rate(d_optimizer, args.d_lr, epoch_i, args.epochs)
            
            best_score = training['best_score']
            best_class_score = training['best_class_score']
            n_iters = training['n_iters']
            validation = validate(
                model, target_test_loader, criterion, task='target', args=args)
            clsNames = validation['classNames']
            log = 'Epoch {}/{} '.format(epoch_i, args.epochs)
            log += 'D/Loss {:.3f} Target/Loss {:.3f} '.format(
                training['d/loss'], training['target/loss'])
            log += '[Val] Target/Loss {:.3f} Target/Acc {:.3f} '.format(
                validation['loss'], validation['acc'])
            log += log_source
            log += 'Time {:.2f}s'.format(time() - start_time)
            logger.info(log)
            # track epoch validation metrics
            wandb.log({
                'val/loss': validation['loss'],
                'val/acc': validation['acc'],
                'val/avg_acc': validation['avgAcc']
            })

            # save
            is_best = (best_score is None or validation['avgAcc'] > best_score)
            best_score = validation['avgAcc'] if is_best else best_score
            best_class_score = validation['classAcc'] if is_best else best_class_score
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                't_optimizer': t_optimizer.state_dict(),
                'epoch': epoch_i,
                'val/avg_acc': best_score,
            }
            save(args.logdir, state_dict, is_best)
            state_dict = {
                'model': discriminator.state_dict(),
                'optimizer': d_optimizer.state_dict(),
                'epoch': epoch_i,
                'val/avg_acc': best_score,
            }
            save_d(args.logdir, state_dict, is_best)
            for cls_idx, clss in enumerate(clsNames):
                logger.info('{}: {}'.format(clss, validation['classAcc'][cls_idx]))
            logger.info('Current val. acc.: {}'.format(validation['avgAcc']))
            logger.info('Best val. acc.: {}'.format(best_score))
            classWiseDict = {}
            for cls_idx, clss in enumerate(clsNames):
                classWiseDict[clss] = validation['classAcc'][cls_idx].item()
    except KeyboardInterrupt as ke:
        logger.info('\n============ Summary ============= \n')
        logger.info('Classwise accuracies: {}'.format(best_class_score))
        logger.info('Best val. acc.: {}'.format(best_score))
    
    return best_score, best_class_score, clsNames


def train(
    model, discriminator,
    source_loader, target_loader, target_test_loader,
    criterion, d_criterion, optimizer, t_optimizer, d_optimizer, 
    best_score, best_class_score, epoch_i, logger, wandb, args=None
):
    model.train()
    discriminator.train()

    best_score = best_score
    best_class_score = best_class_score

    losses, d_losses = AverageMeter(), AverageMeter()
    #n_iters = min(len(source_loader), len(target_loader))
    n_iters = 2500
    valSteps = n_iters//args.num_val
    valStepsList = [valSteps+(x*valSteps) for x in range(args.num_val)]
    vals = valStepsList[:-1]
    source_iter, target_iter = ForeverDataIterator(source_loader), ForeverDataIterator(target_loader)
    for iter_i in range(n_iters):
        source_data, source_label = next(source_iter)
        target_data, target_label = next(target_iter)
        #target_data, target_target, target_conf, target_domain, target_domain_conf = target_iter.next()
        #target_conf = target_conf.to(args.device)
        #target_domain = target_domain.to(args.device)
        #target_domain_conf = target_domain_conf.to(args.device)
        source_data = source_data.to(args.device)
        source_label = source_label.to(args.device)
        target_data = target_data.to(args.device)
        target_label = target_label.to(args.device)
        bs = source_data.size(0)
        D_label_source = torch.tensor(
            [0] * bs, dtype=torch.long).to(args.device)
        D_label_target = torch.tensor(
            [1] * bs, dtype=torch.long).to(args.device)

        # Train source params
        for param in discriminator.parameters():
            param.requires_grad = False
        for param in get_lr_params(model, part='task_specific', tasks=['target']):
            param.requires_grad = False
        for param in get_lr_params(model, part='generic', tasks=['target']):
            param.requires_grad = True
        source_pred, source_feat = model(source_data, task='source')
        lossS = criterion(source_pred, source_label)
        # self-training
        '''
        validSource = (target_domain == 0) & (target_conf >= args.thr)
        validMaskSource = validSource.nonzero(as_tuple=False)[:, 0]
        validTarget = (target_domain == 1) & (target_domain_conf <= args.thr_domain) & (target_conf >= args.thr)
        validMaskTarget = validTarget.nonzero(as_tuple=False)[:, 0]
        validIndexes = torch.cat((validMaskSource, validMaskTarget), 0)
        lossT = criterion(target_pred[validIndexes], target_label[validIndexes])
        loss = lossS + lossT
        '''
        loss = lossS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), bs)

        # Train target params
        for param in get_lr_params(model, part='generic', tasks=['target']):
            param.requires_grad = False
        for param in get_lr_params(model, part='task_specific', tasks=['target']):
            param.requires_grad = True
        target_pred, target_feat = model(target_data, task='target')
        D_output_target = discriminator(target_feat)
        lossG = d_criterion(D_output_target, D_label_source)
        t_optimizer.zero_grad()
        lossG.backward()
        t_optimizer.step()

        # Train discriminator
        for param in discriminator.parameters():
            param.requires_grad = True
        _, source_feat = model(source_data, task='source')
        _, target_feat = model(target_data, task='target')
        D_output_source = discriminator(source_feat.detach())
        D_output_target = discriminator(target_feat.detach())
        D_output = torch.cat([D_output_source, D_output_target], dim=0)
        D_label = torch.cat([D_label_source, D_label_target], dim=0)
        d_loss = d_criterion(D_output, D_label)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        d_losses.update(d_loss.item(), bs)

        # track training metrics
        wandb.log({
            'train/d_loss': d_loss.item(),
            'train/loss_g': lossG.item(),
            'train/loss_s': lossS.item(),
            #'train/loss_t': lossT.item(),
            'train/loss': loss.item()
        })

        if iter_i in vals:
            validation = validate(
                model, target_test_loader, 
                criterion, task='target', args=args)
            clsNames = validation['classNames']
            is_best = (best_score is None or validation['avgAcc'] > best_score)
            best_score = validation['avgAcc'] if is_best else best_score
            best_class_score = validation['classAcc'] if is_best else best_class_score
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                't_optimizer': t_optimizer.state_dict(),
                'epoch': epoch_i,
                'val/avg_acc': best_score,
            }
            save(args.logdir, state_dict, is_best)
            state_dict = {
                'model': discriminator.state_dict(),
                'optimizer': d_optimizer.state_dict(),
                'epoch': epoch_i,
                'val/avg_acc': best_score,
            }
            save_d(args.logdir, state_dict, is_best)
            logger.info('Epoch_{} Iter_{}'.format(epoch_i, iter_i))
            for cls_idx, clss in enumerate(clsNames):
                logger.info('{}: {}'.format(clss, validation['classAcc'][cls_idx]))
            logger.info('Current val. acc.: {}'.format(validation['avgAcc']))
            logger.info('Best val. acc.: {}'.format(best_score))
            classWiseDict = {}
            for cls_idx, clss in enumerate(clsNames):
                classWiseDict[clss] = validation['classAcc'][cls_idx].item()

            # track validation metrics
            wandb.log({
                'val/loss': validation['loss'],
                'val/acc': validation['acc'],
                'val/avg_acc': validation['avgAcc']
            })
            model.train()
            discriminator.train()

    return {'d/loss': d_losses.avg, 'target/loss': losses.avg, 'best_score': best_score, 'best_class_score': best_class_score, 'n_iters': n_iters}


def step(model, data, label, criterion, task, args):
    data, label = data.to(args.device), label.to(args.device)
    output, _ = model(data, task=task)
    loss = criterion(output, label)
    return output, loss


def validate(model, dataloader, criterion, task, args=None):
    model.eval()
    losses = AverageMeter()
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
        for iter_i, (data, label) in enumerate(dataloader):
            bs = label.size(0)
            output, loss = step(model, data, label, criterion, task, args)
            pred_cls = output.data.max(1)[1]
            acc_ev += pred_cls.cpu().eq(label.data).cpu().sum()
            for class_idx, class_id in enumerate(classes):
                idxes = torch.nonzero(label==class_id.to(label.device), as_tuple=False)
                class_acc[class_idx] += pred_cls[idxes].cpu().eq(label[idxes].data).cpu().sum()
                class_len[class_idx] += len(idxes)
            output = torch.softmax(output, dim=1)
            losses.update(loss.item(), bs)
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
        'loss': losses.avg, 'acc': acc, 'avgAcc': avgAcc, 'classAcc': class_acc, 'classNames': classNames,
    }
