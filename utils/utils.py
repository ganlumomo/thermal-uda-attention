import os
import shutil
import torch


def save(log_dir, state_dict, is_best):
    checkpoint_path = os.path.join(log_dir, 'checkpoint.pt')
    torch.save(state_dict, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(log_dir, 'best_model.pt')
        shutil.copyfile(checkpoint_path, best_model_path)


def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG, INFO  # noqa
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    sh = StreamHandler()
    sh.setLevel(INFO)
    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('adda')
    logger.setLevel(INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value
       https://github.com/pytorch/examples/blob/master/imagenet/main.py#L296
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
