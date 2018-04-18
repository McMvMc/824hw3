import random
import os
import numpy as np
from datetime import datetime
from timer import Timer

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import pickle
import network

# mike
# import model
import logger
from RNN import *
from make_dataset import ActRecDataset

DATA_TRAIN_PATH = "./data/annotated_train_set.p"
DATA_TEST_PATH = "./data/randomized_annotated_test_set_no_name_no_num.p"


# class_names = ["smile", "laugh", "chew", "talk", "smoke", "eat", "drink",
# "cartwheel", "clap hands", "climb", "climb stairs", "dive", "fall on the floor", "backhand flip",
# "handstand", "jump", "pull up", "push up", "run", "sit down", "sit up", "somersault", "stand up",
# "turn", "walk", "wave", "brush hair", "catch", "draw sword", "dribble", "golf", "hit something",
# "kick ball", "pick", "pour", "push something", "ride bike", "ride horse", "shoot ball", "shoot bow",
# "shoot gun", "swing baseball bat", "sword exercise", "throw", "fencing", "hug", "kick someone", "kiss",
# "punch", "shake hands", "sword fight"]

def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def count_corr_n(output, gt_cls, batch_size):
    ap = 0.0
    output = output.data.cpu().numpy()
    gt_cls = gt_cls.cpu().numpy()
    for i in range(batch_size):
        gt_num = gt_cls[i]
        max_id = np.argmax(output[i, :])
        if max_id == gt_num:
            ap += 1
    return ap


def eval_net(net, val_data, criterion):
    loss = 0.0
    corr_cnt = 0.0
    n_val = len(val_data)
    batch_size = 1

    for i in range(n_val):
        val_cuda = val_data[i]

        input_vec = torch.Tensor(batch_size, 5120)
        # gt_cls = torch.LongTensor(batch_size*10)
        gt_cls = torch.LongTensor(batch_size)
        # gt_cls = np.zeros(batch_size)
        # label_vec = network.np_to_variable(gt_cls, is_cuda=True)
        # gt
        # gt_num = train_cuda["class_num"]
        # gt_cls_name = train_cuda["class_name"]

        # input
        for i in range(batch_size):
            # gt_cls[i*10:(i+1)*10] = train_cuda[i]["class_num"]
            gt_cls[i] = val_cuda["class_num"]
            input_vec[i] = torch.Tensor(val_cuda["features"])
        input_var = torch.autograd.Variable(input_vec, requires_grad=False).cuda()
        gt_var = torch.autograd.Variable(gt_cls, requires_grad=False).cuda()

        output, hidden = net(input_var)

        # forward
        loss += criterion(output, gt_var)

        corr_cnt += count_corr_n(output, gt_cls, batch_size)

    loss = loss / n_val
    ap = corr_cnt / n_val
    return loss, ap


def test_net(net, test_data):
    net.eval()
    fname = 'part1.2.txt'
    f = open(fname, 'w')
    n_test = len(test_data)

    for i in range(n_test):
        test_cuda = test_data[i]

        input_vec = torch.Tensor(1, 5120)
        input_vec[0] = torch.Tensor(test_cuda["features"])

        input_var = torch.autograd.Variable(input_vec, requires_grad=False).cuda()

        output, hidden = net(input_var)
        output = output.data.cpu().numpy()
        max_id = np.argmax(output[0, :])
        f.write(str(max_id) + '\n')
    f.close()


# hyper-parameters
# ------------
use_relu = True
visualize = True
vis_interval = 500

start_epoch = 0
batch_size = 80
end_epoch = 50
lr_decay_steps = {end_epoch}
# lr_decay = 1./10
lr_decay = 1

lr = 0.08
initial_lr = lr
momentum = 0.9
weight_decay = 1 / 10
disp_interval = 500
log_interval = 500
eval_interval = 2000
snapshot_interval = 5000

# logs
use_tensorboard = False
use_visdom = False
data_log = logger.Logger('./logs1.2/', name='task1.2')
output_dir = 'models1.2/saved_model'
# ------------

# Create network and initialize
net = actrec()
# net.classifier = nn.DataParallel(net.classifier)
# network.weights_normal_init(net, dev=0.01)

# Move model to GPU and set train mode
net.cuda()
# net.train()

# criterion = nn.NLLLoss().cuda()
criterion = nn.CrossEntropyLoss()
params = net.parameters()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

all_train_data = pickle.load(open(DATA_TRAIN_PATH, "rb"))["data"]
test_data = pickle.load(open(DATA_TEST_PATH, "rb"))["data"]
random.shuffle(all_train_data)

n_samples = len(all_train_data)
n_train = int(n_samples * 0.8)
n_val = n_samples - n_train

train_data = all_train_data[:n_train]
val_data = all_train_data[n_train:]

train_sampler = None
val_sampler = None
test_sampler = None
train_dataset = ActRecDataset(train_data)
val_dataset = ActRecDataset(val_data)
test_dataset = ActRecDataset(test_data, test=True)
workers = 4
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    num_workers=workers, pin_memory=True, sampler=train_sampler)
validate_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=(val_sampler is None),
    num_workers=workers, pin_memory=True, sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True, )

for epoch in range(start_epoch, end_epoch + 1):
    print("epoch " + str(epoch))
    # adjust_learning_rate(optimizer, epoch, initial_lr)
    for cur_step, (input, target) in enumerate(train_loader):
        net.train()
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, requires_grad=True).float().cuda()
        target_var = torch.autograd.Variable(target).long()

        # forward
        optimizer.zero_grad()
        output_var, hidden = net.forward(input_var)

        loss = criterion(output_var, target_var)
        train_loss += loss.data[0]
        train_ap = count_corr_n(output_var, target, target.shape[0]) / target.shape[0]
        step_cnt += 1

        # backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss[0])
        global_step = cur_step + epoch * n_train
        # Log to screen
        if cur_step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = (global_step + 1) / duration
            log_text = 'step %d, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
                global_step, train_loss / step_cnt, fps, 1. / fps, lr, momentum, weight_decay)
            print(log_text)

            for name, weights in net.named_parameters():
                tag = name.replace('.', '/')
                data_log.histo_summary(tag, weights.data.cpu().numpy(), global_step)
                data_log.histo_summary(tag + "/gradients", weights.grad.data.cpu().numpy(), global_step)

            re_cnt = True

        # tensorboard + AP
        if cur_step % eval_interval == 0:
            save_name = 'test_' + str(global_step)
            net.eval()
            eval_loss, eval_ap = eval_net(net, val_data, criterion)
            data_log.scalar_summary('eval/acc', eval_ap, global_step)
            data_log.scalar_summary('eval/loss', eval_loss, global_step)
            # print(output[0,:])
        if cur_step % vis_interval == 0:
            print('Logging to Tensorboard')
            data_log.scalar_summary('train/loss', train_loss / step_cnt, global_step)
            data_log.scalar_summary('train/acc', train_ap, global_step)

        # Save model occasionally
        if (cur_step % snapshot_interval == 0) and global_step > 0:
            save_name = os.path.join(output_dir, '{}_{}.h5'.format("task1_1", global_step))
            network.save_net(save_name, net)
            print('Saved model to {}'.format(save_name))

        if re_cnt:
            tp, tf, fg, bg = 0., 0., 0, 0
            train_loss = 0
            step_cnt = 0
            t.tic()
            re_cnt = False

# train_data['data'][ # ]['features'] -- 10 frames, 512 features each
print("finished trainning")

test_net(net, test_data)
