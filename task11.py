from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import os
import numpy as np
from datetime import datetime
from timer import Timer


import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import pickle
import network


# mike
# import model
import logger
from actrec import *
    


DATA_TRAIN_PATH = "./data/annotated_train_set.p"
DATA_TEST_PATH = "./data/randomized_annotated_test_set_no_name_no_num.p"

# class_names = ["smile", "laugh", "chew", "talk", "smoke", "eat", "drink",
# "cartwheel", "clap hands", "climb", "climb stairs", "dive", "fall on the floor", "backhand flip",
# "handstand", "jump", "pull up", "push up", "run", "sit down", "sit up", "somersault", "stand up",
# "turn", "walk", "wave", "brush hair", "catch", "draw sword", "dribble", "golf", "hit something",
# "kick ball", "pick", "pour", "push something", "ride bike", "ride horse", "shoot ball", "shoot bow",
# "shoot gun", "swing baseball bat", "sword exercise", "throw", "fencing", "hug", "kick someone", "kiss",
# "punch", "shake hands", "sword fight"]

def test_net(net, val_data, criterion):

    loss = 0
    ap = 0
    n_val = len(val_data)

    for i in range(n_val):
        val_cuda = val_data[i]

        output = torch.Tensor(10, 51)
        gt_cls = torch.LongTensor(10)
        # gt
        gt_num = val_cuda["class_num"]
        gt_cls[:] = gt_num
        gt_cls_name = val_cuda["class_name"]

        for i in range(10):
            # input
            input_vec = torch.Tensor(val_cuda["features"][i, :])
            input_var = torch.autograd.Variable(input_vec, requires_grad=True).cuda()

            # forward
            output[i, :] = net(input_var)

        # forward
        loss += criterion(output, gt_cls)

        output = output.data.numpy()
        max_id = np.argmax(np.amax(output, axis=0))
        if max_id == gt_num:
            ap += 1

    loss = loss/n_val
    ap = ap/n_val
    return loss , ap


# hyper-parameters
# ------------
use_relu = True
visualize = True
vis_interval = 500

start_step = 0
batch_size = 32
end_step = 500000
lr_decay_steps = {500000}
lr_decay = 1./10

lr = 0.0001
momentum = 0.9
weight_decay = 1/10
disp_interval = 500
log_interval = 500
eval_interval = 2000
snapshot_interval = 5000

# logs
use_tensorboard = False
use_visdom = False
data_log = logger.Logger('./logs/', name='task1.1')
output_dir = 'models/saved_model'
# ------------

# Create network and initialize
net = actrec()
# network.weights_normal_init(net, dev=0.01)

# Move model to GPU and set train mode
net.cuda()
net.train()

# criterion = nn.NLLLoss().cuda()
criterion = nn.CrossEntropyLoss().cuda()
params = net.classifier.parameters()
optimizer = torch.optim.SGD(net.classifier.parameters(), lr=lr,
                            momentum=momentum, weight_decay=weight_decay)

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
n_train = int(n_samples*0.8)
n_val = n_samples - n_train
train_data = all_train_data[:n_train]
val_data = all_train_data[n_train:]

for step in range(start_step, end_step+1, batch_size):
# for step in range(start_step, end_step+1):
    net.train()

    # if step % n_train == 0:
    #     random.shuffle(train_data)

    train_cuda = train_data[step % n_train: (step+batch_size) % n_train]
    # output = torch.Tensor(10, 51)
    gt_cls = torch.LongTensor(batch_size*10)
    # gt
    gt_num = train_cuda["class_num"]
    gt_cls_name = train_cuda["class_name"]

    # input
    for i in range(batch_size):
	    gt_cls[i*10:(i+1)*10] = gt_num
    
    input_vec = torch.Tensor(train_cuda["features"])
    input_var = torch.autograd.Variable(input_vec, requires_grad=True).cuda()

    # forward
    output = net(input_var)

    loss = criterion(output, gt_cls)
    train_loss += loss.data[0]
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, train_loss / step_cnt, fps, 1./fps, lr, momentum, weight_decay)
        print(log_text)

        for name, weights in net.named_parameters():
            tag = name.replace('.', '/')
            data_log.histo_summary(tag, weights.data.cpu().numpy(), step)
            data_log.histo_summary(tag+"/gradients", weights.grad.data.cpu().numpy(),step)

        re_cnt = True

    # tensorboard + AP
    if step % eval_interval == 0:
        save_name = 'test_'+str(step)
        net.eval()
        loss, ap = test_net(net, val_data, criterion)
        data_log.scalar_summary('eval/mAP', ap, step)
        data_log.scalar_summary('eval/loss', loss, step)
        # print(output[0,:])
    if step % vis_interval==0:
        print('Logging to Tensorboard')
        data_log.scalar_summary('train/loss', train_loss/step_cnt, step)

    
    # Save model occasionally 
    if (step % snapshot_interval== 0) and step > 0:
        save_name = os.path.join(output_dir, '{}_{}.h5'.format("task1_1",step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False



# train_data['data'][ # ]['features'] -- 10 frames, 512 features each



