import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np

class ActRec(nn.Module):
    def __init__(self):
        super(ActRec, self).__init__()
        # if use_relu:
        #     self.classifier = nn.Sequential(
        #         nn.Dropout(p=0.5),
        #         nn.Linear(512, 4096),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(4096, 4096),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(4096, num_classes),
        #     )
        # else:
        #     self.classifier = nn.Sequential(
        #         nn.Linear(512, 4096),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(4096, 4096),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(4096, num_classes),
        #     )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 51),
        )

    def forward(self, x):
        o = self.classifier(x)

        # 10, 51 -> 1, 51
        # score = torch.mean(o, 0, keepdim=True)

        # smax = nn.Softmax(dim=1)
        # prob = smax(score)

        # return prob
        return o


def actrec(**kwargs):
    model = ActRec(**kwargs)



    return model        