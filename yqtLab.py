import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import shutil

import yqtUtil.yqtDataset as yqtDataset
import yqtUtil.yqtEnvInfo as yqtEnvInfo
import yqtUtil.yqtNet as yqtNet
import yqtUtil.yqtRun as yqtRun

# 一些外部参数的设置
# batchsize
train_batchsize = 10
test_batchsize = 10
# 打印间隔
train_print_freq = 10
test_print_freq = 10

if os.path.exists("best.pth"):
    os.remove("best.pth")

if os.path.exists("logs_model"):
    shutil.rmtree("logs_model")
os.mkdir("logs_model")

yqtEnvInfo.printInfo()
device = yqtEnvInfo.yqtDevice()

# 建立数据集
filen = "/data/Data/water/data_nor.txt"
train_dataset = yqtDataset.yqtDataset(root_dir=filen, ntype="train")
test_dataset = yqtDataset.yqtDataset(root_dir=filen, ntype="test")

print("train_dataset size:\t", train_dataset.size)
print("test_dataset size:\t", test_dataset.size)

# 加载数据
train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=False, num_workers=4)

# 网络初始化
# input_size=1, hidden_size=100, output_size=1, seq_len=30, num_layers=1, bias=True,
# batch_first=True, dropout=0, bidirectional=False, proj_size=0

model = yqtNet.yqtNet(input_size=1, hidden_size=100, output_size=1, seq_len=30)

if os.path.exists("best.pth"):
    model.load_state_dict(torch.load("best.pth"))
model = model.to(device)

# 训练和测试 保存权重
optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

writer = SummaryWriter("logs_model")

best_prec = 100000
if os.path.exists("best.pth"):
    best_prec = yqtRun.test(model=model, dataLoader=test_loader, criterion=criterion, print_freq=test_print_freq,
                            epoch=0, device=device, writer=writer, type_="reg")

for epoch in range(0, 100):
    yqtRun.train(model=model, dataLoader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                 print_freq=train_print_freq, device=device, writer=writer, type_="reg")

    prec_ = yqtRun.test(model=model, dataLoader=test_loader, criterion=criterion, print_freq=test_print_freq,
                        epoch=epoch, device=device, writer=writer, type_="reg")

    if prec_ < best_prec:
        best_prec = prec_
        torch.save(model.state_dict(), "best.pth")

writer.close()
print('train end Best accuracy or Least loss: ', best_prec)
