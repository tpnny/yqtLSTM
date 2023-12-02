# 模板
# 定义网络结构

import torch
import torch.nn as nn


# LSTM的基础网络例子
# 阅读了batch_first的原因，就容易理解，因为LSTM本身处理数据是同个样本先后的，所以如果要加速，
# 应该是一起读多个样本的第0、第1这样下去。但是即便是batch_first，网络内部也会转成seq_first,
# 同时为了协调dataloader，我觉得batch_first还是习惯true

# 这边batch_size，在网络里直接预设定不是一个好的策略，LSTM实际运行过程中，有的是不能完整填满batch，所以需要动态取这个第一维的batchsize

class yqtNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1, seq_len=30, num_layers=1, bias=True,
                 batch_first=True, dropout=0, bidirectional=False, proj_size=0):
        super().__init__()

        self.batch_size = 2
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        if bidirectional:
            self.num_directions = 2

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, proj_size=proj_size)

        self.linear = nn.Linear(self.seq_len * self.hidden_size, output_size)

        self.hidden_cell = (torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size),
                            torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size))

    def forward(self, input_seq):
        self.batch_size = input_seq.shape[0]
        x, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        x = x.reshape(self.batch_size, -1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    net = yqtNet(input_size=1, hidden_size=100, output_size=1, seq_len=30)
    input_ = torch.randn(2, 30, 1)
    out = net(input_)
    print(out)
    print(out.shape)
