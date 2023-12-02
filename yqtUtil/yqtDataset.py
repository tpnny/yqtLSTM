# 模板
# 自定义数据集
# 可以比较灵活，这里仅做一个示例，实际设计的过程要主动地分离代码和数据
# 然后数据最好是单个文件，这样相当于压缩，读写更快
# 一般最后增减一下维度匹配上网络就可以

from torch.utils.data import Dataset
import numpy as np
import torch


# 10万的数据集
# 8个训练 2个测试
# LSTM 这里我默认时间窗口 30

# 降到10000

class yqtDataset(Dataset):
    def __init__(self, root_dir, ntype="train"):
        self.file = root_dir
        self.type = ntype

        # 直接处理
        self.data = np.loadtxt(self.file)
        if ntype == "train":
            self.data = self.data[:8000]
        else:
            self.data = self.data[8000:]
        self.size = self.data.size - 31

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        x = self.data[idx:idx + 30]
        y = self.data[idx + 30]

        xTensor = torch.tensor(x).to(torch.float32)
        yTensor = torch.tensor(np.array([y])).to(torch.float32)
        xTensor = torch.unsqueeze(xTensor, dim=1)

        # print(xTensor.shape)
        # print(yTensor.shape)
        return xTensor, yTensor


if __name__ == '__main__':
    testPath = "/data/Data/water/data_nor.txt"
    aa = yqtDataset(testPath, "train")
    print(aa.size)
    # 0-19 预测20
    print(aa[0])
    print(aa[7969])

    aa = yqtDataset(testPath, "test")
    print(aa.size)
    print(aa[0])
    print(aa[1969])
