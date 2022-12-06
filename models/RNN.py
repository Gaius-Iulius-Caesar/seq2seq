import torch
from torch import nn


# 隐藏层H, 激活函数f, 样本X, 批量大小n, 维度d, 权重X->H: U, H_t->H_t+1:W, H->O: V
# h = hidden_size, n = batch_size, d = input_size, q = output_size, f = active function
# H:nxh, X:nxd, U: dxh, W: hxh, b0: 1xh, O: nxq, V: hxq, b1: 1xq
# H = f(XU+HW+b0),  O = HV + b1
# H = f([XH][UW]+b0)

# 带输出outputs (pytorch中只计算、输出hidden)

class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = None
                 , acti_func=torch.tanh, device="cpu"):
        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if output_size is None:
            output_size = input_size
        self.output_size = output_size
        self.f = acti_func
        # 隐藏层参数
        self.W_xh = normal((input_size, hidden_size)).requires_grad_(True)
        self.W_hh = normal((hidden_size, hidden_size)).requires_grad_(True)
        self.b_h = torch.zeros(hidden_size, device=device).requires_grad_(True)
        # 输出层参数
        self.W_hq = normal((hidden_size, output_size)).requires_grad_(True)
        self.b_q = torch.zeros(output_size, device=device).requires_grad_(True)
        # 附加梯度

    def initHidden(self, batch_size: int = 1, device="cpu"):
        return torch.zeros((batch_size, self.hidden_size), device=device)

    def forward(self, inputs, hidden):
        # inputs: L x n x input_size
        outputs = []
        for X in inputs:
            hidden = self.f(torch.mm(X, self.W_xh) + torch.mm(hidden, self.W_hh) + self.b_h)
            Y = torch.mm(hidden, self.W_hq) + self.b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), hidden


class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = None
                 , acti_func=torch.tanh, device="cpu"):
        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if output_size is None:
            output_size = input_size
        self.output_size = output_size
        self.f = acti_func
        # 隐藏层参数
        self.W_xh = normal((input_size, hidden_size)).requires_grad_(True)
        self.W_hh = normal((hidden_size, hidden_size)).requires_grad_(True)
        self.b_h = torch.zeros(hidden_size, device=device).requires_grad_(True)
        # 输出层参数
        self.W_hq = normal((hidden_size, output_size)).requires_grad_(True)
        self.b_q = torch.zeros(output_size, device=device).requires_grad_(True)
        # 附加梯度

    def initHidden(self, batch_size: int = 1, device="cpu"):
        return torch.zeros((batch_size, self.hidden_size), device=device)

    def forward(self, input, hidden):
        # input: n x input_size
        hidden = self.f(torch.mm(input, self.W_xh) + torch.mm(hidden, self.W_hh) + self.b_h)
        output = torch.mm(hidden, self.W_hq) + self.b_q
        return hidden, output
