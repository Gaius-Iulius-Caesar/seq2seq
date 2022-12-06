import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.output_size = output_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        def three():
            return (normal((self.input_size, self.hidden_size)),
                    normal((self.hidden_size, self.hidden_size)),
                    torch.zeros(self.hidden_size, device=device))

        W_xi, W_hi, b_i = three()  # 输入门参数
        W_xf, W_hf, b_f = three()  # 遗忘门参数
        W_xo, W_ho, b_o = three()  # 输出门参数
        W_xc, W_hc, b_c = three()  # 候选记忆元参数
        # 输出层参数
        W_hq = normal((self.hidden_size, self.output_size))
        b_q = torch.zeros(self.output_size, device=device)
        # 附加梯度"
        self.params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                       b_c, W_hq, b_q]
        for param in self.params:
            param.requires_grad_(True)

    def init_lstm_state(self, batch_size: int, device="cpu"):
        return (torch.zeros((batch_size, self.hidden_size), device=device),
                torch.zeros((batch_size, self.hidden_size), device=device))

    def forward(self, inputs, state):
        # inputs: L x n x input_size
        # state:init_lstm_state
        # @是矩阵乘法
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
         W_hq, b_q] = self.params
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            Y = (H @ W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H, C)  # output,new_state
