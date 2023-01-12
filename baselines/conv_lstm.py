import torch
import torch.nn as nn
import torch.optim as optim
from torch_moving_mnist.data import MovingMNIST

from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLSTMCell(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, activation, frame_size
    ):
        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)
        output_gate = torch.sigmoid(o_conv + self.W_co * C)
        H = output_gate * self.activation(C)
        return H, C


class ConvLSTM(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, activation, frame_size
    ):
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.convLSTMcell = ConvLSTMCell(
            in_channels, out_channels, kernel_size, padding, activation, frame_size
        )

    def forward(self, X):
        batch_size, _, seq_len, height, width = X.size()
        output = torch.zeros(
            batch_size, self.out_channels, seq_len, height, width, device=device
        )
        H = torch.zeros(batch_size, self.out_channels, height, width, device=device)
        C = torch.zeros(batch_size, self.out_channels, height, width, device=device)
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)
            output[:, :, time_step] = H
        return output


class Seq2Seq(nn.Module):
    def __init__(
        self,
        num_channels,
        num_kernels,
        kernel_size,
        padding,
        activation,
        frame_size,
        num_layers,
    ):
        super(Seq2Seq, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.add_module(
            "convlstm1",
            ConvLSTM(
                in_channels=num_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                frame_size=frame_size,
            ),
        )
        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )
        for l in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{l}",
                ConvLSTM(
                    in_channels=num_kernels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    frame_size=frame_size,
                ),
            )
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
            )

        self.conv = nn.Conv2d(
            in_channels=num_kernels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, X):
        output = self.sequential(X)
        output = self.conv(output[:, :, -1])
        # return nn.Sigmoid()(output)
        return output
