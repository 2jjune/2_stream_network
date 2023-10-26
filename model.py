import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import fastflow
import torchvision.models as models
from torchsummary import summary
from resnet_feature_extracter import Img2Vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_length = 10

# Two-Stream Network 정의
class TwoStreamNetwork(nn.Module):
    def __init__(self, config):
        super(TwoStreamNetwork, self).__init__()

        # Spatial Stream (ResNet-50)
        # self.spatial_stream = models.wide_resnet50_2(pretrained=True).to(device)
        # self.spatial_stream = models.resnet50(pretrained=True).to(device)
        # self.spatial_stream.fc = nn.Flatten()


        #Fast-Flow
        self.spatial_stream = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        ).to(device)
        # x = torch.zeros(1, 3, 256, 256, dtype=torch.float32).cpu()
        # print(type(x), x.shape[1:])
        # # summary(self.spatial_stream, input_size=x.shape[1:])
        # summary(self.spatial_stream, input_size=x[0].size()[1:])
        # print(
        #     "Model A.D. Param#: {}".format(
        #         sum(p.numel() for p in self.spatial_stream.parameters() if p.requires_grad)
        #     )
        # )

        # Temporal Stream (LSTM Autoencoder)
        # self.temporal_stream = EncoderDecoderConvLSTM(nf=256, in_chan=3).to(device)#nf=64
        self.temporal_stream = AutoEncoderRNN(2048, 32, 2).to(device)#nf=64
        # Fusion Stream
        # summary(self.temporal_stream.to(device), input_size=(3,256,256))

        # self.fusion_stream = nn.Sequential(
        #     # nn.Linear(in_features=4096, out_features=512),
        #     nn.Linear(in_features=131072, out_features=2048),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.5),
        #     nn.Linear(in_features=2048, out_features=1),
        #     nn.Sigmoid()
        # )
        self.fusion_stream = nn.Sequential(
            # nn.Linear(in_features=4096, out_features=512),
            # nn.Linear(in_features=131072, out_features=2048),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=131072, out_features=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x_image = x[:,0,:,:,:]
        # reshape_x = x_image.reshape(batch_size * timesteps, C, H, W)

        # Spatial Stream (ResNet-50)
        x_spatial, spatial_loss = self.spatial_stream(x_image)
        # print('spatial_Stream : ', x_spatial.size())
        # print('spatial_Stream : ', spatial_loss)


        # Temporal Stream (LSTM Autoencoder)
        # x_spatial = x_spatial.permute(0, 2, 1)
        frames = torch.stack(tuple(x))
        # rgb = torch.stack([torch.std(frames[:, i, :, :, :], dim=0) for i in range(frames.shape[1])])
        std_r = torch.std(x[:, :, 0, :, :], dim=1, keepdim=True)
        std_g = torch.std(x[:, :, 1, :, :], dim=1, keepdim=True)
        std_b = torch.std(x[:, :, 2, :, :], dim=1, keepdim=True)
        rgb = torch.cat([std_r, std_g, std_b], dim=1)

        # x_temporal = self.temporal_stream(rgb)
        extractor = Img2Vec()
        inputs = extractor.get_vec(image=frames.view(-1,3,256,256))

        inputs = inputs.reshape(-1, sequence_length, 2048).to(device)

        x_temporal, temporal_loss = self.spatial_stream(rgb)
        total_loss = spatial_loss+temporal_loss
        # x_temporal = self.temporal_stream(inputs)
        # x_temporal = self.temporal_stream(x_spatial)


        # Fusion Stream
        # print(f"spatial shape : {x_spatial.cpu().detach().numpy().shape})")
        # print(f"temporal shape : {x_temporal.size()}")
        # x_fusion = torch.cat((x_spatial[:, -1, :], x_temporal[:, -1, :]), dim=1)
        # print('after spatial, temporal', x_spatial, x_temporal)
        x_fusion = torch.cat((x_spatial.to('cuda'), x_temporal.to('cuda')), dim=1)
        x_fusion = self.fusion_stream(x_fusion)

        return x_fusion, total_loss


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.2, bidirectional=bidirectional)
        self.relu = nn.ReLU()

        # initialize weights
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out[:, -1, :].unsqueeze(1)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True,
                            dropout=0.2)

        # initialize weights
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(AutoEncoderRNN, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, bidirectional)

    def forward(self, x):
        encoded_x = self.encoder(x).expand(-1, sequence_length, -1)
        decoded_x = self.decoder(encoded_x)
        decoded_x = decoded_x.view(1, -1)[:,:2048]
        return decoded_x




class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        # print(self.input_dim , self.hidden_dim)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.resnet = models.resnet50(pretrained=True).to(device)
        # self.resnet.fc = nn.Flatten()
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        ).to(device)

        # resnet_out_dim = self.resnet(torch.zeros(1,3,256,256).to(device)).shape[1]
        self.resnet_out = self.resnet(torch.zeros(1, 3, 256, 256).to(device))
        # self.resnet_out_dim = self.resnet_out.view(self.resnet_out.size(0), -1).shape[1]
        # print(self.resnet_out.shape)
        self.resnet_out_dim = self.resnet_out.shape[1]
        # print(self.resnet_out_dim)
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=self.resnet_out.shape[1],
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        # self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,#inchan=3
        #                                        hidden_dim=nf,
        #                                        kernel_size=(3, 3),
        #                                        bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,#밑으로 다 nf = 256
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=3,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            # print(x[:, t, :, :].cpu().detach().numpy().shape)#1,3,256,256

            h_t, c_t = self.encoder_1_convlstm(input_tensor=self.resnet_out, cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            # h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :], cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t, cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector, cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3, cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions
        # print("outputs shape : ",outputs.size())
        # outputs = np.asarray(outputs)
        # outputs = torch.from_numpy(outputs)
        # outputs = outputs.to(device)

        if len(outputs)>0:
            outputs = torch.stack(outputs, 1)
            outputs = outputs.permute(0, 2, 1, 3, 4)
            outputs = self.decoder_CNN(outputs)
            outputs = torch.nn.Sigmoid()(outputs)
            outputs = outputs.permute(0, 2, 1, 3, 4)
            outputs = torch.nn.Flatten()(outputs)
            outputs = outputs.to('cpu')
            outputs = torch.nn.Linear(in_features=outputs.size(1), out_features=8192)(outputs)
            outputs = torch.nn.ReLU()(outputs)
            outputs = torch.nn.Dropout(p=0.5)(outputs)
            outputs = torch.nn.Linear(in_features=8192, out_features=2048)(outputs)


        else:
            outputs = torch.empty((x.size(0), future_step,3,x.size(3),x.size(4)))

        return outputs

    def forward(self, x, future_seq=1, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()
        # x = x.view(-1, 3,256,256)
        # with torch.no_grad():
        #     x = self.resnet(x)
        # x = x.view(b, seq_len, -1, h // 32, w // 32)
        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # print(x.size(), seq_len, future_seq, h_t.cpu().detach().numpy().shape, c_t.cpu().detach().numpy().shape, h_t2.cpu().detach().numpy().shape,
        #       c_t2.cpu().detach().numpy().shape, h_t3.cpu().detach().numpy().shape, c_t3.cpu().detach().numpy().shape, h_t4.cpu().detach().numpy().shape, c_t4.cpu().detach().numpy().shape)
        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        return outputs