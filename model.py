import torch
from torch import nn
import torch.distributions as tdist
import torch.nn.functional as F
import string


class OCR_model(nn.Module):

    def __init__(self, recursive_depth=1, max_word_length=12, embedded_size=512, rnn1_output_size=512):
        super().__init__()

        self.recursive_depth = recursive_depth
        self.max_word_length = max_word_length

        self.cnn1_untied = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
          nn.ReLU(inplace=True)
        )

        self.cnn1_tied = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
          nn.ReLU(inplace=True)
        )

        self.maxpool1 = nn.MaxPool2d(kernel=2, stride=2)

        self.cnn2_untied = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
          nn.ReLU(inplace=True)
        )

        self.cnn2_tied = nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
          nn.ReLU(inplace=True)
        )

        self.maxpool2 = nn.MaxPool2d(kernel=2, stride=2)

        self.cnn3_untied = nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
          nn.ReLU(inplace=True)
        )

        self.cnn3_tied = nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
          nn.ReLU(inplace=True)
        )

        self.maxpool3 = nn.MaxPool2d(kernel=2, stride=2)

        self.cnn1_untied = nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
          nn.ReLU(inplace=True)
        )

        self.cnn1_tied = nn.Sequential(
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
          nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
          nn.Linear(4096, 4096),
          nn.Linear(4096, 4096)
        )

        self.embedded_size = embedded_size
        self.rnn1_output_size = rnn1_output_size

        self.vocab = dict.fromkeys(['<SOW>', '<EOW>'] + list(string.printable[10:36]), 1)
        self.embedding = nn.Embedding(len(self.vocab), self.embedded_size)

        self.rnn1 = nn.LSTMCell(self.embedded_size, 1024)
        self.lin_img = nn.Linear(4096, 4096)
        self.lin_out = nn.Linear(self.rnn1_output_size, 4096)
        self.rnn2 = nn.LSTMCell(4096, self.embedded_size)

        self.output_lin = nn.Linear(self.embedded_size, 27) # number of eng characters+ <eow>

    def forward(self, input):

        # recursive CNN
        x = self.cnn1_untied(input)
        for _ in range(self.recursive_depth):
          x = self.cnn1_tied(x)
        x = self.maxpool1(x)

        x = self.cnn2_untied(input)
        for _ in range(self.recursive_depth):
          x = self.cnn2_tied(x)
        x = self.maxpool2(x)

        x = self.cnn3_untied(input)
        for _ in range(self.recursive_depth):
          x = self.cnn3_tied(x)
        x = self.maxpool3(x)

        x = self.cnn4_untied(input)
        for _ in range(self.recursive_depth):
          x = self.cnn4_tied(x)

        # Flatten image features and send to MLP
        x = self.linear(x.view(x.size(0), -1))

        # Start from <SOW> symbol
        input = self.embedding(torch.tensor([self.vocab['<SOW>']], dtype=torch.long))

        x_proj = self.lin_img(x)

        # nitialize hidden states of rnns
        hidden1 = self.init_hidden(x.size(0), 1024)
        hidden2 = self.init_hidden(x.size(0), 1024)

        # character level language modeling
        y = []
        # организовано циклом, так как новый вход первой рекурентной сети зависит от выходного слоя второй
        for _ in range(self.max_word_length):
          out, hidden1 = self.rnn1(input, hidden1)
          out = self.lin_out(out)
          tau = torch.tanh(out + x_proj)
          alpha = torch.exp(tau) / torch.sum(torch.exp(tau))
          x_att = alpha * x
          input, hidden2 = self.rnn2(x_att, hidden2)

          input_log = F.log_softmax(self.output_lin(x))
          if torch.argmax(input_log) == 1:
            break
          y.append(input_log)

        y = torch.stack(y, dim=0)
        return y

    def init_hidden(self, batch_size, hidden_size):
        normal = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.01]))
        return normal.sample(1, batch_size, hidden_size)
