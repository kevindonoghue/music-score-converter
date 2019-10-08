import numpy as np
import torch
import torch.nn as nn
import os
import json
import time
from skimage import io, transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from pc_to_xml import pc_to_xml
from django.conf import settings

BASE_DIR = settings.BASE_DIR

device='cpu'

tokens = [
    '<START>', '<END>', '<PAD>', 'measure', 'note', 'pitch', 'step', 'alter',
    'octave', 'duration', 'type', 'rest', 'dot', 'staff', 'notations', 'slur',
    'ff', 'f', 'mf', 'mp', 'p', 'pp', 'backup', 'chord'] + list('ABCDEFG') \
+ ['-1'] + list('0123456789') + ['10', '11', '12', '13', '14', '15', '16'] + ['}'] \
+ ['whole', 'half', 'quarter', 'eighth', '16th']

word_to_ix = {word: i for i, word in enumerate(tokens)}
ix_to_word = {str(i): word for i, word in enumerate(tokens)}
len_lexicon = len(word_to_ix)

lstm_hidden_size = 128
fc1_output_size = 128

def get_time_signature_layer(measure_length, height=224, width=224):
    # measure length is 0 for 4/4 and 1 for 3/4
    x = np.zeros((height, width)).astype(np.uint8)
    if measure_length == 12:
        x[:int(height/2)] += 255
    if measure_length == 16:
        x[int(height/2):] += 255
    return x

def get_key_signature_layer(key_number, height=224, width=224):
    # key number is between -7 and 7 inclusive
    x = np.zeros((height, width)).astype(np.uint8)
    splits = np.array_split(x, 15)
    splits[key_number+7] += 255
    return x

class ConvSubunit(nn.Module):
    def __init__(self, input_size, output_size, filter_size, stride, padding, dropout):
        super().__init__()
        self.conv = nn.Conv2d(input_size, output_size, filter_size, stride=stride, padding=padding)
        self.dp = nn.Dropout2d(p=dropout)
        self.bn = nn.BatchNorm2d(output_size)
        self.relu = nn.LeakyReLU()
        self.sequential = nn.Sequential(self.conv, self.dp, self.bn, self.relu)

    def forward(self, x):
        return self.sequential(x)
    
class LargeConvUnit(nn.Module):
    def __init__(self, input_size, output_size, filter_size, stride, padding, dropout):
        super().__init__()
        self.subunit1 = ConvSubunit(input_size, output_size, filter_size, 1, padding, dropout)
        self.subunit2 = ConvSubunit(output_size, output_size, filter_size, 1, padding, dropout)
        self.subunit3 = ConvSubunit(output_size, output_size, filter_size, 1, padding, dropout)
        self.subunit4 = ConvSubunit(output_size, output_size, filter_size, stride, padding, dropout)

    def forward(self, x):
        x = self.subunit1(x)
        cache = x
        x = self.subunit2(x)
        x = self.subunit3(x)
        x = x + cache
        x = self.subunit4(x)
        return x

        
class LargeCNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.cnn = nn.Sequential(LargeConvUnit(3, 32, 3, 2, 1, 0.1), # (224, 224) --> (112, 112)
                                 LargeConvUnit(32, 64, 3, 2, 1, 0.1), # (112, 112) --> (56, 56)
                                 LargeConvUnit(64, 128, 3, 4, 1, 0.1), # (56, 56) --> (14, 14)
                                 LargeConvUnit(128, 256, 3, 7, 1, 0.1)) # (14, 14) --> (2, 2)
        self.fc = nn.Linear(1024, output_size)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
    
    
class Net(nn.Module):
    def __init__(self, save_dir, cnn, len_lexicon, lstm_hidden_size, fc1_output_size, device, num_directions=1):
        super().__init__()
        self.save_dir = save_dir
        self.len_lexicon = len_lexicon
        self.lstm_hidden_size = lstm_hidden_size
        self.fc1_output_size = fc1_output_size
        self.num_directions = num_directions
        self.bidirectional = (num_directions==2)
        self.cnn = cnn
        self.embed = nn.Embedding(num_embeddings=self.len_lexicon, embedding_dim=5)
        self.lstm1 = nn.LSTM(input_size=5,
                             hidden_size=self.lstm_hidden_size,
                             num_layers=2,
                             batch_first=True,
                             dropout=0.3,
                             bidirectional=self.bidirectional)
        self.lstm2 = nn.LSTM(input_size=self.fc1_output_size+self.num_directions*self.lstm_hidden_size,
                             hidden_size=self.lstm_hidden_size,
                             num_layers=2,
                             batch_first=True,
                             dropout=0.3,
                             bidirectional=self.bidirectional)
        self.fc2 = nn.Linear(self.num_directions*self.lstm_hidden_size, self.len_lexicon)
        
    def forward(self, image_input, sequence_input, internal1=None, internal2=None):
        bs = image_input.shape[0]
        sl = sequence_input.shape[1]
        if internal1:
            h1, c1 = internal1
        else:
            h1 = torch.zeros(2*self.num_directions, bs, self.lstm_hidden_size).to(device)
            c1 = torch.zeros(2*self.num_directions, bs, self.lstm_hidden_size).to(device)
        if internal2:
            h2, c2 = internal2
        else:
            h2 = torch.zeros(2*self.num_directions, bs, self.lstm_hidden_size).to(device)
            c2 = torch.zeros(2*self.num_directions, bs, self.lstm_hidden_size).to(device)
        image_output = self.cnn(image_input)
        image_output = image_output.repeat(1, sl).view(bs, sl, self.fc1_output_size)
        sequence_output, (h1, c1) = self.lstm1(self.embed(sequence_input), (h1, c1))
        concatenated = torch.cat([image_output, sequence_output], 2)
        lstm2_out, (h2, c2) = self.lstm2(concatenated, (h2, c2))
        out = self.fc2(lstm2_out)
        return out, (h1, c1), (h2, c2)
                    
             
    def predict(self, arr):
        self.eval()    
        with torch.no_grad():
            arr = arr.view(1,3, 224, 224)
            output_sequence = ['<START>']
            h1 = torch.zeros(2*self.num_directions, 1, self.lstm_hidden_size).to(device)
            c1 = torch.zeros(2*self.num_directions, 1, self.lstm_hidden_size).to(device)
            h2 = torch.zeros(2*self.num_directions, 1, self.lstm_hidden_size).to(device)
            c2 = torch.zeros(2*self.num_directions, 1, self.lstm_hidden_size).to(device)
            while output_sequence[-1] != '<END>' and len(output_sequence)<400:
                sequence_input = torch.Tensor([word_to_ix[output_sequence[-1]]]).type(torch.long).view(1, 1).to(device)
                out, (h1, c1), (h2, c2) = self.forward(arr, sequence_input, (h1, c1), (h2, c2))
                _, sequence_input = out[0, 0, :].max(0)
                output_sequence.append(ix_to_word[str(sequence_input.item())])
        self.train()
        return output_sequence
    
    def predict_stochastic(self, arr):
        self.eval()    
        with torch.no_grad():
            arr = arr.view(1,3, 224, 224)
            output_sequence = ['<START>']
            h1 = torch.zeros(2*self.num_directions, 1, self.lstm_hidden_size).to(device)
            c1 = torch.zeros(2*self.num_directions, 1, self.lstm_hidden_size).to(device)
            h2 = torch.zeros(2*self.num_directions, 1, self.lstm_hidden_size).to(device)
            c2 = torch.zeros(2*self.num_directions, 1, self.lstm_hidden_size).to(device)
            while output_sequence[-1] != '<END>' and len(output_sequence)<400:
                sequence_input = torch.Tensor([word_to_ix[output_sequence[-1]]]).type(torch.long).view(1, 1).to(device)
                out, (h1, c1), (h2, c2) = self.forward(arr, sequence_input, (h1, c1), (h2, c2))
                log_probs = out[0, 0, :].cpu().numpy()
                probs = np.exp(log_probs) / np.exp(log_probs).sum()
                predicted_ix = np.random.choice(len_lexicon, p=probs)
                output_sequence.append(ix_to_word[str(predicted_ix)])
        self.train()
        return output_sequence
    
large_cnn = LargeCNN(fc1_output_size).to(device)
large_net = Net(None, large_cnn, len_lexicon, lstm_hidden_size, fc1_output_size, device).to(device)



def predict_from_image(model, path, measure_length, key_number):
    image = io.imread(path)
    image = rgb2gray(image)
    image = transform.resize(image, (224, 224))
    image = image.reshape(1, 1, 224, 224)*255
    time_sig_layer = get_time_signature_layer(measure_length).reshape(1, 1, 224, 224)
    key_sig_layer = get_key_signature_layer(key_number).reshape(1, 1, 224, 224)
    arr = np.concatenate([image, time_sig_layer, key_sig_layer], axis=1)
    arr = torch.Tensor(arr).type(torch.float).to(device)
    pred_array = model.predict(arr)
    try:
        soup = pc_to_xml(pred_array[1:-1], measure_length, key_number)
    except:
        soup = None
    return soup

def run_model(path, measure_length, key_number):
    large_cnn = LargeCNN(fc1_output_size).to(device)
    large_net = Net(None, large_cnn, len_lexicon, lstm_hidden_size, fc1_output_size, device).to(device)
    large_net.load_state_dict(torch.load(os.path.join(BASE_DIR, 'large_net_checkpoint_iteration_345000.pt'), map_location=torch.device('cpu')))
    soup = predict_from_image(large_net, path, measure_length, key_number)
    return soup