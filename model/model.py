import numpy as np
import torch
import torch.nn as nn
import torchvision
import os
import json
import time

# set to 'cuda' for GPU
device='cpu'

# the set of symbols in the pseudocode translation of xml
tokens = [
    '<START>', '<END>', '<PAD>', 'measure', 'note', 'pitch', 'step', 'alter',
    'octave', 'duration', 'type', 'rest', 'dot', 'staff', 'notations', 'slur',
    'ff', 'f', 'mf', 'mp', 'p', 'pp', 'backup', 'chord'] + list('ABCDEFG') \
+ ['-1'] + list('0123456789') + ['10', '11', '12', '13', '14', '15', '16'] + ['}'] \
+ ['whole', 'half', 'quarter', 'eighth', '16th']

# dictionaries to convert the otkens dictionary to numbers and vice versa
word_to_ix = {word: i for i, word in enumerate(tokens)}
ix_to_word = {str(i): word for i, word in enumerate(tokens)}
len_lexicon = len(word_to_ix)

# directory to access to the data for the model
dataset_dir = 'storage/samples/'

# some hyperparameters for the model
lstm_hidden_size = 128
fc1_output_size = 128
seq_len = 64
batch_size = 64

def get_time_signature_layer(measure_length, height=224, width=224):
    """The time signature data is stored in a channel in the input image.
    This function returns that channel. Here measure length is expected to be in (8, 12, 16)."""
    x = np.zeros((height, width)).astype(np.uint8)
    if measure_length == 12:
        x[:int(height/2)] += 255
    if measure_length == 16:
        x[int(height/2):] += 255
    return x

def get_key_signature_layer(key_number, height=224, width=224):
    """The key signature data is stored in a channel in the input image.
    This function returns that channel. Here measure key number is expected to be in range(-7, 8)"""
    x = np.zeros((height, width)).astype(np.uint8)
    splits = np.array_split(x, 15)
    splits[key_number+7] += 255
    return x

class Dataset():
    """
    The data is stored in folders in dataset_dir such that each folder contains the following:
    - a numpy archive images.npy containing an array of shape (N, 224, 224) where N is the number of images in the sample
    - numpy archives measure_lengths and key_numbers that contain the measure_lengths (8, 12, or 16) or key_numbers (in range(-7, 8)) for the images
    - a json file pc_data.json ('pc' stands for pseudocode) that encodes a list whose nth entry is the pseudocode for the nth image
    
    This class collects these data under a single umbrella. It converts the pseudocode into integers
    and self.sequences is a list of the subsequences of the pseudocode of length seq_len
    """
    def __init__(self, path, seq_len):
        self.seq_len = seq_len
        self.images = np.load(path + 'images.npy')
        self.measure_lengths = np.load(path + 'measure_lengths.npy')
        self.key_numbers = np.load(path + 'key_numbers.npy')
        with open(path + 'pc_data.json') as f:
            self.pc_data = json.load(f)
        self.sequences = []
        self.image_indices = []
        for i, pc in enumerate(self.pc_data):
            if len(pc) < seq_len+1:
                pc = pc + ['<PAD>']*(seq_len+1-len(pc))
            pc_as_ix = np.array([int(word_to_ix[word]) for word in pc])
            for j in range(len(pc)-seq_len):
                self.sequences.append(pc_as_ix[j:j+seq_len+1])
                self.image_indices.append(i)
        
        # self.image_indices is a length of the same length as self.sequences
        # the self.image_indices[n] is the index in self.images of the image corresponding to the sequence in self.sequences
        self.image_indices = np.array(self.image_indices)
        self.sequences = np.array(self.sequences)
            
    def get_batch(self, batch_size, val=False):
        """
        get_batch returns three torch tensors:
        - image_batch has shape (batch_size, 3, 224, 224) and consists of the image concatenated with the key and time signature layers
        - sequence_batch has shape (batch_size, seq_len) and consists of subsequences of the pseudocode of the images in image_batch
        - image_batch_indices has shape (batch_size,) and image_batch_indices[n] is the corresponding index in self.images for image_batch[n]
        """
        validation_partition = int(len(self.sequences)*0.9) # 0.9 can be changed to give a different validation split
        if not val:
            sequence_batch_indices = np.random.choice(len(self.sequences[:validation_partition]), size=batch_size)
        else:
            sequence_batch_indices = np.random.choice(len(self.sequences[validation_partition:]), size=batch_size)
        image_batch_indices = self.image_indices[sequence_batch_indices]
        raw_image_batch = self.images[image_batch_indices].reshape(-1, 1, 224, 224)
        measure_lengths_batch = self.measure_lengths[image_batch_indices]
        key_numbers_batch = self.key_numbers[image_batch_indices]
        measure_lengths_layers = []
        key_numbers_layers = []
        for i in range(batch_size):
            measure_length = measure_lengths_batch[i]
            key_number = key_numbers_batch[i]
            measure_lengths_layers.append(get_time_signature_layer(measure_length))
            key_numbers_layers.append(get_key_signature_layer(key_number))
        measure_lengths_layers = np.array(measure_lengths_layers).reshape(-1, 1, 224, 224)
        key_numbers_layers = np.array(key_numbers_layers).reshape(-1, 1, 224, 224)
        image_batch = np.concatenate([raw_image_batch, measure_lengths_layers, key_numbers_layers], axis=1)
        sequence_batch = self.sequences[sequence_batch_indices]
        image_batch = torch.Tensor(image_batch).type(torch.float).to(device)
        sequence_batch = torch.Tensor(sequence_batch).type(torch.long).to(device)
        return image_batch, sequence_batch, image_batch_indices
    
    
def get_datasets(n):
    """
    Randomly generates n datasets and returns them in a list.
    """
    all_datasets = os.listdir(dataset_dir)
    dataset_sample = []
    for _ in range(n):
        i = np.random.randint(len(all_datasets))
        filename = all_datasets[i]
        dataset = Dataset(dataset_dir + filename + '/', seq_len)
        dataset_sample.append(dataset)
    return dataset_sample

class ConvSubunit(nn.Module):
    """
    A torch module consisting of a single convolutional unit followed by dropout, batchnorm, relu
    """
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
    """
    A torch module consisting of four ConvSubunits.
    """
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
    
class SmallConvUnit(nn.Module):
    """
    A torch module consisting of two ConvSubunits.
    """
    def __init__(self, input_size, output_size, filter_size, stride, padding, dropout):
        super().__init__()
        self.subunit1 = ConvSubunit(input_size, output_size, filter_size, 1, padding, dropout)
        self.subunit2 = ConvSubunit(output_size, output_size, filter_size, stride, padding, dropout)

    def forward(self, x):
        x = self.subunit1(x)
        x = self.subunit2(x)
        return x
    
class Net(nn.Module):
    """
    A net that, once trained, can be used to generate xml code for a given input image.
    
    The architecture consists of two LSTMs and a CNN. The image is processed in the CNN,
    and a subsequence of the pseudocode for that image is processed in the first LSTM.
    The output of the first LSTM is concatenated with the output of the CNN and fed into a second LSTM,
    which is trained to predict the next character in the subsequence.
    """
    def __init__(self, save_dir, cnn, len_lexicon, lstm_hidden_size, fc1_output_size, device, num_directions=1):
        super().__init__()
        self.save_dir = save_dir # directory to save logfile and save the model weights periodically
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
    
    def fit(self, iterations, batch_size, optimizer, loss_fn, print_every=100, save_every=5000, train_time=0, past_iterations=0):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        time_checkpoint = time.time()
        for i in range(iterations):
            self.train()
            if i % 500 == 0:
                dataset = get_datasets(1)[0] # this gets a single dataset every 500 iterations
            arr, seq, _ = dataset.get_batch(batch_size)
            seq1 = seq[:, :-1] # initial sequence
            seq2 = seq[:, 1:] # next characters in the sequence
            out, _, _ = self.forward(arr, seq1)
            out = out.view(-1, self.len_lexicon)
            targets = seq2.reshape(-1)
            loss = loss_fn(out, targets)
            loss.backward() # backprop
            optimizer.step() # gradient descent
            optimizer.zero_grad()
            
            
            train_time += time.time() - time_checkpoint
            time_checkpoint = time.time()
            
            # print out progress every once in a while
            if i % print_every == 0:
                # get an example sequence from the validation set to compute validation loss
                with torch.no_grad():
                    arr_val, seq_val, _ = dataset.get_batch(batch_size, val=True)
                    seq1_val = seq_val[:, :-1]
                    seq2_val = seq_val[:, 1:]
                    out_val, _, _ = self.forward(arr_val, seq1_val)
                    out_val = out_val.view(-1, self.len_lexicon)
                    targets_val = seq2_val.reshape(-1)
                    val_loss = loss_fn(out_val, targets_val)
                    
                # get an example from the training set
                arr, _, image_batch_indices = dataset.get_batch(1, val=False)
                pc = dataset.pc_data[image_batch_indices[0]]
                pc = ' '.join(pc)
                
                # predict that example without randomness
                pred_seq = self.predict(arr)
                pred_seq = ' '.join(pred_seq)
                
                # predict that example with some randomness
                pred_seq2 = self.predict_stochastic(arr)
                pred_seq2 = ' '.join(pred_seq2)
                
                # get an example from the validation set
                arr_val, _, image_batch_indices = dataset.get_batch(1, val=True)
                true_val = dataset.pc_data[image_batch_indices[0]]
                true_val = ' '.join(true_val)
                
                # predict that example
                pred_val = self.predict(arr_val)
                pred_val = ' '.join(pred_val)
                
                # save the predictions and ground truths to a log file
                with open(f'{self.save_dir}log.txt', 'a+') as f:
                    info_string = f"""
                    ----
                    iteration: {i}
                    time elapsed: {train_time}
                    train loss: {loss}
                    val loss: {val_loss}
                    ----
                    pred1: {pred_seq}
                    ----
                    pred2: {pred_seq2}
                    ----
                    true:  {pc}
                    ----
                    pred_val: {pred_val}
                    ----
                    true_val: {true_val}
                    ----



                    """.replace('    ', '')
                    print(info_string)
                    f.write(info_string)
                    
            # every so often reduce learning rate and save the model progress
            # also save a checkpoint json file to be used in self.resume_fit
            if i % save_every == 0 and i != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.99
                torch.save(self.state_dict(), f'{self.save_dir}checkpoint_iteration_{past_iterations+i}.pt')
                with open(f'{self.save_dir}training_info.json', 'w+') as f:
                    json.dump({'train_time': train_time, 'past_iterations': past_iterations+i}, f)
                    
    def resume_fit(self, iterations, optimizer, loss_fn, print_every=100, save_every=5000):
        """
        After stopping the model or loading, can resume fit. Relies on the file training_info.json created by self.fit when the model is saved.
        """
        with open(f'{self.save_dir}training_info.json', 'w+') as f:
            training_info = json.load(f)
        train_time = training_info['train_time']
        past_iterations = training_info['past_iterations']
        checkpoint = torch.load(f'{self.save_dir}checkpoint_iteration_{past_iterations}.pt')
        self.load_state_dict(checkpoint)
        self.fit(iterations, optimizer, loss_fn, print_every=print_every, save_every=save_every, train_time=train_time, past_iterations=past_iterations)
                
             
    def predict(self, arr):
        """
        All sequences start with a <START> token and end with an <END> token.
        This uses the LSTM to recursively predict the next character in the pseudocode
        sequence until it reaches the <END> token, or 400 tokens (whatever comes first).
        The next token is predicted by taking the most probable one.
        """
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
        """
        Like self.predict, except the next token is predicted by sampling from the output distribution.
        """
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
    
# initialze the model and optimizer
large_cnn = LargeCNN(fc1_output_size).to(device)
large_loss_fn = nn.CrossEntropyLoss()
large_net = Net(f'storage/large_model_lr3e-4/', large_cnn, len_lexicon, lstm_hidden_size, fc1_output_size, device).to(device)
large_optimizer = torch.optim.Adam(large_net.parameters(), lr=3e-4)

# # start fit
# large_net.fit(500000, batch_size, large_optimizer, large_loss_fn)

# load save state
# change 'cpu' to 'cuda' if necessary/desired
large_net.load_state_dict(torch.load('large_net_checkpoint_iteration_345000.pt', map_location=torch.device('cpu')))