import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use a pretrained CNN model (ResNet)
        # Other architectures: https://pytorch.org/docs/master/torchvision/models.html
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize all parameters
        
        :param embed_size: Dimensionality of the image embedding
        :param hidden_size: Number of features in the hidden state
        :param vocab_size: Size of the vocabulary
        :param num_layers: Number of layers
        """
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size

        # embedding layer -> words to integers vector
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # LSTM: outputs hidden states of size hidden_size
        # Default: num_layers=1, batch_first=False
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Linear layer to output the scores
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        """
        Predict the following word
        
        :param features: tensor that contains the embedded image features
                         shape: torch.Size([10, 256])
        :param captions: tensor corresponding to the last batch of captions
        
        :return: tensor with size [batch_size, captions.shape[1], vocab_size]
        """
        
        # Embed the caption
        embeddings = self.word_embeddings(captions[:,:-1])
        #print("Embeddings before concatenate: {}".format(embeddings), end="\n\n")
        #print("Word embeddings shape: {}".format(embeddings.shape), end="\n\n")
        
        # Concatenate the features and the embedded captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        #print("Embeddings after concatenate: {}".format(embeddings), end="\n\n")
        
        lstm_output, self.hidden = self.lstm(embeddings)
        
        output = self.linear(lstm_output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        """
        :param inputs: pre-processed image tensors
        
        :return predicted_sentence: predicted sentence (list of tensor ids of length max_len)
        """
        predicted_sentence = []
        
        for i in range(max_len):
            lstm_output, states = self.lstm(inputs, states) # Shape: [1, 1, vocab_size]
            lstm_output = lstm_output[:, -1, :] # use it when batch_first = True
            lstm_output = lstm_output.squeeze(1) # Shape: [1, vocab_size]
            output = self.linear(lstm_output)
            
            _, predicted_index = output.max(1)
            predicted_sentence.append(predicted_index.item())
            
            if predicted_index == 1:
                break
            
            inputs = self.word_embeddings(predicted_index).unsqueeze(1)
        
        return predicted_sentence