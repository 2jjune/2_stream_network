import torch
from torchvision import models
import torch.nn as nn

class Img2Vec():

    def __init__(self, model_path='./fine_tuning_dict.pt'):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            self.model = models.resnet50(pretrained=True).to('cuda')
            self.model.fc = nn.Flatten() # because the model was trained on a cuda machine
        else:
            self.model = models.resnet50(pretrained=True).to('cpu')
            self.model.fc = nn.Flatten()

        self.extraction_layer = self.model._modules.get('avgpool')
        self.layer_output_size = 2048

        self.model = self.model.to(self.device)
        self.model.eval()


    def get_vec(self, image):

        image = image.to(self.device)

        num_imgs = image.size(0)

        my_embedding = torch.zeros(num_imgs, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        return my_embedding.view(num_imgs, -1)