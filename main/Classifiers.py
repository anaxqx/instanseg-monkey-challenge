import torch
import torch.amp
import torch.nn as nn
import torchvision
import pdb
import os

from instanseg.utils.utils import show_images, display_as_grid, download_model
from instanseg.utils.pytorch_utils import torch_fastremap
from pytorch_utils import get_masked_patches
from instanseg.utils.biological_utils import get_mean_object_features
import torch
import tifffile
import matplotlib.pyplot as plt

class PatchClassifier(nn.Module):
    def __init__(self, num_classes = 16,
                 embedding_dim = 64,
                 patch_size = 128,
                 pretrained=False,
                 encoder = 'mobilenet_v3',
                 from_file = None,
                 dropprob = 0.1,
                 dim_in = 4,
                 ):
        super(PatchClassifier, self).__init__()
 
        if isinstance(num_classes,list):
            num_classes = sum(num_classes)
 
        if encoder.startswith("resnet18"):
            if encoder.startswith("resnet18_original"):
                self.model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
                self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)
                self.model.conv1 = torch.nn.Conv2d(dim_in,64,(7,7),  padding=(3, 3), bias=False)
                self.classifier = nn.Linear(embedding_dim, num_classes)
            else:
                self.model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
                self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)
                self.model.conv1 = torch.nn.Conv2d(dim_in,64,(7,7), stride = (1,1), padding=(3, 3), bias=False)
                self.classifier = nn.Linear(embedding_dim, num_classes)
 
        elif encoder.startswith("efficientnet"):
 
            if encoder.endswith("l"):
             #   import torchvision
                self.model = torchvision.models.efficientnet_v2_l('IMAGENET1K_V1')
                self.model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
 
            else:
              #  import torchvision
                self.model = torchvision.models.efficientnet_v2_m('IMAGENET1K_V1')
                self.model.features[0][0] = nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
 
            self.model.classifier = nn.Identity()
            in_feat = 1280
                    
        elif encoder.startswith("mobilenetv4"):
            import timm
 
            if "hybrid" in encoder:
                self.model = timm.create_model('mobilenetv4_hybrid_medium')
                self.model.conv_stem = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
            
            else:
                self.model = timm.create_model('mobilenetv4_conv_large')
                self.model.conv_stem = torch.nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(1, 1), bias = False) #default is onv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier = nn.Identity()
            # self.classifier = nn.Linear(embedding_dim, num_classes)
            in_feat = 1280
 
        elif encoder in ['mobilenet_v3','mobilenet_v3_large','mobilenet_v3_original','mobilenet_v3_large_original']:
            if encoder.startswith('mobilenet_v3_large'):
                self.model = torchvision.models.mobilenet_v3_large(norm_layer = nn.BatchNorm2d )
            else:
                self.model = torchvision.models.mobilenet_v3_small(norm_layer = nn.BatchNorm2d )
 
            if not encoder.endswith('original'):
                self.model.features[0][0] = torch.nn.Conv2d(dim_in, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            else:
                self.model.features[0][0] = torch.nn.Conv2d(dim_in, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
 
            in_feat = self.model.classifier[0].in_features
            self.model.classifier = nn.Identity()
 
        self.classifier = nn.Sequential(
            nn.Linear(in_feat, embedding_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropprob, inplace=True),
            nn.Linear(embedding_dim, num_classes),
        )
 
        if from_file is not None:
            
            file_path = from_file
            checkpoint = torch.load(file_path)
            self.load_state_dict(checkpoint['model_state_dict'], strict = False)
            print(f"Loaded model from {from_file}")

        self.patch_size = patch_size
        self.num_classes = num_classes


    #@torch.jit.unused
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
    
    
    def embed(self, x):

        with torch.no_grad():
            x = self.model(x)
            x = self.classifier[0](x)
        return x
    
    @torch.jit.unused
    def get_model(self):
        return self.model
    @torch.jit.unused
    def get_patches(self, x, label, classes, max_patches = None):

        assert x.ndim == 3, "Input tensor must have shape (C, H, W)"
        assert label.squeeze().ndim == 2, "Label tensor must have shape (H, W)"
        assert classes.squeeze().ndim == 2, "Classes tensor must have shape (H, W)"

        label = torch_fastremap(label.squeeze())[None,None]
        classes = classes.squeeze()[None]

        crops,masks = get_masked_patches(label,x, patch_size=self.patch_size)

        x =(torch.cat((crops,masks),dim= 1) )
        y = get_mean_object_features(classes.float(),label)
        y = torch.round(y).int()

        return x,y
    
    #@torch.jit.unused
    def predict_img(self, x, label, embed: bool = False, is_multiplexed: bool = False):

        assert x.ndim == 3, "Input tensor must have shape (C, H, W)"
        assert label.squeeze().ndim == 2, "Label tensor must have shape (H, W)"

        label = torch_fastremap(label.squeeze())[None,None]
        crops,masks = get_masked_patches(label,x, patch_size=self.patch_size)

        if is_multiplexed:
            x = torch.cat((crops.unsqueeze(2),masks.repeat(1,crops.shape[1],1,1).unsqueeze(2)),dim = 2).flatten(0,1)
        else:
            x =(torch.cat((crops,masks.float()),dim= 1))

        if embed:
            with torch.no_grad():
                y = self.embed(x)
        else:
            with torch.no_grad():
                y = self.forward(x)

        if is_multiplexed:
            if embed:
                return label,y #NOT IMPLEMENTED
            else:
                y = y.argmax(1).float()
                n_channels = crops.shape[1]
                y = y.view(crops.shape[0],n_channels)

        return label,y
    
    @torch.jit.unused
    def semantic_map(self, label, y, embed = False):
        from instanseg.utils.pytorch_utils import remap_values
        semantic_map = label.clone()
        remapping = torch.stack((torch.arange(len(y),device = y.device,dtype = label.dtype)+1,y.argmax(1))).int()
        semantic_map[semantic_map>0] = remap_values(remapping,semantic_map[semantic_map>0]).to(semantic_map.dtype) + 1


        return semantic_map
    
    @torch.jit.unused
    def get_semantic_map(self, x, label):
        label, y = self.predict_img(x,label)
        if label is None:
            return None
        return self.semantic_map(label,y)
    
    @torch.jit.unused
    def show_patches(self, x, label, max_patches = 50):

        label = torch_fastremap(label.squeeze())[None,None]

        crops,masks = get_masked_patches(label,x, patch_size=self.patch_size)
        x =(torch.cat((crops,masks),dim= 1) )[:max_patches]


        display_as_grid(x,ncols = int(x.shape[0]**0.5))
        plt.show()

        print(x.max(),x.min(),x.mean(),x.std())
        plt.show()


def instance_and_semantic_to_panoptic(instance_maps,semantic_maps,num_classes: int = 6):
    if type(instance_maps) == list:
        instance_maps = torch.stack(instance_maps)
    if type(semantic_maps) == list:
        semantic_maps = torch.stack(semantic_maps)
    if instance_maps.ndim == 3:
        instance_maps = instance_maps[None]
    if semantic_maps.ndim == 3:
        semantic_maps = semantic_maps[None]
    panoptic_maps = []
    for instance_map,semantic_map in zip(instance_maps,semantic_maps):
        semantic_map[semantic_map == 0] = num_classes #background last
        semantic_map -= 1 # 0 based indexing
        panoptic = torch.ones((num_classes,instance_map.shape[0],instance_map.shape[1]),device = instance_map.device,dtype = instance_map.dtype)
        panoptic = panoptic * torch.arange(num_classes,device = instance_map.device,dtype = instance_map.dtype)[:,None,None]
        panoptic = (panoptic == semantic_map).int()
        panoptic[:-1] = panoptic[:-1] * instance_map[None]
        panoptic_maps.append(panoptic)
    panoptic = torch.stack(panoptic_maps)
    return panoptic

from torch import nn
class Predictor(nn.Module):
    def __init__(self, 
                instanseg,classifier, 
                predict_embeddings = False, 
                predict_multiplexed: bool = False):
        super().__init__()
        self.instanseg = instanseg
        self.classifier = classifier
        self.predict_embeddings = predict_embeddings
        self.predict_multiplexed = predict_multiplexed
        
    def forward(self, x):
        with torch.no_grad():
            labels = self.instanseg(x, target_segmentation = torch.tensor([0,1]))
            if labels.max() == 0:
                return labels, torch.zeros((0, self.classifier.num_classes),device = x.device, dtype = torch.float32)
           
            for m in self.classifier.modules():
                m.training = True

            if labels.shape[1] == 2:
                labels = labels[:,1][:,None]

            labels,y = self.classifier.predict_img(x[0],labels[0,0], embed = self.predict_embeddings, is_multiplexed = self.predict_multiplexed)

            return labels, y
        
    
