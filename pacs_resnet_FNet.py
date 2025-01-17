
# coding: utf-8

# In[ ]:
import os
from torch.utils.data import Dataset, DataLoader
import os
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, alexnet
import PIL
# from torchlars import LARS
import cv2
import numpy as np

from datasets import *
from pdb import set_trace
import argparse

from imagenet_resnet_18 import resnet18
from optimizers import LARS
from optimizers import LARS2
##################################################### Training f_theta network ###########################################
parser = argparse.ArgumentParser(description='General PyTorch training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--arch',type =str,default = 'resnet')
parser.add_argument('--test_domain', type = str,default = 'sketch')
parser.add_argument("-exp_name","--experiment_name",type=str,default='dum')
parser.add_argument('-wb','--wandb', action = 'store_true')
parser.add_argument('-opt', type = str, default ='SGD')
parser.add_argument('--project_name',type=str)
np.random.seed(0)
CHECKPOINT_DIR = "Models/"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 200
FEATURE_DIM = 256
IMAGE_SIZE = 224
CLASSES = 7
LR = 0.001

src_path = ''
target_path = ''

class DGdata(Dataset):
  def __init__(self, root_dir, image_size, domains=None, transform = None):
  
    self.root_dir = root_dir
    if root_dir[-1] != "/":
      self.root_dir = self.root_dir + "/"
    
    self.categories = ['giraffe', 'horse', 'guitar', 'person', 'dog', 'house', 'elephant']

    if domains is None:
      self.domains = ["photo", "sketch", "art_painting", "cartoon"]
    else:
      self.domains = domains
    
    if transform is None:
      self.transform = transforms.ToTensor()
    else:
      self.transform = transform
    # make a list of all the files in the root_dir
    # and read the labels
    self.img_files = []
    self.labels = []
    self.domain_labels = []
    for domain in self.domains:
      for category in self.categories:
        for image in os.listdir(self.root_dir+domain+'/'+category):
          self.img_files.append(image)
          self.labels.append(self.categories.index(category))
          self.domain_labels.append(self.domains.index(domain))
  
  def __len__(self):
    return len(self.img_files)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    img_path = self.root_dir + self.domains[self.domain_labels[idx]] + "/" + self.categories[self.labels[idx]] + "/" + self.img_files[idx]
    
    image = PIL.Image.open(img_path)
    label = self.labels[idx]

    return self.transform(image), label


class GaussianBlur(object):
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    

class FNet_PACS_ResNet(nn.Module):
 
  def __init__(self, hidden_layer_neurons, output_latent_dim,args):
    super(FNet_PACS_ResNet, self).__init__()
    #resnet = resnet18(pretrained=True, progress=False)
    #self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    if args.arch == 'resnet18':
        self.resnet = resnet18(pretrained = True)
     
    self.fc1 = nn.Linear(self.resnet.feature_dim,  hidden_layer_neurons)
    #self.fc1 = nn.Linear(512,  hidden_layer_neurons)
    self.fc2 = nn.Linear(hidden_layer_neurons, output_latent_dim)
   
  def forward(self, x):
    x = self.resnet(x)
    x = x.squeeze()

    x = self.fc1(x)
    x = F.leaky_relu(x, negative_slope=0.2)

    x = self.fc2(x)
    return x
    

def train_step(x, labels, model, optimizer, tau):
  optimizer.zero_grad()
  # Forward pass
  z = model(x)

  # Calculate loss
  z = F.normalize(z, dim=1)
  pairwise_labels = torch.flatten(torch.matmul(labels, labels.t()))
  logits = torch.flatten(torch.matmul(z, z.t())) / tau
  loss = F.binary_cross_entropy_with_logits(logits, pairwise_labels)
  pred = torch.sigmoid(logits)   # whether two images are similar or not
  accuracy = (pred.round().float() == pairwise_labels).sum()/float(pred.shape[0])

  # Perform train step
  #optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

def training_loop(args,wandb,model, dataset, optimizer, tau=0.1, epochs=200, device=None):
  epoch_wise_loss = []
  epoch_wise_acc = []
  model.train()
  for epoch in (range(epochs)):
    step_wise_loss = []
    step_wise_acc = []
    for image_batch, labels, domains in (dataset):
      image_batch = image_batch.float()
      if dev is not None:
        image_batch, labels = image_batch.to(device), labels.long().to(device)
      labels_onehot = F.one_hot(labels, CLASSES).float()
      loss, accuracy = train_step(image_batch, labels_onehot, model, optimizer, tau)
      step_wise_loss.append(loss)
      step_wise_acc.append(accuracy)


    if (epoch+1)%20 == 0:
      torch.save({'epoch' : epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss}, CHECKPOINT_DIR+"epoch_pacs_resnet_"+args.test_domain+'_'+str(epoch)+".pt")
    epoch_wise_loss.append(np.mean(step_wise_loss))
    epoch_wise_acc.append(np.mean(step_wise_acc))
    print("epoch: {} loss: {:.3f} accuracy: {:.3f} ".format(epoch + 1, np.mean(step_wise_loss), np.mean(step_wise_acc)))

    if args.wandb:
        wandb.log({'tr_loss':epoch_wise_loss})
        wandb.log({'tr_acc':epoch_wise_acc})


  return epoch_wise_loss, epoch_wise_acc, model


args = parser.parse_args()
if args.wandb:
    import wandb
    if os.path.isdir('wand_meta_data/'+args.experiment_name):
        print('not safe')
        sys.exit()
    else:
        os.mkdir('wand_meta_data/'+args.experiment_name)
    wandb.init(project=args.project_name, name = args.experiment_name, resume = True, dir ='wand_meta_data'+"/"+args.experiment_name+'/' )
    wandb.config.update(args,allow_val_change= False)
else:
    wandb=None

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(21)),
                                              transforms.ToTensor(),
                                              AddGaussianNoise(mean=0, std=0.2)] )
# ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)

# dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)

domains_for_train = [ domain for domain in  PACS_DOM_LIST if not  domain == args.test_domain ]

assert len(domains_for_train) <= 3

domain_dataset = Aggregate_DomainDataset(
        dataset_name = 'PACS',
        domain_list = domains_for_train,
        data_split_dir = "/share/data/vision-greg2/xdu/dcorr_content_domain_disentanglement/data/PACS",
        phase = "train",
        image_transform = data_transforms,
        batch_size=BATCH_SIZE,
        num_workers=4,
        use_gpu=True,
        shuffle=True
    )
dataloader = domain_dataset.curr_loader

model = FNet_PACS_ResNet(512, FEATURE_DIM,args)
model = model.to(dev)
if args.opt == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
elif args.opt == 'LARS2':
    optimizer = LARS2(model.parameters(), lr = LR)
elif args.opt == 'LARS':
    optimizer = LARS(model.parameters(), lr = LR)

epoch_wise_loss, epoch_wise_acc, model = training_loop(args,wandb,model, dataloader, optimizer, tau=0.1, epochs=EPOCHS, device=dev)

