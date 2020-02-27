from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageEnhance
from osgeo import gdal
from torchvision import transforms
import glob
import torch as tc
import numpy as np

class FarmDataset(Dataset):
    def __init__(self, istrain=True, isaug=True, isval=False, istest=False):
        self.istrain = istrain
        self.isval = isval
        self.istest = istest
        self.isaug = isaug
        self.trainxformat = './data/train/data1024/*.png'
        self.trainyformat = './data/train/label1024/*.png'
        self.valxformat = './data/val/data1024/*.png'
        self.valyformat = './data/val/label1024/*.png'
        self.testxformat = './data/test/*.png'
        if istrain:
            self.fns = glob.glob(self.trainxformat)
        elif isval:
            self.fns = glob.glob(self.valxformat)
        else: 
            self.fns = glob.glob(self.testxformat)
        self.length = len(self.fns)
        self.transforms = transforms
        
         
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        if self.istrain:        
            imgxname = self.fns[idx]
            sampleimg = Image.open(imgxname)
            imgyname = imgxname.replace('data1024','label1024')
            targetimg = Image.open(imgyname).convert('L')
             
            # Data augmentation for training images
            if self.isaug:
                sampleimg, targetimg = self.imgtrans(sampleimg, targetimg)
             
            sampleimg = transforms.ToTensor()(sampleimg)
            targetimg = np.array(targetimg)
            targetimg = tc.from_numpy(targetimg).long()         # To tensor

            return sampleimg, targetimg
        
        elif self.isval:
            imgxname = self.fns[idx]
            sampleimg = Image.open(imgxname)
            imgyname = imgxname.replace('data1024','label1024')
            targetimg = Image.open(imgyname).convert('L')
             
            sampleimg = transforms.ToTensor()(sampleimg)
            targetimg = np.array(targetimg)
            targetimg = tc.from_numpy(targetimg).long()         # To tensor
            
            return sampleimg,targetimg           
        
        else:
            # Just open the test image for prediction, no need to cut
            return gdal.Open(self.fns[idx])
        
    def imgtrans(self, x, y, outsize=512):
        '''input is a PIL image
           do image data augumentation
           return a PIL image。
        '''
        # Rotate should consider y
        degree = np.random.randint(360)
        x = x.rotate(degree,resample=Image.NEAREST,fillcolor=0)
        y = y.rotate(degree,resample=Image.NEAREST,fillcolor=0)  
         
        # Random do the input image augmentation
        if np.random.random()>0.5:
            # Sharpness
            factor = 0.5+np.random.random()
            enhancer = ImageEnhance.Sharpness(x)
            x = enhancer.enhance(factor)
        if np.random.random()>0.5:
            # Color augument
            factor = 0.5+np.random.random()
            enhancer = ImageEnhance.Color(x)
            x = enhancer.enhance(factor)
        if np.random.random()>0.5:
            # Contrast augument
            factor = 0.5+np.random.random()
            enhancer = ImageEnhance.Contrast(x)
            x = enhancer.enhance(factor)
        if np.random.random()>0.5:
            # Brightness
            factor = 0.5+np.random.random()
            enhancer = ImageEnhance.Brightness(x)
            x = enhancer.enhance(factor)
         
        # Image flip
        transtypes = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        transtype = transtypes[np.random.randint(len(transtypes))]
        x = x.transpose(transtype)
        y = y.transpose(transtype)
         
        return x,y   # Return pil image
