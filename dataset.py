from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from utils import*

SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)

        self.max_lab, self.min_lab = norm_coef_lab()
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b

        img_norm = self.norm_lab(img_lab)
        low_pass_ab = self.get_filter_image(img_norm)

        img_lab = transforms.ToTensor()(img_lab)
        low_pass_ab = transforms.ToTensor()(low_pass_ab)
        
        return {'L': img_lab[[0], ...], 'ab': img_lab[[1, 2], ...], 'ab_low' : low_pass_ab[[1, 2], ...]}
    
    def __len__(self):
        return len(self.paths)
    

    def low_pass(self, img, filter):

        filtered_img = np.zeros((img.shape[0], img.shape[1], img.shape[2] - 1))

        for i in range(1,img.shape[2]):
            img_chan = img[:,:,i]


            f = np.fft.fft2(img_chan)
            f_shifted = np.fft.fftshift(f)

            f_filtered = filter * f_shifted

            f_filtered_shifted = np.fft.ifftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
            filtered_img[:,:,i-1] = np.real(inv_img)

        return filtered_img



    def get_filter_image(self, img):

        r = 5e4 # how narrower the window is
        ham1 = np.hamming(img.shape[0])[:,None] # 1D hamming
        ham2 = np.hamming(img.shape[1])[:,None] # 1D hamming
        ham2d = np.sqrt(np.dot(ham1, ham2.T)) ** r # expand to 2D hamming



        filtered_img = self.low_pass(img, ham2d)

        return filtered_img

    def norm_lab(self,img):

        img[:,:,0] = 2*(img[:,:,0] - self.min_lab['L']) / (self.max_lab['L']- self.min_lab['L']) - 1
        img[:,:,1] = 2*(img[:,:,1] - self.min_lab['a']) / (self.max_lab['a']- self.min_lab['a']) - 1
        img[:,:,2] = 2*(img[:,:,2] - self.min_lab['b']) / (self.max_lab['b']- self.min_lab['b']) - 1

        return img

    

def make_dataloaders(batch_size=4, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader