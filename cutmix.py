import numpy as np
import torch

class Cutmix():
    """ 
    Applies Cutmix transformation for each generated batch
    
    Args: 
        model: model instance
        criterion: loss function instance
        beta: beta value of Beta distribution
        images: incoming images with tensor format
        labels: labels that corresponds with images
    """

    def __init__(self, model, criterion, beta, images, labels, device):
        self.model = model
        self.criterion = criterion
        self.beta = beta
        self.images = images
        self.labels = labels
        self.device = device
        self.loss = None
        self.preds = None
        self.matches = None

    def start_cutmix(self):
        """
        returns loss and prediction of each 'cut mixed' image
        """
        if self.beta>0 and np.random.random()>=0.5: #cutmix executed
            
            lamb = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(self.images.size()[0]).to(self.device)

            target_a = self.labels #original label
            target_b = self.labels[rand_index] #patch label

            #get random generated box points (x1,y1), (x2,y2)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(self.images.size(), lamb) 
            self.images[:, :, bbx1:bbx2, bby1:bby2] = self.images[rand_index, :, bbx1:bbx2, bby1:bby2] #image 
    
            lamb = 1-((bbx2-bbx1) * (bby2-bby1)/(self.images.size()[-1]*self.images.size()[-2]))

            outputs = self.model(self.images)
            self.loss = self.criterion(outputs, target_a)*lamb + self.criterion(outputs, target_b)*(1. -lamb)

        else: # cutmix not executed

            outputs = self.model(self.images)
            self.loss = self.criterion(outputs, self.labels)

        self.preds = torch.argmax(outputs, dim=1)
   
        return self.loss, self.preds, 

    def rand_bbox(self, size, lam): #size: [Batch_size, Channel, Width, Height]

        """
        - returns (x1,y1) (x2,y2) from original image size.
        - each points are randomly generated.
        - generated points implies patchbox size which will be mixed with original image

        Args:
            size: size of the image
            lam: lambda value of randomly generated data by Beta distribution
        """

        W=size[2]
        H=size[3]

        cut_rat = np.sqrt(1. -lam) #get ratio
        cut_w = np.int(W*cut_rat) 
        cut_h = np.int(H*cut_rat)

        #Center coordinate values
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        #Edges of genderated patch
        bbx1 = 0
        bby1 = np.clip(cy-cut_h//2,0,H)
        bbx2 = W
        bby2 = np.clip(cy+cut_h//2,0,H)

        return bbx1, bby1, bbx2, bby2
