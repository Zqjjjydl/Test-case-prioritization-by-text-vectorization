from torch.utils.data import Dataset
import torch

class trainDataset(Dataset): 
    def __init__(self,tempBow,numInput):
        super(trainDataset,self).__init__()
        self.bow=torch.zeros(len(tempBow),numInput)#[documentNum,vocabulary size for bow]
        for (index,document) in enumerate(tempBow):
            for word in document:
                self.bow[index,word[0]]=word[1]
        
    def __getitem__(self, index):
        return self.bow[index]
    def __len__(self): 
        return len(self.bow)