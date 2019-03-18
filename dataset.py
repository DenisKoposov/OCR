from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class WordImageDataset(Dataset):
    """ Some documantation"""

    def __init__(self, csv_file, root_dir, transform=None, preload=False):
        """
        Args:
        """
        data = pd.read_csv(csv_file)
        self.path = data['path']
        self.targets = data['y']
        self.root_dir = root_dir
        self.transform = transform
    

    def __getitem__(self, idx):

        size = (32, 100)
        img_path = os.path.join(self.root_dir, self.path[idx])
        img = Image.open(img_path).resize(size)
        y = self.targets[idx]
        
        if not self.transform is None:
            img = self.transform(img)

        return img, torch.tensor(y)
    
    
    def __len__(self):
        return len(self.path)