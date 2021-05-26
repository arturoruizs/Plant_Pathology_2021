import os
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt 
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from utils import display_grid, one_hot_labels



def data_exploration():
	train_df = pd.read_csv(train_df_path)
	class_freq = train_df['labels'].value_counts()
	labels = class_freq.index.to_list()
	freqs = class_freq.values
	label_df = pd.DataFrame(data = {'labels': labels, 'frequencies': freqs} )
	
	fig = px.bar(label_df, x='labels', y='frequencies')
	fig.show()

	#Creating a sample dataframe to plot one example per class
	sample_df = train_df.groupby('labels').first().reset_index()
	labels,images = sample_df['labels'],sample_df['image']
	display_grid(images,labels,3,4)

class PlantDS(Dataset):

	def __init__(self, root_dir, transforms = None):

		self.root_dir = root_dir
		self.transforms = transforms

		df = pd.read_csv(os.path.join(root_dir, 'plant_pathology/train.csv'))

		self.images = df['image'].values
		self.labels = one_hot_labels(df).values

	def __getitem__(self,idx):

		image = Image.open(os.path.join(self.root_dir,'plant_pathology/train_images', self.images[idx]))
		label = self.labels[idx]

		if self.transforms:

			image = self.transforms(image)
			label = torch.FloatTensor(label)

		return image, label
		
	def __len__(self):
		return len(self.images)


if __name__ == '__main__':

	#Paths
	ROOT_DIR =os.path.dirname(os.path.abspath(os.curdir))
	train_images_path = '/train_images'
	train_df_path = 'train.csv'
	
	data_exploration()

	tensor_plants = PlantDS(ROOT_DIR, transforms = transforms.Compose([transforms.ToTensor(),transforms.Resize((128,128)),transforms.Normalize((0.4871, 0.6265, 0.4081), (0.1883, 0.1652, 0.1961))]))
	train_loader = DataLoader(tensor_plants, batch_size= 64, shuffle=True)















