from sklearn.model_selection import train_test_split
from torchvision.models import resnet101
import matplotlib
import torch
import PIL


model = resnet101()
print("Imported Model")

CLASSES = 50
EPOCHS = 50
INITIAL_LR = 0.001
BATCH_SIZE = 64


dvc remote modify minioserver endpointurl http://172.24.198.42:9000
dvc remote modify minioserver access_key_id daki
dvc remote modify minioserver secret_access_key dakiminio
dvc remote modify minioserver use_ssl false