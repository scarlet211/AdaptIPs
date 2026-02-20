import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from train import train_and_evaluate_model_with_cv
from test import test_model
from only import train_and_evaluate_model_with_only
from test5cv import test_model_5cv
import random



seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

    
def remove_name(data):
    data_new = []
    for i in range(1,len(data),2):
        data_new.append(data[i])
    return data_new


with  open("../data/ST-train.fa") as f:
    ST_train = f.readlines()
    ST_train = [s.strip() for s in ST_train]
with  open("../data/ST-test.fa") as f:
    ST_test = f.readlines()
    ST_test = [s.strip() for s in ST_test]
print(len(ST_train),len(ST_test))
print("———————————————————————————————————————————————————")

ST_train_x = remove_name(ST_train)
ST_test_x = remove_name(ST_test)

print(len(ST_train_x),len(ST_train_x[0]))
print(len(ST_test_x),len(ST_test_x[0]))
print("———————————————————————————————————————————————————")
train_headers = [s[1:] for s in ST_train if s.startswith(">")]
test_headers  = [s[1:] for s in ST_test  if s.startswith(">")]
ST_train_y = np.array([int(h) for h in train_headers], dtype=np.int64)
ST_test_y  = np.array([int(h) for h in test_headers], dtype=np.int64)
# ST_train_y = np.concatenate([np.ones((int(len(ST_train_x)/2),)), np.zeros((int(len(ST_train_x)/2),))], axis=0)  
# ST_test_y = np.concatenate([np.ones((int(len(ST_test_x)/2),)), np.zeros((int(len(ST_test_x)/2),))], axis=0)
print(ST_train_y.shape,ST_test_y.shape)
print("———————————————————————————————————————————————————")


# n = int(input("1:training 2:testing:"))
# n=2
# if n == 1:
#     print("ST_site")
#     print("Model: MMM")
#     train_and_evaluate_model_with_cv(ST_train_x, ST_train_y)
#     print("Model: MMM")
#
# else:
#     test_model(ST_test_x, ST_test_y)


train_and_evaluate_model_with_cv(ST_train_x, ST_train_y,ST_test_x, ST_test_y)
# test_model_5cv(ST_test_x, ST_test_y)

# train_and_evaluate_model_with_only(ST_train_x, ST_train_y)
test_model(ST_test_x, ST_test_y)
