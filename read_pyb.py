import numpy as np
import pickle
import glob
fn = f"KSDD2/split_247.pyb"
with open(f"splits/{fn}", "rb") as f:
    train_samples, test_samples = pickle.load(f)

not_labeled = []
for t in train_samples:
    if(not t[1]):
        not_labeled.append(t)
print(len(not_labeled))

test_path = r"D:\COMIND\mixed-segdec-net-comind2021\datasets\KSDD2_SMALL\test"
test_names = glob.glob(test_path + "\*.png")

test = []
for name in test_names:
    name = name.split("\\")[-1].split('.')[0]
    if "_GT" in name:
        continue
    test.append((int(name), True))

train_path = r"D:\COMIND\mixed-segdec-net-comind2021\datasets\KSDD2_SMALL\train"
train_names = glob.glob(train_path + "\*.png")

train = []
for name in train_names:
    name = name.split("\\")[-1].split('.')[0]
    if "_GT" in name:
        continue
    train.append((int(name), True))

# Store data (serialize)
with open('test.pyb', 'wb') as handle:
    pickle.dump((train,test), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test.pyb', "rb") as f:
    foo, foo2 = pickle.load(f)
