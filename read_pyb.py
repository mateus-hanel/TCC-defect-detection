import numpy as np
import pickle
import glob
fn = f"KSDD2/split_246.pyb"
with open(f"splits/{fn}", "rb") as f:
    train_samples, test_samples = pickle.load(f)
print(len(train_samples))
test_samples
for t in test_samples:
    if(t[1]):
        print(t)

test_path = r"D:\COMIND\mixed-segdec-net-comind2021\datasets\KSDD2_SMALL\test"
test_names = glob.glob(test_path + "\*.png")

test = []
for name in test_names:
    name = name.split("\\")[-1].split('.')[0]
    if "_GT" in name:
        continue
    test.append((int(name), True))

test_path = r"D:\COMIND\mixed-segdec-net-comind2021\datasets\KSDD2_SMALL\train"
test_names = glob.glob(test_path + "\*.png")

test = []
for name in test_names:
    name = name.split("\\")[-1].split('.')[0]
    if "_GT" in name:
        continue
    test.append((int(name), True))

# Store data (serialize)
with open('test.pyb', 'wb') as handle:
    pickle.dump((test_train,test_train), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test.pyb', "rb") as f:
    foo, foo2 = pickle.load(f)
foo2
train_samples
