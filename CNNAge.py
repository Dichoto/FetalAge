# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

os.chdir('F:\\CNN Age')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
class CNNAge(nn.Module):
    def __init__(self, regression=False):
        super().__init__()
        self.regression = regression
        self.CNN3d = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=0, dilation=1, bias=True),
                #nn.LeakyReLU(negative_slope=0.1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                #nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=False),
                nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
                #nn.LeakyReLU(negative_slope=0.1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                #nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=False),
                )
        self.fc1 = nn.Linear(in_features=1920, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        output = self.CNN3d(x)
        output = output.view(-1,1920)
        output = self.fc1(output)
        output = self.fc2(output)
        if self.regression:
            return output
        else:
            return self.sigmoid(output)
        
class VGGAge(nn.Module): # has memory issue.
    def __init__(self):
        super().__init__()
        self.VGG3d = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(),
                nn.Conv3d(in_channels=16,out_channels=16, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
                
                nn.Conv3d(in_channels=16,out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(),
                nn.Conv3d(in_channels=32,out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
                
                nn.Conv3d(in_channels=32,out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(),
                nn.Conv3d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),                
                )
        self.fc1 = nn.Linear(in_features=768, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        output = self.VGG3d(x)
        output = output.view(-1,768)
        output = self.fc1(output)
        output = self.fc2(output)
        return self.sigmoid(output)


        
        
    
# simple test
#mat = torch.randn((20,1,43,51,40))
#net = VGGAge()
#prob = net(mat)        

def train_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        prob = model(data).squeeze(1)
        loss = criterion(prob, target.float())
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()*data.shape[0]
    print('Epoch {} Loss: {:.4f}'.format(epoch, train_loss/len(train_loader.dataset)))
        
def test_model(model, device, test_loader, test_agrp, test_target, test_n_image, mode='Validation', thrd=0.5, return_img_prob=False): 
    # test_loader - dataloader, only load image
    # test_agrp - subject level label
    # test_target - image level label
    # test_n_image - a list containing the number of images each subject has
    
    # image level accuracy
    test_prob = torch.zeros(2)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            dt = data[0].to(device)
            prob = model(dt).squeeze(1).to('cpu')
            test_prob = torch.cat((test_prob, prob))
    test_prob = test_prob[2:]  # predicted image probability
    test_pred = (test_prob>thrd).int()
    acc_img = accuracy_score(test_target, test_pred.numpy())
    f1_img = f1_score(test_target, test_pred)
    auc_img = roc_auc_score(test_target, test_prob.numpy())
    print(mode+'Img Level: ' + 'Acc: {:.4f} | F1: {:.4f} | AUC: {:.4f}'.format(acc_img, f1_img, auc_img))
    
    # subject level accuracy by mean probability   
    count = 0
    test_sub_pred = []
    test_sub_prob = []
    for size in test_n_image:
        mean_prob = test_prob[count:(count+size)].mean()
        test_sub_prob.append(test_prob[count:(count+size)].mean())
        test_sub_pred.append((mean_prob>thrd).item())
        count += size
    test_sub_pred = np.array(test_sub_pred)
    test_sub_prob = np.array(test_sub_prob)
    acc_sub = accuracy_score(test_agrp, test_sub_pred)
    f1_sub = f1_score(test_agrp, test_sub_pred)
    auc_sub = roc_auc_score(test_agrp, test_sub_prob)
    print(mode+'Sub Level: ' + 'Acc: {:.4f} | F1: {:.4f} | AUC: {:.4f}'.format(acc_sub, f1_sub, auc_sub))
    #print('F1: {:.4f}'.format(f1_score(test_agrp, test_sub_pred)))
    #print('AUC: {:.4f}'.format(roc_auc_score(test_agrp, test_sub_prob)))
    if return_img_prob == True:
        return  test_sub_pred, test_sub_prob.round(4), test_prob, {'ImgPerf': np.round((acc_img, f1_img, auc_img),4), 'Subperf': np.round((acc_sub, f1_sub, auc_sub),4)}
    
    return test_sub_pred, test_sub_prob.round(4), {'ImgPerf': np.round((acc_img, f1_img, auc_img),4), 'Subperf': np.round((acc_sub, f1_sub, auc_sub),4)}
#%% import data
TAR_DIR = 'Fetus 3D from 4D W2S1'
sub_file = os.listdir(TAR_DIR)
sub_info = pd.read_excel('Subject Info.xlsx')

age_value = sub_info['GA@Scan_corrected'].values.astype('float32')
#age_group = 1*(age_value>=34).astype('int')
#plt.hist(age_value, bins=40,edgecolor='black', linewidth=1.2)
#plt.title('Histogram of Age')

#stress_value = sub_info['Maternal_Stress'].values.astype('float32')
#stress_group = 1*(stress_value>0).astype('int')
#plt.hist(stress_value, bins=20,edgecolor='black', linewidth=1.2)
#plt.title('Histogram of Stress')

idx_young = np.logical_and(age_value>=26, age_value<29) # qualified young baby, index
idx_old = np.logical_and(age_value>=34, age_value<37) # qualified old baby, index
idx_qualified = np.logical_or(idx_young, idx_old) # put young and old together, index

sub_name = list(sub_info['ID'])

outlier_sub = ['Subject072','Subject174'] # those two subject always lead significant performance degradation in testing data.
#Others often classified: subject 084, 085, 152, 154, 186
rm_outlier = True
if rm_outlier:
    for sub in outlier_sub:
        idx_outlier = sub_name.index(sub)
        idx_qualified[idx_outlier] = False

    
sub_qualified = list(sub_info['ID'][idx_qualified]) # age qualified babies, subject ID
age_value_qualified = age_value[idx_qualified] # actual age value
age_group_qualified = 1*(age_value_qualified>=34).astype('int') # age group old 1, young 0 
 
# check histogram
plt.hist(age_value_qualified, bins=40, edgecolor='black', linewidth=1.2)
plt.title('Age Histogram for Yong and Old baby')


sub_image_dict = dict() # save images of qualified babies in dictionary as subject:image
for sub in sub_qualified:
    sub_image_dict[sub] = np.load(TAR_DIR+'\\'+sub+'.npy')

  
#%% train, validation and test splits
    
#train_sub, test_sub, train_age, test_age, train_agrp, test_agrp = train_test_split(sub_file, age_value, age_group, test_size=15,stratify=age_group) # split by sub, not image
#train_sub, val_sub, train_age, val_age, train_agrp, val_agrp = train_test_split(train_sub, train_age, train_agrp, test_size=10,stratify=train_agrp)

train_sub, test_sub, train_agrp, test_agrp = train_test_split(sub_qualified, age_group_qualified, test_size=8,stratify=age_group_qualified) # split by sub, not image
train_sub, val_sub, train_agrp, val_agrp = train_test_split(train_sub, train_agrp, test_size=8,stratify=train_agrp)

# stack all images from train_sub into training dataset, so does for target
train_image_raw = sub_image_dict[train_sub[0]] 
train_target = np.repeat(train_agrp[0], sub_image_dict[train_sub[0]].shape[0])
for sub, age in zip(train_sub[1:], train_agrp[1:]): 
    train_image_raw = np.concatenate((train_image_raw, sub_image_dict[sub]), axis=0)
    train_target = np.concatenate((train_target, np.repeat(age, sub_image_dict[sub].shape[0])))



# stack images from validation sets, process them use train mean and scaling, also for target
val_image = sub_image_dict[val_sub[0]]
val_target = np.repeat(val_agrp[0], sub_image_dict[val_sub[0]].shape[0])
for sub, age in zip(val_sub[1:], val_agrp[1:]):
    val_image = np.concatenate((val_image, sub_image_dict[sub]), axis=0)
    val_target = np.concatenate((val_target, np.repeat(age, sub_image_dict[sub].shape[0])))



# stack images from testing set, process them using train mean and scaling, also for target
test_image = sub_image_dict[test_sub[0]]
test_target = np.repeat(test_agrp[0], sub_image_dict[test_sub[0]].shape[0])
for sub, age in zip(test_sub[1:], test_agrp[1:]):
    test_image = np.concatenate((test_image, sub_image_dict[sub]), axis=0)
    test_target = np.concatenate((test_target, np.repeat(age, sub_image_dict[sub].shape[0])))


no_val = False
if no_val:
    test_image = np.concatenate((val_image,test_image),axis=0)
    test_target = np.concatenate((val_target,test_target))
    test_sub = val_sub+test_sub
    test_agrp = np.concatenate((val_agrp, test_agrp))

# center training data by subtracting mean image, scale by dividing maximum pixel values
mean_image = np.mean(train_image_raw, axis=0)
train_image = train_image_raw - mean_image
intensity_max = np.max(train_image)

train_image = train_image/intensity_max

val_image_raw = val_image
test_image_raw = test_image

val_image = (val_image - mean_image)/intensity_max

test_image = (test_image - mean_image)/intensity_max
        
# record the number of images each subject has
val_n_image = []
for sub in val_sub:
    val_n_image.append(sub_image_dict[sub].shape[0])

test_n_image = []
for sub in test_sub:
    test_n_image.append(sub_image_dict[sub].shape[0])
 
print(val_sub)
print(test_sub)

xtr, xval, xte = torch.from_numpy(train_image).unsqueeze(1), torch.from_numpy(val_image).unsqueeze(1), torch.from_numpy(test_image).unsqueeze(1)
ytr, yval, yte = torch.from_numpy(train_target), torch.from_numpy(val_target), torch.from_numpy(test_target)

train = D.TensorDataset(xtr, ytr)
val = D.TensorDataset(xval)
test = D.TensorDataset(xte)
trainloader = D.DataLoader(train, batch_size=128, shuffle=True)
valloader = D.DataLoader(val, batch_size=50,shuffle=False,sampler=D.SequentialSampler(val))
testloader= D.DataLoader(test, batch_size=50,shuffle=False,sampler=D.SequentialSampler(test))

#%%
net = CNNAge(regression=False)
#net = VGGAge()
net.float()
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-1, weight_decay=0.001, momentum=0.8, nesterov=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)
criterion = nn.BCELoss()


epoch = 15

for i in range(1, epoch+1):
    scheduler.step()
    train_model(net, device, trainloader, optimizer, criterion, i)
    if i>5:
        val_sub_pred, val_sub_prob, _ = test_model(net,device, valloader, val_agrp, val_target, val_n_image, mode='Validation', thrd=0.5)
        test_sub_pred,test_sub_prob, _ = test_model(net,device, testloader, test_agrp, test_target,test_n_image, mode='Testing', thrd=0.5)
           
print(dict(zip(val_sub,tuple(zip(val_agrp,val_sub_pred))))) # check actual predictions
print(dict(zip(test_sub,tuple(zip(test_agrp,test_sub_pred))))) # check actual predictions
#%%
test_model(net,device, valloader, val_agrp, val_target, val_n_image, mode='Validation', thrd=0.5);
test_model(net,device, testloader, test_agrp, test_target,test_n_image, mode='Testing', thrd=0.5);
#%%
data_split = [test_sub,val_sub,train_sub]
ds_df = pd.DataFrame(data_split)
#ds_df.to_csv('Split_4_Sensitivity Analysis(W2S1).csv', index=False, header=False)
    
#%% check statistics
idx_old = np.where(train_target==1)[0]
idx_young = np.where(train_target==0)[0]        
img_old = train_image_raw[idx_old,:,:,:]
img_young = train_image_raw[idx_young,:,:,:]
np.mean(img_old)
np.mean(img_young)
np.var(img_old)
np.var(img_young)
np.max(img_old)
np.max(img_young)
np.min(img_old)
np.min(img_young)
