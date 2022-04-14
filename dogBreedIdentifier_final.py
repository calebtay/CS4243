#%%
# -*- coding: utf-8 -*-

import pycuda
import numpy as np
from numpy import dtype
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint
import time
import os
import sys, os
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from PIL import Image

path = os.path.dirname(os.path.realpath(__file__))
print(path)

train_data = torch.load(path + '/train_data.pt')
train_label = torch.load(path + '/label_data.pt')
test_data = torch.load(path + '/test_data.pt')
test_label = torch.load(path + '/test_label.pt')
val_data = torch.load(path + '/valid_data.pt')
val_label = torch.load(path + '/valid_label.pt')

print(train_data.size())
print(train_label.view(-1).size()) #[910, 1] -> [910]
print(test_data.size())
print(test_label.size())

mean = train_data.float().mean()
std = train_data.float().std()
train_size = train_data.size()[0]
test_size = test_data.size()[0]
print(mean)
print(std)
print(train_size)
print(test_size)

#%%
# For Training

print("Cuda avaliable = " + str(torch.cuda.is_available()))
print("Device Count = " + str(torch.cuda.device_count()))
id = torch.cuda.current_device()
torch.cuda.empty_cache()
print(torch.cuda.get_device_name(id))
print(torch.cuda.memory_allocated(device=torch.device))

r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
print("Before net " + str(r- a))

criterion = nn.CrossEntropyLoss()
lr= 0.0001
bs= 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torchvision.models.resnet50(pretrained=True)
net = net.to(device)
mean = mean.to(device)
std = std.to(device)

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)
net.fc = net.fc.to(device)

r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
print("After net " + str(r- a))

#%%
# Load Model

net.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + "/model_0.pth"))

#%%

def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == torch.flatten(labels))
    # print(predicted_labels)
    # print(torch.flatten(labels))
    num_matches=indicator.sum()
    # print(num_matches)
    
    return 1-num_matches.float()/bs    

def show_prob_cifar(p):


    p=p.data.squeeze().numpy()

    ft=15
    label = ('Bulldog', 'Chihuahua', 'German Shepherd', 'Golden Retriever', 'Jack Russel Terrier', 'Labrador', 'Pomeranian', 'Poodle', 'Samoyed','Siberian Husky' )
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()
    
def eval_on_test_set():

    running_error=0
    num_batches=0

    for i in range(0,100,bs):

        minibatch_data =  test_data[i:i+bs]
        minibatch_label= test_label[i:i+bs]

        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        inputs = (minibatch_data - mean)/std

        scores = net(inputs) 

        error = get_error( scores , minibatch_label)
        # print(scores)
        # print(minibatch_label)
        # print(error)
        running_error += error.item()

        num_batches+=1


    total_error = running_error/num_batches
    print( 'error rate on test set =', total_error*100 ,'percent')
    
#%%
epochs = 50
min_valid_loss = np.inf

r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
print(r- a)

start=time.time()

for epoch in range(1,epochs):

    if not epoch%5:
        lr = lr / 1.5
        
    optimizer=torch.optim.Adam( net.parameters() , lr=lr )
        
    running_loss=0
    running_error=0
    num_batches=0
    
    shuffled_indices=torch.randperm(train_size)
 
    for count in range(0,train_size,bs):
        
        # FORWARD AND BACKWARD PASS
    
        optimizer.zero_grad()
             
        indices=shuffled_indices[count:count+bs]
        minibatch_data =  train_data[indices]
        minibatch_label=  train_label[indices]
        # print(minibatch_data)
        # print(minibatch_label)

        # print("batch" + str(minibatch_data.size()))
        
        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        inputs = (minibatch_data - mean)/std
        
        inputs.requires_grad_()

        #print("inputs " + str(inputs.size()))
        #r = torch.cuda.memory_reserved(0)
        #a = torch.cuda.memory_allocated(0)
        #print(r- a)

        scores = net(inputs) 

        # print("scores " + str(scores.size()))

        # break

        # print("minibatch_label " + str(minibatch_label.size()))

        loss =  criterion(scores , minibatch_label.view(-1)) 
          
        loss.backward()
        
        optimizer.step()
        

        # COMPUTE STATS
        
        running_loss += loss.detach().item()
        
        error = get_error( scores.detach() , minibatch_label)
        running_error += error.item()
        
        num_batches+=1        

        # break
    
    valid_loss = 0.0

    
    for data, labels in zip(val_data, val_label):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        target = net(torch.unsqueeze(data, 0).float())
        loss = criterion(target,labels)
        valid_loss = loss.item() * data.size(0)

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(net.state_dict(), 'saved_model.pth')
    
    # break
    
    # AVERAGE STATS THEN DISPLAY
    total_loss = running_loss/num_batches
    total_error = running_error/num_batches
    elapsed = (time.time()-start)/60
    
    print('epoch=',epoch, '\t time=', elapsed,'min', '\t lr=', lr  ,'\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
    eval_on_test_set() 
    print(' ')
    
#%%
torch.save(net.state_dict(), 'model_1.pth')

#%%
# choose a picture at random
idx=randint(0, len(test_data))
im=test_data[idx]

# diplay the picture
transform = T.ToPILImage()
img = transform(im)
#img.show()

plt.imshow(img)
im=im.view(1,3,200,200)

# feed it to the net and display the confidence scores
im = im.to(device)
im= (im-mean) / std
scores =  net(im.float()) 
probs= torch.softmax(scores, dim=1)

print(test_label[idx])
show_prob_cifar(probs.cpu())
# %%
