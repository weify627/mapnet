from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from loaddata import PartDataset
from net_mapping import TriangleArea3d,PointNetDenseMap, PointNetMap, TriangleArea, TriangleSide,Cosine
import torch.nn.functional as F
from datetime import datetime
s='0.6z_area' #'over_0.8_area' #define key feature for this model
nepo=25 #define number of epoch for training
data_name='data/0.6z/' #'../3d_ellipsoid/data/over_0.8/' #define dataset directory
npoints=500 #1030
nfaces=1000 #2052
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=1030, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=nepo, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='mapping',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = data_name, classification = True, npoints = opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))
#
test_dataset = PartDataset(root = data_name, classification = True, train = False, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs('history/'+s)
except OSError:
    pass

now=datetime.now().strftime('%Y%m%d_%H%M%S')
target_file = open('history/'+s+'/log_'+s+now+'.txt', 'w')
target_file.write(str(opt))
classifier = PointNetDenseMap(num_points = opt.num_points)
#opt.model='mapping/mapping_model_mse_cos_24.pth'
#opt.model='mapping/20170714_094322/mapping_model_randhalf_area_24.pth'
#opt.model='mapping/20170714_114735/mapping_model_over_0.8_area_24.pth'
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
    printline=(opt.model+' loaded')
    print(printline)
    target_file.write(printline)

optimizer=optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(dataset)/opt.batchSize
criterion=torch.nn.MSELoss()

try:
    os.makedirs('mapping/'+now)
except OSError:
    pass
try:
    os.makedirs('results/'+now)
except OSError:
    pass

c=1
w_marginarea=200000
w_cos=10 #10
w_area=100000000000
for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):

        points, target = data
        points, target = Variable(points.cuda()), Variable(target.cuda(),requires_grad=False)
        points = points.transpose(2,1) 
        optimizer.zero_grad()
        pred_area,pred_side,pred_cos, pred_pts,margin_area = classifier(points,(target)) #,0,s)
## area
	target_area=TriangleArea3d(points.data.transpose(2,1),target.data,0)
	loss_area = w_area*criterion(pred_area,Variable(target_area,requires_grad=False)) #(target)) abs of pred_area
	
#	target_side=TriangleSide(points.data.transpose(2,1),target.data,0)
#	loss_side = criterion(pred_side,Variable(target_side,requires_grad=False))
	target_cos=Cosine(points.data.transpose(2,1),target.data,0)
	loss_cos = w_cos*criterion(pred_cos,Variable(target_cos,requires_grad=False))
	margin_area=w_marginarea*margin_area
	loss=loss_area+loss_cos+margin_area #loss_side #+ loss_area
        loss.backward()
        optimizer.step()
##print train & test loss and save #bachisize test result every 100 iterations
        if i % 100 == 0:
		printline=('[%d: %d/%d] train loss:  %.5f,loss_area: %.5f(weight:%.3f),loss_cos: %.5f(weight:%.3f), margin_area: %.5f(weight:%.3f) \n' %(epoch, i, num_batch, loss.data[0],loss_area.data[0],w_area, loss_cos.data[0],w_cos,margin_area.data[0],w_marginarea))
		print(printline)
	    	target_file.write(printline)
#            j, data = enumerate(testdataloader,0).next()
#            points, target = data
#            points, target = Variable(points.cuda()), Variable(target.cuda(),requires_grad=False)
#            points = points.transpose(2,1) 
#            pred_area,pred_side,pred_cos, _ = classifier(points,(target),0,s)
### area
##	    target_area=TriangleArea(points.data.transpose(2,1),target.data,0)
##	    loss_area = criterion(pred_area,Variable(target_area,requires_grad=False)) #(target))
##	    ratio_area = torch.mean(torch.abs(pred_area.data-target_area).sum(1)/(torch.abs(target_area).sum(1)))
### side
#	    target_side=TriangleSide(points.data.transpose(2,1),target.data,0)
#	    loss_side = criterion(pred_side,Variable(target_side,requires_grad=False))
#	    ratio_side = torch.mean(torch.abs(pred_side.data-target_side).sum(1)/(torch.abs(target_area).sum(1)))
### cosine
#	    target_cos=Cosine(points.data.transpose(2,1),target.data,0)
#	    loss_cos = criterion(pred_cos,Variable(target_cos,requires_grad=False))
### printing
#	    target_file.write(printline)
#	    str_area = '       total area           =  %.3f, rate of change = %.3f\n'%(0.5*torch.mean(torch.abs(target_area).sum(1)),ratio_area)
#	    str_side = '       total length of edge = %.3f, rate of change = %.3f\n'%(torch.mean(torch.abs(target_side).sum(1)),ratio_side)
##	    str_cos = '       average cos / angle =  %.3f, rate of change = %.3f'%(torch.mean(torch.abs(target_cos)[:,0:150]),ratio_cos)
#	    print(str_area) #,torch.min(0.5*(torch.abs(target_area).sum(1))),torch.max(0.5*(torch.abs(target_area).sum(1))))
#	    print(str_side)
##	    print(str_cos)
#	    target_file.write(str_area)
#	    target_file.write(str_side)
##	    target_file.write(str_cos)
	   	ori_f=open('results/'+now+'/'+s+'_input_'+str(c)+'.txt', 'w')
	   	map_f=open('results/'+now+'/'+s+'_output_'+str(c)+'.txt', 'w')
	   	face_f=open('results/'+now+'/'+s+'_face_'+str(c)+'.txt', 'w')
	   	c=c+1
	   	j, data = enumerate(testdataloader,0).next()
	   	points, target = data
	   	points, target = Variable(points.cuda()), Variable(target.cuda(),requires_grad=False)
	   	points = points.transpose(2,1) 
	   	pred_area,pred_side,pred_cos, pred_pts, margin_area = classifier(points,target)#,1,s)
		## cos
	   	target_cos=Cosine(points.data.transpose(2,1),target.data,0)
	   	loss_cos = w_cos*criterion(pred_cos,Variable(target_cos,requires_grad=False))
		ratio_cos = torch.mean((torch.abs(pred_cos.data-target_cos)[:,0:150])/(torch.abs(target_cos)[:,0:150]))
		## margin_area
	   	margin_area=w_marginarea*margin_area
		## area
		target_area=TriangleArea3d(points.data.transpose(2,1),target.data,0)
		loss_area = w_area*criterion(pred_area,Variable(target_area,requires_grad=False)) #(target)) abs of pred_area
		
	   	loss=loss_area+loss_cos+margin_area #loss_side #+ loss_area	
       	    	printline=('[%d: %d/%d] test loss: %.5f,loss_cos: %.5f, margin_area: %.5f \n' %(epoch, i, num_batch, loss.data[0], loss_cos.data[0],margin_area.data[0]))
	   	target_file.write(printline)
	   	points = points.transpose(2,1) 
		for i in range(points.size(0)): 
			for j in range(points.size(1)): 
				for k in range(points.size(2)): 
					ori_f.write(str(points.data[i,j,k]))
					ori_f.write(' ')
				ori_f.write('\n')
				for k in range(pred_pts.size(2)): 
					map_f.write(str(pred_pts.data[i,j,k]))
					map_f.write(' ')
				map_f.write('\n')
			for j in range(target.size(1)): 
				for k in range(target.size(2)): 
					face_f.write(str(target.data[i,j,k]))
					face_f.write(' ')
				face_f.write('\n')
	   	  	
		ori_f.close()
	   	map_f.close()
	   	face_f.close()
#save trained model for every epoch
    torch.save(classifier.state_dict(), '%s/%s/mapping_model_%s_%d.pth' % (opt.outf,now,s, epoch))
target_file.close()


## test the final model
ori_f=open('results/'+now+'/'+s+'_input_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
map_f=open('results/'+now+'/'+s+'_output_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
face_f=open('results/'+now+'/'+s+'_face_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')

_,data = enumerate(testdataloader,0).next()
points, target = data
points, target = Variable(points.cuda()), Variable(target.cuda(),requires_grad=False)
points = points.transpose(2,1) 
pred_area,pred_side,pred_cos, pred_pts, margin_area = classifier(points,target)#,1,s)
points = points.transpose(2,1) 
for i in range(points.size(0)): 
	for j in range(points.size(1)): 
		for k in range(points.size(2)): 
			ori_f.write(str(points.data[i,j,k]))
			ori_f.write(' ')
		ori_f.write('\n')
		for k in range(pred_pts.size(2)): 
			map_f.write(str(pred_pts.data[i,j,k]))
			map_f.write(' ')
		map_f.write('\n')
	for j in range(target.size(1)): 
		for k in range(target.size(2)): 
			face_f.write(str(target.data[i,j,k]))
			face_f.write(' ')
		face_f.write('\n')
ori_f.close()
map_f.close()
face_f.close()

print('input saved as', ori_f.name)
print('output saved as', map_f.name)
print('face saved as', face_f.name)
