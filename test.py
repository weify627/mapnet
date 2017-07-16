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
from net_mapping import PointNetMap, TriangleArea, TriangleSide,Cosine 
import torch.nn.functional as F
from datetime import datetime
s='test_' #'testofmse_cos_square_24.pth'
nepo=0
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=520, help='input batch size')
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

#
test_dataset = PartDataset(root = 'data/consistencycheck', classification = True, train = 2, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

print(len(test_dataset))
#num_classes = len(dataset.classes)
#print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs('history/'+s)
except OSError:
    pass

#target_file = open('history/'+s+'/log_'+s+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')

classifier = PointNetMap(num_points = opt.num_points)
#for k in range(50):
k=24
opt.model='mapping/mapping_model_'+s+str(k)+'.pth' #mapping/mapping_model_mse_cos_square_24.pth' #test_24.pth'
classifier.load_state_dict(torch.load(opt.model))
print('loaded')

optimizer=optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

#num_batch = len(dataset)/opt.batchSize
criterion=torch.nn.MSELoss(size_average=False)

ori_f=open('results/'+s+'_input_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
map_f=open('results/'+s+'_output_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
face_f=open('results/'+s+'_face_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')

for j, data in enumerate(testdataloader,0):
       points, target = data
       points, target = Variable(points.cuda()), Variable(target.cuda(),requires_grad=False)
       points = points.transpose(2,1) 
       pred_area,pred_side,pred_cos, pred_pts, margin_area = classifier(points,(target),1,s)
       points = points.transpose(2,1) 
       for i in range(points.size(0)): 
     	 ori_s=str(points.data[i,0:260,:])
#     	 print(ori_s[:len(ori_s)-46]) 
     	 map_s=str(pred_pts.data[i,0:260,:])
#     	 print(map_s[:len(map_s)-46]) 
      	 ori_f.write(ori_s[:len(ori_s)-47])
      	 map_f.write(map_s[:len(map_s)-47])
     	 ori_s=str(points.data[i,260:,:])
#     	 print(ori_s[:len(ori_s)-46]) 
     	 map_s=str(pred_pts.data[i,260:,:])
#     	 print(map_s[:len(map_s)-47]) 
      	 ori_f.write(ori_s[:len(ori_s)-47])
      	 map_f.write(map_s[:len(map_s)-47])
	 for j in [0,250,500,750]:
     		face_s=str(target.data[i,j:j+250,:])
      		face_f.write(face_s[:len(face_s)-46])
		
ori_f.close()
map_f.close()
face_f.close()
print('input saved as', ori_f.name)
print('output saved as', map_f.name)
