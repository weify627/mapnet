from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F

#used for v0 2d toy model in weekly report 2 
#class STN2d(nn.Module):
#    def __init__(self, num_points = 50):
#        super(STN2d, self).__init__()
#        self.num_points = num_points
#        self.conv1 = torch.nn.Conv1d(2, 64, 1)
#        self.conv2 = torch.nn.Conv1d(64, 128, 1)
#        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#        self.mp1 = torch.nn.MaxPool1d(num_points)
#        self.fc1 = nn.Linear(1024, 512)
#        self.fc2 = nn.Linear(512, 256)
#        self.fc3 = nn.Linear(256, 4)
#        self.relu = nn.ReLU()
#
#        self.bn1 = nn.BatchNorm1d(64)
#        self.bn2 = nn.BatchNorm1d(128)
#        self.bn3 = nn.BatchNorm1d(1024)
#        self.bn4 = nn.BatchNorm1d(512)
#        self.bn5 = nn.BatchNorm1d(256)
#
#
#    def forward(self, x):
#        batchsize = x.size()[0]
#        x = F.relu(self.bn1(self.conv1(x)))
#        x = F.relu(self.bn2(self.conv2(x)))
#        x = F.relu(self.bn3(self.conv3(x)))
#        x = self.mp1(x)
#        x = x.view(-1, 1024)
#
#        x = F.relu(self.bn4(self.fc1(x)))
#        x = F.relu(self.bn5(self.fc2(x)))
#        x = self.fc3(x)
#
##        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
#        iden = Variable(torch.from_numpy(np.array([1,0,0,1]).astype(np.float32))).view(1,4).repeat(batchsize,1)
#        if x.is_cuda:
#            iden = iden.cuda()
#        x = x + iden
##        x = x.view(-1, 3, 3)
#        x = x.view(-1, 2, 2)
#        return x


class STN3d(nn.Module):
    def __init__(self, num_points = 500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


def cosine(s1,s2,is_v):
	cos=torch.sum(s1*s2,1) 
	cos=cos/(torch.norm(s1.add(1e-12),2,1)*torch.norm(s2.add(1e-12),2,1))
	return cos

def Cosine(x,faces,is_variable):
	faces=faces.long()
	cos= (torch.zeros(32,3*faces.size(1)))
	if is_variable:
	    cos = Variable(cos)
	if x.is_cuda:
	    cos=cos.cuda()
	for i in range(faces.size(0)):
	   	v1=torch.index_select(x[i,:,:],0,faces[i,:,0])
	    	v2=torch.index_select(x[i,:,:],0,faces[i,:,1])
	   	v3=torch.index_select(x[i,:,:],0,faces[i,:,2])
		s12=v2-v1
		s23=v3-v2
		s13=v3-v1
		cos1=cosine(s12,s13,is_variable)
		cos2=cosine(s23,-s12,is_variable)
		cos3=cosine(-s13,-s23,is_variable)
		cos_cat=torch.cat((cos1,cos2,cos3),0)
		cos[i,:]=cos_cat
        return (cos)

def TriangleSide(x,faces,is_variable):
	faces=faces.long()
	triangle_side= (torch.zeros(32,3*faces.size(1)))
	if is_variable:
	    triangle_side = Variable(triangle_side)
	if x.is_cuda:
	    triangle_side=triangle_side.cuda()
	for i in range(faces.size(0)):
	   	v1=torch.index_select(x[i,:,:],0,faces[i,:,0])
	    	v2=torch.index_select(x[i,:,:],0,faces[i,:,1])
	   	v3=torch.index_select(x[i,:,:],0,faces[i,:,2])
		s1=torch.nn.functional.pairwise_distance(v3,v2)
		s2=torch.nn.functional.pairwise_distance(v3,v2)
		s3=torch.nn.functional.pairwise_distance(v1,v2)
		s_cat=torch.cat((s1,s2,s3),0)
		triangle_side[i,:]=s_cat
        return (triangle_side)
	
def TriangleArea(x,faces,is_variable):
	faces=faces.long()
	triangle_area= (torch.zeros(32,faces.size(1)))
	if is_variable:
	    triangle_area = Variable(triangle_area)
	if x.is_cuda:
	    triangle_area=triangle_area.cuda()
	for i in range(faces.size(0)):
	    v1=torch.index_select(x[i,:,:],0,faces[i,:,0])
	    v2=torch.index_select(x[i,:,:],0,faces[i,:,1])
	    v3=torch.index_select(x[i,:,:],0,faces[i,:,2])
	    a_sub=(v2[:,1]-v3[:,1])
	    a=(v1[:,0])*a_sub
	    b=v2[:,0]*(v3[:,1]-v1[:,1])
	    c=a+b
	    d = (v3[:,0])*(v1[:,1]-v2[:,1])
	    e=c+d
           
    	    triangle_area[i,:] = e*0.5
        return (triangle_area)

def TriangleArea3d(x,faces,is_variable):
	s1=TriangleArea(x[:,:,1:],faces,is_variable)
	v=torch.index_select(x[:,:,:],2,torch.LongTensor([2,0]).cuda())
	s2=TriangleArea(v,faces,is_variable)
	s3=TriangleArea(x[:,:,0:2],faces,is_variable)
	triangle_area = 0.5*torch.sqrt(s1**2+s2**2+s3**2)
        return (triangle_area)


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 500, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans

#class PointNetCls(nn.Module):
#    def __init__(self, num_points = 2500, k = 2):
#        super(PointNetCls, self).__init__()
#        self.num_points = num_points
#        self.feat = PointNetfeat(num_points, global_feat=True)
#        self.fc1 = nn.Linear(1024, 512)
#        self.fc2 = nn.Linear(512, 256)
#        self.fc3 = nn.Linear(256, k)
#        self.bn1 = nn.BatchNorm1d(512)
#        self.bn2 = nn.BatchNorm1d(256)
#        self.relu = nn.ReLU()
#    def forward(self, x):
#        x, trans = self.feat(x)
#        x = F.relu(self.bn1(self.fc1(x)))
#        x = F.relu(self.bn2(self.fc2(x)))
#        x = self.fc3(x)
#        return F.log_softmax(x), trans

class PointNetDenseMap(nn.Module):
    def __init__(self, num_points = 500, k = 2):
        super(PointNetDenseMap, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x, faces):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1)
	area = TriangleArea(x,faces,1)
	side = TriangleSide(x,faces,1)
	z=Variable(torch.FloatTensor(area.size()).zero_())
        if x.is_cuda:
            z = z.cuda()
	margin=torch.max(z,-area)
	margin_sum=torch.mean(margin.sum(1))
	cos = Cosine(x,faces,1)
        return area,side,cos,x,margin_sum

#used for v1 in weekly report2 3d_ellipsoid
class PointNetMap(nn.Module):
    def __init__(self, num_points = 500,dim=3):
        super(PointNetMap, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 2, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(2)
        self.num_points = num_points
    def forward(self, x, faces,is_test,s): 
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
	x = x.transpose(2,1)

	area = TriangleArea(x,faces,1)
	side = TriangleSide(x,faces,1)
	z=Variable(torch.FloatTensor(area.size()).zero_())
        if x.is_cuda:
            z = z.cuda()
	margin=torch.max(z,-area)
	margin_sum=torch.mean(margin.sum(1))
	cos = Cosine(x,faces,1)
        return area,side,cos,x,margin_sum

if __name__ == '__main__':

    sim_data = Variable(torch.rand(32,2,50))
    faces=torch.LongTensor([[0,1,2],[4,5,6]])
    print(type(faces))
    pointfeat = PointNetMap()
    out, _ = pointfeat(sim_data,faces)
    print('map', out.size())
