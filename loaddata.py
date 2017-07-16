from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json


class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 500,nfaces=2052, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.nfaces = nfaces
        self.root = root

        self.meta = []
        dir_point = os.path.join(self.root, 'points')
        dir_seg = os.path.join(self.root,  'faces')
        fns = sorted(os.listdir(dir_point))
        if train==1:
            fns = fns[:int(len(fns) * 0.9)]
	    print('train')
        elif train==0:
	    print('test')
            fns = fns[int(len(fns) * 0.9):]
	else:
	    fns = fns
	    print(fns)
        self.datapath = []
        for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.datapath.append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.fcs')))
#		print(token)

    def __getitem__(self, index):
        fn = self.datapath[index]
	pts = torch.FloatTensor(self.npoints,3).fill_(0)   #max of 10000 cases:1028(over_0.8)500,520
        get_pts = np.loadtxt(fn[0]).astype(np.float32)
        get_pts = torch.from_numpy(get_pts)
	pts[:get_pts.size(0),:]=get_pts

	faces = torch.LongTensor(self.nfaces,3).fill_(0)   #max of 10000 cases: 2052(over_0.8)983, 987
        get_faces = np.loadtxt(fn[1]).astype(np.int64)
        get_faces = torch.from_numpy(get_faces)
	faces[:get_faces.size(0),:]=get_faces
        return pts, faces

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'data',train=False)
    print(len(d))
    ps, seg = d[1]
    print(ps.size(), ps.type(), seg.size(),seg.type())
    print(ps,seg)
