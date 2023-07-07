import numpy as np
import scipy.io
import scipy.interpolate
import torch
import random
random.seed(10)
from torch import nn
from torch.autograd import grad
import torch_geometric.nn as gnn
from utils import * 

class VariLoss(nn.Module):
    def __init__(self,edges,template):
        super(VariLoss, self).__init__()
        self.edges=edges
        self.template=template

    def forward(self, output, target, tedges):
        MSV_Error = 0
        for i in range(output.shape[0]):
            loss = varifold(target[i],tedges[i]) 
            MSV_Error += loss(output[i]+self.template,self.edges)+ .0000001*SRNF(output[i]+self.template,self.template,self.edges)
        
        return MSV_Error/output.shape[0]

class VariGrad(nn.Module):
    def __init__(self, edges, template):
        super().__init__()
        self.edges=edges
        self.template=template

    def forward(self, x, e):
        if len(x[0].shape) < 2:
            print("INPUT DOES NOT HAVE BATCH DIMENSION")
        ls=[]
        for i in range(len(x)):
            loss=varifold(x[i],e[i])
            qtemplate = self.template.clone().requires_grad_(True)
            [g] = grad(loss(qtemplate,self.edges), qtemplate, create_graph=True)
            ls+=[g.detach()]
        return ls

    
class Encoder(nn.Module):
    def __init__(self, edges, template):
        super().__init__()
        self.N = template.shape[1]
        self.dim = template.shape[0]  
        self.VG = VariGrad(edges, template).cuda()
        self.edges=edges.T
        self.conv1=gnn.GCNConv(self.dim,2*self.dim)  
        self.bn1 = gnn.BatchNorm(2*self.dim)
        self.conv2=gnn.GCNConv(2*self.dim,2*self.dim)
        self.bn2 = gnn.BatchNorm(2*self.dim)
        self.conv3=gnn.GCNConv(2*self.dim,2*self.dim)
        self.bn3 = gnn.BatchNorm(2*self.dim)
        
        
        self.encoder = nn.Sequential(
            nn.Linear(self.N*self.dim*2,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
        )

    def forward(self,x,e):
        x=self.VG(x,e)
        v=torch.cat(x, dim=1).double()
        ei=torch.cat([self.edges+self.N*i for i in range(len(x))],dim=1).long()
        o=self.conv1(v.T,ei) 
        o=self.bn1(o)
        o=o.relu()
        o=self.conv2(o,ei)
        o=self.bn2(o)
        o=o.relu()
        o=self.conv3(o,ei) 
        o=self.bn3(o)
        o=o.relu()
        o=self.encoder(o.reshape(len(x),-1))    
        return o
    
    
class Decoder(nn.Module):
    def __init__(self,template):
        super().__init__()
        
        self.N = template.shape[1]
        self.dim = template.shape[0]
        self.lin = nn.Sequential( 
            nn.Linear(32,32),
            nn.ReLU(),    
            nn.Linear(32,32),
            nn.ReLU(),    
            nn.Linear(32,64),
            nn.ReLU(),    
            nn.Linear(64,128),
            nn.ReLU(),    
            nn.Linear(128,256),
            nn.ReLU(),  
            nn.Linear(256,self.N*self.dim),       
        )


    def forward(self, x):
        o=self.lin(x).reshape(-1,self.dim,self.N)
        return o
    
class Classifier(nn.Module):
    def __init__(self,edges, template, classes):
        super().__init__()
        self.N = template.shape[1]
        self.dim = template.shape[0]  
        self.VG = VariGrad(edges, template).cuda()
        
        self.edges=edges.T
        self.conv1=gnn.GCNConv(self.dim,8*self.dim)  
        self.bn1 = gnn.BatchNorm(8*self.dim)
        self.conv2=gnn.GCNConv(8*self.dim,32*self.dim)
        self.bn2 = gnn.BatchNorm(32*self.dim)
        self.conv3=gnn.GCNConv(32*self.dim,32*self.dim)
        self.bn3 = gnn.BatchNorm(32*self.dim)
        
        self.lin = nn.Sequential(
            
            nn.Linear(32*self.dim*self.N,512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256,classes)
            
        )

    def forward(self,x,e):
        x=self.VG(x,e)
        v=torch.cat(x, dim=1).double()
        ei=torch.cat([self.edges+self.N*i for i in range(len(x))],dim=1).long()
        o=self.conv1(v.T,ei) 
        o=self.bn1(o)
        o=o.relu()
        o=self.conv2(o,ei)
        o=self.bn2(o)
        o=o.relu()
        o=self.conv3(o,ei) 
        o=self.bn3(o)
        o=o.relu()
        o=self.lin(o.reshape(len(x),-1))
        return o
    
    
class VG_C:
    def __init__(self, template, template_edges,trainsize, testsize, classes):
        self.template=template
        self.template_edges=template_edges
        self.C=Classifier(template_edges, template, classes).cuda().double()
        self.Loss=nn.CrossEntropyLoss()
        self.trainsize=trainsize
        self.testsize =testsize
        self.optimizer = torch.optim.Adam(self.C.parameters(),lr = 5e-5,weight_decay = 1e-8)
   
        
    def load_data(self,array, arraye, labels, perm=True):
        if perm:
            p=torch.randperm(array.shape[0])  
            array=array[p]
            arraye=arraye[p]
            labels=labels[p]
        vg = []
        data = []
        d_edges = []
        for i in range(array.shape[0]):
            x=torch.from_numpy(array[i][0].astype(np.float64)).cuda().transpose(0,1)
            e=torch.from_numpy(arraye[i][0].astype(np.int64)).cuda()
            data+=[x]
            d_edges += [e]

        trainingedges =d_edges[:self.trainsize]
        trainingdata =data[:self.trainsize]
        traininglabels =labels[:self.trainsize]
        testedges = d_edges[self.trainsize:self.trainsize+self.testsize]
        testdata = data[self.trainsize:self.trainsize+self.testsize]
        testlabels = labels[self.trainsize:self.trainsize+self.testsize]  
        return trainingedges,trainingdata, traininglabels,testedges,testdata, testlabels
        
    def evaluate_loss(self,data,edges,labels,batchsize,verbose,train= False):
        loss = 0
        for i in range(int(len(data)/batchsize)):  
            out = self.C(data[i*batchsize:(i+1)*batchsize], edges[i*batchsize:(i+1)*batchsize])
            batch_loss = self.Loss(out,labels[i*batchsize:(i+1)*batchsize])
            if verbose:
                print ("Batch:"+str(i)+"/"+str(int(len(data)/batchsize)),batch_loss.item()/batchsize, end="\r")
            if train:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step() 
            loss+=batch_loss/(len(data)*batchsize)
        return loss
    
    def evaluate_accuracy(self,data,edges,labels,batchsize):
        correct = 0
        for i in range(int(len(data)/batchsize)):  
            out = self.C(data[i*batchsize:(i+1)*batchsize], edges[i*batchsize:(i+1)*batchsize])
            pred_choice =out.max(1)[1]
            correct+=(pred_choice ==  labels[i*batchsize:(i+1)*batchsize]).sum()
        return 100*correct/(int(len(data)/batchsize)*batchsize)
    
    def train(self, epochs,trainingdata,trainingedges,traininglabels,testdata,testedges,testlabels, batchsize=10, perm = True, verbose= True):
        for epoch in range(epochs):
            if perm:
                p=torch.randperm(self.trainsize) 
                traininglabels=traininglabels[p]
                trainingdata = [trainingdata[i] for i in p.tolist()]
                trainingedges = [trainingedges[i] for i in p.tolist()]            
            epochloss=self.evaluate_loss(trainingdata,trainingedges,traininglabels,batchsize,verbose,train=True)
            validation_accuracy=self.evaluate_accuracy(testdata,testedges,testlabels,batchsize)
            if verbose:
                print("Epoch: "+ str(epoch),epochloss.item(),validation_accuracy.item())
                
    def save(self,classifier_file):
        torch.save(self.C.state_dict(), classifier_file)
        
    def load(self,classifier_file):
        self.C.load_state_dict(torch.load(classifier_file))

        
class VG_R:
    def __init__(self, template, template_edges,trainsize, testsize):
        self.template=template
        self.template_edges=template_edges
        self.E=Encoder(template_edges, template).cuda().double()
        self.D=Decoder(template).cuda().double()
        self.VL=VariLoss(template_edges, template).cuda().double()
        self.trainsize=trainsize
        self.testsize =testsize
        params_to_optimize = [{'params': self.E.parameters()},{'params': self.D.parameters()}]
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=0.001, betas=(0.9, 0.999))
   
        
    def load_data(self,array, arraye, perm=True):
        if perm:
            p=torch.randperm(array.shape[0])  
            array=array[p]
            arraye=arraye[p]
        vg = []
        data = []
        d_edges = []
        for i in range(array.shape[0]):
            x=torch.from_numpy(array[i][0].astype(np.float64)).cuda().transpose(0,1)
            e=torch.from_numpy(arraye[i][0].astype(np.int64)).cuda()
            data+=[x]
            d_edges += [e]

        trainingedges =d_edges[:self.trainsize]
        trainingdata =data[:self.trainsize]
        testedges = d_edges[self.trainsize:self.trainsize+self.testsize]
        testdata = data[self.trainsize:self.trainsize+self.testsize]  
        return trainingedges,trainingdata,testedges,testdata
        
    def evaluate_loss(self,data,edges,batchsize,verbose,train= False):
        loss = 0
        for i in range(int(len(data)/batchsize)):  
            reconstructed = self.D(self.E(data[i*batchsize:(i+1)*batchsize], edges[i*batchsize:(i+1)*batchsize]))
            batch_loss = self.VL(reconstructed, data[i*batchsize:(i+1)*batchsize], edges[i*batchsize:(i+1)*batchsize])
            if verbose:
                print ("Batch:"+str(i)+"/"+str(int(len(data)/batchsize)),batch_loss.item()/batchsize, end="\r")
            if train:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step() 
            loss+=batch_loss/(len(data))
        return loss
    
    def train(self, epochs,trainingdata,trainingedges,testdata,testedges, batchsize=10, perm = True, verbose= True):
        for epoch in range(epochs):
            if perm:
                p=torch.randperm(self.trainsize)    
                trainingdata = [trainingdata[i] for i in p.tolist()]
                trainingedges = [trainingedges[i] for i in p.tolist()]            
            epochloss=self.evaluate_loss(trainingdata,trainingedges,batchsize,verbose,train=True)
            validation_loss=self.evaluate_loss(testdata,testedges,batchsize,verbose=False,train=False)
            if verbose:
                print("Epoch: "+ str(epoch),epochloss.item(),validation_loss.item())
                
    def save(self,encoder_file, decoder_file):
        torch.save(self.E.state_dict(), encoder_file)
        torch.save(self.D.state_dict(), decoder_file)
        
    def load(self,encoder_file, decoder_file):
        self.E.load_state_dict(torch.load(encoder_file))
        self.D.load_state_dict(torch.load(decoder_file))