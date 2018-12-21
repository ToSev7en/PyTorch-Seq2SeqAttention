import torch
import torch.nn as nn
import time

class BasicModule(nn.Module):
   '''
   封装 nn.Module，主要提供 save 和 load 两个方法
   '''

   def __init__(self,opt=None):
       super(BasicModule,self).__init__()
       # 模型的默认名字
       self.model_name = str(type(self))

   def load(self, path):
       '''
       可加载指定路径的模型
       '''
       self.load_state_dict(torch.load(path))

   def save(self, name=None):
       '''
       保存模型，默认使用“模型名字+时间”作为文件名，
       如 AlexNet_0710_23:57:29.pth
       '''
       if name is None:
           prefix = 'checkpoints/' + self.model_name + '_'
           name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
       torch.save(self.state_dict(), name)
       return name


class Flat(nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)