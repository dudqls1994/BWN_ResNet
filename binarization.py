import torch.nn as nn
import numpy

class Bin():
    def __init__(self, model):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        # First filter & last filter binarization.
        start_range = 0
        end_range = count_Conv2d-1
        self.bin_range = numpy.linspace(start_range,end_range, end_range-start_range+1).astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    # weight filter binarization. Float weight-> Binary weight
    def F2B(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.target_modules[index].data.clamp(-1.0, 1.0) # clamp weight filter --> like tanh function
            self.saved_params[index].copy_(self.target_modules[index].data) # save float weight filter

            n = self.target_modules[index].data[0].nelement()  # sum of weight element
            s = self.target_modules[index].data.size()  # weight filter size
            m = self.target_modules[index].data.norm(1, 3, keepdim=True) \
                .sum(2, keepdim=True).sum(1, keepdim=True).div(n)  # calculate mean of weight filter (BWN)
            self.target_modules[index].data = \
                self.target_modules[index].data.sign()#.mul(m.expand(s))



    # Binary weight -> float weight
    def B2F(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    # STE(Straight Through Estimator
    def STE(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0

            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)