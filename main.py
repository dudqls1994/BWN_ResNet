import os
from PIL import Image
import argparse

import binarization
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from model import vgg

import torch.nn.init as init

#Set My Image Dataset
class ImageDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    # init about dataset
    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    # read imagedataset(image, labels)
    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)
        return {'image': image, 'label': self.labels[index]}

    # return dataset size
    def __len__(self):
        return self.length


# save best accuracy & best model parameter
def save_state(model, best_acc):
    print('--------save model--------')
    state = {
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
    }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)
    torch.save(state, 'checkpoint/'+'RFBNN_' + args.model +'_'+ str(args.input) +'_'+ str(args.lr)+'_'+ str(args.binarize) + '.pt') # save checkpoint

# train function
def train(epoch):
    model.train()
    for i_batch, item in enumerate(train_loader):
        # weight binarize
        if args.binarize:
            binmodel.F2B()
        images = item['image'].to(device)
        labels = item['label'].to(device)

        # confirm weight filter
        # for m in model.modules():
        #  if isinstance(m, nn.Conv2d):
        #      print(m.weight.data.size())
        #      print(m.weight.data)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Backward
        loss = criterion(outputs, labels)
        loss.backward()
        # B--> F and STE
        if args.binarize:
            binmodel.B2F()
            binmodel.STE()

        optimizer.step()

        if (i_batch + 1) % batch == 0:
            print('Epoch [{}], Loss: {:.4f}, lr : {}'
                  .format( epoch, loss.item(),optimizer.param_groups[0]['lr']))
    return

# test function
def test(epoch):
    global best_acc
    model.eval()
    correct = 0
    total = 0
    # weight binarize
    if args.binarize:
        binmodel.F2B()
    for item in test_loader:
        images = item['image'].to(device)
        labels = item['label'].to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
    # B--> F
    if args.binarize:
        binmodel.B2F()
    acc = 100 * correct / total
    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc) # model save
    print('{} Test images --> Test Accuracy : {} %'.format(total,100 * correct / total))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return


def adjust_learning_rate(optimizer, epoch):
    update_list = [15,30]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store',default= 10, type=int,
                        help='insert batch size')
    parser.add_argument('--epoch', action='store',default= 40, type=int,
                        help='insert epoch size')
    parser.add_argument('--input', action='store',default= 32, type=int,
                        help='insert input size')
    parser.add_argument('--model', action='store', default='vgg8',
                        help='structure of model')
    parser.add_argument('--lr', action='store', default=0.1,type=float,
                        help='insert learning rate size')
    parser.add_argument('--load', action='store', default=None,
                        help='insert your pretrained model path')
    parser.add_argument('--binarize', action='store', default=False,
                        help='if you want bnn, set true')
    args = parser.parse_args()
    print('==> Options:', args)

    epoch = args.epoch
    batch = args.batch

    #data augementation
    transform_train = transforms.Compose([
        transforms.Resize((args.input, args.input)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.input, args.input)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])
    #YB Dataset Load
    train_dataset = ImageDataset(data_set_path="/mnt/hdd2/ProjectAd/YBdata/train",
                                 transforms=transform_train)
    test_dataset = ImageDataset(data_set_path="/mnt/hdd2/ProjectAd/YBdata/test",
                                transforms=transform_test)
    #Set parameter
    train_loader = DataLoader(train_dataset,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch,
                             shuffle=False,
                             num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = train_dataset.num_classes
    #set model
    if args.model == 'vgg8':
        model = vgg.Net().to(device)
    else:
        raise Exception(args.arch + 'is not support')
    # Use pretrained model
    if not args.load:
        print('Init model weight')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight,mode='fan_out', nonlinearity='relu') #He init

    else:
        print('Load pretrained model', args.load, '...') #load pretrained model
        pretrained_model = torch.load(args.load)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=0.0005, momentum=0.9)
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
       params += [{'params':[value], 'lr': args.lr,'momentum':0.9}]
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=0.0005, momentum=0.9)

    # model binarization.
    binmodel = binarization.Bin(model)

    for epoch in range(1, epoch):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
