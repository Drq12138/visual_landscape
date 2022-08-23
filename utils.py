import random
from time import time
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import resnet
from torch.autograd.variable import Variable
from scipy import interpolate
import h5py
from sklearn.decomposition import PCA
from tqdm import tqdm

import math


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=1):
    print('Random Seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(args):
    if args.datasets == 'MNIST':
        print('normal dataset!')
        mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
        mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)
    elif args.datasets == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.smalldatasets:
            percent = args.smalldatasets
            path = os.path.join("../data", 'cifar10_' + str(percent) + '_smalldataset')
            # path = 'cifar10_' + str(percent) +  '_smalldataset'
            print('Use ', percent, 'of Datasets')
            print('path:', path)
            ################################################################
            # Use small datasets

            if os.path.exists(path):
                print('read dataset!')
                trainset = torch.load(path)
            else:
                print('make dataset!')
                trainset = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]), download=True)
                N = int(percent * len(trainset))
                trainset.targets = trainset.targets[:N]
                trainset.data = trainset.data[:N]

                torch.save(trainset, path)
                print(N)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            print('dataset size: ', len(train_loader.dataset))
        else:
            print('normal dataset!')
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]), download=True),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.datasets == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        if args.smalldatasets:
            percent = args.smalldatasets
            path = os.path.join("../data", 'cifar100_' + str(percent) + '_smalldataset')
            # path = 'cifar100_' + str(percent) +  '_smalldataset'
            print('Use ', percent, 'of Datasets')
            print('path:', path)
            ################################################################
            # Use small datasets

            if os.path.exists(path):
                print('load dataset!')
                trainset = torch.load(path)
            else:
                print('create dataset!')
                trainset = datasets.CIFAR100(root='../data', train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]), download=True)
                N = int(percent * len(trainset))
                trainset.targets = trainset.targets[:N]
                trainset.data = trainset.data[:N]

                torch.save(trainset, path)
                print(N)

            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            print('dataset size: ', len(train_loader.dataset))

        else:
            print('normal dataset!')
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(root='../data', train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]), download=True),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def get_model(args):
    if args.datasets == 'CIFAR10' or args.datasets == 'MNIST':
        num_class = 10
    elif args.datasets == 'CIFAR100':
        num_class = 100

    if args.datasets == 'CIFAR100':
        if args.arch == 'vgg16':
            from models.vgg import vgg16_bn
            net = vgg16_bn()
        elif args.arch == 'vgg13':
            from models.vgg import vgg13_bn
            net = vgg13_bn()
        elif args.arch == 'vgg11':
            from models.vgg import vgg11_bn
            net = vgg11_bn()
        elif args.arch == 'vgg19':
            from models.vgg import vgg19_bn
            net = vgg19_bn()
        elif args.arch == 'densenet121':
            from models.densenet import densenet121
            net = densenet121()
        elif args.arch == 'densenet161':
            from models.densenet import densenet161
            net = densenet161()
        elif args.arch == 'densenet169':
            from models.densenet import densenet169
            net = densenet169()
        elif args.arch == 'densenet201':
            from models.densenet import densenet201
            net = densenet201()
        elif args.arch == 'googlenet':
            from models.googlenet import googlenet
            net = googlenet()
        elif args.arch == 'inceptionv3':
            from models.inceptionv3 import inceptionv3
            net = inceptionv3()
        elif args.arch == 'inceptionv4':
            from models.inceptionv4 import inceptionv4
            net = inceptionv4()
        elif args.arch == 'inceptionresnetv2':
            from models.inceptionv4 import inception_resnet_v2
            net = inception_resnet_v2()
        elif args.arch == 'xception':
            from models.xception import xception
            net = xception()
        elif args.arch == 'resnet18':
            from resnet import resnet18
            net = resnet18()
        elif args.arch == 'resnet34':
            from resnet import resnet34
            net = resnet34()
        elif args.arch == 'resnet50':
            from resnet import resnet50
            net = resnet50()
        elif args.arch == 'resnet101':
            from resnet import resnet101
            net = resnet101()
        elif args.arch == 'resnet152':
            from resnet import resnet152
            net = resnet152()
        elif args.arch == 'preactresnet18':
            from models.preactresnet import preactresnet18
            net = preactresnet18()
        elif args.arch == 'preactresnet34':
            from models.preactresnet import preactresnet34
            net = preactresnet34()
        elif args.arch == 'preactresnet50':
            from models.preactresnet import preactresnet50
            net = preactresnet50()
        elif args.arch == 'preactresnet101':
            from models.preactresnet import preactresnet101
            net = preactresnet101()
        elif args.arch == 'preactresnet152':
            from models.preactresnet import preactresnet152
            net = preactresnet152()
        elif args.arch == 'resnext50':
            from models.resnext import resnext50
            net = resnext50()
        elif args.arch == 'resnext101':
            from models.resnext import resnext101
            net = resnext101()
        elif args.arch == 'resnext152':
            from models.resnext import resnext152
            net = resnext152()
        elif args.arch == 'shufflenet':
            from models.shufflenet import shufflenet
            net = shufflenet()
        elif args.arch == 'shufflenetv2':
            from models.shufflenetv2 import shufflenetv2
            net = shufflenetv2()
        elif args.arch == 'squeezenet':
            from models.squeezenet import squeezenet
            net = squeezenet()
        elif args.arch == 'mobilenet':
            from models.mobilenet import mobilenet
            net = mobilenet()
        elif args.arch == 'mobilenetv2':
            from models.mobilenetv2 import mobilenetv2
            net = mobilenetv2()
        elif args.arch == 'nasnet':
            from models.nasnet import nasnet
            net = nasnet()
        elif args.arch == 'attention56':
            from models.attention import attention56
            net = attention56()
        elif args.arch == 'attention92':
            from models.attention import attention92
            net = attention92()
        elif args.arch == 'seresnet18':
            from models.senet import seresnet18
            net = seresnet18()
        elif args.arch == 'seresnet34':
            from models.senet import seresnet34
            net = seresnet34()
        elif args.arch == 'seresnet50':
            from models.senet import seresnet50
            net = seresnet50()
        elif args.arch == 'seresnet101':
            from models.senet import seresnet101
            net = seresnet101()
        elif args.arch == 'seresnet152':
            from models.senet import seresnet152
            net = seresnet152()
        elif args.arch == 'wideresnet':
            from models.wideresidual import wideresnet
            net = wideresnet()
        elif args.arch == 'stochasticdepth18':
            from models.stochasticdepth import stochastic_depth_resnet18
            net = stochastic_depth_resnet18()
        elif args.arch == 'efficientnet':
            from models.efficientnet import efficientnet
            net = efficientnet(1, 1, 100, bn_momentum=0.9)
        elif args.arch == 'stochasticdepth34':
            from models.stochasticdepth import stochastic_depth_resnet34
            net = stochastic_depth_resnet34()
        elif args.arch == 'stochasticdepth50':
            from models.stochasticdepth import stochastic_depth_resnet50
            net = stochastic_depth_resnet50()
        elif args.arch == 'stochasticdepth101':
            from models.stochasticdepth import stochastic_depth_resnet101
            net = stochastic_depth_resnet101()
        else:
            net = resnet.__dict__[args.arch](num_classes=num_class)

        return net
    return resnet.__dict__[args.arch](num_classes=num_class)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_weight_list(net):
    """
    return weight of model
    :param net:
    :return: list, element of tensor
    """
    return [p.data for p in net.parameters()]


def get_random_weights(weights):
    return [torch.randn(w.size()).cuda() for w in weights]


def get_random_states(states):
    return [torch.randn(w.size()) for k, w in states.items()]


def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm() / direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
    assert (len(direction) == len(states))
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def create_random_direction(weight_list, weight_type='weight', ignore='biasbn', norm='filter'):
    """
    given model weight list, return random direction
    :param weight_list:
    :param weight_type: default weight
    :param ignore:
    :param norm:
    :return: list of direction, element of tensor
    """

    if weight_type == 'weight':
        direction_list = get_random_weights(weight_list)
        normalize_directions_for_weights(direction_list, weight_list, norm, ignore)
    # elif weight_type == 'state':
    #     states = model.state_dict()
    #     direction = get_random_states(states)
    #     normalize_directions_for_states(direction, states, norm, ignore)
    return direction_list


def divide_param(vec, weight_list, weight_type='weight'):
    """
    given flat vec, divide it according to weight
    :param vec: tensor
    :param weight_list: list of tensor
    :param weight_type:
    :return: list of tensor
    """
    index = 0
    new_vec = []
    if weight_type == 'weight':
        for w in weight_list:
            new_vec.append((vec[index:index + w.numel()]).view(w.size()))
            index += w.numel()
    elif weight_type == 'state':
        for k, s in weight_list.items():
            new_vec.append(torch.tensor((vec[index:index + s.numel()])).view(s.size()).cuda())
            index += s.numel()

    return new_vec


def create_pca_direction(model, weight_list, weight_type, filesnames, ignore='biasbn', norm='filter'):
    if weight_type == 'weight':
        mats = []
        for file in filesnames:
            checkpoint = torch.load(file)
            model.load_state_dict(checkpoint['state_dict'])
            temp_weight_list = get_weight_list(model)
            temp_weight_flat = flat_param(temp_weight_list, weight_type)
            mats.append(temp_weight_flat.cpu())
        mats = np.vstack(mats)
        mats_new = mats[:-1] - mats[-1]

        pca = PCA(n_components=2)
        pca.fit_transform(mats_new)
        principalComponents = np.array(pca.components_)
        x_direction_flat = torch.tensor(principalComponents[0, :])
        x_direction_list = divide_param(x_direction_flat, weight_list)
        y_direction_flat = torch.tensor(principalComponents[1, :])
        y_direction_list = divide_param(y_direction_flat, weight_list)
        normalize_directions_for_weights(x_direction_list, weight_list, norm, ignore)
        normalize_directions_for_weights(y_direction_list, weight_list, norm, ignore)

        return x_direction_list, y_direction_list
    elif weight_type == 'state':
        states = model.state_dict()
        mats = []
        for file in filesnames:
            checkpoint = torch.load(file)
            model.load_state_dict(checkpoint['state_dict'])
            temp_states = model.state_dict()
            temp_flat = flat_param(temp_states, weight_type)
            mats.append(temp_flat)

        mats = np.vstack(mats)
        mats_new = mats[:-1] - mats[-1]

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(mats_new)
        x_direction = np.array(principalComponents[0, :])
        x_direction = divide_param(x_direction, states, weight_type)
        y_direction = np.array(principalComponents[1, :])
        y_direction = divide_param(y_direction, states, weight_type)
        normalize_directions_for_states(x_direction, states, norm, ignore)
        normalize_directions_for_states(y_direction, states, norm, ignore)
        return x_direction, y_direction


def train_net(model, train_loader, optimizer, criterion, epoch):
    model.train()
    losses = AverageMeter()

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))

    return losses.avg


def test(model, test_loader, criterion):
    model.eval()
    accuracys = AverageMeter()
    losses = AverageMeter()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = Variable(data).cuda(), Variable(target).cuda()

            output = model(data)

            loss = criterion(output, target)
            acc = accuracy(output.data, target)[0]
            accuracys.update(acc.item(), data.size(0))
            losses.update(loss.item(), data.size(0))

    return accuracys.avg, losses.avg


def eval_loss(model, criterion, dataloader):
    model.eval()
    losses = AverageMeter()
    accuracy_ave = AverageMeter()
    with torch.no_grad():
        for batch_idx, (input_data, target) in enumerate(dataloader):
            input_data = input_data.cuda()
            target = target.cuda()

            output = model(input_data)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            accuaray = predicted.eq(target).sum().item()

            losses.update(loss.item(), input_data.shape[0])
            accuracy_ave.update(accuaray, input_data.shape[0])

    return losses.avg, accuracy_ave.avg


def h5_to_vtp(vals, xcoordinates, ycoordinates, name, base_dir, log=False, zmax=-1, interp=-1, show_points=False,
              show_polys=True):
    # set this to True to generate points
    # show_points = False
    # #set this to True to generate polygons
    # show_polys = True

    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals[:].ravel()

    # Interpolate the resolution up to the desired amount
    if interp > 0:
        m = interpolate.interp2d(xcoordinates[0, :], ycoordinates[:, 0], vals, kind='cubic')
        x_array = np.linspace(min(x_array), max(x_array), interp)
        y_array = np.linspace(min(y_array), max(y_array), interp)
        z_array = m(x_array, y_array).ravel()

        x_array, y_array = np.meshgrid(x_array, y_array)
        x_array = x_array.ravel()
        y_array = y_array.ravel()

    vtp_file = os.path.join(base_dir, name)
    if zmax > 0:
        z_array[z_array > zmax] = zmax
        vtp_file += "_zmax=" + str(zmax)

    if log:
        z_array = np.log(z_array + 0.1)
        vtp_file += "_log"
    vtp_file += ".vtp"
    print("Here's your output file:{}".format(vtp_file))

    number_points = len(z_array)
    print("number_points = {} points".format(number_points))

    matrix_size = int(math.sqrt(number_points))
    print("matrix_size = {} x {}".format(matrix_size, matrix_size))

    poly_size = matrix_size - 1
    print("poly_size = {} x {}".format(poly_size, poly_size))

    number_polys = poly_size * poly_size
    print("number_polys = {}".format(number_polys))

    min_value_array = [min(x_array), min(y_array), min(z_array)]
    max_value_array = [max(x_array), max(y_array), max(z_array)]
    min_value = min(min_value_array)
    max_value = max(max_value_array)

    averaged_z_value_array = []

    poly_count = 0
    for column_count in range(poly_size):
        stride_value = column_count * matrix_size
        for row_count in range(poly_size):
            temp_index = stride_value + row_count
            averaged_z_value = (z_array[temp_index] + z_array[temp_index + 1] +
                                z_array[temp_index + matrix_size] +
                                z_array[temp_index + matrix_size + 1]) / 4.0
            averaged_z_value_array.append(averaged_z_value)
            poly_count += 1

    avg_min_value = min(averaged_z_value_array)
    avg_max_value = max(averaged_z_value_array)

    output_file = open(vtp_file, 'w')
    output_file.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    output_file.write('  <PolyData>\n')

    if (show_points and show_polys):
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(
                number_points, number_points, number_polys))
    elif (show_polys):
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(
                number_points, number_polys))
    else:
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="">\n'.format(
                number_points, number_points))

    # <PointData>
    output_file.write('      <PointData>\n')
    output_file.write(
        '        <DataArray type="Float32" Name="zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
            min_value_array[2], max_value_array[2]))
    for vertexcount in range(number_points):
        if (vertexcount % 6) is 0:
            output_file.write('          ')
        output_file.write('{}'.format(z_array[vertexcount]))
        if (vertexcount % 6) is 5:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 6) is not 5:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </PointData>\n')

    # <CellData>
    output_file.write('      <CellData>\n')
    if (show_polys and not show_points):
        output_file.write(
            '        <DataArray type="Float32" Name="averaged zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
                avg_min_value, avg_max_value))
        for vertexcount in range(number_polys):
            if (vertexcount % 6) is 0:
                output_file.write('          ')
            output_file.write('{}'.format(averaged_z_value_array[vertexcount]))
            if (vertexcount % 6) is 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) is not 5:
            output_file.write('\n')
        output_file.write('        </DataArray>\n')
    output_file.write('      </CellData>\n')

    # <Points>
    output_file.write('      <Points>\n')
    output_file.write(
        '        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
            min_value, max_value))
    for vertexcount in range(number_points):
        if (vertexcount % 2) is 0:
            output_file.write('          ')
        output_file.write('{} {} {}'.format(x_array[vertexcount], y_array[vertexcount], z_array[vertexcount]))
        if (vertexcount % 2) is 1:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 2) is not 1:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Points>\n')

    # <Verts>
    output_file.write('      <Verts>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_points - 1))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) is 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount))
            if (vertexcount % 6) is 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) is not 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_points))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) is 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount + 1))
            if (vertexcount % 6) is 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) is not 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Verts>\n')

    # <Lines>
    output_file.write('      <Lines>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Lines>\n')

    # <Strips>
    output_file.write('      <Strips>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Strips>\n')

    # <Polys>
    output_file.write('      <Polys>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1))
    if (show_polys):
        polycount = 0
        for column_count in range(poly_size):
            stride_value = column_count * matrix_size
            for row_count in range(poly_size):
                temp_index = stride_value + row_count
                if (polycount % 2) is 0:
                    output_file.write('          ')
                output_file.write('{} {} {} {}'.format(temp_index, (temp_index + 1), (temp_index + matrix_size + 1),
                                                       (temp_index + matrix_size)))
                if (polycount % 2) is 1:
                    output_file.write('\n')
                else:
                    output_file.write(' ')
                polycount += 1
        if (polycount % 2) is 1:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys))
    if (show_polys):
        for polycount in range(number_polys):
            if (polycount % 6) is 0:
                output_file.write('          ')
            output_file.write('{}'.format((polycount + 1) * 4))
            if (polycount % 6) is 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (polycount % 6) is not 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Polys>\n')

    output_file.write('    </Piece>\n')
    output_file.write('  </PolyData>\n')
    output_file.write('</VTKFile>\n')
    output_file.write('')
    output_file.close()

    print("Done with file:{}".format(vtp_file))


def flat_param(parameter, weight_type='weight'):
    """
    given param list,return the flat one
    :param parameter:
    :param weight_type:
    :return: tensor
    """
    if weight_type == 'weight':
        # return np.concatenate([ar.cpu().flatten() for ar in parameter], axis=None)
        return torch.cat([ar.view(-1) for ar in parameter])
    elif weight_type == 'state':
        return np.concatenate([ar.cpu().flatten() for (k, ar) in parameter.items()], axis=None)


def cal_path(model, weight_list, filenames, direction, dataloader, criterion):
    """
    plot the path of training process
    :param model:
    :param weight_list:
    :param filenames:
    :param direction: list:[dx_list, dy_list]
    :param dataloader:
    :param criterion:
    :return:
    """
    dx_list, dy_list = direction[0], direction[1]
    matrix = [flat_param(dx_list).cpu(), flat_param(dy_list).cpu()]
    matrix = np.vstack(matrix)
    matrix = matrix.T
    origin_weight_flat = flat_param(weight_list)
    coefs = []
    path_loss = []
    path_acc = []
    for file in tqdm(filenames):
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['state_dict'])
        temp_weight = get_weight_list(model)
        temp_flat = flat_param(temp_weight)
        b = temp_flat - origin_weight_flat
        coefs.append(np.hstack(np.linalg.lstsq(matrix, b.cpu(), rcond=None)[0]))
        acc, loss = test(model, dataloader, criterion)
        path_loss.append(loss)
        path_acc.append(acc)
    return np.array(coefs), np.array(path_loss), np.array(path_acc)


#
# def get_coefs(model, weight, state, filesnames, direction, dataloader, criterion, weight_type='weight'):
#     dx, dy = direction[0], direction[1]
#     matrix = [flat_param(dx), flat_param(dy)]
#     matrix = np.vstack(matrix)
#     matrix = matrix.T
#
#     if weight_type == 'weight':
#         origin_weight = flat_param(weight, weight_type)
#         coefs = []
#         path_loss = []
#         path_acc = []
#         for file in tqdm(filesnames):
#             checkpoint = torch.load(file)
#             model.load_state_dict(checkpoint['state_dict'])
#
#             temp_weight = get_weight_list(model)
#
#             temp_flat = flat_param(temp_weight)
#             b = temp_flat - origin_weight
#             coefs.append(np.hstack(np.linalg.lstsq(matrix, b, rcond=None)[0]))
#             acc, loss = test(model, dataloader, criterion)
#             path_loss.append(loss)
#             path_acc.append(acc)
#
#     elif weight_type == 'state':
#         origin_state = flat_param(state, weight_type)
#         coefs = []
#         path_loss = []
#         path_acc = []
#         for file in tqdm(filesnames):
#             checkpoint = torch.load(file)
#             model.load_state_dict(checkpoint['state_dict'])
#
#             temp_state = model.state_dict()
#
#             temp_flat = flat_param(temp_state, weight_type)
#             b = temp_flat - origin_state
#             coefs.append(np.hstack(np.linalg.lstsq(matrix, b, rcond=None)[0]))
#             acc, loss = test(model, dataloader, criterion)
#             path_loss.append(loss)
#             path_acc.append(acc)
#
#     return np.array(coefs), np.array(path_loss), np.array(path_acc)


def get_model_grad_vec(model):
    # Return the model grad as a vector

    vec = []
    for name, param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)


def get_delta(model, dataloader, criterion):
    """
    given model at certain point, return grad vector
    :param model:
    :param dataloader:
    :param criterion:
    :return: [batch_num, param_sum] saved all the grad of each batch
    """
    model.train()
    total_delta = []
    losses = AverageMeter()

    for batch_id, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        batch_delta = get_model_grad_vec(model)
        total_delta.append(batch_delta)
        losses.update(loss.item(), data.size(0))
    return torch.stack(total_delta, 0), losses.avg


def decomposition_delta(delta_direction_vector, temp_weight_list, origin_weight):
    origin_bias = flat_param(temp_weight_list) - flat_param(origin_weight)
    origin_bias = torch.tensor(origin_bias).cuda()

    project_delta = (torch.dot(delta_direction_vector, origin_bias)) / (
        torch.dot(origin_bias, origin_bias)) * origin_bias
    # max_d = max(project_delta)
    # norm = project_delta.norm()
    search_direction_vector = project_delta - delta_direction_vector

    return search_direction_vector

    pass


def set_weigth(model, weight_list):
    for (p, w) in zip(model.parameters(), weight_list):
        p.data = w


# def cal_loss_direction(model, dataloader, criterion, temp_weight_list, search_direction_vector, t):
#     search_direction_list = divide_param(search_direction_vector, temp_weight_list)
#     for (p, w, s) in zip(model.parameters(), temp_weight_list, search_direction_list):
#         p.data = w + t * s.type_as(w)
#
#     acc, loss = test(model, dataloader, criterion)
#     return acc, loss
#

def cal_direction_weight(temp_weight_list, search_direction_vector, t):
    new_weight_list = []
    search_direction_list = divide_param(search_direction_vector, temp_weight_list)
    for (w, s) in zip(temp_weight_list, search_direction_list):
        new_weight_list.append(w + t * s.type_as(w))

    return new_weight_list


def back_tracking_line_search(model, dataloader, criterion, temp_weight_list, temp_loss, search_direction_vector,
                              delta_direction_vector, alpha=0.5,
                              beta=0.5):
    bias_dot = torch.dot(search_direction_vector, delta_direction_vector)
    last_t = 0.2
    count = 0
    while True:
        new_weight_list = cal_direction_weight(temp_weight_list, search_direction_vector, last_t)
        set_weigth(model, new_weight_list)
        acc, new_loss = test(model, dataloader, criterion)
        count += 1
        if (new_loss < temp_loss + alpha * last_t * bias_dot) or (count > 10):
            break

        last_t = last_t * beta

    return last_t, new_loss, new_weight_list, count
