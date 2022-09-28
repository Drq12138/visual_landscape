import torch
from utils import flat_param, divide_param, set_weight, get_model_grad_vec
import numpy as np
from torch.autograd.variable import Variable

def update_grad(model, weight_matrix, final_weight_list, origin_loss, direction, dataloader, criterion, args):
    """
    given model and train path, calculate the coordinate of the path and save the grad vector based on the direction
    :param model:
    :param weight_matrix: [100,D]
    :param final_weight_list:
    :param direction:
    :param dataloader:
    :param criterion:
    :return:
    """
    relative_weight_matrix = weight_matrix - weight_matrix[-1, :]

    temp_dx = flat_param(direction[0])
    temp_dy = flat_param(direction[1])
    for epoch in range(args.epoch):
        for batch_id, (data, target) in enumerate(dataloader):

            matrix = [temp_dx.cpu(), temp_dy.cpu()]
            matrix = np.vstack(matrix)
            matrix = matrix.T
            grad_dx = torch.zeros_like(temp_dx)
            grad_dy = torch.zeros_like(temp_dy)
            temp_grad_loss = 0
            for weight_idx in range(len(relative_weight_matrix) - 1):

                temp_weight = relative_weight_matrix[weight_idx, :]
                coefs = np.linalg.lstsq(matrix, temp_weight, rcond=None)[0]
                project_weight = matrix @ coefs.T + weight_matrix[-1, :]
                project_weight_tensor = torch.tensor(project_weight).cuda()
                project_weight_list = divide_param(project_weight_tensor, final_weight_list)
                set_weight(model, project_weight_list)
                data = Variable(data).cuda()
                target = Variable(target).cuda()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                temp_grad_loss += (loss.item() - origin_loss[weight_idx])**2
                grad_vector = get_model_grad_vec(model)

                grad_dx += 2*(loss.item()-origin_loss[weight_idx])*coefs[0]*grad_vector
                grad_dy += 2 * (loss.item() - origin_loss[weight_idx]) * coefs[1] * grad_vector
            print('epoch: {}   batch: {}   loss: {}'.format(epoch, batch_id, temp_grad_loss))
            temp_dx -= grad_dx*args.lr
            temp_dy -= grad_dy * args.lr



    return temp_dx, temp_dy


    #             temp_weight_tensor = torch.tensor(temp_weight + weight_matrix[-1, :]).cuda()
    #             temp_weight_list = divide_param(temp_weight_tensor, final_weight_list)
    #
    #
    #
    #
    #             set_weight(model, temp_weight_list)
    #             temp_output = model(data)
    #             temp_loss = criterion(temp_output, target)
    #             temp_loss = loss.item()
    #             temp_grad_vector = get_model_grad_vec(model)
    #
    # for i in range(len(weight_matrix)):
    #     temp_weight = torch.tensor(weight_matrix[i, :]).cuda()
    #     temp_weight_list = divide_param(temp_weight, final_weight_list)
    #     set_weight(model, temp_weight_list)
    #     acc, loss = test(model, dataloader, criterion)
    #
    #     print("epoch:{}, acc: {}, loss:{} ".format(i, acc, loss))