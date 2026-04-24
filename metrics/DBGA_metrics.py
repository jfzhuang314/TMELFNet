import torch
import numpy as np

def weighting_DSC(y_pred, y_true, class_weights, smooth = 0.1):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability, BNC格式
        y_true [n_classes, x, y, z] one-hot code, BNC格式
        class_weights
        smooth = 1.0
    '''
    # smooth = 1.0
    mdsc = 0.0
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mdsc += w*((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))
        
    return mdsc


def weighting_SEN(y_pred, y_true, class_weights, smooth = 0.1):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    # smooth = 1.
    msen = 0.0
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        msen += w*((intersection + smooth) / (true_flat.sum() + smooth))
        
    return msen


def weighting_PPV(y_pred, y_true, class_weights, smooth =  0.1):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    # smooth = 1.
    mppv = 0.0
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mppv += w*((intersection + smooth) / (pred_flat.sum() + smooth))
        
    return mppv


def weighting_IOU(y_pred, y_true, class_weights, smooth= 0.1):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    # smooth = 1.
    miou = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        union = pred_flat.sum() + true_flat.sum() - intersection
        w = class_weights[c] / class_weights.sum()
        miou += w * ((intersection + smooth) / (union + smooth))

    return miou

def weighting_ACC(y_pred, y_true, class_weights, smooth= 0.1):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    # smooth = 1.
    macc = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    # 计算总样本数
    total_samples = one_hot.numel() // n_classes

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        # 计算每个类别下正确预测的样本数（正例和负例）
        intersection = (pred_flat * true_flat).sum()
        # w = class_weights[c] / class_weights.sum()
        # macc += w * ((correct_predictions + smooth) / (total_samples + smooth))
        macc += intersection

    return macc/total_samples

def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    loss = 0.
    n_classes = y_pred.shape[-1]
    
    for c in range(0, n_classes):
        pred_flat = y_pred[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
       
        # with weight
        w = class_weights[c]/class_weights.sum()
        loss += w*(1 - ((2. * intersection + smooth) /
                         (pred_flat.sum() + true_flat.sum() + smooth)))
       
    return loss


def DSC(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
        
    return dsc


def SEN(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    sen = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
        
    return sen


def PPV(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    ppv = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
        
    return ppv



def accuracy_check2(y_pred, y_true):
    # print('y_pred size: ', y_pred.size())
    #n_pts = y_pred.shape[1] * y_pred.shape[0] # multiply the number of points per mesh with batch size
    n_pts = y_pred.shape[1] 
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    acc = 0.0
    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        #print('intersection: ', intersection)
        acc += intersection

    return acc/n_pts


if __name__ == '__main__':

    y_pred = torch.zeros((3, 4, 4, 2), dtype=torch.float32)
    y_true = torch.zeros((3, 4, 4, 2), dtype=torch.float32)

    # 随机生成第 0 类数据
    mask_0 = torch.rand((4, 4, 2)) > 0.3
    y_pred[0][mask_0] = 0.8 + 0.2 * torch.rand_like(y_pred[0][mask_0])
    y_pred[1][mask_0] = 0.1 * torch.rand_like(y_pred[1][mask_0])
    y_pred[2][mask_0] = 0.1 * torch.rand_like(y_pred[2][mask_0])
    y_true[0][mask_0] = 1

    # 随机生成第 1 类数据
    mask_1 = torch.rand((4, 4, 2)) > 0.5
    y_pred[0][mask_1] = 0.1 * torch.rand_like(y_pred[0][mask_1])
    y_pred[1][mask_1] = 0.8 + 0.2 * torch.rand_like(y_pred[1][mask_1])
    y_pred[2][mask_1] = 0.1 * torch.rand_like(y_pred[2][mask_1])
    y_true[1][mask_1] = 1

    # 生成稀少的第 2 类数据
    mask_2 = torch.rand((4, 4, 2)) > 0.9
    y_pred[0][mask_2] = 0.1 * torch.rand_like(y_pred[0][mask_2])
    y_pred[1][mask_2] = 0.1 * torch.rand_like(y_pred[1][mask_2])
    y_pred[2][mask_2] = 0.8 + 0.2 * torch.rand_like(y_pred[2][mask_2])
    y_true[2][mask_2] = 1

    # 确保每个位置概率和为 1
    y_pred = y_pred / y_pred.sum(dim=0, keepdim=True)

    # 类别权重，稀少类别的权重更高
    class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    print(y_pred.shape, y_true.shape)

    # 调用 weighting_ACC 函数
    for smooth in [1.0, 0.1, 0.01]:
        macc = weighting_ACC(y_pred, y_true, class_weights, smooth)
        miou = weighting_IOU(y_pred, y_true, class_weights, smooth)
        dsc = weighting_DSC(y_pred, y_true, class_weights, smooth)
        sen = weighting_SEN(y_pred, y_true, class_weights, smooth)
        ppv = weighting_PPV(y_pred, y_true, class_weights, smooth)

        print(f"smooth:{smooth},macc:{macc}, miou:{miou}, dsc:{dsc}, sen:{sen}, ppv:{ppv}")