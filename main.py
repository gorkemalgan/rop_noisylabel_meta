import os, sys, gc
import time
import pandas as pd 
import numpy as np
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from PIL import Image 

import torch
from torch import nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
from torch.utils.tensorboard import SummaryWriter

from utils import *
from resnet_torch import resnet50

class torch_dataset(Dataset): 
    def __init__(self, img_paths, labels, transform): 
        self.transform = transform
        self.img_paths = img_paths
        self.labels = labels
    def __getitem__(self, index):  
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('RGB')    
        img = self.transform(image)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.img_paths)

def normal_train(model_path, epochs, net, train_dl, val_dl, test_dl, epoch_offset=0):
    NUM_TRAINDATA = len(train_dl.dataset)
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    if not os.path.exists(model_path):
        for epoch in range(epochs): 
            start_epoch = time.time()
            train_accuracy = AverageMeter()
            train_loss = AverageMeter()

            lr = lr_scheduler(epoch)
            set_learningrate(optimizer, lr)
            net.train()

            for batch_idx, (images, labels) in enumerate(train_dl):
                start = time.time()
                
                # training images and labels
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)

                # compute output
                output, _feats = net(images,get_feat=True)
                _, predicted = torch.max(output.data, 1)

                # training
                loss = criterion_cce(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), labels.size(0)) 
                train_loss.update(loss.item())
                
                if VERBOSE == 2:
                    template = "Progress: {:6.5f}, Accuracy: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                    sys.stdout.write(template.format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_loss.avg, time.time()-start))
            if VERBOSE == 2:
                sys.stdout.flush()  
                
            # evaluate on validation and test data
            val_accuracy, val_loss = evaluate(net, meta_dl, criterion_cce)
            test_accuracy, test_loss = evaluate(net, test_dl, criterion_cce)

            if SAVE_LOGS == 1:
                summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
                summary_writer.add_scalar('test_loss', test_loss, epoch)
                summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
                summary_writer.add_scalar('val_loss', val_loss, epoch)
                summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)

            if VERBOSE > 0:
                template = 'Epoch {}, Accuracy(train,val,test): {:3.1f}/{:3.1f}/{:3.1f}, Loss(train,val,test): {:4.3f}/{:4.3f}/{:4.3f},Learning rate: {}, Time: {:3.1f}({:3.2f})'
                print(template.format(epoch + 1, 
                                    train_accuracy.percentage, val_accuracy, test_accuracy,
                                    train_loss.avg, val_loss, test_loss,  
                                    lr, time.time()-start_epoch, (time.time()-start_epoch)/3600))   
        torch.save(net.cpu().state_dict(), model_path)
        net.to(DEVICE) 
    else:
        net.load_state_dict(torch.load(model_path, map_location=DEVICE))  
        val_accuracy, val_loss = evaluate(net, meta_dl, criterion_cce)
        test_accuracy, test_loss = evaluate(net, test_dl, criterion_cce)
        if VERBOSE > 0:
            print('Pretrained model, Accuracy(val,test): {:3.1f}/{:3.1f}, Loss(val,test): {:4.3f}/{:4.3f}'.format(val_accuracy, test_accuracy,val_loss, test_loss))
        if SAVE_LOGS == 1:
            summary_writer.add_scalar('test_loss', test_loss, epochs+epoch_offset-1)
            summary_writer.add_scalar('test_accuracy', test_accuracy, epochs+epoch_offset-1)
            summary_writer.add_scalar('val_loss', val_loss, epochs+epoch_offset-1)
            summary_writer.add_scalar('val_accuracy', val_accuracy, epochs+epoch_offset-1)

def meta_train(alpha, beta, gamma, epochs, net, feature_encoder, train_dl, meta_dl, test_dl, epoch_offset=0):
    print('alpha:{}, beta:{}, gamma:{}, epochs:{}'.format(alpha, beta, gamma, epochs))

    NUM_TRAINDATA = len(train_dl.dataset)
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    def meta_training_loop(meta_epoch, index, output, labels, yy, images_meta, labels_meta, feats):
        meta_net.train()
        # meta training for predicted labels
        lc = criterion_meta(output, yy)                                            # classification loss
        # train for classification loss with meta-learning
        net.zero_grad()
        grads = torch.autograd.grad(lc, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        for grad in grads:
            grad.detach()
        fast_weights = OrderedDict((name, param - alpha*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))  
        fast_out = net.forward(images_meta,fast_weights)   

        loss_meta = criterion_cce(fast_out, labels_meta)
        loss_compatibility = criterion_cce(yy, labels)
        loss_all = loss_meta + gamma*loss_compatibility

        optimizer_meta_net.zero_grad()
        meta_net.zero_grad()
        loss_all.backward(retain_graph=True)
        optimizer_meta_net.step()

        # update labels
        meta_net.eval()
        _yy = meta_net(feats).detach()
        new_y[meta_epoch+1,index,:] = _yy.cpu().numpy()
        del grads

        # training base network
        lc = criterion_meta(output, _yy)                                # classification loss
        le = -torch.mean(torch.mul(softmax(output), logsoftmax(output)))# entropy loss
        loss = lc + le                                                  # overall loss
        optimizer.zero_grad()
        net.zero_grad()
        loss.backward()
        optimizer.step()
        meta_net.train()
        
        return loss

    def meta_training():
        # initialize parameters and buffers
        t_meta_loader_iter = iter(meta_dl) 
        labels_yy = np.zeros(NUM_TRAINDATA)
        test_acc_best = 0
        val_acc_best = 0
        epoch_best = 0
        for epoch in range(epochs): 
            start_epoch = time.time()
            train_accuracy = AverageMeter()
            train_loss = AverageMeter()
            train_accuracy_meta = AverageMeter()
            label_similarity = AverageMeter()
            meta_epoch = epoch

            lr = lr_scheduler(epoch)
            set_learningrate(optimizer, lr)
            net.train()
            meta_net.train()
            grads_dict = OrderedDict((name, 0) for (name, param) in meta_net.named_parameters()) 

            for batch_idx, (images, labels) in enumerate(train_dl):
                start = time.time()
                index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
                
                # training images and labels
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)

                # compute output
                output, _ = net(images,get_feat=True)
                _, predicted = torch.max(output.data, 1)

                # predict labels
                feats = torch.tensor(features[index], dtype=torch.float, device=DEVICE)
                yy = meta_net(feats)
                _, labels_yy[index] = torch.max(yy.cpu(), 1)
                # meta training images and labels
                try:
                    images_meta, labels_meta = next(t_meta_loader_iter)
                except StopIteration:
                    t_meta_loader_iter = iter(meta_dl)
                    images_meta, labels_meta = next(t_meta_loader_iter)
                    images_meta, labels_meta = images_meta[:labels.size(0)], labels_meta[:labels.size(0)]
                images_meta, labels_meta = images_meta.to(DEVICE), labels_meta.to(DEVICE)
                images_meta, labels_meta = torch.autograd.Variable(images_meta), torch.autograd.Variable(labels_meta)
                
                #with torch.autograd.detect_anomaly():
                loss = meta_training_loop(meta_epoch, index, output, labels, yy, images_meta, labels_meta, feats)

                train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), labels.size(0)) 
                train_loss.update(loss.item())
                train_accuracy_meta.update(predicted.eq(torch.tensor(labels_yy[index]).to(DEVICE)).cpu().sum().item(), predicted.size(0)) 
                label_similarity.update(labels.eq(torch.tensor(labels_yy[index]).to(DEVICE)).cpu().sum().item(), labels.size(0))

                # keep log of gradients
                for tag, parm in meta_net.named_parameters():
                    if parm.grad != None:
                        grads_dict[tag] += parm.grad.data.cpu().numpy()
                del feats
                gc.collect()

                if VERBOSE == 2:
                    template = "Progress: {:6.5f}, Accuracy: {:5.4f}, Accuracy Meta: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                    sys.stdout.write(template.format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_accuracy_meta.percentage, train_loss.avg, time.time()-start))
            if VERBOSE == 2:
                sys.stdout.flush()           

            if SAVE_LOGS == 1:
                np.save(log_dir + "y.npy", new_y)
            # evaluate on validation and test data
            val_accuracy, val_loss = evaluate(net, meta_dl, criterion_cce)
            test_accuracy, test_loss = evaluate(net, test_dl, criterion_cce)
            if val_accuracy > val_acc_best: 
                val_acc_best = val_accuracy
                test_acc_best = test_accuracy
                epoch_best = epoch

            if SAVE_LOGS == 1:
                summary_writer.add_scalar('train_loss', train_loss.avg, epoch+epoch_offset)
                summary_writer.add_scalar('test_loss', test_loss, epoch+epoch_offset)
                summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch+epoch_offset)
                summary_writer.add_scalar('test_accuracy', test_accuracy, epoch+epoch_offset)
                summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch+epoch_offset)
                summary_writer.add_scalar('val_loss', val_loss, epoch+epoch_offset)
                summary_writer.add_scalar('val_accuracy', val_accuracy, epoch+epoch_offset)
                summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch+epoch_offset)
                summary_writer.add_scalar('label_similarity', label_similarity.percentage, epoch+epoch_offset)
                for tag, parm in meta_net.named_parameters():
                    summary_writer.add_histogram('grads_'+tag, grads_dict[tag], epoch+epoch_offset)

            if VERBOSE > 0:
                template = 'Epoch {}, Accuracy(train,meta_train,val,test): {:3.1f}/{:3.1f}/{:3.1f}/{:3.1f}, Loss(train,val,test): {:4.3f}/{:4.3f}/{:4.3f}, Label similarity: {:6.3f}, Hyper-params(alpha,beta,gamma): {:3.2f}/{:5.4f}/{:3.2f}, Time: {:3.1f}({:3.2f})'
                print(template.format(epoch + 1, 
                                    train_accuracy.percentage, train_accuracy_meta.percentage, val_accuracy, test_accuracy,
                                    train_loss.avg, val_loss, test_loss,  
                                    label_similarity.percentage, alpha, beta, gamma,
                                    time.time()-start_epoch, (time.time()-start_epoch)/3600))

        print('Train acc: {:3.1f}, Validation acc: {:3.1f}-{:3.1f}, Test acc: {:3.1f}-{:3.1f}, Best epoch: {}, Num meta-data: {},Hyper-params(alpha,beta,gamma): {:3.2f}/{:5.4f}/{:3.2f}'.format(
            train_accuracy.percentage, val_accuracy, val_acc_best, test_accuracy, test_acc_best, epoch_best, args.metadata_num, alpha, beta, gamma))
        if SAVE_LOGS == 1:
            summary_writer.close()
            torch.save(net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))
        return val_acc_best, test_acc_best, epoch_best

    def init_labels():
        new_y = np.zeros([NUM_META_EPOCHS+1,NUM_TRAINDATA,NUM_CLASSES])
        y_init = np.zeros([NUM_TRAINDATA,NUM_CLASSES])
        for batch_idx, (_, labels) in enumerate(train_dl):
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
            onehot = torch.zeros(labels.size(0), NUM_CLASSES).scatter_(1, labels.view(-1, 1), 1).cpu().numpy()
            y_init[index, :] = onehot
        new_y[0] = y_init
        return new_y

    def extract_features():
        features = np.zeros((NUM_TRAINDATA,NUM_FEATURES))
        outs = np.zeros((NUM_TRAINDATA,NUM_CLASSES))
        losses = np.zeros(NUM_TRAINDATA)
        c = nn.CrossEntropyLoss(reduction='none').to(DEVICE)
        for batch_idx, (images, labels) in enumerate(train_dl):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
            output, feats = feature_encoder(images,get_feat=True)
            features[index] = feats.cpu().detach().numpy()
            outs[index] = output.cpu().detach().numpy()
            loss = c(output, labels)
            losses[index] = loss.detach().cpu().numpy()
        return features, losses, outs

    class MetaNet(nn.Module):
        def __init__(self, input, output):
            super(MetaNet, self).__init__()
            layer1_size = input
            layer2_size = int(input/2)
            self.linear1 = nn.Linear(layer1_size, layer1_size)
            self.linear2 = nn.Linear(layer1_size, layer2_size)
            self.linear3 = nn.Linear(layer2_size, output)
            self.bn1 = nn.BatchNorm1d(layer1_size)
            self.bn2 = nn.BatchNorm1d(layer2_size)
        def forward(self, x):
            x = F.relu(self.bn1(self.linear1(x)))
            x = F.relu(self.bn2(self.linear2(x)))
            out = self.linear3(x)
            return softmax(out)
    meta_net = MetaNet(NUM_FEATURES, NUM_CLASSES).to(DEVICE)
    optimizer_meta_net = torch.optim.Adam(meta_net.parameters(), beta, weight_decay=1e-4)
    meta_net.train()

    # initialize predicted labels with given labels
    new_y = init_labels()
    # extract features for all training data
    features, _, _ = extract_features() 

    # meta training
    return meta_training()

def get_model(num_classes):
    try:
        model = resnet50(pretrained=True)
    except:
        model = resnet50(pretrained=False)
        model.load_state_dict(torch.load('resnet50.pt'))
    model.fc = nn.Linear(2048,2)
    model.to(DEVICE)
    model.train()
    return model

def lr_scheduler(epoch):
    if epoch < 5:
        return 1e-3
    else:
        return 1e-4

def set_learningrate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(net, dataloader, criterion):
    eval_accuracy = AverageMeter()
    eval_loss = AverageMeter()

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs) 
            loss = criterion(outputs, targets) 
            _, predicted = torch.max(outputs.data, 1) 
            eval_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
            eval_loss.update(loss.item())
    return eval_accuracy.percentage, eval_loss.avg

def get_dataset_dr():
    dr_image_dir = os.path.join('data', 'DR')
    df = pd.read_csv(os.path.join(dr_image_dir, 'trainLabels_cropped.csv'))
    df['path'] = df['image'].map(lambda x: os.path.join(dr_image_dir,'resized_train_cropped/resized_train_cropped','{}.jpeg'.format(x)))
    df = df.drop(columns=['image'])
    df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
    df['level'] = (df['level'] > 1).astype(int) # Disease or no disease
    #print(df.head(10))

    img_paths, labels = df['path'].tolist(), df['level'].tolist() 
    x_train, x_val, y_train, y_val = train_test_split(img_paths, labels, test_size=0.2, random_state=RANDOM_SEED)

    train_ds = torch_dataset(x_train, y_train,transform_train)
    val_ds = torch_dataset(x_val, y_val,transform_val)
    train_dataloader = torch.utils.data.DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS)
    val_dataloader = torch.utils.data.DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)
    return train_dataloader, val_dataloader

def read_dataset_rop():         
    dr_image_dir = os.path.join('data', 'ROP')
    df = pd.read_csv(os.path.join(dr_image_dir, 'images.csv'))
    df['path'] = df['image'].map(lambda x: os.path.join(dr_image_dir,'ROP-All2-imgs','{}.jpeg'.format(x)))
    df = df.drop(columns=['image'])
    #df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
    img_paths, labels_saban, labels_berker, labels_banu = np.array(df['path'].tolist()), np.array(df['saban'].tolist()), np.array(df['berker'].tolist()), np.array(df['banu'].tolist())

    # correction for berker
    labels_berker[labels_berker=='nan'] = labels_saban[labels_berker=='nan']

    labels_saban[labels_saban=='normal'] = 0
    labels_saban[labels_saban=='preplus'] = 1
    labels_saban[labels_saban=='plus'] = 2
    labels_berker[labels_berker=='normal'] = 0
    labels_berker[labels_berker=='preplus'] = 1
    labels_berker[labels_berker=='plus'] = 2
    labels_banu[labels_banu=='normal'] = 0
    labels_banu[labels_banu=='preplus'] = 1
    labels_banu[labels_banu=='plus'] = 2

    return img_paths, labels_saban.astype(int), labels_berker.astype(int), labels_banu.astype(int)

def split_train_meta_test(labels_saban, labels_berker, labels_banu):
    consensus_sbe = labels_saban == labels_berker
    consensus_sba = labels_saban == labels_banu
    consensus_bb = labels_berker == labels_banu
    consensus_idx = np.bitwise_and(consensus_sbe,consensus_sba,consensus_bb)
    consensus_idx = np.where(consensus_idx)[0]
    nonconsensus_idx = np.setdiff1d(np.arange(len(labels_saban)), consensus_idx)
    idx_rest, meta_idx, _, _ = train_test_split(consensus_idx, labels_saban[consensus_idx], test_size=args.metadata_num, random_state=RANDOM_SEED)
    idx_rest, test_idx, _, _ = train_test_split(idx_rest, labels_saban[idx_rest], test_size=args.testdata_num, random_state=RANDOM_SEED)
    train_idx = np.concatenate((nonconsensus_idx, idx_rest), axis=0)
    return np.array(train_idx), np.array(meta_idx), np.array(test_idx)

def get_dataset_labels(labels_saban, labels_berker, labels_banu, algorithm):
    if algorithm == 'saban':
        return labels_saban
    elif algorithm == 'berker':
        return labels_saban
    elif algorithm == 'banu':
        return labels_saban
    elif algorithm == 'mv':
        onehot_saban = np.zeros((labels_saban.size, NUM_CLASSES))
        onehot_saban[np.arange(labels_saban.size),labels_saban] = 1
        onehot_berker = np.zeros((labels_berker.size, NUM_CLASSES))
        onehot_berker[np.arange(labels_berker.size),labels_saban] = 1
        onehot_banu = np.zeros((labels_banu.size, NUM_CLASSES))
        onehot_banu[np.arange(labels_banu.size),labels_banu] = 1
        labels_all = onehot_saban + onehot_berker + onehot_banu
        return np.argmax(labels_all, axis=1)
    elif algorithm == 'soft':
        onehot_saban = np.zeros((labels_saban.size, NUM_CLASSES))
        onehot_saban[np.arange(labels_saban.size),labels_saban] = 1
        onehot_berker = np.zeros((labels_berker.size, NUM_CLASSES))
        onehot_berker[np.arange(labels_berker.size),labels_saban] = 1
        onehot_banu = np.zeros((labels_banu.size, NUM_CLASSES))
        onehot_banu[np.arange(labels_banu.size),labels_banu] = 1
        labels_all = onehot_saban + onehot_berker + onehot_banu
        return labels_all/3
    else:
        assert True , 'wrong labeling algorithm!'

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--batch_size', required=False, type=int, default=32,
        help="Number of gpus to be used")
    parser.add_argument('-i', '--gpu_id', required=False, type=int, default=0,
        help="GPU ids to be used")
    parser.add_argument('-f', '--folder_log', required=False, type=str,
        help="Folder name for logs")
    parser.add_argument('-v', '--verbose', required=False, type=int, default=0,
        help="Details of prints: 0(silent), 1(not silent)")
    parser.add_argument('-w', '--num_workers', required=False, type=int, default=2,
        help="Number of parallel workers to parse dataset")
    parser.add_argument('--save_logs', required=False, type=int, default=1,
        help="Either to save log files (1) or not (0)")
    parser.add_argument('-u', '--use_saved', required=False, type=int, default=1,
        help="Either to use presaved files (1) or not (0)")
    parser.add_argument('--seed', required=False, type=int, default=42,
        help="Random seed to be used in simulation")
    
    parser.add_argument('-a', '--alpha', required=False, type=float,
        help="Learning rate for meta iteration")
    parser.add_argument('-b', '--beta', required=False, type=float,
        help="Beta paramter")
    parser.add_argument('-g', '--gamma', required=False, type=float,
        help="Gamma paramter")
    parser.add_argument('-s1', '--stage1', required=False, type=int, default=1,
        help="Epoch num to end stage1 (straight training)")
    parser.add_argument('-s2', '--stage2', required=False, type=int, default=10,
        help="Epoch num to end stage2 (meta training)")
    parser.add_argument('-s3', '--stage3', required=False, type=int, default=1,
        help="")

    parser.add_argument('-m', '--metadata_num', required=False, type=int, default=200,
        help="Number of samples to be used as meta-data")
    parser.add_argument('-t', '--testdata_num', required=False, type=int, default=200,
        help="Number of samples to be used as meta-data")
    parser.add_argument('-l', '--labeler', required=False, type=str, default='soft',
        help="saban, berker, banu, mv, soft")

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    GPU_ID = args.gpu_id
    VERBOSE = args.verbose
    NUM_WORKERS = args.num_workers
    RANDOM_SEED = args.seed
    SAVE_LOGS = args.save_logs
    USE_SAVED = args.use_saved
    NUM_TESTDATA = 100
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_RESIZED = (640,480)
    NUM_CLASSES = 3
    NUM_FEATURES = 2048
    NUM_META_EPOCHS = args.stage2 - args.stage1

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    transform_train = transforms.Compose([
        transforms.Resize(IMG_RESIZED),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ]) 

    transform_val = transforms.Compose([
        transforms.Resize(IMG_RESIZED),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ]) 

    # loss functions
    logsoftmax = nn.LogSoftmax(dim=1).to(DEVICE)
    softmax = nn.Softmax(dim=1).to(DEVICE)
    criterion_cce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCELoss()
    criterion_meta = lambda output, labels: torch.mean(softmax(output)*(logsoftmax(output+1e-10)-torch.log(labels+1e-10)))

    net = get_model(2)
    feature_encoder = get_model(NUM_CLASSES)

    # if logging
    if SAVE_LOGS == 1:
        log_folder = args.folder_log if args.folder_log else 'a{}_b{}_g{}_s{}_m{}_{}'.format(args.alpha, args.beta, args.gamma, args.stage1, args.metadata_num, current_time)
        log_dir = 'logs/{}/'.format(log_folder)
        log_dir_hp = 'logs_hp/{}/'.format(log_folder)
        #clean_emptylogs()
        create_folder(log_dir)
        summary_writer = SummaryWriter(log_dir)
        create_folder(log_dir_hp)
        hp_writer = SummaryWriter(log_dir_hp)

    # get diabetic retinopathy dataset
    train_dl_dr, val_dl_dr = get_dataset_dr()
    # get rop dataset
    img_paths, labels_saban, labels_berker, labels_banu = read_dataset_rop()
    idx_train, idx_meta, idx_test = split_train_meta_test(labels_saban, labels_berker, labels_banu)
    x_train, x_meta, y_meta, x_test, y_test = img_paths[idx_train], img_paths[idx_meta], labels_saban[idx_meta], img_paths[idx_test], labels_saban[idx_test]
    y_train = get_dataset_labels(labels_saban[idx_train], labels_berker[idx_train], labels_banu[idx_train], args.labeler)
    train_ds = torch_dataset(x_train, y_train,transform_train)
    meta_ds  = torch_dataset(x_meta, y_meta,transform_val)
    test_ds  = torch_dataset(x_test, y_test,transform_val)
    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS)
    meta_dl  = torch.utils.data.DataLoader(meta_ds,batch_size=BATCH_SIZE,shuffle=False)
    test_dl  = torch.utils.data.DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False)
    
    if False:
        print('Saban and Berker consensus: {}'.format(np.sum(labels_saban == labels_berker)))
        print('Saban and Banu consensus: {}'.format(np.sum(labels_saban == labels_banu)))
        print('Berker and Banu consensus: {}'.format(np.sum(labels_berker == labels_banu)))
        print('All consensus: {}'.format(np.sum(labels_saban[labels_berker == labels_banu] == labels_berker[labels_berker == labels_banu])))
        print('All different: {}'.format(np.sum(np.bitwise_and(labels_saban[labels_berker != labels_banu] != labels_berker[labels_berker != labels_banu], labels_saban[labels_berker != labels_banu] != labels_banu[labels_berker != labels_banu]))))

    start_train = time.time()
    print("Device: {}/{}, Batch size: {}, Num-MetaData: {}, Seed: {}".format(DEVICE, GPU_ID, BATCH_SIZE, args.metadata_num, RANDOM_SEED))

    model_s1_path = "models1_dr_{}_{}".format(args.stage1,RANDOM_SEED)
    model_s2_path = "models2_dr_{}_{}_{}".format(args.stage1, args.stage2,RANDOM_SEED)

    # pre-train solely on diabetic retinopathy data
    normal_train(model_s1_path, args.stage1, net, train_dl_dr, val_dl_dr, val_dl_dr)
    # change the last layer for rop dataset and freeze last of the layers for learning
    # on rop dataset, we only train fc layers
    net.fc = nn.Linear(2048,NUM_CLASSES) 
    net.to(DEVICE)
    for param in net.parameters():
        param.requires_grad = False 
    for param in net.fc.parameters():
        param.requires_grad = True
    net.train()

    # train solely on rop with noisy labels
    if args.stage2 > 0:
        model_s2_path = "models2_dr_{}_{}_{}".format(args.stage1, args.stage2,RANDOM_SEED)
        normal_train(model_s2_path, args.stage2, net, train_dl, meta_dl, test_dl, epoch_offset=args.stage1)
    else:
        model_s2_path = model_s1_path
        
    # meta-train
    feature_encoder.load_state_dict(torch.load(model_s2_path, map_location=DEVICE))  
    feature_encoder.eval()
    val_acc_best, test_acc_best, epoch_best = meta_train(args.alpha, args.beta, args.gamma, args.stage3, net, feature_encoder, train_dl, meta_dl, test_dl, epoch_offset=args.stage1+args.stage2)

    if SAVE_LOGS:
        NUM_TRAINDATA = len(train_dl.dataset)
        # write log for hyperparameters
        hp_writer.add_hparams({'alpha':args.alpha, 'beta': args.beta, 'gamma':args.gamma, 'stage1':args.stage1, 'stage2':args.stage2, 'stage2':args.stage3, 'num_meta':args.metadata_num, 'num_train': NUM_TRAINDATA}, 
                                {'val_accuracy': val_acc_best, 'test_accuracy': test_acc_best, 'epoch_best':epoch_best})
        hp_writer.close()
    print('Total training duration: {:3.2f}h'.format((time.time()-start_train)/3600))