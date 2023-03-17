'''
train or finetune hybrid model:
image+image or image+radio
'''

import argparse
import os
import time
import sys
import yaml

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import distributed
from torch.utils.tensorboard import SummaryWriter

from models.mergenet import TwoResNet, HandAddResNet
from models.linear_layer import LinearClassifier, HandAddLinearClassifier
from util import AverageMeter, adjust_learning_rate, group_parameters, str2list, cal_score, score_list
from data_load import ImageDataset, MixDataset

from sklearn.metrics import f1_score


def parse_option(settings_path):
    with open(settings_path) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # file settings
    data_path = settings["file_settings"]["data_path"]
    train_csv_path = settings["file_settings"]["train_csv_path"]
    val_csv_path = settings["file_settings"]["val_csv_path"]
    save_folder = settings["file_settings"]["save_folder"]
    pretrain_model_path = settings["file_settings"]["pretrain_model_path"]

    # data settings
    data_shape = settings["data_settings"]["data_shape"] # (z, x, y)
    crop_scale = settings["data_settings"]["crop_scale"]
    radio_dim = settings["data_settings"]["radio_dim"]
    classify_num = settings["data_settings"]["classify_num"]
    feat_dim = settings["data_settings"]["feat_dim"]

    # train settings
    train_mode = settings["train_settings"]["train_mode"]
    model = settings["train_settings"]["model"]
    loss_weight = settings["train_settings"]["loss_weight"]
    
    epochs = settings["train_settings"]["epochs"]
    batch_size = settings["train_settings"]["batch_size"]
    learning_rate_base = settings["train_settings"]["learning_rate_base"]
    learning_rate_classifier = settings["train_settings"]["learning_rate_classifier"]
    lr_decay_epochs = settings["train_settings"]["lr_decay_epochs"]

    # ArgumentParser
    parser = argparse.ArgumentParser()

    # file settings
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--train_csv_path", type=str, default=train_csv_path)
    parser.add_argument("--val_csv_path", type=str, default=val_csv_path)
    parser.add_argument("--save_folder", type=str, default=save_folder)
    parser.add_argument("--pretrain_model_path", type=str, default=pretrain_model_path)
    
    # data settings
    parser.add_argument("--data_shape", type=str, default=data_shape)
    parser.add_argument("--crop_scale", type=float, default=crop_scale)
    parser.add_argument("--radio_dim", type=int, default=radio_dim)
    parser.add_argument("--classify_num", type=int, default=classify_num)

    # train settings
    parser.add_argument("--train_mode", type=str, default=train_mode,
                        choices=["image+image", "image+radio"])
    parser.add_argument("--model", type=str, default=model,
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--loss_weight", type=str, default=loss_weight)
    parser.add_argument("--print_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=20)
    
    parser.add_argument("--epochs", type=int, default=epochs)                                            
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument('--learning_rate_base', type=float, default=learning_rate_base)
    parser.add_argument('--learning_rate_classifier', type=float, default=learning_rate_classifier)
    parser.add_argument('--lr_decay_epochs', type=str, default=lr_decay_epochs)
    
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--feat_dim', type=int, default=feat_dim, help='dim of feat for inner product')

    opt = parser.parse_args()

    # change settings
    opt.train_csv_path = [string.strip() for string in opt.train_csv_path.split(",")]
    opt.val_csv_path = [string.strip() for string in opt.val_csv_path.split(",")]

    opt.data_shape = str2list(opt.data_shape)
    opt.loss_weight = str2list(opt.loss_weight)
    opt.lr_decay_epochs = str2list(opt.lr_decay_epochs)
    
    save_file_name = settings_path.split("/")[-1]
    opt.save_folder = os.path.join(opt.save_folder, save_file_name)
    count = len(opt.train_csv_path)
    for i in range(0, count):
        savedir = os.path.join(opt.save_folder, str(i))
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

    return opt


def get_train_val_loader(args, countnum):
    # transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((args.data_shape[1], args.data_shape[2]), 
                                    scale=(args.crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize([0.5] * args.data_shape[0], [0.5] * args.data_shape[0])
    ])

    val_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize([0.5] * args.data_shape[0], [0.5] * args.data_shape[0])
    ])

    # dataset definition
    if args.train_mode == "image+radio":
        train_dataset = MixDataset(args.data_path, args.train_csv_path[countnum], train_transform)
        val_dataset = MixDataset(args.data_path, args.val_csv_path[countnum], val_transform)

    elif args.train_mode == "image+image":
        train_dataset = ImageDataset(args.data_path, args.train_csv_path[countnum], train_transform,
                                     self_supervised=True)
        val_dataset = ImageDataset(args.data_path, args.val_csv_path[countnum], val_transform,
                                   self_supervised=True)

    # load data
    print("number of train: {}".format(len(train_dataset)))
    print("number of val: {}".format(len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        sampler=None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader


def set_model(args):
    if args.model == "resnet18":
        image_in_dim = 512
    elif args.model == "resnet50" or "resnet101":
        image_in_dim = 2048
    
    if args.train_mode == "image+radio":
        model = HandAddResNet(fc_in_dim=args.radio_dim, model_name=args.model, 
                              in_channel=args.data_shape[0], classify_num=args.classify_num, low_dim=args.feat_dim)
        classifier = HandAddLinearClassifier(image_in_dim=image_in_dim, radio_in_dim=256, 
                                             classify_num=args.classify_num)

    elif args.train_mode == "image+image":
        model = TwoResNet(model_name=args.model, in_channel=args.data_shape[0], 
                          classify_num=args.classify_num)
        classifier = LinearClassifier(in_dim=image_in_dim, classify_num=args.classify_num)

    model = nn.DataParallel(model)

    # load pre-trained model
    if args.pretrain_model_path != "":
        print('==> loading pre-trained model')
        ckpt = torch.load(args.pretrain_model_path)
        model.load_state_dict(ckpt['model'])
        print("==> loaded checkpoint '{}' (epoch {})".format(args.pretrain_model_path, ckpt['epoch']))
        print('==> done')

    model = model.cuda()
    classifier = classifier.cuda()

    # weight
    if args.loss_weight == []:
        weight = None
    else:
        weight=torch.FloatTensor(args.loss_weight).cuda()
    
    criterion = nn.CrossEntropyLoss(weight=weight).cuda()

    return model, classifier, criterion


def set_optimizer(args, model, classifier):
    if args.train_mode == "image+image":
        optimizer = optim.SGD(
            [
                {"params": model.parameters(), "lr": args.learning_rate_base},
                {"params": classifier.parameters(), "lr": args.learning_rate_classifier}
            ],
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

    elif args.train_mode == "image+radio":
        model_parameters = group_parameters(model)
        # (radio_net) (deep_net)
        optimizer = torch.optim.SGD([
            {"params": model_parameters['base_parameters'],
             "lr": args.learning_rate_base},
            {"params": model_parameters['new_parameters'],
             "lr": args.learning_rate_classifier},
            {"params": classifier.parameters(),
             "lr": args.learning_rate_classifier},
        ], momentum=args.momentum, weight_decay=args.weight_decay)

    return optimizer


def train(epoch, train_loader, model, classifier, criterion, optimizer, args):
    """
    one epoch training
    """
    model.train()
    classifier.train()

    losses = AverageMeter()
    pred_labels = []
    truth_labels = []

    if args.train_mode == "image+image":
        for idx, (data, label, _) in enumerate(train_loader):
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # forward
            feat_l, feat_ab = model(data, truth_features=True)
            output = classifier(feat_l, feat_ab)
            loss = criterion(output, label)

            with torch.no_grad():
                _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
                pred = pred.t().squeeze()
            pred = pred.cpu().numpy().tolist()
            truth = label.cpu().numpy().tolist()
            pred_labels.extend(pred)
            truth_labels.extend(truth)

            losses.update(loss.item(), data.size(0))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print info
            if idx % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       epoch, idx, len(train_loader), loss=losses))
                sys.stdout.flush()

    elif args.train_mode == "image+radio":
        for idx, (image_data, radio_data, label, _) in enumerate(train_loader):
            image_data = image_data.cuda(non_blocking=True)
            radio_data = radio_data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # forward
            feat_image, feat_radio = model(image_data, radio_data, truth_features=True)
            output = classifier(feat_image, feat_radio)
            loss = criterion(output, label)

            with torch.no_grad():
                _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
                pred = pred.t().squeeze()
            pred = pred.cpu().numpy().tolist()
            truth = label.cpu().numpy().tolist()
            pred_labels.extend(pred)
            truth_labels.extend(truth)

            losses.update(loss.item(), image_data.size(0))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print info
            if idx % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, idx, len(train_loader), loss=losses))
                sys.stdout.flush()

    score = cal_score(truth_labels, pred_labels)
    return score, losses.avg


def validate(val_loader, model, classifier, criterion, args, test_flag=False):
    """
    evaluation
    """
    losses = AverageMeter()
    pred_labels = []
    truth_labels = []

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    if args.train_mode == "image+image":
        with torch.no_grad():
            for idx, (data, label, _) in enumerate(val_loader):
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                # compute output
                feat_l, feat_ab = model(data, truth_features=True)
                output = classifier(feat_l.detach(), feat_ab.detach())
                loss = criterion(output, label)

                # measure accuracy and record loss
                _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
                pred = pred.t().squeeze()
                pred = pred.cpu().numpy().tolist()
                truth = label.cpu().numpy().tolist()
                pred_labels.extend(pred)
                truth_labels.extend(truth)

                losses.update(loss.item(), data.size(0))

                if idx % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                           idx, len(val_loader), loss=losses))

    elif args.train_mode == "image+radio":
        with torch.no_grad():
            for idx, (image_data, radio_data, label, _) in enumerate(val_loader):
                image_data = image_data.cuda(non_blocking=True)
                radio_data = radio_data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                # compute output
                feat_image, feat_radio = model(image_data, radio_data, truth_features=True)
                output = classifier(feat_image.detach(), feat_radio.detach())
                loss = criterion(output, label)

                # measure accuracy and record loss
                _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
                pred = pred.t().squeeze()
                pred = pred.cpu().numpy().tolist()
                truth = label.cpu().numpy().tolist()
                pred_labels.extend(pred)
                truth_labels.extend(truth)

                losses.update(loss.item(), image_data.size(0))

                if idx % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        idx, len(val_loader), loss=losses))

    if test_flag:
        return truth_labels, pred_labels
    else:
        score = cal_score(truth_labels, pred_labels)
        return score, losses.avg


def main():
    settings_path = input("settings file:")
    settings_path = os.path.join("../Experiments/Settings/", settings_path)

    args = parse_option(settings_path)

    count = len(args.train_csv_path)

    for i in range(0, count):
        best_score = 0
        save_folder = os.path.join(args.save_folder, str(i))
        # set the data loader
        train_loader, val_loader = get_train_val_loader(args, i)
        # set the model
        model, classifier, criterion = set_model(args)
        # set optimizer
        optimizer = set_optimizer(args, model, classifier)
        cudnn.benchmark = True
        # tensorboard
        writer = SummaryWriter(log_dir=save_folder)

        # train routine
        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(epoch, args, optimizer)
            print("==> training...")

            train_score, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, args)
            writer.add_scalar('train_score', train_score, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)

            print("==> testing...")
            test_score, test_loss = validate(val_loader, model, classifier, criterion, args)
            writer.add_scalar('test_score', test_score, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)

            # save the best model
            if test_score > best_score:
                best_score = test_score
                state = {
                    "opt": args,
                    "epoch": epoch,
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score
                }
                save_name = os.path.join(save_folder, "best_model.pth")
                torch.save(state, save_name)

            # save model
            if epoch % args.save_freq == 0:
                print("==> saving...")
                state = {
                    "opt": args,
                    "epoch": epoch,
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score
                }
                save_name = "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
                save_name = os.path.join(save_folder, save_name)
                print("saving regular model!")
                torch.save(state, save_name)
        
        # test routine
        state = torch.load(os.path.join(save_folder, "best_model.pth"))
        model.load_state_dict(state["model"])
        classifier.load_state_dict(state["classifier"])
        model = model.cuda()
        classifier = classifier.cuda()
        truth_labels, pred_labels = validate(val_loader, model, classifier, criterion, args, test_flag=True)
        score_list(truth_labels, pred_labels, save_folder)


if __name__ == '__main__':
    main()