'''
train self supervised model: image+image or image+radio
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
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from data_load import ImageDataset, MixDataset
from models.mergenet import TwoResNet
from models.mergenet_nodrop import HandAddResNet
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss
from util import AverageMeter, adjust_learning_rate, str2list


def parse_option(settings_path):
    with open(settings_path) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # file settings
    data_path = settings["file_settings"]["data_path"]
    train_csv_path = settings["file_settings"]["train_csv_path"]
    save_folder = settings["file_settings"]["save_folder"]

    # data settings
    data_shape = settings["data_settings"]["data_shape"] # (z, x, y)
    crop_scale = settings["data_settings"]["crop_scale"]
    radio_dim = settings["data_settings"]["radio_dim"]
    classify_num = settings["data_settings"]["classify_num"]
    feat_dim = settings["data_settings"]["feat_dim"]

    # train settings
    train_mode = settings["train_settings"]["train_mode"]
    model = settings["train_settings"]["model"]
    
    epochs = settings["train_settings"]["epochs"]
    batch_size = settings["train_settings"]["batch_size"]
    learning_rate = settings["train_settings"]["learning_rate"]
    lr_decay_epochs = settings["train_settings"]["lr_decay_epochs"]

    # ArgumentParser
    parser = argparse.ArgumentParser()

    # file settings
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--train_csv_path", type=str, default=train_csv_path)
    parser.add_argument("--save_folder", type=str, default=save_folder)
    
    # data settings
    parser.add_argument("--data_shape", type=str, default=data_shape)
    parser.add_argument("--crop_scale", type=float, default=crop_scale)
    parser.add_argument("--radio_dim", type=int, default=radio_dim)
    parser.add_argument("--classify_num", type=int, default=classify_num)

    # train settings
    parser.add_argument("--train_mode", type=str, default=train_mode,
                        choices=["image+image", "image+radio"])
    parser.add_argument('--model', type=str, default=model,
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--print_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=50)
    
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--lr_decay_epochs', type=str, default=lr_decay_epochs)
    
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # nce settings
    parser.add_argument('--softmax', action='store_true',
                        help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=2048)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=feat_dim, help='dim of feat for inner product')

    opt = parser.parse_args()

    # change settings
    opt.data_shape = str2list(opt.data_shape)
    opt.lr_decay_epochs = str2list(opt.lr_decay_epochs)

    save_file_name = settings_path.split("/")[-1]
    opt.save_folder = os.path.join(opt.save_folder, save_file_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_train_loader(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((args.data_shape[1], args.data_shape[2]), 
                                    scale=(args.crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize([0.5] * args.data_shape[0], [0.5] * args.data_shape[0])
    ])

    # dataset definition
    if args.train_mode == "image+image":
        train_dataset = ImageDataset(args.data_path, args.train_csv_path, train_transform,
                                     self_supervised=True)
    elif args.train_mode == "image+radio":
        train_dataset = MixDataset(args.data_path, args.train_csv_path, train_transform)

    # load data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        sampler=None
    )
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    if args.train_mode == "image+image":
        model = TwoResNet(model_name=args.model, in_channel=args.data_shape[0], classify_num=args.classify_num)

    elif args.train_mode == "image+radio":
        model = HandAddResNet(fc_in_dim=args.radio_dim, model_name=args.model, 
                              in_channel=args.data_shape[0], classify_num=args.classify_num, low_dim=args.feat_dim)

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    model = nn.DataParallel(model)
    model = model.cuda()
    contrast = contrast.cuda()
    criterion_ab = criterion_ab.cuda()
    criterion_l = criterion_l.cuda()
    cudnn.benchmark = True
    return model, contrast, criterion_ab, criterion_l


def set_optimizer(args, model):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD([
    #     {"params": model.module.radio_net.parameters(),
    #         "lr": args.learning_rate * args.learning_rate_ratio},
    #     {"params": model.module.deep_net.parameters()},
    # ], lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, args):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()

    if args.train_mode == "image+image":
        for idx, (data, _, index) in enumerate(train_loader):
            bsz = data.size(0)
            if torch.cuda.is_available():
                data = data.cuda()
                index = index.cuda()

            # forward
            feat_l, feat_ab = model(data, self_supervised_features=True)
            out_l, out_ab = contrast(feat_l, feat_ab, index)

            l_loss = criterion_l(out_l)
            ab_loss = criterion_ab(out_ab)
            l_prob = out_l[:, 0].mean()
            ab_prob = out_ab[:, 0].mean()

            loss = l_loss + ab_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save info
            losses.update(loss.item(), bsz)
            l_loss_meter.update(l_loss.item(), bsz)
            l_prob_meter.update(l_prob.item(), bsz)
            ab_loss_meter.update(ab_loss.item(), bsz)
            ab_prob_meter.update(ab_prob.item(), bsz)

            torch.cuda.synchronize()

            # print info
            if (idx + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                      'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), loss=losses, lprobs=l_prob_meter,
                       abprobs=ab_prob_meter))

                sys.stdout.flush()

    elif args.train_mode == "image+radio":
        for idx, (image_data, radio_data, _, index) in enumerate(train_loader):
            bsz = image_data.size(0)
            if torch.cuda.is_available():
                image_data = image_data.cuda()
                radio_data = radio_data.cuda()
                index = index.cuda()

            # forward
            image_feature, radio_feature = model(image_data, radio_data,
                                                 self_supervised_features=True)
            out_image, out_radio = contrast(image_feature, radio_feature, index)

            image_loss = criterion_l(out_image)
            radio_loss = criterion_ab(out_radio)
            image_prob = out_image[:, 0].mean()
            radio_prob = out_radio[:, 0].mean()

            loss = image_loss + radio_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save info
            losses.update(loss.item(), bsz)
            l_loss_meter.update(image_loss.item(), bsz)
            l_prob_meter.update(image_prob.item(), bsz)
            ab_loss_meter.update(radio_loss.item(), bsz)
            ab_prob_meter.update(radio_prob.item(), bsz)

            torch.cuda.synchronize()

            # print info
            if (idx + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'image_p {imageprobs.val:.3f} ({imageprobs.avg:.3f})\t'
                      'radio_p {radioprobs.val:.3f} ({radioprobs.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), loss=losses, imageprobs=l_prob_meter,
                    radioprobs=ab_prob_meter))

                sys.stdout.flush()

    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg


def main():
    settings_path = input("settings file:")
    settings_path = os.path.join("../Experiments/Settings", settings_path)

    args = parse_option(settings_path)
    # set the loader
    train_loader, n_data = get_train_loader(args)
    # set the model
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)
    # set the optimizer
    optimizer = set_optimizer(args, model)
    # tensorboard
    writer = SummaryWriter(log_dir=args.save_folder)

    # routine
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)
        
        print("==> training...")
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l,
                                                 criterion_ab, optimizer, args)

        # tensorboard logger
        writer.add_scalar('l_loss(image)', l_loss, epoch)
        writer.add_scalar('l_prob(image)', l_prob, epoch)
        writer.add_scalar('ab_loss(radio)', ab_loss, epoch)
        writer.add_scalar('ab_prob(radio)', ab_prob, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.save_folder,
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()