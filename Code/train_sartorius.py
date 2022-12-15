import os
import os.path as osp
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deeplab import DeepLab
from models import CPNet, UNet, DResUNet, CPNetSAM, UNetSAM, DResUNetSAM
from data import get_train_data, get_val_data, get_test_data
from dynamics import diameters, compute_masks
from metrics import  average_precision, acc_prec_rec_iou
from transforms import reshape_and_normalize_data, random_rotate_and_resize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Flow Tracking and UNet-like network based Instance Segmentation")
    parser.add_argument('--model', type=str, default='cpnet', choices=['cpnet', 'unet', 'dresunet', 'deeplab'], help='model name (default: resnet)')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='backbone network for dresunet (default: resnet50)')
    parser.add_argument('--use-tmap', action='store_true', default=False, help='whether to use tmap (default: auto)')
    parser.add_argument('--tmap', type=str, default='weight', choices=['weight', 'output'], help='tmap used in loss weight or output (default: weight)')
    parser.add_argument('--use-sam', action='store_true', default=False, help='whether to use spatial attention module (default: auto)')
    parser.add_argument('--sam-position', type=str, default='mid', choices=['mid', 'tail'], help='spatial attention module used in mid or tail (default: mid)')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--nepochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--diam-mean', type=float, default=17.0, help='mean of the diameter')
    args = parser.parse_args()

    use_gpu = True
    device=0
    if use_gpu and torch.cuda.is_available():
        # device = torch.device(f'cuda:{device}')
        device = torch.device('cuda')
        use_gpu=True
    else:
        device = torch.device('cpu')
        use_gpu=False

    x_shape = (520, 704)
    args.nchannels = 1
    args.nclasses = 4 if args.use_tmap and args.tmap == "output" else 3
    print('============================== Model =================================')
    if args.use_sam:
        if args.model == "cpnet":
            net = CPNetSAM(args.nchannels, args.nclasses, sam_position=args.sam_position)
        elif args.model == "unet":
            net = UNetSAM(args.nchannels, args.nclasses, sam_position=args.sam_position)
        else:
            net = DResUNetSAM(args.nchannels, args.nclasses, sam_position=args.sam_position, backbone=args.backbone)
    else:
        if args.model == "cpnet":
            net = CPNet(args.nchannels, args.nclasses)
        elif args.model == "unet":
            net = UNet(args.nchannels, args.nclasses)
        elif args.model == "deeplab":
            net = DeepLab(in_channels=args.nchannels, num_classes=args.nclasses)
        else:
            net = DResUNet(args.nchannels, args.nclasses, backbone=args.backbone)
    if use_gpu:
        net = net.to(device)

    print('============================== Optimizer and Loss Function =================================')
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    def tmap_w_loss_fn(gts, preds):
        criterion1 = nn.MSELoss(reduction='mean')
        criterion2 = nn.BCEWithLogitsLoss(reduction='mean', weight = (1 + gts[:, 3]))
        veci = 5.*gts[:, 1:3]
        lbl = (gts[:, 0] > .5).float()
        loss1 = criterion1(preds[:, :2], veci)
        loss2 = criterion2(preds[:, 2], lbl)
        return .5*loss1+loss2

    def tmap_o_loss_fn(gts, preds):
        criterion1 = nn.MSELoss(reduction='mean')
        criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
        criterion3 = nn.MSELoss(reduction='mean')
        veci = 5.*gts[:, 1:3]
        lbl = (gts[:, 0] > .5).float()
        cont = gts[: ,3]
        loss1 = criterion1(preds[:, :2], veci)
        loss2 = criterion2(preds[:, 2]*(1+preds[:, 3]), lbl)
        loss3 = criterion3(preds[:, 3], cont)
        return .5*loss1+loss2+.5*loss3

    def fv_loss_fn(gts, preds):
        criterion1 = nn.MSELoss(reduction='mean')
        criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
        veci = 5.*gts[:, 1:3]
        lbl = (gts[:, 0] > .5).float()
        loss1 = criterion1(preds[:, :2], veci)
        loss2 = criterion2(preds[:, 2], lbl)
        return .5*loss1+loss2

    if not args.use_tmap:
        loss_fn = fv_loss_fn
    else:
        if args.tmap == "weight":
            loss_fn = tmap_w_loss_fn
        else:
            loss_fn = tmap_o_loss_fn

    print('============================== Data =================================')
    train_data, train_flows, train_image_names, train_cell_types = get_train_data(use_tmap=args.use_tmap)
    val_data, val_flows, val_image_names, val_cell_types = get_val_data(use_tmap=args.use_tmap)
    test_data, test_flows, test_image_names, test_cell_types = get_test_data(use_tmap=args.use_tmap)

    print('============================== Data Preprocessing =================================')
    train_data, _, _ = reshape_and_normalize_data(train_data, test_data=None, channels=None, normalize=True)
    val_data, _, _ = reshape_and_normalize_data(val_data, test_data=None, channels=None, normalize=True)
    test_data, _, _ = reshape_and_normalize_data(test_data, test_data=None, channels=None, normalize=True)

    print('============================== Log file =================================')
    exp_dir = osp.join("../Experiments", "{}_sam_{}_loss_{}".format((args.model if args.model == "cpnet" or args.model == "unet" else args.model+"_"+args.backbone),
                                                                    ("none" if not args.use_sam else args.sam_position),
                                                                    ("fv" if not args.use_tmap else args.tmap)))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if osp.exists(osp.join(exp_dir, 'config.yaml')):
        os.remove(osp.join(exp_dir, 'config.yaml'))
    with open(osp.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)
    if osp.exists(osp.join(exp_dir, 'val.csv')):
        os.remove(osp.join(exp_dir, 'val.csv'))
    with open(osp.join(exp_dir, 'val.csv'), 'w') as f:
        eval_head = ["epoch", "image_id", "cell_type", "ap_50", "ap_55", "ap_60", "ap_65", "ap_70", "ap_75", "ap_80", "ap_85", "ap_90", "ap_95", "ap", "acc", "prec", "rec", "iou", "cellprob_threshold"]
        f.write(','.join(eval_head) + '\n')
    if osp.exists(osp.join(exp_dir, 'test.csv')):
        os.remove(osp.join(exp_dir, 'test.csv'))
    with open(osp.join(exp_dir, 'test.csv'), 'w') as f:
        eval_head = ["image_id", "cell_type", "ap_50", "ap_55", "ap_60", "ap_65", "ap_70", "ap_75", "ap_80", "ap_85", "ap_90", "ap_95", "ap", "acc", "prec", "rec", "iou", "cellprob_threshold"]
        f.write(','.join(eval_head) + '\n')
    if osp.exists(osp.join(exp_dir, 'losses.csv')):
        os.remove(osp.join(exp_dir, 'losses.csv'))
    with open(osp.join(exp_dir, 'losses.csv'), 'w') as f:
        eval_head = ["epoch", "train_loss"]
        f.write(','.join(eval_head) + '\n')

    diam_mean = args.diam_mean
    train_rescale = True
    train_diams = np.array([diameters(train_flows[k][0])[0] for k in range(len(train_flows))])
    train_diam_mean = train_diams[train_diams > 0].mean()
    if train_rescale:
        train_diams[train_diams < 5] = 5.
        train_scale_range = 0.5
    else:
        train_scale_range = 1.0
    val_rescale = train_diams / diam_mean
    test_rescale = train_diams / diam_mean

    nimg_train = len(train_data)
    nimg_val = len(val_data)
    nimg_test = len(test_data)
    nimg_per_epoch = None
    inds_all = np.zeros((0,), 'int32')
    if nimg_per_epoch is None or nimg_train > nimg_per_epoch:
        nimg_per_epoch = nimg_train
    while len(inds_all) < args.nepochs * nimg_per_epoch:
        rperm = np.random.permutation(nimg_train)
        inds_all = np.hstack((inds_all, rperm))

    best_mean_ap_t0 = 0.0
    best_mean_ap_t4 = 0.0
    torch.save({
        'epoch': -1,
        'arch': net.__class__.__name__,
        'model_state_dict': net.state_dict(),
        'best_mean_ap': 0.0,
    }, osp.join(exp_dir, 'best_model_t0.pth.tar'))
    torch.save({
        'epoch': -1,
        'arch': net.__class__.__name__,
        'model_state_dict': net.state_dict(),
        'best_mean_ap': 0.0,
    }, osp.join(exp_dir, 'best_model_t4.pth.tar'))
    for iepoch in range(args.nepochs):
        print(f'============================== Epoch {iepoch} =================================')
        net.train()
        rperm = inds_all[iepoch * nimg_per_epoch:(iepoch + 1) * nimg_per_epoch]
        lavg, nsum = 0., 0
        for ibatch in range(0, nimg_per_epoch, args.batch_size):
            inds = rperm[ibatch:ibatch + args.batch_size]
            rsc = train_diams[inds] / diam_mean if train_rescale else np.ones(len(inds), np.float32)
            imgs, lbls, scale = random_rotate_and_resize(
                [train_data[i] for i in inds],
                Y=[train_flows[i][1:] for i in inds],
                rescale=rsc, scale_range=train_scale_range)
            imgs, lbls = torch.from_numpy(imgs).float(), torch.from_numpy(lbls).float()
            if use_gpu:
                imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            preds = net(imgs)[0]
            loss = loss_fn(lbls, preds)
            loss.backward()
            train_loss = loss.item()
            optimizer.step()
            train_loss *= len(imgs)
            lavg += train_loss
            nsum += len(imgs)
        lavg = lavg / nsum
        with open(osp.join(exp_dir, 'losses.csv'), 'a') as f:
            log = [iepoch, lavg]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        print(f"=====Train Loss: {lavg}")

        net.eval()
        masks_gt_val = [val_flows[i][0].astype(int) for i in range(len(val_flows))]
        masks_pred_val_t0 = []
        masks_pred_val_t4 = []
        for ival in range(0, nimg_val):
            imgs = torch.from_numpy(np.array([val_data[ival]])).float()
            if use_gpu:
                imgs = imgs.to(device)
            with torch.no_grad():
                preds = net(imgs)[0]
            cellprob = preds[:, 2].cpu().numpy()[0]
            if args.use_tmap and args.tmap == "output":
                contourprob = preds[:, 3].cpu().numpy()[0]
                cellprob = cellprob*(1 + contourprob)
            dP = preds[:, :2].cpu().numpy()[0]
            masks_pred_val_t0.append(compute_masks(dP, cellprob)[0].astype(int))
            masks_pred_val_t4.append(compute_masks(dP, cellprob, cellprob_threshold=0.4)[0].astype(int))
        ap_val_t0 = average_precision(masks_gt_val, masks_pred_val_t0, threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        ap_val_t4 = average_precision(masks_gt_val, masks_pred_val_t4, threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        meanap_val_t0 = np.mean(ap_val_t0[0], axis=1)
        meanap_val_t4 = np.mean(ap_val_t4[0], axis=1)
        acc_prec_rec_ious_val_t0 = acc_prec_rec_iou(masks_gt_val, masks_pred_val_t0)
        acc_prec_rec_ious_val_t4 = acc_prec_rec_iou(masks_gt_val, masks_pred_val_t4)
        for image_idx in range(len(val_image_names)):
            image_name = val_image_names[image_idx]
            cell_type = val_cell_types[image_idx]
            with open(osp.join(exp_dir, 'val.csv'), 'a') as f:
                log = [iepoch, image_name, cell_type] + list(ap_val_t0[0][image_idx]) + [meanap_val_t0[image_idx]] + list(acc_prec_rec_ious_val_t0[image_idx]) + [0.0]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            with open(osp.join(exp_dir, 'val.csv'), 'a') as f:
                log = [iepoch, image_name, cell_type] + list(ap_val_t4[0][image_idx]) + [meanap_val_t4[image_idx]] + list(acc_prec_rec_ious_val_t4[image_idx])+ [0.4]
                log = map(str, log)
                f.write(','.join(log) + '\n')
        print(f"===== Val MeanAP threshold 0.0: {np.mean(meanap_val_t0):.4}, Mean IOU: {np.mean(acc_prec_rec_ious_val_t0[:, 3]):.4}")
        print(f"===== Val MeanAP threshold 0.4: {np.mean(meanap_val_t4):.4}, Mean IOU: {np.mean(acc_prec_rec_ious_val_t4[:, 3]):.4}")
        is_best_t0 = np.mean(meanap_val_t0) > best_mean_ap_t0
        if is_best_t0:
            best_mean_ap_t0 = np.mean(meanap_val_t0)
            torch.save({
                'epoch': iepoch,
                'arch': net.__class__.__name__,
                'model_state_dict': net.state_dict(),
                'best_mean_ap': best_mean_ap_t0,
            }, osp.join(exp_dir, 'best_model_t0.pth.tar'))
            print("Best val model t0 saved")
        is_best_t4 = np.mean(meanap_val_t4) > best_mean_ap_t4
        if is_best_t4:
            best_mean_ap_t4 = np.mean(meanap_val_t4)
            torch.save({
                'epoch': iepoch,
                'arch': net.__class__.__name__,
                'model_state_dict': net.state_dict(),
                'best_mean_ap': best_mean_ap_t4,
            }, osp.join(exp_dir, 'best_model_t4.pth.tar'))
            print("Best val model t0 saved")

    best_model_resume = torch.load(osp.join(exp_dir, 'best_model_t0.pth.tar'))
    net.load_state_dict(best_model_resume['model_state_dict'])
    print(f"Best t0 model in epoch {best_model_resume['epoch']}, MeanAP {best_model_resume['best_mean_ap']}")
    masks_gt_test = [test_flows[i][0].astype(int) for i in range(len(test_flows))]
    masks_pred_test = []
    for itest in range(0, nimg_test):
        imgs = torch.from_numpy(np.array([test_data[itest]])).float()
        if use_gpu:
            imgs = imgs.to(device)
        with torch.no_grad():
            preds = net(imgs)[0]
        cellprob = preds[:, 2].cpu().numpy()[0]
        if args.use_tmap and args.tmap == "output":
            contourprob = preds[:, 3].cpu().numpy()[0]
            cellprob = cellprob * (1 + contourprob)
        dP = preds[:, :2].cpu().numpy()[0]
        masks_pred_test.append(compute_masks(dP, cellprob)[0].astype(int))
    ap_test = average_precision(masks_gt_test, masks_pred_test,
                                threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    meanap_test = np.mean(ap_test[0], axis=1)
    acc_prec_rec_ious_test = acc_prec_rec_iou(masks_gt_test, masks_pred_test)
    for image_idx in range(len(test_image_names)):
        image_name = test_image_names[image_idx]
        cell_type = test_cell_types[image_idx]
        with open(osp.join(exp_dir, 'test.csv'), 'a') as f:
            log = [image_name, cell_type] + list(ap_test[0][image_idx]) + [meanap_test[image_idx]] + list(
                acc_prec_rec_ious_test[image_idx]) + [0.0]
            log = map(str, log)
            f.write(','.join(log) + '\n')
    print(f"Test MeanAP: {np.mean(meanap_test):.4}, Mean IOU: {np.mean(acc_prec_rec_ious_test[:, 3]):.4}")

    best_model_resume = torch.load(osp.join(exp_dir, 'best_model_t4.pth.tar'))
    net.load_state_dict(best_model_resume['model_state_dict'])
    print(f"Best t4 model in epoch {best_model_resume['epoch']}, MeanAP {best_model_resume['best_mean_ap']}")
    masks_gt_test = [test_flows[i][0].astype(int) for i in range(len(test_flows))]
    masks_pred_test = []
    for itest in range(0, nimg_test):
        imgs = torch.from_numpy(np.array([test_data[itest]])).float()
        if use_gpu:
            imgs = imgs.to(device)
        with torch.no_grad():
            preds = net(imgs)[0]
        cellprob = preds[:, 2].cpu().numpy()[0]
        if args.use_tmap and args.tmap == "output":
            contourprob = preds[:, 3].cpu().numpy()[0]
            cellprob = cellprob * (1 + contourprob)
        dP = preds[:, :2].cpu().numpy()[0]
        masks_pred_test.append(compute_masks(dP, cellprob, cellprob_threshold=0.4)[0].astype(int))
    ap_test = average_precision(masks_gt_test, masks_pred_test,
                                threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    meanap_test = np.mean(ap_test[0], axis=1)
    acc_prec_rec_ious_test = acc_prec_rec_iou(masks_gt_test, masks_pred_test)
    for image_idx in range(len(test_image_names)):
        image_name = test_image_names[image_idx]
        cell_type = test_cell_types[image_idx]
        with open(osp.join(exp_dir, 'test.csv'), 'a') as f:
            log = [image_name, cell_type] + list(ap_test[0][image_idx]) + [meanap_test[image_idx]] + list(
                acc_prec_rec_ious_test[image_idx]) + [0.4]
            log = map(str, log)
            f.write(','.join(log) + '\n')
    print(f"Test MeanAP: {np.mean(meanap_test):.4}, Mean IOU: {np.mean(acc_prec_rec_ious_test[:, 3]):.4}")
