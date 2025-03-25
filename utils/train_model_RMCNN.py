from utils.utils import *
import torch.nn as nn
from model.optimizer import MixOptimizer
from pytorch_metric_learning import losses
import time

def interaug(args, data_loader):  
    timg = data_loader.dataset.x
    label = data_loader.dataset.y
    batch_size = args.batch_size
    aug_data = []
    aug_label = []
    for cls4aug in range(args.num_class):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]

        tmp_aug_data = np.zeros((int(batch_size / args.num_class), 1, args.in_chan, args.signal_length))
        for ri in range(int(batch_size / args.num_class)):
            for rj in range(8):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]
        aug_data.append(tmp_aug_data)
        aug_label.append(np.ones(int(batch_size / args.num_class)) * cls4aug)

    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]
    if args.CUDA:
        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
    else:
        aug_data = torch.from_numpy(aug_data).float().cpu()
        aug_label = torch.from_numpy(aug_label).long().cpu()
    return aug_data, aug_label

def train_one_epoch(args, data_loader, net, loss_fn_clf, optimizer, loss_fn_inter=None):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        
        if args.CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        else:
            x_batch, y_batch = x_batch.cpu(), y_batch.cpu()

        if args.is_aug:
            aug_data, aug_label = interaug(args, data_loader)
            x_batch = torch.cat((x_batch, aug_data))
            y_batch = torch.cat((y_batch, aug_label))  

        fe, out = net(x_batch)
        loss_clf = loss_fn_clf(out, y_batch)
        loss = args.coefficient_clf * loss_clf
        if args.enable_inter:
            loss_inter = loss_fn_inter(fe, y_batch)
            loss += args.coefficient_inter * loss_inter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      

        _, pred = torch.max(out, 1)
        tl.add(loss)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
    return tl.item(), pred_train, act_train


def predict(args, data_loader, net, loss_fn_clf, loss_fn_inter=None):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if args.CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            else:
                x_batch, y_batch = x_batch.cpu(), y_batch.cpu()
            fe, out = net(x_batch)
            loss_clf = loss_fn_clf(out, y_batch)
            loss = args.coefficient_clf * loss_clf
            if args.enable_inter:
                loss_inter = loss_fn_inter(fe, y_batch)
                loss += args.coefficient_inter * loss_inter
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def train(args, data_train, label_train, data_val, label_val, subject):
    seed_all(args.random_seed)
    save_name = 'sub' + str(subject)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    model = get_model(args)
    if args.CUDA:
        model = model.cuda()
    else:
        model = model.cpu()

    CE = nn.CrossEntropyLoss()
    tripletloss = losses.TripletMarginLoss(margin=args.tripletloss_margin)
    optimizer = torch.optim.Adam([
        {'params':filter(lambda p: p.requires_grad, model.parameters()), 'lr':args.learning_rate},                 
        ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
	 patience=args.patience_scheduler,verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, 
		 min_lr=0, eps=args.scheduler_eps)
    optimizer = MixOptimizer(optimizer)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['min_loss'] = 1e10
    trlog['target_loss'] = 0.0

    # timer = Timer()
    patience = args.patience
    counter = 0
    time_comsume = 0
    for epoch in range(1, args.max_epoch + 1):
        print('sub{} stage1 epoch{}'.format(subject, epoch))

        start = time.time()
        loss_train, pred_train, act_train = train_one_epoch(
            args, data_loader=train_loader, net=model, loss_fn_clf=CE, optimizer=optimizer, loss_fn_inter=tripletloss
            )
        end = time.time()
        print (str(end-start))
        time_comsume += end-start
        acc_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('tra, loss={:.4f} acc={:.4f}'
              .format(loss_train, acc_train))

        loss_val, pred_val, act_val = predict(
            args, data_loader=val_loader, net=model, loss_fn_clf=CE, loss_fn_inter=tripletloss
            )
        scheduler.step(loss_val)
        acc_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('val, loss={:.4f} acc={:.4f}\n'.
              format(loss_val, acc_val))

        if loss_val <= trlog['min_loss']:
            trlog['max_acc'] = acc_val
            trlog['min_loss'] = loss_val
            trlog['target_loss'] = loss_train
            # if fold:
            #     save_model('candidate')
            counter = 0
            # save model here for reproduce
            model_name_reproduce = save_name + '.pth'
            save_path = os.path.join(args.save_path_model_param, 'model_first_stage')
            ensure_path(save_path)
            model_name_reproduce = os.path.join(save_path, model_name_reproduce)
            torch.save(model.state_dict(), model_name_reproduce)
            
        else:
            counter += 1
            if epoch > args.min_epoch:
                if counter >= patience:
                    print('early stopping')
                    break

        trlog['train_loss'].append(loss_train.item())
        trlog['train_acc'].append(acc_train.item())
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val.item())

        # print('ETA:{}/{} SUB:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch), subject))
    # save the training log file
    save_path = os.path.join(args.save_path_log_param, 'log_first_stage')
    ensure_path(save_path)
    torch.save(trlog, os.path.join(save_path, save_name))

    print('average time:'+ str(time_comsume / epoch))

    return trlog['max_acc'], trlog['target_loss']


def test(args, data, label,  subject):
    # set_up(args)
    seed_all(args.random_seed)
    test_loader = get_dataloader(data, label, args.batch_size)

    model = get_model(args)
    if args.CUDA:
        model = model.cuda()
    else:
        model = model.cpu()

    CE = nn.CrossEntropyLoss()
    tripletloss = losses.TripletMarginLoss(margin=args.tripletloss_margin)
    model_name_reproduce = 'sub' + str(subject) + '.pth'
    load_path_final = os.path.join(args.save_path_model_param, 'model_second_stage', model_name_reproduce)
    model.load_state_dict(torch.load(load_path_final))
    loss, pred, act = predict(
        args, data_loader=test_loader, net=model, loss_fn_clf=CE, loss_fn_inter=tripletloss
    )
    acc, cm = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f}'.format(loss, acc))
    return pred, act

def combine_train(args, data_train, label_train, data_val, label_val, subject, target_loss, fold=None):
    seed_all(args.random_seed)
    save_name = 'sub' + str(subject)
    if fold:
        save_name = 'sub' + str(subject) + '_fold' + str(fold)
    # set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    model = get_model(args)
    if args.CUDA:
        model = model.cuda()
    else:
        model = model.cpu()
    model.load_state_dict(torch.load(os.path.join(args.save_path_model_param, 'model_first_stage', save_name + '.pth')))

    CE = nn.CrossEntropyLoss()
    tripletloss = losses.TripletMarginLoss(margin=args.tripletloss_margin)
    optimizer = torch.optim.Adam([
        {'params':filter(lambda p: p.requires_grad, model.parameters()), 'lr':args.learning_rate},                 
        ])


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
	 patience=args.patience_scheduler,verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, 
		 min_lr=0, eps=args.scheduler_eps)
    optimizer = MixOptimizer(optimizer)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['min_loss'] = 1e10

    # timer = Timer()
    patience = args.patience
    counter = 0

    for epoch in range(1, args.max_epoch_cmb + 1):
        if fold:
            print('sub{} stage2 fold{} epoch{}'.format(subject, fold, epoch))
        else:
            print('sub{} stage2 epoch{}'.format(subject, epoch))
        loss_train, pred_train, act_train = train_one_epoch(
            args, data_loader=train_loader, net=model, loss_fn_clf=CE, optimizer=optimizer, loss_fn_inter=tripletloss
            )

        acc_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('tra, loss={:.4f} acc={:.4f}'
              .format(loss_train, acc_train))

        loss_val, pred_val, act_val = predict(
            args, data_loader=val_loader, net=model, loss_fn_clf=CE, loss_fn_inter=tripletloss
            )
        scheduler.step(loss_val)
        acc_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('val, loss={:.4f} acc={:.4f}\n'.
              format(loss_val, acc_val))

        if loss_val <= trlog['min_loss']:
            trlog['max_acc'] = acc_val
            trlog['min_loss'] = loss_val
            # save_model('final_model')
            counter = 0
            # save model here for reproduce
            model_name_reproduce = save_name + '.pth'
            save_path = os.path.join(args.save_path_model_param, 'model_second_stage')
            ensure_path(save_path)
            model_name_reproduce = os.path.join(save_path, model_name_reproduce)
            torch.save(model.state_dict(), model_name_reproduce)
            
        else:
            counter += 1

        if epoch > args.min_epoch_cmb:
            # if counter >= patience:
            #     print('early stopping')
            #     break
            if loss_val <= target_loss:
                print('loss reach!')
                break

        trlog['train_loss'].append(loss_train.item())
        trlog['train_acc'].append(acc_train.item())
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val.item())

        # print('ETA:{}/{} SUB:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch), subject))
    # save the training log file
    save_path = os.path.join(args.save_path_log_param, 'log_second_stage')
    ensure_path(save_path)
    torch.save(trlog, os.path.join(save_path, save_name))
