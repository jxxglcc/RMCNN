'''
#####################################################################################################################
Date       : 1st, Sep., 2022
---------------------------------------------------------------------------------------------------------------------
Discription: Trainning file of Graph-CSPNet for holdout scenario on BCIC-IV-2a. 
#######################################################################################################################
'''

import time
import pandas as pd
import numpy as np
import random


#import torch and sklearn
from torch.autograd import Variable
import torch.nn.functional as F
import torch as th
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from sklearn.model_selection import StratifiedShuffleSplit


#import util folder
from utils.model import Graph_CSPNet_Basic
#from utils.functional import MixOptimizer
from utils.early_stopping import EarlyStopping
from utils.load_data import load_HG, dataloader_in_main
from utils.args import args_parser
import utils.geoopt as geoopt



def adjust_learning_rate(optimizer, epoch):
    optimizer.lr = args.initial_lr * (args.decay ** (epoch // 100))


def main(args, train, val, test, train_y, val_y, test_y, graph_matrix, adjacency_matrix, sub, total_sub, validation):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    if validation:
        train       = Variable(torch.from_numpy(train)).double()
        val         = Variable(torch.from_numpy(val)).double()
        test        = Variable(torch.from_numpy(test)).double()

        train_y     = Variable(torch.LongTensor(train_y))
        val_y       = Variable(torch.LongTensor(val_y))
        test_y      = Variable(torch.LongTensor(test_y))
          
        train_dataset = dataloader_in_main(train, train_y)
        val_dataset   = dataloader_in_main(val,  val_y)
        test_dataset  = dataloader_in_main(test, test_y)


        train_kwargs = {'batch_size': args.train_batch_size}
        if use_cuda:
              cuda_kwargs ={'num_workers': 1,
                            # 'sampler': train_sampler,
                              'pin_memory': True,
                              'shuffle': True     
              }
              train_kwargs.update(cuda_kwargs)
              
        valid_kwargs = {'batch_size': args.valid_batch_size}
        if use_cuda:
              cuda_kwargs ={'num_workers': 1,
                            # 'sampler':valid_sampler,
                              'pin_memory': True,
                              'shuffle': True     
              }
              valid_kwargs.update(cuda_kwargs)

        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
              cuda_kwargs ={'num_workers': 1,
                              'pin_memory': True,
                              'shuffle': True      
              }
              test_kwargs.update(cuda_kwargs)

        train_loader  = torch.utils.data.DataLoader(dataset= train_dataset, **train_kwargs)
        valid_loader  = torch.utils.data.DataLoader(dataset= val_dataset, **valid_kwargs)
        test_loader   = torch.utils.data.DataLoader(dataset= test_dataset,  **test_kwargs)
    else:
        train       = Variable(torch.from_numpy(train)).double()
        test        = Variable(torch.from_numpy(test)).double()
        train_y     = Variable(torch.LongTensor(train_y))
        test_y      = Variable(torch.LongTensor(test_y))

        train_dataset = dataloader_in_main(train, train_y)
        test_dataset  = dataloader_in_main(test, test_y)

        train_kwargs  = {'batch_size': args.train_batch_size}

        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                            'pin_memory': True,
                            'shuffle': True     
        }
            train_kwargs.update(cuda_kwargs)

        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs  = {'num_workers': 1,
                            'pin_memory': True,
                            'shuffle': True      
            }
            test_kwargs.update(cuda_kwargs)

        train_loader  = torch.utils.data.DataLoader(dataset= train_dataset, **train_kwargs)
        test_loader   = torch.utils.data.DataLoader(dataset= test_dataset,  **test_kwargs)


    model = Graph_CSPNet_Basic(channel_num = train.shape[1],
        P = Variable(torch.from_numpy(graph_matrix)).double().to(device),  
        mlp = args.mlp,
        dataset = 'HG',
        ).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, parameter in model.named_parameters():
        print(name, ':', parameter.size())

    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.initial_lr)

    early_stopping = EarlyStopping(
        alg_name = 'Graph_CSPNet', 
        path_w   = args.weights_folder_path + 'Graph_CSPNet' + '_checkpoint.pt', 
        patience = args.patience, 
        verbose  = True, 
        )

    print('#####Start Trainning######')

    for epoch in range(1, args.epochs+1):
        start = time.time()

        adjust_learning_rate(optimizer, epoch)

        model.train()

        train_correct = 0

        print('----#------#-----#-----#-----#-----#-----#-----')
        print('['+'Graph_CSPNet'+': Sub No.{}/{}, Epoch {}/{}]:'.format(sub, total_sub, epoch, args.epochs))   
        for batch_idx, (batch_train, batch_train_y) in enumerate(train_loader):

            optimizer.zero_grad()
            logits = model(batch_train.to(device))
            output = F.log_softmax(logits, dim = -1)
            loss   = F.nll_loss(output, batch_train_y.to(device))
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                pred    = output.data.max(1, keepdim=True)[1]
                train_correct += pred.eq(batch_train_y.to(device).data.view_as(pred)).long().cpu().sum()
                torch.save(model.state_dict(), args.weights_folder_path + 'Graph_CSPNet'+'_model.pth')
                torch.save(optimizer.state_dict(), args.weights_folder_path+'optimizer.pth')
                # print('Completed {:.0f}%'.format(100. * (1+batch_idx) / len(train_loader)))
        end = time.time()
        print (str(end-start))
        print('Trainning loss {:.10f} Acc.: {:.4f}'.format(loss.cpu().detach().numpy(), train_correct.item()/len(train_loader.dataset)))
                    

        if validation:
            # print('#####Start Validation######')
            valid_losses  = []
            valid_loss    =  0
            valid_correct =  0

            model.eval()

            for batch_idx, (batch_valid, batch_valid_y) in enumerate(valid_loader):

                logits         = model(batch_valid.to(device))
                output         = F.log_softmax(logits, dim = -1)
                valid_loss    += F.nll_loss(output, batch_valid_y.to(device))
                valid_losses.append(valid_loss.item())
                pred           = output.data.max(1, keepdim=True)[1]
                valid_correct += pred.eq(batch_valid_y.to(device).data.view_as(pred)).long().cpu().sum()

            print('Validate loss: {:.10f} Acc: {:.4f}'.format(sum(valid_losses), valid_correct.item()/len(valid_loader.dataset)))
            
            early_stopping(np.average(valid_losses), model)
            
            if early_stopping.early_stop:
              print("Early Stopping!")
              break
        else:
            pass
        

    print('###############################################################')
    print('START TESTING')
    print('###############################################################')

    
    model.eval()
    test_loss    = 0
    test_correct = 0
    with torch.no_grad():
        for batch_idx, (batch_test, batch_test_y) in enumerate(test_loader):

            logits        = model(batch_test.to(device))
            output        = F.log_softmax(logits, dim = -1)
            test_loss    += F.nll_loss(output, batch_test_y.to(device))            
            test_pred     = output.data.max(1, keepdim=True)[1]
            test_correct += test_pred.eq(batch_test_y.to(device).data.view_as(test_pred)).long().cpu().sum()

            print('-----------------------------------')
            print('Testing Batch {}:'.format(batch_idx))
            print('  Pred Label:', test_pred.view(1, test_pred.shape[0]).cpu().numpy()[0])
            print('Ground Truth:', batch_test_y.numpy())


    return test_correct.item()/len(test_loader.dataset), test_loss.item()/len(test_loader.dataset)


if __name__ == '__main__':

    args   = args_parser()

    seed_n = args.seed
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    alg_df = pd.DataFrame(columns=['Test Acc'])

    print('############Start Task#################')
    
    for sub in range(args.start_No, args.end_No + 1):

        HG_dataset = load_HG(sub,
            alg_name = 'Graph_CSPNet',
            scenario = 'Holdout',
            path='./High-Gamma/data'  
            )

        alg_record = []

        # start      = time.time()

        x_train_stack, x_valid_stack, x_test_stack, y_train, y_valid, y_test = HG_dataset.generate_training_valid_test_set_Holdout()

        graph_M, adj_M = HG_dataset.LGT_graph_matrix_fn()


        acc, loss = main(
        args       = args, 
        train      = x_train_stack, 
        val        = x_valid_stack,
        test       = x_test_stack, 
        train_y    = y_train,
        val_y      = y_valid,
        test_y     = y_test,
        graph_matrix = graph_M,
        adjacency_matrix = adj_M, 
        sub        = sub, 
        total_sub  = args.end_No - args.start_No + 1, 
        validation = True,
        )

        print('##############################################################')

        print('Graph_CSPNet' + ' Testing Loss.: {:4f} Acc: {:4f}'.format(loss, acc))



        alg_record.append(acc)
        
        alg_df.loc[sub] = alg_record
 
    alg_df.to_csv(args.folder_name \
        + time.strftime("[%Y_%m_%d_%H_%M_%S]", time.localtime()) \
        + 'Graph_CSPNet' \
        +'_Sub(' \
        + str(args.start_No) \
        +'-' \
        +str(args.end_No) \
        +')' \
        +'_' \
        + str(args.epochs)\
        + '_ho.csv'\
        , index = False)
    

