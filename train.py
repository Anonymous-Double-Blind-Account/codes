import os, argparse, torch
from torch.utils.data import DataLoader
import numpy as np

from dataset.build_datasets import build_slios
from model.vqmodel import UEFL
from utils import running_uefl_avg, validate, plot_lc, entropy, silo_training

# UEFL training
def main(args):
    print(args)

    # check cuda
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('training device:', device)

    # load data silos
    silotr = []
    silote = []
    silotr, silote = build_slios(args)
    
    # model
    if args.data == 'cifar10' or args.data == 'cifar100' or args.data == 'gtsrb':
        input_ch = 3 # rgb dataset
    else:
        input_ch = 1 # gray dataset
    mainmodel = UEFL(input_ch=input_ch, dim=args.dim, depth=args.depth, num_codes=args.num_codes, data=args.data, enc = args.encoder, silo_kinds=args.num_dist, seg=args.seg, ema=args.ema)

    # results folder
    results_folder = os.path.join('./results', args.workdir)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # training
    global_accs = [] # accuracy
    global_losses = [] # test loss
    global_vqlosses = [] # vq codebook loss
    global_ppls = [] # perplexity
    entropys = [] # uncertainty
    etp = np.ones(args.num_silo) # uncertainty initialization
    threshold = 0 # threshold initialization
    iteration = 0

    while(etp.max() > threshold): # keep running until all the uncertainty is below the threshold
        iteration += 1
        print('======================================== Iterations {}========================================'.format(iteration))
        # update book_idx for uncertain silos
        if iteration == 1:
            book_idx = [0]*args.num_silo  # all silos use the same shared codebook at the beginning
        else:
            for i in range(args.num_silo):
                if etp[i] > threshold:
                    book_idx[i] = iteration - 1 # each iteration, add a new codebook for uncertain silos
        print('book_idx: ', book_idx)

        codebooks = []
        # keep training for only a few more rounds
        num_round = args.round if iteration == 1 else args.round_plus
        for r in range(num_round):
            if r%10 == 0:
                print('---------------------------------------- Round {}----------------------------------------'.format(r+1))

            ########### training ###########
            avg_model = None
            for s in range(len(silotr)):
                datatr = silotr[s]
                datate = silote[s]
                train_loader = DataLoader(datatr, batch_size=args.batchsz, shuffle=True)
                test_loader = DataLoader(datate, batch_size=args.batchte, shuffle=False)
                lr = args.lr # learning rate
                # load local codebooks for each silo
                if r>0:
                    mainmodel.load_codebooks(codebooks[s])
                # train local model for each silo, locally initialize the model if it's the first round of additional iterations with Kmeans
                init_and_ext_flag = True if r == 0 and iteration > 1 else False
                localmodel, test_loss, acc, vqloss, ppl  = silo_training(train_loader, test_loader, mainmodel, device, args, lr, book_idx[s], init_and_ext=init_and_ext_flag)
                # update local codebooks
                if r == 0:
                    codebooks.append(localmodel.get_codebooks())
                else:
                    codebooks[s] = localmodel.get_codebooks()
                # update global model, ignore codebooks
                avg_model = running_uefl_avg(avg_model, localmodel.state_dict(), 1/len(silotr))
                
                if r%10 == 0:
                    print('silo {}_local: \ttest loss:{:.4f} \tvq loss:{:.4f} \taccuracy:{:.4f} \tperplexity:{:.4f}'.format(s+1, test_loss, vqloss, acc, ppl))
            if r == 0:
                mainmodel.extend_codebooks(iteration) # extend the codewords capacity for the global model, extend once for each iteration
            mainmodel.load_state_dict(avg_model)

            ########### evaluation ###########
            global_acc = []
            global_loss = []
            global_vqloss = []
            global_ppl = []
            predictions = []
            for t in range(args.step): # inference multiple times for uncertainty evaluation
                accuracy = []
                predction = []
                loss = []
                vqlosses = []
                ppls = []
                for s in range(args.num_silo): # for each silo
                    datate = silote[s]
                    test_loader = DataLoader(datate, batch_size=args.batchte, shuffle=False)
                    mainmodel.load_codebooks(codebooks[s]) # load local codebooks for each silo
                    test_loss, acc, pred, vqloss, ppl = validate(test_loader, mainmodel, device, args, book_idx[s])
                    accuracy.append(float("%.4f" % acc))
                    loss.append(float("%.4f" % test_loss))
                    vqlosses.append(float("%.4f" % vqloss))
                    ppls.append(float("%.4f" % ppl))
                    predction.append(pred)
                global_acc.append(accuracy)
                predictions.append(predction)
                global_vqloss.append(vqlosses)
                global_loss.append(loss)
                global_ppl.append(ppls)
            predictions = np.asarray(predictions)
            global_acc = np.asarray(global_acc)
            global_loss = np.asarray(global_loss)
            global_vqloss = np.asarray(global_vqloss)
            global_ppl = np.asarray(global_ppl)

            ent = entropy(predictions) # calculate entropy as uncertainty
            print('round {}: \ttest loss:{} \tvq loss:{} \tacc:{} \tentropy:{}'.format(r+1, 
                    np.mean(global_loss, 0), np.mean(global_vqloss, 0), np.mean(global_acc, 0), np.mean(ent, 1)))
            global_accs.append( np.mean(global_acc, 0))
            global_losses.append( np.mean(global_loss, 0))
            global_vqlosses.append(np.mean(global_vqloss, 0))
            global_ppls.append(np.mean(global_ppl, 0))
            entropys.append(np.mean(ent, 1))
            # torch.save(mainmodel.state_dict(), modelpath)
        
        if args.plot:
            plot_lc(global_accs, args.round+(iteration-1)*args.round_plus, os.path.join(results_folder, 'learning curve_global_Accuracy')) # accuracy
            plot_lc(global_losses, args.round+(iteration-1)*args.round_plus, os.path.join(results_folder, 'learning curve_global_TestLoss')) # test loss
            plot_lc(global_vqlosses, args.round+(iteration-1)*args.round_plus, os.path.join(results_folder, 'learning curve_global_VQLoss')) # vq codebook loss
            plot_lc(global_ppls, args.round+(iteration-1)*args.round_plus, os.path.join(results_folder, 'learning curve_global_Perplexity')) # perplexity
            plot_lc(entropys, args.round+(iteration-1)*args.round_plus, os.path.join(results_folder, 'learning curve_Entropy')) # uncertainty

        # uncertainty evaluation
        etp = entropys[-1]
        etp = np.asarray(etp)
        # compute threshold
        if iteration == 1:
            benchmark = etp.min() if args.min else etp.mean() # use mean or min as the benchmark
            threshold = (1+args.thd) * benchmark

    print('#########################################################=============Done!=============#########################################################')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # data
    argparser.add_argument('--data', type=str, help='which dataset to use(mnist/fmnist/gtsrb/cifar10/cifar100)', default='mnist')
    argparser.add_argument('--num_silo', type=int, help='number of silos', default=9)
    argparser.add_argument('--num_dist', type=int, help='number of distributions', default=3)
    argparser.add_argument('--sample', type=int, help='how many samples for each data silo', default=2000)
    argparser.add_argument('--noise', action='store_true', help='add noise to data silo or not')

    # model
    argparser.add_argument('--encoder', type=str, help='cnn/vgg', default='cnn')
    argparser.add_argument('--dim', type=int, help='the 1st cnov layer channel number(128)', default=128)
    argparser.add_argument('--depth', type=int, help='number of conv blocks', default=3)
    argparser.add_argument('--num_codes', type=int, help='VQ codebook size', default=64)
    argparser.add_argument('--seg', type=int, help='how many segments', default=1)
    argparser.add_argument('--ema', action='store_true', help='whether use ema or not')

    # training
    argparser.add_argument('--dev', type=str, help='cuda device or cpu', default='cuda:0')
    argparser.add_argument('--round', type=int, help='number of federated learning rounds', default=20) # 10, 20, or 50
    argparser.add_argument('--round_plus', type=int, help='number of additional rounds', default=10) # 5
    argparser.add_argument('--epoch', type=int, help='number of local training epochs', default=20) # 5, 10 or 20
    argparser.add_argument('--step', type=int, help='inference times for uncertainty', default=20)
    argparser.add_argument('--batchsz', type=int, help='local batch size', default=1024)
    argparser.add_argument('--batchte', type=int, help='local batch size', default=100)
    argparser.add_argument('--lr', type=float, help='learning rate', default=10e-5)
    argparser.add_argument('--gamma', type=float, help='vq loss weight', default=0.1)
    argparser.add_argument('--thd', type=float, help='threshold for uncertainty evaluation', default=0.1)
    argparser.add_argument('--min', action='store_true', help='whether use minimum uncertainty as the threshold or not')
    argparser.add_argument('--workdir', type=str, help='cuda device or cpu', default='mnist_s9d3s2000_c256r20+10e20s20')
    argparser.add_argument('--plot', action='store_true', help='plot learning curves or not')
    args = argparser.parse_args()
    main(args)
