from model.train import train_eval_loop
import sys
import os
if __name__ == '__main__':
    fold=int(sys.argv[1])
    for lr in [3e-4, 1e-4, 5e-5]:
        print("fold :  %s, lr : %s, PID: %s" % (fold, lr, os.getpid()))
        #for fold in range(5):
        train_eval_loop(lr=lr, dropout=0.1, batch_size=64, fold=fold)
