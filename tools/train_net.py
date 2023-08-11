# created by Iran R. Roman <iran@ccrma.stanford.edu>
from desreo.models import build_model
from desreo.datasets import snoop_dogg_loader

import numpy as np
import torch
import scipy
from torchdiffeq import odeint

def train_epoch(
    train_loader,
    model,
    optimizer,
    cfg,
    writer=None,
):
    """
    Perform training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        cfg (CfgNode): configs. 
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()

    for cur_iter, (inputs, targets) in enumerate(
        train_loader
    ):
        inputs = inputs.cuda()
        targets = targets.cuda()

        dur = inputs.shape[1]/model.fs
        t = torch.linspace(0, dur, int(dur * model.fs)).cuda()

        # prepare model for input data
        model.x = inputs
        model.t = t
        init_conds = (torch.tensor(inputs.shape[0]*[[0.1+1j*0]]).cuda(), torch.tensor(inputs.shape[0]*[[model.f0]]).cuda())
        
        # model inference
        pred_y, pred_f = odeint(model, init_conds, t[:-1], method='rk4')
        pred_y = pred_y[...,0].t()
        pred_f = pred_f[...,0].t()

        # calculate the loss and optimize
        inputs_peaks = [scipy.signal.find_peaks(x.cpu().numpy())[0] for x in inputs]
        min_peaks = np.min([len(p) for p in inputs_peaks])
        top_indices = [np.argsort(inputs[ix,x].cpu().numpy())[::-1][:min_peaks] for ix, x in enumerate(inputs_peaks)]
        top_indices = torch.tensor(np.array([np.array(inputs_peaks[it])[t] for it, t in enumerate(top_indices)])).cuda()
        z_peaks = torch.gather(pred_y,1,top_indices)
        circ_mean = torch.mean(torch.exp(1j*torch.angle(z_peaks)),axis=1)
        R = torch.abs(circ_mean)
        loss = -torch.log(torch.mean(R))

        print('loss:\t',"{:.3f}".format(float(loss.detach())),'( R:',"{:.2f}".format(float(torch.mean(R).detach())),')')
        if np.isnan(loss.detach().cpu()):
            return loss.detach().cpu()
        loss.backward()
        optimizer.step()

        # correct model parameters
        #model.alpha.data.clamp_(0,max=0)
        model.alpha.data.clamp_(-float('inf'),max=0)
        model.beta2.data.clamp_(-float('inf'),max=0)
        model.beta1.data.clamp_(0.01,max=float('inf'))
        model.cs.data.clamp_(0.01,max=float('inf'))
        model.cr.data.clamp_(0.01,max=float('inf'))
        model.cw.data.clamp_(0.01,max=float('inf'))
        return loss.detach().cpu()


def eval_epoch(
    eval_loader,
    model,
    cfg,
    writer=None,
):
    """
    Perform training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        cfg (CfgNode): configs. 
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.eval()

    with torch.no_grad():
        for cur_iter, (inputs, targets) in enumerate(
            eval_loader
        ):
            inputs = inputs.cuda()
            targets = targets.cuda()

            dur = inputs.shape[1]/model.fs
            t = torch.linspace(0, dur, int(dur * model.fs)).cuda()

            # prepare model for input data
            model.x = inputs
            model.t = t
            init_conds = (torch.tensor(inputs.shape[0]*[[0.1+1j*0]]).cuda(), torch.tensor(inputs.shape[0]*[[model.f0]]).cuda())
            
            # model inference
            pred_y, pred_f = odeint(model, init_conds, t[:-1], method='rk4')
            pred_y = pred_y[...,0].t()
            pred_f = pred_f[...,0].t()

            # calculate the loss and optimize
            inputs_peaks = [scipy.signal.find_peaks(x.cpu().numpy())[0] for x in inputs]
            min_peaks = np.min([len(p) for p in inputs_peaks])
            top_indices = [np.argsort(inputs[ix,x].cpu().numpy())[::-1][:min_peaks] for ix, x in enumerate(inputs_peaks)]
            top_indices = torch.tensor(np.array([np.array(inputs_peaks[it])[t] for it, t in enumerate(top_indices)])).cuda()
            z_peaks = torch.gather(pred_y,1,top_indices)
            circ_mean = torch.mean(torch.exp(1j*torch.angle(z_peaks)),axis=1)
            R = torch.abs(circ_mean)
            loss = -torch.log(torch.mean(R))

            print(' (val):\t',"{:.3f}".format(float(loss.detach())),'( R:',"{:.2f}".format(float(torch.mean(R).detach())),')')
            return float(loss.cpu())



def train(cfg):
    """
    Train a DESREO model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode):
    """
    # Set random seed from configs.
    if cfg.NUMPY_SEED:
        np.random.seed(cfg.NUMPY_SEED)
    if cfg.TORCH_SEED:
        torch.manual_seed(cfg.TORCH_SEED)

    # Build the model and print parameters.
    model = build_model(cfg)
    print('built DESREO model with parameters:')
    print('alpha\t',"{:.3f}".format(float(model.alpha.detach())))
    print('beta1\t',"{:.3f}".format(float(model.beta1.detach())))
    print('beta2\t',"{:.3f}".format(float(model.beta2.detach())))
    print('cs\t',"{:.3f}".format(float(model.cs.detach())))
    print('cr\t',"{:.3f}".format(float(model.cr.detach())))
    print('cw\t',"{:.3f}".format(float(model.cw.detach())))
    print('f0\t',"{:.3f}".format(float(model.f0.detach())),'\n')

    # Construct the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    # Create the train and val loaders.
    train_loader = snoop_dogg_loader(cfg, "train")
    val_loader = snoop_dogg_loader(cfg, "val")

    ### # set up writer for logging to Tensorboard format.
    ### writer = tb.TensorboardWriter(cfg)

    # Perform the training loop.
    patience = 0
    best_loss  = float('inf')
    while patience < cfg.SOLVER.PATIENCE_LIM:

        # Train for one epoch.
        loss = train_epoch(
            train_loader,
            model,
            optimizer,
            cfg,
        )
        if np.isnan(loss):
            print('*************************************')
            print('*************************************')
            print('*************************************')
            print(f'nan loss found with training data')
            print('*************************************')
            print('*************************************')
            print('*************************************')
            print('\n\n\nreinitalizing model')
            # Build the model and print parameters.
            model = build_model(cfg)
            print('built DESREO model with parameters:')
            print('alpha\t',"{:.3f}".format(float(model.alpha.detach())))
            print('beta1\t',"{:.3f}".format(float(model.beta1.detach())))
            print('beta2\t',"{:.3f}".format(float(model.beta2.detach())))
            print('cs\t',"{:.3f}".format(float(model.cs.detach())))
            print('cr\t',"{:.3f}".format(float(model.cr.detach())))
            print('cw\t',"{:.3f}".format(float(model.cw.detach())))
            print('f0\t',"{:.3f}".format(float(model.f0.detach())),'\n')
            patience = 0
            best_loss  = float('inf')
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
            continue
        loss = eval_epoch(
            val_loader,
            model,
            cfg,
        )
        if np.isnan(loss):
            print('*************************************')
            print('*************************************')
            print('*************************************')
            print(f'nan loss found with validation data')
            print('*************************************')
            print('*************************************')
            print('*************************************')
            print('\n\n\nreinitalizing model')
            # Build the model and print parameters.
            model = build_model(cfg)
            print('built DESREO model with parameters:')
            print('alpha\t',"{:.3f}".format(float(model.alpha.detach())))
            print('beta1\t',"{:.3f}".format(float(model.beta1.detach())))
            print('beta2\t',"{:.3f}".format(float(model.beta2.detach())))
            print('cs\t',"{:.3f}".format(float(model.cs.detach())))
            print('cr\t',"{:.3f}".format(float(model.cr.detach())))
            print('cw\t',"{:.3f}".format(float(model.cw.detach())))
            print('f0\t',"{:.3f}".format(float(model.f0.detach())),'\n')
            patience = 0
            best_loss  = float('inf')
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
            continue
        if loss < best_loss:
            best_loss = loss
            patience = 0
            print('\tnew best validation loss found:')
            print("\t{:.3f}".format(best_loss))
            print('\twith model parameters:')
            print('\talpha\t',"{:.3f}".format(float(model.alpha.detach())))
            print('\tbeta1\t',"{:.3f}".format(float(model.beta1.detach())))
            print('\tbeta2\t',"{:.3f}".format(float(model.beta2.detach())))
            print('\tcs\t',"{:.3f}".format(float(model.cs.detach())))
            print('\tcr\t',"{:.3f}".format(float(model.cr.detach())))
            print('\tcw\t',"{:.3f}".format(float(model.cw.detach())))
            print('\tf0\t',"{:.3f}".format(float(model.f0.detach())),'\n')
        patience += 1
    print('*************************************')
    print('patience elapsed!')
    print('*************************************\n\n\n')

    train(cfg)

