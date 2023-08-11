# created by Iran R. Roman <iran@ccrma.stanford.edu>
import torch
import math

class desreo(torch.nn.Module):

    def __init__(self, cfg):
        #, f0, alpha, beta1, beta2, cs, cw, cr, x, fs, t)

        super().__init__()

        if cfg.DESREO.RND_INIT.ENABLE:
            f0_center = cfg.DESREO.RND_INIT.F0_INIT_CENT
            #self.alpha = torch.tensor(0.0)
            self.alpha = -torch.abs(torch.randn(1))
            self.beta1 = torch.abs(torch.randn(1))
            self.beta2 = -torch.abs(torch.randn(1))
            #self.beta2 = torch.tensor(0.0)
            self.f0 = torch.randn(1) + f0_center
            self.cs = torch.abs(torch.randn(1))
            self.cr = torch.abs(torch.randn(1))
            self.cw = torch.abs(torch.randn(1))
        else:
            self.alpha = torch.tensor(cfg.DESSEO.ALPHA)
            self.beta1 = torch.tensor(cfg.DESSEO.BETA1)
            self.beta2 = torch.tensor(cfg.DESSEO.BETA2)
            self.f0 = torch.tensor(cfg.DESSEO.F0)
            self.cs = torch.tensor(cfg.DESSEO.CS)
            self.cr = torch.tensor(cfg.DESSEO.CR)
            self.cw = torch.tensor(cfg.DESSEO.CW)
        self.alpha = torch.nn.Parameter(self.alpha.cuda())
        self.beta1 = torch.nn.Parameter(self.beta1.cuda())
        self.beta2 = torch.nn.Parameter(self.beta2.cuda())
        self.f0    = torch.nn.Parameter(self.f0.cuda())
        self.cs    = torch.nn.Parameter(self.cs.cuda())
        self.cr    = torch.nn.Parameter(self.cr.cuda())
        self.cw    = torch.nn.Parameter(self.cw.cuda())

        self.fs = cfg.ENV_FS

    def forward(self, t, states):

        z,f = states
        w = f * 2 * math.pi
        zp2 = torch.pow(torch.abs(z),2)
        zp4 = torch.pow(torch.abs(z),4)

        X = self.get_force(t)
        F = self.cs * X * (1 / (1 - torch.conj(z)))
        freq_learn = (self.cr / torch.abs(z)) * (torch.sin(torch.angle(z))) * F
        elast = self.cw * (w - (self.f0 * 2 * math.pi)) / (self.f0 * 2 * math.pi)

        z_dot = f * (z * (self.alpha + 1j*2*math.pi + self.beta1 * zp2 + self.beta2 * zp4 / (1 - zp2)) + F )
        w_dot = torch.real(f * ( -freq_learn - elast))

        return [z_dot, w_dot/(2 * math.pi)]

    def get_force(self,t):
      t_samps = t * self.fs
      t_frac = t_samps % 1
      t_indx = (t_samps // 1).type(torch.LongTensor)
      if t_frac != 0:
        return (1-t_frac) * self.x[:,[t_indx]] + t_frac * self.x[:,[t_indx+1]]
      else:
        return self.x[:,[t_indx]]

def build_model(cfg):

    return desreo(cfg)

