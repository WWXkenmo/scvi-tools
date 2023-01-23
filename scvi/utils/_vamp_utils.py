from torch.autograd import Variable
import math

def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    if mu.is_cuda:
        eps = torch.cuda.FloatTensor(std.size()).normal_()
    else:
        eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu)


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )