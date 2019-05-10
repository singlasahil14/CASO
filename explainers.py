import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import multiprocessing as mp
import viz

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean_tensor = torch.cuda.FloatTensor(mean).view(1, -1, 1, 1)
std_tensor = torch.cuda.FloatTensor(std).view(1, -1, 1, 1)

def forward(model, x):
    x_norm = (x - mean_tensor)/std_tensor
    logits = model(x_norm)
    return logits

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, 2, dim=1, keepdim=True) + 1e-8
    return d

def _kl_div(log_probs, probs):
    kld = F.kl_div(log_probs, probs, size_average=False)
    return kld / log_probs.shape[0]

def _prox(delta, lr, lambda1):
    """Compute the proximal operator of delta."""
    prod = lr*lambda1
    delta = torch.where(delta<-prod, delta + prod, 
            torch.where(delta<prod, torch.zeros_like(delta), delta - prod))
    return delta

class Explainer:

    def default_gradient(self, model, x):
        x_var = Variable(x, requires_grad=True)
        logits = forward(model, x_var)
        y = logits.max(1)[1]
        self.max_prob, self.max_index = F.softmax(logits, 1).max(1)
        #print('max probability: ', max_prob)
        loss = F.cross_entropy(logits, y, reduction='sum')
        x_grad, = torch.autograd.grad(loss, x_var)
        return x_grad

    def get_input_grad(self, x, logits, y, create_graph=False,
                       cross_entropy=True):
        if cross_entropy:
            loss = F.cross_entropy(logits, y)
            x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)
        else:
            grad_out = torch.zeros_like(logits.data)
            grad_out.scatter_(1, y.data.unsqueeze(0).t(), 1.0)
            x_grad, = torch.autograd.grad(logits, x,
                                          grad_outputs=grad_out,
                                          create_graph=create_graph)
        return x_grad

    def explain(self, model, x, y=None):
        '''Explain model prediction of y given x.
        Args:
            model (torch.nn.Module):
                The model to explain.
            x (torch.cuda.FloatTensor):
                Input tensor.
            y (torch.cuda.LongTensor or None):
                The class to explain. If None use prediction.
        Returns
            torch.cuda.FloatTensor:
                Saliency mapping with the same shape as input x.
        '''
        # TODO add optional y
        pass

class VanillaGradExplainer(Explainer):
    """Regular input gradient explanation."""

    def __init__(self, times_input=False):
        """
        Args:
            times_input: Whether to multiply input as postprocessing.
        """
        self.times_input = times_input

    def explain(self, model, x):
        x = Variable(x, requires_grad=True)
        logits = forward(model, x)
        max_logit, y = logits.max(1)
        x_grad, = torch.autograd.grad(max_logit, x)
        if self.times_input:
            x_grad *= x.data
        return x_grad

class CASO(Explainer):
    def __init__(self, second_order=True, smooth=False, full_hessian=False):
        self._second_order = second_order
        self._smooth = smooth
        self._full_hessian = full_hessian

    def _get_W(self, x, model, batch_size=32):
        x_expand = x.expand(batch_size, -1, -1, -1)
        logits = forward(model, x_expand)
        num_classes = logits.shape[1]
        ws = []
        for i in range(0, num_classes, batch_size):
            logits_trace = torch.trace(logits[:,i:])
            x_grad, = torch.autograd.grad(logits_trace, x_expand, retain_graph=True)
            ws.append(x_grad)
        W = torch.cat(ws, 0)
        W = W[:num_classes]
        W = W.view(num_classes, -1).t()
        return W

    def _initialize_delta(self,x, init):
        '''Initialize the delta vector that becomse the saliency.'''
        batch_size, n_chs, height, width = x.shape
        if init == 'zero':
            delta = torch.zeros(batch_size, n_chs * width * height).cuda()
        elif init == 'random':
            delta = 1e-3*torch.randn(batch_size, n_chs * height * width).cuda()
        delta = nn.Parameter(delta, requires_grad=True)
        return delta

    def _full_eigen(self, model, x):
        batch_size = x.shape[0]
        assert batch_size==1
        x = Variable(x, requires_grad=True)
        logits = forward(model, x)
        self.max_prob, self.max_index = F.softmax(logits, dim=1).max(1)
        y = logits.max(1)[1]
        loss = F.cross_entropy(logits, y)
        x_grad, = torch.autograd.grad(loss, x)
        x_grad = x_grad.view(batch_size, -1)

        W = self._get_W(x, model)
        probs = F.softmax(logits, 1)
        W = W.cpu().detach()
        probs = probs.cpu().detach()

        D = torch.diag(probs[0, :])
        A = (D - probs.t().mm(probs))
        sigma_A, U_A = torch.symeig(-A, eigenvectors=True)
        rank_A = torch.sum(sigma_A<0)
        sigma_A = -sigma_A[:rank_A]
        U_A = U_A[:, :rank_A]

        sigma_A_sqrt = torch.sqrt(sigma_A)
        sigma_A_sqrt = torch.diag(sigma_A_sqrt)
        chol_A = U_A.mm(sigma_A_sqrt)
        B = W.mm(chol_A)
        BTB = B.t().mm(B)
        rank_B = np.linalg.matrix_rank(BTB)

        sigma_B_sq, V_B = torch.symeig(-BTB, eigenvectors=True)
        sigma_B_sq = -sigma_B_sq[:rank_B]
        V_B = V_B[:, :rank_B]

        sigma_B_inv = torch.rsqrt(sigma_B_sq)        
        sigma_B_inv = torch.diag(sigma_B_inv)
        HEV = V_B.mm(sigma_B_inv)
        HEV = B.mm(HEV)

        HEV = HEV.type(torch.cuda.FloatTensor)
        W = W.type(torch.cuda.FloatTensor)
        A = A.type(torch.cuda.FloatTensor)
        hvp_fn = lambda z: ((z.mm(W)).mm(A)).mm(W.t())
        x_grad_data = x_grad.detach()
        sigma_H = sigma_B_sq.detach()
            
        return x_grad_data, sigma_H, HEV, hvp_fn

    def _power_eigen(self, model, x, return_vec=False, smooth=False):
        batch_size, n_chs, height, width = x.shape
        x_flat = Variable(x.view(batch_size, -1), requires_grad=True)
        x = x_flat.view(x.shape)
        logits = forward(model, x)
        y = logits.max(1)[1]
        max_prob, max_index = F.softmax(logits, 1).max(1)
        self.max_prob = max_prob
        self.max_index = max_index
        loss = F.cross_entropy(logits, y, reduction='sum')
        x_grad = torch.autograd.grad(loss, x_flat, create_graph=True)[0]

        if smooth:
            ev = torch.rand(1, n_chs*height*width)
        else:
            ev = torch.rand(batch_size, n_chs*height*width)
        ev = ev - 0.5
        ev = F.normalize(ev.view(ev.shape[0], -1)).cuda()
        ev = Variable(ev, requires_grad=True)
        def hvp_fn(delta):
            hvp = torch.autograd.grad((x_grad*delta).sum(), x_flat, retain_graph=True)[0]
            return hvp
        for iterat in range(10):
            hvp = hvp_fn(ev)
            if smooth:
                hvp = hvp.mean(0, keepdim=True)
            sigma_H = (ev*hvp).sum(1, keepdim=True)
            ev = F.normalize(hvp)
        x_grad_data = x_grad.detach()
        sigma_H = sigma_H.detach()
        if return_vec:
            ev = ev.detach()
            return x_grad_data, sigma_H, ev, hvp_fn
        else:
            return x_grad_data, sigma_H, hvp_fn

    def explain(self, model, x, lambda1=0., lambda2=20., n_iter=10, optim='proximal', lr=3e-2, 
                init='zero', n_samples=16, stddev_spread=0.15, times_input=False, magnitude=False):
        assert lambda2 > 0
        delta = self._iterate(model, x, lambda1=lambda1, lambda2=lambda2, n_iter=n_iter, 
                             optim=optim, lr=lr, init=init, n_samples=n_samples, 
                             stddev_spread=stddev_spread)
        if magnitude:
            delta = delta.view(delta.shape[0], -1)
            delta = (delta*delta)
            delta = delta/torch.sum(delta, 1, keepdim=True)
        delta = delta.view(x.shape).data
        if times_input:
            delta *= x.data
        return delta

    def _iterate(self, model, x, lambda1=0., lambda2=20., n_iter=10, optim='proximal', lr=1e-1, 
                init='zero', n_samples=16, stddev_spread=0.15):
        assert init in ['zero', 'random']
        assert optim in ['sgd', 'lbfgs', 'adam', 'proximal']
        x_orig = x
        if self._smooth:
            stddev = stddev_spread * (x.max() - x.min())
            x_repeat = x.repeat(n_samples, 1, 1, 1)
            noise = torch.randn(x_repeat.shape).cuda() * stddev
            x = noise + x_repeat.clone()

        if self._second_order:
            if self._full_hessian:
                x_grad, sigma_H, HEV, hvp_fn = self._full_eigen(model, x)
                if lambda1 == 0:
                    sigma_H = sigma_H.type(torch.cuda.FloatTensor)
                    delta = self._hessian_inverse(sigma_H, HEV, x_grad, lambda2)
                    return delta
            else:
                if self._smooth:
                    x_grad, sigma_H, fn = self._power_eigen(model, x, smooth=True)
                    x_grad = x_grad.mean(0, keepdim=True)
                    hvp_fn = lambda z: fn(z).mean(0, keepdim=True)
                else:
                    x_grad, sigma_H, fn = self._power_eigen(model, x, smooth=False)
                    hvp_fn = fn
                lambda2 = sigma_H + lambda2
        else:
            hvp_fn = lambda z: 0.
            x_grad = self.default_gradient(model, x)
            if self._smooth:
                x_grad = x_grad.mean(0, keepdim=True)
        x_grad = x_grad.detach().view(x_grad.shape[0], -1)
        delta = self._initialize_delta(x_orig, init)
        if optim=='proximal':
            delta = self._proximal(n_iter, lr, delta, x_grad, 
                                   hvp_fn, lambda1, lambda2)
        else:
            if optim == 'sgd':
                optimizer = torch.optim.SGD([delta], lr=lr, momentum=0.9)
            elif optim == 'lbfgs':
                optimizer = torch.optim.LBFGS([delta], lr=lr, history_size=50)
            elif optim =='adam':
                optimizer = torch.optim.Adam([delta], lr=lr)
            delta = self._optimize(optimizer, n_iter, delta,
                                   x_grad, hvp_fn, lambda1, lambda2)
        return delta

    def _optimize(self, optimizer, n_iter, delta, x_grad, hvp_fn,
                  lambda1, lambda2):
        delta = delta.view(delta.shape[0], -1)
        for i in range(n_iter):
            def closure():
                optimizer.zero_grad()
                _, loss, _ = self._loss_and_grad(delta,
                            x_grad, hvp_fn, lambda1, lambda2)
                loss.backward()
                return loss
            # update delta
            optimizer.step(closure)
        return delta

    def _loss_and_grad(self, delta, x_grad, hvp_fn,
                       lambda1, lambda2):
        t1 = (x_grad*delta).sum(1, keepdim=True)
        l1 = delta.abs().sum(1, keepdim=True)
        l1 = lambda1 * l1

        l2 = 0.5 * (delta * delta).sum(1, keepdim=True)
        l2 = lambda2 * l2
        grad = (lambda2 * delta) - x_grad

        if self._second_order:
            hvp = hvp_fn(delta)
            t2 = 0.5 * (delta * hvp).sum(1, keepdim=True)
            grad += -hvp
        else:
            t2 = torch.zeros_like(l2)
        t1 = torch.sum(t1)
        l1 = torch.sum(l1)
        t2 = torch.sum(t2)
        l2 = torch.sum(l2)

        smooth_loss = (- t1 - t2 + l2)
        loss = (- t1 - t2 + l1 + l2)
        return smooth_loss, loss, grad

    def _proximal(self, n_iter, lr, delta, x_grad, hvp_fn,
                  lambda1, lambda2):
        B = delta
        B_prev = B
        k = 1
        while True:
            mom = (k - 2)/(k + 1)
            V = B + mom*(B - B_prev)
            g_V, loss_V, grad_V = self._loss_and_grad(V, x_grad, 
                                    hvp_fn, lambda1, lambda2)

            B_prev = B
            ls_beta = 0.5
            t = lr
            while True:
                # Compute the update based on gradient, then apply prox.
                B = V - t * grad_V
                B = _prox(B, t, lambda1)

                if ls_beta is None:
                    break
                # The line search condition is to exit when g(b) <= g(v) +
                # grad_v.T@(b - v) + (1/2t)||b - v||^2.
                g_B, loss, grad_B = self._loss_and_grad(B, x_grad, 
                                      hvp_fn, lambda1, lambda2)
                B_V_diff = B - V
                # grad_v.T@(b - v):
                c_2 = (grad_V * B_V_diff).sum()
                # (1/2t)||b - v||^2:
                c_3 = ((B_V_diff ** 2).sum()) / (2. * t)

                upper_bound = g_V + c_2 + c_3
                if g_B <= upper_bound:
                    break
                else:
                    t *= ls_beta
            k = k + 1
            if (k-1)==n_iter:
                break
        return B

class IntegrateGradExplainer(Explainer):
    '''Integrated gradient. The final input multiplication is optional.

    See https://arxiv.org/abs/1703.01365.
    '''
    def __init__(self, n_iter=100, times_input=False):
        self.n_iter = n_iter
        self.times_input = times_input

    def explain(self, model, x):
        grad = 0
        x_data = x.clone()
        for alpha in np.arange(1 / self.n_iter, 1.0, 1 / self.n_iter):
            x_var = Variable(x_data * alpha, requires_grad=True)
            output = forward(model, x_var)
            max_logit, y = output.max(1)
            g, = torch.autograd.grad(max_logit, x_var)
            grad += g
        if self.times_input:
            grad *= x_data
        grad = grad / self.n_iter
        return grad

class SmoothGradExplainer(Explainer):
    '''
    See https://arxiv.org/abs/1706.03825.
    '''
    def __init__(self, base_explainer=None, stdev_spread=0.15,
                 n_samples=16, magnitude=False, times_input=False):
        if base_explainer is None:
            base_explainer = VanillaGradExplainer()
        self.base_explainer = base_explainer
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        self.times_input = times_input

    def explain(self, model, x):
        stdev = self.stdev_spread * (x.max() - x.min())
        total_gradients = 0
        for i in range(self.n_samples):
            noise = torch.randn(x.shape).cuda() * stdev
            x_var = noise + x.clone()
            grad = self.base_explainer.explain(model, x_var)
            total_gradients += grad
        total_gradients /= self.n_samples
        if self.magnitude:
            total_gradients = total_gradients*total_gradients
        if self.times_input:
            total_gradients *= x
        return total_gradients
