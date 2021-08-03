import math
import copy

import torch
import torch.distributed as dist

__all__ = [
    'power_method',
    'conjugate_gradient_method',
    'reduce_params'
]


def power_method(mvp_fn,
                 model,
                 top_n=1,
                 max_iters=100,
                 tol=1e-3,
                 is_distributed=False,
                 print_progress=False,
                 random_seed=None):
    # main logic is adopted from https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py
    # modified interface and format
    # modified for various matrices and distributed memory run

    assert top_n >= 1
    assert max_iters >= 1

    params = [p for p in model.parameters() if p.requires_grad]

    def _report(message):
        if print_progress:
            print(message)

    eigvals = []
    eigvecs = []
    for i in range(top_n):
        _report(f'start power iteration for lambda({i+1}).')
        vec = [torch.randn_like(p) for p in params]
        if is_distributed:
            vec = _flatten_parameters(vec)
            dist.broadcast(vec, src=0)
            vec = _unflatten_like_parameters(vec, params)

        eigval = None
        last_eigval = None
        # power iteration
        for j in range(max_iters):
            vec = _orthnormal(vec, eigvecs)
            Mv = _mvp(mvp_fn, vec, random_seed=random_seed)
            eigval = _group_product(Mv, vec).item()
            if j > 0:
                diff = abs(eigval - last_eigval) / (abs(last_eigval) + 1e-6)
                _report(f'{j}/{max_iters} diff={diff}')
                if diff < tol:
                    break
            last_eigval = eigval
            vec = Mv
        eigvals.append(eigval)
        eigvecs.append(vec)

    # sort both in descending order
    eigvals, eigvecs = (list(t) for t in zip(*sorted(zip(eigvals, eigvecs))[::-1]))

    return eigvals, eigvecs


def conjugate_gradient_method(mvp_fn,
                              b,
                              init_x=None,
                              damping=1e-3,
                              max_iters=None,
                              tol=1e-8,
                              preconditioner=None,
                              print_progress=False,
                              random_seed=None,
                              save_log=False):
    """
    Solve (A + d * I)x = b by conjugate gradient method.
    d: damping
    Return x when x is close enough to inv(A) * b.
    """
    if max_iters is None:
        n_dim = sum([_b.numel() for _b in b])
        max_iters = n_dim

    def _call_mvp(v):
        return _mvp(mvp_fn, v, random_seed, damping)

    x = init_x
    if x is None:
        x = [torch.zeros_like(_b) for _b in b]
        r = copy.deepcopy(b)
    else:
        Ax = _call_mvp(x)
        r = _group_add(b, Ax, -1)

    if preconditioner is None:
        p = copy.deepcopy(r)
        last_rz = _group_product(r, r)
    else:
        p = preconditioner.precondition_vector(r)
        last_rz = _group_product(r, p)

    b_norm = math.sqrt(_group_product(b, b))

    log = []
    for i in range(max_iters):
        Ap = _call_mvp(p)
        alpha = last_rz / _group_product(p, Ap)
        x = _group_add(x, p, alpha)
        r = _group_add(r, Ap, -alpha)
        rr = _group_product(r, r)
        err = math.sqrt(rr) / b_norm
        log.append({'step': i + 1, 'error': err})
        if print_progress:
            print(f'{i+1}/{max_iters} err={err}')
        if err < tol:
            break
        if preconditioner is None:
            z = r
            rz = rr
        else:
            z = preconditioner.precondition_vector(r)
            rz = _group_product(r, z)

        beta = rz / last_rz  # Fletcher-Reeves
        p = _group_add(z, p, beta)
        last_rz = rz

    if save_log:
        return x, log
    else:
        return x


def _mvp(mvp_fn,
         vec,
         random_seed=None,
         damping=None):
    if random_seed:
        # for matrices that are not deterministic (e.g., fisher_mc)
        torch.manual_seed(random_seed)
    Mv = mvp_fn(vec)
    if damping:
        Mv = _group_add(Mv, vec, damping)
    return Mv


def reduce_params(params, is_master=True, all_reduce=False):
    # pack
    packed_tensor = _flatten_parameters(params)
    if all_reduce:
        # all-reduce
        dist.all_reduce(packed_tensor)
    else:
        dist.reduce(packed_tensor, dst=0)
    if all_reduce or is_master:
        # unpack
        rst = _unflatten_like_parameters(packed_tensor, params)
    else:
        rst = None

    dist.barrier()

    return rst


def _flatten_parameters(params):
    vec = []
    for param in params:
        vec.append(param.flatten())
    return torch.cat(vec)


def _unflatten_like_parameters(vec, params):
    pointer = 0
    rst = []
    for param in params:
        numel = param.numel()
        rst.append(vec[pointer:pointer + numel].view_as(param))
        pointer += numel
    return rst


def _group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def _group_add(xs, ys, alpha=1.):
    return [x.add(y.mul(alpha)) for x, y in zip(xs, ys)]


def _normalization(v):
    s = _group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def _orthnormal(w, v_list):
    for v in v_list:
        w = _group_add(w, v, alpha=-_group_product(w, v))
    return _normalization(w)
