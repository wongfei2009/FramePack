# Better Flow Matching UniPC by Lvmin Zhang
# (c) 2025
# CC BY-SA 4.0
# Attribution-ShareAlike 4.0 International Licence


import torch

from tqdm.auto import trange


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


class FlowMatchUniPC:
    def __init__(self, model, extra_args, variant='bh1'):
        self.model = model
        self.variant = variant
        self.extra_args = extra_args
        # Extract movement_scale from extra_args (default to 0.1 if not specified)
        self.movement_scale = extra_args.get('movement_scale', 0.1)

    def model_fn(self, x, t):
        return self.model(x, t, **self.extra_args)

    def update_fn(self, x, model_prev_list, t_prev_list, t, order):
        assert order <= len(model_prev_list)
        dims = x.dim()

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = - torch.log(t_prev_0)
        lambda_t = - torch.log(t)
        model_prev_0 = model_prev_list[-1]

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = - torch.log(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h[0]
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError('Bad variant!')

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        use_predictor = len(D1s) > 0

        if use_predictor:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], device=b.device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None
            rhos_p = None

        if order == 1:
            rhos_c = torch.tensor([0.5], device=b.device)
        else:
            rhos_c = torch.linalg.solve(R, b)

        x_t_ = expand_dims(t / t_prev_0, dims) * x - expand_dims(h_phi_1, dims) * model_prev_0

        if use_predictor:
            pred_res = torch.tensordot(D1s, rhos_p, dims=([1], [0]))
        else:
            pred_res = 0

        x_t = x_t_ - expand_dims(B_h, dims) * pred_res
        
        # We don't need to pass movement_scale to the model
        # It's used directly in the sample method
        model_t = self.model_fn(x_t, t)

        if D1s is not None:
            corr_res = torch.tensordot(D1s, rhos_c[:-1], dims=([1], [0]))
        else:
            corr_res = 0

        D1_t = (model_t - model_prev_0)
        x_t = x_t_ - expand_dims(B_h, dims) * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    def sample(self, x, sigmas, callback=None, disable_pbar=False):
        order = min(3, len(sigmas) - 2)
        model_prev_list, t_prev_list = [], []
        for i in trange(len(sigmas) - 1, disable=disable_pbar):
            vec_t = sigmas[i].expand(x.shape[0])

            # For i=5 (25 steps), apply the movement effect directly here
            if i == 5:  # for 25 steps, empirical value
                import math
                
                # Get movement_scale from extra_args, defaulting to 0.1 if not provided
                movement_scale = self.movement_scale
                
                # move along H/W axis with sin curve
                dim = -1  # -2 for H, -1 for W
                f = x.shape[2]
                size = x.shape[dim]
                
                for j in range(f):
                    amount = math.sin((j / f) * math.pi * 2) * size * movement_scale
                    print(f"amount: {amount}, j: {j}, f: {f}, size: {size}")
                    x[:, :, j, :, :] = torch.roll(x[:, :, j, :, :], int(amount), dims=dim)

            if i == 0:
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
            elif i < order:
                init_order = i
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, init_order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            else:
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)

            model_prev_list = model_prev_list[-order:]
            t_prev_list = t_prev_list[-order:]

            if callback is not None:
                callback_data = {'x': x, 'i': i, 'denoised': model_prev_list[-1]}
                
                # Check if cache info is available and pass it to the callback
                if hasattr(model_prev_list[-1], 'cache_info'):
                    callback_data['cache_info'] = getattr(model_prev_list[-1], 'cache_info')
                
                # Pass the movement_scale to the callback
                callback_data['movement_scale'] = self.movement_scale
                
                # Call the callback with the data
                callback(callback_data)

        return model_prev_list[-1]


def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=False, variant='bh1'):
    assert variant in ['bh1', 'bh2']
    return FlowMatchUniPC(model, extra_args=extra_args, variant=variant).sample(noise, sigmas=sigmas, callback=callback, disable_pbar=disable)
