import os
import argparse
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS

def load_gray_image_no_norm(path, size=None, device='cpu'):
    im = Image.open(path).convert('L')
    if size is not None:
        im = im.resize(size, Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32)
    return torch.from_numpy(arr.copy())[None, None].to(device)

class TSCMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, num_basis=32):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.act = nn.Tanh()

        self.basis_proj = nn.Linear(hidden_dim, num_basis, bias=True)

        self.final = nn.Linear(num_basis, out_dim, bias=True)

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.xavier_uniform_(self.basis_proj.weight)
        nn.init.zeros_(self.basis_proj.bias)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):

        h = self.act(self.lin(x))
        b = torch.sin(self.basis_proj(h))
        out = self.final(b)
        return out

class TSCMPre(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hidden_dim=64, num_layers=4, num_basis=32, max_disp=20.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(TSCMLayer(in_dim, hidden_dim, hidden_dim, num_basis))
            elif i == num_layers - 1:
                layers.append(TSCMLayer(hidden_dim, out_dim, hidden_dim, num_basis))
            else:
                layers.append(TSCMLayer(hidden_dim, hidden_dim, hidden_dim, num_basis))
        self.layers = nn.ModuleList(layers)
        self.max_disp = float(max_disp)

    def forward(self, coords):
        x = coords
        for layer in self.layers:
            x = layer(x)

        out = torch.tanh(x) * self.max_disp
        return out

class TSCMNet(nn.Module):
    def __init__(self, in_dim=10, out_dim=2, hidden_dim=64, num_layers=4, num_basis=32, max_disp=20.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(TSCMLayer(in_dim, hidden_dim, hidden_dim, num_basis))
            elif i == num_layers - 1:
                layers.append(TSCMLayer(hidden_dim, out_dim, hidden_dim, num_basis))
            else:
                layers.append(TSCMLayer(hidden_dim, hidden_dim, hidden_dim, num_basis))
        self.layers = nn.ModuleList(layers)
        self.max_disp = float(max_disp)

    def forward(self, coords):
        x = coords
        for layer in self.layers:
            x = layer(x)

        out = torch.tanh(x) * self.max_disp
        return out

def make_base_grid(H, W, device):
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([grid_x, grid_y], dim=-1)
    return base

def uv_to_flow_grid(uv, H, W):
    device = uv.device
    scale_x = 2.0 / (W - 1) if W > 1 else 0.0
    scale_y = 2.0 / (H - 1) if H > 1 else 0.0
    u = uv[:,0,:,:] * scale_x
    v = uv[:,1,:,:] * scale_y
    flow = torch.stack([u, v], dim=-1)
    return flow

def warp_ref(I_ref, uv, mode='bicubic', base_grid=None, flow_scale=None):
    B, C, H, W = I_ref.shape
    base = (make_base_grid(H, W, I_ref.device).unsqueeze(0)
            if base_grid is None else base_grid)
    if flow_scale is None:
        flow_scale = uv.new_tensor([2.0 / (W - 1) if W > 1 else 0.0,
                                    2.0 / (H - 1) if H > 1 else 0.0]).view(1,1,1,2)

    grid = (base + uv.permute(0, 2, 3, 1) * flow_scale).clamp(-1.0, 1.0)
    try:
        return F.grid_sample(I_ref, grid, mode=mode,
                             padding_mode='border', align_corners=True)
    except Exception:
        return F.grid_sample(I_ref, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)

def run_full_forward(model, points, coords_batch):
    if coords_batch >= points.shape[0]:
        return model(points)
    return torch.cat([model(chunk) for chunk in torch.split(points, coords_batch, dim=0)], dim=0)

def make_sobel_kernel(device, dtype=torch.float32):
    return torch.tensor([
        [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]],
        [[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]],
    ], device=device, dtype=dtype)

def compute_image_gradients(img, sobel_kernel=None):
    if sobel_kernel is None:
        sobel_kernel = make_sobel_kernel(img.device, img.dtype)

    return F.conv2d(img, sobel_kernel, padding=1)

def gradient_consistency_loss(I_ref_warp, I_def=None,
                              sobel_kernel=None, grad_def_cached=None):
    grad_ref = compute_image_gradients(I_ref_warp, sobel_kernel)
    if grad_def_cached is None:
        if I_def is None:
            raise ValueError('I_def或grad_def_cached至少提供一个')
        grad_def_cached = compute_image_gradients(I_def, sobel_kernel)

    return (grad_ref - grad_def_cached).abs().sum(dim=1).mean()

def charbonnier_loss(Iw, I_t, eps=1e-3):
    diff = Iw - I_t
    return torch.mean(torch.sqrt(diff * diff + eps * eps))

def laplacian_smoothness_loss(uv_map):
    u = uv_map[:, 0:1, :, :]
    v = uv_map[:, 1:2, :, :]

    def laplace_loss(field):
        up    = field[:, :, :-2, 1:-1]
        down  = field[:, :, 2:, 1:-1]
        left  = field[:, :, 1:-1, :-2]
        right = field[:, :, 1:-1, 2:]
        center = field[:, :, 1:-1, 1:-1]
        neighbor_mean = (up + down + left + right) / 4.0
        return F.mse_loss(center, neighbor_mean, reduction='mean')

    return 0.5 * (laplace_loss(u) + laplace_loss(v))

def build_inputs_from_neighbor_buffer(neighbor_uv, H, W, device, coords_t=None):
    if not torch.is_tensor(neighbor_uv):
        nb = torch.as_tensor(neighbor_uv, dtype=torch.float32, device=device)
    else:
        nb = neighbor_uv.to(device=device, dtype=torch.float32)
    if nb.shape != (H, W, 2):
        raise ValueError(f"neighbor_uv 应为 {(H,W,2)}，实际是 {tuple(nb.shape)}")

    if coords_t is None:
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        xy = torch.stack((xx, yy), -1)
    else:
        xy = coords_t.view(H, W, 2)

    inp = torch.empty((H, W, 10), device=nb.device, dtype=nb.dtype)
    inp[..., :2] = xy

    inp[1:, :, 2:4] = nb[:-1, :, :]
    inp[0, :, 2:4] = nb[0, :, :]

    inp[:, :-1, 4:6] = nb[:, 1:, :]
    inp[:, -1, 4:6] = nb[:, -1, :]

    inp[:-1, :, 6:8] = nb[1:, :, :]
    inp[-1, :, 6:8] = nb[-1, :, :]

    inp[:, 1:, 8:10] = nb[:, :-1, :]
    inp[:, 0, 8:10] = nb[:, 0, :]
    return inp.view(H * W, 10)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_numeric_array(path, name):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext in ('.txt', '.csv'):
        delimiter = ',' if ext == '.csv' else None
        arr = np.loadtxt(path, delimiter=delimiter)
    elif ext in ('.png', '.bmp', '.tif', '.tiff'):
        arr = np.asarray(Image.open(path).convert('L'))
    else:
        raise ValueError(
            f"Unsupported {name} format '{ext}'. Use .npy, .txt, .csv, .png, .bmp, .tif, or .tiff."
        )
    return np.asarray(arr)

def normalize_field_shape(field, expected_shape, name):
    arr = np.asarray(field).squeeze()
    if arr.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape} after squeezing; got {arr.shape}."
        )
    return arr.astype(np.float64, copy=False)

def evaluate_displacement(uv_pred, u_gt, v_gt, mask=None):
    if uv_pred.ndim != 3 or uv_pred.shape[-1] != 2:
        raise ValueError(f"uv_pred must have shape (H, W, 2), got {uv_pred.shape}.")
    H, W, _ = uv_pred.shape
    expected = (H, W)
    u_gt = normalize_field_shape(u_gt, expected, 'u_gt')
    v_gt = normalize_field_shape(v_gt, expected, 'v_gt')

    valid = np.isfinite(uv_pred[..., 0]) & np.isfinite(uv_pred[..., 1])
    valid &= np.isfinite(u_gt) & np.isfinite(v_gt)
    if mask is not None:
        mask = normalize_field_shape(mask, expected, 'mask')
        valid &= np.isfinite(mask) & (mask > 0)

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError('The evaluation mask contains no valid pixels.')

    result = {'valid_pixels': n_valid}
    for idx, (name, gt) in enumerate((('u', u_gt), ('v', v_gt))):
        error = uv_pred[..., idx].astype(np.float64, copy=False)[valid] - gt[valid]
        result[name] = {
            'MAE': float(np.mean(np.abs(error))),
            'MSE': float(np.mean(error ** 2)),
            'RMSE': float(np.sqrt(np.mean(error ** 2))),
        }
    return result

def print_evaluation_metrics(metrics):
    print(f"Evaluation valid pixels: {metrics['valid_pixels']}")
    for component in ('u', 'v'):
        m = metrics[component]
        print(
            f"{component}-component: MAE={m['MAE']:.6e}, "
            f"MSE={m['MSE']:.6e}, RMSE={m['RMSE']:.6e}"
        )

def train_pinn_two_stage(
    ref_path, deform_path,
    out_dir='neighbor_prediction_out', device='cuda', H=480, W=480,
    max_disp=15.0, seed=0, preheat_epochs=1000,
    formal_adam_epochs=3000, formal_lbfgs_epochs=200,
    stage1_adam_lr=1e-4, stage1_adam_wd=1e-5,
    stage2_adam_lr=1e-4, stage2_adam_wd=5e-5,
    lbfgs_lr=0.1, coords_batch=230400,
    hidden_dim=96, num_hidden_layers=6,
    lbfgs_inner_max_iter=1, log_pre=100, log_adam=50, log_lbfgs=10,
    warp_mode='bicubic', u_gt_path=None, v_gt_path=None, mask_path=None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device if str(device).startswith('cuda') and torch.cuda.is_available() else 'cpu')
    set_random_seed(seed)
    if coords_batch <= 0:
        raise ValueError('coords_batch must be positive')
    if (u_gt_path is None) != (v_gt_path is None):
        raise ValueError('Provide both --u_gt and --v_gt for quantitative evaluation.')
    if mask_path is not None and u_gt_path is None:
        raise ValueError('--mask can only be used together with --u_gt and --v_gt.')
    if warp_mode not in ('bilinear', 'nearest', 'bicubic'):
        raise ValueError("warp_mode must be one of: 'bilinear', 'nearest', or 'bicubic'")

    I_ref = load_gray_image_no_norm(ref_path, size=(W, H), device=device)
    I_def = load_gray_image_no_norm(deform_path, size=(W, H), device=device)
    assert I_ref.shape[-2:] == (H, W) and I_def.shape[-2:] == (H, W)

    xs = np.linspace(-1.0, 1.0, W)
    ys = np.linspace(-1.0, 1.0, H)
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    coords = np.stack((gx, gy), axis=-1).reshape(-1, 2).astype(np.float32)
    coords_t = torch.from_numpy(coords).to(device)
    base_grid = make_base_grid(H, W, device).unsqueeze(0)
    flow_scale = torch.tensor(
        [2.0 / (W - 1) if W > 1 else 0.0,
         2.0 / (H - 1) if H > 1 else 0.0],
        dtype=I_ref.dtype, device=device,
    ).view(1, 1, 1, 2)
    sobel_kernel = make_sobel_kernel(device, I_ref.dtype)
    with torch.no_grad():
        grad_def_cached = compute_image_gradients(I_def, sobel_kernel)

    preheat_model = TSCMPre(
        in_dim=2, out_dim=2, hidden_dim=hidden_dim,
        num_layers=num_hidden_layers, max_disp=max_disp
    ).to(device)
    opt_pre = Adam(preheat_model.parameters(), lr=stage1_adam_lr, weight_decay=stage1_adam_wd)

    # Stage I: Coordinate-based pre-training
    print('=== STAGE I: Coordinate-based pre-training (Adam) ===')
    for epoch in range(1, preheat_epochs + 1):
        preheat_model.train()
        opt_pre.zero_grad(set_to_none=True)
        uv_map = run_full_forward(preheat_model, coords_t, coords_batch).view(1, H, W, 2).permute(0, 3, 1, 2).contiguous()
        warped = warp_ref(I_ref, uv_map, mode=warp_mode, base_grid=base_grid, flow_scale=flow_scale)
        diff = warped - I_def
        photo_loss = torch.sqrt(diff.square() + 1e-6).mean()
        grad = gradient_consistency_loss(warped, sobel_kernel=sobel_kernel, grad_def_cached=grad_def_cached)
        smooth = 0.0
        loss = photo_loss + 0.1 * grad + 0.1 * smooth
        loss.backward()
        opt_pre.step()
        if epoch == 1 or (log_pre > 0 and epoch % log_pre == 0):
            print(f'Preheat Adam {epoch}/{preheat_epochs}: loss={loss.detach().item():.6e}, photo={photo_loss.detach().item():.6e}, grad={grad.detach().item():.6e}')

    # Stage II: Neighborhood-guided training

    model = TSCMNet(
        in_dim=10, out_dim=2, hidden_dim=hidden_dim,
        num_layers=num_hidden_layers, max_disp=max_disp
    ).to(device)
    pre_dict = preheat_model.state_dict()
    model_dict = model.state_dict()
    copied = 0
    coord_columns_copied = False
    with torch.no_grad():
        for k, v in pre_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k].copy_(v)
                copied += 1
            elif k == 'layers.0.lin.weight' and k in model_dict:
                dest = model_dict[k]
                if v.ndim == 2 and dest.ndim == 2 and v.shape[0] == dest.shape[0] and v.shape[1] == 2 and dest.shape[1] == 10:
                    dest[:, :2].copy_(v)
                    coord_columns_copied = True
    model.load_state_dict(model_dict)
    print(f'Transferred {copied} matching tensors; first-layer coordinate weights transferred: {coord_columns_copied}.')

    preheat_model.eval()
    with torch.no_grad():
        neighbor_uv = run_full_forward(preheat_model, coords_t, coords_batch).view(H, W, 2).detach().contiguous()
    opt_adam = Adam(model.parameters(), lr=stage2_adam_lr, weight_decay=stage2_adam_wd)

    del pre_dict, model_dict, preheat_model, opt_pre
    if preheat_epochs:
        del uv_map, warped, diff, photo_loss, grad, smooth, loss

    print('=== STAGE II: Neighborhood-guided training (Adam) ===')
    actual_adam_epochs = 0
    for epoch in range(1, formal_adam_epochs + 1):
        model.train()
        opt_adam.zero_grad(set_to_none=True)
        inputs = build_inputs_from_neighbor_buffer(neighbor_uv, H, W, device, coords_t)
        uv_map = run_full_forward(model, inputs, coords_batch).view(1, H, W, 2).permute(0, 3, 1, 2).contiguous()
        warped = warp_ref(I_ref, uv_map, mode=warp_mode, base_grid=base_grid, flow_scale=flow_scale)
        diff = warped - I_def
        photo_loss = torch.sqrt(diff.square() + 1e-6).mean()
        smooth = laplacian_smoothness_loss(uv_map)
        grad = gradient_consistency_loss(warped, sobel_kernel=sobel_kernel, grad_def_cached=grad_def_cached)
        loss = photo_loss + 0.1 * grad + 0.1 * smooth
        loss.backward()
        opt_adam.step()

        with torch.no_grad():
            neighbor_uv = run_full_forward(model, inputs, coords_batch).view(H, W, 2).detach().contiguous()
        actual_adam_epochs = epoch

        show = (epoch == 1 or (log_adam > 0 and epoch % log_adam == 0))
        if show:
            print(
                f'Formal Adam {epoch}/{formal_adam_epochs}: '
                f'loss={loss.detach().item():.6e}, '
                f'photo={photo_loss.detach().item():.6e}, '
                f'smooth={smooth.detach().item():.6e}, '
                f'grad={grad.detach().item():.6e}'
            )

    if actual_adam_epochs:
        del uv_map, warped, diff, photo_loss, smooth, grad, loss, inputs

    print('=== STAGE II: Neighborhood-guided refinement (L-BFGS) ===')
    inner_iter = int(lbfgs_inner_max_iter)
    if inner_iter < 1:
        raise ValueError("lbfgs_inner_max_iter must be >= 1")
    opt_lbfgs = LBFGS(
        model.parameters(), lr=lbfgs_lr, max_iter=inner_iter,
        history_size=20, line_search_fn='strong_wolfe'
    )
    lbfgs_inputs = None
    closure_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        opt_lbfgs.zero_grad(set_to_none=True)
        uv = run_full_forward(model, lbfgs_inputs, coords_batch).view(1, H, W, 2).permute(0, 3, 1, 2).contiguous()
        warped_now = warp_ref(I_ref, uv, mode=warp_mode, base_grid=base_grid, flow_scale=flow_scale)
        diff_now = warped_now - I_def
        photo_loss = torch.sqrt(diff_now.square() + 1e-6).mean()
        grad = gradient_consistency_loss(warped_now, sobel_kernel=sobel_kernel, grad_def_cached=grad_def_cached)
        smooth = laplacian_smoothness_loss(uv)
        loss = photo_loss + 0.1 * grad + 0.1 * smooth
        loss.backward()
        return loss

    for it in range(1, formal_lbfgs_epochs + 1):
        lbfgs_inputs = build_inputs_from_neighbor_buffer(neighbor_uv, H, W, device, coords_t)

        loss_val = opt_lbfgs.step(closure)
        with torch.no_grad():
            uv_next = run_full_forward(model, lbfgs_inputs, coords_batch)
            neighbor_uv = uv_next.view(H, W, 2).detach().contiguous()
        if it == 1 or (log_lbfgs > 0 and it % log_lbfgs == 0):
            print(f'L-BFGS outer {it}/{formal_lbfgs_epochs}: loss={loss_val.detach().item():.6e}, total_closure_calls={closure_calls}')

    model.eval()
    with torch.no_grad():
        inputs_final = build_inputs_from_neighbor_buffer(neighbor_uv, H, W, device, coords_t)
        uv_final = run_full_forward(model, inputs_final, coords_batch).view(H, W, 2).contiguous()

    uv_np = uv_final.cpu().numpy()
    metrics = None
    if u_gt_path is not None:
        u_gt = load_numeric_array(u_gt_path, 'u_gt')
        v_gt = load_numeric_array(v_gt_path, 'v_gt')
        mask = load_numeric_array(mask_path, 'mask') if mask_path is not None else None
        metrics = evaluate_displacement(uv_np, u_gt, v_gt, mask)
        print_evaluation_metrics(metrics)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, k, name in zip(axes, (0, 1), ('u', 'v')):
        image = ax.imshow(uv_np[..., k], origin='lower', cmap='jet')
        ax.set_title(f'Predicted {name} displacement (pixel)')
        ax.axis('off')
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    png_path = os.path.join(out_dir, 'final_uv.png')
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved prediction visualization:', png_path)
    return {'uv': uv_np, 'png_path': png_path, 'metrics': metrics}

def main():
    p = argparse.ArgumentParser(
        description='NeighborDIC two-stage training with optional quantitative evaluation'
    )
    p.add_argument('--ref', type=str)
    p.add_argument('--deform', type=str)
    p.add_argument('--out', type=str, default='out')
    p.add_argument('--H', type=int)
    p.add_argument('--W', type=int)
    p.add_argument('--max_disp', type=float, default=15.0)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int)

    p.add_argument('--preheat_epochs', type=int, default=1000)
    p.add_argument('--formal_adam_epochs', type=int, default=3000)
    p.add_argument('--formal_lbfgs_epochs', type=int, default=200)

    p.add_argument('--stage1_adam_lr', type=float, default=1e-4)
    p.add_argument('--stage1_adam_wd', type=float, default=1e-5)
    p.add_argument('--stage2_adam_lr', type=float, default=1e-4)
    p.add_argument('--stage2_adam_wd', type=float, default=5e-5)
    p.add_argument('--lbfgs_lr', type=float, default=0.1)
    p.add_argument('--lbfgs_inner_max_iter', type=int, default=1)

    p.add_argument('--coords_batch', type=int)
    p.add_argument('--hidden_dim', type=int, default=96)
    p.add_argument('--num_hidden_layers', type=int, default=6)
    p.add_argument('--warp_mode', type=str, choices=('bilinear', 'nearest', 'bicubic'), default='bicubic')

    p.add_argument('--u_gt', type=str, default=None)
    p.add_argument('--v_gt', type=str, default=None)
    p.add_argument('--mask', type=str, default=None)

    p.add_argument('--log_pre', type=int, default=100)
    p.add_argument('--log_adam', type=int, default=50)
    p.add_argument('--log_lbfgs', type=int, default=10)

    args = p.parse_args()
    result = train_pinn_two_stage(
        ref_path=args.ref,
        deform_path=args.deform,
        out_dir=args.out,
        device=args.device,
        H=args.H,
        W=args.W,
        max_disp=args.max_disp,
        seed=args.seed,
        preheat_epochs=args.preheat_epochs,
        formal_adam_epochs=args.formal_adam_epochs,
        formal_lbfgs_epochs=args.formal_lbfgs_epochs,
        stage1_adam_lr=args.stage1_adam_lr,
        stage1_adam_wd=args.stage1_adam_wd,
        stage2_adam_lr=args.stage2_adam_lr,
        stage2_adam_wd=args.stage2_adam_wd,
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_inner_max_iter=args.lbfgs_inner_max_iter,
        coords_batch=args.coords_batch,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        warp_mode=args.warp_mode,
        u_gt_path=args.u_gt,
        v_gt_path=args.v_gt,
        mask_path=args.mask,
        log_pre=args.log_pre,
        log_adam=args.log_adam,
        log_lbfgs=args.log_lbfgs,
    )
    print('Final uv shape:', result['uv'].shape)
    if result['metrics'] is None:
        print('Quantitative evaluation skipped (no ground-truth fields supplied).')

if __name__ == '__main__':
    main()
