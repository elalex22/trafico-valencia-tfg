"""
Loop de entrenamiento genérico + métricas.
"""

import time
import numpy as np
import torch

from config import MODELS_DIR


def persistencia_mae(Y, M, horizon=1):
    """MAE/RMSE del modelo trivial Y[t+horizon] = Y[t], en escala real (veh/h)."""
    T = Y.shape[0]
    Y_now = np.asarray(Y[:T - horizon])
    Y_future = np.asarray(Y[horizon:])
    M_future = np.asarray(M[horizon:])

    Y_now_real = np.expm1(np.clip(Y_now, 0, None))
    Y_future_real = np.expm1(np.clip(Y_future, 0, None))

    ae = np.abs(Y_now_real - Y_future_real) * M_future
    se = (Y_now_real - Y_future_real) ** 2 * M_future
    total = M_future.sum()
    return ae.sum() / max(total, 1), np.sqrt(se.sum() / max(total, 1))


def masked_mae(y_pred, y_true, mask):
    return ((y_pred - y_true).abs() * mask).sum() / (mask.sum() + 1e-8)


def evaluate(model, loader, device, edge_index, is_gcn=True, batches_limit=None):
    model.eval()
    tot_ae_r = tot_se_r = tot_n = 0.0
    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            if batches_limit and i >= batches_limit:
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x, edge_index) if is_gcn else model(x)
            pred_r = torch.expm1(pred.clamp(min=0))
            y_r = torch.expm1(y)
            tot_ae_r += ((pred_r - y_r).abs() * m).sum().item()
            tot_se_r += ((pred_r - y_r) ** 2 * m).sum().item()
            tot_n += m.sum().item()
    return {
        'mae_real': tot_ae_r / max(tot_n, 1),
        'rmse_real': (tot_se_r / max(tot_n, 1)) ** 0.5,
    }


def entrenar(nombre, model, ds_train, ds_val, ds_test, device, persistencia_test,
             edge_index=None, is_gcn=True, batch_size=16, epochs=10, lr=1e-3,
             patience=3, log_every=200):
    from torch.utils.data import DataLoader

    print(f'\n{"═"*70}\nEXPERIMENTO: {nombre}\n{"═"*70}')

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'  batches train: {len(loader_train)}')
    print(f'  parámetros: {sum(p.numel() for p in model.parameters()):,}')

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=2, min_lr=1e-5)

    best_val = float('inf')
    epochs_sin_mejora = 0
    ckpt = MODELS_DIR / f'{nombre}.pt'

    for epoch in range(epochs):
        print(f'\n  ── Epoch {epoch+1}/{epochs} (lr={opt.param_groups[0]["lr"]:.2e}) ──')
        model.train()
        tot_loss, nb = 0.0, 0
        t0 = t_last = time.time()

        for i, (x, y, m) in enumerate(loader_train):
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x, edge_index) if is_gcn else model(x)
            loss = masked_mae(pred, y, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot_loss += loss.item(); nb += 1

            if (i + 1) % log_every == 0:
                speed = log_every / (time.time() - t_last)
                t_last = time.time()
                print(f'    batch {i+1}/{len(loader_train)} | loss={tot_loss/nb:.4f} | {speed:.1f} batch/s')

        val_m = evaluate(model, loader_val, device, edge_index, is_gcn)
        scheduler.step(val_m['mae_real'])
        print(f'    train_loss={tot_loss/nb:.4f} | val_mae={val_m["mae_real"]:.2f} veh/h | ({time.time()-t0:.0f}s)')

        if val_m['mae_real'] < best_val:
            best_val = val_m['mae_real']
            torch.save(model.state_dict(), ckpt)
            epochs_sin_mejora = 0
            print(f'    ✓ guardado (best_val={best_val:.2f})')
        else:
            epochs_sin_mejora += 1
            print(f'    sin mejora ({epochs_sin_mejora}/{patience})')
            if epochs_sin_mejora >= patience:
                print('    → Early stopping')
                break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    test_m = evaluate(model, loader_test, device, edge_index, is_gcn)
    print(f'\n  ═ TEST ═\n  MAE:  {test_m["mae_real"]:.2f} veh/h\n  RMSE: {test_m["rmse_real"]:.2f} veh/h')
    print(f'  vs persistencia: {persistencia_test:.2f} veh/h ({test_m["mae_real"]-persistencia_test:+.2f})')
    return test_m