import torch

def min_snr_loss(loss, alpha_bars_batched, pred_type, gamma = 5):
    """
    Applies min-SNR-gamma to the training loss to stabilize it.
    args:
        - loss = training loss
        - alpha_bars_batched (tensor) = alpha_cum_prod extracted with latent shape
        - gamma (int) = minimum value to use in the min-SNR-gamma equation
    out:
        - snr_loss = weighted loss averaged over the last three dimension
    """
    snr = alpha_bars_batched / (1 - alpha_bars_batched)

    if pred_type == 'v':
        denom = snr + 1
    elif pred_type == 'epsilon':
        denom = snr
    elif pred_type == 'x_0':
        denom = 1
    else:
        raise ValueError(f"pred_type {pred_type} not supported, use [v, x_0, epsilon]")
    
    w = torch.min(snr, gamma * torch.ones_like(snr)) / denom
    avg_loss = loss.mean(dim = (1, 2, 3))
    snr_loss = avg_loss * w.reshape(w.shape[0])
    snr_loss = snr_loss.mean()

    return snr_loss
