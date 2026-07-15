import torch

def min_snr_loss(loss, alpha_bars_batched, gamma = 5):
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
    w = torch.min(snr, gamma * torch.ones_like(snr)) / (1 + snr)
    avg_loss = loss.mean(dim = (1, 2, 3))
    snr_loss = avg_loss * snr
    snr_loss = snr_loss.mean()
    return snr_loss
