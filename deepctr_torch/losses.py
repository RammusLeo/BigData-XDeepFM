import torch
import torch.nn.functional as F
import torch.nn as nn

def contrastive_loss(logits, labels, tau=0.4):
    """
    Compute the contrastive loss as defined in the given formula.

    Args:
        logits (torch.Tensor): Logits matrix of shape (N, D), embeddings before activation.
        labels (torch.Tensor): Tensor of shape (N,).
        tau (float): Temperature scaling factor.

    Returns:
        torch.Tensor: Contrastive loss value.
    """
    N = logits.size(0)  # Batch size
    device = logits.device

    # Compute pairwise similarities
    similarity = torch.mm(logits, logits.T) / tau  # Shape: (N, N)

    # Mask to exclude diagonal (self-similarity)
    mask_self = torch.eye(N, dtype=torch.bool, device=device)

    # Positive pair mask: same label, exclude self
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # Shape: (N, N)
    mask_positive = labels_equal & ~mask_self

    total_loss = 0.0

    for i in range(N):
        # Positive samples for i-th sample
        pos_indices = mask_positive[i].nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:  # Skip if no positive pairs
            continue

        # Numerator: Positive similarities
        numerator = torch.exp(similarity[i, pos_indices])

        # Denominator: All except self
        mask_all_except_self = ~mask_self[i]
        denominator = torch.exp(similarity[i, mask_all_except_self]).sum()

        # Log-loss
        loss_i = -torch.log(numerator / denominator).mean()
        total_loss += loss_i

    total_loss /= N
    return total_loss

def combined_loss(y_pred, labels, z, alpha=0.9, tau=0.4):
    """
    Compute a weighted combination of contrastive loss and BCE loss.

    Args:
        logits (torch.Tensor): Logits matrix of shape (N, D), embeddings before activation.
        labels (torch.Tensor): Tensor of shape (N,) for contrastive loss.
        targets (torch.Tensor): Binary targets for BCE loss, shape (N,).
        alpha (float): Weight for the contrastive loss.
        tau (float): Temperature scaling factor.

    Returns:
        torch.Tensor: Weighted combined loss.
    """
    # Contrastive loss
    loss_contrastive = contrastive_loss(z, labels, tau)

    # z_bce = z[:, 1]  # Select logits for the positive class (class 1)
    # loss_bce = F.binary_cross_entropy_with_logits(z_bce, labels.float())
    criterion = nn.CrossEntropyLoss()
    loss_bce = criterion(z, labels.long())


    # Weighted sum of the losses
    total_loss = 0.1 * (1 - alpha) * loss_contrastive + alpha * loss_bce
    
    # print(f"Contrastive Loss: {loss_contrastive.item()}, BCE Loss: {loss_bce.item()}")
    return total_loss

# Example Usage
if __name__ == "__main__":
    # Dummy logits (batch size 4, embedding dimension 5)
    z = torch.randn(4, 2, requires_grad=True)
    logits = nn.Sigmoid()(z)
    y_pred = torch.argmax(logits, dim=1)
    # Labels for contrastive loss
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    # Alpha for weighting losses
    alpha = 0.9
    loss = combined_loss(labels, y_pred, z, logits, alpha=alpha, tau=0.4)
    print(f"Combined Loss: {loss.item()}")
