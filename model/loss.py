from torch import nn
import torch


class LogSoftmax(nn.LogSoftmax):
    """
    LogSoftmax layer that can handle infinity values.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute log(softmax(tensor))

        :param torch.Tensor tensor: FloatTensor whose log-softmax value will be computed
        :rtype: torch.FloatTensor
        :return: LogSoftmax result.
        """
        # Find maximum values
        max_t = tensor.max(dim=self.dim, keepdim=True).values
        # Reset maximum as zero if it is a finite value.
        tensor = (tensor - max_t.masked_fill(~torch.isfinite(max_t), 0.0))

        # If a row's elements are all infinity, set the row as zeros to avoid NaN.
        all_inf_mask = torch.isinf(tensor).all(dim=self.dim, keepdim=True)
        if all_inf_mask.any().item():
            tensor = tensor.masked_fill(all_inf_mask, 0.0)

        # Forward nn.LogSoftmax.
        return super().forward(tensor)


class SmoothedCrossEntropyLoss(nn.Module):
    """
    Computes cross entropy loss with uniformly smoothed targets.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100, reduction: str = 'batchmean'):
        """
        Cross entropy loss with uniformly smoothed targets.

        :param float smoothing: Label smoothing factor, between 0 and 1 (exclusive; default is 0.1)
        :param int ignore_index: Index to be ignored. (PAD_ID by default)
        :param str reduction: Style of reduction to be done. One of 'batchmean'(default), 'none', or 'sum'.
        """
        assert 0 < smoothing < 1, "Smoothing factor should be in (0.0, 1.0)"
        assert reduction in {'batchmean', 'none', 'sum'}
        super().__init__()

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Computes cross entropy loss with uniformly smoothed targets.
        Since the entropy of smoothed target distribution is always same, we can compute this with KL-divergence.

        :param torch.Tensor input: Log probability for each class. This is a Tensor with shape [B, C]
        :param torch.LongTensor target: List of target classes. This is a LongTensor with shape [B]
        :rtype: torch.Tensor
        :return: Computed loss
        """
        # target=target.flatten()
        target = target.flatten()
        softmax_input = nn.LogSoftmax(dim=1)(input)
        # input=input.
        softmax_input = softmax_input.transpose(1, 2).flatten(0, 1)

        target = target.view(-1, 1)

        # Prepare smoothed target
        # Set all probability of the targets which should be ignored as zero.
        # Since D_KL(p, q) = p (log(p) - log(q)), by setting p(x) ??0, these target cannot affect loss anymore.
        smoothed_target = torch.zeros(softmax_input.shape, requires_grad=False, device=target.device)
        # softmax_input=LogSoftmax(dim=-1)(input)

        # Set target values zero if predicted values are masked with -inf.
        for r, row in enumerate(softmax_input):
            tgt = target[r].item()
            if tgt == self.ignore_index:
                continue

            finites = torch.isfinite(row)
            n_cls = finites.sum().item()
            assert n_cls > 0

            smoothing_prob = self.smoothing / n_cls
            smoothed_target[r].masked_fill_(finites, smoothing_prob)
            smoothed_target[r, tgt] = 1.0 - self.smoothing + smoothing_prob

        # Compute loss: - p log q

        loss = - smoothed_target * softmax_input.masked_fill(~torch.isfinite(softmax_input), 0.0)

        if self.reduction == 'batchmean':
            return loss.sum() / softmax_input.shape[0]
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
