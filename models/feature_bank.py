"""Identity feature bank used by the IF2F contrastive loss."""

import torch
from torch import nn


class FeatureBankPlus(nn.Module):
    def __init__(self, dim=512, num_identities=3612, queue_length=2, temperature=0.07):
        super().__init__()
        self.num_identities = num_identities
        self.queue_length = queue_length
        self.temperature = temperature
        queue = torch.randn(num_identities, queue_length, dim)
        self.register_buffer("queue", nn.functional.normalize(queue, dim=-1))
        self.register_buffer(
            "queue_ptr", torch.zeros(num_identities, dtype=torch.long)
        )

    @torch.no_grad()
    def _update(self, keys, identities):
        for key, identity_tensor in zip(keys, identities):
            identity = int(identity_tensor)
            if identity < 0 or identity >= self.num_identities:
                raise IndexError(
                    f"Identity label {identity} is outside the feature bank range "
                    f"[0, {self.num_identities - 1}]."
                )
            pointer = int(self.queue_ptr[identity])
            self.queue[identity, pointer] = key
            self.queue_ptr[identity] = (pointer + 1) % self.queue_length

    def _negative_keys(self, identity):
        mask = torch.arange(self.num_identities, device=self.queue.device) != identity
        return self.queue[mask].flatten(0, 1)

    def forward(self, anchors, positives, original_a, original_b, identities):
        self._update(anchors.detach(), identities)
        self._update(positives.detach(), identities)

        positive_logits = torch.einsum("bd,bd->b", anchors, positives).unsqueeze(1)
        negative_logits = []
        for index, identity_tensor in enumerate(identities):
            negatives = self._negative_keys(int(identity_tensor))
            negatives = torch.cat([original_a, original_b, negatives], dim=0)
            negative_logits.append(
                torch.einsum("bd,md->bm", anchors[index : index + 1], negatives.detach())
            )

        logits = torch.cat([positive_logits, torch.cat(negative_logits, dim=0)], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return logits, labels
