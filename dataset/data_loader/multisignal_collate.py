"""Collate for the multi-signal dict batch contract."""
from torch.utils.data import default_collate

_REQUIRED = ('frames', 'channel_mask', 'labels', 'label_mask', 'filename', 'chunk_id')


def multisignal_collate(batch):
    for item in batch:
        missing = [k for k in _REQUIRED if k not in item]
        if missing:
            raise KeyError(f"multisignal batch item missing keys: {missing}")
    return default_collate(batch)
