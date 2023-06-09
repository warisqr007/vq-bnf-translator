import logging

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class Tacotron2Loss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False, bce_pos_weight=20.0, num_labels=128
    ):
        """Initialize Tactoron2 loss module.

        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
            bce_pos_weight (float): Weight of positive sample of stop token.

        """
        super(Tacotron2Loss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=torch.tensor(bce_pos_weight)
        )
        self.ctc_criterion = torch.nn.CTCLoss(blank=num_labels, reduction='none')

        # NOTE(kan-bayashi): register pre hook function for the compatibility
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    # after_outs, before_outs, logits, logits_labels, ys, ys_labels, stop_labels, olens
    def forward(self, after_outs, before_outs, logits, logits_labels, \
                ys, ys_labels, stop_labels, ilens, olens):
            #self, after_outs, before_outs, logits, ys, labels, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        """

        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(masks)
            after_outs = after_outs.masked_select(masks)
            before_outs = before_outs.masked_select(masks)
            stop_labels = stop_labels.masked_select(masks[:, :, 0])
            logits = logits.masked_select(masks[:, :, 0])
            # logits_labels = logits_labels.masked_select(masks)
            ys_labels = ys_labels.masked_select(masks[:, :, 0])

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, ys) + self.l1_criterion(before_outs, ys)
        mse_loss = self.mse_criterion(after_outs, ys) + self.mse_criterion(
            before_outs, ys
        )
        bce_loss = self.bce_criterion(logits, stop_labels)


        # print(logits_labels.shape)
        # (N, T, C) -> (T, N, C), required by torch.nn.CTCLoss()
        log_probs = torch.nn.functional.log_softmax(logits_labels.transpose(0,1), dim=2)
        ctc_loss = self.ctc_criterion(log_probs, ys_labels, olens, olens).mean()

        # make weighted mask and apply it
        if self.use_weighted_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(ys.size(0) * ys.size(2))
            logit_weights = weights.div(ys.size(0))

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()
            bce_loss = (
                bce_loss.mul(logit_weights.squeeze(-1))
                .masked_select(masks.squeeze(-1))
                .sum()
            )
            ctc_loss = (
                ctc_loss.mul(logit_weights.squeeze(-1))
                .masked_select(masks.squeeze(-1))
                .sum()
            )

        return l1_loss, mse_loss, bce_loss, ctc_loss

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Apply pre hook function before loading state dict.

        From v.0.6.1 `bce_criterion.pos_weight` param is registered as a parameter but
        old models do not include it and as a result, it causes missing key error when
        loading old model parameter. This function solve the issue by adding param in
        state dict before loading as a pre hook function
        of the `load_state_dict` method.

        """
        key = prefix + "bce_criterion.pos_weight"
        if key not in state_dict:
            state_dict[key] = self.bce_criterion.pos_weight