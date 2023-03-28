"""BNF Quantizer related modules."""

from distutils.command.config import config
import imp
import logging
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from src.vector_quantize import VectorQuantize


class Quantizer(torch.nn.Module):
    """BNF Quantizer module.

    """

    def __init__(self, args):
        # initialize base classes
        torch.nn.Module.__init__(self)

        self.vq = VectorQuantize(
                    dim = args.codebook_embed_size,
                    codebook_size =  args.codebook_size,
                    kmeans_iters = args.kmeans_iters,
                    use_cosine_sim = args.use_cosine_sim   # set this to True
                )
        self.commitment_cost = args.commitment_cost

    def forward(self, xs, ilens, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded acoustic features (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Loss value.

        """
        # remove unnecessary padded part (for multi-gpus)
        max_ilen = max(ilens)
        if max_ilen != xs.shape[1]:
            xs = xs[:, :max_ilen]


        ##### Vector quantize BNFs ######
        xs_quantized, indices, commit_loss = self.vq(xs)
        #print(f'xs : {xs_quantized.shape} \n prosody_vec : {prosody_vec.shape} \nIndices: {indices.shape}')

        loss = self.commitment_cost*commit_loss

        return loss, commit_loss

    def inference(self, x, *args, **kwargs):
        """Generate the sequence of features given the sequences of acoustic features.

        Args:
            x (Tensor): Input sequence of acoustic features (T, idim).
            inference_args (Namespace):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).

        """
        

        # forward encoder
        x = x.unsqueeze(0)
        x_quantized, indices, _ = self.vq(x)

        return x_quantized.squeeze(0), indices.squeeze(0)