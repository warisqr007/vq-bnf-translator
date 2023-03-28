# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Voice Transformer Network (Transformer-VC) related modules."""

from distutils.command.config import config
import imp
import logging
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_transformer import subsequent_mask
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import (
    Tacotron2Loss as TransformerLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.e2e_tts_transformer import (
    GuidedMultiHeadAttentionLoss,  # noqa: H301
    TTSPlot,  # noqa: H301
)
from vector_quantize_pytorch import VectorQuantize
# from src.prosody_modules import ECAPA_TDNN
# from src.prosody_modules import ProsodyEncoder
from synthesizer.src.ecapa_tdnn import ECAPA_TDNN

class Transformer(TTSInterface, torch.nn.Module):
    """VC Transformer module.

    This is a module of the Voice Transformer Network
    (a.k.a. VTN or Transformer-VC) described in
    `Voice Transformer Network: Sequence-to-Sequence
    Voice Conversion Using Transformer with
    Text-to-Speech Pretraining`_,
    which convert the sequence of acoustic features
    into the sequence of acoustic features.

    .. _`Voice Transformer Network: Sequence-to-Sequence
        Voice Conversion Using Transformer with
        Text-to-Speech Pretraining`:
        https://arxiv.org/pdf/1912.06813.pdf

    """

    @property
    def attention_plot_class(self):
        """Return plot class for attention weight plot."""
        return TTSPlot

    def __init__(self,args):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # fill missing arguments
        # args = fill_missing_args(args, self.add_arguments)

        # store hyperparameters
        self.idim = args.idim
        self.odim = args.odim
        self.whereusespkd = args.whereusespkd
        self.spk_embed_dim = args.spk_embed_dim
        self.prosody_emb_dim = args.prosody_emb_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = args.spk_embed_integration_type
        self.use_scaled_pos_enc = args.use_scaled_pos_enc
        self.reduction_factor = args.reduction_factor
        self.encoder_reduction_factor = args.encoder_reduction_factor
        self.transformer_input_layer = args.transformer_input_layer
        self.loss_type = args.loss_type
        self.use_guided_attn_loss = args.use_guided_attn_loss
        if self.use_guided_attn_loss:
            if args.num_layers_applied_guided_attn == -1:
                self.num_layers_applied_guided_attn = args.elayers
            else:
                self.num_layers_applied_guided_attn = (
                    args.num_layers_applied_guided_attn
                )
            if args.num_heads_applied_guided_attn == -1:
                self.num_heads_applied_guided_attn = args.aheads
            else:
                self.num_heads_applied_guided_attn = args.num_heads_applied_guided_attn
            self.modules_applied_guided_attn = args.modules_applied_guided_attn

        # use idx 0 as padding idx
        padding_idx = 0

        # define projection layer
        # if self.spk_embed_dim is not None:
        #     if self.whereusespkd == 'atinput':
        #         if self.spk_embed_integration_type == "add":
        #             self.projection = torch.nn.Linear(self.spk_embed_dim, args.idim)
        #         else:
        #             self.projection = torch.nn.Linear(
        #                 args.idim + self.spk_embed_dim, args.idim
        #             )
        #     else:
        #         if self.spk_embed_integration_type == "add":
        #             self.projection = torch.nn.Linear(self.spk_embed_dim, args.adim)
        #         else:
        #             self.projection = torch.nn.Linear(
        #                 args.adim + self.spk_embed_dim, args.adim
        #             )
        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define transformer encoder
        if args.eprenet_conv_layers != 0:
            # encoder prenet
            encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=args.idim,
                    elayers=0,
                    econv_layers=args.eprenet_conv_layers,
                    econv_chans=args.eprenet_conv_chans,
                    econv_filts=args.eprenet_conv_filts,
                    use_batch_norm=args.use_batch_norm,
                    dropout_rate=args.eprenet_dropout_rate,
                    padding_idx=padding_idx,
                    input_layer=torch.nn.Linear(
                        args.idim * args.encoder_reduction_factor, args.idim
                    ),
                ),
                torch.nn.Linear(args.eprenet_conv_chans, args.adim),
            )
        elif args.transformer_input_layer == "linear":
            encoder_input_layer = torch.nn.Linear(
                args.idim * args.encoder_reduction_factor, args.adim
            )
        else:
            encoder_input_layer = args.transformer_input_layer
        self.encoder = Encoder(
            idim=args.idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=args.transformer_enc_dropout_rate,
            positional_dropout_rate=args.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=args.transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.encoder_normalize_before,
            concat_after=args.encoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size,
        )

        # self.encoder = ConformerEncoder(
        #     idim = args.idim,
        #     attention_dim=args.adim,
        #     attention_heads=args.aheads,
        #     linear_units=2048,
        #     num_blocks=2,
        #     dropout_rate=0.1,
        #     positional_dropout_rate=0.1,
        #     attention_dropout_rate=0.0,
        #     input_layer="conv2d",
        #     normalize_before=True,
        #     concat_after=False,
        #     positionwise_layer_type="linear",
        #     positionwise_conv_kernel_size=1,
        #     macaron_style=False,
        #     pos_enc_layer_type="abs_pos",
        #     selfattention_layer_type="selfattn",
        #     activation_type="swish",
        #     use_cnn_module=True,
        #     zero_triu=False,
        #     cnn_module_kernel=15,
        #     padding_idx=-1,
        #     stochastic_depth_rate=0.0,
        #     intermediate_layers=None,
        #     ctc_softmax=None,
        #     conditioning_layer_dim=None,
        # )

        # self.vq = VectorQuantize(
        #             dim = args.codebook_embed_size,
        #             codebook_size =  args.codebook_size,
        #             use_cosine_sim = True   # set this to True
        #         )
        # self.commitment_cost = args.commitment_cost

        # self.prosody_encoder = ECAPA_TDNN()

        # self.prosody_spk_projection = torch.nn.Linear(
        #     args.adim + self.spk_embed_dim + self.prosody_emb_dim, args.adim
        # )

        self.prosody_encoder = ECAPA_TDNN() #ProsodyEncoder(80, self.prosody_emb_dim) #ECAPA_TDNN()

        self.spk_projection = torch.nn.Linear(
            args.adim + self.spk_embed_dim, args.adim
        )

        self.prosody_projection = torch.nn.Linear(
            args.adim + self.prosody_emb_dim, args.adim
        )

        # define transformer decoder
        if args.dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(
                    idim=args.odim,
                    n_layers=args.dprenet_layers,
                    n_units=args.dprenet_units,
                    dropout_rate=args.dprenet_dropout_rate,
                ),
                torch.nn.Linear(args.dprenet_units, args.adim),
            )
        else:
            decoder_input_layer = "linear"
        self.decoder = Decoder(
            odim=-1,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.transformer_dec_dropout_rate,
            positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
            self_attention_dropout_rate=args.transformer_dec_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_enc_dec_attn_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=pos_enc_class,
            normalize_before=args.decoder_normalize_before,
            concat_after=args.decoder_concat_after,
        )

        # define final projection
        self.feat_out = torch.nn.Linear(args.adim, args.odim * args.reduction_factor)
        self.prob_out = torch.nn.Linear(args.adim, args.reduction_factor)

        # define postnet
        self.postnet = (
            None
            if args.postnet_layers == 0
            else Postnet(
                idim=args.idim,
                odim=args.odim,
                n_layers=args.postnet_layers,
                n_chans=args.postnet_chans,
                n_filts=args.postnet_filts,
                use_batch_norm=args.use_batch_norm,
                dropout_rate=args.postnet_dropout_rate,
            )
        )

        # define loss function
        self.criterion = TransformerLoss(
            use_masking=args.use_masking,
            use_weighted_masking=args.use_weighted_masking,
            bce_pos_weight=args.bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(
                sigma=args.guided_attn_loss_sigma,
                alpha=args.guided_attn_loss_lambda,
            )

        # initialize parameters
        self._reset_parameters(
            init_type=args.transformer_init,
            init_enc_alpha=args.initial_encoder_alpha,
            init_dec_alpha=args.initial_decoder_alpha,
        )

        # load pretrained model
        # if args.pretrained_model is not None:
        #     self.load_pretrained_model(args.pretrained_model)

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def _add_first_frame_and_remove_last_frame(self, ys):
        ys_in = torch.cat(
            [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
        )
        return ys_in

    def forward(self, xs, ilens, ys, olens, spembs=None, prosody_vec=None, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded acoustic features (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors
                (B, spk_embed_dim).

        Returns:
            Tensor: Loss value.

        """
        #print(f'xs : {xs.shape} \n ilens : {ilens.shape} \n ys : {ys.shape} \n olens : {olens.shape} \n spembs :{spembs.shape} prosody_vec: {prosody_vec.shape}')

        # remove unnecessary padded part (for multi-gpus)
        max_ilen = max(ilens)
        max_olen = max(olens)
        if max_ilen != xs.shape[1]:
            xs = xs[:, :max_ilen]
        if max_olen != ys.shape[1]:
            ys = ys[:, :max_olen]
            # labels = labels[:, :max_olen]

        
        # if self.whereusespkd == 'atinput':
        #     xs = self._integrate_with_spk_in_embed(xs,spembs)
        # thin out input frames for reduction factor
        # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)
        if self.encoder_reduction_factor > 1:
            B, Lmax, idim = xs.shape
            if Lmax % self.encoder_reduction_factor != 0:
                xs = xs[:, : -(Lmax % self.encoder_reduction_factor), :]
            xs_ds = xs.contiguous().view(
                B,
                int(Lmax / self.encoder_reduction_factor),
                idim * self.encoder_reduction_factor,
            )
            ilens_ds = ilens.new(
                [ilen // self.encoder_reduction_factor for ilen in ilens]
            )
        else:
            xs_ds, ilens_ds = xs, ilens
        
        


        x_masks = self._source_mask(ilens)
        hs, hs_masks = self.encoder(xs, x_masks)


        prosody_embs = self.prosody_encoder(prosody_vec)
        prosody_embs = F.normalize(prosody_embs).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.prosody_projection(torch.cat([hs, prosody_embs], dim=-1))


        spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.spk_projection(torch.cat([hs, spembs], dim=-1))

        

        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.reduction_factor > 1:
            ys_in = ys[:, self.reduction_factor - 1 :: self.reduction_factor]
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # if conv2d, modify mask. Use ceiling division here
        if "conv2d" in self.transformer_input_layer:
            ilens_ds_st = ilens_ds.new(
                [((ilen - 2 + 1) // 2 - 2 + 1) // 2 for ilen in ilens_ds]
            )
        else:
            ilens_ds_st = ilens_ds

        # forward decoder
        y_masks = self._target_mask(olens_in)
        zs, _ = self.decoder(ys_in, y_masks, hs, hs_masks)
        # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, Lmax//r, r) -> (B, Lmax//r * r)
        # logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            assert olens.ge(
                self.reduction_factor
            ).all(), "Output length must be greater than or equal to reduction factor."
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]
            # labels = labels[:, :max_olen]
            # labels = torch.scatter(
            #     labels, 1, (olens - 1).unsqueeze(1), 1.0
            # )  # see #3388

        # calculate loss values
        # return after_outs, before_outs, logits, ys, labels, olens
        l1_loss, l2_loss, bce_loss = self.criterion(
            after_outs, before_outs, ys, olens
        )
        # loss = l1_loss + l2_loss 
        # loss = self.commitment_cost*commit_loss + l2_loss
        loss = l2_loss 
        # if self.loss_type == "L1":
        #     loss = l1_loss 
        # elif self.loss_type == "L2":
        #     loss = l2_loss 
        # elif self.loss_type == "L1+L2":
        #     loss = l1_loss + l2_loss 
        # else:
        #     raise ValueError("unknown --loss-type " + self.loss_type)
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"l2_loss": l2_loss.item()},
            {"bce_loss": 0},
            {"loss": loss.item()},
        ]

        # calculate guided attention loss
        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                    reversed(range(len(self.encoder.encoders)))
                ):
                    att_ws += [
                        self.encoder.encoders[layer_idx].self_attn.attn[
                            :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_in, T_in)
                enc_attn_loss = self.attn_criterion(
                    att_ws, ilens_ds_st, ilens_ds_st
                )  # TODO(unilight): is changing to ilens_ds_st right?
                loss = loss + enc_attn_loss
                report_keys += [{"enc_attn_loss": enc_attn_loss.item()}]
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                    reversed(range(len(self.decoder.decoders)))
                ):
                    att_ws += [
                        self.decoder.decoders[layer_idx].self_attn.attn[
                            :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_out)
                dec_attn_loss = self.attn_criterion(att_ws, olens_in, olens_in)
                loss = loss + dec_attn_loss
                report_keys += [{"dec_attn_loss": dec_attn_loss.item()}]
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                    reversed(range(len(self.decoder.decoders)))
                ):
                    att_ws += [
                        self.decoder.decoders[layer_idx].src_attn.attn[
                            :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_in)
                enc_dec_attn_loss = self.attn_criterion(
                    att_ws, ilens_ds_st, olens_in
                )  # TODO(unilight): is changing to ilens_ds_st right?
                loss = loss + enc_dec_attn_loss
                report_keys += [{"enc_dec_attn_loss": enc_dec_attn_loss.item()}]

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        # self.reporter.report(report_keys)

        return loss, None, after_outs, before_outs, ys, olens

    def inference(self, x, spemb=None, prosody_vec=None, *args, **kwargs):
        """Generate the sequence of features given the sequences of acoustic features.

        Args:
            x (Tensor): Input sequence of acoustic features (T, idim).
            inference_args (Namespace):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).

        """
        # get options
        # threshold = inference_args.threshold
        # minlenratio = inference_args.minlenratio
        # maxlenratio = inference_args.maxlenratio
        # use_att_constraint = getattr(
        #     inference_args, "use_att_constraint", False
        # )  # keep compatibility
        # if use_att_constraint:
        #     logging.warning(
        #         "Attention constraint is not yet supported in Transformer. Not enabled."
        #     )

        # thin out input frames for reduction factor
        # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)
        olens = prosody_vec.shape[0] // self.reduction_factor
        if self.encoder_reduction_factor > 1:
            Lmax, idim = x.shape
            if Lmax % self.encoder_reduction_factor != 0:
                x = x[: -(Lmax % self.encoder_reduction_factor), :]
            x_ds = x.contiguous().view(
                int(Lmax / self.encoder_reduction_factor),
                idim * self.encoder_reduction_factor,
            )
        else:
            x_ds = x

        # forward encoder
        x_ds = x_ds.unsqueeze(0)
        hs, _ = self.encoder(x_ds, None)



        prosody_vec = prosody_vec.unsqueeze(0)
        prosody_embs = self.prosody_encoder(prosody_vec)
        prosody_embs = F.normalize(prosody_embs).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.prosody_projection(torch.cat([hs, prosody_embs], dim=-1))

        
        spembs = spemb.unsqueeze(0)
        spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.spk_projection(torch.cat([hs, spembs], dim=-1))

        # set limits of length
        # maxlen = int(hs.size(1) * maxlenratio / self.reduction_factor)
        # minlen = int(hs.size(1) * minlenratio / self.reduction_factor)

        # initialize
        idx = 0
        ys = hs.new_zeros(1, 1, self.odim)
        outs, probs = [], []

        # forward decoder step-by-step
        z_cache = self.decoder.init_state(x)
        while True:
            # update index
            idx += 1

            # calculate output and stop prob at idx-th step
            y_masks = subsequent_mask(idx).unsqueeze(0).to(x.device)
            z, z_cache = self.decoder.forward_one_step(
                ys, y_masks, hs, cache=z_cache
            )  # (B, adim)
            outs += [
                self.feat_out(z).view(self.reduction_factor, self.odim)
            ]  # [(r, odim), ...]
            # probs += [torch.sigmoid(self.prob_out(z))[0]]  # [(r), ...]

            # update next inputs
            ys = torch.cat(
                (ys, outs[-1][-1].view(1, 1, self.odim)), dim=1
            )  # (1, idx + 1, odim)

            # get attention weights
            att_ws_ = []
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) and "src" in name:
                    att_ws_ += [m.attn[0, :, -1].unsqueeze(1)]  # [(#heads, 1, T),...]
            if idx == 1:
                att_ws = att_ws_
            else:
                # [(#heads, l, T), ...]
                att_ws = [
                    torch.cat([att_w, att_w_], dim=1)
                    for att_w, att_w_ in zip(att_ws, att_ws_)
                ]

            # check whether to finish generation
            if idx > olens-1:
                outs = (
                    torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2)
                )  # (L, odim) -> (1, L, odim) -> (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                # probs = torch.cat(probs, dim=0)
                break

        # concatenate attention weights -> (#layers, #heads, L, T)
        att_ws = torch.stack(att_ws, dim=0)

        return outs, att_ws
    
    def get_prosody_embed(self, prosody_vec, *args, **kwargs):
        prosody_vec = prosody_vec.unsqueeze(0)
        prosody_embs = self.prosody_encoder(prosody_vec)

        return prosody_embs.squeeze(0)

    def calculate_all_attentions(
        self,
        xs,
        ilens,
        ys,
        olens,
        spembs=None,
        prosody_vec = None,
        skip_output=False,
        keep_tensor=False,
        *args,
        **kwargs
    ):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded acoustic features (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors
                (B, spk_embed_dim).
            skip_output (bool, optional): Whether to skip calculate the final output.
            keep_tensor (bool, optional): Whether to keep original tensor.

        Returns:
            dict: Dict of attention weights and outputs.

        """
        with torch.no_grad():
            # thin out input frames for reduction factor
            # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)
            if self.encoder_reduction_factor > 1:
                B, Lmax, idim = xs.shape
                if Lmax % self.encoder_reduction_factor != 0:
                    xs = xs[:, : -(Lmax % self.encoder_reduction_factor), :]
                xs_ds = xs.contiguous().view(
                    B,
                    int(Lmax / self.encoder_reduction_factor),
                    idim * self.encoder_reduction_factor,
                )
                ilens_ds = ilens.new(
                    [ilen // self.encoder_reduction_factor for ilen in ilens]
                )
            else:
                xs_ds, ilens_ds = xs, ilens

            # forward encoder
            x_masks = self._source_mask(ilens_ds)
            hs, hs_masks = self.encoder(xs_ds, x_masks)

            # integrate speaker embedding
            if self.spk_embed_dim is not None:
                hs = self._integrate_with_spk_embed(hs, spembs)

            
            # Add prosody information
            hs = self._integrate_with_prosody_embed(hs, hs_masks, prosody_vec)
            # thin out frames for reduction factor
            # (B, Lmax, odim) ->  (B, Lmax//r, odim)
            if self.reduction_factor > 1:
                ys_in = ys[:, self.reduction_factor - 1 :: self.reduction_factor]
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                ys_in, olens_in = ys, olens

            # add first zero frame and remove last frame for auto-regressive
            ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

            # forward decoder
            y_masks = self._target_mask(olens_in)
            zs, _ = self.decoder(ys_in, y_masks, hs, hs_masks)

            # calculate final outputs
            if not skip_output:
                before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
                if self.postnet is None:
                    after_outs = before_outs
                else:
                    after_outs = before_outs + self.postnet(
                        before_outs.transpose(1, 2)
                    ).transpose(1, 2)

        # modifiy mod part of output lengths due to reduction factor > 1
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])

        # store into dict
        att_ws_dict = dict()
        if keep_tensor:
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention):
                    att_ws_dict[name] = m.attn
            if not skip_output:
                att_ws_dict["before_postnet_fbank"] = before_outs
                att_ws_dict["after_postnet_fbank"] = after_outs
        else:
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention):
                    attn = m.attn.cpu().numpy()
                    if "encoder" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                    elif "decoder" in name:
                        if "src" in name:
                            attn = [
                                a[:, :ol, :il]
                                for a, il, ol in zip(
                                    attn, ilens.tolist(), olens_in.tolist()
                                )
                            ]
                        elif "self" in name:
                            attn = [
                                a[:, :l, :l] for a, l in zip(attn, olens_in.tolist())
                            ]
                        else:
                            logging.warning("unknown attention module: " + name)
                    else:
                        logging.warning("unknown attention module: " + name)
                    att_ws_dict[name] = attn
            if not skip_output:
                before_outs = before_outs.cpu().numpy()
                after_outs = after_outs.cpu().numpy()
                att_ws_dict["before_postnet_fbank"] = [
                    m[:l].T for m, l in zip(before_outs, olens.tolist())
                ]
                att_ws_dict["after_postnet_fbank"] = [
                    m[:l].T for m, l in zip(after_outs, olens.tolist())
                ]

        return att_ws_dict

    # def _integrate_with_spk_embed(self, hs, spembs):
    #     """Integrate speaker embedding with hidden states.

    #     Args:
    #         hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
    #         spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

    #     Returns:
    #         Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)

    #     """
    #     if self.spk_embed_integration_type == "add":
    #         # apply projection and then add to hidden states
    #         spembs = self.projection(F.normalize(spembs))
    #         hs = hs + spembs.unsqueeze(1)
    #     elif self.spk_embed_integration_type == "concat":
    #         # concat hidden states with spk embeds and then apply projection
    #         spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
    #         hs = self.projection(torch.cat([hs, spembs], dim=-1))
    #     else:
    #         raise NotImplementedError("support only add or concat.")

    #     return hs

    # def _integrate_with_spk_in_embed(self, xs, spembs):
    #     """Integrate speaker embedding with hidden states.

    #     Args:
    #         hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
    #         spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

    #     Returns:
    #         Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)

    #     """
    #     if self.spk_embed_integration_type == "add":
    #         # apply projection and then add to hidden states
    #         spembs = self.projection(F.normalize(spembs))
    #         xs = xs + spembs.unsqueeze(1)
    #     elif self.spk_embed_integration_type == "concat":
    #         # concat hidden states with spk embeds and then apply projection
    #         spembs = F.normalize(spembs).unsqueeze(1).expand(-1, xs.size(1), -1)
    #         xs = self.projection(torch.cat([xs, spembs], dim=-1))
            
    #     else:
    #         raise NotImplementedError("support only add or concat.")

    #     return xs

    # def _integrate_with_prosody_embed(self, hs, hs_masks, prosody_vec):
    #     #spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
    #     prosody_vec, _ = self.prosody_encoder(prosody_vec, None)
    #     prosody_vec = self.prosody_bottleneck(prosody_vec.transpose(1, 2)).transpose(1, 2)

    #     #config III:
    #     # prosody_vec_att = self.prosody_attention(prosody_vec, hs, hs, None)
    #     # hs =  prosody_vec_att + prosody_vec

    #     #config IV:
    #     prosody_vec_att = self.prosody_attention(prosody_vec, hs, hs, None)
    #     #print(f'hs : {hs.shape} \n prosody_vec : {prosody_vec.shape} \n prosody att : {prosody_vec_att.shape}')

    #     #hs =  prosody_vec_att + hs #
    #     hs = self.prosody_projection(torch.cat([prosody_vec, prosody_vec_att], dim=-1))

    #     return hs

    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                    [[1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _target_mask(self, olens):
        """Make masks for masked self-attention.

        Args:
            olens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> olens = [5, 3]
            >>> self._target_mask(olens)
            tensor([[[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]],
                    [[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training.

        keys should match what `chainer.reporter` reports.
        If you add the key `loss`, the reporter will report `main/loss`
            and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss`
            and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ["loss", "l1_loss", "l2_loss", "bce_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]
        if self.use_guided_attn_loss:
            if "encoder" in self.modules_applied_guided_attn:
                plot_keys += ["enc_attn_loss"]
            if "decoder" in self.modules_applied_guided_attn:
                plot_keys += ["dec_attn_loss"]
            if "encoder-decoder" in self.modules_applied_guided_attn:
                plot_keys += ["enc_dec_attn_loss"]

        return plot_keys


