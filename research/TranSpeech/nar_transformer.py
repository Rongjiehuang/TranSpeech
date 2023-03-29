# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch import Tensor
import ipdb
from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoderDecoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.iterative_refinement_generator import DecoderOut
import torch.nn.functional as F
from fairseq.models.speech_to_text import S2TTransformerEncoder
from fairseq.models.speech_to_speech.modules.ctc_decoder import CTCDecoder
from fairseq.models.speech_to_speech.modules.stacked_embedding import StackedEmbedding
from fairseq.models.nat.nonautoregressive_transformer import _mean_pooling
from fairseq.models.transformer import (
    Linear,
    TransformerDecoder,
    TransformerModelBase,
    Embedding,
)
from fairseq.models.nat.cmlm_transformer import _skeptical_unmasking
from fairseq.models.nat import NATransformerModel, FairseqNATDecoder, ensemble_decoder

logger = logging.getLogger(__name__)


class S2STransformerEncoder(S2TTransformerEncoder):
    """Based on S2T transformer encoder, with support
    to incorporate target speaker embedding."""

    def __init__(self, args):
        super().__init__(args)

        self.spk_emb_proj = None
        if args.target_speaker_embed:
            self.spk_emb_proj = Linear(
                args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim
            )
        self.ensemble_models = None

    def forward(
        self, src_tokens, src_lengths, tgt_speaker=None, return_all_hiddens=False
    ):
        '''
        src_tokens: B, L, 80
        src_lengths: B
        tgt_speaker: None
        return_all_hiddens: True
        '''
        out = super().forward(src_tokens, src_lengths, return_all_hiddens)

        if self.spk_emb_proj:
            x = out["encoder_out"][0]
            seq_len, bsz, _ = x.size()
            tgt_speaker_emb = tgt_speaker.view(1, bsz, -1).expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, tgt_speaker_emb], dim=2))
            out["encoder_out"][0] = x

        return out


class TransformerUnitDecoder(TransformerDecoder):
    """Based on Transformer decoder, with support to decoding stacked units"""

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn, output_projection
        )
        self.n_frames_per_step = args.n_frames_per_step
        self.out_proj_n_frames = (
            Linear(
                self.output_embed_dim,
                self.output_embed_dim * self.n_frames_per_step,
                bias=False,
            )
            if self.n_frames_per_step > 1
            else None
        )
        self.ensemble_models = None
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, args.encoder_embed_dim, None)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        normalize = False,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # x: B, tgt_len ,256
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            bsz, seq_len, d = x.size()
            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)
            x = self.output_layer(x.view(bsz, seq_len, self.n_frames_per_step, d)) # [B, T, 512] -> [B, T, dictionary_size]
            x = x.view(bsz, seq_len * self.n_frames_per_step, -1)
            if (
                incremental_state is None and self.n_frames_per_step > 1
            ):  # teacher-forcing mode in training
                x = x[
                    :, : -(self.n_frames_per_step - 1), :
                ]  # remove extra frames after <eos>

        x = F.log_softmax(x, -1) if normalize else x
        return x, extra

    def upgrade_state_dict_named(self, state_dict, name):
        if self.n_frames_per_step > 1:
            move_keys = [
                (
                    f"{name}.project_in_dim.weight",
                    f"{name}.embed_tokens.project_in_dim.weight",
                )
            ]
            for from_k, to_k in move_keys:
                if from_k in state_dict and to_k not in state_dict:
                    state_dict[to_k] = state_dict[from_k]
                    del state_dict[from_k]

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks) # B x 256
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset: # false
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt




class S2STransformerMultitaskModelBase(NATransformerModel): # NATransformerModel / FairseqEncoderDecoderModel
    @classmethod
    def build_encoder(cls, args, src_dict=None, embed_tokens=None):  # Here we don't use src_dict and embed_tokens
        encoder = S2STransformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_multitask_decoder(cls, args, tgt_dict, in_dim):
        decoder_args = args.decoder_args
        decoder_args.encoder_embed_dim = in_dim
        if args.decoder_type == "transformer":
            base_multitask_text_transformer_decoder_arch(decoder_args)
            task_decoder = TransformerDecoder(
                decoder_args,
                tgt_dict,
                embed_tokens=TransformerModelBase.build_embedding(
                    decoder_args,
                    tgt_dict,
                    decoder_args.decoder_embed_dim,
                ),
            )
        elif args.decoder_type == "ctc":
            task_decoder = CTCDecoder(
                dictionary=tgt_dict,
                in_dim=in_dim,
            )
        else:
            raise NotImplementedError(
                "currently only support multitask decoder_type 'transformer', 'ctc'"
            )

        return task_decoder

    @classmethod
    def build_model(cls, args, task):
        encoder = cls.build_encoder(args)
        decoder = (
            cls.build_decoder(args, task.target_dictionary)
            if task.args.target_is_code
            else cls.build_decoder(args)
        )
        base_model = cls(args, encoder, decoder)

        # set up multitask decoders
        base_model.multitask_decoders = {}
        for task_name, task_obj in task.multitask_tasks.items():
            in_dim = (
                args.encoder_embed_dim
                if task_obj.args.input_from == "encoder"
                else args.decoder_embed_dim
            )
            task_decoder = cls.build_multitask_decoder(
                task_obj.args, task_obj.target_dictionary, in_dim
            )

            setattr(base_model, f"{task_name}_decoder", task_decoder)
            decoder_model_cls = (
                FairseqEncoderModel
                if task_obj.args.decoder_type == "ctc"
                else FairseqLanguageModel
            )
            base_model.multitask_decoders[task_name] = decoder_model_cls(
                getattr(base_model, f"{task_name}_decoder")
            )

        return base_model

    # def forward_encoder(self, src_tokens, src_lengths, speaker=None, **kwargs):
    #     return self.encoder(
    #         src_tokens, src_lengths=src_lengths, tgt_speaker=speaker, **kwargs
    #     )

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

@register_model("nar_transformer") # S2STransformerEncoder + TransformerUnitDecoder
class NARS2UTTransformerModel(S2STransformerMultitaskModelBase):
    """
    Direct speech-to-speech translation model with S2T Transformer encoder + Transformer discrete unit decoder
    https://arxiv.org/abs/2107.05604
    """

    @staticmethod
    def add_args(parser):
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )
        # speaker
        parser.add_argument(
            "--speaker-embed-dim",
            type=int,
            metavar="N",
            help="speaker embedding dimension",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, PlaceHolder=None): # embed_tokens
        num_embeddings = len(tgt_dict)
        padding_idx = tgt_dict.pad()
        embed_tokens = StackedEmbedding(
            num_embeddings,
            args.decoder_embed_dim,
            padding_idx,
            num_stacked=args.n_frames_per_step,
        )
        return TransformerUnitDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )

    def forward(
        self,
        src_tokens, # B, src_len_source, 80
        src_lengths, # B
        prev_output_tokens, # B, tgt_len
        tgt_tokens=None,
        tgt_speaker=None,
        return_all_hiddens=False,
    ):

        # assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            tgt_speaker=tgt_speaker,
            return_all_hiddens=return_all_hiddens, # True
        )
        '''
        encoder_out: {
        'encoder_out': (src_len, B, 256),
        'encoder_padding_mask': Null list,
        'encoder_embedding': Null list,
        'encoder_states': list, 12 X (src_len, B, 256),
        'src_tokens': Null list,
        'src_lengths': Null list}
        '''

        # length prediction
        length_out = self.decoder.forward_length(  # [B, 256]
            normalize=False, encoder_out=encoder_out
        )

        length_tgt = self.decoder.forward_length_prediction(   # [B]
            length_out, encoder_out, tgt_tokens
        )

        decoder_out = self.decoder(
            prev_output_tokens,  # prev_output_tokens -> tgt_tokens
            encoder_out=encoder_out,
            normalize=False,
        )

        word_ins_mask = prev_output_tokens.eq(self.unk)  # [B, T]

        '''
        decoder_out: {
        (B, L1, dictionary size),
        'attn': list, list, 1 X (B, tgt_len, src_len),
        'inner_states': list, 1 X (tgt_len, B, 256),
        '''
        if return_all_hiddens:
            # For CrossEntropy loss
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
            decoder_out[-1]["encoder_padding_mask"] = encoder_out["encoder_padding_mask"]
            decoder_out[-1]["word_ins_mask"] = word_ins_mask
            # For Length loss
            decoder_out[-1]["length_tgt"] = length_tgt
            decoder_out[-1]["length_out"] = length_out
            decoder_out[-1]["factor"] = self.decoder.length_loss_factor
        return decoder_out



    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs): # Only Use During Inference
        step = decoder_out.step
        max_step = decoder_out.max_step


        # output_tokens: [0 (start), 3 (mask), 3, 3, ..., 2 (stop), ..., 1 (pad)] TODO: a mismatch between output_tokens and _tokens
        output_tokens = decoder_out.output_tokens       # token/mask/score: [B, T]
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        out, extra = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens, #  [B, T]
            encoder_out=encoder_out,  # ['encoder_out', 'encoder_padding_mask', 'encoder_embedding', 'encoder_states', 'fc_results', 'src_tokens', 'src_lengths']
        )
        _scores, _tokens = out.max(-1)
        # tokens: [75, 67, ..., 24, 76, 75, ..., 75]

        # 加入mask
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(  # p%分数最低的 加入mask
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )
            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())


        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


    def initialize_output_tokens(self, encoder_out, src_lengths):
        '''
        Change src_tokens (from NATransformerModel) to  src_lengths for speech to speech translation
        '''


        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_lengths, max_length)

        initial_output_tokens = src_lengths.new_zeros(
            src_lengths.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        # initial_output_tokens[:, 0] = self.bos
        # initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        # initial_output_tokens[:, 0] = self.bos
        # initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )


def base_multitask_text_transformer_decoder_arch(args):
    args.dropout = getattr(args, "dropout", 0.3)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_layers = getattr(args, "decoder_layers", 2)

    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)

    # decoder layer
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)

    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)


def base_s2st_transformer_encoder_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)

    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.conv_version = getattr(args, "conv_version", "s2t_transformer")
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 256)


@register_model_architecture(
    model_name="nar_transformer", arch_name="nar_s2ut_transformer"
)
def s2ut_architecture_base(args):
    base_s2st_transformer_encoder_architecture(args)

    # decoder
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)

@register_model_architecture("nar_transformer", "nar_s2ut_transformer_fisher")
def s2ut_architecture_fisher(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    s2ut_architecture_base(args)
