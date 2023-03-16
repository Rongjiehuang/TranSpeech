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
from fairseq.models.transformer import (
    Linear,
    TransformerDecoder,
    TransformerModelBase,
    Embedding,
)
from fairseq.models.nat.cmlm_transformer import _skeptical_unmasking
from fairseq.models.nat import NATransformerModel, FairseqNATDecoder, ensemble_decoder
from fairseq.models.speech_to_text.s2t_conformer import S2TConformerEncoder
from research.TranSpeech.nar_transformer import (
    TransformerUnitDecoder,
    S2STransformerMultitaskModelBase,
    base_s2st_transformer_encoder_architecture,
    NARS2UTTransformerModel,
    s2ut_architecture_fisher,
    s2ut_architecture_base,
)


logger = logging.getLogger(__name__)


class S2SConformerEncoder(S2TConformerEncoder):
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


class S2SConformerMultitaskModelBase(S2STransformerMultitaskModelBase): # NATransformerModel / FairseqEncoderDecoderModel
    @classmethod
    def build_encoder(cls, args, src_dict=None, embed_tokens=None):  # Here we don't use src_dict and embed_tokens
        encoder = S2SConformerEncoder(args)
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


@register_model("nar_conformer")  # S2SConformerEncoder + ConformerUnitDecoder
class NARS2UTConformerModel(S2SConformerMultitaskModelBase, NARS2UTTransformerModel):
    """
    Direct speech-to-speech translation model with S2T Transformer encoder + Transformer discrete unit decoder
    https://arxiv.org/abs/2107.05604
    """
    @staticmethod
    def add_args(parser):
        NARS2UTTransformerModel.add_args(parser)
        # Conformer
        parser.add_argument("--input-feat-per-channel", default=80)
        parser.add_argument("--depthwise-conv-kernel-size", default=31)
        parser.add_argument("--input-channels", default=1)
        parser.add_argument(
            "--attn-type",
            default=None,
            help="If not specified uses fairseq MHA. Other valid option is espnet",
        )
        parser.add_argument(
            "--pos-enc-type",
            default="abs",
            help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
        )


@register_model_architecture("nar_conformer", "nar_s2ut_conformer")
def s2ut_comformer_architecture_fisher(args):
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    s2ut_architecture_base(args)


@register_model_architecture("nar_conformer", "nar_s2ut_conformer_fisher")
def s2ut_comformer_architecture_fisher(args):
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    s2ut_architecture_base(args)
