# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent TTNN implementation of the Audio Spectrogram Transformer (AST).

Bounty: tenstorrent/tt-metal#52054
Model: MIT/ast-finetuned-audioset-10-10-0.4593
"""

from .ttnn_ast import (
    TtASTForAudioClassification,
    TtASTModel,
    TtASTEmbeddings,
    TtASTEncoderLayer,
    custom_preprocessor,
)

__all__ = [
    "TtASTForAudioClassification",
    "TtASTModel",
    "TtASTEmbeddings",
    "TtASTEncoderLayer",
    "custom_preprocessor",
]
