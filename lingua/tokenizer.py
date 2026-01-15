# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
import logging
import os
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tiktoken
from sentencepiece import SentencePieceProcessor
from tiktoken.load import load_tiktoken_bpe

logger = logging.getLogger(__name__)


@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: Optional[str] = None
    tokenizers: Optional[List[Dict[str, Any]]] = None
    load_supermapping: Optional[bool] = False
    dropout: float = 0.0
    seed: Optional[int] = 42


class Tokenizer(abc.ABC):
    mapping: Dict[str, Any] = {}

    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens,skip_special_tokens:bool=None):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass

    def load_supermapping(self, base_path: str, path: str) -> None:
        import json

        mapping_path = Path(base_path, f"{path.replace('/', '--')}_super_mapping.json")
        if not mapping_path.exists():
            import huggingface_hub as hf_hub

            try:
                assert os.environ.get("HF_HUB_OFFLINE") != "1"
                repo_id = f"gsaltintas/supertokenizer-{path.replace('/', '-')}"
                mapping_path = hf_hub.hf_hub_download(
                    repo_id, f"{path.replace('/', '--')}_super_mapping.json"
                )
                logger.info(f"Downloaded super mapping from HF Hub {repo_id} to {mapping_path}")
            except:
                mapping_path = Path(base_path, f"{path.replace('/', '--')}_super_mapping.json")
                logger.warning(f"Failed to download super mapping from HF Hub {repo_id}. Trying local path {mapping_path}")
        assert os.path.isfile(mapping_path), mapping_path
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        self.mapping = {int(k): v for k, v in mapping.items()}
        logger.info(f"Loaded super mapping from {mapping_path}")

    def encode_to_supermapping(self, tokens: List[str], add_bos: bool, add_eos: bool) -> List[int]:
        ids = []
        token_ids = self.encode(tokens, add_bos=add_bos, add_eos=add_eos)
        if len(self.mapping) == 0:
            return token_ids
        return [self.mapping[tid] for tid in token_ids if tid in self.mapping ]

class MockTokenizer(Tokenizer):
    n_words: int = 256

    def encode(self, tokens, add_bos, add_eos):
        return tokens


class ByteTokenizer(Tokenizer):
    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + list(s.encode()) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int],skip_special_tokens:bool=None):
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode("utf-8", errors="backslashreplace")

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is None:
            tokens = self.encode(text)

        decoded_chars, offsets = [], []
        byte_pos = 0
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode("utf-8", errors="ignore")
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode("utf-8"))

        return decoded_chars, offsets


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int],skip_special_tokens:bool=None):
        return self.sp_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets


DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|fim_prefix|>": 2,
    "<|fim_middle|>": 3,
    "<|fim_end_fill|>": 253,
    "<|fim_pad|>": 254,
    "<|fim_suffix|>": 255,
}
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
DEFAULT_SPECIAL_TOKENS = {
    "bos": ["<|begin_of_text|>", "<s>", "<bos>"],
    "eos": ["<|end_of_text|>", "</s>", "<eos>"],
    "pad": ["<pad>"],
}
ALIGNED_BOS = "~SPECIAL~ALIGNED~BOS~SYMBOL~"

class TikTokenTokenizer(Tokenizer):

    def __init__(self, model_path: str) -> None:
        try:
            # on Vulcan need to first load these because there is no internet connection on the compute nodes
            self.tkt_model = tiktoken.encoding_for_model(model_path)
        except:
            mergeable_ranks = load_tiktoken_bpe(model_path)
            all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
            missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
            for id in missing_ids:
                all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
            for name in all_special_tokens_with_ids:
                all_special_tokens_with_ids[name] += len(mergeable_ranks)
            logger.error(f"Failed to load TikToken model from {model_path}")
            mergeable_ranks = load_tiktoken_bpe(model_path)
            self.tkt_model = tiktoken.core.Encoding(
                name=Path(model_path).stem,
                pat_str=DEFAULT_TIKTOKEN_PATTERN,
                mergeable_ranks=mergeable_ranks,
                special_tokens=all_special_tokens_with_ids,
            )

        try:
            self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        except:
            self.bos_id: int = None
        self.eos_id: int = self.tkt_model.encode_single_token("<|endoftext|>")

        self.n_words: int = self.tkt_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert isinstance(s, str)
        
        add_bos = self.bos_id is not None and add_bos

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return (
            [self.bos_id] * add_bos
            + sum(self.tkt_model.encode_ordinary_batch(subs), start=[])
            + [self.eos_id] * add_eos
        )

    def decode(self, tokens: List[int],skip_special_tokens:bool=None):
        return self.tkt_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is not None:
            token_bytes = self.tkt_model.decode_tokens_bytes(tokens)
        else:
            token_bytes = self.tkt_model.decode_tokens_bytes(
                self.tkt_model.encode(text, allowed_special="all")
            )

        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        substrs = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return substrs, offsets

def find_id(tokenizer, surfaces: Sequence[str]):
    """Look through surfaces to see if any are in the tokenizer's vocab."""
    token_id = None
    for surface in surfaces:
        token_id = tokenizer.token_to_id(surface)
        if token_id is not None:
            logger.info("Found id for special token: %s", surface)
            break
    else:
        logger.warning("No id found for special token.")
    return token_id


class HFTokenizer(Tokenizer):

    def __init__(self, model_path: str, dropout: float = 0) -> None:
        try:
            import transformers
            # Try to load as a transformers.Tokenizer as it includes more
            # information about things like bos/eos
            transformers_tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path
            )
            logger.info("Loaded Transformers Tokenizer from %s", model_path)
            # Extract the underlying tokenizers.Tokenizer to get access to things
            # like the offests.
            self.hf_tokenizer = transformers_tokenizer._tokenizer
            logger.info(
                "Extracted Tokenizers Tokenizer from Transformers Tokenizer"
            )

            if dropout > 0:
                try:
                    self.hf_tokenizer.model.dropout = dropout
                    logger.info("Set tokenizer dropout to %f", dropout)
                except Exception as e:
                    logger.warning("Failed to set tokenizer dropout: %s", e)

            # Find special tokens based on the transformers.Tokenizer
            bos_token = transformers_tokenizer.bos_token
            logger.info(
                "Found bos_token: %s based on Transformers Tokenizer.", bos_token
            )
            self.bos_id = transformers_tokenizer.convert_tokens_to_ids(bos_token)

            eos_token = transformers_tokenizer.eos_token
            logger.info(
                "Found eos_token: %s based on Transformers Tokenizer.", eos_token
            )
            self.eos_id = transformers_tokenizer.convert_tokens_to_ids(eos_token)

            pad_token = transformers_tokenizer.pad_token
            logger.info(
                "Found pad_token: %s based on Transformers Tokenizer.", pad_token
            )
            if pad_token is not None:
                # It is ok for this not be set for models that don't have a pad
                # because it isn't set for some the other lingua implementations.
                self.pad_id = transformers_tokenizer.convert_tokens_to_ids(pad_token)

        except:
            import tokenizers
            # If we failed to load as a transformers.Tokenizer, load as a
            # tokenizers.Tokenizer
            self.hf_tokenizer = tokenizers.Tokenizer.from_file(model_path)
            logger.info("Loaded Tokenizers Tokenizer.")
            if dropout > 0:
                try:
                    self.hf_tokenizer._tokenizer.model.dropout = dropout
                    logger.info("Set tokenizer dropout to %f", dropout)
                except Exception as e:
                    logger.warning("Failed to set tokenizer dropout: %s", e)

            # We need to infer the special tokens. If you used a different
            # special token, it needs to be added tothe DEFAULT_SPECIAL_TOKENS
            # dict.
            logger.info("Infering bos id.")
            self.bos_id = find_id(self.hf_tokenizer, DEFAULT_SPECIAL_TOKENS["bos"])
            logger.info("Infering eos id.")
            self.eos_id = find_id(self.hf_tokenizer, DEFAULT_SPECIAL_TOKENS["eos"])
            logger.info("Infering pad id.")
            self.pad_id = find_id(self.hf_tokenizer, DEFAULT_SPECIAL_TOKENS["pad"])

        self.n_words = self.hf_tokenizer.get_vocab_size()

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        """Convert a string to a list of tokens."""
        # Never add bos/eos special tokens because we are using a
        # tokenizers.Tokenizer which doesn't auto add them.
        encoded = self.hf_tokenizer.encode(s, add_special_tokens=False).ids
        # Add bos/eos as needed, easy because we are not processing batches.
        if add_bos and self.bos_id is not None:
            encoded = [self.bos_id] + encoded
        if add_eos and self.eos_id is not None:
            encoded = encoded + [self.eos_id]
        return encoded

    def decode(self, tokens: List[int]):
        """Convert a list of tokens to a stirng."""
        return self.hf_tokenizer.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Get the offsets (and surface) for each token in the original string."""
        if tokens is not None:
            logger.warning(
                "`tokens` passed to `get_token_offsets`, but are ignored with the HFTokenizer."
            )

        # Don't add special tokens so we don't need to handle things like the
        # offset of the bos token.
        encoding = self.hf_tokenizer.encode(text, add_special_tokens=False)
        # Slice the original text instead of using encoding.tokens to avoid the
        # fact that tokenizers uses Ġ instead of space.
        substrs = [text[s:e] for s, e in encoding.offsets]
        return substrs, encoding.offsets


class ByT5HFTokenizer(HFTokenizer):

    def __init__(self, model_path: str) -> None:
        import transformers
        self.hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.bos_token = self.hf_tokenizer.pad_token
        self.eos_token = self.hf_tokenizer.eos_token
        self.bos_id = self.hf_tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_id = self.hf_tokenizer.convert_tokens_to_ids(self.eos_token)

        self.n_words = self.hf_tokenizer.vocab_size

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        """Convert a string to a list of tokens."""
        # Never add bos/eos special tokens because we are using a
        # tokenizers.Tokenizer which doesn't auto add them.
        encoded = self.hf_tokenizer.encode(s, add_special_tokens=False)
        # Add bos/eos as needed, easy because we are not processing batches.
        if add_bos and self.bos_id is not None:
            encoded = [self.bos_id] + encoded
        if add_eos and self.eos_id is not None:
            encoded = encoded + [self.eos_id]
        return encoded

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Get the offsets (and surface) for each token in the original string."""
        return None, None


class SimplifiedHFTokenizer(HFTokenizer):

    def __init__(self, model_path: str, dropout: float = 0) -> None:
        import transformers
        # Try to load as a transformers.Tokenizer as it includes more
        # information about things like bos/eos
        transformers_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path
        )
        logger.info("Loaded Transformers Tokenizer from %s", model_path)
        # Extract the underlying tokenizers.Tokenizer to get access to things
        # like the offests.
        self.hf_tokenizer = transformers_tokenizer._tokenizer
        logger.info(
            "Extracted Tokenizers Tokenizer from Transformers Tokenizer"
        )
        if dropout > 0:
            try:
                self.hf_tokenizer.model.dropout = dropout
                logger.info("Set tokenizer dropout to %f", dropout)
            except Exception as e:
                logger.warning("Failed to set tokenizer dropout: %s", e)
        special_tokens = getattr(transformers_tokenizer, "special_tokens_map", {})
        if "bert" in model_path:
            self.bos_token = special_tokens.get("cls_token")
        elif "t5" in model_path:
            self.bos_token = special_tokens.get("pad_token")
        else:
            self.bos_token = special_tokens.get("bos_token")
        logger.info(
            "Found bos_token: %s based on Transformers Tokenizer.", self.bos_token
        )
        if self.bos_token is not None:
            self.bos_id = transformers_tokenizer.convert_tokens_to_ids(self.bos_token)
        else:
            self.bos_id = None
        logger.info(
            "Found bos_id: %s based on Transformers Tokenizer.", self.bos_id
        )

        if "bert" in model_path:
            self.eos_token = special_tokens.get("pad_token")
        else:
            self.eos_token = special_tokens.get("eos_token")
        logger.info(
            "Found eos_token: %s based on Transformers Tokenizer.", self.eos_token
        )
        if self.eos_token is not None:
            self.eos_id = transformers_tokenizer.convert_tokens_to_ids(self.eos_token)
        else:
            self.eos_id = None
        logger.info(
            "Found eos_id: %s based on Transformers Tokenizer.", self.eos_id
        )

        self.n_words = self.hf_tokenizer.get_vocab_size()

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )


class TokenMonsterTokenizer(Tokenizer):

    def __init__(self, model_path: str):
        import tokenmonster
        self.tokenizer = tokenmonster.load(model_path)
        self.n_words = self.tokenizer.vocab_size
        self.bos_id = None
        self.eos_id = None

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        token_ids = self.tokenizer.tokenize(s)
        if token_ids is None:
            return np.array([], dtype=np.longlong)
        return token_ids.astype(np.longlong)

    def decode(self, tokens: List[int]):
        return self.tokenizer.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        return None, None


class TekkenTokenizer(Tokenizer):
    def __init__(self):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        tok = MistralTokenizer.v3(is_tekken=True)
        self.tokenizer = tok.instruct_tokenizer.tokenizer

        self.n_words = self.tokenizer.n_words
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        return self.tokenizer.encode(s, add_bos, add_eos)

    def decode(self, tokens: List[int]):
        if tokens[0] == self.bos_id:
            tokens = tokens[1:]
        if tokens[-1] == self.eos_id:
            tokens = tokens[:-1]
        return self.tokenizer.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        return None, None

class SupersetTokenizer(Tokenizer):
    n_words: int = 851586
    def __init__(self, tokenizers: List[Dict[str, str]], rng_state: Dict[str, Any] = None):
        self.tokenizers = {}
        ## todo: need to load mappings too
        import os
        for tokenizer_info in tokenizers:
            name = tokenizer_info["name"]
            path = tokenizer_info.get("path", None)
            kwargs = {}
            dropout = tokenizer_info.get("dropout", 0)
            if dropout > 0:
                kwargs["dropout"] = dropout
            load_supermapping = tokenizer_info.get("load_supermapping", False)
            try:
                tokenizer = build_tokenizer(name, path, **kwargs)
                encoding_path = path
                if name == "tiktoken":
                    encoding_path = f"tiktoken/{path}"
                elif name == "tokenmonster":
                    encoding_path = f"tokenmonster/{path}"
                elif name == "tekken":
                    encoding_path = f"mistralai/{path}"
                if load_supermapping:
                    logger.info(f"Loading supermapping for the tokenizer {path}")
                    tokenizer.load_supermapping(f"{os.environ.get('PROJECT')}/tokenizers/super_mappings", encoding_path)
                else:
                    logger.info(f"Not loading supermapping for the tokenizer {path}")
                self.tokenizers[f"{name}/{path}"] = tokenizer
            except Exception as e:
                logger.error("Error loading tokenizer %s from  %s. %s",path, name, e)
        if len(self.tokenizers) == 0:
            raise ValueError("No valid tokenizers provided.")
        
        logger.info(f"Number of tokenizers loaded: {len(self.tokenizers)}")
        if rng_state is not None:
            rng = np.random.default_rng()
            rng.bit_generator.state = rng_state
            self.rng = rng
            logger.info("Restored RNG state for supertokenizer.")
        else:
            self.rng = np.random.default_rng(seed=42)
            logger.info("Initialized new RNG for supertokenizer.")
        import json

        import huggingface_hub as hf_hub
        try:
            assert os.environ.get("HF_HUB_OFFLINE") != "1"
            repo_id = "gsaltintas/supertokenizer-super_vocab"
            path = hf_hub.hf_hub_download(repo_id, "super_vocab.json")
        except:
            path = f"{os.environ.get('PROJECT')}/tokenizers/supertokenizer/super_vocab.json"
        with open(path, "r") as f:
            self.super_vocab = json.load(f)
        # align bos eos with llama
        self.bos_id = self.super_vocab.get(ALIGNED_BOS)
        self.eos_id = self.super_vocab.get("<|end_of_text|>")
        self.bos_token, self.eos_token = ALIGNED_BOS, "<|end_of_text|>"
        logger.info(
            "Setting bos_token: %s with id %d.", self.bos_token, self.bos_id
        )
        logger.info(
            "Setting eos_token: %s with id %d.", self.eos_token, self.eos_id
        )

    def encode(self, tokens, add_bos, add_eos):
        tokenizer_key = self.rng.choice(list(self.tokenizers.keys()))
        tokenizer = self.tokenizers[tokenizer_key]
        ids = tokenizer.encode_to_supermapping(tokens, add_bos=False, add_eos=False)   
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        # logger.debug(f"Selected tokenizer {tokenizer_key}, length of ids: {len(ids)}, add_bos: {add_bos}, add_eos: {add_eos}")
        return ids

    def decode(self, tokens: List[int],skip_special_tokens:bool=None):
        pass

    def get_token_offsets(self, text: str, tokens: List[int] | None = None) -> Tuple[List[str] | List[int]]:
        return None, None

def build_tokenizer(name: str, path: Optional[Union[str, List[Dict[str, str]]]] = None, tokenizers: Optional[List[Dict[str, str]]]=None, dropout: float = 0, rng_state: Dict[str, Any] = None) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "mock":
        return MockTokenizer()
    elif name == "sp":
        return SentencePieceTokenizer(path)
    elif name == "tiktoken":
        return TikTokenTokenizer(path)
    elif name == "huggingface" and "byt5" in path:
        return ByT5HFTokenizer(path)
    elif name == "huggingface":
        return SimplifiedHFTokenizer(path, dropout=dropout)
    elif name == "tokenmonster":
        return TokenMonsterTokenizer(path)
    elif name == "tekken":
        return TekkenTokenizer()
    elif name == "supertokenizer":
        return SupersetTokenizer(tokenizers, rng_state=rng_state)
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
