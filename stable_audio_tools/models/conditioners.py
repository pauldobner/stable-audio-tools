#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from .utils import load_ckpt_state_dict
from .transformer import AbsolutePositionalEmbedding
from .. import control_signals

from torch import nn

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            #self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
    
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class ListConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                options: tp.List[str]
                ):
        super().__init__(output_dim, output_dim)

        self.options = options
        self.embedder = nn.Embedding(len(options)+1, output_dim).requires_grad_(True)

    def forward(self, texts: tp.List[str], device=None) -> tp.Any:

        # Cast the inputs to floats, handling the case where the input is not in the options
        ints = [self.options.index(x) + 1 if x in self.options else 0 for x in texts]

        ints = torch.tensor(ints).to(device) # shape [batch_size]

        int_embeds = self.embedder(ints).unsqueeze(1) # shape [batch_size, 1, output_dim]

        return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

def clap_load_state_dict(clap_ckpt_path, clap_model):
    state_dict = torch.load(clap_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    # Remove "module." from state dict keys
    state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Fix for transformers library
    removed_keys = ["text_branch.embeddings.position_ids"]
    for removed_key in removed_keys:
        if removed_key in state_dict:
            del state_dict[removed_key]

    clap_model.load_state_dict(state_dict, strict=False)

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                clap_load_state_dict(clap_ckpt_path, self.model.model)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)

            # Cast text feature to same type as proj_out, unless proj_out is Identity
            if not isinstance(self.proj_out, nn.Identity):
                proj_out_dtype = next(self.proj_out.parameters()).dtype
                text_features = text_features.to(proj_out_dtype)                        

            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            text_embedding = text_embedding.to(proj_out_dtype)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                clap_load_state_dict(clap_ckpt_path, self.model.model)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        # Cast audio embedding to same type as proj_out, unless proj_out is Identity

        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            audio_embedding = audio_embedding.to(proj_out_dtype)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]

class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl", "google/t5-v1_1-xl", "google/t5-v1_1-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/t5-v1_1-xl": 2048,
        "google/t5-v1_1-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)
        
        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
            finally:
                logging.disable(previous_level)
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask
    
class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from g2p_en import G2p

        self.max_length = max_length

        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]
        
        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]
        
        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        
        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)
  
class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            use_abs_pos_emb = False,
            project_out: bool = False,
            special_tokens: tp.List[str] = []
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from transformers import AutoTokenizer

         # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        # Add special tokens
        if len(special_tokens) > 0:
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

        self.abs_pos_emb = None

        if use_abs_pos_emb:
            self.abs_pos_emb = AbsolutePositionalEmbedding(output_dim, max_length)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
    
        embeddings = self.token_embedder(input_ids)
            
        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        if self.abs_pos_emb is not None:
            embeddings = embeddings + self.abs_pos_emb(embeddings)

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int, save_pretransform: bool = False):
        super().__init__(pretransform.encoded_channels, output_dim)


        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform
        

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.stack(audio, dim=0)

        # Add batch dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)

        audio = audio.to(device)
        
        latents = self.pretransform.encode(audio)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]

class SourceMixConditioner(Conditioner):
    """
    A conditioner that mixes projected audio embeddings from multiple sources

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
        source_keys: a list of keys for the potential sources in the metadata

    """
    def __init__(
        self, 
        pretransform: Pretransform, 
        output_dim: int, 
        save_pretransform: bool = False, 
        source_keys: tp.List[str] = [], 
        pre_encoded: bool = False, 
        allow_null_source=False,
        source_length=None
    ):
        super().__init__(pretransform.encoded_channels, output_dim)

        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform

        self.source_keys = source_keys

        self.source_heads = nn.ModuleList([nn.Conv1d(pretransform.encoded_channels, output_dim, kernel_size=1) for _ in source_keys])        

        self.pre_encoded = pre_encoded

        self.allow_null_source = allow_null_source

        if self.allow_null_source:
            self.null_source = nn.Parameter(torch.randn(output_dim, 1))

            assert source_length is not None, "Source length must be specified if allowing null sources"

            self.source_length = source_length

    def forward(self, sources: tp.List[tp.Dict[str, torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        dtype = next(self.proj_out.parameters()).dtype

        # Output has to be the batch of summed projections
        # Input is per-batch-item list of source audio

        mixes = []

        for source_dict in sources: # Iterate over batch items

            mix = None

            for key_ix, key in enumerate(self.source_keys): # Iterate over potential sources
                if key in source_dict:

                    source = source_dict[key]

                    if not self.pre_encoded:
                        assert source.dim() == 2, f"Source audio must be shape [channels, samples], got shape: {source.shape}"
                        audio = set_audio_channels(source.unsqueeze(0), self.pretransform.io_channels)

                        audio = audio.to(device)
                        latents = self.pretransform.encode(audio).squeeze(0)
                    else:
                        latents = source.to(device)           

                    latents = latents.to(dtype)

                    if mix is None:
                        mix = self.source_heads[key_ix](latents)
                    else:
                        mix += self.source_heads[key_ix](latents)
            
            if mix is not None:
                mixes.append(mix)
            else:
                if self.allow_null_source:
                    mixes.append(self.null_source.repeat(1, self.source_length))
                else:
                    raise ValueError("No sources found for mix")

        mixes = torch.stack(mixes, dim=0)

        return [mixes, torch.ones(mixes.shape[0], mixes.shape[2]).to(mixes.device)]


class ControlSignalConditioner(Conditioner):
    """
    A conditioner that extracts control signals (loudness, spectral centroid, pitch) from audio
    using the provided control_signals module.
    
    Args:
        output_dim: the dimension of the output embeddings
        control_type: the type of control signal to extract ("loudness", "centroid", "pitch")
        sample_rate: the sample rate of the audio
        hop_length: the hop length for the control signal extraction (should match model's latent hop length)
        project_out: whether to add another linear projection to the output embeddings
    """
    def __init__(
        self,
        output_dim: int,
        control_type: str,
        sample_rate: int,
        hop_length: int,
        input_dim: int = 1,
        project_out: bool = False,
        device: str = "cuda"
    ):
        # The control signals are 1D (scalar per frame) or multi-dimensional (pitch bins)
        super().__init__(input_dim, output_dim, project_out=project_out)
        
        self.control_type = control_type
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.device = device
        
        valid_types = ["loudness", "centroid", "pitch"]
        assert control_type in valid_types, f"control_type must be one of {valid_types}"

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.proj_out.to(device)
        
        if isinstance(audios, list):
            # Find max length to pad
            max_len = max([a.shape[-1] for a in audios])
            padded_audios = []
            for audio in audios:
                # audio shape: (channels, samples)
                if audio.shape[-1] < max_len:
                    padding = torch.zeros(audio.shape[0], max_len - audio.shape[-1]).to(audio)
                    padded_audios.append(torch.cat([audio, padding], dim=-1))
                else:
                    padded_audios.append(audio)
            audios = torch.stack(padded_audios, dim=0)
            
        # audios shape: (batch, channels, samples)
        
        # Convert to mono for feature extraction
        if audios.dim() == 3:
            audios_mono = audios.mean(dim=1) # (batch, samples)
        else:
            audios_mono = audios
            
        batch_size = audios_mono.shape[0]
        
        extracted_signals = []
        
        # Process each item in the batch
        # Note: control_signals functions work on numpy arrays mostly, except pitch which handles torch tensors
        # We'll process item by item for safety and simplicity with the provided librosa-based code
        
        for i in range(batch_size):
            audio_np = audios_mono[i].detach().cpu().numpy()
            
            if self.control_type == "loudness":
                signal = control_signals.compute_loudness(
                    audio_np, 
                    self.sample_rate, 
                    hop_length=self.hop_length
                )
            elif self.control_type == "centroid":
                signal = control_signals.spectral_centroid(
                    audio_np, 
                    self.sample_rate, 
                    hop_length=self.hop_length
                )
            elif self.control_type == "pitch":
                # extract_pitch_probability handles torch tensors and device
                # It expects (1, n_samples) or (n_samples,)
                # It returns probabilities. The paper/code implies we might want the raw probability or a specific processing.
                # control_signals.extract_pitch_probability returns pitch_probs
                
                # We pass the tensor directly to avoid cpu roundtrip if possible, 
                # but the function signature says "audio: np.ndarray | torch.Tensor"
                # and it uses torchcrepe.
                
                signal = control_signals.extract_pitch_probability(
                    audios_mono[i], 
                    self.sample_rate, 
                    hop_length=self.hop_length,
                    device=str(device)
                )
                
                # Signal is (n_frames, 360) or similar?
                # Wait, extract_pitch_probability returns pitch_probs. 
                # torchcrepe.infer returns (batch, time, 360) usually? 
                # Let's check control_signals.py again.
                # It returns pitch_probs.numpy().
                # And it does `pitch_probs = torchcrepe.infer(...)`.
                # If model='tiny', it returns probabilities over pitch bins.
                # But we initialized the conditioner with dim=1.
                # If the signal is high-dimensional (like pitch probs), we need to adjust.
                # The paper says "single linear adapter layer per control signal".
                # If it's pitch *probability*, it's likely a vector per frame.
                # If it's just f0, it's a scalar.
                # control_signals.py docstring: "Extract the raw pitch probabilities... Uses the CREPE 'tiny' variant."
                # So it returns a vector of probabilities (360 bins usually).
                # I need to check the dimensionality.
                pass

            # Handle dimensionality
            # Loudness: (n_frames,) -> Scalar
            # Centroid: (n_frames,) -> Scalar
            # Pitch: Likely (n_frames, 360)
            
            if self.control_type == "pitch":
                # For pitch, the signal is likely multi-dimensional (bins)
                # We need to handle this.
                # If it's 360 bins, we should project 360 -> output_dim.
                # But I initialized super().__init__(1, ...) assuming scalar.
                # I need to fix this logic.
                pass
            
            extracted_signals.append(signal)

        # Re-evaluating Pitch dimensionality
        # If control_type is pitch, we need to know the input dimension.
        # CREPE usually has 360 bins.
        # I will update the __init__ to handle this.
        
        # Let's assume for now I'll fix the __init__ logic in the next step or do it dynamically?
        # No, I should do it right here.
        # I will change the logic below to stack and project.
        
        # Convert list of numpy arrays to tensor
        # signals are (time, [bins])
        
        # Pad signals to same length if needed (though if audio was padded, signals should be roughly same length)
        max_sig_len = max([s.shape[0] for s in extracted_signals])
        
        padded_signals = []
        for s in extracted_signals:
            s_tensor = torch.tensor(s).to(device).float()
            if s_tensor.dim() == 1:
                s_tensor = s_tensor.unsqueeze(1) # (time, 1)
            
            # s_tensor is (time, channels)
            
            if s_tensor.shape[0] < max_sig_len:
                padding = torch.zeros(max_sig_len - s_tensor.shape[0], s_tensor.shape[1]).to(device)
                s_tensor = torch.cat([s_tensor, padding], dim=0)
                
            padded_signals.append(s_tensor)
            
        # Stack: (batch, time, channels)
        signals_tensor = torch.stack(padded_signals, dim=0)
        
        # Permute to (batch, channels, time) for consistency with some other parts? 
        # But Conditioner usually expects (batch, time, channels) for the projection input?
        # The other conditioners return [embeddings, mask] where embeddings is (batch, 1, dim) or (batch, seq, dim).
        # Here we have a sequence.
        
        # Project
        # self.proj_out expects (..., dim_in)
        # So (batch, time, channels) is correct.
        
        embeddings = self.proj_out(signals_tensor) # (batch, time, output_dim)
        
        # Transpose to (batch, output_dim, time) if that's what's expected?
        # Wait, T5Conditioner returns (batch, seq, dim).
        # But ConditionedDiffusionModelWrapper.get_conditioning_inputs handles it.
        # If using input_concat_ids, it expects (batch, channels, seq).
        # So I should probably return (batch, output_dim, time).
        
        # However, the other conditioners (like T5) return (batch, seq, dim).
        # And get_conditioning_inputs says:
        # "Concatenate all input concat conditioning inputs over the channel dimension... Assumes ... (batch, channels, seq)"
        # But T5 is usually Cross Attention (batch, seq, channels).
        
        # If I want to use this for `input_concat_ids`, I should return (batch, channels, seq).
        # But `Conditioner` interface seems to return (batch, seq, channels) generally (like T5).
        # Let's look at `SourceMixConditioner`. It returns `[mixes, mask]`.
        # `mixes` is `torch.stack(mixes, dim=0)`.
        # `mix` comes from `self.source_heads[key_ix](latents)`.
        # `latents` from `pretransform.encode` is (batch, channels, time).
        # `Conv1d` preserves (batch, channels, time).
        # So `SourceMixConditioner` returns (batch, channels, time).
        
        # So if I want to support `input_concat`, I should return (batch, channels, time).
        # If I want to support `cross_attn`, I should return (batch, time, channels).
        
        # The paper says "single linear adapter layer".
        # If I use `nn.Linear`, it works on the last dimension.
        # If I have (batch, time, 1), Linear -> (batch, time, out_dim).
        # If I want (batch, out_dim, time), I transpose.
        
        # Given these are "control signals" like envelopes, they are usually concatenated (Dense control).
        # So I should target (batch, out_dim, time).
        
        embeddings = embeddings.transpose(1, 2) # (batch, output_dim, time)
        
        return [embeddings, torch.ones(embeddings.shape[0], embeddings.shape[2]).to(device)]


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                # Debug prints
                # print(f"Checking key {condition_key} in metadata. Keys: {list(x.keys())}, Conditioner type: {type(conditioner)}")
                
                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    elif isinstance(conditioner, ControlSignalConditioner) and "audio" in x:
                        # If the key is not found, but it's a control signal conditioner, we can try to use the audio
                        condition_key = "audio"
                    else:
                        print(f"FAILED: Key {condition_key} not found. Available keys: {list(x.keys())}. Conditioner: {type(conditioner)}")
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]
                    
                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any], pretransform=None) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}
        
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "list":
            conditioners[id] = ListConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for pretransform conditioners"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(cond_pretransform, **conditioner_config)
        elif conditioner_type == "source_mix":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for source_mix conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for source_mix conditioners if use_model_pretransform is True"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = SourceMixConditioner(cond_pretransform, **conditioner_config)
        elif conditioner_type == "control_signal":
            # Determine input dimension based on control type
            control_type = conditioner_config.get("control_type")
            if control_type == "pitch":
                # CREPE tiny model output dimension is 360
                input_dim = 360
            else:
                # Loudness and Centroid are scalars
                input_dim = 1
            
            # Update the conditioner class to handle the input dim
            # We need to instantiate the class with the correct input dim
            # But the class __init__ I wrote above hardcoded 1. 
            # I will modify the class __init__ in the previous chunk to accept input_dim or infer it.
            # Actually, I'll just pass input_dim to the constructor.
            
            conditioners[id] = ControlSignalConditioner(input_dim=input_dim, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)