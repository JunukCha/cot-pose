import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import torch
from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralConfig
from peft import PeftModel
from PIL import Image
from transformers import BitsAndBytesConfig

from posegpt.models.posegpt_full_mask import PoseGPTFullMask

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import sys
sys.path.append('posescript')
from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.generative_caption.model_generative_caption import DescriptionGenerator
from text2pose.retrieval.model_retrieval import PoseText


def load_unipose_model(config, model_path, model_base, device_map='auto', torch_dtype=None, **kwargs):
    # load tokenizer
    if model_path.endswith('/'): model_path = model_path[:-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # load model
    print('Loading LLaVA from base model...')
    lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = PoseGPTFullMask.from_pretrained(
            model_base,
            low_cpu_mem_usage=True,
            attn_implementation=None,
            torch_dtype=torch_dtype,
            config=lora_cfg_pretrained,
            tokenizer=tokenizer,
            device_map=device_map,
            pose_vqvae_codebook_size=config.pose_vqvae_config.params.quantizer.params.nb_code,
            evaluate_task=None)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    model.model.mm_projector[0].weight = torch.nn.Parameter(torch.empty(4096, 2304, device=model.device, dtype=model.dtype))

    model.get_model().load_hmr_vit_backbone(**config)

    print('Loading additional LLaVA weights...')
    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    non_lora_trainables = {(k[len('base_model.model.'):] if k.startswith('base_model.model.') else k): v for k, v in non_lora_trainables.items()}
    model.resize_token_embeddings(len(tokenizer)) # type: ignore
    model.load_state_dict(non_lora_trainables, strict=False)

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    # build pose vqvae model
    model.get_model().load_pose_vqvae(**config)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        raise NotImplementedError
    image_processor = vision_tower.image_processor
    model.get_pose_vqvae().to(model.device).to(torch_dtype)
    model.get_hmr_vit_backbone().to(model.device).to(torch_dtype)
    return model, image_processor


def load_unipose_model_4bit(
    config,
    merged_model_dir: str,
    device_map='auto',
    torch_dtype=None,
    **kwargs
):
    # Load tokenizer from the merged model directory
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir, use_fast=False)

    # Load LoRA-merged configuration
    lora_cfg = LlavaMistralConfig.from_pretrained(merged_model_dir)

    # Prepare 4-bit quantization settings
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Load the model with 4-bit quantization
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = PoseGPTFullMask.from_pretrained(
            merged_model_dir,
            load_in_4bit=True,
            quantization_config=bnb_cfg,
            low_cpu_mem_usage=True,
            attn_implementation=None,
            torch_dtype=torch_dtype,
            config=lora_cfg,
            tokenizer=tokenizer,
            device_map=device_map,
            pose_vqvae_codebook_size=config.pose_vqvae_config.params.quantizer.params.nb_code,
            evaluate_task=None
        )

    # Load HMR backbone and VQ-VAE
    model.get_model().load_hmr_vit_backbone(**config)
    model.get_model().load_pose_vqvae(**config)

    # Ensure vision tower is ready
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        raise NotImplementedError("Vision tower failed to load.")
    image_processor = vision_tower.image_processor

    # Move additional modules to proper device and dtype
    model.get_pose_vqvae().to(model.device).to(torch_dtype)
    model.get_hmr_vit_backbone().to(model.device).to(torch_dtype)

    return model, image_processor

def load_unipose_model_wo_4bit(
    config,
    merged_model_dir: str,
    device_map='auto',
    torch_dtype=None,
    **kwargs
):
    # Load tokenizer from the merged model directory
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir, use_fast=False)

    # Load LoRA-merged configuration
    lora_cfg = LlavaMistralConfig.from_pretrained(merged_model_dir)

    # Load the model with 4-bit quantization
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = PoseGPTFullMask.from_pretrained(
            merged_model_dir,
            low_cpu_mem_usage=True,
            attn_implementation=None,
            torch_dtype=torch_dtype,
            config=lora_cfg,
            tokenizer=tokenizer,
            device_map=device_map,
            pose_vqvae_codebook_size=config.pose_vqvae_config.params.quantizer.params.nb_code,
            evaluate_task=None
        )

    # Load HMR backbone and VQ-VAE
    model.get_model().load_hmr_vit_backbone(**config)
    model.get_model().load_pose_vqvae(**config)

    # Ensure vision tower is ready
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        raise NotImplementedError("Vision tower failed to load.")
    image_processor = vision_tower.image_processor

    # Move additional modules to proper device and dtype
    model.get_pose_vqvae().to(model.device).to(torch_dtype)
    model.get_hmr_vit_backbone().to(model.device).to(torch_dtype)

    return model, image_processor

def load_cot_pose_model(config, model_path, model_base, device_map='auto', torch_dtype=None, **kwargs):
    # load tokenizer
    if model_path.endswith('/'): model_path = model_path[:-1]
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    # load model
    print('Loading LLaVA from base model...')
    lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = PoseGPTFullMask.from_pretrained(
            model_base,
            low_cpu_mem_usage=True,
            attn_implementation=None,
            torch_dtype=torch_dtype,
            config=lora_cfg_pretrained,
            tokenizer=tokenizer,
            device_map=device_map,
            pose_vqvae_codebook_size=config.pose_vqvae_config.params.quantizer.params.nb_code,
            evaluate_task=None)

    model.get_model().load_hmr_vit_backbone(**config)

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    # build pose vqvae model
    model.get_model().load_pose_vqvae(**config)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        raise NotImplementedError
    image_processor = vision_tower.image_processor
    model.get_pose_vqvae().to(model.device).to(torch_dtype)
    model.get_hmr_vit_backbone().to(model.device).to(torch_dtype)
    return model, image_processor

def load_model_description_generator(model_path, device):
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)

	# load checkpoint & model info
	ckpt = torch.load(model_path, weights_only=False)
	text_decoder_name = ckpt['args'].text_decoder_name
	transformer_mode = ckpt['args'].transformer_mode
	encoder_latentD = ckpt['args'].latentD
	decoder_latentD = ckpt['args'].decoder_latentD
	decoder_nlayers = ckpt['args'].decoder_nlayers
	decoder_nhead = ckpt['args'].decoder_nhead
	num_body_joints = getattr(ckpt['args'], 'num_body_joints', 52)

	# load model
	model = DescriptionGenerator(text_decoder_name=text_decoder_name,
								transformer_mode=transformer_mode,
								decoder_nlayers=decoder_nlayers,
								decoder_nhead=decoder_nhead,
								encoder_latentD=encoder_latentD,
								decoder_latentD=decoder_latentD,
								num_body_joints=num_body_joints
								).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)

	return model, get_tokenizer_name(text_decoder_name)

def load_model_retrieval(model_path, device):
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)

	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	text_encoder_name = ckpt['args'].text_encoder_name
	transformer_topping = getattr(ckpt['args'], 'transformer_topping', None)
	latentD = ckpt['args'].latentD
	num_body_joints = getattr(ckpt['args'], 'num_body_joints', 52)

	# load model
	model = PoseText(text_encoder_name=text_encoder_name,
				  	 transformer_topping=transformer_topping,
					 latentD=latentD,
					 num_body_joints=num_body_joints
					 ).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)

	return model, get_tokenizer_name(text_encoder_name)

def setup_models(model_paths, checkpoint, _load_model_func, device=None):

	device = device if device else "cuda"

	# load models
	models = []
	tokenizer_names = []
	for i, mp in enumerate(model_paths):
		if ".pth" not in mp:
			mp = mp + f"/checkpoint_{checkpoint}.pth"
			print(f"Checkpoint not specified (model {i}). Using {checkpoint} checkpoint.")

		m, ten = _load_model_func(mp, device)
		models.append(m)
		tokenizer_names.append(ten)

	return models, tokenizer_names
