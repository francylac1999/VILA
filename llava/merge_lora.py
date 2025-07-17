import os
import torch
from transformers import AutoConfig
from llava.model.builder import prepare_config_for_eval
from llava.model import LlavaLlamaModel
from peft import PeftModel

def merge_lora_weights(model_path: str, model_base_path: str, merged_model_path: str):
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    # Configurazione base
    lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    lora_cfg_pretrained.use_cache = False

    print("Loading LLaVA from base model...")
    config = AutoConfig.from_pretrained(model_base_path)
    prepare_config_for_eval(config, kwargs)

    # Caricamento modello base
    model = LlavaLlamaModel.from_pretrained(model_base_path, low_cpu_mem_usage=True, config=config, **kwargs)

    # Controllo su dimensione dei token
    token_num, token_dim = model.llm.lm_head.out_features, model.llm.lm_head.in_features
    if model.llm.lm_head.weight.shape[0] != token_num:
        model.llm.lm_head.weight = torch.nn.Parameter(
            torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype)
        )
        model.llm.embed_tokens.weight = torch.nn.Parameter(
            torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype)
        )

    # Caricamento pesi non-LoRA se esistono
    non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)

    # Merge dei LoRA
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, model_path, device_map="cpu", low_cpu_mem_usage=True)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Salvataggio
    print(f"Saving merged model to: {merged_model_path}")
    model.save_pretrained(merged_model_path)
    print("Done.")

# Esempio d'uso
if __name__ == "__main__":
    model_path = "/home/workspace/VILA/runs/train/NVILA-Lite-2B-finetune-efficient-qlora-5-epochs-PIC4SER/model/checkpoint-6140"
    model_base_path = "/home/workspace/NVILA-Lite-2B"
    merged_model_path = "/home/workspace/NVILA-Lite-2B-merged_adapters-qlora-PIC4SER-5-epochs"
    merge_lora_weights(model_path, model_base_path, merged_model_path)
