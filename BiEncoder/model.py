from sentence_transformers import models
from transformers import T5Config, AutoModel, LlamaConfig


class Transformer(models.Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        elif isinstance(config, LlamaConfig):
            self._load_llama_model(model_name_or_path, config, cache_dir)
        else:
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def _load_llama_model(self, model_name_or_path, config, cache_dir):
        """Loads the encoder model from Llama"""
        from transformers import LlamaModel
        from peft import get_peft_model, LoraConfig, TaskType

        self.auto_model = LlamaModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        self.auto_model = get_peft_model(self.auto_model, peft_config)
        # model.print_trainable_parameters()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.auto_model.config.pad_token_id = self.tokenizer.pad_token_id
