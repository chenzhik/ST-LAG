import torch
from torch import nn
from typing import Any, Optional, Tuple, Union
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPPreTrainedModel, CLIPTextTransformer, CLIPVisionTransformer, CLIPOutput
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class CLIPModel_2(CLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

        self.max_token_count = 77
        self.empty_embeds = None

        # Initialize weights and apply final processing
        self.post_init()

    def calc_similarity(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # text_embeds = text_outputs[1]
        # text_embeds = self.text_projection(text_embeds)
        last_hidden_state = text_outputs[0]
        text_embeds_per_token = self.text_projection(last_hidden_state[0,:])

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds_per_token = self.text_projection(last_hidden_state[0,:])

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_text = torch.matmul(text_embeds_per_token, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        # convert to weights
        logits_per_token = (logits_per_text / logits_per_text[-1]).transpose(0,1)
        logits_per_token[:, 0] = 1.0
        per_token_weights = torch.nn.functional.pad(logits_per_token, (0, self.max_token_count-len(logits_per_text)), mode='constant', value=1.0)

        # norm
        per_token_weights -= per_token_weights.min(1, keepdim=True)[0]
        per_token_weights /= per_token_weights.max(1, keepdim=True)[0]
        # mean_value = per_token_weights.mean()
        # per_token_weights_ = torch.where(per_token_weights < mean_value, per_token_weights/2.0, per_token_weights)
        per_token_weights_ = per_token_weights / 1.5
        return (logits_per_text, logits_per_image, per_token_weights_)

    def prompt_reweighting(self, per_token_weights, prompt_embeds, tokenizers=None, text_encoders=None, attention_mask=None, disable_empty_z=False):
        """pipe.text_encoder, pipe.text_tokenizer"""
        
        
        if self.empty_embeds == None:
            self.empty_embeds = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                empty_token_ids = torch.tensor([tokenizer.bos_token_id] +
                                        [tokenizer.eos_token_id] +
                                        [tokenizer.pad_token_id] * (self.max_token_count - 2),
                                        dtype=torch.int, device=text_encoder.device).unsqueeze(0)
                self.empty_embeds.append(text_encoder(empty_token_ids, output_hidden_states=True, return_dict=True))
        
        # return concatnated PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED feature in SDXL
        empty_z = torch.cat([self.empty_embeds[i].hidden_states[-2] for i in range(len(self.empty_embeds))], dim=-1)
        
        z = prompt_embeds
        
        batch_weights_expanded = per_token_weights.reshape(per_token_weights.shape + (1,)).expand(z.shape).to(z)
        z_delta_from_empty = z - empty_z
        this_weighted_z = empty_z + (z_delta_from_empty * batch_weights_expanded)
        if disable_empty_z:
            this_weighted_z = z  * batch_weights_expanded
        return this_weighted_z 
    

if __name__ == "__main__":
    """ref: https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts"""
    
    clip_path = "/mnt/afs_longfuchen/chenzhikai/pretrain/huggingface/laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    model = CLIPModel_2.from_pretrained(clip_path)
    processor = CLIPProcessor.from_pretrained(clip_path)

    image = Image.open("./test.png")
    image2 = Image.open("./test.png")
    images = [image, image2]
    text = "night view, moon, cat, sofa, rabbit"
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    outputs = model.calc_similarity(**inputs)
    per_token_weights = outputs[2] 
    prompt_embeds, empty_embeds = torch.rand(1, 77, 768), torch.rand(1, 77, 768)
    model.prompt_reweighting(per_token_weights, prompt_embeds, empty_embeds)