import torch
from torch.nn import functional as F


def add_unet_crossattn_map_store(unet, cross_attn_map_store):
    def store_attn_map(name, cross_attn_map_store):
        def store_attn_map_(attn_map):
            cross_attn_map_store[name] = attn_map
        
        return store_attn_map_

    for name, module in unet.named_modules():
        if 'attn2' in name and module.__class__.__name__ == 'CrossAttention':
            module.cross_attn_map_store = store_attn_map(name, cross_attn_map_store)

    return unet


def fuse_object_embeddings(
    inputs_embeds,  # (b, 77, 1024)
    image_token_mask,   # (b, 77)
    object_embeds,  # (b, max_num_objects, 1, 1536)
    num_objects,    # (b, 1)
    fuse_fn=torch.add,
):
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    seq_length = inputs_embeds.shape[1]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )

    valid_ids = torch.arange(max_num_objects, device=object_embeds.device)[None, :] < num_objects
    valid_object_embeds = flat_object_embeds[valid_ids.flatten()]

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)

    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds)
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    image_token_idx,
    image_token_idx_mask,
    loss_fn,
):
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn
        )
        loss += layer_loss
    return loss / num_layers


def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    object_token_idx,
    object_token_idx_mask,
    loss_fn,
):
    b, num_heads, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents ** 0.5)

    # Resize the object segmentation maps to the size of the cross attention scores
    object_segmaps = F.interpolate(
        object_segmaps, size=(size, size), mode="bilinear", antialias=True
    )  # (b, max_num_objects, size, size)

    object_segmaps = object_segmaps.view(
        b, max_num_objects, -1
    )  # (b, max_num_objects, num_noise_latents)


    cross_attention_scores = cross_attention_scores.view(
        b, num_heads, num_noise_latents, num_text_tokens
    )

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(b, 1, 1, max_num_objects).expand(
            b, num_heads, num_noise_latents, max_num_objects
        ),
    )  # (b, num_heads, num_noise_latents, max_num_objects)

    object_segmaps = (
        object_segmaps.permute(0, 2, 1)
        .unsqueeze(1)
        .expand(b, num_heads, num_noise_latents, max_num_objects)
    )

    loss = loss_fn(object_token_attn_prob, object_segmaps)

    loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
    object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / object_token_cnt).mean()

    return loss