from transformers import CLIPModel, CLIPProcessor


def load_clip_model(model_name, device):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor