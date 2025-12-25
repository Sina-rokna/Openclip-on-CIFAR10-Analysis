import open_clip
import torch 

def load_clip(args, device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model = model.to(device).eval()
    return model, tokenizer, preprocess