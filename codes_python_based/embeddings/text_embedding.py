import torch
import torch.nn.functional as Func

def compute_text_embeddings(model, tokenizer, classnames, device, args):
    """
    Compute text embeddings based on template_mode.
    """
    if args.template_mode == 'shared':
        texts = []
        for t in args.templates:
            for c in classnames:
                texts.append(t.format(c) if "{}" in t else t.format(classname=c))
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(tokens)
        Embedding_shape = text_emb.shape[-1]
        text_emb = text_emb.reshape(len(args.templates), len(classnames), Embedding_shape).mean(0)  # (number of templates, number of classes, Embedding shape) 
                                                                                            # -> (number of classes, D) -> (number of classes, Embedding shape)

    elif args.template_mode == 'per_class':
            # specialized template per class → no ensembling, just one per class
            if len(args.per_class_templates) != len(classnames):
                raise ValueError(f"Expected {len(classnames)} per-class templates, got {len(args.per_class_templates)}")
            texts = []
            for template, classname in zip(args.per_class_templates, classnames):
                texts.append(template.format(classname=classname))
            
            tokens = tokenizer(texts).to(device)
            with torch.no_grad():
                text_emb = model.encode_text(tokens)  
                
    text_emb = Func.normalize(text_emb, dim=-1) # normalize our embeddings 
    return text_emb