import torch
import torch.nn.functional as Func
from embeddings.text_embedding import compute_text_embeddings
from tqdm import tqdm
from collections import defaultdict


def run_zero_shot(model, test_loader, classnames, tokenizer, args, device):
    text_emb = compute_text_embeddings(model, tokenizer, classnames, device, args= args)
    correct, total = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            img_emb = Func.normalize(model.encode_image(imgs), dim=-1)
            preds = (img_emb @ text_emb.T).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # per-class accuracy
            for c in range(len(classnames)):
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c] += mask.sum().item()

    print(f"Zero-shot accuracy: {correct / total:.4f}")

    print("Per-class accuracy:")
    for c, name in enumerate(classnames):
        acc = class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        print(f"  {name:10s}: {acc:.4f}")