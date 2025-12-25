import torch.nn as nn
import copy 
import torch.nn.functional as Func
from tqdm import tqdm
from embeddings.text_embedding import compute_text_embeddings
import torch 

class ProjectionMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)


def run_img_txt_mlp(model, tokenizer, train_loader, val_loader, test_loader, classnames, args, device):
    # freeze CLIP
    if args.freeze_clip: 
        for p in model.parameters():
            p.requires_grad = False

    # text embeddings 
    text_embeddings = compute_text_embeddings(model, tokenizer, classnames, device, args = args)

    img_head = ProjectionMLP(
        input_dim=text_embeddings.shape[1],  ### TODO 
        hidden_dim=args.proj_hidden_dim,
        output_dim=text_embeddings.shape[1] ### TODO which size? 
    ).to(device)

    txt_head = ProjectionMLP(
        input_dim=text_embeddings.shape[1],   ### TODO 
        hidden_dim=args.proj_hidden_dim,
        output_dim=text_embeddings.shape[1] ### TODO which size? 
    ).to(device)

    optimizer = torch.optim.Adam(
        list(img_head.parameters()) + list(txt_head.parameters()),
        lr=args.proj_lr
    )

    best_val_acc = 0.0
    best_img_state, best_txt_state = None, None

    # Training loop
    for epoch in range(args.proj_epochs):
        img_head.train()
        txt_head.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                img_emb = Func.normalize(model.encode_image(imgs), dim=-1)
            img_proj = Func.normalize(img_head(img_emb), dim=-1)
            txt_proj = Func.normalize(txt_head(text_embeddings), dim=-1)
            logits = img_proj @ txt_proj.T
            loss = Func.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, loss={total_loss:.4f}")
        # Validation
        img_head.eval()
        txt_head.eval()

        correct, total = 0, 0
        with torch.no_grad():
            txt_proj = Func.normalize(txt_head(text_embeddings), dim=-1)
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                img_emb = Func.normalize(model.encode_image(imgs), dim=-1)
                img_proj = Func.normalize(img_head(img_emb), dim=-1)

                preds = (img_proj @ txt_proj.T).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Validation accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_img_state = copy.deepcopy(img_head.state_dict())
            best_txt_state = copy.deepcopy(txt_head.state_dict())

    # Test
    img_head.load_state_dict(best_img_state)
    txt_head.load_state_dict(best_txt_state)

    img_head.eval()
    txt_head.eval()

    correct, total = 0, 0
    class_correct = {c:0 for c in range(len(classnames))}
    class_total = {c:0 for c in range(len(classnames))}

    with torch.no_grad():
        txt_proj = Func.normalize(txt_head(text_embeddings), dim=-1)

        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            img_emb = Func.normalize(model.encode_image(imgs), dim=-1)
            img_proj = Func.normalize(img_head(img_emb), dim=-1)
            preds = (img_proj @ txt_proj.T).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # per-class tracking
            for c in range(len(classnames)):
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c] += mask.sum().item()

    print(f"Image–Text MLP Test Accuracy: {correct / total:.4f}")


    # per-class accuracy
    print("\nPer-class test accuracy:")
    for c, name in enumerate(classnames):
        acc_c = class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        print(f"  {name:10s}: {acc_c:.4f}")