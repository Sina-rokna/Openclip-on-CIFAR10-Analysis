from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch 
from tqdm import tqdm 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def run_linear_probe(train_feats, train_labels, test_feats, test_labels, args, classnames =
                    ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']):
    clf = LogisticRegression(C=args.logreg_C, max_iter=args.logreg_max_iter)
    clf.fit(train_feats, train_labels)
    preds = clf.predict(test_feats)
    acc = clf.score(test_feats, test_labels)
    print(f"Linear probe accuracy: {acc:.6f}")
    if classnames is not None:
        print("\nPer-class accuracy:")
        report = classification_report(test_labels, preds, target_names=classnames, digits=4, zero_division=0)
        print(report)

class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes))
    def forward(self, x):
        return self.net(x)

def run_mlp_probe(train_feats, train_labels, val_feats, val_labels, test_feats, test_labels,args, device, classnames =
                    ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']):
    train_dataset = TensorDataset(torch.tensor(train_feats, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_feats, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_feats, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.mlp_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.mlp_batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.mlp_batch_size)

    MLP_probe = MLPProbe(
        in_dim=train_feats.shape[1],
        hidden_dim=args.mlp_hidden_dim,
        num_classes=len(set(train_labels))
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MLP_probe.parameters(), lr=args.mlp_lr)
    best_acc, best_state = 0.0, None

    # training loop
    for epoch in range(args.mlp_epochs):
        MLP_probe.train()
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.mlp_epochs}")

        for x, y in bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(MLP_probe(x), y)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=loss.item())
        # validation
        MLP_probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = MLP_probe(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Validation acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = MLP_probe.state_dict()

    MLP_probe.load_state_dict(best_state)

    # test
    MLP_probe.eval()
    correct, total = 0, 0
    class_correct = {c:0 for c in range(len(classnames))} if classnames else None
    class_total = {c:0 for c in range(len(classnames))} if classnames else None

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = MLP_probe(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            # per-class tracking
            if classnames:
                for c in range(len(classnames)):
                    mask = y == c
                    class_correct[c] += (preds[mask] == y[mask]).sum().item()
                    class_total[c] += mask.sum().item()
                    
    print(f"MLP probe test accuracy: {correct / total:.4f}")

    # per-class accuracy
    if classnames:
        print("\nPer-class test accuracy:")
        for c, name in enumerate(classnames):
            acc_c = class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
            print(f"  {name:10s}: {acc_c:.4f}")
