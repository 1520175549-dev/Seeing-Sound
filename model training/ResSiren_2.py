import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


TARGET_WIDTH = 126


# ── Dataset ──────────────────────────────────────────────────────────────────
class UrbanSoundDataset(Dataset):
    def __init__(self, csv_path, fold_list, root_dir, target_mel_dir=None):
        df = pd.read_csv(csv_path)
        temp_info = df[df['fold'].isin(fold_list)]
        self.root_dir       = Path(root_dir)
        self.target_mel_dir = Path(target_mel_dir) if target_mel_dir else None

        exists = temp_info.apply(
            lambda x: (self.root_dir / f"fold{int(x['fold'])}" / x['file_name']).exists(), axis=1
        )
        self.data_info = temp_info[exists].reset_index(drop=True)

    def __len__(self):
        return len(self.data_info)

    def _load_npy(self, path):
        data = np.load(path)
        if data.ndim == 3:
            data = data.squeeze(0)
        w = data.shape[1]
        if w < TARGET_WIDTH:
            data = np.pad(data, ((0, 0), (0, TARGET_WIDTH - w)), mode='constant')
        elif w > TARGET_WIDTH:
            data = data[:, :TARGET_WIDTH]
        return torch.from_numpy(data).float().unsqueeze(0)   # [1, H, W]

    def __getitem__(self, index):
        row       = self.data_info.iloc[index]
        file_path = self.root_dir / f"fold{int(row['fold'])}" / row['file_name']
        label     = int(row['final_label']) - 1              # 0-indexed

        data = self._load_npy(file_path)

        has_clean  = False
        clean_data = torch.zeros_like(data)

        if self.target_mel_dir:
            clean_name = row.get('clean_npy_name', '')
            if pd.isna(clean_name):
                clean_name = ''
            clean_name = str(clean_name).strip()
            if clean_name and clean_name != 'nan':
                clean_path = self.target_mel_dir / f"fold{int(row['fold'])}" / clean_name
                if clean_path.exists():
                    clean_data = self._load_npy(clean_path)
                    has_clean  = True

        return data, torch.tensor(label).long(), clean_data, torch.tensor(has_clean).bool()


# ── ResBlk ───────────────────────────────────────────────────────────────────
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return F.relu(out)


# ── ResSiren ──────────────────────────────────────────────────────────────────
class ResSiren(nn.Module):
    def __init__(self):
        super(ResSiren, self).__init__()
        self.conv1  = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        self.blk1     = ResBlk(32,  64,  stride=1)
        self.blk2     = ResBlk(64,  128, stride=1)
        self.blk3     = ResBlk(128, 256, stride=1)
        self.blk4     = ResBlk(256, 512, stride=1)
        self.outlayer = nn.Linear(512, 3)
        self.decoder  = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,  1,   kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, w  = x.shape[2], x.shape[3]
        feat  = F.relu(self.conv1(x))
        feat  = self.blk1(feat)9
        feat  = self.blk2(feat)
        feat  = self.blk3(feat)
        feat  = self.blk4(feat)
        pooled = F.adaptive_avg_pool2d(feat, [1, 1]).view(feat.size(0), -1)
        logits = self.outlayer(pooled)
        recon  = self.decoder(feat)
        recon  = F.interpolate(recon, size=(h, w), mode='bilinear', align_corners=False)
        return logits, recon


# ── Loss (placeholder, overridden in __main__ with weighted version) ──────────
cls_criterion = nn.CrossEntropyLoss()
LAMBDA_RECON  = 0.5


def compute_loss(logits, labels, recon, clean, has_clean):
    cls_loss   = cls_criterion(logits, labels)
    recon_loss = torch.tensor(0.0, device=logits.device)

    valid = has_clean
    if valid.sum() > 0:
        pred_mel = recon[valid]
        gt_mel   = clean[valid]
        gt_flat  = gt_mel.flatten(2)
        gt_min   = gt_flat.min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        gt_max   = gt_flat.max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        gt_norm  = (gt_mel - gt_min) / (gt_max - gt_min + 1e-8)
        recon_loss = F.mse_loss(pred_mel, gt_norm)

    total = cls_loss + LAMBDA_RECON * recon_loss
    return total, cls_loss, recon_loss


# ── Decoder evaluation ────────────────────────────────────────────────────────
def evaluate_decoder(model, device, loader):
    model.eval()
    mse_list, psnr_list = [], []

    with torch.no_grad():
        for data, labels, clean, has_clean in loader:
            data, labels     = data.to(device), labels.to(device)
            clean, has_clean = clean.to(device), has_clean.to(device)
            _, recon         = model(data)

            valid = has_clean
            if valid.sum() == 0:
                continue

            pred_mel = recon[valid]
            gt_mel   = clean[valid]
            gt_flat  = gt_mel.flatten(2)
            gt_min   = gt_flat.min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            gt_max   = gt_flat.max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            gt_norm  = (gt_mel - gt_min) / (gt_max - gt_min + 1e-8)

            mse  = F.mse_loss(pred_mel, gt_norm, reduction='none').mean(dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            mse_list.append(mse.cpu())
            psnr_list.append(psnr.cpu())

    if not mse_list:
        print("  [Decoder] no GT samples in this split.")
        return {}

    mse_all  = torch.cat(mse_list)
    psnr_all = torch.cat(psnr_list)
    metrics  = {'MSE': mse_all.mean().item(), 'PSNR': psnr_all.mean().item(), 'n': len(mse_all)}
    print(f"  [Decoder] n={metrics['n']}  MSE={metrics['MSE']:.4f}  PSNR={metrics['PSNR']:.2f} dB")
    return metrics


# ── Train / Test ──────────────────────────────────────────────────────────────
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_index, (data, target, clean, has_clean) in enumerate(train_loader):
        data, target     = data.to(device), target.to(device)
        clean, has_clean = clean.to(device), has_clean.to(device)

        optimizer.zero_grad()
        logits, recon      = model(data)
        loss, cls_l, rec_l = compute_loss(logits, target, recon, clean, has_clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_index % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_index * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_index / len(train_loader):.0f}%)]'
                  f'  loss={loss.item():.4f}  cls={cls_l.item():.4f}  recon={rec_l.item():.4f}')

    return total_loss / len(train_loader)


def test_model(model, device, test_loader):
    model.eval()
    correct   = 0
    test_loss = 0
    with torch.no_grad():
        for data, target, clean, has_clean in test_loader:
            data, target     = data.to(device), target.to(device)
            clean, has_clean = clean.to(device), has_clean.to(device)
            logits, recon    = model(data)
            loss, _, _       = compute_loss(logits, target, recon, clean, has_clean)
            test_loss       += loss.item()
            pred             = logits.argmax(dim=1)
            correct         += pred.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"  [Cls]  Loss={avg_loss:.4f}  Accuracy={accuracy:.3f}%")
    return accuracy


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    CSV_PATH       = "/home/tg07382a/Desktop/total_mel_index.csv"
    ROOT_DIR       = "/home/tg07382a/Desktop/total_mel_dataset"
    TARGET_MEL_DIR = "/home/tg07382a/Desktop/target_mel_dataset"

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS     = 5

    # weighted loss: inverse frequency, weights on correct device
    # label counts: horn=3759, siren=4259, other=7340
    class_weights = torch.tensor([7340/3759, 7340/4259, 1.0]).to(DEVICE)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    folds      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fold_group = [folds[i:i + 2] for i in range(0, len(folds), 2)]

    print("Start training!")
    print(f"Device: {DEVICE}")
    print(f"Class weights: horn={class_weights[0]:.3f}  siren={class_weights[1]:.3f}  other={class_weights[2]:.3f}\n")

    final_results       = []
    all_decoder_metrics = []
    dec_metrics         = {}

    for i, test_folds in enumerate(fold_group):
        print(f"\n--- Round {i + 1} / {len(fold_group)} (Test Folds: {test_folds}) ---")
        train_folds = [f for f in folds if f not in test_folds]

        train_dataset = UrbanSoundDataset(CSV_PATH, train_folds, ROOT_DIR, TARGET_MEL_DIR)
        test_dataset  = UrbanSoundDataset(CSV_PATH, test_folds,  ROOT_DIR, TARGET_MEL_DIR)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        model    = ResSiren().to(DEVICE)
        optimize = optim.Adam(model.parameters(), lr=0.001)

        train_losses      = []
        test_accuracies   = []
        decoder_mse_curve = []
        best_accuracy     = 0

        for epoch in range(1, EPOCHS + 1):
            avg_loss    = train_model(model, DEVICE, train_loader, optimize, epoch)
            accuracy    = test_model(model, DEVICE, test_loader)
            dec_metrics = evaluate_decoder(model, DEVICE, test_loader)

            train_losses.append(avg_loss)
            test_accuracies.append(accuracy)
            if dec_metrics:
                decoder_mse_curve.append(dec_metrics['MSE'])

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f"ressiren_multitask_round{i + 1}.pth")

        final_results.append(best_accuracy)
        if dec_metrics:
            all_decoder_metrics.append(dec_metrics)

        n_plots = 3 if decoder_mse_curve else 2
        plt.figure(figsize=(6 * n_plots, 5))

        plt.subplot(1, n_plots, 1)
        plt.plot(range(1, EPOCHS + 1), train_losses, 'r-o', label='Train Loss')
        plt.title(f'Round {i + 1} - Training Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()

        plt.subplot(1, n_plots, 2)
        plt.plot(range(1, EPOCHS + 1), test_accuracies, 'b-o', label='Test Acc')
        plt.title(f'Round {i + 1} - Classification Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.grid(True); plt.legend()

        if decoder_mse_curve:
            plt.subplot(1, n_plots, 3)
            plt.plot(range(1, len(decoder_mse_curve) + 1), decoder_mse_curve, 'g-o', label='Decoder MSE')
            plt.title(f'Round {i + 1} - Decoder MSE (↓)')
            plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.grid(True); plt.legend()

        plt.tight_layout()
        plt.savefig(f'round_{i + 1}.png')
        plt.close()

    print("\n--- All Rounds Completed! ---")
    print(f"Best Accuracies per Round : {final_results}")
    print(f"Mean Classification Acc   : {np.mean(final_results):.2f}%")
    if all_decoder_metrics:
        print(f"Mean Decoder MSE          : {np.mean([m['MSE']  for m in all_decoder_metrics]):.4f}")
        print(f"Mean Decoder PSNR         : {np.mean([m['PSNR'] for m in all_decoder_metrics]):.2f} dB")