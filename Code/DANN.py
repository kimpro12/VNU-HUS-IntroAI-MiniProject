import argparse
import copy
import math
import os
import random
import sys
from typing import Tuple, Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# -----------------------------------------------------------------------------
# Utils & Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Thiết lập seed để đảm bảo tính tái lập (reproducibility) cho quá trình huấn luyện.
    
    Args:
        seed (int): Giá trị seed cố định. Mặc định là 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Lớp tiện ích để tính toán và lưu trữ giá trị trung bình và hiện tại.
    Thường dùng để theo dõi loss hoặc accuracy qua từng batch.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset các chỉ số về 0."""
        self.sum: float = 0.0
        self.cnt: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Cập nhật giá trị mới.
        
        Args:
            val (float): Giá trị cần cập nhật (ví dụ: loss của batch).
            n (int): Số lượng phần tử (ví dụ: batch_size).
        """
        self.sum += float(val) * n
        self.cnt += n

    @property
    def avg(self) -> float:
        """
        Trả về giá trị trung bình hiện tại.
        
        Returns:
            float: Giá trị trung bình (sum / count).
        """
        return self.sum / max(1, self.cnt)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Tính toán độ chính xác (accuracy) của dự đoán.
    
    Args:
        logits (torch.Tensor): Đầu ra chưa qua softmax của model, shape (Batch, Classes).
        y (torch.Tensor): Nhãn thực tế (Ground truth), shape (Batch,).
        
    Returns:
        float: Độ chính xác trung bình của batch.
    """
    return (logits.argmax(dim=1) == y).float().mean().item()


# -----------------------------------------------------------------------------
# Gradient Reversal Layer (GRL) & Model Architecture
# -----------------------------------------------------------------------------

class GradReverse(torch.autograd.Function):
    """
    Implement Gradient Reversal Layer (GRL).
    Trong quá trình forward, nó hoạt động như hàm identity.
    Trong quá trình backward, nó đảo ngược gradient (nhân với -lambda).
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        """
        Forward pass: Giữ nguyên input.
        
        Args:
            ctx: Context object để lưu trữ thông tin cho backward pass.
            x (torch.Tensor): Input tensor.
            lambd (float): Hệ số đảo ngược gradient.
            
        Returns:
            torch.Tensor: Output tensor (giống hệt input x).
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: Đảo ngược gradient.
        
        Args:
            ctx: Context object đã lưu lambd từ forward.
            grad_output (torch.Tensor): Gradient từ lớp phía sau truyền về.
            
        Returns:
            Tuple[torch.Tensor, None]: Gradient đã đảo ngược cho x, và None cho lambd.
        """
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Wrapper function để gọi GradReverse."""
    return GradReverse.apply(x, lambd)


def dann_lambda_schedule(global_step: int, max_steps: int, gamma: float = 10.0) -> float:
    """
    Tính toán giá trị lambda theo lịch trình (schedule) được đề xuất trong paper DANN.
    Lambda tăng dần từ 0 lên 1 trong quá trình huấn luyện để ổn định model lúc đầu.
    
    Args:
        global_step (int): Bước hiện tại (current iteration).
        max_steps (int): Tổng số bước huấn luyện.
        gamma (float): Hệ số điều chỉnh độ dốc của đường cong.
        
    Returns:
        float: Giá trị lambda tại bước hiện tại.
    """
    p = float(global_step) / float(max_steps)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


class FeatureExtractor(nn.Module):
    """
    Mạng Convolutional để trích xuất đặc trưng (features) từ ảnh.
    Đầu vào: Ảnh (3, 32, 32). Đầu ra: Vector đặc trưng phẳng (flattened).
    """
    def __init__(self) -> None:
        super().__init__()
        # Input: 3 channels (RGB), 32x32 images
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5), nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5), nn.BatchNorm2d(50), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(True)
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Batch ảnh đầu vào.
            
        Returns:
            torch.Tensor: Features đã được làm phẳng. Size: (Batch, 1250) với ảnh 32x32.
        """
        return self.flatten(self.conv(x))


class LabelClassifier(nn.Module):
    """
    Bộ phân loại nhãn (Label Predictor).
    Dự đoán nhãn lớp (0-9) từ vector đặc trưng.
    """
    def __init__(self, in_dim: int = 1250, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 100), nn.BatchNorm1d(100), nn.ReLU(True),
            nn.Dropout2d(),
            """
            Trên [B, N] dùng Dropout2d sẽ dính warning, nó sẽ biến [B, N] thành [B, N, 1, 1]
            và theo quy tắc của Dropout2d nó sẽ ngẫu nhiên trong 1 batch sẽ tắt bất kì 50%(50% là mặc định của pytorch) channel
            hay trong trường hợp này thì nó sẽ tắt đi 50% neuron bất kì trong tất cả các mẫu trong batch
            Đây là 1 trick regularization khiến accuracy nếu dùng Dropout thường hay Dropout1d thì chỉ được tầm 76-77% accuracy,
            còn dùng cái này thì được tận 82%-84% accuracy
            """
            
            nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(True),
            nn.Linear(100, num_classes), nn.LogSoftmax(dim=1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class DomainClassifier(nn.Module):
    """
    Bộ phân loại miền (Domain Discriminator).
    Phân biệt xem features đến từ Source (0) hay Target (1).
    """
    def __init__(self, in_dim: int = 1250) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 100), nn.BatchNorm1d(100), nn.ReLU(True),
            nn.Linear(100, 2), nn.LogSoftmax(dim=1)
        )

    def forward(self, features: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Vector đặc trưng.
            lambd (float): Hệ số gradient reversal.
            
        Returns:
            torch.Tensor: Logits dự đoán domain (Source/Target).
        """
        # Áp dụng Gradient Reversal Layer trước khi đưa vào mạng fully connected
        features = grad_reverse(features, lambd)
        return self.fc(features)


class DANN(nn.Module):
    """
    Mô hình Domain-Adversarial Neural Network (DANN) hoàn chỉnh.
    Bao gồm: Feature Extractor, Label Classifier, và Domain Classifier.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.feat = FeatureExtractor()
        self.label = LabelClassifier(num_classes=num_classes)
        self.domain = DomainClassifier()

    def forward(self, x: torch.Tensor, lambd: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass trả về cả dự đoán nhãn và dự đoán domain.
        
        Args:
            x (torch.Tensor): Input images.
            lambd (float): Hệ số lambda cho GRL.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (label_output, domain_output)
        """
        raw_features = self.feat(x)
        label_output = self.label(raw_features)
        domain_output = self.domain(raw_features, lambd)
        return label_output, domain_output

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Chỉ trích xuất đặc trưng (dùng cho training loop tùy chỉnh)."""
        return self.feat(x)

    @torch.no_grad()
    def get_top_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất đặc trưng ở lớp sâu hơn (gần output layer) để visualize t-SNE.
        Lấy output của layer Linear(100, 100) cuối cùng trong LabelClassifier.
        """
        raw_features = self.feat(x)
        # Truy cập vào các layer con của LabelClassifier để lấy features cao cấp
        top_feature_extractor = nn.Sequential(*list(self.label.fc.children())[:-2])
        return top_feature_extractor(raw_features)


# -----------------------------------------------------------------------------
# Data Loading & Early Stopping
# -----------------------------------------------------------------------------

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Định nghĩa các phép biến đổi ảnh cho MNIST (Source) và MNIST-M (Target).
    
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (transform_source, transform_target)
    """
    # MNIST là ảnh xám (1 channel), cần duplicate thành 3 channels để khớp với MNIST-M (RGB)
    mnist_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307] * 3, std=[0.3081] * 3)
    ])
    mnistm_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    return mnist_tf, mnistm_tf


def build_loaders(mnistm_root: str, batch_size: int, num_workers: int, tgt_split: List[float], seed: int) -> Dict[str, DataLoader]:
    """
    Tạo các DataLoader cho cả Source và Target domain.
    
    Args:
        mnistm_root (str): Đường dẫn thư mục gốc chứa MNIST-M.
        batch_size (int): Kích thước batch.
        num_workers (int): Số luồng tải dữ liệu.
        tgt_split (List[float]): Tỷ lệ chia tập Target [train, val, test].
        seed (int): Random seed.
        
    Returns:
        Dict[str, DataLoader]: Dictionary chứa các loader keys: 'src_train', 'src_val', ...
    """
    set_seed(seed)
    mnist_tf, mnistm_tf = get_transforms()

    # Chuẩn bị dữ liệu Source (MNIST)
    src_train_full = datasets.MNIST("./data", train=True, download=True, transform=mnist_tf)
    src_test = datasets.MNIST("./data", train=False, download=True, transform=mnist_tf)
    
    # Chia Source Train thành Train/Val (90/10)
    n_train = int(0.9 * len(src_train_full))
    n_val = len(src_train_full) - n_train
    src_train, src_val = random_split(src_train_full, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # Chuẩn bị dữ liệu Target (MNIST-M)
    train_dir = os.path.join(mnistm_root, "training")
    test_dir = os.path.join(mnistm_root, "testing")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(f"Không tìm thấy thư mục '{train_dir}' hoặc '{test_dir}'. Vui lòng kiểm tra lại path.")

    # Gộp toàn bộ MNIST-M lại rồi chia theo tỷ lệ mong muốn
    tgt_full = ConcatDataset([ImageFolder(train_dir, mnistm_tf), ImageFolder(test_dir, mnistm_tf)])
    t_train, t_val, t_test = [int(p * len(tgt_full)) for p in tgt_split]
    # Cộng phần dư vào test set để đảm bảo tổng số lượng khớp
    t_test += len(tgt_full) - sum((t_train, t_val, t_test))
    
    tgt_train, tgt_val, tgt_test = random_split(tgt_full, [t_train, t_val, t_test], generator=torch.Generator().manual_seed(seed))

    def create_loader(ds, is_train: bool) -> DataLoader:
        return DataLoader(ds, batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True, drop_last=is_train)

    return {
        "src_train": create_loader(src_train, True), "src_val": create_loader(src_val, False), "src_test": create_loader(src_test, False),
        "tgt_train": create_loader(tgt_train, True), "tgt_val": create_loader(tgt_val, False), "tgt_test": create_loader(tgt_test, False)
    }


class EarlyStopping:
    """
    Cơ chế dừng sớm (Early Stopping) để tránh overfitting.
    Theo dõi validation loss và lưu lại trạng thái model tốt nhất.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, warmup_epochs: int = 0) -> None:
        """
        Args:
            patience (int): Số epoch chấp nhận không cải thiện trước khi dừng.
            min_delta (float): Ngưỡng cải thiện tối thiểu.
            warmup_epochs (int): Số epoch đầu bỏ qua không check early stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[Dict[str, Any]] = None

    def step(self, current_epoch: int, val_loss: float, model: nn.Module) -> bool:
        """
        Kiểm tra điều kiện dừng.
        
        Args:
            current_epoch (int): Epoch hiện tại.
            val_loss (float): Loss trên tập validation.
            model (nn.Module): Model hiện tại để lưu state_dict nếu tốt nhất.
            
        Returns:
            bool: True nếu cần dừng training, False nếu tiếp tục.
        """
        if current_epoch < self.warmup_epochs:
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        return self.counter >= self.patience


# -----------------------------------------------------------------------------
# Training & Evaluation Functions
# -----------------------------------------------------------------------------

def train_source_only_one_epoch(model: nn.Module, opt: torch.optim.Optimizer, device: torch.device, loader: DataLoader) -> float:
    """
    Huấn luyện model chỉ trên dữ liệu Source (không có domain adaptation).
    
    Returns:
        float: Giá trị Loss trung bình của epoch.
    """
    model.train()
    ce_loss = nn.NLLLoss()
    meter = AverageMeter()
    
    for x_s, y_s in loader:
        x_s, y_s = x_s.to(device), y_s.to(device)
        opt.zero_grad()
        # lambd=0 để vô hiệu hóa ảnh hưởng của domain classifier (nếu có)
        y_s_logits, _ = model(x_s, lambd=0)
        loss = ce_loss(y_s_logits, y_s)
        loss.backward()
        opt.step()
        meter.update(loss.item(), x_s.size(0))
        
    return meter.avg


def train_dann_one_epoch(
    model: nn.Module, 
    opt: torch.optim.Optimizer, 
    device: torch.device, 
    loaders: Dict[str, DataLoader], 
    epoch: int, 
    total_epochs: int, 
    gamma: float, 
    dom_weight: float
) -> Dict[str, float]:
    """
    Huấn luyện DANN một epoch. Kết hợp dữ liệu Source (có nhãn) và Target (không nhãn).
    
    Args:
        model (nn.Module): DANN model.
        opt (torch.optim.Optimizer): Optimizer.
        device (torch.device): CPU/GPU.
        loaders (Dict): Dictionary chứa các dataloader.
        epoch (int): Index epoch hiện tại.
        total_epochs (int): Tổng số epochs.
        gamma (float): Tham số cho lambda schedule.
        dom_weight (float): Trọng số của domain loss trong tổng loss.
        
    Returns:
        Dict[str, float]: Dictionary chứa 'cls_loss' (phân loại) và 'dom_loss' (domain).
    """
    model.train()
    ce_loss = nn.NLLLoss()
    
    # Tạo iterator để lấy dữ liệu song song từ 2 loader
    src_iter = iter(loaders["src_train"])
    tgt_iter = iter(loaders["tgt_train"])
    n_iter = min(len(src_iter), len(tgt_iter))
    
    log = {"cls_loss": AverageMeter(), "dom_loss": AverageMeter()}

    for i in range(n_iter):
        # Lấy batch từ Source (có nhãn y_s)
        x_s, y_s = next(src_iter)
        # Lấy batch từ Target (chỉ cần ảnh x_t, bỏ qua nhãn giả định)
        x_t, _ = next(tgt_iter)
        
        x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)

        # Tính toán lambda động dựa trên tiến độ training
        lambd = dann_lambda_schedule(epoch * n_iter + i, total_epochs * n_iter, gamma)
        
        opt.zero_grad()

        # 1. Trích xuất đặc trưng chung
        f_s = model.features(x_s)
        f_t = model.features(x_t)

        # 2. Tính Classification Loss (chỉ trên Source)
        y_s_logits = model.label(f_s)
        classification_loss = ce_loss(y_s_logits, y_s)

        # 3. Tính Domain Loss (trên cả Source và Target)
        # Domain Source có nhãn 0, Domain Target có nhãn 1
        d_s_logits = model.domain(f_s, lambd)
        d_t_logits = model.domain(f_t, lambd)
        
        d_s_labels = torch.zeros(x_s.size(0), dtype=torch.long, device=device)
        d_t_labels = torch.ones(x_t.size(0), dtype=torch.long, device=device)
        
        domain_loss = ce_loss(d_s_logits, d_s_labels) + ce_loss(d_t_logits, d_t_labels)

        # 4. Tổng hợp loss và Backpropagation
        total_loss = classification_loss + dom_weight * domain_loss
        total_loss.backward()
        opt.step()

        log["cls_loss"].update(classification_loss.item(), x_s.size(0))
        log["dom_loss"].update(domain_loss.item(), x_s.size(0) + x_t.size(0))

    return {k: v.avg for k, v in log.items()}


@torch.no_grad()
def evaluate_source(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Đánh giá model trên tập dữ liệu Source (có nhãn).
    
    Returns:
        Tuple[float, float]: (Validation Loss, Accuracy)
    """
    model.eval()
    ce_loss = nn.NLLLoss()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, lambd=0)
        loss_meter.update(ce_loss(logits, y).item(), x.size(0))
        acc_meter.update(accuracy(logits, y), x.size(0))
        
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_target(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Đánh giá độ chính xác phân loại trên tập Target (dùng nhãn thực tế để kiểm chứng).
    
    Returns:
        float: Accuracy trên tập Target.
    """
    model.eval()
    meter = AverageMeter()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, lambd=0)
        meter.update(accuracy(logits, y), x.size(0))
        
    return meter.avg


@torch.no_grad()
def evaluate_domain_accuracy(model: nn.Module, loaders: Dict[str, DataLoader], device: torch.device) -> float:
    """
    Đánh giá khả năng phân biệt domain của Discriminator.
    Lý tưởng nhất cho DANN là Accuracy này tiến về 50% (Discriminator bị lừa hoàn toàn).
    
    Returns:
        float: Domain Accuracy.
    """
    model.eval()
    meter = AverageMeter()
    src_iter = iter(loaders["src_test"])
    tgt_iter = iter(loaders["tgt_test"])
    n_iter = min(len(src_iter), len(tgt_iter))

    for _ in range(n_iter):
        x_s, _ = next(src_iter)
        x_t, _ = next(tgt_iter)
        x_s, x_t = x_s.to(device), x_t.to(device)

        d_s_labels = torch.zeros(x_s.size(0), dtype=torch.long, device=device)
        d_t_labels = torch.ones(x_t.size(0), dtype=torch.long, device=device)

        _, d_s_logits = model(x_s, lambd=0)
        _, d_t_logits = model(x_t, lambd=0)

        meter.update(accuracy(d_s_logits, d_s_labels), x_s.size(0))
        meter.update(accuracy(d_t_logits, d_t_labels), x_t.size(0))
        
    return meter.avg


@torch.no_grad()
def visualize_tsne_like_paper(model: nn.Module, loaders: Dict[str, DataLoader], device: torch.device, title: str, filename: str, max_samples: int = 1000) -> None:
    """
    Tạo biểu đồ t-SNE để trực quan hóa phân bố đặc trưng của Source và Target.
    Lưu biểu đồ thành file ảnh.
    
    Args:
        model (nn.Module): Model DANN.
        loaders (Dict): Dataloaders.
        device (torch.device): CPU/GPU.
        title (str): Tiêu đề biểu đồ.
        filename (str): Tên file lưu.
        max_samples (int): Số lượng mẫu tối đa để plot (tránh quá tải t-SNE).
    """
    print(f"Generating t-SNE plot: {title}...")
    model.eval()

    def extract_top_features(loader: DataLoader, limit: int) -> np.ndarray:
        features = []
        count = 0
        for x, _ in loader:
            if count >= limit:
                break
            # Lấy features từ layer sâu nhất trước output
            f = model.get_top_features(x.to(device)).cpu().numpy()
            features.append(f)
            count += x.size(0)
        return np.concatenate(features, axis=0)

    # Lấy features mẫu
    src_features = extract_top_features(loaders["src_test"], max_samples // 2)
    tgt_features = extract_top_features(loaders["tgt_test"], max_samples // 2)

    # Gán nhãn domain (0: Source, 1: Target) để tô màu
    all_features = np.concatenate([src_features, tgt_features], axis=0)
    all_domains = np.concatenate([np.zeros(src_features.shape[0]), np.ones(tgt_features.shape[0])])

    # Chạy thuật toán t-SNE
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    features_2d = tsne.fit_transform(all_features)

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 8))
    colors = ["#0000ff", "#ff0000"]  # Blue for Source, Red for Target
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1], hue=all_domains,
        palette=colors, s=10, alpha=0.5
    )
    plt.title(title, fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.legend(["Source (MNIST)", "Target (MNIST-M)"])
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to: {filename}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Hàm chính điều phối toàn bộ luồng chương trình."""
    parser = argparse.ArgumentParser(description="DANN for MNIST -> MNIST-M with Source-Only Baseline")
    parser.add_argument("--mnistm-root", type=str, required=True, help="Path to MNIST-M dataset folder")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for EACH phase (Source-Only and DANN)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=10.0, help="Gamma for lambda scheduler")
    parser.add_argument("--dom-weight", type=float, default=1.0, help="Weight for domain loss.")
    parser.add_argument("--tgt-split", type=float, nargs=3, default=(0.8, 0.1, 0.1), help="Split ratio for Target (Train/Val/Test)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Epochs to wait before activating early stopping for DANN.")
    args = parser.parse_args()

    # Thiết lập thiết bị và seed
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        loaders = build_loaders(args.mnistm_root, args.batch_size, args.num_workers, args.tgt_split, args.seed)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # =========================================================================
    #   PHASE 1: SOURCE-ONLY BASELINE TRAINING
    #   Mục đích: Đo hiệu suất khi KHÔNG có Domain Adaptation để làm chuẩn so sánh.
    # =========================================================================
    print("\n" + "="*80)
    print(" " * 25 + "PHASE 1: SOURCE-ONLY BASELINE")
    print("="*80)
    
    model_source_only = DANN(num_classes=10).to(device)
    opt_source_only = torch.optim.AdamW(model_source_only.parameters(), lr=args.lr)
    early_stopper_so = EarlyStopping(patience=args.patience)

    for epoch in range(args.epochs):
        train_loss = train_source_only_one_epoch(model_source_only, opt_source_only, device, loaders["src_train"])
        val_loss, val_acc = evaluate_source(model_source_only, loaders["src_val"], device)
        
        print(f"[Source-Only] Epoch {epoch+1:02d}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc*100:.2f}%")
        
        if early_stopper_so.step(epoch, val_loss, model_source_only):
            print("Early stopping triggered.")
            break
            
    # Load lại weight tốt nhất
    if early_stopper_so.best_state:
        print("Restoring best model from Source-Only phase...")
        model_source_only.load_state_dict(early_stopper_so.best_state)

    print("\n--- Evaluating Source-Only Baseline Model ---")
    _, src_test_acc = evaluate_source(model_source_only, loaders["src_test"], device)
    tgt_test_acc_baseline = evaluate_target(model_source_only, loaders["tgt_test"], device)
    print(f"  Source (MNIST) Test Accuracy:   {src_test_acc*100:.2f}%")
    print(f"  Target (MNIST-M) Test Accuracy: {tgt_test_acc_baseline*100:.2f}%  <-- DOMAIN GAP")

    # =========================================================================
    #   PHASE 2: DOMAIN ADAPTATION WITH DANN
    #   Mục đích: Train DANN để thu nhỏ khoảng cách giữa 2 domain.
    # =========================================================================
    print("\n" + "="*80)
    print(" " * 27 + "PHASE 2: DANN TRAINING")
    print("="*80)
    
    # Khởi tạo model mới cho DANN
    model_dann = DANN(num_classes=10).to(device)
    opt_dann = torch.optim.AdamW(model_dann.parameters(), lr=args.lr, weight_decay=1e-3)
    early_stopper_dann = EarlyStopping(patience=args.patience, warmup_epochs=args.warmup_epochs)

    # Visualize phân bố trước khi train DANN (Sử dụng model chưa train hoặc model source-only đều được, ở đây dùng init mới)
    visualize_tsne_like_paper(model_dann, loaders, device, "Before Adaptation (Random Init)", "tsne_paper_before.png")

    for epoch in range(args.epochs):
        train_log = train_dann_one_epoch(model_dann, opt_dann, device, loaders, epoch, args.epochs, args.gamma, args.dom_weight)
        
        # Đánh giá
        src_val_loss, src_val_acc = evaluate_source(model_dann, loaders["src_val"], device)
        tgt_val_acc = evaluate_target(model_dann, loaders["tgt_val"], device)
        dom_val_acc = evaluate_domain_accuracy(model_dann, loaders, device)
        
        # Dùng Source Validation Loss để quyết định Early Stopping (tiêu chí an toàn nhất)
        should_stop = early_stopper_dann.step(epoch, src_val_loss, model_dann)
        
        print(
            f"[DANN] Epoch {epoch+1:02d}: "
            f"cls_loss={train_log['cls_loss']:.3f} | dom_loss={train_log['dom_loss']:.3f} | "
            f"src_val_loss={src_val_loss:.4f} | src_acc={src_val_acc*100:.2f}% | "
            f"tgt_acc={tgt_val_acc*100:.2f}% | dom_acc={dom_val_acc*100:.2f}%"
        )
        
        if should_stop:
            print(f"Early stopping triggered after epoch {epoch+1}.")
            break

    # Load lại weight tốt nhất của DANN
    if early_stopper_dann.best_state:
        print("Restoring best model from DANN phase...")
        model_dann.load_state_dict(early_stopper_dann.best_state)

    print("\n--- Final Evaluation After DANN ---")
    _, src_test_acc_dann = evaluate_source(model_dann, loaders["src_test"], device)
    tgt_test_acc_dann = evaluate_target(model_dann, loaders["tgt_test"], device)
    dom_acc_after = evaluate_domain_accuracy(model_dann, loaders, device)
    
    print(f"  Source (MNIST) Test Accuracy:   {src_test_acc_dann*100:.2f}%")
    print(f"  Target (MNIST-M) Test Accuracy: {tgt_test_acc_dann*100:.2f}%")
    print(f"  Final Domain Accuracy:          {dom_acc_after*100:.2f}% (Closer to 50% is better)")
    
    improvement = tgt_test_acc_dann - tgt_test_acc_baseline
    print(f"\n  IMPROVEMENT over Source-Only: +{improvement*100:.2f}%")

    visualize_tsne_like_paper(model_dann, loaders, device, "After Adaptation", "tsne_paper_after.png")
    
    # Lưu model tốt nhất
    save_path = "checkpoints/best_dann_final.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model_dann.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()