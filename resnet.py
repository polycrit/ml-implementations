import io
import csv

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Features, Value
from tqdm import tqdm


SAMPLE_RATE = 22050
DURATION_SEC = 5
TARGET_LENGTH = SAMPLE_RATE * DURATION_SEC


# transforms raw audio into ResNet-friendly format
class MelSpec:
    def __init__(self):
        # create a Mel spectrogram (parameters inspired by pytorch docs)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )
        # turn linear amplitued into decibels (log scale) which is how humands perceive loudness
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform, sample_rate):
        # convert stereo to mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        # resample files to match fixed sample rate
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, SAMPLE_RATE
            )

        # padding/truncating
        # if too short, pads with zeros (which corresponds to silence)
        # if too long, crops the audio to keep only the first 5 seconds
        if waveform.shape[0] < TARGET_LENGTH:
            padding = TARGET_LENGTH - waveform.shape[0]
            waveform = nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:TARGET_LENGTH]

        spec = self.mel_spectrogram(waveform.unsqueeze(0))
        spec = self.amplitude_to_db(spec)

        # daata normalization so the NN trains faster
        spec = (spec - spec.mean()) / (spec.std() + 1e-9)
        return spec


REPO = "philgzl/fsd50k"
NUM_CLASSES = 200


# downloads vocabulary.csv, creates a mapping for label names to numeric IDs
def _load_vocabulary():
    path = hf_hub_download(REPO, "ground_truth/vocabulary.csv", repo_type="dataset")
    label_to_idx = {}
    with open(path) as f:
        for row in csv.reader(f):
            label_to_idx[row[1]] = int(row[0])
    return label_to_idx


# downloads main csv files, creates a mapping for filenames to primary labels
def _load_ground_truth(csv_name, split_filter=None):
    path = hf_hub_download(REPO, f"ground_truth/{csv_name}", repo_type="dataset")
    fname_to_label = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_filter and row["split"] != split_filter:
                continue
            fname_to_label[row["fname"]] = row["labels"].split(",")[0]
    return fname_to_label


# a pytorch dataset used to feed fsd50k data into the model
class FSD50KDataset(Dataset):
    def __init__(self, split):
        # map the requested split to correct file structure
        hf_split = {"train": "dev", "validation": "dev", "test": "eval"}[split]

        # load data
        self.data = load_dataset(
            REPO,
            split=hf_split,
            features=Features({"audio": Value("binary"), "name": Value("string")}),
        )

        # convert audio
        self.to_mel = MelSpec()
        self.num_classes = NUM_CLASSES

        # load the vocabulary and ground truths
        self.label_to_idx = _load_vocabulary()
        split_filter = split if split in ("train", "val") else None
        if split == "validation":
            split_filter = "val"
        csv_name = "dev.csv" if hf_split == "dev" else "eval.csv"
        self.fname_to_label = _load_ground_truth(csv_name, split_filter)

        # create a list of valid indices, ensuring we only try to load files that we actually have labels for
        self.indices = [
            i
            for i, row in enumerate(self.data)
            if self._fname(row["name"]) in self.fname_to_label
        ]

    @staticmethod
    def _fname(name):
        basename = name.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0]
        return stem.rsplit("_", 1)[0]

    # actual dataset length
    def __len__(self):
        return len(self.indices)

    # fetching an item (for training loop)
    def __getitem__(self, idx):
        # retrieve raw audio
        row = self.data[self.indices[idx]]
        audio_bytes = row["audio"]

        # decode audio
        waveform, sr = sf.read(io.BytesIO(audio_bytes))
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # transform
        spectrogram = self.to_mel(waveform, sr)

        # get label
        fname = self._fname(row["name"])
        primary_label = self.fname_to_label[fname]
        label_idx = self.label_to_idx[primary_label]

        return spectrogram, label_idx


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet34(nn.Module):
    LAYER_CONFIG = [
        (3, 64, 1),
        (4, 128, 2),
        (6, 256, 2),
        (3, 512, 2),
    ]

    def __init__(self, in_channels=1, num_classes=200):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        prev_channels = 64
        res_layers = []
        for num_blocks, channels, stride in self.LAYER_CONFIG:
            res_layers.append(
                self._make_layer(prev_channels, channels, num_blocks, stride)
            )
            prev_channels = channels
        self.res_layers = nn.Sequential(*res_layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, stride):
        blocks = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_layers(x)
        x = self.head(x)
        return x


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    num_correct = 0
    num_samples = 0

    desc = "Train" if training else "Eval"
    progress = tqdm(loader, desc=desc, leave=False)

    with torch.set_grad_enabled(training):
        for spectrograms, labels in progress:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            logits = model(spectrograms)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            batch_size = spectrograms.size(0)
            total_loss += loss.item() * batch_size
            num_correct += (logits.argmax(dim=1) == labels).sum().item()
            num_samples += batch_size

            progress.set_postfix(
                loss=total_loss / num_samples,
                acc=num_correct / num_samples,
            )

    avg_loss = total_loss / num_samples
    accuracy = num_correct / num_samples
    return avg_loss, accuracy


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    epochs=20,
    batch_size=32,
    lr=1e-3,
    weight_decay=0.01,
    num_workers=2,
    save_path="resnet34_fsd50k_best.pt",
):
    device = get_device()
    print(f"Using device: {device}")

    print("Loading FSD50K training split...")
    train_dataset = FSD50KDataset("train")
    print(f"  {len(train_dataset)} samples, {train_dataset.num_classes} classes")

    print("Loading FSD50K validation split...")
    val_dataset = FSD50KDataset("validation")
    print(f"  {len(val_dataset)} validation samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = ResNet34(in_channels=1, num_classes=train_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    best_val_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:>3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_accuracy:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model (val_acc={val_accuracy:.4f})")

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")

    print("Loading FSD50K test split...")
    test_dataset = FSD50KDataset("test")
    test_loader = DataLoader(
        test_dataset,
        batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    model.load_state_dict(torch.load("resnet34_fsd50k_best.pt", weights_only=True))
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  Test accuracy={test_accuracy:.4f}")
