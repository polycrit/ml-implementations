import sys
import modal

app = modal.App("resnet34-fsd50k")

image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("resnet.py", "/root/resnet.py")
)

volume = modal.Volume.from_name("resnet-fsd50k-vol", create_if_missing=True)


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=60 * 60 * 12,
)
def train_resnet():
    import os

    os.environ["HF_HOME"] = "/vol/huggingface_cache"

    sys.path.insert(0, "/root")
    from resnet import train

    train(
        epochs=20,
        batch_size=128,
        lr=5e-4,
        weight_decay=0.01,
        num_workers=4,
        save_path="/vol/resnet34_dropout_fsd50k_best.pt",
    )

    volume.commit()
    print("Checkpoint saved to volume at /vol/resnet34_dropout_fsd50k_best.pt")


@app.local_entrypoint()
def main():
    train_resnet.remote()
