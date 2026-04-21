from pathlib import Path, PurePosixPath

import modal

SUNET_ID = "iddah"

if SUNET_ID == "":
    raise NotImplementedError(f"Please set the SUNET_ID in {__file__}")

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"basics-{SUNET_ID}")
user_volume = modal.Volume.from_name(f"basics-{SUNET_ID}", create_if_missing=True, version=2)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = modal.Image.debian_slim().apt_install("wget", "gzip")
    
    image = image.add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True).uv_sync()
    image = image.add_local_python_source("cs336_systems")

    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    f"/root/{DATA_PATH}": user_volume,
}


def secrets(include_huggingface_secret: bool = False) -> list[modal.Secret]:
    secrets = [modal.Secret.from_dict({"SOME_ENV_VAR": "some-value"}), modal.Secret.from_name("my-secret")]
    return secrets