"""Download and verify the Kather 2016 5000-tile dataset."""
from pathlib import Path
import argparse
import hashlib
import shutil
import urllib.request
import zipfile

URL = ("https://zenodo.org/records/53169/files/"
       "Kather_texture_2016_image_tiles_5000.zip?download=1")
MD5 = "0ddbebfc56344752028fda72602aaade"

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", type=Path,
        default=Path("docker-project/external_datasets/kather2016")
    )
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "Kather_texture_2016_image_tiles_5000.zip"

    if not archive.exists():
        print("Downloading from Zenodo...", flush=True)
        temporary = archive.with_suffix(".zip.part")
        try:
            with urllib.request.urlopen(URL, timeout=120) as response:
                with temporary.open("wb") as out:
                    shutil.copyfileobj(response, out)
            temporary.replace(archive)
        finally:
            if temporary.exists():
                temporary.unlink()

    digest = hashlib.md5()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise RuntimeError("Kather ZIP MD5 mismatch. Refusing extraction.")
    print("KATHER ZIP MD5: PASS")

    destination = root / "extracted"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path = (destination / item.filename).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise RuntimeError("Unsafe ZIP member")
        z.extractall(destination)

    image_dirs = []
    for d in destination.rglob("*"):
        if d.is_dir():
            images = [
                x for x in d.iterdir()
                if x.is_file() and x.suffix.lower() in (".tif", ".tiff")
            ]
            if images:
                image_dirs.append((d, len(images)))

    counts = sorted(n for _, n in image_dirs)
    if counts != [625] * 8:
        raise RuntimeError(f"Unexpected Kather class counts: {counts}")

    print("KATHER DATASET: PASS")
    print("CLASSES: 8")
    print("IMAGES: 5000")
    print("UNIFIED SPLIT: train=3496, val=744, test=760")
    print("EXTRACTED:", destination)

if __name__ == "__main__":
    main()
