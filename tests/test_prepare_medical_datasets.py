import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_medical_datasets as pmd  # noqa: E402


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _gradient(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


def test_find_duplicates_reports_exact_and_near_only(tmp_path):
    base = _gradient(0)
    near = base.copy()
    near[0, 0] = (base[0, 0].astype(int) + 3).clip(0, 255).astype(np.uint8)
    _save(tmp_path / "histopathology" / "kather2016" / "TUM" / "a.png", base)
    _save(tmp_path / "histopathology" / "nct_crc_100k" / "TUM" / "a_copy.png", base)
    _save(tmp_path / "histopathology" / "nct_crc_100k" / "TUM" / "a_near.png", near)
    _save(tmp_path / "histopathology" / "kather2016" / "ADI" / "other.png", _gradient(1))
    _save(tmp_path / "_downloads" / "ignored.png", base)

    rows = pmd.find_duplicates(tmp_path, threshold=5)

    exact = {r["path"] for r in rows if r["group_type"] == "exact"}
    near_paths = {r["path"] for r in rows if r["group_type"] == "near"}
    a = str(Path("histopathology/kather2016/TUM/a.png"))
    copy = str(Path("histopathology/nct_crc_100k/TUM/a_copy.png"))
    near_p = str(Path("histopathology/nct_crc_100k/TUM/a_near.png"))
    assert exact == {a, copy}
    assert near_paths == {a, copy, near_p}
    assert all("other.png" not in r["path"] and "_downloads" not in r["path"] for r in rows)
    assert {r["dataset"] for r in rows} == {"histopathology/kather2016", "histopathology/nct_crc_100k"}

    out = tmp_path / "dups.csv"
    pmd.write_duplicates_csv(rows, out)
    assert out.read_text(encoding="utf-8").splitlines()[0] == "group_type,group_id,dataset,path,sha256,ahash"


def test_find_duplicates_distinct_images_not_reported(tmp_path):
    for i in range(5):
        _save(tmp_path / "fundus" / "x" / "c" / f"{i}.png", _gradient(10 + i))
    assert pmd.find_duplicates(tmp_path, threshold=5) == []


class _FakeBlood:
    calls = []

    def __init__(self, split, download, size, root):
        _FakeBlood.calls.append((split, download, size))
        n = {"train": 3, "val": 1, "test": 2}[split]
        self.imgs = np.full((n, 4, 4, 3), 100, dtype=np.uint8)
        self.labels = np.array([[i % 2] for i in range(n)])


def test_export_medmnist_writes_imagefolder_splits(tmp_path, monkeypatch):
    import medmnist

    _FakeBlood.calls = []
    monkeypatch.setattr(medmnist, "BloodMNIST", _FakeBlood)

    out = pmd.export_medmnist("bloodmnist_224", tmp_path, download=False)

    assert out == tmp_path / "hematology" / "bloodmnist_224"
    assert _FakeBlood.calls == [("train", False, 224), ("val", False, 224), ("test", False, 224)]
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png"))
    assert files == [
        "test/0_basophil/test_000000.png",
        "test/1_eosinophil/test_000001.png",
        "train/0_basophil/train_000000.png",
        "train/0_basophil/train_000002.png",
        "train/1_eosinophil/train_000001.png",
        "val/0_basophil/val_000000.png",
    ]
    with Image.open(out / "train/0_basophil/train_000000.png") as img:
        assert img.size == (4, 4)

    # Re-running skips existing files.
    pmd.export_medmnist("bloodmnist_224", tmp_path, download=False)
    assert len(list(out.rglob("*.png"))) == 6


def test_chestmnist_is_not_exportable():
    assert all(v["flag"] != "chestmnist" for v in pmd.MEDMNIST_DATASETS.values())
    with pytest.raises(KeyError):
        pmd.export_medmnist("chestmnist_224", Path("unused"))


class _FakeResponse:
    def __init__(self, data: bytes, status: int, fail_after: int | None = None):
        self.data, self.status, self.pos, self.fail_after = data, status, 0, fail_after
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n=-1):
        if self.fail_after is not None and self.pos >= self.fail_after:
            raise ConnectionResetError("connection dropped")
        end = len(self.data) if n < 0 else self.pos + n
        if self.fail_after is not None:
            end = min(end, self.fail_after)
        chunk, self.pos = self.data[self.pos:end], end
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_resumes_after_drop_and_only_renames_when_complete(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 40
    calls = []

    def fake_urlopen(request, timeout=None):
        start = int(request.headers.get("Range", "bytes=0-")[6:-1])
        calls.append(start)
        if len(calls) == 1:  # first attempt drops mid-way
            return _FakeResponse(payload, 200, fail_after=3000)
        return _FakeResponse(payload[start:], 206)

    monkeypatch.setattr(pmd.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "data.zip"
    pmd._download_file("https://example.invalid/data.zip", dest, wait=0)

    assert calls == [0, 3000]
    assert dest.read_bytes() == payload
    assert not (tmp_path / "data.zip.part").exists()
