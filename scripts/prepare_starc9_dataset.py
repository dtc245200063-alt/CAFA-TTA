
import csv
import hashlib
import json
import platform
import zipfile

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import PIL

from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data/raw/starc9"

OUTPUT_DIR = ROOT / "data/processed/starc9"

EXPERIMENT_DIR = ROOT / "experiments/starc9_external"

ARCHIVE_PATH = (
    RAW_DIR
    / "STANFORD-CRC-HE-VAL-SMALL-NORMALIZED.zip"
)

MANIFEST_PATH = RAW_DIR / "dataset_manifest.json"

INSPECTION_PATH = RAW_DIR / "dataset_inspection.json"

PROTOCOL_PATH = EXPERIMENT_DIR / "external_protocol.json"

DATASET_PATH = OUTPUT_DIR / "starc9_target_28.npz"

INDEX_PATH = OUTPUT_DIR / "sample_index.csv"

PREPARATION_PATH = (
    OUTPUT_DIR / "preparation_manifest.json"
)

EXPECTED_PROTOCOL_HASH = (
    "0880BA90A14046C10E006F99B53CAAB6E"
    "013D46F1C62B56D38CEBD3A992E9295"
)

EXPECTED_ARCHIVE_HASH = (
    "088281DE09BFE4D57B4A8E356A99CDB9B"
    "002A08A0FF7C44910AEE0B4DE81D3F3"
)

LABEL_MAPPING = {
    "ADI": 0,
    "LYM": 3,
    "MUC": 4,
    "MUS": 5,
    "NOR": 6,
    "TUM": 8,
}

TARGET_CLASS_IDS = [0, 3, 4, 5, 6, 8]

EXPECTED_IMAGES = 12000

IMAGE_SIZE = 28


# =========================================================
# UTILITIES
# =========================================================

def require(condition, message):

    if not condition:
        raise RuntimeError(message)


def read_json(path):

    require(
        path.is_file(),
        f"Missing JSON file: {path}"
    )

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def sha256_file(path):

    require(
        path.is_file(),
        f"Missing file: {path}"
    )

    hasher = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):

            hasher.update(chunk)

    return hasher.hexdigest().upper()


def sha256_bytes(data):

    return hashlib.sha256(
        data
    ).hexdigest().upper()


def relative_path(path):

    return path.relative_to(
        ROOT
    ).as_posix()


def write_json(path, data):

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


# =========================================================
# VALIDATE FROZEN PROTOCOL
# =========================================================

def validate_protocol():

    print("\nVALIDATING FROZEN PROTOCOL")
    print("-" * 75)

    actual_protocol_hash = sha256_file(
        PROTOCOL_PATH
    )

    require(
        actual_protocol_hash
        == EXPECTED_PROTOCOL_HASH,
        "External protocol SHA256 mismatch"
    )

    protocol = read_json(
        PROTOCOL_PATH
    )

    manifest = read_json(
        MANIFEST_PATH
    )

    inspection = read_json(
        INSPECTION_PATH
    )

    require(
        protocol["configuration_frozen"] is True,
        "External protocol not frozen"
    )

    require(
        protocol["target_dataset"] == "STARC-9",
        "Unexpected target dataset"
    )

    require(
        protocol["target_images"]
        == EXPECTED_IMAGES,
        "Unexpected target image count"
    )

    require(
        protocol["label_mapping"]
        == LABEL_MAPPING,
        "Protocol label mapping mismatch"
    )

    require(
        protocol["target_class_ids"]
        == TARGET_CLASS_IDS,
        "Target class IDs mismatch"
    )

    require(
        protocol["input_size"] == [28, 28],
        "Input size mismatch"
    )

    require(
        protocol["input_color_mode"] == "RGB",
        "Input color mode mismatch"
    )

    require(
        protocol["resize_interpolation"]
        == "bilinear",
        "Resize interpolation mismatch"
    )

    require(
        protocol["resize_antialias"] is True,
        "Antialias configuration mismatch"
    )

    require(
        protocol["normalization_mean"]
        == [0.5, 0.5, 0.5],
        "Normalization mean mismatch"
    )

    require(
        protocol["normalization_std"]
        == [0.5, 0.5, 0.5],
        "Normalization std mismatch"
    )

    require(
        protocol["target_labels_used_for_adaptation"]
        is False,
        "Target-label adaptation must be disabled"
    )

    require(
        protocol["restrict_output_logits"] is False,
        "Output restriction must be disabled"
    )

    require(
        protocol["restrict_source_prototypes"]
        is False,
        "Prototype restriction must be disabled"
    )

    require(
        protocol["synthetic_corruption"] is False,
        "Unexpected synthetic corruption"
    )

    require(
        manifest["archive_integrity_verified"]
        is True,
        "Archive integrity not verified"
    )

    require(
        inspection["all_images_decoded"] is True,
        "Image inspection incomplete"
    )

    require(
        inspection["label_mapping"]
        == LABEL_MAPPING,
        "Inspection label mapping mismatch"
    )

    require(
        inspection["included_images"]
        == EXPECTED_IMAGES,
        "Inspection sample count mismatch"
    )

    require(
        protocol["dataset_revision"]
        == manifest["revision"]
        == inspection["revision"],
        "Dataset revision mismatch"
    )

    require(
        manifest["sha256"].upper()
        == EXPECTED_ARCHIVE_HASH,
        "Dataset manifest archive hash mismatch"
    )

    require(
        protocol["archive_sha256"].upper()
        == EXPECTED_ARCHIVE_HASH,
        "Protocol archive hash mismatch"
    )

    require(
        inspection["archive_sha256"].upper()
        == EXPECTED_ARCHIVE_HASH,
        "Inspection archive hash mismatch"
    )

    print("Protocol SHA256: PASS")

    for relative, expected in (
        protocol["input_hashes"].items()
    ):

        actual = sha256_file(
            ROOT / relative
        )

        require(
            actual == expected.upper(),
            f"Frozen input hash mismatch: {relative}"
        )

        print(
            f"Input hash: PASS | {relative}"
        )

    print(
        "\nChecking original ZIP SHA256...",
        flush=True,
    )

    archive_hash = sha256_file(
        ARCHIVE_PATH
    )

    require(
        archive_hash == EXPECTED_ARCHIVE_HASH,
        "Original archive SHA256 mismatch"
    )

    require(
        ARCHIVE_PATH.stat().st_size
        == manifest["size_bytes"],
        "Original archive size mismatch"
    )

    print("Archive SHA256: PASS")
    print("Frozen protocol validation: PASS")

    return protocol, manifest, inspection


# =========================================================
# PREPARE TARGET DATASET
# =========================================================

def prepare_dataset():

    images = np.empty(
        (
            EXPECTED_IMAGES,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    labels = np.empty(
        EXPECTED_IMAGES,
        dtype=np.int64,
    )

    indices = np.arange(
        EXPECTED_IMAGES,
        dtype=np.int64,
    )

    records = []

    class_counts = Counter()

    original_sizes = Counter()

    original_hash_groups = defaultdict(list)

    resized_hash_groups = defaultdict(list)

    source_paths = set()

    print("\nPREPARING STARC-9 TARGET IMAGES")
    print("-" * 75)

    with zipfile.ZipFile(
        ARCHIVE_PATH,
        "r",
    ) as archive:

        selected = []

        all_images = [
            item
            for item in archive.infolist()
            if not item.is_dir()
            and PurePosixPath(
                item.filename
            ).suffix.lower() == ".png"
        ]

        require(
            len(all_images) == 18000,
            "Unexpected total archive image count"
        )

        for item in all_images:

            parts = PurePosixPath(
                item.filename
            ).parts

            require(
                len(parts) == 3,
                f"Unexpected path: {item.filename}"
            )

            require(
                parts[0]
                == "STANFORD-CRC-HE-VAL-SMALL",
                f"Unexpected archive directory: {item.filename}"
            )

            class_name = parts[1]

            if class_name in LABEL_MAPPING:

                selected.append(item)

        selected.sort(
            key=lambda item: item.filename
        )

        require(
            len(selected) == EXPECTED_IMAGES,
            "Unexpected selected image count"
        )

        print(
            "Selected images:",
            len(selected)
        )

        print(
            "Resize:",
            "Pillow bilinear -> 28 x 28"
        )

        print(
            "Storage:",
            "uint8 RGB"
        )

        print(
            "Normalization:",
            "Deferred until model evaluation"
        )

        for index, item in enumerate(
            selected
        ):

            parts = PurePosixPath(
                item.filename
            ).parts

            class_name = parts[1]

            label = LABEL_MAPPING[
                class_name
            ]

            require(
                parts[2].startswith(
                    class_name + "_"
                ),
                "Filename-label mismatch"
            )

            require(
                item.filename not in source_paths,
                f"Duplicate source path: {item.filename}"
            )

            source_paths.add(
                item.filename
            )

            raw_bytes = archive.read(
                item
            )

            original_hash = sha256_bytes(
                raw_bytes
            )

            with archive.open(item) as file:

                with Image.open(file) as image:

                    require(
                        image.mode == "RGB",
                        f"Unexpected image mode: {item.filename}"
                    )

                    original_width = image.width

                    original_height = image.height

                    require(
                        image.size in [
                            (224, 224),
                            (256, 256),
                        ],
                        f"Unexpected image size: {item.filename}"
                    )

                    resized = image.resize(
                        (
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                        ),
                        resample=Image.Resampling.BILINEAR,
                    )

                    array = np.asarray(
                        resized,
                        dtype=np.uint8,
                    )

            require(
                array.shape
                == (IMAGE_SIZE, IMAGE_SIZE, 3),
                f"Unexpected resized shape: {item.filename}"
            )

            require(
                array.dtype == np.uint8,
                "Unexpected resized image dtype"
            )

            images[index] = array

            labels[index] = label

            resized_hash = sha256_bytes(
                array.tobytes()
            )

            original_hash_groups[
                original_hash
            ].append(index)

            resized_hash_groups[
                resized_hash
            ].append(index)

            class_counts[
                class_name
            ] += 1

            original_sizes[
                (
                    original_width,
                    original_height,
                )
            ] += 1

            records.append({
                "sample_id": index,
                "archive_path": item.filename,
                "original_class": class_name,
                "mapped_label": label,
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": IMAGE_SIZE,
                "processed_height": IMAGE_SIZE,
                "original_sha256": original_hash,
                "resized_pixel_sha256": resized_hash,
            })

            completed = index + 1

            if (
                completed % 1000 == 0
                or completed == EXPECTED_IMAGES
            ):

                print(
                    f"[{completed:5d}/{EXPECTED_IMAGES}] "
                    "Images prepared",
                    flush=True,
                )

    require(
        len(records) == EXPECTED_IMAGES,
        "Sample index count mismatch"
    )

    require(
        len(source_paths) == EXPECTED_IMAGES,
        "Non-unique source paths"
    )

    require(
        set(class_counts) == set(LABEL_MAPPING),
        "Prepared class set mismatch"
    )

    require(
        all(
            count == 2000
            for count in class_counts.values()
        ),
        "Prepared class distribution mismatch"
    )

    return (
        images,
        labels,
        indices,
        records,
        class_counts,
        original_sizes,
        original_hash_groups,
        resized_hash_groups,
    )


# =========================================================
# DUPLICATE ANALYSIS
# =========================================================

def analyze_duplicates(groups, labels):

    duplicate_groups = [
        members
        for members in groups.values()
        if len(members) > 1
    ]

    duplicate_extra_images = sum(
        len(members) - 1
        for members in duplicate_groups
    )

    duplicate_involved_images = sum(
        len(members)
        for members in duplicate_groups
    )

    cross_label_groups = []

    for members in duplicate_groups:

        unique_labels = sorted({
            int(labels[index])
            for index in members
        })

        if len(unique_labels) > 1:

            cross_label_groups.append({
                "sample_ids": members,
                "labels": unique_labels,
            })

    return {
        "unique_hashes": len(groups),
        "duplicate_groups": len(
            duplicate_groups
        ),
        "duplicate_extra_images": (
            duplicate_extra_images
        ),
        "duplicate_involved_images": (
            duplicate_involved_images
        ),
        "cross_label_duplicate_groups": len(
            cross_label_groups
        ),
        "cross_label_details": (
            cross_label_groups
        ),
    }


# =========================================================
# VALIDATE PREPARED ARRAYS
# =========================================================

def validate_arrays(
    images,
    labels,
    indices,
):

    require(
        images.shape
        == (12000, 28, 28, 3),
        "Invalid image array shape"
    )

    require(
        images.dtype == np.uint8,
        "Invalid image dtype"
    )

    require(
        labels.shape == (12000,),
        "Invalid label shape"
    )

    require(
        labels.dtype == np.int64,
        "Invalid label dtype"
    )

    require(
        indices.shape == (12000,),
        "Invalid index shape"
    )

    require(
        indices.dtype == np.int64,
        "Invalid index dtype"
    )

    require(
        np.array_equal(
            indices,
            np.arange(
                EXPECTED_IMAGES,
                dtype=np.int64,
            ),
        ),
        "Sample index sequence mismatch"
    )

    require(
        np.unique(labels).tolist()
        == TARGET_CLASS_IDS,
        "Prepared label IDs mismatch"
    )

    label_counts = Counter(
        labels.tolist()
    )

    require(
        all(
            count == 2000
            for count in label_counts.values()
        ),
        "Prepared labels are not balanced"
    )

    require(
        int(images.min()) >= 0,
        "Invalid minimum pixel value"
    )

    require(
        int(images.max()) <= 255,
        "Invalid maximum pixel value"
    )

    constant_images = int(
        np.count_nonzero(
            np.ptp(
                images.reshape(
                    EXPECTED_IMAGES,
                    -1,
                ),
                axis=1,
            ) == 0
        )
    )

    print("\nPREPARED ARRAY VALIDATION")
    print("-" * 75)

    print("Images shape:", images.shape)

    print("Images dtype:", images.dtype)

    print("Labels shape:", labels.shape)

    print("Labels dtype:", labels.dtype)

    print("Indices shape:", indices.shape)

    print(
        "Pixel range:",
        int(images.min()),
        "to",
        int(images.max()),
    )

    print(
        "Image memory (MiB):",
        round(
            images.nbytes / 1024**2,
            4,
        )
    )

    print(
        "Label IDs:",
        np.unique(labels).tolist()
    )

    print(
        "Label counts:",
        dict(sorted(label_counts.items()))
    )

    print(
        "Constant pixel images:",
        constant_images
    )

    print("Array validation: PASS")

    return constant_images


# =========================================================
# SAVE PREPARED DATASET
# =========================================================

def save_dataset(
    images,
    labels,
    indices,
    records,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_dataset = DATASET_PATH.with_suffix(
        ".npz.tmp"
    )

    with temporary_dataset.open("wb") as file:

        np.savez_compressed(
            file,
            images=images,
            labels=labels,
            indices=indices,
        )

    temporary_dataset.replace(
        DATASET_PATH
    )

    temporary_index = INDEX_PATH.with_suffix(
        ".csv.tmp"
    )

    with temporary_index.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    temporary_index.replace(
        INDEX_PATH
    )

    print("\nOUTPUT FILES")
    print("-" * 75)

    print(
        "Dataset:",
        DATASET_PATH
    )

    print(
        "Dataset size (MiB):",
        round(
            DATASET_PATH.stat().st_size
            / 1024**2,
            4,
        )
    )

    print(
        "Sample index:",
        INDEX_PATH
    )


# =========================================================
# VERIFY SAVED FILES
# =========================================================

def verify_saved_dataset(
    images,
    labels,
    indices,
    records,
):

    with np.load(
        DATASET_PATH,
        allow_pickle=False,
    ) as data:

        require(
            set(data.files)
            == {"images", "labels", "indices"},
            "Unexpected NPZ keys"
        )

        require(
            np.array_equal(
                data["images"],
                images,
            ),
            "Saved images mismatch"
        )

        require(
            np.array_equal(
                data["labels"],
                labels,
            ),
            "Saved labels mismatch"
        )

        require(
            np.array_equal(
                data["indices"],
                indices,
            ),
            "Saved indices mismatch"
        )

    with INDEX_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        saved_records = list(
            csv.DictReader(file)
        )

    require(
        len(saved_records)
        == EXPECTED_IMAGES,
        "Saved sample index count mismatch"
    )

    for index, row in enumerate(
        saved_records
    ):

        original = records[index]

        require(
            int(row["sample_id"]) == index,
            "Saved sample ID mismatch"
        )

        require(
            int(row["mapped_label"])
            == int(labels[index]),
            "Saved sample label mismatch"
        )

        require(
            row["archive_path"]
            == original["archive_path"],
            "Saved archive path mismatch"
        )

        require(
            row["original_class"]
            == original["original_class"],
            "Saved class name mismatch"
        )

        require(
            row["original_sha256"]
            == original["original_sha256"],
            "Saved original image hash mismatch"
        )

        actual_hash = sha256_bytes(
            images[index].tobytes()
        )

        require(
            row["resized_pixel_sha256"]
            == actual_hash,
            f"Saved resized image hash mismatch: {index}"
        )

    print("\nSAVED DATASET VERIFICATION")
    print("-" * 75)

    print("NPZ array integrity: PASS")

    print("CSV sample index integrity: PASS")

    print(
        "Per-image resized SHA256: "
        "12000 / 12000 PASS"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 75)

    print(
        "CAFA-TTA | STARC-9 DATASET PREPARATION"
    )

    print("=" * 75)

    print(
        "Python:",
        platform.python_version()
    )

    print("NumPy:", np.__version__)

    print("Pillow:", PIL.__version__)

    print("Device: CPU")

    (
        protocol,
        dataset_manifest,
        inspection,
    ) = validate_protocol()

    existing_outputs = [
        path
        for path in [
            DATASET_PATH,
            INDEX_PATH,
            PREPARATION_PATH,
        ]
        if path.exists()
    ]

    require(
        not existing_outputs,
        "Prepared output already exists. "
        "Refusing to overwrite existing dataset: "
        + str(existing_outputs)
    )

    (
        images,
        labels,
        indices,
        records,
        class_counts,
        original_sizes,
        original_hash_groups,
        resized_hash_groups,
    ) = prepare_dataset()

    constant_images = validate_arrays(
        images,
        labels,
        indices,
    )

    original_duplicates = analyze_duplicates(
        original_hash_groups,
        labels,
    )

    resized_duplicates = analyze_duplicates(
        resized_hash_groups,
        labels,
    )

    print("\nDUPLICATE ANALYSIS")
    print("-" * 75)

    print(
        "Original file duplicate groups:",
        original_duplicates[
            "duplicate_groups"
        ]
    )

    print(
        "Original extra duplicate images:",
        original_duplicates[
            "duplicate_extra_images"
        ]
    )

    print(
        "Original cross-label duplicate groups:",
        original_duplicates[
            "cross_label_duplicate_groups"
        ]
    )

    print(
        "Resized pixel duplicate groups:",
        resized_duplicates[
            "duplicate_groups"
        ]
    )

    print(
        "Resized extra duplicate images:",
        resized_duplicates[
            "duplicate_extra_images"
        ]
    )

    print(
        "Resized cross-label duplicate groups:",
        resized_duplicates[
            "cross_label_duplicate_groups"
        ]
    )

    print(
        "Samples removed:",
        0
    )

    save_dataset(
        images,
        labels,
        indices,
        records,
    )

    verify_saved_dataset(
        images,
        labels,
        indices,
        records,
    )

    dataset_hash = sha256_file(
        DATASET_PATH
    )

    index_hash = sha256_file(
        INDEX_PATH
    )

    script_hash = sha256_file(
        Path(__file__).resolve()
    )

    channel_mean = (
        images.mean(axis=(0, 1, 2))
        / 255.0
    )

    channel_std = (
        images.std(axis=(0, 1, 2))
        / 255.0
    )

    notes = []

    if (
        original_duplicates[
            "duplicate_groups"
        ] > 0
    ):

        notes.append(
            "Exact duplicate original PNG byte "
            "content was detected."
        )

    if (
        resized_duplicates[
            "duplicate_groups"
        ] > 0
    ):

        notes.append(
            "Exact duplicate pixel arrays were "
            "detected after resizing."
        )

    if (
        resized_duplicates[
            "cross_label_duplicate_groups"
        ] > 0
    ):

        notes.append(
            "Some identical resized pixel arrays "
            "have different mapped labels."
        )

    if constant_images > 0:

        notes.append(
            "Constant pixel images were detected."
        )

    report = {
        "project": "CAFA-TTA",
        "stage": "STARC-9 dataset preparation",

        "dataset": "STARC-9",

        "subset": dataset_manifest[
            "subset"
        ],

        "dataset_revision": dataset_manifest[
            "revision"
        ],

        "archive_sha256": (
            EXPECTED_ARCHIVE_HASH
        ),

        "protocol": relative_path(
            PROTOCOL_PATH
        ),

        "protocol_sha256": (
            EXPECTED_PROTOCOL_HASH
        ),

        "prepared_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "device": "cpu",
        },

        "total_archive_images": 18000,
        "prepared_images": EXPECTED_IMAGES,
        "excluded_incompatible_images": 6000,

        "samples_removed_after_selection": 0,

        "label_mapping": LABEL_MAPPING,

        "target_class_ids": (
            TARGET_CLASS_IDS
        ),

        "class_distribution": dict(
            sorted(class_counts.items())
        ),

        "original_image_sizes": {
            f"{width}x{height}": count
            for (width, height), count
            in sorted(original_sizes.items())
        },

        "preprocessing": {
            "color_mode": "RGB",
            "resize_backend": "Pillow",
            "resize_interpolation": "bilinear",
            "resize_antialias": True,
            "output_size": [28, 28],
            "synthetic_corruption": False,
            "normalization_applied": False,
            "normalization_deferred_to_evaluation": True,
            "evaluation_normalization_mean": [
                0.5, 0.5, 0.5
            ],
            "evaluation_normalization_std": [
                0.5, 0.5, 0.5
            ],
        },

        "arrays": {
            "images_shape": list(
                images.shape
            ),
            "images_dtype": str(
                images.dtype
            ),
            "labels_shape": list(
                labels.shape
            ),
            "labels_dtype": str(
                labels.dtype
            ),
            "indices_shape": list(
                indices.shape
            ),
            "indices_dtype": str(
                indices.dtype
            ),
            "pixel_min": int(
                images.min()
            ),
            "pixel_max": int(
                images.max()
            ),
            "images_nbytes": int(
                images.nbytes
            ),
        },

        "descriptive_pixel_statistics": {
            "channel_mean_0_to_1": (
                channel_mean.tolist()
            ),
            "channel_std_0_to_1": (
                channel_std.tolist()
            ),
            "used_for_model_normalization": False,
        },

        "constant_pixel_images": (
            constant_images
        ),

        "original_byte_duplicate_analysis": (
            original_duplicates
        ),

        "resized_pixel_duplicate_analysis": (
            resized_duplicates
        ),

        "sample_order": (
            "sorted_archive_paths"
        ),

        "input_hashes": {
            relative_path(
                MANIFEST_PATH
            ): sha256_file(
                MANIFEST_PATH
            ),

            relative_path(
                INSPECTION_PATH
            ): sha256_file(
                INSPECTION_PATH
            ),

            relative_path(
                ARCHIVE_PATH
            ): EXPECTED_ARCHIVE_HASH,

            relative_path(
                PROTOCOL_PATH
            ): EXPECTED_PROTOCOL_HASH,
        },

        "output_hashes": {
            relative_path(
                DATASET_PATH
            ): dataset_hash,

            relative_path(
                INDEX_PATH
            ): index_hash,
        },

        "preparation_script": (
            relative_path(
                Path(__file__).resolve()
            )
        ),

        "preparation_script_sha256": (
            script_hash
        ),

        "archive_integrity_verified": True,

        "prepared_array_integrity_verified": True,

        "sample_index_integrity_verified": True,

        "per_image_resized_hashes_verified": True,

        "dataset_extracted_to_disk": False,

        "configuration_frozen": True,

        "target_labels_used_for_subset_selection": True,

        "target_labels_used_for_adaptation": False,

        "model_evaluation_performed": False,

        "notes": notes,

        "status": "dataset_preparation_completed",
    }

    write_json(
        PREPARATION_PATH,
        report,
    )

    print("\n" + "=" * 75)

    print(
        "STARC-9 PREPARATION SUMMARY"
    )

    print("=" * 75)

    print("Dataset: STARC-9")

    print(
        "Prepared images:",
        len(images),
        "/ 12000",
    )

    print(
        "Target classes:",
        len(class_counts)
    )

    print(
        "Class distribution:",
        dict(sorted(class_counts.items()))
    )

    print(
        "Original image sizes:",
        dict(original_sizes)
    )

    print(
        "Images shape:",
        images.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    print(
        "Unique sample IDs:",
        len(np.unique(indices))
    )

    print(
        "Pixel range:",
        int(images.min()),
        "to",
        int(images.max()),
    )

    print(
        "Channel mean [0,1]:",
        channel_mean.tolist()
    )

    print(
        "Channel std [0,1]:",
        channel_std.tolist()
    )

    print(
        "Original duplicate groups:",
        original_duplicates[
            "duplicate_groups"
        ]
    )

    print(
        "Resized duplicate groups:",
        resized_duplicates[
            "duplicate_groups"
        ]
    )

    print(
        "Cross-label resized duplicate groups:",
        resized_duplicates[
            "cross_label_duplicate_groups"
        ]
    )

    print(
        "Samples removed after selection: 0"
    )

    print(
        "Dataset SHA256:",
        dataset_hash
    )

    print(
        "Sample index SHA256:",
        index_hash
    )

    print(
        "Protocol SHA256:",
        EXPECTED_PROTOCOL_HASH
    )

    print(
        "Script SHA256:",
        script_hash
    )

    print("Data integrity: PASS")

    print(
        "Frozen protocol integrity: PASS"
    )

    print(
        "Normalization applied: NO"
    )

    print(
        "Model evaluation performed: NO"
    )

    if notes:

        print("\nDATA QUALITY NOTES:")

        for number, note in enumerate(
            notes,
            start=1,
        ):

            print(
                f"{number}. {note}"
            )

    else:

        print(
            "Data quality notes: NONE"
        )

    print(
        "\nSaved dataset:",
        DATASET_PATH
    )

    print(
        "Saved sample index:",
        INDEX_PATH
    )

    print(
        "Saved manifest:",
        PREPARATION_PATH
    )

    print(
        "Final status: DATASET PREPARATION COMPLETED"
    )

    print("=" * 75)


if __name__ == "__main__":

    main()
