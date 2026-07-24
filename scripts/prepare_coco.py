import argparse
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from src.latent_diffusion.data.download_utils import download_file, extract_zip, remove_file, verify_paths


COCO_ARCHIVES = {
    "train2017.zip": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "required_paths": [
            "train2017",
        ],
    },
    "val2017.zip": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "required_paths": [
            "val2017",
        ],
    },
    "annotations_trainval2017.zip": {
        "url": (
            "http://images.cocodataset.org/annotations/"
            "annotations_trainval2017.zip"
        ),
        "required_paths": [
            "annotations/captions_train2017.json",
            "annotations/captions_val2017.json",
        ],
    },
}


REQUIRED_COCO_PATHS = [
    "train2017",
    "val2017",
    "annotations",
    "annotations/captions_train2017.json",
    "annotations/captions_val2017.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the COCO 2017 dataset."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/coco"),
        help="Directory where COCO 2017 will be prepared.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIP archives after extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload archives and overwrite extracted files.",
    )

    return parser.parse_args()


def paths_exist(
    root: Path,
    relative_paths: list[str],
) -> bool:
    return all((root / path).exists() for path in relative_paths)


def is_valid_zip(path: Path) -> bool:
    """
    Check whether a path contains a complete, readable ZIP archive.
    """
    if not path.is_file():
        return False

    try:
        with ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except (BadZipFile, OSError):
        return False


def prepare_archive(
    archive_name: str,
    url: str,
    required_paths: list[str],
    output_dir: Path,
    archive_dir: Path,
    overwrite: bool,
) -> Path:
    """
    Download and extract one COCO archive when necessary.
    """
    archive_path = archive_dir / archive_name

    if not overwrite and paths_exist(output_dir, required_paths):
        print(f"✓ {archive_name} already extracted")
        return archive_path

    if overwrite or not is_valid_zip(archive_path):
        print(f"Downloading {archive_name}...")

        download_file(
            url=url,
            destination=archive_path,
            overwrite=overwrite,
        )
    else:
        print(f"✓ {archive_name} already downloaded")

    print(f"Extracting {archive_name}...")

    extract_zip(
        archive_path=archive_path,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    if not paths_exist(output_dir, required_paths):
        formatted_paths = "\n".join(
            f"  - {output_dir / path}"
            for path in required_paths
            if not (output_dir / path).exists()
        )

        raise RuntimeError(
            f"Extraction of {archive_name} did not produce all "
            f"expected paths:\n{formatted_paths}"
        )

    print(f"✓ {archive_name} extracted")

    return archive_path


def prepare_coco(
    output_dir: str | Path,
    keep_archives: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Download, extract, and verify COCO 2017 train/validation data.

    Args:
        output_dir:
            Destination directory for the prepared COCO dataset.
        keep_archives:
            Whether downloaded ZIP archives should be preserved.
        overwrite:
            Whether archives and extracted files should be replaced.
    """
    output_dir = Path(output_dir)
    archive_dir = output_dir / "archives"

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing COCO 2017 in {output_dir.resolve()}\n")

    if not overwrite and paths_exist(output_dir, REQUIRED_COCO_PATHS):
        print("✓ COCO 2017 is already prepared")
        return

    archive_paths = []

    for archive_name, archive_config in COCO_ARCHIVES.items():
        archive_path = prepare_archive(
            archive_name=archive_name,
            url=archive_config["url"],
            required_paths=archive_config["required_paths"],
            output_dir=output_dir,
            archive_dir=archive_dir,
            overwrite=overwrite,
        )

        archive_paths.append(archive_path)

    print("\nVerifying COCO dataset...")

    verify_paths(
        root=output_dir,
        required_paths=REQUIRED_COCO_PATHS,
    )

    print("✓ Dataset verification passed")

    if not keep_archives:
        print("\nRemoving downloaded archives...")

        for archive_path in archive_paths:
            remove_file(archive_path, missing_ok=True)

        try:
            archive_dir.rmdir()
        except OSError:
            pass

        print("✓ Archives removed")

    print("\nCOCO 2017 is ready.")


def main() -> None:
    args = parse_args()

    prepare_coco(
        output_dir=args.output_dir,
        keep_archives=args.keep_archives,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()