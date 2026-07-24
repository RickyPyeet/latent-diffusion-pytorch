from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile 

from tqdm.auto import tqdm

def download_file(url: str,
                  destination: str | Path,
                  overwrite: bool = True,
                  chunk_size: int = 1024 * 1024):
    if chunk_size <= 0:
        raise ValueError(f"Chunk size must be greater than 0")

    destination = Path(destination)
    destination.parent.mkdir(parents = True, exist_ok = True)

    if destination.exists() and not overwrite:
        existing_size = destination.stat().st_size
    else:
        existing_size = 0

    headers = {}

    if existing_size > 0:
        headers['Range'] = f"bytes={existing_size}-"
    
    request = Request(url, headers = {**headers, "User-Agent": 'latent_diffusion_pytorch'})

    try:
        response = urlopen(request)
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Failed to load {url}: {error}") from error

    status_code = getattr(response, "status", None)

    # HTTP 206 means the server accepted the Range request.
    resumed = existing_size > 0 and status_code == 206

    if existing_size > 0 and not resumed:
        existing_size = 0

    content_length = response.headers.get("Content-Length")
    remaining_size = int(content_length) if content_length is not None else None

    total_size = (existing_size + remaining_size if remaining_size is not None else None)

    file_mode = "ab" if resumed else "wb"

    try:
        with (
            destination.open(file_mode) as file,
            tqdm(
                total=total_size,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as progress_bar,
        ):
            while True:
                chunk = response.read(chunk_size)

                if not chunk:
                    break

                file.write(chunk)
                progress_bar.update(len(chunk))

    except OSError as error:
        raise RuntimeError(f"Failed to write downloaded file to {destination}: {error}") from error
    finally:
        response.close()

    if total_size is not None:
        downloaded_size = destination.stat().st_size

        if downloaded_size != total_size:
            raise RuntimeError(f"Incomplete download for {destination}. Expected {total_size} bytes, found {downloaded_size}")

    return destination


def extract_zip(archive_path: str | Path,
                output_dir: str | Path,
                overwrite: bool = False) -> Path:
    """
    Extract a ZIP archive into an output directory.
    Existing extracted files are preserved unless overwrite is enabled.
    Args:
        archive_path: Path to the ZIP archive.
        output_dir: Directory where archive contents are extracted.
        overwrite: Whether existing extracted files should be replaced.
    Out:
        Extraction directory.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(archive_path, "r") as archive:
            members = archive.infolist()

            for member in tqdm(
                members,
                desc=f"Extracting {archive_path.name}",
                unit="file",
            ):
                destination = output_dir / member.filename

                if destination.exists() and not overwrite:
                    continue

                archive.extract(member, output_dir)

    except BadZipFile as error:
        raise RuntimeError(f"Invalid ZIP archive: {archive_path}") from error
    except OSError as error:
        raise RuntimeError(f"Failed to extract {archive_path}: {error}") from error

    return output_dir


def verify_paths(root: str | Path, required_paths: list[str | Path]) -> None:
    """
    Verify that required files and directories exist below a root directory.
    Args:
        root: Root directory containing the expected paths.
        required_paths: Paths expected relative to root.
    """
    root = Path(root)

    missing_paths = [root / path for path in required_paths if not (root / path).exists()]

    if missing_paths:
        formatted_paths = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Dataset verification failed. Missing paths:\n {formatted_paths}")


def remove_file(path: str | Path, missing_ok: bool = True) -> None:
    """
    Remove a file from disk.
    Args:
        path: File to remove
        missing_ok: Whether a missing file should be ignored
    """
    path = Path(path)

    if not path.exists():
        if missing_ok:
            return

        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Expected a file, found directory: {path}")

    path.unlink()