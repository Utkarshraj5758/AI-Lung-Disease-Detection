"""Robust DICOM -> PIL conversion for single chest radiographs."""

from __future__ import annotations

from io import BytesIO
from os import PathLike
from typing import BinaryIO

import numpy as np
import pydicom
from PIL import Image
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut, apply_voi_lut

from lungxray.data import minmax_to_uint8

_DECODER_HINT = "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"


def dicom_to_pil(file: str | PathLike[str] | BinaryIO | bytes) -> Image.Image:
    """Decode a DICOM file to an 8-bit PIL image ("L" for greyscale, "RGB" for colour).

    Greyscale data gets the modality LUT (rescale slope/intercept) and then the VOI
    LUT / window, is min-max scaled to uint8, and MONOCHROME1 is inverted so that dense
    tissue is bright. Only the first frame of multi-frame data is used.

    Raises:
        ValueError: not DICOM, no pixel data, or pixel data that cannot be decoded
            (e.g. a compressed transfer syntax without a decoder plugin installed).
    """
    if isinstance(file, (bytes, bytearray)):
        file = BytesIO(file)
    try:
        ds = pydicom.dcmread(file)
    except InvalidDicomError as e:
        raise ValueError("Not a valid DICOM file (no DICOM header found).") from e
    if "PixelData" not in ds:
        raise ValueError("DICOM file has no pixel data")

    try:
        arr = ds.pixel_array
    except Exception as e:
        file_meta = getattr(ds, "file_meta", None)
        syntax = file_meta.get("TransferSyntaxUID") if file_meta is not None else None
        if syntax is not None and syntax.is_compressed:
            raise ValueError(
                f"Cannot decode compressed DICOM pixel data ({syntax.name}). "
                f"Install a decoder plugin, e.g. `{_DECODER_HINT}`."
            ) from e
        raise ValueError("Could not decode the DICOM pixel data.") from e

    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        arr = arr[0]

    if int(ds.get("SamplesPerPixel", 1)) == 1:
        arr = apply_modality_lut(arr, ds)
        arr = apply_voi_lut(arr, ds)
        grey = minmax_to_uint8(arr)
        if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            grey = 255 - grey
        return Image.fromarray(grey)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unsupported DICOM pixel layout {arr.shape}")
    # pydicom converts YBR colour spaces to RGB by default.
    rgb = arr if arr.dtype == np.uint8 else minmax_to_uint8(arr)
    return Image.fromarray(rgb)
