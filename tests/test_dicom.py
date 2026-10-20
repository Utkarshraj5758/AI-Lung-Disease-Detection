from io import BytesIO

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import JPEGLSLossless, generate_uid

from lungxray.dicom import dicom_to_pil


def _gradient(rows=32, cols=48, lo=1000, hi=60000):
    return np.linspace(lo, hi, rows * cols).reshape(rows, cols).astype(np.uint16)


def test_16_bit_maps_to_full_uint8_range(make_dicom):
    img = dicom_to_pil(make_dicom(_gradient()))
    arr = np.asarray(img)
    assert img.mode == "L" and arr.dtype == np.uint8 and arr.shape == (32, 48)
    assert arr.min() == 0 and arr.max() == 255
    assert np.all(np.diff(arr.ravel().astype(int)) >= 0)  # monotone gradient preserved


def test_monochrome1_is_inverted(make_dicom):
    pixels = _gradient()
    mono2 = np.asarray(dicom_to_pil(make_dicom(pixels, "MONOCHROME2")))
    mono1 = np.asarray(dicom_to_pil(make_dicom(pixels, "MONOCHROME1")))
    np.testing.assert_array_equal(mono1, 255 - mono2)
    assert mono1[0, 0] == 255 and mono1[-1, -1] == 0


def test_multi_frame_uses_first_frame(make_dicom):
    first = _gradient()
    frames = np.stack([first, first[::-1, ::-1], np.zeros_like(first)])
    from_frames = np.asarray(dicom_to_pil(make_dicom(frames)))
    from_single = np.asarray(dicom_to_pil(make_dicom(first)))
    np.testing.assert_array_equal(from_frames, from_single)


def test_voi_window_is_applied(make_dicom):
    pixels = _gradient(lo=0, hi=4000)
    windowed = np.asarray(dicom_to_pil(make_dicom(pixels, WindowCenter=2000, WindowWidth=200)))
    # Values outside the narrow window saturate, so most pixels are 0 or 255.
    assert np.mean((windowed == 0) | (windowed == 255)) > 0.9


def test_modality_lut_is_applied(make_dicom):
    pixels = _gradient()
    flipped = np.asarray(dicom_to_pil(make_dicom(pixels, RescaleSlope=-1, RescaleIntercept=0)))
    assert flipped[0, 0] == 255 and flipped[-1, -1] == 0


def test_compressed_without_decoder_raises_clear_error():
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = JPEGLSLossless
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    ds.SOPInstanceUID = generate_uid()
    ds.Rows = ds.Columns = 8
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation, ds.SamplesPerPixel = 0, 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = encapsulate([b"\xff\xd8\xff\xf7not really jpeg-ls"])
    ds["PixelData"].VR = "OB"
    buffer = BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    with pytest.raises(ValueError, match="pylibjpeg"):
        dicom_to_pil(buffer.getvalue())


def test_not_dicom_and_no_pixel_data_raise(make_dicom):
    with pytest.raises(ValueError, match="Not a valid DICOM"):
        dicom_to_pil(b"definitely not a dicom file")
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    ds.SOPInstanceUID = generate_uid()
    buffer = BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    with pytest.raises(ValueError, match="no pixel data"):
        dicom_to_pil(buffer.getvalue())
