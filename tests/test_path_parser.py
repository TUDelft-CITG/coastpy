from pathlib import Path

import pytest

from coastpy.io.utils import PathParser


def test_local_filepath():
    fp = "/Users/calkoen/data/tmp/typology/train/release/file.tif"
    pp = PathParser(fp)

    assert pp.protocol == "file"
    assert pp.to_filepath() == Path(fp)
    assert pp.filename == "file.tif"
    assert pp.bucket == ""  # Should be empty for local paths
    assert pp.key == ""  # Should be empty for local paths

    # Changing protocol to cloud should raise an error
    with pytest.raises(ValueError):  # noqa: PT011
        pp.to_cloud_uri()


def test_azure_cloud_uri():
    uri = "az://tmp/s2-l2a-composite/release/2025-01-17/file.tif"
    pp = PathParser(uri)
    pp.to_cloud_uri()

    assert pp.protocol == "az"
    assert pp.bucket == "tmp"
    assert pp.key == "s2-l2a-composite/release/2025-01-17"
    assert pp.filename == "file.tif"
    assert pp.to_cloud_uri() == uri

    # Changing to HTTPS should fail without account_name
    with pytest.raises(ValueError):  # noqa: PT011
        pp.to_https_url()

    # Changing to HTTPS should succeed with account_name
    assert pp.to_https_url(account_name="coclico") == (
        "https://coclico.blob.core.windows.net/tmp/s2-l2a-composite/release/2025-01-17/file.tif"
    )


def test_azure_cloud_uri_only_filename():
    uri = "az://tmp/file.tif"
    pp = PathParser(uri)

    assert pp.protocol == "az"
    assert pp.bucket == "tmp"
    assert pp.key == ""  # No directory structure
    assert pp.filename == "file.tif"
    assert pp.to_cloud_uri() == uri

    # Ensure HTTPS conversion works correctly
    assert pp.to_https_url(account_name="coclico") == (
        "https://coclico.blob.core.windows.net/tmp/file.tif"
    )


def test_https_url_parsing():
    url = "https://coclico.blob.core.windows.net/tmp/s2-l2a-composite/release/file.tif"
    pp = PathParser(url)

    assert pp.protocol == "https"
    assert pp.account_name == "coclico"
    assert pp.bucket == "tmp"
    assert pp.key == "s2-l2a-composite/release"
    assert pp.filename == "file.tif"
    assert pp.to_https_url() == url


def test_dynamic_filename():
    uri = "az://tmp/s2-l2a-composite/release/file.tif"
    pp = PathParser(uri)

    assert pp.filename == "file.tif"

    # Adding band and resolution should update filename dynamically
    pp.band = "nir"
    assert pp.filename == "file_nir.tif"

    pp.resolution = "10m"
    assert pp.filename == "file_nir_10m.tif"

    # Ensure cloud URI and HTTPS URL reflect the new filename
    assert pp.to_cloud_uri() == "az://tmp/s2-l2a-composite/release/file_nir_10m.tif"
    assert pp.to_https_url(account_name="coclico") == (
        "https://coclico.blob.core.windows.net/tmp/s2-l2a-composite/release/file_nir_10m.tif"
    )


def test_invalid_paths():
    # Unsupported protocol should raise error
    with pytest.raises(ValueError):  # noqa: PT011
        PathParser("ftp://example.com/file.tif")


def test_protocol_change():
    fp = "/Users/calkoen/data/tmp/file.tif"
    pp = PathParser(fp)

    assert pp.protocol == "file"
    assert pp.to_filepath() == Path(fp)

    # Switching to cloud protocol
    pp.protocol = "az"
    pp.bucket = "tmp"
    pp.key = "data"

    assert pp.to_cloud_uri() == "az://tmp/data/file.tif"

    # Switching to HTTPS should fail without account_name
    with pytest.raises(ValueError):  # noqa: PT011
        pp.to_https_url()

    # With account_name it should work
    assert pp.to_https_url(account_name="coclico") == (
        "https://coclico.blob.core.windows.net/tmp/data/file.tif"
    )
