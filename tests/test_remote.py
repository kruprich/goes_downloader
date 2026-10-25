"""The bucket: object URLs and listing on both clouds, against canned S3 XML and GCS JSON pages."""

import asyncio
import json
import os

import pytest

from conftest import REPO
from goes_downloader import remote
from goes_downloader.config import load_config
from goes_downloader.day import infer_satellite_label_from_bucket

PREFIX = "GLM-L2-LCFA/2025/300/18/"
NAMES = [f"{PREFIX}OR_GLM-L2-LCFA_G19_s2025300180{i}000_e2025300180{i}200_c2025300180{i}220.nc" for i in range(5)]


def load(tmp_path, bucket):
    raw = json.load(open(os.path.join(REPO, "config.json")))
    raw["bucket"] = bucket
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return load_config(str(path))


def s3_page(names, token=None):
    contents = "".join(f"<Contents><Key>{n}</Key><Size>{1000 + len(n)}</Size><StorageClass>STANDARD</StorageClass></Contents>" for n in names)
    more = f"<IsTruncated>true</IsTruncated><NextContinuationToken>{token}</NextContinuationToken>" if token else "<IsTruncated>false</IsTruncated>"
    return (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Name>noaa-goes19</Name><Prefix>{PREFIX}</Prefix>{more}{contents}</ListBucketResult>").encode()


def gcs_page(names, token=None):
    page = {"items": [{"name": n, "size": str(1000 + len(n))} for n in names]}
    return {**page, "nextPageToken": token} if token else page


@pytest.mark.parametrize("bucket,url", [
    ("s3://noaa-goes19", "https://noaa-goes19.s3.amazonaws.com/ABI-L2-MCMIPC/2025/300/18/f.nc"),
    ("gs://gcp-public-data-goes-19", "https://storage.googleapis.com/gcp-public-data-goes-19/ABI-L2-MCMIPC/2025/300/18/f.nc"),
])
def test_object_url(bucket, url):
    assert remote.object_url(bucket, "ABI-L2-MCMIPC/2025/300/18/f.nc") == url


@pytest.mark.parametrize("bucket,expected", [
    ("s3://noaa-goes19", "s3://noaa-goes19"),
    ("gs://gcp-public-data-goes-19/", "gs://gcp-public-data-goes-19"),
])
def test_bucket_is_a_url(tmp_path, bucket, expected):
    cfg = load(tmp_path, bucket)
    assert cfg.bucket == expected
    assert infer_satellite_label_from_bucket(cfg.bucket) == "GOES-19"


@pytest.mark.parametrize("bucket", ["noaa-goes19", "https://noaa-goes19.s3.amazonaws.com", "s3://noaa-goes19/ABI-L2-MCMIPC", "az://goes"])
def test_bad_buckets_are_rejected(tmp_path, bucket):
    with pytest.raises(ValueError, match="bucket must be"):
        load(tmp_path, bucket)


@pytest.mark.parametrize("cloud", ["s3", "gs"])
def test_listing_pages_through_the_prefix(tmp_path, monkeypatch, cloud):
    """Every page is read, each with the token the previous one returned; sizes come back as ints."""
    cfg = load(tmp_path, "s3://noaa-goes19" if cloud == "s3" else "gs://gcp-public-data-goes-19")
    pages = {None: (NAMES[:2], "tok/1+="), "tok/1+=": (NAMES[2:4], "tok2"), "tok2": (NAMES[4:], None)}
    requests = []

    async def fake_get(cfg, session, url, params=None, *, as_json):
        requests.append((url, dict(params)))
        token = params.get("continuation-token" if cloud == "s3" else "pageToken")
        names, next_token = pages[token]
        return s3_page(names, next_token) if cloud == "s3" else gcs_page(names, next_token)

    monkeypatch.setattr(remote, "http_get", fake_get)
    objects = asyncio.run(remote.list_object_names(cfg, None, PREFIX))
    assert objects == {n: 1000 + len(n) for n in NAMES}
    assert len(requests) == 3 and all(p["prefix"] == PREFIX for _, p in requests)
    if cloud == "s3":
        assert {url for url, _ in requests} == {"https://noaa-goes19.s3.amazonaws.com/"}
        assert all(p["list-type"] == "2" and p["max-keys"] == "1000" for _, p in requests)
    else:
        assert {url for url, _ in requests} == {"https://storage.googleapis.com/storage/v1/b/gcp-public-data-goes-19/o"}
