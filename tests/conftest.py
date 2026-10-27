import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from tests.phantom import write_case
from src.config import PipelineConfig
from src.pipeline import NeuroScanX


@pytest.fixture(scope="session")
def case_paths(tmp_path_factory):
    return write_case(str(tmp_path_factory.mktemp("case")), shape=(48, 56, 40), noise=0.05)


@pytest.fixture(scope="session")
def cfg():
    c = PipelineConfig()
    c.xai.use_llm = "off"
    return c


@pytest.fixture(scope="session")
def prepared(case_paths, cfg):
    nsx = NeuroScanX(cfg)
    mri, rep = nsx.load(*(case_paths[k] for k in ("T1", "T1ce", "T2", "FLAIR")))
    assert rep.ok
    return nsx, nsx.prepare(mri, rep, case_paths["GT"])


@pytest.fixture(scope="session")
def result(prepared):
    nsx, prep = prepared
    return nsx.analyze(prep, "edge_spatial_fcm")
