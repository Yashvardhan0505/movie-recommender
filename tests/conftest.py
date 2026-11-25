import pytest

from recsys import data
from recsys.synthetic import write_synthetic


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    folder = write_synthetic(tmp_path_factory.mktemp("ml") / "ml-1m", n_users=1200, n_items=600, seed=0)
    return data.load(folder)
