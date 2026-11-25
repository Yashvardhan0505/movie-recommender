from recsys.split import temporal_split


def test_periods_do_not_overlap(dataset):
    val, test = temporal_split(dataset.ratings, 0.8, 0.9)
    assert val.cutoff < test.cutoff
    assert val.history["ts"].max() < val.cutoff <= val.target["ts"].min()
    assert val.target["ts"].max() < test.cutoff <= test.target["ts"].min()
    assert test.history["ts"].max() < test.cutoff
    assert len(val.history) + len(val.target) == len(test.history)
    assert len(test.history) + len(test.target) == len(dataset.ratings)


def test_warm_and_new_users(dataset):
    _, test = temporal_split(dataset.ratings)
    warm, new = set(test.warm_users()), set(test.new_users())
    seen_before = set(test.history["user"])
    assert warm and new
    assert warm <= seen_before
    assert not new & seen_before
    assert warm | new == set(test.target["user"])
