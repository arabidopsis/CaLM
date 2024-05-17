from numpy.random import default_rng, Generator


_RNG = default_rng()


def set_random_seed(seed: int):
    global _RNG  # pylint: disable=global-statement
    _RNG = default_rng(seed)


def get_rng() -> Generator:
    return _RNG
