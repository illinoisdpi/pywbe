from pyWBE import hello
from pyWBE.first_submodule import hello_first_sm


def test_hello_world():
    assert hello.hello_world() is None


def test_hello_first_sm():
    assert hello_first_sm.hello_firstsm() is None
