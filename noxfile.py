import nox


@nox.session()
def run_pytest(session):
    session.install("-r", "requirements_dev.txt")
    session.install("-e", ".")
    session.run("pytest")


@nox.session()
def test_linting(session):
    session.install("flake8")
    session.run("flake8", "src")
    session.run("flake8", "tests")


@nox.session()
def run_type_checker(session):
    session.install("mypy")
    session.run("mypy", "src")
