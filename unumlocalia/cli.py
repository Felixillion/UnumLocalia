import argparse

from unumlocalia import __version__
from unumlocalia.widgets import launch


def main():

    parser = argparse.ArgumentParser(
        prog="unumlocalia"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    parser.parse_args()

    launch()