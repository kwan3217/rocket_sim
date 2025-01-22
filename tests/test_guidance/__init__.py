"""
Describe purpose of this script here

Created: 1/22/25
"""
# __init__.py in the directory you want to skip
import pytest
pytestmark = pytest.mark.skip(reason="Skipping all tests in this directory")

def main():
    pass


if __name__ == "__main__":
    main()
