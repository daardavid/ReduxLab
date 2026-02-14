#!/usr/bin/env python3
"""
Test runner script for PCA project.
Runs all unit tests and generates coverage report.
"""

import subprocess
import sys
import os


def run_tests():
    """Run all tests with coverage."""
    print("Running PCA project tests...")

    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("pytest not found. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"]
        )

    # Check if coverage is available
    try:
        import coverage
    except ImportError:
        print("coverage not found. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "coverage", "pytest-cov"]
        )

    # Run tests with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--verbose",
        "--tb=short",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=20",
    ]

    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_specific_test(test_file):
    """Run a specific test file."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        f"tests/{test_file}",
        "--verbose",
        "--tb=short",
    ]
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode
    except Exception as e:
        print(f"Error running test {test_file}: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        exit_code = run_specific_test(test_file)
    else:
        exit_code = run_tests()
    sys.exit(exit_code)
