#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import datetime
from sys import platform

# Global variables
build_dir = "build/"
install_dir = "_install/"
build_commands = ["cmake", "--build", ".", "--config", "Release", "--parallel"]
test_arguments = ["-test-frames", "10"]
names_to_avoid = ["gpu_monitor", "offscreen"]

def header(name):
    """Print a header with a given name."""
    print("")
    print("***************************************************************************")
    print(f"  {name}")
    print("***************************************************************************")

def create_directory_if_not_exists(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_cmake(commands):
    """Run CMake commands."""
    try:
        subprocess.run(commands, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running CMake: {e}")

def build():
    """Build the project."""
    header("BUILDING PROJECT")

    # Create build and install directories if they do not exist
    create_directory_if_not_exists(build_dir)
    create_directory_if_not_exists(install_dir)

    # Change to the build directory
    os.chdir(build_dir)

    # Run CMake to generate build files
    run_cmake(["cmake", ".."])

    # Call cmake to build the release config in parallel
    run_cmake(build_commands)

    # Change back to the original directory
    os.chdir("..")

    # Call cmake to install
    run_cmake(["cmake", "--install", "build", "--prefix", install_dir])

import os

def print_log_result(log_file):
    # Print lines containing "Testing Time" in the log file
    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            lines = file.readlines()
            print("Results:")
            for line in lines:
                if "Testing Time" in line:
                    print(line.strip())


def test():
    """Run tests on the built executables."""
    print("Executing test function")
    test_dir = os.path.join(install_dir, "bin_x64/")

    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' does not exist.")
        return

    current_dir = os.getcwd()
    os.chdir(test_dir)

    # Find all executables in the test directory
    executables = [
        f
        for f in os.listdir(".")
        if os.path.isfile(os.path.join(".", f)) and (f.endswith(".exe") or f.endswith("_app"))
    ]

    # Call each executable with options --test and --screenshot
    returncode = 0
    for executable in executables:
        executable_path = os.path.join(".", executable)
        args = [executable_path]

        if not any(name in executable for name in names_to_avoid):
            args += ["-test"]
            args += test_arguments
            args += ["-screenshot", "snap_" + executable[:-4] + ".jpg"]
            try:
                header(f"Testing '{executable}'")
                subprocess.run(args, check=True)
                log_file = "log_" + executable[:-4] + ".txt"
                print_log_result(log_file)

            except subprocess.CalledProcessError as e:
                print(f"Error occurred while testing: {e}")
                returncode = e.returncode
                # Ignore errors from ray_query_position_fetch in CI
                if ("CI" in os.environ) and ('ray_query_position_fetch' in executable) and datetime.datetime.now().date() < datetime.date(year=2024, month=2, day=1):
                    print("Ignored")
                    returncode = 0

    os.chdir(current_dir)
    if returncode != 0:
        sys.exit(returncode)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the command line options
    parser.add_argument("--build", action="store_true", help="Execute build function")
    parser.add_argument("--test", action="store_true", help="Execute test function")

    # Parse the command line arguments
    args = parser.parse_args()

    if args.build:
        build()

    if args.test:
        test()

    # If no arguments provided, call all functions
    if not any(vars(args).values()):
        build()
        test()
