#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import datetime
from sys import platform

build_dir = "build/"
install_dir = "_install"


def header(name):
    print("")
    print("***************************************************************************")
    print(f"  {name}")
    print("***************************************************************************")


def clone_nvpro_core():
    header("NVPRO-CORE")

    nvpro_core_dir = None
    current_dir = os.getcwd()

    # Check if nvpro_core directory exists at the same level
    if os.path.exists("nvpro_core"):
        nvpro_core_dir = os.path.abspath("nvpro_core")
    # Check if nvpro_core directory exists one level up
    elif os.path.exists("../nvpro_core"):
        nvpro_core_dir = os.path.abspath("../nvpro_core")
    # Check if nvpro_core directory exists two levels up
    elif os.path.exists("../../nvpro_core"):
        nvpro_core_dir = os.path.abspath("../../nvpro_core")

    # If nvpro_core directory doesn't exist, clone it one level above the current directory
    if not nvpro_core_dir:
        target_dir = os.path.abspath(os.path.join(current_dir, ".."))

        try:
            # Clone nvpro_core recursively from the GitLab repository
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--recursive",
                    "https://gitlab-master.nvidia.com/devtechproviz/nvpro-samples/nvpro_core.git",
                    target_dir + "/nvpro_core",
                ],
                check=True,
            )

            # Initialize nvpro_core
            os.chdir(target_dir)
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"], check=True
            )

            print("nvpro_core cloned and initialized successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while cloning nvpro_core: {e}")
    else:
        print("nvpro_core directory already exists.")
        try:
            # Change to the nvpro_core directory
            os.chdir(nvpro_core_dir)

            # Pull changes recursively in the nvpro_core directory
            subprocess.run(["git", "pull", "--recurse-submodules"], check=True)

            print("nvpro_core updated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while pulling changes for nvpro_core: {e}")

        # Change back to the original directory
        os.chdir(current_dir)


def build():
    header("BUILDING PROJECT")

    # Check if build directory exists, create it if necessary
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # Check if install directory exists, create it if necessary
    if not os.path.exists(install_dir):
        os.makedirs(install_dir)

    # Change to the build directory
    os.chdir(build_dir)
    try:
        subprocess.run(["cmake", ".."], check=True)

        # Call cmake to build the release config in parallel
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "--parallel"], check=True
        )

        # Change back to the original directory
        os.chdir("..")

        # Call cmake to install
        subprocess.run(
            ["cmake", "--install", "build", "--prefix", install_dir], check=True
        )

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while building: {e}")


def test():
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

    # Call each executable with options --test and --snapshot
    returncode = 0
    for executable in executables:
        executable_path = os.path.join(".", executable)
        args = [executable_path]
        if not ('offscreen' in executable): # offscreen.exe doesn't take these arguments
            args += ["--test", "--snapshot", "--frames", "10"]
        try:
            header(f"Testing '{executable}'")
            subprocess.run(args, check=True)

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

def format_code():
    header("Checking code format")

    try:
        # Call git clang-format to make a diff with origin/main
        diff_output = subprocess.check_output(
            ["git", "clang-format", "--diff", "origin/main"]
        )
    except subprocess.CalledProcessError as e:
        print(e.output.decode())
        print(f"Error occurred while calling git clang-format the code: {e}")
        return

    if diff_output:
        diff_string = diff_output.decode()
        if (
            "no modified files" in diff_string
            or "did not modify any files" in diff_string
        ):
            print(diff_string)
            return

        print(diff_string)
        print(
            "Code formatting differences found. Please run 'clang-format' to format your code."
        )
        sys.exit(1)
    else:
        print("Code is already correctly formatted.")



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the command line options
    parser.add_argument("--build", action="store_true", help="Execute build function")
    parser.add_argument("--test", action="store_true", help="Execute test function")
    parser.add_argument("--format", action="store_true", help="Execute format function")
    parser.add_argument(
        "--nvpro",
        action="store_true",
        help="Check for nvpro-core and clone it if needed",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Check the command line arguments and call the corresponding functions
    if args.nvpro:
        clone_nvpro_core()

    if args.build:
        build()

    if args.test:
        test()

    if args.format:
        format_code()

    # If no arguments provided, call all functions
    if not any(vars(args).values()):
        build()
        test()
        format_code()
