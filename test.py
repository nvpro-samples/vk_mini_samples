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
    print("Running CMake with the following commands:", commands)
    try:
        subprocess.run(commands, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running CMake: {e}")


def build(shader_language):
    """Build the project with the specified shader language."""
    header("BUILDING PROJECT")

    # Create build and install directories if they do not exist
    create_directory_if_not_exists(build_dir)
    create_directory_if_not_exists(install_dir)

    # Change to the build directory
    os.chdir(build_dir)

    # Run CMake to generate build files
    cmake_args = ["cmake", ".."]

    if shader_language == "GLSL":
        cmake_args += ["-DUSE_SHADER_LANGUAGE=GLSL"]
    elif shader_language == "HLSL":
        cmake_args += ["-DUSE_SHADER_LANGUAGE=HLSL"]
    elif shader_language == "SLANG":
        cmake_args += ["-DUSE_SHADER_LANGUAGE=SLANG"]
    else:
        print("Invalid shader language specified.")
        return

    run_cmake(cmake_args)

    # Call cmake to build the release config in parallel
    run_cmake(build_commands)

    # Change back to the original directory
    os.chdir("..")

    # Remove all .exe files in the bin_x64 directory
    bin_x64_dir = os.path.join(install_dir, "bin_x64/")
    if os.path.exists(bin_x64_dir):
        for file in os.listdir(bin_x64_dir):
            if file.endswith(".exe"):
                os.remove(os.path.join(bin_x64_dir, file))

    # Call cmake to install
    run_cmake(["cmake", "--install", "build", "--prefix", install_dir])


def extract_testing_time(log_file):
    """Extract testing time from the log file."""
    testing_time = None
    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "Testing Time" in line:
                    testing_time = line.split(":")[1].strip()
                    break
    if testing_time is None:
        testing_time = "N/A"
    return testing_time


def test(shader_language):
    """Run tests on the built executables."""
    print(f"Executing test function with shader language: {shader_language}")
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
        if os.path.isfile(os.path.join(".", f))
        and (f.endswith(".exe") or f.endswith("_app"))
    ]

    # Initialize a list to store test results
    test_results = []

    # Call each executable with options --test and --screenshot
    for executable in executables:
        executable_path = os.path.join(".", executable)
        args = [executable_path]

        if not any(name in executable for name in names_to_avoid):
            log_file = f"log_{executable[:-4]}_{shader_language}.txt"
            image_file = f"snap_{executable[:-4]}_{shader_language}.jpg"

            args += ["-test"]
            args += test_arguments
            args += ["-screenshot", image_file]
            args += ["-logfile", log_file]
            try:
                header(
                    f"Testing '{executable}' with shader language: {shader_language}"
                )
                subprocess.run(args, check=True)
                testing_time = extract_testing_time(log_file)
                test_results.append(
                    (executable, 0, testing_time)
                )  # Success, so error code is 0

            except subprocess.CalledProcessError as e:
                print(f"Error occurred while testing: {e}")
                returncode = e.returncode
                test_results.append((executable, returncode, "N/A"))

                # Ignore errors from ray_query_position_fetch in CI
                if (
                    ("CI" in os.environ)
                    and ("ray_query_position_fetch" in executable)
                    and datetime.datetime.now().date()
                    < datetime.date(year=2024, month=2, day=1)
                ):
                    print("Ignored")
                    returncode = 0
                    test_results[-1] = (
                        executable,
                        returncode,
                        "N/A",
                    )  # Update the test result

    os.chdir(current_dir)
    if any(error_code != 0 for _, error_code, time in test_results):
        sys.exit(1)

    return test_results


def finalReport(shader_language, test_results):
    print("\nFinal Report for", shader_language, ":")
    print("-" * 60)
    print("{:<30} | {:<12} | {:<8}".format("Executable", "Error Code", "Time"))
    print("-" * 60)
    for test_result in test_results:
        executable, error_code, testing_time = test_result
        print("{:<30} | {:<12} | {:<8}".format(executable, error_code, testing_time))
    print("-" * 60)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the command line options
    parser.add_argument("--test", action="store_true", help="Execute test function")
    parser.add_argument(
        "--build",
        nargs="?",
        default=None,
        choices=["GLSL", "HLSL", "SLANG"],
        help="Execute build function with optional shader language",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    shading_language = "GLSL"
    if args.build:
      if args.build in ["GLSL", "HLSL", "SLANG"]:
          shading_language = args.build
          build(shading_language)  # Build with specified shader language
      else:
          print("Invalid shader language specified.")
          sys.exit(1)

    if args.test:
        test_results = test(shading_language)  # Test with specified shader language
        finalReport(shading_language, test_results)
      
    #    # If no argument is provided, call build with all shader languages
    #    for language in ["GLSL", "HLSL", "SLANG"]:
    #        if args.build:
    #            build(language)
    #        test_results_all_languages.append(test(language))
    #    # Print the final report for all shader languages
    #    print("\nFinal Report for all shader languages:")
    #    for shader_language, test_results in zip(
    #        ["GLSL", "HLSL", "SLANG"], test_results_all_languages
    #    ):
    #        finalReport(shader_language, test_results)
