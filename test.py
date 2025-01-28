import os
import sys
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum, auto
import logging


class ShaderLanguage(Enum):
    GLSL = "GLSL"
    HLSL = "HLSL"
    SLANG = "SLANG"

    def __str__(self):
        return self.value


class ReturnCode(Enum):
    SUCCESS = 0
    BUILD_ERROR = 1
    TEST_ERROR = 2
    ENVIRONMENT_ERROR = 3


# Constants
BUILD_DIR = Path("build")
INSTALL_DIR = Path("_install")
BUILD_COMMANDS = ["cmake", "--build", ".", "--config", "Release", "--parallel"]
TEST_ARGUMENTS = ["-test-frames", "10"]
NAMES_TO_AVOID = ["gpu_monitor", "offscreen"]


# Define the executables and their arguments explicitly
EXECUTABLES_WITH_ARGS = [
    ("vk_mini_barycentric_wireframe", ["--headless"]),
    ("vk_mini_compute_multi_threaded", ["--headless"]),
    ("vk_mini_compute_only", ["--headless"]),
    ("vk_mini_gltf_raytrace", ["--headless", "--frames", "100"]),
    ("vk_mini_image_ktx", ["--headless"]),
    ("vk_mini_image_viewer", ["--headless"]),
    ("vk_mini_line_stipple", ["--headless"]),
    ("vk_mini_memory_budget", ["--headless", "--frames", "1000"]),
    ("vk_mini_mm_opacity", ["--headless"]),
    ("vk_mini_msaa", ["--headless"]),
    ("vk_mini_offscreen", ["--size", "480", "360"]),
    ("vk_mini_ray_query", ["--headless", "--frames", "100"]),
    ("vk_mini_ray_query_position_fetch", ["--headless"]),
    ("vk_mini_realtime_analysis",["--headless", "--frames", "100", "--winSize", "800", "600"],),
    ("vk_mini_rectangle", ["--headless"]),
    ("vk_mini_ser_pathtrace", ["--headless", "--frames", "100"]),
    ("vk_mini_shader_object", ["--headless"]),
    ("vk_mini_shader_printf", ["--headless"]),
    ("vk_mini_simple_polygons", ["--headless"]),
    ("vk_mini_solid_color", ["--headless"]),
    ("vk_mini_texture_3d", ["--headless"]),
    ("vk_mini_tiny_shader_toy", ["--headless"]),
]


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def header(name: str) -> None:
    """Print a header with a given name."""
    logger.info("*" * 75)
    logger.info(f"  {name}")
    logger.info("*" * 75)


def create_directory_if_not_exists(directory: Path) -> None:
    """Create a directory if it does not exist."""
    directory.mkdir(parents=True, exist_ok=True)


def run_command(commands: List[str]) -> int:
    """Run a command and handle potential errors."""
    logger.info(f"Running command: {' '.join(commands)}")
    try:
        result = subprocess.run(commands, check=True, text=True, capture_output=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while running command: {e}")
        logger.error(f"Command output:\n{e.stdout}\n{e.stderr}")
        return e.returncode


def simple_progress(iterable, desc: str):
    """A simple progress indicator using only standard libraries."""
    total = len(iterable)
    for i, item in enumerate(iterable, 1):
        yield item
        sys.stdout.write(f"\r{desc}: {i}/{total} ({i/total:.1%})\n")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def build(shader_language: ShaderLanguage) -> ReturnCode:
    """Build the project with the specified shader language."""
    header(f"BUILDING PROJECT WITH {shader_language.name}")

    try:
        create_directory_if_not_exists(BUILD_DIR)
        create_directory_if_not_exists(INSTALL_DIR)

        os.chdir(BUILD_DIR)

        cmake_args = [
            "cmake",
            "..",
            f"-DUSE_SHADER_LANGUAGE={shader_language.name}",
        ]

        run_command(cmake_args)
        run_command(BUILD_COMMANDS)

        os.chdir("..")

        bin_x64_dir = INSTALL_DIR / "bin_x64"
        if bin_x64_dir.exists():
            exe_files = list(bin_x64_dir.glob("*.exe"))
            for file in simple_progress(exe_files, "Removing old executables"):
                file.unlink()

        run_command(["cmake", "--install", "build", "--prefix", str(INSTALL_DIR)])
        return ReturnCode.SUCCESS
    except Exception as e:
        logger.error(f"Build failed: {e}")
        return ReturnCode.BUILD_ERROR


def extract_testing_time(log_file: Path) -> str:
    """Extract testing time from the last line of the log file."""
    if log_file.exists():
        content = log_file.read_text()
        lines = content.splitlines()
        if lines:
            last_line = lines[-1].strip()
            if "ms" in last_line:
                try:
                    # Extract the value before 'ms'
                    return last_line.split()[1] + " ms"
                except IndexError:
                    pass
    return "N/A"


def test(
    shader_language: ShaderLanguage,
) -> Tuple[List[Tuple[str, int, str]], ReturnCode]:
    """Run tests on the built executables."""
    logger.info(f"Executing test function with shader language: {shader_language.name}")
    test_dir = INSTALL_DIR / "bin_x64"

    if not test_dir.exists():
        logger.error(f"Test directory '{test_dir}' does not exist.")
        return [], ReturnCode.ENVIRONMENT_ERROR

    current_dir = Path.cwd()
    os.chdir(test_dir)

    try:
        test_results = []
        overall_result = ReturnCode.SUCCESS

        for executable, args in EXECUTABLES_WITH_ARGS:
            try:
                header(
                    f"Testing '{executable}' with shader language: {shader_language.name}"
                )
                return_code = run_command([executable] + args)
                log_file = f"log_{executable}.txt"
                testing_time = extract_testing_time(Path(log_file))

                result = ReturnCode.SUCCESS
                if return_code != 0:
                    result = ReturnCode.TEST_ERROR
                    overall_result = ReturnCode.TEST_ERROR

                test_results.append((executable, result.value, testing_time))
            except subprocess.CalledProcessError as e:
                logger.error(f"Error occurred while testing: {e}")
                returncode = ReturnCode.TEST_ERROR.value
                test_results.append((executable, returncode, "N/A"))
                overall_result = ReturnCode.TEST_ERROR

                # Ignore errors from ray_query_position_fetch in CI
                if (
                    ("CI" in os.environ)
                    and ("ray_query_position_fetch" in executable)
                    and datetime.datetime.now().date()
                    < datetime.date(year=2024, month=2, day=1)
                ):
                    logger.warning(
                        "Ignored error for ray_query_position_fetch in CI environment"
                    )
                    test_results[-1] = (executable, ReturnCode.SUCCESS.value, "N/A")

        return test_results, overall_result
    finally:
        os.chdir(current_dir)


def final_report(
    shader_language: ShaderLanguage, test_results: List[Tuple[str, int, str]]
) -> None:
    logger.info(f"\nFinal Report for {shader_language.name}:")
    logger.info("-" * 80)
    logger.info("{:<35} | {:<12} | {:<8}".format("Executable", "Return Code", "Time"))
    logger.info("-" * 80)
    for executable, return_code, testing_time in test_results:
        logger.info(
            "{:<35} | {:<12} | {:<8}".format(
                executable, ReturnCode(return_code).name, testing_time
            )
        )
    logger.info("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and test the project with different shader languages."
    )
    parser.add_argument("--test", action="store_true", help="Execute test function")
    parser.add_argument(
        "--build",
        type=ShaderLanguage,
        choices=list(ShaderLanguage),
        help="Execute build function with specified shader language (GLSL, HLSL, or SLANG)",
    )

    args = parser.parse_args()

    if args.build is None and not args.test:
        parser.error("At least one of --build or --test must be specified")

    return args


def main() -> ReturnCode:
    args = parse_args()

    shader_language = args.build or ShaderLanguage.GLSL

    overall_result = ReturnCode.SUCCESS

    if args.build:
        logger.info(f"Building with shader language: {shader_language}")
        build_result = build(shader_language)
        if build_result != ReturnCode.SUCCESS:
            return build_result

    if args.test:
        test_results, test_result = test(shader_language)
        final_report(shader_language, test_results)
        if test_result != ReturnCode.SUCCESS:
            overall_result = test_result

    return overall_result


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(result.value)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(ReturnCode.ENVIRONMENT_ERROR.value)
