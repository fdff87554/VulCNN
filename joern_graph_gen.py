import argparse
import logging
import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ]

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract Code Property Graphs (CPGs) using Joern"
    )
    parser.add_argument(
        "-i", "--input", help="Input directory path", type=str, required=True
    )
    parser.add_argument(
        "-o", "--output", help="Output directory path", type=str, required=True
    )
    parser.add_argument(
        "-t",
        "--type",
        help="Process type: parse or export",
        type=str,
        choices=["parse", "export"],
        required=True,
    )
    parser.add_argument(
        "-r",
        "--repr",
        help="Representation type: pdg or lineinfo_json",
        type=str,
        choices=["pdg", "lineinfo_json"],
        default="pdg",
    )
    parser.add_argument(
        "-j", "--joern_path", help="Joern CLI path", type=str, required=True
    )
    parser.add_argument(
        "-l", "--log_file", help="Log file path", type=str, default="joern_process.log"
    )
    return parser.parse_args()


def setup_environment(joern_path: Path, logger: logging.Logger):
    """Set up environment variables and validate Joern installation"""
    joern_path = joern_path.resolve()
    if not joern_path.exists():
        logger.error(f"Joern path does not exist: {joern_path}")
        exit(1)

    os.environ["PATH"] = f"{joern_path}{os.pathsep}{os.environ['PATH']}"
    os.environ["JOERN_HOME"] = str(joern_path)

    # 只檢查 joern 執行檔是否存在和可執行
    joern_executable = joern_path / "joern"
    if not os.access(str(joern_executable), os.X_OK):
        logger.error(f"Joern executable is not executable: {joern_executable}")
        exit(1)

    logger.info(f"Joern installation found at: {joern_path}")
    return


def run_subprocess(
    cmd: List[str], error_msg: str, logger: logging.Logger
) -> Optional[str]:
    """Run a subprocess command and handle potential errors"""
    try:
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        logger.info(f"Command executed successfully. Output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"{error_msg}: {e}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}", exc_info=True)
    return None


def process_file(
    file: Path,
    outdir: Path,
    record_file: Path,
    process_func: Callable,
    logger: logging.Logger,
):
    """Process a single file and record the result"""
    with record_file.open("r+") as f:
        processed_files = set(f.read().splitlines())
        name = file.stem
        if name in processed_files:
            logger.info(f"File already processed: {name}")
            return

        if process_func(file, outdir):
            f.write(f"{name}\n")
            logger.info(f"Added {name} to record file")


def joern_parse(
    file: Path, outdir: Path, joern_path: Path, logger: logging.Logger
) -> bool:
    """Parse a C file using Joern"""
    name = file.stem
    logger.info(f"Processing file: {name}")
    out_file = outdir / f"{name}.bin"

    if out_file.exists():
        logger.info(f"Output file already exists: {out_file}")
        return True

    cmd = [
        str(joern_path / "joern-parse"),
        str(file),
        "--language",
        "c",
        "--output",
        str(out_file),
    ]

    return run_subprocess(cmd, f"Error parsing file {file}", logger) is not None


def joern_export(
    bin_file: Path, outdir: Path, repr: str, joern_path: Path, logger: logging.Logger
) -> bool:
    """Export parsed binary file to PDG or JSON format"""
    logger.info(f"Starting export process: {bin_file}")
    name = bin_file.stem
    out_file = outdir / name

    if repr == "pdg":
        cmd = [
            str(joern_path / "joern-export"),
            str(bin_file),
            "--repr",
            "pdg",
            "--out",
            str(out_file),
        ]

        if run_subprocess(cmd, f"Error exporting PDG: {bin_file}", logger) is None:
            return False

        return merge_pdg_files(out_file, logger)
    else:  # JSON export
        return export_json(bin_file, out_file, joern_path, logger)


def merge_pdg_files(out_file: Path, logger: logging.Logger) -> bool:
    """Merge multiple PDG files into a single .dot file"""
    if not out_file.is_dir():
        logger.info(f"PDG output is already a file: {out_file}")
        return True

    pdg_files = list(out_file.glob("*.dot"))
    if not pdg_files:
        logger.warning(f"No .dot files found in {out_file}")
        return False

    merged_dot = out_file.with_suffix(".dot")
    logger.info(f"Merging PDG files to: {merged_dot}")

    try:
        with merged_dot.open("w", encoding="utf-8") as outfile:
            outfile.write("digraph G {\n")
            for pdg in pdg_files:
                logger.info(f"Processing PDG file: {pdg}")
                with pdg.open(encoding="utf-8") as infile:
                    content = infile.read()
                    content = content.replace(
                        "digraph G {", "", 1).rsplit("}", 1)[0]
                    outfile.write(f"subgraph cluster_{
                                  pdg.stem} {{\n{content}\n}}\n")
            outfile.write("}")

        logger.info(f"Successfully merged PDG files to: {merged_dot}")

        # 安全地刪除原始 PDG 目錄
        backup_dir = out_file.with_name(out_file.name + "_backup")
        logger.info(f"Creating backup of original PDG directory: {backup_dir}")
        shutil.move(str(out_file), str(backup_dir))
        logger.info(f"Original PDG directory moved to: {backup_dir}")

        logger.info(f"Removing backup directory: {backup_dir}")
        shutil.rmtree(backup_dir)
        logger.info(f"Backup directory removed: {backup_dir}")

        return True
    except Exception as e:
        logger.error(f"Error in merge_pdg_files: {e}", exc_info=True)
        return False


def export_json(
    bin_file: Path, out_file: Path, joern_path: Path, logger: logging.Logger
) -> bool:
    """Export binary file to JSON format"""
    out_file = out_file.with_suffix(".json")
    script_path = Path("graph-for-funcs.sc").resolve()
    if not script_path.exists():
        logger.error(f"Script file does not exist: {script_path}")
        return False

    cmd = [
        str(joern_path / "joern"),
        "--script",
        str(script_path),
        "--params",
        f"inputPath={bin_file},outputPath={out_file}",
    ]

    if run_subprocess(cmd, f"Error exporting JSON: {bin_file}", logger) is None:
        return False

    logger.info(f"Successfully exported JSON: {out_file}")
    return True


def main():
    args = parse_arguments()
    logger = setup_logging(Path(args.log_file))
    logger.info(
        f"Starting Joern graph generation process with parameters: {args}")

    joern_path = Path(args.joern_path).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    logger.info(f"Joern path: {joern_path}")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    setup_environment(joern_path, logger)

    pool_num = os.cpu_count() or 1
    logger.info(f"Using process pool with {pool_num} workers")

    with Pool(pool_num) as pool:
        if args.type == "parse":
            process_parse(pool, input_path, output_path, joern_path, logger)
        elif args.type == "export":
            process_export(pool, input_path, output_path,
                           args.repr, joern_path, logger)
        else:
            logger.error(f"Invalid process type: {args.type}")

    logger.info("Joern graph generation process completed")


def process_parse(pool, input_path, output_path, joern_path, logger):
    files = list(input_path.glob("*.c"))
    logger.info(f"Found {len(files)} C files to parse")
    record_file = output_path / "parse_res.txt"
    record_file.touch(exist_ok=True)
    pool.map(
        partial(
            process_file,
            outdir=output_path,
            record_file=record_file,
            process_func=partial(
                joern_parse, joern_path=joern_path, logger=logger),
            logger=logger,
        ),
        files,
    )


def process_export(pool, input_path, output_path, repr, joern_path, logger):
    bins = list(input_path.glob("*.bin"))
    logger.info(f"Found {len(bins)} binary files to export")
    record_file = output_path / "export_res.txt"
    record_file.touch(exist_ok=True)
    pool.map(
        partial(
            process_file,
            outdir=output_path,
            record_file=record_file,
            process_func=partial(
                joern_export, repr=repr, joern_path=joern_path, logger=logger
            ),
            logger=logger,
        ),
        bins,
    )


if __name__ == "__main__":
    main()
