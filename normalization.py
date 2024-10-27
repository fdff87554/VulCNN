import argparse
import re

from pathlib import Path

from clean_gadget import clean_gadget


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Code normalization")
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input directory containing the code to normalize"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output directory to store the normalized code"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Increase output verbosity"
    )

    return parser.parse_args()


def remove_comments(code: str) -> str:
    code_without_comments = re.sub(
        r'(?<!:)//.*?$|/\*[\s\S]*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        lambda match: match.group() if match.group().startswith(("'", '"')) else '',
        code,
        flags=re.MULTILINE
    )

    cleaned_lines = [
        line for line in code_without_comments.splitlines() if line.strip()]
    return '\n'.join(cleaned_lines)


def process_file(input_path: Path, output_path: Path, verbose: bool = False) -> None:
    try:
        if verbose:
            print(f"Processing file: {input_path}")

        with input_path.open("r", encoding="utf-8") as file:
            code = file.read()

        code_without_comments = remove_comments(code)

        cleaned_code = clean_gadget(code_without_comments.splitlines())

        with output_path.open("w", encoding="utf-8") as file:
            file.write("\n".join(cleaned_code))

        if verbose:
            print(f"Processed: {input_path} -> {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")


def normalize(input_dir: Path, output_dir: Path, verbose: bool = False) -> None:
    for category_folder in input_dir.iterdir():
        if not category_folder.is_dir():
            continue
        relative_path = category_folder.relative_to(input_dir)
        output_folder = output_dir / relative_path
        output_folder.mkdir(parents=True, exist_ok=True)
        for file_path in category_folder.glob('*'):
            if not file_path.is_file():
                continue
            output_file = output_folder / file_path.name
            process_file(file_path, output_file, verbose)


def main() -> None:
    args = parse_options()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    print(f"Starting normalization process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    normalize(input_dir, output_dir, args.verbose)
    print("Normalization completed successfully.")


if __name__ == "__main__":
    main()
