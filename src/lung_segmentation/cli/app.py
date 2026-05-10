"""Gradio app CLI."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Gradio lung segmentation app.")
    parser.add_argument(
        "--model-dir",
        default="artifacts/models",
        help="Directory with .pkl models.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=None, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from lung_segmentation.app import create_demo

    demo = create_demo(args.model_dir)
    launch_kwargs = {"server_name": args.host, "share": args.share}
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
