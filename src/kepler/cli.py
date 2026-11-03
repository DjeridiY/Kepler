"""Command line: `kepler run CLIP` from a phone clip to a sealed prediction, and quick sealed home bets."""

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

from kepler import home
from kepler.agent import OpenAICompatLLM, Setup
from kepler.budget import Budget, BudgetExceeded
from kepler.measure import MeasureError
from kepler.pipeline import RunError, run
from kepler.sam import MetaTransport, Transport

_KEYS = {"meta": "META_API_KEY", "baseten": "BASETEN_API_KEY"}
_SIGNALS = {"pendulum": "angle", "fall": "height"}


def _box(text: str) -> tuple[int, int, int, int]:
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected X1,Y1,X2,Y2")
    return parts[0], parts[1], parts[2], parts[3]


def _outputs(p: argparse.ArgumentParser, default_out: str) -> None:
    p.add_argument("--out", help=f"output directory (default: {default_out})")
    p.add_argument("--cap", type=float, default=20.0, help="spending cap in USD (default 20)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kepler", description="Film an experiment, find the law, predict the motion.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="track, fit, predict and seal one clip")
    p.add_argument("clip")
    _outputs(p, "<CLIP without extension>_kepler/")
    p.add_argument("--tracker", choices=("sam", "color"), default="sam")
    p.add_argument("--box", type=_box, help="start box X1,Y1,X2,Y2 for the color tracker")
    p.add_argument("--provider", choices=tuple(_KEYS), default="meta")
    p.add_argument("--px-per-m", type=float)
    p.add_argument("--experiment", choices=tuple(_SIGNALS))
    p.add_argument("--prompt", help="short noun phrase naming the moving object")

    p = commands.add_parser("bet", help="period of the shortened pendulum, from a clip of the long one")
    p.add_argument("clip_a")
    p.add_argument("--length-a", type=float, required=True, help="filmed pendulum length (m)")
    p.add_argument("--length-b", type=float, required=True, help="shortened pendulum length (m)")
    p.add_argument("--prompt", default="pendulum bob")
    _outputs(p, "kepler_bet/")

    p = commands.add_parser("cup", help="where to put the cup for a ball rolling off the table")
    p.add_argument("clips", nargs="+")
    p.add_argument("--table-height", type=float, required=True, help="table height (m)")
    p.add_argument("--cup-height", type=float, default=0.0, help="cup height (m, default 0)")
    p.add_argument("--ball-diameter", type=float, default=0.04, help="ball diameter (m, default 0.04)")
    p.add_argument("--prompt", default="ball")
    _outputs(p, "kepler_cup/")

    p = commands.add_parser("weigh", help="mass of an object from its bounce on an elastic, against a reference")
    p.add_argument("ref_clip")
    p.add_argument("obj_clip")
    p.add_argument("--ref-mass", type=float, required=True, help="reference mass (kg)")
    p.add_argument("--prompt-ref", default="bottle")
    p.add_argument("--prompt-obj", default="hanging object")
    _outputs(p, "kepler_weigh/")
    return parser


def _budget(cap: float) -> Budget:
    path = Path.home() / ".kepler" / "budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return Budget(str(path), cap)


def _run(args: argparse.Namespace) -> int:
    llm_key = os.environ.get(_KEYS[args.provider])
    meta_key = os.environ.get("META_API_KEY")
    if not llm_key or (args.tracker == "sam" and not meta_key):
        missing = _KEYS[args.provider] if not llm_key else "META_API_KEY"
        print(f"error: set {missing} in the environment", file=sys.stderr)
        return 2
    setup = None
    if args.experiment and args.prompt:
        setup = Setup(args.experiment, args.prompt, _SIGNALS[args.experiment], "given on the command line")
    out = args.out or str(Path(args.clip).with_suffix("")) + "_kepler"
    try:
        result = run(
            args.clip,
            out,
            transport=MetaTransport(meta_key) if meta_key else None,
            llm=OpenAICompatLLM(args.provider, llm_key),
            budget=_budget(args.cap),
            tracker=args.tracker,
            box0=args.box,
            px_per_m=args.px_per_m,
            setup=setup,
        )
    except BudgetExceeded as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except RunError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    for f in fields(result):
        value = getattr(result, f.name)
        if value is not None and f.name != "entries":
            print(f"{f.name}: {value}")
    print(f"video: {Path(out) / 'kepler.mp4'}")
    return 0


def _measure(args: argparse.Namespace, transport: Transport, budget: Budget) -> tuple[dict, str]:
    """The result and the headline to call out before the take."""
    if args.command == "bet":
        r = home.bet(args.clip_a, args.length_a, args.length_b, prompt=args.prompt, transport=transport, budget=budget)
        return r, f"PERIOD B  {r['period_b_pred_s']:.3f} s"
    if args.command == "cup":
        r = home.cup(
            args.clips, args.table_height, prompt=args.prompt, transport=transport,
            cup_height_m=args.cup_height, ball_diameter_m=args.ball_diameter, budget=budget,
        )  # fmt: skip
        spread_cm = r["spread_m"] * 100 / 2
        return r, f"PUT THE CUP AT  {r['d_pred_m'] * 100:.1f} cm  (±{spread_cm:.1f} cm over {len(r['runs'])} runs)"
    r = home.weigh(
        args.ref_clip, args.obj_clip, args.ref_mass, prompt_ref=args.prompt_ref, prompt_obj=args.prompt_obj,
        transport=transport, budget=budget,
    )  # fmt: skip
    return r, f"MASS  {r['mass_kg'] * 1000:.0f} g"


def _home(args: argparse.Namespace) -> int:
    meta_key = os.environ.get("META_API_KEY")
    if not meta_key:
        print("error: set META_API_KEY in the environment", file=sys.stderr)
        return 2
    try:
        result, headline = _measure(args, MetaTransport(meta_key), _budget(args.cap))
    except BudgetExceeded as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except (RunError, MeasureError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    out = Path(args.out or f"kepler_{args.command}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "seal.txt").write_text(f"sha256 {result['seal_sha256']}\ncomputed {result['computed']}\n")
    print(headline)
    for key, value in result.items():
        if key != "runs":
            print(f"{key}: {value}")
    print(f"saved: {out / 'result.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return _run(args) if args.command == "run" else _home(args)


if __name__ == "__main__":
    sys.exit(main())
