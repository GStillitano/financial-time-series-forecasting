from pprint import pprint

from financial_forecasting.cli import build_parser


if __name__ == "__main__":
    parser = build_parser("Run all forecasting experiments.")
    args = parser.parse_args()
    from financial_forecasting.pipeline import run_all

    pprint(run_all(results_dir=args.results_dir, live=args.live))
