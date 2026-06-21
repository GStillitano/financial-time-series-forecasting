"""Script to run all forecasting experiments sequentially."""

from pprint import pprint

from financial_forecasting.cli import build_parser


if __name__ == "__main__":
    # Setup parser and parse arguments
    parser = build_parser("Run all forecasting experiments.")
    args = parser.parse_args()
    
    from financial_forecasting.pipeline import run_all

    # Run all experiments and pretty-print the results summary
    pprint(run_all(results_dir=args.results_dir, live=args.live))
