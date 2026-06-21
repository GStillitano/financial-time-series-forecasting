"""Script to run the Bitcoin forecasting experiment."""

from pprint import pprint

from financial_forecasting.cli import build_parser
from financial_forecasting.config import BITCOIN_CONFIG


if __name__ == "__main__":
    # Setup parser and parse arguments
    parser = build_parser("Run the Bitcoin forecasting experiment.")
    args = parser.parse_args()
    
    from financial_forecasting.pipeline import run_experiment

    # Run the Bitcoin experiment and pretty-print the results summary
    pprint(run_experiment(BITCOIN_CONFIG, results_dir=args.results_dir, live=args.live))
