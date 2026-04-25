from pprint import pprint

from financial_forecasting.cli import build_parser
from financial_forecasting.config import TESLA_CONFIG


if __name__ == "__main__":
    parser = build_parser("Run the Tesla forecasting experiment.")
    args = parser.parse_args()
    from financial_forecasting.pipeline import run_experiment

    pprint(run_experiment(TESLA_CONFIG, results_dir=args.results_dir, live=args.live))
