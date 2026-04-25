from pprint import pprint

from financial_forecasting.config import TESLA_CONFIG
from financial_forecasting.pipeline import run_experiment


if __name__ == "__main__":
    pprint(run_experiment(TESLA_CONFIG))

