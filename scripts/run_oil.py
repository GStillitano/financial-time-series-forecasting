from pprint import pprint

from financial_forecasting.config import OIL_CONFIG
from financial_forecasting.pipeline import run_experiment


if __name__ == "__main__":
    pprint(run_experiment(OIL_CONFIG))

