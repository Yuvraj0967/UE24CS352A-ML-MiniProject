"""One-command training entry point.

    python train_model.py          # download data, train K-NN / NB / LogReg, save the best model
    python train_model.py --lstm   # additionally train the Bi-LSTM + GloVe model (needs PyTorch)

Artifacts are written to models/ (model.pkl and vectorizer.pkl are used by demo.py).
"""
import argparse

from src import train_baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lstm", action="store_true", help="also train the Bi-LSTM (requires torch)")
    args = parser.parse_args()

    train_baselines.main()
    print("SUCCESS: model.pkl and vectorizer.pkl saved inside 'models/'")
    if args.lstm:
        from src import train_lstm
        train_lstm.main()


if __name__ == "__main__":
    main()
