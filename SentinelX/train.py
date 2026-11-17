import argparse
from src.pipeline import train_system

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentinelX hybrid NIDS")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU worker processes (default: logical CPUs - 1)")
    args = parser.parse_args()
    train_system(args.workers)
