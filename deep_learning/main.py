"""Neural Network & Representation Learning demos.

Run specific demo:
    python deep_learning/main.py 1    # backpropagation
    python deep_learning/main.py 2    # weight initialization
    python deep_learning/main.py 3    # LayerNorm vs BatchNorm
    python deep_learning/main.py 4    # residual connections
    python deep_learning/main.py 5    # dropout & regularization
    python deep_learning/main.py 6    # overfitting demonstration

Run all:
    python deep_learning/main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEMOS = {
    '1': ('demo_01_backprop', 'Backpropagation & Gradient Flow'),
    '2': ('demo_02_init', 'Weight Initialization Strategies'),
    '3': ('demo_03_norm', 'LayerNorm vs BatchNorm'),
    '4': ('demo_04_residual', 'Residual Connections'),
    '5': ('demo_05_dropout', 'Dropout & Regularization'),
    '6': ('demo_06_overfit', 'Overfitting — Memorize vs Learn'),
}


def main():
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key in DEMOS:
            mod_name, desc = DEMOS[key]
            print(f"Running Demo {key}: {desc}")
            __import__(f'deep_learning.demos.{mod_name}').demos.__dict__[
                mod_name].run()
        else:
            print(f"Unknown demo: {key}. Available: {list(DEMOS.keys())}")
    else:
        for key in sorted(DEMOS.keys(), key=int):
            mod_name, desc = DEMOS[key]
            print(f"\n{'=' * 55}")
            print(f"Demo {key}: {desc}")
            print(f"{'=' * 55}")
            mod = __import__(
                f'deep_learning.demos.{mod_name}', fromlist=['run'])
            mod.run()


if __name__ == '__main__':
    main()
