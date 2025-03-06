# SpS-SpecDec- A Speculative Decoding Library

## Overview

This library implements an optimized speculative decoding algorithm for autoregressive models, inspired by DeepMind's research. It enables faster inference using an approximate model to propose multiple candidate tokens, validated by a larger target model.

## Features

- Supports models from Hugging Face's `transformers` library
- Implements speculative sampling with configurable `gamma`
- Benchmarking tools to compare performance
- Easy-to-use CLI and scriptable API

## Structure:

Directory structure looks something like this ..

```bash
speculative-decoding-lib/
│── examples/                     # Example scripts demonstrating usage
│   ├── example_basic.py
│   ├── example_benchmark.py
│── speculative_decoding/         # Core library
│   ├── __init__.py
│   ├── main.py                   # Main speculative decoding pipeline
│   ├── sampler.py                 # Implements speculative sampling logic
│   ├── models.py                  # Model loading and handling utilities
│   ├── utils.py                   # Helper functions (e.g., logging, benchmarks)
│── tests/                         # Unit tests
│   ├── test_sampler.py
│   ├── test_models.py
│── scripts/                       # CLI scripts
│   ├── run_speculative.py         # CLI script to run the library
│── benchmarks/                    # Profiling and benchmark results
│── requirements.txt               # Dependencies
│── setup.py                       # Package installation setup
│── README.md                      # Documentation and usage guide
│── LICENSE                        # Open-source license
```

## Installation

```bash
pip install git+https://github.com/jayeshthk/SpS-SpecDec.git
```

## Usage

### Basic Example

```python
from specdec.main import SpeculativeDecoder

decoder = SpeculativeDecoder(
    approx_model_name="meta-llama/Llama-2-7b-hf",
    target_model_name="meta-llama/Llama-2-70b-hf",
    gamma=4, max_tokens=20, device="cuda"
)
input_text = "Tell me a fun fact about space."
output = decoder.speculative_sample(input_text)
print("Generated Text:", output)
```

### CLI Usage

```bash
python scripts/run_speculative.py --input "Explain black holes." --benchmark
```

## Benchmarking

To compare performance, run:

```bash
python scripts/run_specdec.py --input "Tell me a joke." --benchmark
```

## References

```
@inproceedings{leviathan2023fast,
  title={Fast inference from transformers via speculative decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  booktitle={International Conference on Machine Learning},
  pages={19274--19286},
  year={2023},
  organization={PMLR}
}

@article{chen2023accelerating,
  title={Accelerating large language model decoding with speculative sampling},
  author={Chen, Charlie and Borgeaud, Sebastian and Irving, Geoffrey and Lespiau, Jean-Baptiste and Sifre, Laurent and Jumper, John},
  journal={arXiv preprint arXiv:2302.01318},
  year={2023}
}
```

## Limitations and Strengths of the SpecDec Library

One big downside of the library is that it really depends on having a good draft model—if the draft model makes bad guesses, too many tokens get rejected, which kinda defeats the whole speedup. Also, this technique only works with autoregressive models, so if you were hoping to use it with something like BERT, well... nope. The gamma parameter is also a bit tricky—it needs to be just right. Too high? Too many rejections. Too low? You lose out on the speed boost. Another thing is that it eats up more GPU memory, since you're running two models at once. And honestly, if you're just generating a few tokens, the speedup isn't even noticeable, so standard decoding works just fine in those cases.

That said, this thing can be crazy fast—we’re talking 2-2.5x faster inference without messing up the quality. It works with all the big transformer models and comes with benchmarking tools so you can tweak it for max performance. The draft model is also lightweight, so it keeps compute costs down, which is great if you’re running on a tight setup. Plus, the API and CLI are super easy to use, and if you care about reproducibility, there’s a seed option to keep things consistent.

## License

MIT License. See `LICENSE` for details.
