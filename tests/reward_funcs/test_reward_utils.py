"""Tests for reward utils."""
# %% Imports
# Third Party Imports
from numpy.random import rand

# Punch Clock Imports
from punchclock.reward_funcs.reward_utils import cropArray, lookupPreprocessor

# %% Test lookupPreprocessor
print("\nTest lookupPreprocessor...")
func_names = [
    "trace",
    "sum_rows",
    {
        "preprocessor": "crop_array",
        "config": {
            "a": 1,
            "b": 2,
        },
    },
    sum,
    {
        "preprocessor": "divide",
        "config": {"x1": 1},
    },
]

for name in func_names:
    preprocessor = lookupPreprocessor(name)
    print(f"preprocessor = {preprocessor}")

# %% Test cropArray
print("\nTest cropArray...")
x = rand(6, 3)
x_crop = cropArray(
    x=x,
    indices_or_sections=[2, 4],
    axis=0,
    section_to_keep=0,
)
print(f"x = \n{x}")
print(f"x_crop = \n{x_crop}")
