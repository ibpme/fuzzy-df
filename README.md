# fuzzy-df

`fuzzy-df` is a Python package that provides utilities for performing fuzzy matching and merging on pandas DataFrames and Series. It leverages the power of `rapidfuzz` for efficient similarity computations and integrates seamlessly with pandas.

## Features

- **Fuzzy Matching**: Match elements between two pandas Series based on similarity scores.
- **Fuzzy Merging**: Merge pandas DataFrames or Series using fuzzy matching logic.
- **Customizable**: Configure similarity score thresholds and specify custom column names for scores.

## Installation

You can install `fuzzy-df` using pip:

```bash
pip install fuzzy-df
```

## Usage

### Fuzzy Matching

Use the `fuzz_match` function to perform fuzzy matching between two pandas Series:

```python
import pandas as pd
from fuzzy_df.match import fuzz_match

comp_left = pd.Series(["apple", "banana", "cherry"])
comp_right = pd.Series(["apples", "bananas", "grape"])

matches = fuzz_match(comp_left, comp_right, score_cutoff=70)
print(matches)
```

**Output:**

```
    left_index  right_index  score
0           0            0   90.0
1           1            1   85.0
```

### Fuzzy Merging

Use the `fuzz_merge` function to merge two pandas DataFrames or Series based on fuzzy matching:

```python
import pandas as pd
from fuzzy_df.merge import fuzz_merge

left = pd.DataFrame({"id": [1, 2], "name": ["apple", "banana"]})
right = pd.DataFrame({"id": [3, 4], "name": ["apples", "bananas"]})

merged = fuzz_merge(left, right, left_on="name", right_on="name", score_cutoff=70)
print(merged)
```

## Building

### Building with `uv`

To build the project using `uv`, follow these steps:

1. **Run the build command**: Navigate to the project directory and execute the following command:

   ```bash
   uv build
   ```

   This will package the project and prepare it for distribution.

2. **Verify the build**: After the build process completes, you should see the generated distribution files in the `dist/` directory. You can verify them by listing the contents:

   ```bash
   ls dist/
   ```

3. **Install the built package locally** (optional): To test the built package, you can install it locally using pip:

   ```bash
   pip install dist/fuzzy_df-<version>.tar.gz
   ```

Replace `<version>` with the actual version number of the package.

For more information on `uv`, refer to its [documentation](https://github.com/ultraviolet/uv).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests on the [GitHub repository](https://github.com/ibpme/fuzzy-df).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Developed by Iman-Budi Pranakasih. For inquiries, contact [ibpranakasih@gmail.com](mailto:ibpranakasih@gmail.com).
