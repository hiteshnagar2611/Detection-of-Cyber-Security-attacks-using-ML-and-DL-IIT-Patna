# ML Model Training and Evaluation with k-Fold Cross-Validation

This script trains and evaluates multiple machine learning models using k-fold cross-validation. It includes various algorithms and saves performance metrics for each model.

## Requirements

- Python 3.6 or higher
- Required Python libraries:
  - scikit-learn
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - datetime
  - os

You can install the necessary libraries using the following command:

```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

## Script Overview

The script performs the following steps:

1. **Imports and Setup**:
   - Imports necessary libraries for machine learning, data manipulation, and plotting.
   - Sets up logging and output file handling.

2. **Data Loading and Preparation**:
   - Loads the dataset and prepares it for training and testing.
   - Splits the data into training and testing sets.

3. **Model Training and Evaluation**:
   - Defines multiple machine learning models.
   - Performs k-fold cross-validation for each model.
   - Trains and evaluates the models on each fold.
   - Calculates performance metrics such as accuracy, precision, recall, and F1 score.
   - Plots and saves confusion matrices for each model and fold.
   - Logs the results and performance metrics to an output file.

4. **Stacking Classifier**:
   - Defines and trains a stacking classifier with k-fold cross-validation.
   - Saves performance metrics and confusion matrices for the stacking classifier.

5. **Cleaning Up**:
   - Deletes intermediate files generated during the process.
   - Logs the start and end times of the script execution to a file.

## How to Use

1. Clone the repository or download the script.

2. Ensure you have the required libraries installed (see Requirements section).

3. Place your dataset in the same directory as the script. Modify the script to load your dataset appropriately.

4. Run the script using the following command:

   ```bash
   python ML_65_Algo_k_fold_Intern.py
   ```

5. The script will perform k-fold cross-validation on multiple models and save the results to the specified output files.

## Output

- Confusion matrix plots for each model and fold.
- Performance metrics for each model and fold, logged to an output file.
- Execution start and end times logged to a `time.txt` file.

## Notes

- Ensure your dataset is compatible with the script. You may need to modify the data loading and preprocessing steps according to your dataset.
- The script uses k-fold cross-validation to ensure robust evaluation of the models.
- Modify the list of models or add new models as needed.

## Contributing

If you have any suggestions or improvements, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
