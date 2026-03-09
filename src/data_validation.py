import numpy as np
import pandas as pd
from pathlib import Path
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeOfType,
    ExpectColumnValuesToNotBeNull,
    ExpectColumnDistinctValuesToEqualSet
    )

def validate_data():
    print("Validating data...")
    root_path = Path(__file__).parent.parent
    data_path = root_path / 'data' / 'raw_data.csv'

    df = pd.read_csv(data_path)
    
    context = gx.get_context()
    data_source = context.data_sources.add_pandas(name="my_pandas_data_source")
    data_asset = data_source.add_dataframe_asset(name="raw_data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name="my_batch")
    
    suite = gx.ExpectationSuite(name="my_suite")

    suite.add_expectation(ExpectColumnValuesToBeBetween(column='temp', min_value=-15, max_value=50))
    suite.add_expectation(ExpectColumnValuesToBeBetween(column='humidity', min_value=0, max_value=100))
    suite.add_expectation(ExpectColumnDistinctValuesToEqualSet(column='holiday', value_set={0, 1}))
    for c in ['temp', 'humidity', 'holiday']:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=c))
        suite.add_expectation(ExpectColumnValuesToBeOfType(column=c, type_='int64'))

    validation_definition = gx.ValidationDefinition(
        name="validation_definition",
        data=batch_definition,
        suite=suite
    )

    checkpoint = gx.Checkpoint(
        name="checkpoint",
        validation_definitions=[validation_definition],
        result_format="SUMMARY"
    )

    context.suites.add(suite)
    context.validation_definitions.add(validation_definition)
    context.checkpoints.add(checkpoint)

    results = checkpoint.run(batch_parameters={"dataframe": df})
    if results.success:
        return True
    else:
        try:
            context.view_validation_result(results)
        except gx.exceptions.NoDataDocsError:
            pass
        return False

if __name__ == "__main__":
    success = validate_data()
    if success:
        with open(".validate_success.txt", "w") as f:
            f.write("Validation completed successfully")
        print("✓ Validation successed")
    else:
        print("✗ Validation failed")
        exit(1)