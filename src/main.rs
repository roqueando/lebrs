//use lebrs::neural::train;
use polars::{lazy::dsl::col, prelude::*};
use lebrs::neural::Dataset;

#[allow(dead_code)]
struct SplittedDataset {
    x_train: DataFrame,
    x_test: DataFrame,
    y_train: DataFrame,
    y_test: DataFrame,
}

fn load_matches() -> PolarsResult<DataFrame> {
    let path = "data/stats1.csv";
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.into()))
        .unwrap()
        .finish()
}

fn data_split(df: LazyFrame, test_percent: f64, train_percent: f64) -> SplittedDataset {
    let train_frac = Series::new("train_frac", &[train_percent]);
    let test_frac = Series::new("test_frac", &[test_percent]);

    let x_train = df
        .clone()
        .filter(col("win").eq(0))
        .collect()
        .unwrap()
        .sample_frac(&train_frac, true, true, Some(200))
        .unwrap()
        .drop("win")
        .unwrap();

    let x_test = df
        .clone()
        .filter(col("win").eq(0))
        .collect()
        .unwrap()
        .sample_frac(&test_frac, true, true, Some(200))
        .unwrap()
        .drop("win")
        .unwrap();

    let y_train = df
        .clone()
        .filter(col("win").eq(1))
        .collect()
        .unwrap()
        .sample_frac(&train_frac, true, true, Some(200))
        .unwrap()
        .drop("win")
        .unwrap();

    let y_test = df
        .clone()
        .filter(col("win").eq(1))
        .collect()
        .unwrap()
        .sample_frac(&test_frac, true, true, Some(200))
        .unwrap()
        .drop("win")
        .unwrap();

    SplittedDataset {
        x_train,
        x_test,
        y_train,
        y_test,
    }
}

fn main() {
    let df = load_matches().unwrap();
    let data = df.clone().lazy().select([
        col("win"),
        col("item1"),
        col("item2"),
        col("item3"),
        col("item4"),
        col("item5"),
        col("item6"),
    ]);

    let dataset = data_split(data, 0.2, 0.8);
    //train();
}
