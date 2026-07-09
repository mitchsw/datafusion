// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Tests for the parquet scan metrics registration granularity controlled by
//! `datafusion.explain.per_file_metrics` (see [`ParquetFileMetrics::new`]).

use std::sync::Arc;

use crate::parquet::utils::MetricsFinder;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::metrics::MetricsSet;
use parquet::arrow::ArrowWriter;
use tempfile::NamedTempFile;

/// Write a single-column parquet file and return (file name, size).
fn write_file(values: &[i32]) -> (NamedTempFile, String, u64) {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(values.to_vec()))],
    )
    .unwrap();

    let temp_file = tempfile::Builder::new()
        .prefix("metrics_granularity")
        .suffix(".parquet")
        .tempfile_in(std::path::Path::new(""))
        .expect("tempfile creation");

    let mut writer =
        ArrowWriter::try_new(temp_file.reopen().unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let file_name = temp_file.path().to_string_lossy().to_string();
    let file_name = if cfg!(target_os = "windows") {
        file_name.replace('\\', "/")
    } else {
        file_name
    };
    let file_size = temp_file.path().metadata().unwrap().len();
    (temp_file, file_name, file_size)
}

/// Scan two parquet files placed in a single file group (one partition) and
/// return the resulting [`DataSourceExec`] metrics.
async fn scan_two_files_single_partition(per_file_metrics: bool) -> MetricsSet {
    let mut config = SessionConfig::new();
    config.options_mut().explain.per_file_metrics = per_file_metrics;
    let ctx = SessionContext::new_with_config(config);

    let (_f1, name1, size1) = write_file(&[1, 2, 3]);
    let (_f2, name2, size2) = write_file(&[4, 5, 6]);

    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
    let source = Arc::new(ParquetSource::new(Arc::clone(&schema)));

    // Both files in a single group => a single partition that opens both files.
    let file_group = FileGroup::new(vec![
        PartitionedFile::new(name1, size1),
        PartitionedFile::new(name2, size2),
    ]);
    let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
        .with_file_group(file_group)
        .build();

    let plan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
    let results = datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();
    let total_rows = results.iter().map(|b| b.num_rows()).sum::<usize>();
    assert_eq!(total_rows, 6);

    MetricsFinder::find_metrics(plan.as_ref()).unwrap()
}

fn has_filename_label(metrics: &MetricsSet) -> bool {
    metrics
        .iter()
        .any(|m| m.labels().iter().any(|l| l.name() == "filename"))
}

/// Distinct `bytes_scanned` metrics registered (one per `ParquetFileMetrics`).
fn bytes_scanned_registrations(metrics: &MetricsSet) -> usize {
    metrics
        .iter()
        .filter(|m| m.value().name() == "bytes_scanned")
        .count()
}

#[tokio::test]
async fn default_scan_registers_per_partition_metrics() {
    let metrics = scan_two_files_single_partition(false).await;

    assert!(
        !has_filename_label(&metrics),
        "per-partition metrics must not carry a `filename` label: {metrics:#?}"
    );
    assert_eq!(
        bytes_scanned_registrations(&metrics),
        1,
        "two files in one partition should share a single metrics set: {metrics:#?}"
    );
}

#[tokio::test]
async fn per_file_scan_registers_per_file_metrics() {
    let metrics = scan_two_files_single_partition(true).await;

    assert!(
        has_filename_label(&metrics),
        "per-file metrics must carry a `filename` label: {metrics:#?}"
    );
    assert_eq!(
        bytes_scanned_registrations(&metrics),
        2,
        "each of the two files should register its own metrics set: {metrics:#?}"
    );
}
