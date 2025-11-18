from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.datasets.registry import list_golden_datasets, load_golden_dataset

# List available datasets
datasets = list_golden_datasets()
print(datasets)



try:
    dataset = GoldenDataset('hotpotqa')
    print(dataset.validate())
    print(dataset.head(5))
    print(dataset.count())
    print(dataset.stats())

    for record in dataset:
        print(f"Question: {record.user_input}")
        print(f"Answer: {record.reference}")
        print(f"Evidence: {record.reference_contexts}")
        print(f"Evidence_ids: {record.reference_context_ids}")
        break  # Show first record
    print("=" * 100)
    for record in load_golden_dataset("xquad"):
        print(f"Question: {record.user_input}")
        print(f"Answer: {record.reference}")
        print(f"Evidence: {record.reference_contexts}")
        print(f"Evidence_ids: {record.reference_context_ids}")
        break  # Show first record
except ValueError as e:
    print(f"Error: {e}")
    print("Available datasets need subset parameter")
