

mkdir ./test_results
python ./ptlreg/AlignStructuredDataset.py \
  -p ./ptlreg/test_data/structured/primary \
  -s ./ptlreg/test_data/structured/secondary \
  -o ./test_results/ \
  -n "waterfall1_test" \

python ./ptlreg/AlignUnstructuredDataset.py \
  -i ./ptlreg/test_data/unstructured \
  -o ./test_results/ -n "structured_test" \
  -n "unstructured_test" \
  -t 300