# DADO



## Framework
![DADO](figure/system.pdf)

## Dataset
### Generated/Gold Datasets
The datasets, including the original train/validation/test data, the generated training data, as well as label names are available in subfolder `./dataset`:
| Dataset | # Train | # Test | # Class | Task  | Domain |
| ------  | ------- | ----- | ----------- | ----------- | ----------- |
| Amazon | 13.8k | 1.2k | 23 | Multiclass | Product Review |
| Reddit | 26.6k | 2.3k | 45 |Multiclass | Web Post |
| SST-2 | 6.9k | 1.8k | 2 |Multiclass | Movie Review |



###  Dataset Attributes
Please see the subfolders on the `./datasets` directory for attribute information.

## Code for Training Data Generation
See `gen_train_data` for details,

## Code for Classifier Training
See `train_classifier` for details, 

```
run_amazon.sh
run_reddit.sh
run_sst.sh
```



## Citation

