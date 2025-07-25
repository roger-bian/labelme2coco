### Use labelme environment
```
conda create -n labelme
conda activate labelme
pip install labelme
```

### Convert command
```
# It generates:
#   - data_dataset_coco/JPEGImages
#   - data_dataset_coco/Visualization
#   - data_dataset_coco/annotations.json


python ./labelme2coco.py {input_folder} {output_folder} --labels labels.txt
```

#### Example
```
python ./labelme2coco.py data_annotated data_dataset_coco --labels labels.txt
```