# Test-case-prioritization-by-text-vectorization
Project for software testing course.

To run codeBERT test case prioritization:
- Download pretrained model, unzip it and put it under folder `model`.
- Satisfy package versions in requirements.txt
- Then run:

```
cd blackBox
python blackbox.py <algorithm>
```

The pretrained model has been sent to TA via email because of its large size.

Also, other black box baseline methods can be run by changing line 55 in blackbox.py

To run white box baseline methods, run:

```
cd whiteBox
python whiteBox/prioritize.py <metric> <algorithm>
```

To train a new CodeBERT ,  run:

```
sh ./training/run_unsup_example.sh
```















