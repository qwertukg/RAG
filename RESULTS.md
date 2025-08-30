```
PS C:\Users\danii\PycharmProjects\RAG> py test_damp.py --idx 0,1,2,3,4,5,6,7,8,9 --train-n 60000
[Info] Device: cuda
DAMP far: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:39<00:00,  4.98s/it]
DAMP near:  12%|███████████████████████████████████▌                                                                                                                                                                                                                                                        | 1/8 [00:09<01:04,  9.18s/it] 
detectors λ=0.70 (candidates): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:01<00:00, 533.39it/s] 
[TRAIN] Detectors built: 61
Class memory (batch): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 31.30it/s] 
[Check] bits_on over 128 tests — min/med/max: 0 / 7.0 / 16 | zero_frac=1.56%
Batch 10 — acc=0.400
{'idx': 0, 'prediction': 7, 'confidence': 0.35, 'truth': 7, 'bits_on': 7}
{'idx': 1, 'prediction': 0, 'confidence': 0.3, 'truth': 2, 'bits_on': 6}
{'idx': 2, 'prediction': 0, 'confidence': 0.1, 'truth': 1, 'bits_on': 2}
{'idx': 3, 'prediction': 0, 'confidence': 0.35, 'truth': 0, 'bits_on': 7}
{'idx': 4, 'prediction': 3, 'confidence': 0.6, 'truth': 4, 'bits_on': 12}
{'idx': 5, 'prediction': 1, 'confidence': 0.3, 'truth': 1, 'bits_on': 6}
{'idx': 6, 'prediction': 2, 'confidence': 0.42857142857142855, 'truth': 4, 'bits_on': 10}
{'idx': 7, 'prediction': 0, 'confidence': 0.25, 'truth': 9, 'bits_on': 5}
{'idx': 8, 'prediction': 0, 'confidence': 0.2, 'truth': 5, 'bits_on': 4}
{'idx': 9, 'prediction': 9, 'confidence': 0.5238095238095238, 'truth': 9, 'bits_on': 12}
```

```
PS C:\Users\danii\PycharmProjects\RAG> py test_damp.py --idx 0,1,2,3,4,5,6,7,8,9 --train-n 60000 --target-density 0.12                                                                                  
[Info] Device: cuda
DAMP far: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:38<00:00,  4.87s/it] 
DAMP near:  12%|███████████████████▏                                                                                                                                     | 1/8 [00:09<01:08,  9.82s/it] 
detectors λ=0.70 (candidates): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:01<00:00, 542.42it/s] 
[TRAIN] Detectors built: 61
Class memory (batch): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 29.96it/s] 
[Check] bits_on over 128 tests — min/med/max: 0 / 7.0 / 20 | zero_frac=0.78%
Batch 10 — acc=0.100
{'idx': 0, 'prediction': 0, 'confidence': 0.125, 'truth': 7, 'bits_on': 2}
{'idx': 1, 'prediction': 8, 'confidence': 0.5, 'truth': 2, 'bits_on': 8}
{'idx': 2, 'prediction': 7, 'confidence': 0.25, 'truth': 1, 'bits_on': 3}
{'idx': 3, 'prediction': 1, 'confidence': 0.45454545454545453, 'truth': 0, 'bits_on': 9}
{'idx': 4, 'prediction': 3, 'confidence': 0.8571428571428571, 'truth': 4, 'bits_on': 6}
{'idx': 5, 'prediction': 1, 'confidence': 0.5, 'truth': 1, 'bits_on': 14}
{'idx': 6, 'prediction': 3, 'confidence': 0.625, 'truth': 4, 'bits_on': 6}
{'idx': 7, 'prediction': 3, 'confidence': 0.125, 'truth': 9, 'bits_on': 2}
{'idx': 8, 'prediction': 0, 'confidence': 0.3, 'truth': 5, 'bits_on': 6}
{'idx': 9, 'prediction': 7, 'confidence': 0.5, 'truth': 9, 'bits_on': 14}
```

```
py test_damp.py --idx 0,1,2,3,4,5,6,7,8,9 --train-n 60000 --proto 36x36 --steps-far 8 --steps-near 10 --p-per-step 24576 --min-near-steps 3 --attempts 8192 --mu-e-build 0.012 --eps 7.0 --lam-d 0.62 --mu-e-detect 0.015 --mu-d 0.06 --detect-k 1024 --target-density 0.12
[Info] Device: cuda
DAMP far: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:56<00:00,  7.06s/it]
DAMP near:  20%|██████████████████████████████▍                                                                                                                         | 2/10 [00:21<01:24, 10.51s/it]
detectors λ=0.62 (candidates): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1296/1296 [00:02<00:00, 552.88it/s]
[TRAIN] Detectors built: 57
Class memory (batch): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 22.23it/s]
[Check] bits_on over 128 tests — min/med/max: 2 / 12.0 / 20 | zero_frac=0.00%
Batch 10 — acc=0.100
{'idx': 0, 'prediction': 1, 'confidence': 0.7142857142857143, 'truth': 7, 'bits_on': 6}
{'idx': 1, 'prediction': 1, 'confidence': 0.4, 'truth': 2, 'bits_on': 15}
{'idx': 2, 'prediction': 3, 'confidence': 0.5, 'truth': 1, 'bits_on': 6}
{'idx': 3, 'prediction': 0, 'confidence': 0.3333333333333333, 'truth': 0, 'bits_on': 14}
{'idx': 4, 'prediction': 0, 'confidence': 0.5, 'truth': 4, 'bits_on': 12}
{'idx': 5, 'prediction': 0, 'confidence': 0.5, 'truth': 1, 'bits_on': 6}
{'idx': 6, 'prediction': 1, 'confidence': 0.375, 'truth': 4, 'bits_on': 16}
{'idx': 7, 'prediction': 1, 'confidence': 0.5555555555555556, 'truth': 9, 'bits_on': 8}
{'idx': 8, 'prediction': 0, 'confidence': 0.46153846153846156, 'truth': 5, 'bits_on': 13}
{'idx': 9, 'prediction': 1, 'confidence': 0.4, 'truth': 9, 'bits_on': 15}
```

```
 py test_damp_v.py --idx 0,1,2,3,4,5,6,7,8,9 --train-n 60000 --proto 128x128 --attempts 16384 --lam-d 0.75 --mu-e-build 0.05 --eps 30 --min-samples 10 --show --save data.npz
[Info] Device: cuda
DAMP far: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:05<00:00,  8.23s/it]
DAMP near:  12%|█████████████████████████████▎                                                                                                                                                                                                            | 1/8 [00:14<01:43, 14.72s/it] 
detectors λ=0.75 (candidates): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16384/16384 [23:20<00:00, 11.70it/s]
[TRAIN] Detectors built: 290
Class memory (batch): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:01<00:00, 14.65it/s]
[SAVE] Артефакты сохранены в data.npz
{"idx": 0, "prediction": 7, "confidence": 0.0375, "truth": 7, "bits_on": 3}
{"idx": 1, "prediction": 6, "confidence": 0.5238095238095238, "truth": 2, "bits_on": 48}
{"idx": 2, "prediction": 1, "confidence": 0.4875, "truth": 1, "bits_on": 158}
{"idx": 3, "prediction": 0, "confidence": 0.4074074074074074, "truth": 0, "bits_on": 34}
{"idx": 4, "prediction": 0, "confidence": 0.0625, "truth": 4, "bits_on": 5}
{"idx": 5, "prediction": 1, "confidence": 0.449438202247191, "truth": 1, "bits_on": 178}
{"idx": 6, "prediction": 4, "confidence": 0.0875, "truth": 4, "bits_on": 7}
{"idx": 7, "prediction": 3, "confidence": 0.225, "truth": 9, "bits_on": 18}
{"idx": 8, "prediction": 6, "confidence": 0.575, "truth": 5, "bits_on": 46}
{"idx": 9, "prediction": 7, "confidence": 0.6578947368421053, "truth": 9, "bits_on": 109}
```

```
py test_damp_v.py --idx 0,1,2,3,4,5,6,7,8,9 --train-n 60000 --proto 128x128 --attempts 16384 --lam-d 0.75 --mu-e-build 0.05 --eps 50 --min-samples 10 --show --save data.npz
[Info] Device: cuda
DAMP far: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:04<00:00,  8.12s/it]
DAMP near:  12%|█████████████████████████████▎                                                                                                                                                                                                            | 1/8 [00:14<01:39, 14.17s/it] 
detectors λ=0.75 (candidates): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16384/16384 [44:57<00:00,  6.07it/s]
[TRAIN] Detectors built: 242
Class memory (batch): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:01<00:00, 14.91it/s]
[SAVE] Артефакты сохранены в data.npz
Batch 10 — acc=0.400
{'idx': 0, 'prediction': 7, 'truth': 7, 'confidence': 0.057971014492753624, 'bits_on': 4}
{'idx': 1, 'prediction': 6, 'truth': 2, 'confidence': 0.4782608695652174, 'bits_on': 33}
{'idx': 2, 'prediction': 1, 'truth': 1, 'confidence': 0.43312101910828027, 'bits_on': 156}
{'idx': 3, 'prediction': 0, 'truth': 0, 'confidence': 0.2753623188405797, 'bits_on': 19}
{'idx': 4, 'prediction': 0, 'truth': 4, 'confidence': 0.0, 'bits_on': 0}
{'idx': 5, 'prediction': 1, 'truth': 1, 'confidence': 0.4233128834355828, 'bits_on': 163}
{'idx': 6, 'prediction': 5, 'truth': 4, 'confidence': 0.11594202898550725, 'bits_on': 8}
{'idx': 7, 'prediction': 0, 'truth': 9, 'confidence': 0.043478260869565216, 'bits_on': 3}
{'idx': 8, 'prediction': 6, 'truth': 5, 'confidence': 0.37681159420289856, 'bits_on': 26}
{'idx': 9, 'prediction': 7, 'truth': 9, 'confidence': 0.7701149425287356, 'bits_on': 85}
```