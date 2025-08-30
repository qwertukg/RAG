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