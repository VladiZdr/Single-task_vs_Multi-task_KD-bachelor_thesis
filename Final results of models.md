# Final results of models

## Ledgar

- epochs = 10, KD without teacher annealing, T = 1, full datasets

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ledgar_Teacher | 8 | 0.8252 | 0.8214 | 0.8833 | label_15 (1.00) | label_8 (0.00) | 24.5 smp/s |
| ledgar_Baseline | 9 | 0.5588 | 0.7933 | 0.8631 | label_14 (1.00) | label_8 (0.00) | 341.2 smp/s |
| ledgar_Single_task_KD_Student | 9 | 0.5621 | 0.7930 | 0.8624 | label_14 (1.00) | label_8 (0.00) | 356.9 smp/s |
| ledgar_Mix_05 | 9 | 0.5589 | 0.7926 | 0.8630 | label_14 (1.00) | label_8 (0.00) | 343.8 smp/s |
| ledgar_Mix_07 | 9 | 0.5598 | 0.7915 | 0.8622 | label_14 (1.00) | label_8 (0.00) | 380.5 smp/s |
| [TF-IDF] ledgar_tfidf | 10 | 3.6479 | 0.1999 | 0.4269 | label_11 (0.95) | label_0 (0.00) | 48464.6 smp/s |

## UNFAIR-TOS

- epochs = 10, KD without teacher annealing, T = 1, full datasets

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unfair_tos_Teacher | 10 | 0.0243 | 0.8335 | 0.8117 | label_6 (0.97) | label_2 (0.73) | 30.4 smp/s |
| unfair_tos_Baseline | 8 | 0.0232 | 0.7816 | 0.7438 | label_5 (0.96) | label_2 (0.68) | 438.7 smp/s |
| unfair_tos_Single_task_KD_Student | 8 | 0.0234 | 0.7724 | 0.7377 | label_5 (0.96) | label_0 (0.67) | 361.5 smp/s |
| unfair_tos_Mix_05 | 8 | 0.0233 | 0.7805 | 0.7418 | label_5 (0.96) | label_0 (0.68) | 339.4 smp/s |
| unfair_tos_Mix_07 | 8 | 0.0233 | 0.7724 | 0.7377 | label_5 (0.96) | label_0 (0.67) | 345.4 smp/s |
| [TF-IDF] unfair_tos_tfidf | 1 | 0.6859 | 0.0284 | 0.0285 | label_0 (0.06) | label_5 (0.00) | 82059.9 smp/s |

## MULTI_TASK

- epochs = 10, KD without teacher annealing, T = 1, full datasets

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [MultiTask] multi_task_model_supervised | 10 | 0.4889 | 0.7432 | 0.7698 | - | - | - |
| └─ LEDGAR | - | 0.5612 | 0.7953 | 0.8639 | label_14 (1.00) | label_8 (0.00) | 373.0 smp/s |
| └─ UNFAIR-ToS | - | 0.0414 | 0.6910 | 0.6758 | label_6 (0.87) | label_7 (0.45) | 406.3 smp/s |
| [MultiTask] multi_task_model_kd | 10 | 0.4929 | 0.7373 | 0.7691 | - | - | - |
| └─ LEDGAR | - | 0.5658 | 0.7928 | 0.8640 | label_14 (1.00) | label_8 (0.00) | 363.4 smp/s |
| └─ UNFAIR-ToS | - | 0.0420 | 0.6818 | 0.6742 | label_5 (0.85) | label_7 (0.47) | 322.5 smp/s |
| [MultiTask] multi_task_kd_mix_05 | 10 | 0.4891 | 0.7438 | 0.7724 | - | - | - |
| └─ LEDGAR | - | 0.5615 | 0.7950 | 0.8643 | label_14 (1.00) | label_8 (0.00) | 347.5 smp/s |
| └─ UNFAIR-ToS | - | 0.0416 | 0.6927 | 0.6805 | label_5 (0.88) | label_7 (0.47) | 333.4 smp/s |
| [MultiTask] multi_task_kd_mix_07 | 10 | 0.4905 | 0.7385 | 0.7708 | - | - | - |
| └─ LEDGAR | - | 0.5630 | 0.7926 | 0.8642 | label_14 (1.00) | label_8 (0.00) | 349.7 smp/s |
| └─ UNFAIR-ToS | - | 0.0417 | 0.6844 | 0.6773 | label_5 (0.85) | label_7 (0.47) | 329.4 smp/s |

## INCREASED TEMPERATURE RESULTS (T = 2, T = 4)

- epochs = 10, KD without teacher annealing, full datasets

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ledgar_KD_T2 | 9 | 0.8421 | 0.7821 | 0.8590 | label_14 (1.00) | label_4 (0.00) | 374.7 smp/s |
| unfair_tos_KD_T2 | 10 | 0.0291 | 0.7795 | 0.7500 | label_5 (0.96) | label_4 (0.68) | 382.3 smp/s |
| ledgar_KD_T4 | 9 | 1.0475 | 0.7580 | 0.8546 | label_93 (0.99) | label_4 (0.00) | 339.1 smp/s |
| unfair_tos_KD_T4 | 9 | 0.0322 | 0.7846 | 0.7493 | label_5 (0.96) | label_0 (0.67) | 341.3 smp/s |
| [MultiTask] multi_task_kd_t2 | 10 | 0.7279 | 0.7272 | 0.7623 | - | - | - |
| └─ LEDGAR | - | 0.8382 | 0.7790 | 0.8595 | label_26 (0.99) | label_8 (0.00) | 340.7 smp/s |
| └─ UNFAIR-ToS | - | 0.0458 | 0.6753 | 0.6652 | label_6 (0.85) | label_7 (0.48) | 335.8 smp/s |
| [MultiTask] multi_task_kd_t4 | 10 | 0.9154 | 0.7108 | 0.7591 | - | - | - |
| └─ LEDGAR | - | 1.0552 | 0.7534 | 0.8530 | label_26 (0.99) | label_4 (0.00) | 391.0 smp/s |
| └─ UNFAIR-ToS | - | 0.0500 | 0.6681 | 0.6651 | label_6 (0.83) | label_7 (0.47) | 391.4 smp/s |

## TEACHER ANNEALING

- epochs = 10, full datasets
- *KD WEIGHT 1.0 → 0.0*
- *T = 2*

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ledgar_KD_Annealing | 9 | 0.7250 | 0.7841 | 0.8585 | label_14 (1.00) | label_8 (0.00) | 380.5 smp/s |
| unfair_tos_KD_Annealing | 10 | 0.0264 | 0.7893 | 0.7579 | label_6 (0.94) | label_0 (0.68) | 385.5 smp/s |
| [MultiTask] multi_task_kd_annealing | 10 | 0.5998 | 0.7624 | 0.7896 | - | - | - |
| ├─ LEDGAR | - | 0.6912 | 0.7890 | 0.8607 | label_14 (1.00) | label_8 (0.00) | 414.6 smp/s |
| └─ UNFAIR-ToS | - | 0.0340 | 0.7358 | 0.7186 | label_6 (0.93) | label_7 (0.61) | 415.4 smp/s |

## LOW RESSOURCE EXPERIMENTS

       *LEDGAR IS BALANCED TO MATCH UNFAIR-TOS DATASET SIZE*

*T = 2*

*EPOCHS INCREASED PROPROTIONALLY TO MATCH OBSERVED SAMPLES ON FULL DATASET*

- **SINGLE-TASK RESULTS**
- **UNFAIR-TOS**

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unfair_tos_Teacher_LR30 | 13 | 0.0274 | 0.7574 | 0.7549 | label_5 (0.92) | label_7 (0.60) | 20.7 smp/s |
| unfair_tos_Supervised_LR30 | 26 | 0.0383 | 0.6253 | 0.6182 | label_7 (0.73) | label_3 (0.50) | 346.9 smp/s |
| unfair_tos_KD_LR30 | 19 | 0.0436 | 0.6582 | 0.6566 | label_5 (0.92) | label_7 (0.33) | 380.7 smp/s |
| unfair_tos_Teacher_LR40 | 7 | 0.0223 | 0.8147 | 0.8043 | label_6 (1.00) | label_7 (0.67) | 17.8 smp/s |
| unfair_tos_Supervised_LR40 | 21 | 0.0295 | 0.7712 | 0.7263 | label_6 (0.93) | label_2 (0.62) | 348.5 smp/s |
| unfair_tos_KD_LR40 | 19 | 0.0335 | 0.7674 | 0.7200 | label_5 (0.96) | label_0 (0.62) | 353.9 smp/s |
| unfair_tos_Teacher_LR50 | 8 | 0.0254 | 0.8313 | 0.7946 | label_6 (0.97) | label_2 (0.67) | 17.7 smp/s |
| unfair_tos_Supervised_LR50 | 18 | 0.0283 | 0.7710 | 0.7317 | label_6 (0.97) | label_2 (0.59) | 338.1 smp/s |
| unfair_tos_KD_LR50 | 14 | 0.0333 | 0.7581 | 0.7182 | label_6 (0.97) | label_2 (0.62) | 347.7 smp/s |
| unfair_tos_Teacher_LR60 | 9 | 0.0258 | 0.8073 | 0.7869 | label_5 (0.96) | label_2 (0.72) | 18.2 smp/s |
| unfair_tos_Supervised_LR60 | 17 | 0.0281 | 0.7418 | 0.7059 | label_5 (0.92) | label_2 (0.64) | 340.8 smp/s |
| unfair_tos_KD_LR60 | 15 | 0.0351 | 0.7446 | 0.7170 | label_6 (0.93) | label_4 (0.62) | 362.5 smp/s |
| unfair_tos_Teacher_LR70 | 12 | 0.0250 | 0.8095 | 0.7913 | label_6 (0.97) | label_7 (0.71) | 17.8 smp/s |
| unfair_tos_Supervised_LR70 | 14 | 0.0246 | 0.7575 | 0.7303 | label_6 (0.93) | label_4 (0.65) | 339.5 smp/s |
| unfair_tos_KD_LR70 | 8 | 0.0294 | 0.7590 | 0.7368 | label_6 (0.97) | label_3 (0.60) | 355.4 smp/s |
| unfair_tos_Teacher_LR80 | 12 | 0.0244 | 0.8258 | 0.8043 | label_6 (0.97) | label_0 (0.72) | 21.0 smp/s |
| unfair_tos_Supervised_LR80 | 10 | 0.0237 | 0.7580 | 0.7233 | label_5 (0.96) | label_2 (0.66) | 343.1 smp/s |
| unfair_tos_KD_LR80 | 13 | 0.0295 | 0.7716 | 0.7416 | label_6 (0.94) | label_3 (0.67) | 344.0 smp/s |
| unfair_tos_Teacher_LR90 | 5 | 0.0243 | 0.8212 | 0.8011 | label_6 (1.00) | label_2 (0.69) | 21.2 smp/s |
| unfair_tos_Supervised_LR90 | 8 | 0.0257 | 0.7675 | 0.7277 | label_6 (0.94) | label_0 (0.65) | 357.7 smp/s |
| unfair_tos_KD_LR90 | 11 | 0.0295 | 0.7765 | 0.7500 | label_5 (0.96) | label_2 (0.67) | 340.2 smp/s |
- **LEDGAR**

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ledgar_Teacher_LR3 | 31 | 1.3969 | 0.6805 | 0.7973 | label_16 (0.99) | label_8 (0.00) | 18.7 smp/s |
| ledgar_Supervised_LR3 | 25 | 1.1673 | 0.5878 | 0.7301 | label_26 (0.98) | label_4 (0.00) | 404.4 smp/s |
| ledgar_KD_LR3 | 32 | 1.4157 | 0.5583 | 0.7270 | label_26 (0.98) | label_4 (0.00) | 318.2 smp/s |
| ledgar_Teacher_LR4 | 15 | 1.2421 | 0.6936 | 0.8033 | label_80 (0.99) | label_8 (0.00) | 18.3 smp/s |
| ledgar_Supervised_LR4 | 24 | 1.0574 | 0.6045 | 0.7452 | label_26 (0.98) | label_4 (0.00) | 350.1 smp/s |
| ledgar_KD_LR4 | 25 | 1.3057 | 0.5618 | 0.7376 | label_26 (0.98) | label_1 (0.00) | 284.6 smp/s |
| ledgar_Teacher_LR5 | 17 | 1.2104 | 0.7109 | 0.8137 | label_80 (0.99) | label_8 (0.00) | 18.1 smp/s |
| ledgar_Supervised_LR5 | 20 | 0.9792 | 0.6259 | 0.7642 | label_26 (0.98) | label_8 (0.00) | 345.6 smp/s |
| ledgar_KD_LR5 | 20 | 1.2507 | 0.5773 | 0.7519 | label_26 (0.98) | label_1 (0.00) | 327.6 smp/s |
| ledgar_Teacher_LR6 | 14 | 1.1902 | 0.7094 | 0.8115 | label_16 (0.99) | label_8 (0.00) | 18.4 smp/s |
| ledgar_Supervised_LR6 | 17 | 0.9686 | 0.6194 | 0.7634 | label_26 (0.98) | label_4 (0.00) | 338.1 smp/s |
| ledgar_KD_LR6 | 16 | 1.2350 | 0.5755 | 0.7514 | label_26 (0.98) | label_3 (0.00) | 311.5 smp/s |
| ledgar_Teacher_LR7 | 12 | 1.1660 | 0.7215 | 0.8169 | label_43 (0.99) | label_8 (0.00) | 18.0 smp/s |
| ledgar_Supervised_LR7 | 13 | 0.9415 | 0.6274 | 0.7675 | label_26 (0.99) | label_4 (0.00) | 341.9 smp/s |
| ledgar_KD_LR7 | 14 | 1.2054 | 0.5793 | 0.7546 | label_26 (0.98) | label_1 (0.00) | 325.6 smp/s |
| ledgar_Teacher_LR8 | 13 | 1.0806 | 0.7304 | 0.8218 | label_26 (0.99) | label_4 (0.10) | 22.3 smp/s |
| ledgar_Supervised_LR8 | 12 | 0.9152 | 0.6343 | 0.7730 | label_26 (0.99) | label_4 (0.00) | 348.0 smp/s |
| ledgar_KD_LR8 | 13 | 1.1999 | 0.5891 | 0.7610 | label_26 (0.98) | label_3 (0.00) | 342.0 smp/s |
| ledgar_Teacher_LR9 | 9 | 0.9865 | 0.7389 | 0.8243 | label_16 (0.99) | label_4 (0.08) | 20.2 smp/s |
| ledgar_Supervised_LR9 | 11 | 0.9177 | 0.6283 | 0.7715 | label_26 (0.99) | label_4 (0.00) | 334.2 smp/s |
| ledgar_KD_LR9 | 11 | 1.1814 | 0.5838 | 0.7578 | label_26 (0.98) | label_1 (0.00) | 356.1 smp/s |
| ledgar_Teacher_LR10 | 9 | 0.9602 | 0.7422 | 0.8273 | label_15 (0.99) | label_8 (0.00) | 21.8 smp/s |
| ledgar_Baseline_LR10 | 10 | 0.8988 | 0.6293 | 0.7747 | label_26 (0.98) | label_4 (0.00) | 343.6 smp/s |
| ledgar_KD_Student_LR10 | 10 | 1.1416 | 0.5769 | 0.7577 | label_26 (0.98) | label_1 (0.00) | 336.9 smp/s |
- **MULTI-TASK RESULTS**

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [MultiTask] MT_Supervised_LR30_3 | 20 | 0.9937 | 0.5785 | 0.6619 | - | - | - |
| ├─ LEDGAR | - | 1.1481 | 0.5832 | 0.7308 | label_26 (0.98) | label_4 (0.00) | 339.2 smp/s |
| └─ UNFAIR-ToS | - | 0.0381 | 0.5738 | 0.5930 | label_5 (0.88) | label_7 (0.25) | 341.3 smp/s |
| [MultiTask] MT_KD_LR30_3 | 32 | 1.2237 | 0.5642 | 0.6694 | - | - | - |
| ├─ LEDGAR | - | 1.4132 | 0.5585 | 0.7335 | label_26 (0.97) | label_1 (0.00) | 336.3 smp/s |
| └─ UNFAIR-ToS | - | 0.0512 | 0.5699 | 0.6053 | label_5 (0.85) | label_7 (0.12) | 320.5 smp/s |
| [MultiTask] MT_Supervised_LR40_4 | 17 | 0.9310 | 0.6406 | 0.7053 | - | - | - |
| ├─ LEDGAR | - | 1.0762 | 0.6006 | 0.7439 | label_26 (0.98) | label_4 (0.00) | 333.4 smp/s |
| └─ UNFAIR-ToS | - | 0.0324 | 0.6805 | 0.6667 | label_5 (0.88) | label_2 (0.58) | 371.4 smp/s |
| [MultiTask] MT_KD_LR40_4 | 25 | 1.1346 | 0.6280 | 0.7163 | - | - | - |
| ├─ LEDGAR | - | 1.3121 | 0.5610 | 0.7419 | label_26 (0.98) | label_1 (0.00) | 375.6 smp/s |
| └─ UNFAIR-ToS | - | 0.0361 | 0.6950 | 0.6906 | label_5 (0.81) | label_7 (0.62) | 374.7 smp/s |
| [MultiTask] MT_Supervised_LR50_5 | 19 | 0.8594 | 0.6461 | 0.7063 | - | - | - |
| ├─ LEDGAR | - | 0.9932 | 0.6299 | 0.7621 | label_26 (0.98) | label_1 (0.00) | 368.8 smp/s |
| └─ UNFAIR-ToS | - | 0.0320 | 0.6624 | 0.6505 | label_5 (0.83) | label_4 (0.55) | 339.2 smp/s |
| [MultiTask] MT_KD_LR50_5 | 18 | 1.1183 | 0.6023 | 0.6927 | - | - | - |
| ├─ LEDGAR | - | 1.2930 | 0.5708 | 0.7485 | label_26 (0.98) | label_1 (0.00) | 383.1 smp/s |
| └─ UNFAIR-ToS | - | 0.0372 | 0.6338 | 0.6369 | label_5 (0.83) | label_4 (0.45) | 346.3 smp/s |
| [MultiTask] MT_Supervised_LR60_6 | 13 | 0.8470 | 0.6611 | 0.7284 | - | - | - |
| ├─ LEDGAR | - | 0.9790 | 0.6218 | 0.7638 | label_26 (0.98) | label_4 (0.00) | 395.0 smp/s |
| └─ UNFAIR-ToS | - | 0.0306 | 0.7004 | 0.6930 | label_5 (0.88) | label_3 (0.55) | 342.5 smp/s |
| [MultiTask] MT_KD_LR60_6 | 13 | 1.0931 | 0.6376 | 0.7198 | - | - | - |
| ├─ LEDGAR | - | 1.2638 | 0.5619 | 0.7457 | label_26 (0.98) | label_1 (0.00) | 350.4 smp/s |
| └─ UNFAIR-ToS | - | 0.0370 | 0.7133 | 0.6940 | label_6 (0.90) | label_4 (0.59) | 344.6 smp/s |
| [MultiTask] MT_Supervised_LR70_7 | 14 | 0.8168 | 0.6831 | 0.7416 | - | - | - |
| ├─ LEDGAR | - | 0.9443 | 0.6359 | 0.7742 | label_26 (0.98) | label_4 (0.00) | 335.0 smp/s |
| └─ UNFAIR-ToS | - | 0.0277 | 0.7304 | 0.7090 | label_5 (0.88) | label_3 (0.57) | 413.3 smp/s |
| [MultiTask] MT_KD_LR70_7 | 13 | 1.0772 | 0.6426 | 0.7300 | - | - | - |
| ├─ LEDGAR | - | 1.2458 | 0.5689 | 0.7516 | label_26 (0.98) | label_3 (0.00) | 345.6 smp/s |
| └─ UNFAIR-ToS | - | 0.0336 | 0.7162 | 0.7084 | label_5 (0.88) | label_3 (0.64) | 334.2 smp/s |
| [MultiTask] MT_Supervised_LR80_8 | 12 | 0.7918 | 0.6848 | 0.7472 | - | - | - |
| ├─ LEDGAR | - | 0.9156 | 0.6361 | 0.7773 | label_26 (0.98) | label_4 (0.00) | 328.8 smp/s |
| └─ UNFAIR-ToS | - | 0.0257 | 0.7334 | 0.7171 | label_5 (0.88) | label_4 (0.59) | 330.6 smp/s |
| [MultiTask] MT_KD_LR80_8 | 12 | 1.0579 | 0.6434 | 0.7281 | - | - | - |
| ├─ LEDGAR | - | 1.2238 | 0.5762 | 0.7565 | label_26 (0.98) | label_1 (0.00) | 359.9 smp/s |
| └─ UNFAIR-ToS | - | 0.0307 | 0.7106 | 0.6997 | label_5 (0.88) | label_4 (0.56) | 370.6 smp/s |
| [MultiTask] MT_Supervised_LR90_9 | 10 | 0.7917 | 0.6760 | 0.7380 | - | - | - |
| ├─ LEDGAR | - | 0.9153 | 0.6410 | 0.7788 | label_26 (0.98) | label_4 (0.00) | 336.6 smp/s |
| └─ UNFAIR-ToS | - | 0.0269 | 0.7109 | 0.6973 | label_5 (0.88) | label_2 (0.63) | 340.6 smp/s |
| [MultiTask] MT_KD_LR90_9 | 11 | 1.0316 | 0.6424 | 0.7302 | - | - | - |
| ├─ LEDGAR | - | 1.1934 | 0.5728 | 0.7552 | label_64 (0.98) | label_1 (0.00) | 379.2 smp/s |
| └─ UNFAIR-ToS | - | 0.0304 | 0.7119 | 0.7052 | label_5 (0.88) | label_4 (0.62) | 376.9 smp/s |
| [MultiTask] multi_task_super_lr100_10 | 10 | 0.7741 | 0.6914 | 0.7555 | - | - | - |
| ├─ LEDGAR | - | 0.8950 | 0.6379 | 0.7788 | label_26 (0.98) | label_4 (0.00) | 379.7 smp/s |
| └─ UNFAIR-ToS | - | 0.0257 | 0.7449 | 0.7322 | label_5 (0.88) | label_3 (0.62) | 388.4 smp/s |
| [MultiTask] multi_task_kd_lr100_10 | 10 | 1.0108 | 0.6468 | 0.7341 | - | - | - |
| ├─ LEDGAR | - | 1.1692 | 0.5714 | 0.7551 | label_26 (0.97) | label_1 (0.00) | 351.4 smp/s |
| └─ UNFAIR-ToS | - | 0.0300 | 0.7223 | 0.7131 | label_5 (0.92) | label_7 (0.56) | 339.9 smp/s |

## DIFFERENT SEEDS RESULTS

       *LEDGAR IS BALANCED TO MATCH UNFAIR-TOS DATASET SIZE*

*T = 2*

*EPOCHS INCREASED PROPROTIONALLY TO MATCH OBSERVED SAMPLES ON FULL DATASET*

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ledgar_Baseline_S2 | 10 | 0.5578 | 0.7899 | 0.8583 | label_14 (1.00) | label_8 (0.00) | 351.2 smp/s |
| unfair_tos_Baseline_S2 | 10 | 0.0223 | 0.7828 | 0.7453 | label_6 (0.97) | label_1 (0.66) | 370.5 smp/s |
| ledgar_Single_task_KD_Student_S2 | 10 | 0.8493 | 0.7811 | 0.8587 | label_93 (0.99) | label_8 (0.00) | 340.0 smp/s |
| unfair_tos_Single_task_KD_Student_S2 | 8 | 0.0289 | 0.8059 | 0.7655 | label_6 (0.97) | label_0 (0.68) | 341.6 smp/s |
| ledgar_KD_Annealing_S2 | 7 | 0.7686 | 0.7742 | 0.8529 | label_93 (0.99) | label_8 (0.00) | 348.4 smp/s |
| unfair_tos_KD_Annealing_S2 | 8 | 0.0267 | 0.7895 | 0.7560 | label_6 (0.97) | label_0 (0.67) | 328.2 smp/s |
| unfair_tos_Supervised_LR40_S2 | 20 | 0.0298 | 0.7037 | 0.6982 | label_5 (0.92) | label_7 (0.53) | 338.6 smp/s |
| ledgar_Supervised_LR4_S2 | 25 | 1.0257 | 0.6146 | 0.7576 | label_26 (0.98) | label_8 (0.00) | 325.3 smp/s |
| unfair_tos_KD_LR40_S2 | 12 | 0.0356 | 0.6936 | 0.6902 | label_5 (0.89) | label_7 (0.55) | 330.7 smp/s |
| ledgar_KD_LR4_S2 | 24 | 1.2573 | 0.5785 | 0.7481 | label_26 (0.98) | label_1 (0.00) | 366.4 smp/s |
| unfair_tos_Supervised_LR60_S2 | 13 | 0.0281 | 0.7082 | 0.6798 | label_5 (0.82) | label_2 (0.54) | 348.5 smp/s |
| ledgar_Supervised_LR6_S2 | 16 | 0.9434 | 0.6366 | 0.7728 | label_26 (0.98) | label_1 (0.00) | 350.5 smp/s |
| unfair_tos_KD_LR60_S2 | 17 | 0.0354 | 0.7172 | 0.6971 | label_5 (0.88) | label_2 (0.56) | 326.6 smp/s |
| ledgar_KD_LR6_S2 | 17 | 1.2123 | 0.5866 | 0.7596 | label_26 (0.98) | label_3 (0.00) | 357.4 smp/s |
| unfair_tos_Supervised_LR80_S2 | 7 | 0.0237 | 0.7520 | 0.7451 | label_6 (0.97) | label_7 (0.53) | 334.2 smp/s |
| ledgar_Supervised_LR8_S2 | 12 | 0.9073 | 0.6502 | 0.7811 | label_26 (0.98) | label_8 (0.00) | 334.6 smp/s |
| unfair_tos_KD_LR80_S2 | 13 | 0.0304 | 0.7900 | 0.7645 | label_6 (0.94) | label_2 (0.68) | 324.9 smp/s |
| ledgar_KD_LR8_S2 | 13 | 1.1775 | 0.5942 | 0.7659 | label_26 (0.98) | label_3 (0.00) | 353.9 smp/s |
| ledgar_Baseline_LR10_S2 | 10 | 0.8928 | 0.6437 | 0.7831 | label_26 (0.98) | label_4 (0.00) | 323.2 smp/s |
| ledgar_KD_Student_LR10_S2 | 10 | 1.1323 | 0.5923 | 0.7638 | label_26 (0.98) | label_1 (0.00) | 337.3 smp/s |
| ledgar_Baseline_S3 | 10 | 0.5571 | 0.7900 | 0.8627 | label_14 (1.00) | label_8 (0.00) | 353.2 smp/s |
| unfair_tos_Baseline_S3 | 6 | 0.0244 | 0.7649 | 0.7371 | label_5 (0.96) | label_0 (0.68) | 333.7 smp/s |
| ledgar_Single_task_KD_Student_S3 | 10 | 0.8347 | 0.7726 | 0.8590 | label_39 (0.99) | label_8 (0.00) | 331.0 smp/s |
| unfair_tos_Single_task_KD_Student_S3 | 10 | 0.0310 | 0.7561 | 0.7297 | label_5 (0.96) | label_2 (0.65) | 329.1 smp/s |
| ledgar_KD_Annealing_S3 | 9 | 0.7169 | 0.7822 | 0.8576 | label_14 (1.00) | label_8 (0.00) | 347.3 smp/s |
| unfair_tos_KD_Annealing_S3 | 9 | 0.0279 | 0.7589 | 0.7391 | label_5 (0.96) | label_7 (0.63) | 352.0 smp/s |
| unfair_tos_Supervised_LR40_S3 | 19 | 0.0304 | 0.7364 | 0.6888 | label_5 (0.96) | label_2 (0.56) | 382.4 smp/s |
| ledgar_Supervised_LR4_S3 | 22 | 1.0659 | 0.6060 | 0.7512 | label_26 (0.98) | label_8 (0.00) | 338.6 smp/s |
| unfair_tos_KD_LR40_S3 | 22 | 0.0354 | 0.7421 | 0.6959 | label_5 (0.92) | label_0 (0.59) | 346.0 smp/s |
| ledgar_KD_LR4_S3 | 25 | 1.3019 | 0.5693 | 0.7423 | label_26 (0.98) | label_1 (0.00) | 335.5 smp/s |
| unfair_tos_Supervised_LR60_S3 | 15 | 0.0277 | 0.7221 | 0.7034 | label_5 (0.96) | label_7 (0.50) | 346.5 smp/s |
| ledgar_Supervised_LR6_S3 | 17 | 0.9850 | 0.6178 | 0.7616 | label_26 (0.98) | label_8 (0.00) | 341.2 smp/s |
| unfair_tos_KD_LR60_S3 | 15 | 0.0378 | 0.7018 | 0.6917 | label_5 (0.89) | label_3 (0.57) | 347.5 smp/s |
| ledgar_KD_LR6_S3 | 17 | 1.2403 | 0.5869 | 0.7586 | label_26 (0.98) | label_3 (0.00) | 351.6 smp/s |
| unfair_tos_Supervised_LR80_S3 | 8 | 0.0228 | 0.7290 | 0.7097 | label_5 (0.92) | label_2 (0.65) | 408.7 smp/s |
| ledgar_Supervised_LR8_S3 | 12 | 0.9505 | 0.6278 | 0.7702 | label_26 (0.98) | label_8 (0.00) | 345.1 smp/s |
| unfair_tos_KD_LR80_S3 | 13 | 0.0327 | 0.7536 | 0.7391 | label_5 (0.96) | label_7 (0.63) | 340.8 smp/s |
| ledgar_KD_LR8_S3 | 12 | 1.2087 | 0.5854 | 0.7589 | label_26 (0.98) | label_1 (0.00) | 335.5 smp/s |
| ledgar_Baseline_LR10_S3 | 10 | 0.9258 | 0.6269 | 0.7728 | label_26 (0.98) | label_4 (0.00) | 338.5 smp/s |
| ledgar_KD_Student_LR10_S3 | 10 | 1.1580 | 0.5888 | 0.7642 | label_26 (0.98) | label_1 (0.00) | 334.1 smp/s |

| Model / Task Sub-Split | Epoch | Loss | Macro-F1 | Micro-F1 | Best Label (F1) | Worst Label (F1) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [MultiTask] multi_task_model_supervised_S2 | 10 | 0.4905 | 0.7268 | 0.7476 | - | - | - |
| ├─ LEDGAR | - | 0.5624 | 0.7914 | 0.8615 | label_14 (1.00) | label_8 (0.00) | 372.2 smp/s |
| └─ UNFAIR-ToS | - | 0.0456 | 0.6623 | 0.6336 | label_5 (0.89) | label_7 (0.52) | 336.9 smp/s |
| [MultiTask] multi_task_model_kd_S2 | 10 | 0.7246 | 0.7219 | 0.7505 | - | - | - |
| ├─ LEDGAR | - | 0.8334 | 0.7729 | 0.8584 | label_26 (0.99) | label_8 (0.00) | 378.7 smp/s |
| └─ UNFAIR-ToS | - | 0.0514 | 0.6708 | 0.6426 | label_6 (0.89) | label_7 (0.54) | 333.4 smp/s |
| [MultiTask] multi_task_kd_annealing_S2 | 10 | 0.5972 | 0.7658 | 0.7796 | - | - | - |
| ├─ LEDGAR | - | 0.6880 | 0.7822 | 0.8587 | label_80 (0.99) | label_8 (0.00) | 337.0 smp/s |
| └─ UNFAIR-ToS | - | 0.0357 | 0.7494 | 0.7005 | label_6 (0.97) | label_2 (0.59) | 338.2 smp/s |
| [MultiTask] MT_Supervised_LR40_4_S2 | 21 | 0.9219 | 0.6398 | 0.7109 | - | - | - |
| ├─ LEDGAR | - | 1.0654 | 0.6073 | 0.7532 | label_26 (0.98) | label_5 (0.00) | 394.1 smp/s |
| └─ UNFAIR-ToS | - | 0.0343 | 0.6723 | 0.6685 | label_5 (0.88) | label_7 (0.50) | 356.0 smp/s |
| [MultiTask] MT_KD_LR40_4_S2 | 22 | 1.1371 | 0.6214 | 0.6975 | - | - | - |
| ├─ LEDGAR | - | 1.3150 | 0.5514 | 0.7339 | label_26 (0.98) | label_1 (0.00) | 357.6 smp/s |
| └─ UNFAIR-ToS | - | 0.0367 | 0.6914 | 0.6610 | label_5 (0.88) | label_0 (0.60) | 395.9 smp/s |
| [MultiTask] MT_Supervised_LR60_6_S2 | 16 | 0.8343 | 0.6635 | 0.7264 | - | - | - |
| ├─ LEDGAR | - | 0.9643 | 0.6347 | 0.7711 | label_26 (0.98) | label_4 (0.00) | 358.5 smp/s |
| └─ UNFAIR-ToS | - | 0.0304 | 0.6923 | 0.6817 | label_5 (0.83) | label_7 (0.59) | 393.8 smp/s |
| [MultiTask] MT_KD_LR60_6_S2 | 17 | 1.0738 | 0.6398 | 0.7105 | - | - | - |
| ├─ LEDGAR | - | 1.2413 | 0.5810 | 0.7524 | label_26 (0.98) | label_3 (0.00) | 346.0 smp/s |
| └─ UNFAIR-ToS | - | 0.0369 | 0.6985 | 0.6685 | label_5 (0.88) | label_4 (0.46) | 395.6 smp/s |
| [MultiTask] MT_Supervised_LR80_8_S2 | 12 | 0.8088 | 0.6850 | 0.7424 | - | - | - |
| ├─ LEDGAR | - | 0.9347 | 0.6386 | 0.7760 | label_15 (0.98) | label_4 (0.00) | 381.5 smp/s |
| └─ UNFAIR-ToS | - | 0.0293 | 0.7313 | 0.7088 | label_6 (0.90) | label_2 (0.59) | 368.3 smp/s |
| [MultiTask] MT_KD_LR80_8_S2 | 13 | 1.0558 | 0.6502 | 0.7278 | - | - | - |
| ├─ LEDGAR | - | 1.2212 | 0.5801 | 0.7553 | label_26 (0.98) | label_1 (0.00) | 382.7 smp/s |
| └─ UNFAIR-ToS | - | 0.0324 | 0.7203 | 0.7003 | label_5 (0.88) | label_2 (0.60) | 381.6 smp/s |
| [MultiTask] multi_task_super_lr100_10_S2 | 9 | 0.7804 | 0.6915 | 0.7514 | - | - | - |
| ├─ LEDGAR | - | 0.9022 | 0.6373 | 0.7802 | label_26 (0.98) | label_4 (0.00) | 419.7 smp/s |
| └─ UNFAIR-ToS | - | 0.0273 | 0.7458 | 0.7227 | label_5 (0.88) | label_2 (0.64) | 414.4 smp/s |
| [MultiTask] multi_task_kd_lr100_10_S2 | 10 | 1.0089 | 0.6571 | 0.7331 | - | - | - |
| ├─ LEDGAR | - | 1.1669 | 0.5802 | 0.7567 | label_26 (0.98) | label_1 (0.00) | 359.7 smp/s |
| └─ UNFAIR-ToS | - | 0.0311 | 0.7339 | 0.7095 | label_5 (0.88) | label_2 (0.63) | 308.2 smp/s |
| [MultiTask] multi_task_model_supervised_S3 | 10 | 0.4859 | 0.7249 | 0.7491 | - | - | - |
| ├─ LEDGAR | - | 0.5575 | 0.7919 | 0.8623 | label_14 (1.00) | label_8 (0.00) | 360.0 smp/s |
| └─ UNFAIR-ToS | - | 0.0427 | 0.6579 | 0.6360 | label_6 (0.85) | label_7 (0.52) | 376.1 smp/s |
| [MultiTask] multi_task_model_kd_S3 | 10 | 0.7291 | 0.7141 | 0.7400 | - | - | - |
| ├─ LEDGAR | - | 0.8385 | 0.7799 | 0.8578 | label_93 (0.99) | label_8 (0.00) | 343.6 smp/s |
| └─ UNFAIR-ToS | - | 0.0519 | 0.6482 | 0.6221 | label_6 (0.88) | label_7 (0.47) | 362.3 smp/s |
| [MultiTask] multi_task_kd_annealing_S3 | 10 | 0.5986 | 0.7508 | 0.7683 | - | - | - |
| ├─ LEDGAR | - | 0.6897 | 0.7871 | 0.8588 | label_14 (1.00) | label_8 (0.00) | 376.8 smp/s |
| └─ UNFAIR-ToS | - | 0.0350 | 0.7145 | 0.6779 | label_5 (0.88) | label_0 (0.60) | 406.3 smp/s |
| [MultiTask] MT_Supervised_LR40_4_S3 | 15 | 0.9701 | 0.6178 | 0.6973 | - | - | - |
| ├─ LEDGAR | - | 1.1217 | 0.5832 | 0.7392 | label_26 (0.98) | label_8 (0.00) | 352.6 smp/s |
| └─ UNFAIR-ToS | - | 0.0317 | 0.6524 | 0.6554 | label_5 (0.80) | label_7 (0.56) | 356.9 smp/s |
| [MultiTask] MT_KD_LR40_4_S3 | 21 | 1.1354 | 0.6203 | 0.7069 | - | - | - |
| ├─ LEDGAR | - | 1.3133 | 0.5630 | 0.7391 | label_26 (0.98) | label_5 (0.00) | 341.5 smp/s |
| └─ UNFAIR-ToS | - | 0.0345 | 0.6777 | 0.6746 | label_6 (0.81) | label_4 (0.41) | 340.0 smp/s |
| [MultiTask] MT_Supervised_LR60_6_S3 | 17 | 0.8542 | 0.6723 | 0.7402 | - | - | - |
| ├─ LEDGAR | - | 0.9879 | 0.6286 | 0.7654 | label_26 (0.98) | label_8 (0.00) | 353.2 smp/s |
| └─ UNFAIR-ToS | - | 0.0265 | 0.7161 | 0.7151 | label_5 (0.92) | label_7 (0.55) | 359.8 smp/s |
| [MultiTask] MT_KD_LR60_6_S3 | 13 | 1.1236 | 0.6307 | 0.7153 | - | - | - |
| ├─ LEDGAR | - | 1.2994 | 0.5652 | 0.7432 | label_26 (0.97) | label_3 (0.00) | 345.6 smp/s |
| └─ UNFAIR-ToS | - | 0.0357 | 0.6961 | 0.6873 | label_5 (0.85) | label_4 (0.51) | 345.7 smp/s |
| [MultiTask] MT_Supervised_LR80_8_S3 | 11 | 0.8153 | 0.6805 | 0.7438 | - | - | - |
| ├─ LEDGAR | - | 0.9429 | 0.6325 | 0.7698 | label_26 (0.98) | label_8 (0.00) | 342.8 smp/s |
| └─ UNFAIR-ToS | - | 0.0261 | 0.7285 | 0.7178 | label_5 (0.83) | label_3 (0.65) | 348.9 smp/s |
| [MultiTask] MT_KD_LR80_8_S3 | 13 | 1.0766 | 0.6583 | 0.7402 | - | - | - |
| ├─ LEDGAR | - | 1.2458 | 0.5784 | 0.7541 | label_26 (0.97) | label_1 (0.00) | 341.9 smp/s |
| └─ UNFAIR-ToS | - | 0.0298 | 0.7381 | 0.7262 | label_5 (0.92) | label_4 (0.65) | 343.8 smp/s |
| [MultiTask] multi_task_super_lr100_10_S3 | 10 | 0.7832 | 0.7008 | 0.7617 | - | - | - |
| ├─ LEDGAR | - | 0.9058 | 0.6402 | 0.7783 | label_26 (0.98) | label_8 (0.00) | 408.0 smp/s |
| └─ UNFAIR-ToS | - | 0.0246 | 0.7615 | 0.7451 | label_5 (0.92) | label_7 (0.67) | 387.4 smp/s |
| [MultiTask] multi_task_kd_lr100_10_S3 | 10 | 1.0187 | 0.6626 | 0.7478 | - | - | - |
| ├─ LEDGAR | - | 1.1785 | 0.5808 | 0.7581 | label_26 (0.97) | label_1 (0.00) | 341.4 smp/s |
| └─ UNFAIR-ToS | - | 0.0297 | 0.7445 | 0.7374 | label_5 (0.92) | label_7 (0.63) | 331.4 smp/s |