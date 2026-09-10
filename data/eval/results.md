# Eval results (n=220)

## Intent

| system   |   accuracy |   macro-F1 |
|----------|------------|------------|
| agent    |      0.664 |      0.69  |
| majority |      0.177 |      0.038 |
| keyword  |      0.482 |      0.524 |
| tfidf_lr |      0.6   |      0.566 |

### Agent per intent
| intent               |    P |    R |   F1 |   n |
|----------------------|------|------|------|-----|
| playback_issue       | 0.79 | 0.74 | 0.77 |  31 |
| login_account        | 0.65 | 0.95 | 0.77 |  21 |
| billing_subscription | 0.68 | 0.82 | 0.74 |  39 |
| downloads_library    | 1    | 1    | 1    |   8 |
| device_connect       | 0.75 | 0.67 | 0.71 |   9 |
| content_missing      | 0.58 | 0.55 | 0.56 |  33 |
| feature_feedback     | 0.76 | 0.54 | 0.63 |  57 |
| other                | 0.32 | 0.36 | 0.34 |  22 |

### Agent confusion (rows = gold, cols = predicted)
| gold \ pred          |   playback |   login_ac |   billing_ |   download |   device_c |   content_ |   feature_ |   other |
|----------------------|------------|------------|------------|------------|------------|------------|------------|---------|
| playback_issue       |         23 |          1 |          2 |          0 |          0 |          3 |          2 |       0 |
| login_account        |          0 |         20 |          1 |          0 |          0 |          0 |          0 |       0 |
| billing_subscription |          1 |          2 |         32 |          0 |          0 |          1 |          0 |       3 |
| downloads_library    |          0 |          0 |          0 |          8 |          0 |          0 |          0 |       0 |
| device_connect       |          0 |          1 |          0 |          0 |          6 |          0 |          2 |       0 |
| content_missing      |          2 |          0 |          4 |          0 |          0 |         18 |          2 |       7 |
| feature_feedback     |          2 |          3 |          7 |          0 |          2 |          5 |         31 |       7 |
| other                |          1 |          4 |          1 |          0 |          0 |          4 |          4 |       8 |

## Escalation
| system           |   esc P |   esc R |   esc F1 |   auto P |   auto R |   auto F1 |   unsafe auto |   unsafe rate |   esc share |
|------------------|---------|---------|----------|----------|----------|-----------|---------------|---------------|-------------|
| agent            |   0.444 |   0.926 |    0.6   |    0.882 |    0.324 |     0.474 |             6 |         0.027 |       0.768 |
| always_escalate  |   0.368 |   1     |    0.538 |    0     |    0     |     0     |             0 |         0     |       1     |
| never_escalate   |   0     |   0     |    0     |    0.632 |    1     |     0.774 |            81 |         0.368 |       0     |
| keyword_escalate |   0.838 |   0.383 |    0.525 |    0.727 |    0.957 |     0.826 |            50 |         0.227 |       0.168 |

unsafe auto = predicted auto when the human said escalate.

### Agent escalation reasons (first rule hit)
| reason               |   n |
|----------------------|-----|
| account_intent       |  78 |
| no_similar_history   |  33 |
| needs_account_access |  29 |
| low_confidence       |  24 |
| anger_legal_churn    |   4 |
| repeat_contact       |   1 |

### All signals fired
| signal               |   n |
|----------------------|-----|
| needs_account_access | 126 |
| account_intent       |  78 |
| no_similar_history   |  40 |
| low_confidence       |  25 |
| anger_legal_churn    |  11 |
| repeat_contact       |   2 |
| pii                  |   1 |

## Reply quality (LLM judge, n=220)
| system        |   grounded |   helpful |   tone |   safe |   overall |   % overall>=4 |
|---------------|------------|-----------|--------|--------|-----------|----------------|
| agent         |       4.48 |      1.63 |   4.96 |   4.94 |      3.16 |           25   |
| reply_trivial |       4.09 |      1.47 |   4.97 |   4.98 |      3.11 |           19.1 |
| reply_nearest |       4.56 |      1.26 |   4.95 |   4.86 |      2.96 |           11.4 |

### Paired: agent vs baseline on overall
| baseline      |   win |   tie |   loss |   win rate |
|---------------|-------|-------|--------|------------|
| reply_trivial |    46 |   141 |     33 |      0.209 |
| reply_nearest |    65 |   131 |     24 |      0.295 |

## Inter-annotator agreement (n=60)
| field           |   Cohen kappa |   raw agreement |
|-----------------|---------------|-----------------|
| intent          |         0.899 |           0.917 |
| should_escalate |         0.927 |           0.967 |
