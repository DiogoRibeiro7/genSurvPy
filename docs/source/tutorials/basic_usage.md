# Basic Usage Tutorial

This tutorial covers the fundamentals of generating survival data with gen_surv.

## Your First Dataset

Let's start with the simplest case - generating data from a Cox proportional hazards model:

```python
from gen_surv import generate
import pandas as pd

# Generate basic CPHM data
 df = generate(
     model="cphm",
     n=200,
     beta=0.7,
     covar=1.5,
     model_cens="exponential",
     cens_par=2.0,
     seed=42  # For reproducibility
 )

print(f"Dataset shape: {df.shape}")
print(f"Event rate: {df['status'].mean():.2%}")
print("\nFirst 5 rows:")
print(df.head())
```

## Understanding Parameters

### Common Parameters

All models share these parameters:

- `n`: Sample size (number of individuals)
- `model_cens`: Censoring type ("uniform" or "exponential")
- `cens_par`: Censoring distribution parameter
- `seed`: Random seed for reproducibility

### Model-Specific Parameters

Each model has unique parameters. For CPHM:

- `beta`: Covariate effect (hazard ratio = exp(beta))
- `covar`: Range for uniform covariate generation [0, covar]

## Censoring Mechanisms

gen_surv supports two censoring types:

### Uniform Censoring
```python
# Censoring times uniformly distributed on [0, cens_par]
df_uniform = generate(
    model="cphm",
    n=100,
    beta=0.5,
    covar=2.0,
    model_cens="uniform",
    cens_par=3.0
)
```

### Exponential Censoring
```python
# Censoring times exponentially distributed with mean cens_par
df_exponential = generate(
    model="cphm",
    n=100,
    beta=0.5,
    covar=2.0,
    model_cens="exponential",
    cens_par=2.0
)
```

## Exploring Your Data

Basic data exploration:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Event rate by covariate level
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Histogram of survival times
ax1.hist(df['time'], bins=20, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Time')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Observed Times')

# Event rate vs covariate
df['covar_bin'] = pd.cut(df['covariate'], bins=5)
event_rate = df.groupby('covar_bin')['status'].mean()
event_rate.plot(kind='bar', ax=ax2, rot=45)
ax2.set_ylabel('Event Rate')
ax2.set_title('Event Rate by Covariate Level')

plt.tight_layout()
plt.show()
```

## Next Steps

- Try different models: {doc}`model_comparison`
- Learn advanced features: {doc}`advanced_features`  
- See integration examples: {doc}`integration_examples`
