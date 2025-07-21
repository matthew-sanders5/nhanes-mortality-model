# NHANES Mortality Risk Model

This repository contains code for training and evaluating a machine learning model to predict all-cause mortality using NHANES 1999–2018 data and public-use Linked Mortality Files.

## Files
- `behavior_model.py`: trains model on NHANES 1999–2008 and outputs internal validation results
- `behavior_temporal.py`: evaluates trained model on NHANES 2009–2018
- Outputs are saved in `/results/`

## Data
NHANES data and mortality linkage files must be downloaded separately:
- https://www.cdc.gov/nchs/nhanes/
- https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
