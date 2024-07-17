# VIP Connected Health Deep Learning Project

## Conda Setup

To install and activate the Conda environment from environment.yml:

```text
conda env create -f environment.yml
conda activate ptb-xl-env
```

When adding any additional dependencies e.g. numpy, pandas etc.

```text
conda activate ptb-xl-env
conda env export > environment.yml
```

To update the environment list

```text
conda env update --file environment.yml --prune
```

Then make a pull request which will show the changes to the Conda environment in environment.yml

## PTB-XL Database Notes

Note that for the PTB-XL database the report information is in German.
E.g. "sinusrhythmus linkstyp t abnormal, wahrscheinlich inferiorer myokardschaden qt-verlängerung"
translates to: "Sinus rhythm left type T abnormal, probably inferior myocardial damage, QT prolongation."
