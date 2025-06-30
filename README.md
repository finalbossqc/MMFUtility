The .mmf file format is used to represent segmented voxelized models. This repo contains scripts to parse .mmf files into a human readable .json file format, which can be used for simulations. 

To use the scripts, first clone the repo

```git clone  ```

cd into the repository

```cd ./MMFUtility```

and create a new directory called BioModels

```mkdir BioModels```

Inside this directory is where you will put your .mmf file and the accompanying gzipped raw data. Suppose that we have the files

```BioModels/<model>.mmf```

and

```BioModels/<model>.raw.gz```

Simply run the bash script ```extract.sh```

```bash extract.sh <model>```
