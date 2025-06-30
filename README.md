The .mmf file format is used to represent segmented voxelized models. This repo contains scripts to parse .mmf files into a human readable .json file format, which can be used for simulations. 

To use the scripts, first clone the repo

```git clone  ```

cd into the repository

```cd ./MMFUtility```

The .mmf format consists of a metadata file (.mmf) along with a raw data file (.raw or .raw.gz or some other compressed format). If the .raw file is gzipped, use ```src/unzip.py``` to unzip the archive or do it manually.

```python3 src/unzip.py <rawdata>.raw.gz```

Then use the ```extractraw.py``` to transform the raw data into a json file.

```python3 src/extractraw.py <metadata>.mmf <rawdata>.raw.gz --output <outputdir>.json```

Include the ```--coords``` flag if you want to have the voxel coordinates and mappings in ```<outputdir>.json```

```python3 src/extractraw.py <metadata>.mmf <rawdata>.raw.gz --output <outputdir>.json --coords```
