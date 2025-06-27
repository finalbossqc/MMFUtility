#! bash/sh

name=$1
cd $(dirname $0)

python3 ../src/unzip.py ../BioModels/${name}.raw.gz ../json
python3 ../src/extractraw.py ../BioModels/${name}.mmf ../json/${name}.raw --output ../json/${name}.json --coords