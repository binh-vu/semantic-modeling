#!/bin/bash

set -e
set -o pipefail

export PYTHONPATH=$(pwd)/pysm

# remove all cache files
echo ">> remove debug folder"
rm -rf debug

# this file is run to generate clean data sources, karma model json & r2rml are generated and commited to the repository before
echo ">> museum_edm"
python -m preprocessing.museum_edm.x01_make_karma_sources
echo ">> museum_crm"
python -m preprocessing.museum_crm.x02_make_karma_sources