#!/bin/bash

cd semantic_modeling

cargo tarpaulin --out Xml
pycobertura show --format html --output coverage.html cobertura.xml