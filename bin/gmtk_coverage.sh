#!/bin/bash

cd gmtk

cargo tarpaulin --out Xml
pycobertura show --format html --output coverage.html cobertura.xml