#!/usr/bin/env bash

DIR="/workspace/semantic-modeling/"
cd $DIR

cp -a algorithm $DIR/containers/dev_environment/source_code/
cp -a exec $DIR/containers/dev_environment/source_code/
cp -a mira $DIR/containers/dev_environment/source_code/
cp -a gmtk $DIR/containers/dev_environment/source_code/
cp -a rdb2rdf $DIR/containers/dev_environment/source_code/
cp -a utils $DIR/containers/dev_environment/source_code/
cp -a Cargo.toml $DIR/containers/dev_environment/source_code/
