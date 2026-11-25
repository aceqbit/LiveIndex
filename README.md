# LiveIndex

LiveIndex is a hybrid search engine (in-house BM25, HNSW vectors, reciprocal rank fusion, optional ONNX reranker) in which every write gets an epoch and every query reads one consistent snapshot, so deletes, updates and permission changes take effect at the acknowledged epoch. It ships with `liveindex-check`, a stateful property-test suite that checks any retriever against six freshness and permission invariants.

## Problem

## Architecture

## Quick start

## Results (generated)

## Limits
