---
title: "NeuroTK: Dataset Validation for Neurology Brain Imaging"
authors:
  - name: Sakshi Rathi
    affiliation: 1
affiliations:
  - name: University of Minnesota
    index: 1
date: 2026
bibliography: paper.bib
---

## Summary

NeuroTK is an open-source software toolkit for dataset validation and quality assurance of neurology brain imaging data stored in NIfTI format. The software is designed to surface structural, geometric, and annotation-related issues in imaging datasets prior to downstream analysis, modeling, or benchmarking. NeuroTK provides deterministic validation, standardized preprocessing, and machine- and human-readable reports, enabling reproducible auditing and documentation of dataset quality.

## Statement of Need

Neurology brain imaging datasets are often heterogeneous due to variations in acquisition protocols, scanners, reconstruction pipelines, and clinical workflows. Common issues include inconsistent voxel spacing, orientation mismatches, malformed affine matrices, missing or incomplete annotations, and undocumented preprocessing steps. These issues frequently remain undetected until late stages of analysis, where they can invalidate results or complicate reproducibility.

Existing medical imaging frameworks primarily focus on model development, training, or inference, and typically assume that input datasets are already clean and standardized. NeuroTK addresses this gap by providing a dedicated, lightweight toolkit for dataset-level and file-level validation and standardization. The software enables researchers and clinicians to explicitly identify, document, and correct dataset quality issues before downstream use, improving reproducibility, transparency, and reviewer confidence. NeuroTK is intended for use in research pipelines, benchmarking studies, dataset releases, and clinical research audits.

## References
