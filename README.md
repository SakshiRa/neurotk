NeuroTK is a dataset validation toolkit for neurology brain imaging in NIfTI format.

It addresses dataset quality assurance for brain imaging by reporting structural and metadata issues across images and optional labels. It does not perform training or inference and is CPU-only by design.

Install with:
pip install neurotk

Example:
neurotk validate --images imagesTr --labels labelsTr --out report.json

NeuroTK reports issues but does not modify or fix data.
