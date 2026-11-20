# Third-Party Notices

## CAFA-TTA Research Project

The MIT license in the root LICENSE file applies to the original
CAFA-TTA source code and associated documentation for which the
project author holds the necessary licensing rights.

Third-party materials remain subject to their respective licenses
and copyright notices.

---

## 1. TENT

CAFA-TTA evaluates TENT as a test-time adaptation baseline.

The local implementation follows the entropy minimization and
BatchNorm adaptation principles described in the original work.

The upstream copyright and license notice are retained below
to acknowledge the reference implementation.

### Reference

Wang, D., Shelhamer, E., Liu, S., Olshausen, B., and Darrell, T. (2021).

Tent: Fully Test-Time Adaptation by Entropy Minimization.

International Conference on Learning Representations (ICLR).

- Paper: https://arxiv.org/abs/2006.10726
- Repository: https://github.com/DequanWang/tent
- License: https://github.com/DequanWang/tent/blob/master/LICENSE

### Upstream License

```text
MIT License

Copyright (c) 2021 Dequan Wang and Evan Shelhamer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


---

## 2. External Software Dependencies

CAFA-TTA uses third-party Python packages, including PyTorch,
torchvision, MedMNIST, NumPy, SciPy, pandas, scikit-learn,
scikit-image, Pillow, and Matplotlib.

These dependencies retain their respective upstream licenses.

Dependency versions are recorded in:

- [requirements.txt](requirements.txt)
- [Frozen requirements](experiments/reproducibility/requirements_frozen.txt)

The project-level MIT license does not replace the licenses
of these external packages.

---

## 3. Dataset Licenses

PathMNIST and STARC-9 remain subject to their respective
dataset licenses and attribution requirements.

See [Dataset Licenses and Attribution](docs/DATASET_LICENSES.md).

---

## 4. Research Artifact Distribution

Model checkpoints, source prototypes, medical image datasets,
sample-linked metadata, and detailed prediction artifacts
are subject to separate publication and redistribution review.

This notice does not establish permission to redistribute
all third-party datasets or derived research artifacts.
