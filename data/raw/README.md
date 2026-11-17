# NSL-KDD raw data

Place `KDDTrain+.txt` and `KDDTest+.txt` in this directory. To download the
files from the pinned public mirror and verify their Git blob hashes, run:

```powershell
python scripts\inspect_nsl_kdd.py --download
```

The official NSL-KDD page reports that the original downloads are no longer
available. The files are therefore fetched from
[Jehuty4949/NSL_KDD](https://github.com/Jehuty4949/NSL_KDD), pinned at commit
`27bbbdf5c5cccf9970b8b6037a0f19dc09c5a0a9`. The dataset page states that it may
be redistributed provided the dataset and its reference paper are cited.

Citation:

> M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the
> KDD CUP 99 Data Set,” Second IEEE Symposium on Computational Intelligence for
> Security and Defense Applications (CISDA), 2009.

Do not edit the raw source files. Generated data profiles are written under
`reports/`, not into this directory.
