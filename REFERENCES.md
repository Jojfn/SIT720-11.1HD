# References (IEEE numbered style)

Entries [1]–[13] are transcribed from the reference list of the reproduced paper, so their
details are exact. Entries [14]–[22] are standard works cited in support of the Part 2 design.

---

[1] F. Montori, K. Liao, M. De Giosa, P. P. Jayaraman, L. Bononi, T. Sellis, and
D. Georgakopoulos, "A metadata-assisted cascading ensemble classification framework for automatic
annotation of open IoT data," *IEEE Internet of Things Journal*, vol. 10, no. 15, pp. 13401–13413,
Aug. 2023.

[2] F. Montori, K. Liao, P. P. Jayaraman, L. Bononi, T. Sellis, and D. Georgakopoulos,
"Classification and annotation of open Internet of Things datastreams," in *Proc. Int. Conf. Web
Information Systems Engineering (WISE)*, 2018, pp. 209–224.

[3] E. Alpaydin and C. Kaynak, "Cascading classifiers," *Kybernetika*, vol. 34, no. 4,
pp. 369–374, 1998.

[4] A. F. R. Rahman and M. C. Fairhurst, "Serial combination of multiple experts: A unified
evaluation," *Pattern Analysis and Applications*, vol. 2, no. 4, pp. 292–311, 1999.

[5] L. Rokach, "Ensemble-based classifiers," *Artificial Intelligence Review*, vol. 33, nos. 1–2,
pp. 1–39, 2010.

[6] A. Bagnall, J. Lines, A. Bostrom, J. Large, and E. Keogh, "The great time series
classification bake off: A review and experimental evaluation of recent algorithmic advances,"
*Data Mining and Knowledge Discovery*, vol. 31, no. 3, pp. 606–660, 2017.

[7] X. Li and J. Lin, "Linear time complexity time series classification with bag-of-pattern
features," in *Proc. IEEE Int. Conf. Data Mining (ICDM)*, 2017, pp. 277–286.

[8] J. Grabocka, N. Schilling, M. Wistuba, and L. Schmidt-Thieme, "Learning time-series
shapelets," in *Proc. 20th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2014,
pp. 392–401.

[9] P. Schäfer, "The BOSS is concerned with time series classification in the presence of noise,"
*Data Mining and Knowledge Discovery*, vol. 29, no. 6, pp. 1505–1530, 2015.

[10] F. J. Damerau, "A technique for computer detection and correction of spelling errors,"
*Communications of the ACM*, vol. 7, no. 3, pp. 171–176, 1964.

[11] J.-P. Calbimonte, Z. Yan, H. Jeung, Ó. Corcho, and K. Aberer, "Deriving semantic sensor
metadata from raw measurements," in *Proc. 5th Int. Workshop on Semantic Sensor Networks*, 2012,
pp. 33–48.

[12] P. M. James, R. J. Dawson, N. Harris, and J. Jonczyk, *Urban Observatory Environment*.
Newcastle upon Tyne, U.K.: Newcastle Univ., 2014, doi: 10.17634/154300-19.

[13] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine
Learning Research*, vol. 12, no. 85, pp. 2825–2830, 2011.

[14] P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features,"
in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2001, pp. 511–518.

[15] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[16] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *The Annals of
Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

[17] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, "Long-tail learning
via logit adjustment," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[18] G. E. Hinton, "Training products of experts by minimizing contrastive divergence," *Neural
Computation*, vol. 14, no. 8, pp. 1771–1800, 2002.

[19] L. I. Kuncheva and C. J. Whitaker, "Measures of diversity in classifier ensembles and their
relationship with the ensemble accuracy," *Machine Learning*, vol. 51, no. 2, pp. 181–207, 2003.

[20] M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for
classification tasks," *Information Processing & Management*, vol. 45, no. 4, pp. 427–437, 2009.

[21] T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*, vol. 27, no. 8,
pp. 861–874, 2006.

[22] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority
over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357,
2002.

---

## Where each is used

| Ref | Used for |
|---|---|
| [1] | The reproduced paper |
| [2] | The authors' earlier version of the annotation problem, which MACE extends |
| [3]–[5], [14] | Cascading and serial ensembles: the lineage MACE belongs to |
| [6]–[9] | Time-series classification baselines, and why BOPF/LTS/BOSS are out of scope |
| [10] | The Damerau–Levenshtein distance behind DDL-NLP |
| [11], [12] | Provenance of the Swissex and Urban Observatory data sets |
| [13], [15], [16] | Implementation of the bag-of-summaries classifiers |
| [17] | Logit adjustment, the long-tail correction trialled in Part 2 |
| [18] | Products of experts, the multiplicative pooling rule in MACE-SF |
| [19] | Ensemble diversity: why correlated members favour averaging over products |
| [20] | Macro versus weighted averaging, justifying the reported F1 |
| [21] | ROC/AUC, the metric the paper omits and Part 2 relies on |
| [22] | Class imbalance, the condition ThingSpeak's 108:1 distribution creates |

## Software

`scikit-learn` [13] provides the bag-of-summaries classifiers and evaluation metrics. `rapidfuzz`
(v3.14) supplies the Damerau–Levenshtein distance used by DDL-NLP; the authors used
`pyxdameraulevenshtein`, which no longer builds on current Python.
