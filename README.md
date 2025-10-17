Sleep Stage Analysis Script

This repository contains a script for rapid and efficient sleep stage analysis using the YASA and Visbrain Python libraries.

YASA performs automatic sleep stage classification (Wake, N1, N2, N3, REM) from EEG, EOG, and EMG data.

Only epochs with a confidence level above 66% are retained. This threshold is arbitrary but supported by previous research (e.g.,
Carro-Domínguez, M., Huwiler, S., Oberlin, S. et al. Pupil size reveals arousal level fluctuations in human sleep. Nat Commun 16, 2070 (2025). https://doi.org/10.1038/s41467-025-57289-5
).

Epochs below the 66% confidence threshold are flagged as potential artifacts in Visbrain for manual review.

Annotations exported from Visbrain are merged with YASA’s automatic results to create the final hypnogram, providing an integrated view of automatic and manual scoring.

  Name                    Version                   Build  Channel
ipython                   8.18.1                   pypi_0    pypi
ipywidgets                8.1.7                    pypi_0    pypi
matplotlib                3.9.4                    pypi_0    pypi
matplotlib-inline         0.1.7                    pypi_0    pypi
mne                       1.8.0                    pypi_0    pypi
numpy                     2.0.2                    pypi_0    pypi
pandas                    2.3.3                    pypi_0    pypi
pip                       25.2               pyhc872135_0
pyqt5                     5.15.11                  pypi_0    pypi
scikit-learn              1.6.1                    pypi_0    pypi
scipy                     1.13.1                   pypi_0    pypi
seaborn                   0.13.2                   pypi_0    pypi
setuptools                80.9.0           py39haa95532_0
visbrain                  0.4.5                    pypi_0    pypi
vispy                     0.15.2                   pypi_0    pypi
yasa                      0.6.5                    pypi_0    pypi
