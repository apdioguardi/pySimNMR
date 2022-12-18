# pySimNMR

Simulation software for solid-state nuclear magnetic resonance (NMR) spectra. Frequency/field swept spectra of both single crystals and powders. Methods for 

## Getting Started

You will need to install python version 3.5 or higher to run this code, with 64bit python recommended for dealing with large arrays, especially for the calculation of powder spectra. As of 2020-09-01 the software has been tested with 3.8.3.

### Prerequisites

Dependencies include:
- numpy (tested with version 1.19.0)
- matplotlib (tested with version 3.2.2)
- lmfit (tested with version 1.0.1, with dependency package scipy 1.5.1)

### Installing

After downloading the files one can simply run the one of the simulation files, e.g., ```python plot_freq_spectrum_multisite.py``` after modifying the input parameters at the beginning of the file. This is, of course, not ideal and will be updated in the future to a graphical user interface that will allow one to more easily interact with the program and stop from editing the source code. However, as this is home-made software, I would like to make it available for researchers as soon as possible.

## Running the tests

No standardized testing is available yet, just try to run a simulation/fit and see what tracebacks appear.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Adam Paul Dioguardi** - *Initial work* - [apdioguardi](https://github.com/apdioguardi)
* **Piotr Lepucki** - *addition of offset, linear, and gaussian backgrounds*

## License

(need to check this with IFW) This project is licensed under the GNU-GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* We thank H.-J. Grafe for help with testing, basic physics, and mentorship. We thank N. J. Curro for mentorship and the pre-python version on which our the exact diagonalization methodology is based, and also for discussion of properly encoding the character of the parent eigenstates.