# An Information-Driven Perspective to Evaluate Regression Algorithms

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code and experimental framework developed for the thesis: **An Information-Driven Perspective to Evaluate Regression Algorithms**. 

The project implements a novel information-based criterion to evaluate optimality in regression algorithms.

---

## 📂 Repository Structure

- `src/`: Core Python modules and functional implementations.
- `*.bash`: Shell scripts used to orchestrate experiments and call Python functions.
- `requirements.txt`: List of required Python dependencies.
- `results/`: (Optional) Directory where experimental outputs are stored.

---

## 🚀 Getting Started

### Prerequisites
- A Unix-based environment (Linux or macOS) is recommended for running the `.bash` scripts.
- Python 3.9 or higher.

### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/benjaminc001/info_teacher.git](https://github.com/benjaminc001/info_teacher.git)
   cd info_teacher
   ```

2. **Set up a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running Experiments

The experimental workflow is automated via Bash scripts located in the root directory. These scripts handle the sequence of execution and parameter passing to the Python modules in `src/`.

To run the main experiment pipeline, use:

```bash
chmod +x run_experiments.bash
./run_experiments.bash
```

> **Note:** Please ensure all data paths are correctly configured in the `.bash` files before execution.

---

## 🛠️ Technical Stack

- **Language:** Python 3.9
- **Automation:** Bash Scripting
- **Key Libraries:**  NumPy, PyTorch, Pandas, Numba, Matplotlib, SkLearn.

---

## 🎓 Citation & Acknowledgments

If you find this work useful for your research, please cite it as follows:

```bibtex
@misc{repositorio,
  author = {Benjamín Castro},
  title = {Experimental Setting Source Code},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/benjaminc001/info_teacher}},
  note = {Accessed: May 6, 2026}
}
```
---

## 📧 Contact

**Benjamin Castro** benjamincastro@ug.uchile.cl
Project Link: [https://github.com/benjaminc001/info_teacher](https://github.com/benjaminc001/info_teacher)
```