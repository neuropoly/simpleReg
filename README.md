# SimpleReg

### Dependencies
- `bash` terminal
- Python >= 3.11

### Installation
1. Open a `bash` terminal and go into the directory in which you want to install the library.

2. Create the installation directory:
   ```bash
   mkdir simpleReg
   cd simpleReg
   ```

3. Create and activate a virtual environment using one of the following options (highly recommended):
   - venv
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   - conda env
   ```
   conda create -n myenv python=3.11
   conda activate myenv
   ```

4. Install this repository using one of the following options:
   - Git clone (for developpers)
   > **Note:** If you pull a new version from GitHub, make sure to rerun this command with the flag `--upgrade`
   ```bash
   git clone https://github.com/neuropoly/simplereg.git
   python3 -m pip install -e simplereg
   ```
   - PyPI installation (for inference only)
   ```bash
   python3 -m pip install simplereg
   ```

### Launch with an initial transform

You can start the app with an initial transform (ITK `.txt/.tfm`, numeric `.mat/.txt`, or `.npy` 4x4 matrix):

```bash
/Users/benjamindeleener/code/simpleReg/venv/bin/python scripts/start_app.py --initial-transform /path/to/transform.txt
```

By default, the initial transform resets the current transform stack before being applied.
Use this flag to append instead:

```bash
/Users/benjamindeleener/code/simpleReg/venv/bin/python scripts/start_app.py --initial-transform /path/to/transform.txt --append-initial-transform
```


