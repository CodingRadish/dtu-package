# Welcome
```bash
pip install git+https://github.com/CodingRadish/dtu-package.git
```
or 
```bash
C:/path/to/python.exe -m pip install --upgrade --force-reinstall git+https://github.com/FredslundMagnus/dtu-package.git
```

# Setup
Run this on your pc and it will show you commands you should use on the server:
```python
from dtu import setup

setup(github_link="https://github.com/CodingRadish/dtu-package.git", python="3.10.7", packages=["numpy", "pandas", "numba", "matplotlib"])
```
If this is your second project set
```python
first_time: bool = True
```

# Server
Example of main.py
```python
from dtu import Parameters, dtu

@dtu
class Defaults(Parameters):
    name: str = "local"
    instances: int = 1
    GPU: bool = False
    time: int = 3600

    b: float = 2.0
    a: int = 1
    d: str = "fd"

    def run(self, b: float, d: str, a: int) -> None:
        print(b,d, self.time)


Defaults.start()
```

For `list[int]` CLI overrides used through `MYARGS` (for example in `submit_cpu.sh`), you can now use either bracket syntax with spaces or a separator format:
```bash
--channels 16 32 64 --strides 1 2 2
-channels [16, 32, 64] -strides [1, 2, 2]
-channels 16-32-64 -strides 1-2-2
```

example of generate.py
```python
from main import Defaults

Defaults("Test1", b=4, d="dsf")
```
It will check types

## Create own package
https://packaging.python.org/en/latest/tutorials/packaging-projects/
