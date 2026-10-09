from __future__ import annotations
import ast
import importlib
import inspect
import shlex
from dataclasses import MISSING, dataclass, field
from inspect import signature
from sys import argv
from typing import Any, Callable, TypeVar, overload

try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform


_T = TypeVar("_T")


def _prepare_mutable_defaults(cls):
    for name, annotation in getattr(cls, "__annotations__", {}).items():
        if is_list_of_int_annotation(annotation) and type(getattr(cls, name, None)) is list:
            default_value = tuple(getattr(cls, name))
            setattr(cls, name, field(default_factory=lambda default=default_value: list(default)))
    return cls


@dataclass_transform(field_specifiers=(field,))
@overload
def dtu(cls: type[_T], **kwargs: Any) -> type[_T]:
    ...


@overload
def dtu(cls: None = None, **kwargs: Any) -> Callable[[type[_T]], type[_T]]:
    ...


def dtu(cls: type[_T] | None = None, **kwargs: Any) -> type[_T] | Callable[[type[_T]], type[_T]]:
    def wrap(inner_cls):
        prepared_cls = _prepare_mutable_defaults(inner_cls)
        return dataclass(prepared_cls, **kwargs)
    if cls is None:
        return wrap
    return wrap(cls)


def _get_transfer_format(module, class_name, args, kwargs, symbol="~") -> str:
    kw = str(kwargs)
    return str(module.__name__ + symbol + class_name + symbol + str(args) + symbol + kw).replace(" ", "@").replace("{", "#.").replace("}", ".#").replace(",", "#.#")


def _get_par_str(class_name, args, kwargs: dict[str, object]) -> str:
    _args = [colorize(arg) for arg in args]
    _kwargs = [f"<c>{key}</c><k>=</k>{colorize(value)}" for key, value in kwargs.items()]
    return f"<d>{class_name}</d><k>(</k>{'<k>,</k> '.join(_args+_kwargs)}<k>)</k>"


def check_primitives(args, kwargs: dict):
    for arg in list(args) + list(kwargs.values()):
        if type(arg) not in {int, str, bool, float}:
            raise Exception("Please only use the types {int, str, bool, float} in a Parameter")


def is_list_of_int_annotation(annotation: object) -> bool:
    return annotation == list[int] or annotation == "list[int]"


def is_list_of_int(value: object) -> bool:
    return type(value) is list and all(type(item) is int for item in value)


def serialize_for_cli(value: object) -> str:
    if is_list_of_int(value):
        return "-".join(str(item) for item in value)
    return shlex.quote(str(value))


def parse_for_type(value: str, expected_type: object) -> object:
    if expected_type is str or expected_type == "str":
        return value
    if expected_type in {int, "int"}:
        return int(ast.literal_eval(value))
    if expected_type in {float, "float"}:
        return float(ast.literal_eval(value))
    if expected_type in {bool, "bool"}:
        if value in {"true", "false"}:
            return value == "true"
        if value in {"True", "False"}:
            return value == "True"
        parsed = ast.literal_eval(value)
        if type(parsed) is bool:
            return parsed
        raise ValueError
    if is_list_of_int_annotation(expected_type):
        stripped = value.strip()
        if stripped.isdigit():
            return [int(stripped)]
        parts_by_space = stripped.split()
        if len(parts_by_space) > 1:
            cleaned = [part.rstrip(",") for part in parts_by_space]
            if all(part.isdigit() for part in cleaned):
                return [int(part) for part in cleaned]
        separators = {separator for separator in ("-", ".") if separator in value}
        if len(separators) == 1 and all(char.isdigit() or char in separators for char in value):
            parts = value.split(separators.pop())
            if any(part == "" for part in parts):
                raise ValueError
            return [int(part) for part in parts]
        parsed = ast.literal_eval(value)
        if not is_list_of_int(parsed):
            raise ValueError
        return parsed
    return value


class _Parameter(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)

        old_init = x.__init__

        def init(self, *args, **kwargs):
            check_primitives(args, kwargs)
            self._get_transfer_format = _get_transfer_format(
                inspect.getmodule(self),
                self.__class__.__name__,
                args,
                kwargs
            )
            self._par_str = _get_par_str(
                self.__class__.__name__,
                args,
                kwargs
            )
            old_init(self, *args, **kwargs)
        x.__init__ = init
        return x


class Parameter(metaclass=_Parameter):
    pass

class GPU:
    """
    GPU.v16
    GPU.v32
    GPU.a40
    GPU.a80
    """
    v16: GPU
    v32: GPU
    a40: GPU
    a80: GPU

    def __init__(self, name: str) -> None:
        self.name = name

    
GPU.v16 = GPU("v16")
GPU.v32 = GPU("v32")
GPU.a40 = GPU("a40")
GPU.a80 = GPU("a80")


def relive(_code: str) -> _Parameter:
    module_name, class_name,  args, kwargs = _code.replace("@", " ").replace("#.#", ",").replace(".#", "}").replace("#.", "{").split("~")
    module = importlib.import_module(module_name)
    _class = module.__getattribute__(class_name)
    return _class.__call__(*eval(args), **eval(kwargs))


def setup(github_link: str, python: str = "3.9.6", packages: list[str] = ["torch", "torchvision", "matplotlib"], first_time: bool = True):
    """
    github_link="https://github.com/FredslundMagnus/dtu-package.git"
    python="3.10.12" # see module available for newest
    packages=["torch", "torchvision", "matplotlib"]
    """
    name = github_link.split("/")[-1][:-4]
    newline = "\n"
    if first_time:
        print("""
Do these step by step:
cd ~
vi .profile
i (insert mode)
PATH=$PATH:$HOME/bin:. (Change to this)
esc esc :wq enter (quiting and saving)
mkdir bin
""")
    print(f"""
Copy all this and put it in the server terminal:
cd ~/Desktop
mkdir {name}
cd {name}
module load python3/{python}
python3 -m venv project-env
source project-env/bin/activate
python -m pip install git+https://github.com/FredslundMagnus/dtu-package.git{(newline + "python -m pip install " + " ".join(packages)) if packages else ""}
git config --global credential.helper store
[wandb login]
git clone {github_link}
yes | cp project-env/bin/dtu_server ~/bin/dtu
cd {name}
deactivate
dtu
""")


def check(params, features):
    for key, value in params.items():
        if key not in features:
            raise Exception(f'The feature "{key}" does not exist.')
        expected_type = features[key]
        if is_list_of_int_annotation(expected_type):
            if not is_list_of_int(value):
                raise Exception(f'The feature "{key}" should be of type list[int].')
            continue
        if value.__class__ not in {int, str, bool, float} and value.__class__.__class__ is not _Parameter:
            raise Exception(f"Problem with {key}: {value}. You can only user int, str, bool, float or objects with metaclass=Parameter")
        if value.__class__ != expected_type:
            if value.__class__ == int and (expected_type == float or expected_type == 'float'):
                params[key] = float(value)
            else:
                _class_ = expected_type.__name__ if hasattr(expected_type, "__name__") else expected_type
                if value.__class__.__name__ != _class_:
                    raise Exception(f'The feature "{key}" should be of type {_class_}.')
                # else:
                #     params[key] = value.__name__


def colorize(obj: object) -> str:
    if type(obj) in {int, float}:
        return f"<f>{obj}</f>"
    if type(obj) is str:
        return f'<j>"{obj}"</j>'
    if type(obj) is bool:
        return f"<e>{obj}</e>"
    if is_list_of_int(obj):
        return f"<j>{obj}</j>"


def print_parameters(values: dict[str, object], override: dict[str, object]) -> None:
    for key, value in override.items():
        values[key] = value
    a = 20
    print("""
<style>
c { color: #9cdcfe; font-family: 'Verdana', sans-serif;} /* VARIABLE */
d { color: #4EC9B0; font-family: 'Verdana', sans-serif;} /* CLASS */
e { color: #569cd6; font-family: 'Verdana', sans-serif;} /* BOOL */
f { color: #b5cea8; font-family: 'Verdana', sans-serif;} /* NUMBERS */
j { color: #ce9178; font-family: 'Verdana', sans-serif;} /* STRING */
k { font-family: 'Verdana', sans-serif;} /* SYMBOLS */
</style>
""")
    print("# Parameters\n")
    print("| PARAMETER".ljust(a) + "| TYPE".ljust(a) + "| VALUE".ljust(a) + "|")
    print("|".ljust(a, "-") + "|".ljust(a, "-") + "|".ljust(a, "-") + "|")
    for key, value in values.items():
        if key not in {"instances", "cls", "self", "isServer", "ID"}:
            print(f"| <c>{key}</c>".ljust(a) + f"| <d>{type(value).__name__}</d>".ljust(a) + f"| {colorize(value) if type(type(value)) is type else value._par_str}".ljust(a-1) + " |")
    print("\n# Output\n")
    print("```")


def createFolders(name, folders, file):
    for folder in folders:
        file.write(f"mkdir -p outputs/{name}/{folder}\n")


def change_parameter(params: dict[str, object]) -> dict[str, object]:
    for key, value in params.items():
        if value.__class__.__class__ is _Parameter:
            params[key] = value._get_transfer_format
    return params


def genExperiments(features, folders, file, name, n, gpu, **params):
    createFolders(name, folders, file)
    check(params, features)
    params = change_parameter(params)
    for i in range(n):
        params['ID'] = i
        arguments = " ".join(f"-{param_name} {serialize_for_cli(value)}" for param_name, value in params.items())
        file.write(f'bsub -o "outputs/{name}/Markdown/{name}_{i}.md" -J "{name}_{i}" -env MYARGS="-name {name}-{i} {arguments}" < submit_{"cpu" if gpu is None else ("gpu_" + gpu.name)}.sh\n')


class Parameters():
    name: str = "local"
    ID: int = 0
    folders: list[str] = []
    instances: int = 1
    GPU: None | GPU = None
    time: int = 3600
    isServer: bool = False
    __first__: bool = True

    def __post_init__(self):
        if Parameters.__first__:
            file = open('experiments.sh', 'w')
            file.write('#!/bin/sh\n')
            Parameters.__first__ = False
        else:
            file = open('experiments.sh', 'a')
        self.folders = list(self.folders)
        features_with_GPU, folders = dict(self.__annotations__), ['Markdown'] + self.folders
        features = {name: value for name, value in features_with_GPU.items() if name != "GPU"}
        genExperiments(features, folders, file, self.name, self.instances, self.GPU, **{k: v for k, v in self.__dict__.items() if k not in {"name", "instances", "folders", "GPU"}})
        file.close()

    @classmethod
    def start(cls) -> None:
        override = cls.override(argv[1:])
        values = {name: value for name, value in cls.__dict__.items() if name[0] != "_" and name != "run" and name != "GPU"}
        for name, dataclass_field in getattr(cls, "__dataclass_fields__", {}).items():
            if name in values:
                continue
            if dataclass_field.default is not MISSING:
                values[name] = dataclass_field.default
            elif dataclass_field.default_factory is not MISSING:
                values[name] = dataclass_field.default_factory()
        values['cls'] = cls
        values['self'] = cls
        values['isServer'] = len(argv[1:]) > 1
        args = [(override[name] if name in override else values[name]) for name in signature(cls.run).parameters]
        annotations = [(v.name, v.annotation) for v in signature(cls.run).parameters.values() if v.name not in {"cls", "self"}]

        isRunning: bool = cls.__module__ == "__main__"
        if isRunning:
            for name, annotation in annotations:
                if name == "isServer":
                    if annotation != bool and annotation != 'bool':
                        raise TypeError(f"The type of 'isServer' should be 'bool' in run!")
                elif cls.__annotations__[name] != annotation:
                    _class_ = cls.__annotations__[name].__name__ if hasattr(cls.__annotations__[name], "__name__") else cls.__annotations__[name]
                    raise TypeError(f"The type of '{name}' should be '{_class_}' in run!")
            if values['isServer']:
                print_parameters(values, override)
            cls.run(*args)
            print("```")

    @classmethod
    def override(cls, args) -> dict[str, object]:
        if len(args) == 0:
            return {}
        temp = {}
        i = 0
        while i < len(args):
            _key = args[i]
            if not _key.startswith("-"):
                raise ValueError(f"Expected argument key prefixed with '-', got: {_key}")
            i += 1
            key: str = _key.lstrip("-")
            _type = cls.__annotations__[key] if key != "ID" else int
            if i >= len(args):
                raise ValueError(f"Missing value for argument: {_key}")
            value = args[i]
            if is_list_of_int_annotation(_type):
                values = [value]
                if value.startswith("[") and not value.endswith("]"):
                    i += 1
                    while i < len(args):
                        values.append(args[i])
                        if args[i].endswith("]"):
                            break
                        i += 1
                    if not values[-1].endswith("]"):
                        raise ValueError(f"Could not parse list value for argument: {_key}")
                elif value.isdigit():
                    while i + 1 < len(args) and args[i + 1].isdigit():
                        i += 1
                        values.append(args[i])
                if len(values) > 1:
                    value = " ".join(values)
                else:
                    value = values[0]
            try:
                _type: type = eval(_type) if isinstance(_type, str) else _type
            except NameError:
                value = relive(value)
            else:
                value = parse_for_type(value, _type)
            if type(_type) is _Parameter:
                value = relive(value)
            temp[key] = value
            i += 1
        return temp
