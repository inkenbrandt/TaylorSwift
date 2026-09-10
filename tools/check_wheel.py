"""Exercise a built wheel outside the checkout, including its typing contract."""

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheel_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "dist"
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Expected exactly one wheel in dist/")
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        environment = work / "venv"
        venv.create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["MPLBACKEND"] = "Agg"

        def run(*args: str) -> None:
            subprocess.run(args, cwd=work, env=env, check=True)

        run("uv", "pip", "install", "--python", str(python), str(wheels[0]), "mypy>=1.13")
        smoke = work / "smoke.py"
        smoke.write_text('''
from importlib.metadata import distribution
from pathlib import Path
import sys
import TaylorSwift as ts

assert "TaylorSwift.plotting" not in sys.modules
assert "numpy" not in sys.modules
dist = distribution("taylorswift-spectra")
assert ts.__version__ == dist.version
package = Path(ts.__file__).parent
assert package.is_relative_to(Path(sys.prefix))
assert (package / "py.typed").is_file()
assert (package / "__init__.pyi").is_file()
for name in ts.__all__:
    assert getattr(ts, name) is not None, name
fig, axes = ts.plot_cospectra([])
assert axes.shape == (2, 2)
fig.savefig("cospectra.png")
assert Path("cospectra.png").stat().st_size > 0
''', encoding="utf-8")
        run(str(python), "-I", str(smoke))
        consumer = work / "consumer.py"
        consumer.write_text('''
from typing import assert_type
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
import TaylorSwift as ts
from TaylorSwift import SiteConfig, SpectralResult, plot_cospectra

config = SiteConfig(z_measurement=3.0, z_canopy=0.3)
assert_type(config, ts.SiteConfig)
assert_type(ts.__version__, str)
results: list[SpectralResult] = []
fig, axes = plot_cospectra(results)
assert_type(fig, Figure)
assert_type(axes, NDArray[np.object_])
fig.savefig("example.pdf")
# Unused-ignore checking proves these errors are actually detected.
plot_cospectra(results).savefig("bad.pdf")  # type: ignore[attr-defined]
plot_cospectra("bad input")  # type: ignore[arg-type]
ts.missing_public_export()  # type: ignore[attr-defined]
SiteConfig(z_measurement="bad", z_canopy=0.3)  # type: ignore[arg-type]
''', encoding="utf-8")
        (work / "mypy.ini").write_text("[mypy]\npython_version = 3.12\nstrict = True\n", encoding="utf-8")
        run(str(python), "-m", "mypy", "--config-file", "mypy.ini", "consumer.py")


if __name__ == "__main__":
    main()
