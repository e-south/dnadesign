"""Resolve package-page links and build source archives with portable metadata."""

import gzip
import os
import re
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def package_link(match):
    target = match.group(1)
    if target.startswith(("https://", "http://", "#", "mailto:")):
        return match.group(0)
    path = target.split("#", 1)[0]
    if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
        return f"](https://raw.githubusercontent.com/e-south/dnadesign/v{PROJECT['version']}/{target})"
    kind = "tree" if (ROOT / path).is_dir() else "blob"
    return f"](https://github.com/e-south/dnadesign/{kind}/v{PROJECT['version']}/{target})"


def write_catalog(destination):
    if PROJECT["name"] != "dnadesign-tools":
        return
    relative = Path("dnadesign/ops/catalog/builtin.json")
    if (ROOT / "docs/runbooks/README.md").is_file():
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from dnadesign.ops.catalog.packaged import compile_catalog

            content = compile_catalog(ROOT, version=PROJECT["version"])
        finally:
            sys.path.pop(0)
    else:
        # Source archives carry the catalog compiled from the release metadata.
        content = (ROOT / "src" / relative).read_text(encoding="utf-8")
    target = Path(destination) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


class PortableBuild(build_py):
    @staticmethod
    def _resource_package(package):
        parts = package.split(".")
        return (
            len(parts) >= 3
            and parts[0] == "dnadesign"
            and parts[2]
            in {
                "assets",
                "campaigns",
                "datasets",
                "docs",
                "examples",
                "images",
                "jobs",
                "presets",
                "resources",
                "styles",
                "workspaces",
            }
        )

    def find_package_modules(self, package, package_dir):
        # Top-level example/data shelves are resources, never executable packages.
        # Runtime packages such as <tool>.src.workspaces remain ordinary source.
        if self._resource_package(package):
            return []
        return super().find_package_modules(package, package_dir)

    def find_data_files(self, package, src_dir):
        # Approved resources are listed on the owning tool package. Do not let
        # setuptools' implicit .pyi/py.typed rules discover local shelf contents.
        if self._resource_package(package):
            return []
        return super().find_data_files(package, src_dir)

    def run(self):
        # setuptools copies current inputs but does not remove deleted resources.
        # Rebuild its staging directory so incremental wheels cannot retain them.
        staging = Path(self.build_lib)
        build_base = Path(self.get_finalized_command("build").build_base).resolve()
        resolved = staging.resolve()
        if (
            staging.is_symlink()
            or resolved == build_base
            or not resolved.is_relative_to(build_base)
            or resolved in (ROOT.resolve(), *ROOT.resolve().parents)
            or resolved.is_relative_to((ROOT / "src").resolve())
        ):
            raise ValueError("build_lib must be a staging subdirectory of build_base, outside the source tree")
        if staging.exists() and not self.dry_run:
            shutil.rmtree(staging)
        super().run()
        if not self.dry_run:
            write_catalog(self.build_lib)


class PortableSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        write_catalog(Path(base_dir) / "src")

    def make_archive(self, base_name, format, root_dir=None, base_dir=None, owner=None, group=None):
        if format != "gztar":
            return super().make_archive(base_name, format, root_dir, base_dir, owner, group)
        epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))

        def metadata(member):
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mtime = epoch
            member.pax_headers = {}
            return member

        archive_path = Path(f"{base_name}.tar.gz")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        source = Path(root_dir or ".") / (base_dir or ".")
        with archive_path.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    archive.add(source, arcname=base_dir or ".", filter=metadata)
        return str(archive_path)


setup(
    long_description=re.sub(r"\]\(([^)]+)\)", package_link, (ROOT / "README.md").read_text()),
    long_description_content_type="text/markdown",
    cmdclass={"sdist": PortableSdist, "build_py": PortableBuild},
)
