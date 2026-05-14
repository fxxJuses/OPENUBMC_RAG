"""Schema-aware JSON parser for openUBMC MDS models, IPMI commands, and CSR configs."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.models.code_chunk import CodeChunk, Symbol


class JsonParser(BaseParser):
    @property
    def language(self) -> str:
        return "json"

    @property
    def supported_extensions(self) -> list[str]:
        return [".json", ".sr"]

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        rel_path = str(file_path)
        name = file_path.name

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_chunk(content, rel_path, repo_name)

        if name == "service.json":
            return self._parse_service_json(data, content, rel_path, repo_name)
        elif name == "model.json":
            return self._parse_model_json(data, content, rel_path, repo_name)
        elif name == "ipmi.json":
            return self._parse_ipmi_json(data, content, rel_path, repo_name)
        elif name == "types.json":
            return self._parse_types_json(data, content, rel_path, repo_name)
        elif file_path.suffix == ".sr":
            return self._parse_sr_file(data, content, rel_path, repo_name)
        else:
            return self._parse_generic_json(data, content, rel_path, repo_name)

    def _parse_service_json(
        self, data: dict, content: str, rel_path: str, repo_name: str
    ) -> list[CodeChunk]:
        """Parse MDS service.json — single chunk with dependency/interface metadata."""
        symbols = []
        deps = data.get("dependencies", [])
        if isinstance(deps, list):
            for dep in deps:
                if isinstance(dep, str):
                    symbols.append(Symbol(
                        name=dep, kind="dependency",
                        line_start=1, line_end=1, language="json",
                    ))
                elif isinstance(dep, dict) and "name" in dep:
                    symbols.append(Symbol(
                        name=dep["name"], kind="dependency",
                        line_start=1, line_end=1, language="json",
                    ))

        required = data.get("required", [])
        if isinstance(required, list):
            for iface in required:
                if isinstance(iface, str):
                    symbols.append(Symbol(
                        name=iface, kind="interface",
                        line_start=1, line_end=1, language="json",
                    ))

        metadata = {
            "service_name": data.get("name", ""),
            "version": data.get("version", ""),
            "dependencies": [s.name for s in symbols if s.kind == "dependency"],
            "required_interfaces": [s.name for s in symbols if s.kind == "interface"],
        }

        return [CodeChunk(
            chunk_id=str(uuid.uuid4()),
            content=content,
            file_path=rel_path,
            repo_name=repo_name,
            language="json",
            component_name=repo_name,
            start_line=1,
            end_line=len(content.splitlines()),
            chunk_type="mds_service",
            symbols=symbols,
            metadata=metadata,
        )]

    def _parse_model_json(
        self, data: dict, content: str, rel_path: str, repo_name: str
    ) -> list[CodeChunk]:
        """Parse MDS model.json — each top-level class key becomes a chunk."""
        chunks = []
        # Try to find the classes/models section
        models = data
        if "classes" in data:
            models = data["classes"]
        elif "models" in data:
            models = data["models"]

        if not isinstance(models, dict):
            return self._fallback_chunk(content, rel_path, repo_name)

        lines = content.splitlines()
        for class_name, class_def in models.items():
            if not isinstance(class_def, dict):
                continue

            # Find line range for this class
            start_line, end_line = self._find_json_key_range(lines, class_name)

            # Extract properties as symbols
            symbols = [Symbol(
                name=class_name, kind="class",
                line_start=start_line, line_end=end_line, language="json",
            )]
            props = class_def.get("properties", {})
            for prop_name in (props if isinstance(props, dict) else []):
                symbols.append(Symbol(
                    name=prop_name, kind="variable",
                    line_start=start_line, line_end=end_line, language="json",
                ))

            chunk_content = json.dumps({class_name: class_def}, indent=2, ensure_ascii=False)
            chunks.append(CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_content,
                file_path=rel_path,
                repo_name=repo_name,
                language="json",
                component_name=repo_name,
                start_line=start_line,
                end_line=end_line,
                chunk_type="mds_model",
                symbols=symbols,
                metadata={"mds_class": class_name},
            ))

        return chunks if chunks else self._fallback_chunk(content, rel_path, repo_name)

    def _parse_ipmi_json(
        self, data: dict, content: str, rel_path: str, repo_name: str
    ) -> list[CodeChunk]:
        """Parse MDS ipmi.json — each command under 'cmds' becomes a chunk."""
        chunks = []
        cmds = data.get("cmds", data)

        if not isinstance(cmds, dict):
            return self._fallback_chunk(content, rel_path, repo_name)

        lines = content.splitlines()
        for cmd_name, cmd_def in cmds.items():
            if not isinstance(cmd_def, dict):
                continue

            start_line, end_line = self._find_json_key_range(lines, cmd_name)

            symbols = [Symbol(
                name=cmd_name, kind="ipmi_command",
                line_start=start_line, line_end=end_line, language="json",
            )]
            netfn = cmd_def.get("netfn", "")
            cmd_byte = cmd_def.get("cmd", "")
            metadata = {
                "netfn": str(netfn),
                "cmd": str(cmd_byte),
            }

            chunk_content = json.dumps({cmd_name: cmd_def}, indent=2, ensure_ascii=False)
            chunks.append(CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_content,
                file_path=rel_path,
                repo_name=repo_name,
                language="json",
                component_name=repo_name,
                start_line=start_line,
                end_line=end_line,
                chunk_type="mds_ipmi_cmd",
                symbols=symbols,
                metadata=metadata,
            ))

        return chunks if chunks else self._fallback_chunk(content, rel_path, repo_name)

    def _parse_types_json(
        self, data: dict, content: str, rel_path: str, repo_name: str
    ) -> list[CodeChunk]:
        """Parse MDS types.json — each struct/enum definition becomes a chunk."""
        chunks = []
        defs = data.get("defs", data)

        if not isinstance(defs, dict):
            return self._fallback_chunk(content, rel_path, repo_name)

        lines = content.splitlines()
        for type_name, type_def in defs.items():
            if not isinstance(type_def, dict):
                continue

            start_line, end_line = self._find_json_key_range(lines, type_name)
            kind = type_def.get("type", "struct")

            symbols = [Symbol(
                name=type_name, kind="interface" if kind == "enum" else "class",
                line_start=start_line, line_end=end_line, language="json",
            )]

            chunk_content = json.dumps({type_name: type_def}, indent=2, ensure_ascii=False)
            chunks.append(CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_content,
                file_path=rel_path,
                repo_name=repo_name,
                language="json",
                component_name=repo_name,
                start_line=start_line,
                end_line=end_line,
                chunk_type="mds_type_def",
                symbols=symbols,
                metadata={"type_kind": kind},
            ))

        return chunks if chunks else self._fallback_chunk(content, rel_path, repo_name)

    def _parse_sr_file(
        self, data: dict, content: str, rel_path: str, repo_name: str
    ) -> list[CodeChunk]:
        """Parse CSR .sr file — extract topology and objects."""
        chunks = []

        if "ManagementTopology" in data:
            topology = data["ManagementTopology"]
            chunks.append(CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=json.dumps({"ManagementTopology": topology}, indent=2, ensure_ascii=False),
                file_path=rel_path,
                repo_name=repo_name,
                language="json",
                component_name=repo_name,
                start_line=1,
                end_line=len(content.splitlines()),
                chunk_type="csr_topology",
                symbols=[Symbol(name="ManagementTopology", kind="class", line_start=1, line_end=1, language="json")],
            ))

        objects = data.get("Objects", data.get("objects", {}))
        if isinstance(objects, dict):
            lines = content.splitlines()
            for obj_name, obj_def in objects.items():
                if not isinstance(obj_def, dict):
                    continue
                start_line, end_line = self._find_json_key_range(lines, obj_name)
                # Extract class prefix (e.g., "Eeprom_BCU" -> "Eeprom")
                class_prefix = obj_name.split("_")[0] if "_" in obj_name else obj_name

                symbols = [Symbol(
                    name=obj_name, kind="class",
                    line_start=start_line, line_end=end_line, language="json",
                )]
                chunk_content = json.dumps({obj_name: obj_def}, indent=2, ensure_ascii=False)
                chunks.append(CodeChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=chunk_content,
                    file_path=rel_path,
                    repo_name=repo_name,
                    language="json",
                    component_name=repo_name,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="csr_object",
                    symbols=symbols,
                    metadata={"class_prefix": class_prefix},
                ))

        return chunks if chunks else self._fallback_chunk(content, rel_path, repo_name)

    def _parse_generic_json(
        self, data: dict, content: str, rel_path: str, repo_name: str
    ) -> list[CodeChunk]:
        """Generic JSON file — single chunk."""
        return self._fallback_chunk(content, rel_path, repo_name)

    def _fallback_chunk(self, content: str, rel_path: str, repo_name: str) -> list[CodeChunk]:
        return [CodeChunk(
            chunk_id=str(uuid.uuid4()),
            content=content,
            file_path=rel_path,
            repo_name=repo_name,
            language="json",
            component_name=repo_name,
            start_line=1,
            end_line=len(content.splitlines()),
            chunk_type="config_block",
        )]

    def _find_json_key_range(self, lines: list[str], key: str) -> tuple[int, int]:
        """Best-effort: find the line range of a JSON key in the file."""
        start = 1
        end = len(lines)
        key_pattern = f'"{key}"'
        for i, line in enumerate(lines):
            if key_pattern in line:
                start = i + 1
                # Find closing brace
                depth = 0
                for j in range(i, len(lines)):
                    depth += lines[j].count("{") - lines[j].count("}")
                    if depth <= 0 and j > i:
                        end = j + 1
                        break
                break
        return start, end
