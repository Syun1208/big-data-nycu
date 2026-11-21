from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from pathlib import Path

import graphviz

from src.data.classes.settings import GraphSettings

logger = logging.getLogger(__name__)

GRAPH_FORMATS = ("png",)
GRAPH_DPI = "300"


@dataclass(frozen=True, slots=True)
class TableSchema:
    name: str
    primary_key: str
    row_time: str
    foreign_keys: tuple[tuple[str, str], ...]
    attributes: tuple[str, ...]


SCHEMA = (
    TableSchema("Users", "user_id", "first_seen_at", (), ("github_user_id", "login", "is_bot", "is_cohort")),
    TableSchema("Repositories", "repo_id", "first_seen_at", (), ("github_repo_id", "name", "org_id")),
    TableSchema(
        "PullRequests",
        "pr_id",
        "first_seen_at",
        (("repo_id", "Repositories"), ("author_id", "Users"), ("merged_by_id", "Users")),
        ("number", "created_at", "merged_at*", "closed_at*"),
    ),
    TableSchema(
        "Issues",
        "issue_id",
        "first_seen_at",
        (("repo_id", "Repositories"), ("author_id", "Users"), ("assignee_id", "Users"), ("pull_request_id", "PullRequests")),
        ("number", "is_pull_request", "created_at", "closed_at*"),
    ),
    TableSchema(
        "IssueComments",
        "comment_id",
        "created_at",
        (("issue_id", "Issues"), ("repo_id", "Repositories"), ("author_id", "Users")),
        ("is_pull_request", "body"),
    ),
    TableSchema(
        "PRReviews",
        "review_id",
        "first_seen_at",
        (("pr_id", "PullRequests"), ("repo_id", "Repositories"), ("reviewer_id", "Users")),
        ("state", "submitted_at", "dismissed_at*"),
    ),
    TableSchema(
        "PRReviewComments",
        "comment_id",
        "created_at",
        (
            ("pr_id", "PullRequests"),
            ("repo_id", "Repositories"),
            ("author_id", "Users"),
            ("pull_request_review_id", "PRReviews"),
            ("in_reply_to_id", "PRReviewComments"),
        ),
        ("path", "body"),
    ),
    TableSchema(
        "Pushes",
        "push_id",
        "created_at",
        (("user_id", "Users"), ("repo_id", "Repositories")),
        ("ref_type", "is_new_ref"),
    ),
)

ER_FONT = "DejaVu Sans"
ER_HEADER = "#e3ecf5:#a9c3dc"
ER_BORDER = "#7f9db9"
ER_LINE = "#333333"
ER_ICON = "#3d6d99"
ER_TYPE = "#555555"
ER_PRIMARY = "#d4a017"
ER_FOREIGN = "#c0392b"
ER_COLUMN = "#3fb6c9"
ER_JUNCTION = "0.11"
ER_LEFT_HUB = "Users"
ER_RIGHT_HUB = "Repositories"
ER_MIDDLE = ("Pushes", "IssueComments", "Issues", "PullRequests", "PRReviews", "PRReviewComments")

def render(
    *,
    graph: object,
    output_dir: Path,
    name: str,
) -> list[Path]:
    from graphviz import ExecutableNotFound

    paths = []
    for file_format in GRAPH_FORMATS:
        graph.format = file_format
        try:
            rendered = graph.render(
                filename=name,
                directory=str(output_dir),
                cleanup=True,
            )
        except ExecutableNotFound as error:
            (output_dir / name).unlink(missing_ok=True)
            source = output_dir / f"{name}.gv"
            graph.save(filename=str(source))
            raise RuntimeError(
                f"Graphviz `dot` is not installed. Run `conda install -c conda-forge graphviz`. The DOT source is in {source}."
            ) from error
        paths.append(Path(rendered))
    for path in paths:
        stamp_dpi(path=path)
    return paths


def stamp_dpi(*, path: Path) -> None:
    from PIL import Image

    dpi = int(GRAPH_DPI)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        image.load()
        image.save(path, dpi=(dpi, dpi))


class SchemaGraphBuilder:
    def __init__(self, *, column_types: dict[tuple[str, str], str]) -> None:
        self._column_types = column_types

    def build(self) -> object:  
        graph = graphviz.Digraph(
            "schema",
            graph_attr={"rankdir": "TB", "splines": "ortho", "nodesep": "0.3", "ranksep": "0.55", "newrank": "true",
                        "dpi": GRAPH_DPI},
            node_attr={"shape": "plaintext", "fontname": ER_FONT, "fontsize": "22", "margin": "0"},
            edge_attr={"color": ER_LINE, "penwidth": "1.4", "arrowsize": "1.1"},
        )
        tables = {table.name: table for table in SCHEMA}
        for name in (ER_LEFT_HUB, ER_RIGHT_HUB):
            graph.node(name, label=self._table_label(table=tables[name]))
        for name in ER_MIDDLE:
            graph.node(name, label=self._table_label(table=tables[name]), group=name)
        self._bus(graph=graph, hub=ER_LEFT_HUB, side="left")
        self._bus(graph=graph, hub=ER_RIGHT_HUB, side="right")
        with graph.subgraph() as column:
            column.attr(rank="same")
            for name in ER_MIDDLE:
                column.node(name)
            for upper, lower in zip(ER_MIDDLE, ER_MIDDLE[1:]):
                column.edge(upper, lower, style="invis")
        for parent, child in self._inner_pairs():
            if parent != child:
                graph.edge(parent, child, dir="both", arrowtail="teetee", arrowhead="crowtee", constraint="false")
        return graph

    @staticmethod
    def _bus(*, graph: object, hub: str, side: str) -> None:
        points = [f"{hub}_bus_{index}" for index in range(len(ER_MIDDLE))]
        with graph.subgraph() as bus:
            bus.attr(rank="same")
            for index, (point, child) in enumerate(zip(points, ER_MIDDLE)):
                junction = 0 < index < len(points) - 1
                size = ER_JUNCTION if junction else "0.01"
                bus.node(point, shape="point", width=size, height=size, color=ER_LINE, fillcolor=ER_LINE,
                         style="filled", label="", group=child)
            for upper, lower in zip(points, points[1:]):
                bus.edge(upper, lower, dir="none")
        middle = points[len(points) // 2]
        if side == "left":
            graph.edge(hub, middle, dir="back", arrowtail="teetee")
            for point, child in zip(points, ER_MIDDLE):
                graph.edge(point, child, arrowhead="crowtee")
        else:
            graph.edge(middle, hub, arrowhead="teetee")
            for point, child in zip(points, ER_MIDDLE):
                graph.edge(child, point, dir="back", arrowtail="crowtee")

    @staticmethod
    def _inner_pairs() -> list[tuple[str, str]]:
        pairs = []
        for table in SCHEMA:
            for _, target in table.foreign_keys:
                if target not in (ER_LEFT_HUB, ER_RIGHT_HUB) and (target, table.name) not in pairs:
                    pairs.append((target, table.name))
        return pairs

    def _row(self, *, table: str, column: str, marker: str) -> str:
        name = column.rstrip("*")
        sql_type = self._column_types.get((table, name), "")
        if marker == ER_FOREIGN and dict(next(t.foreign_keys for t in SCHEMA if t.name == table)).get(name) == table:
            sql_type = f"{sql_type} ↺"
        return (
            f'<TR><TD ALIGN="LEFT"><FONT COLOR="{marker}">◆</FONT> {html.escape(column)} '
            f'<FONT COLOR="{ER_TYPE}">{sql_type}</FONT></TD></TR>'
        )

    def _table_label(self, *, table: TableSchema) -> str:
        rows = [
            f'<TR><TD BGCOLOR="{ER_HEADER}" GRADIENTANGLE="270" ALIGN="LEFT">'
            f'<FONT COLOR="{ER_ICON}">▦</FONT> <B>{table.name}</B></TD></TR>'
        ]
        rows.append(self._row(table=table.name, column=table.primary_key, marker=ER_PRIMARY))
        rows += [self._row(table=table.name, column=column, marker=ER_FOREIGN) for column, _ in table.foreign_keys]
        rows += [self._row(table=table.name, column=column, marker=ER_COLUMN) for column in table.attributes]
        rows.append(self._row(table=table.name, column=table.row_time, marker=ER_COLUMN))
        return (
            f'<<TABLE BORDER="1" COLOR="{ER_BORDER}" BGCOLOR="#ffffff" CELLBORDER="0" CELLSPACING="0" '
            f'CELLPADDING="3">{"".join(rows)}</TABLE>>'
        )


def sql_type(*, arrow_type: str) -> str:
    if arrow_type.startswith("int"):
        return "BIGINT"
    if arrow_type in ("bool", "boolean"):
        return "BOOLEAN"
    if arrow_type.startswith("timestamp"):
        return "TIMESTAMP"
    if arrow_type in ("double", "float"):
        return "DOUBLE"
    return "TEXT"


def read_column_types(*, db_dir: Path) -> dict[tuple[str, str], str]:
    import pyarrow.parquet as pq

    types = {}
    for table in SCHEMA:
        for column in pq.read_schema(db_dir / f"{table.name}.parquet"):
            types[(table.name, column.name)] = sql_type(arrow_type=str(column.type))
    return types


class GraphReport:
    def __init__(
        self,
        *,
        settings: GraphSettings,
        output_dir: Path,
    ) -> None:
        self._settings = settings
        self._output_dir = output_dir

    def run(self) -> list[Path]:
        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        paths = render(
            graph=SchemaGraphBuilder(column_types=read_column_types(db_dir=self._settings.data.db_dir)).build(),
            output_dir=self._output_dir,
            name="schema_graph",
        )
        for path in paths:
            logger.info("💾 %s", path)
        return paths
