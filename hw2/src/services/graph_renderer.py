from __future__ import annotations

import html
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import graphviz
import pandas as pd

from src.data.classes.settings import GraphSettings
from src.data.loaders.relational_db import RelationalDatabase

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

FONT = "Libertinus Serif"
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
INK = "#000000"
MUTED = "#555555"
GRAPH_FONT = "Libertinus Serif"
NODE_STYLES = {
    "cohort": {"fillcolor": "#d9c8ae", "color": "#c0a378", "fontcolor": "#2e2e2e"},
    "user": {"fillcolor": "#d0d4da", "color": "#a5abb6", "fontcolor": "#2e2e2e"},
    "repo": {"fillcolor": "#f16667", "color": "#eb2728", "fontcolor": "#ffffff"},
    "pr": {"fillcolor": "#57c7e3", "color": "#23b3d7", "fontcolor": "#2e2e2e"},
    "issue": {"fillcolor": "#ffd86e", "color": "#edba39", "fontcolor": "#2e2e2e"},
}
RELATION_NAMES = {
    "push": "PUSHED",
    "open_pr": "OPENED",
    "open_issue": "OPENED",
    "issue_comment": "COMMENTED",
    "pr_review": "REVIEWED",
    "pr_review_comment": "REVIEW_COMMENTED",
    "author": "AUTHORED",
    "in": "IN",
}
NODE_LINE_CHARS = 14
USER_EDGE_LENGTH = "2.9"
IN_EDGE_LENGTH = "3.5"
NODE_MAX_LINES = 3


def edge_label(*, names: Sequence[str]) -> str:
    rows = "".join(f"<TR><TD><B>{html.escape(name)}</B></TD></TR>" for name in names)
    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1" BGCOLOR="white">{rows}</TABLE>>'


def circle_label(*, text: str) -> str:
    words = text.replace("/", "/ ").replace("-", "- ").split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip() if not current.endswith(("/", "-")) else f"{current}{word}"
        if len(candidate) <= NODE_LINE_CHARS:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    lines = [line if len(line) <= NODE_LINE_CHARS else line[: NODE_LINE_CHARS - 1] + "…" for line in lines]
    if len(lines) > NODE_MAX_LINES:
        lines = lines[:NODE_MAX_LINES]
        lines[-1] = lines[-1][: NODE_LINE_CHARS - 1] + "…"
    return "<<B>" + "<BR/>".join(html.escape(line) for line in lines) + "</B>>"


def shorten(
    *,
    text: object,
    limit: int,
) -> str:
    value = str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


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
            graph_attr={"rankdir": "TB", "splines": "ortho", "nodesep": "0.55", "ranksep": "0.9", "newrank": "true",
                        "dpi": GRAPH_DPI},
            node_attr={"shape": "plaintext", "fontname": ER_FONT, "fontsize": "13", "margin": "0"},
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
            f'CELLPADDING="4">{"".join(rows)}</TABLE>>'
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


class EntityGraphBuilder:
    def __init__(
        self,
        *,
        settings: GraphSettings,
        database: RelationalDatabase,
    ) -> None:
        self._settings = settings
        self._database = database
        self._users = database.users.set_index("user_id")
        self._repositories = database.repositories.set_index("repo_id")
        self._pull_requests = database.pull_requests.set_index("pr_id")
        self._issues = database.issues.set_index("issue_id")
        self._nodes: dict[str, tuple[str, str]] = {}
        self._edges: dict[tuple[str, str, str], int] = {}

    def build(self) -> object:
        start = pd.Timestamp(self._settings.start_time)
        end = pd.Timestamp(self._settings.end_time)
        user_ids = self._resolve_users(github_ids=self._settings.github_user_ids)
        self._add_pushes(user_ids=user_ids, start=start, end=end)
        self._add_pull_requests(user_ids=user_ids, start=start, end=end)
        self._add_issues(user_ids=user_ids, start=start, end=end)
        self._add_issue_comments(user_ids=user_ids, start=start, end=end)
        self._add_reviews(user_ids=user_ids, start=start, end=end)
        self._add_review_comments(user_ids=user_ids, start=start, end=end)
        self._drop_redundant_authorship()

        graph = graphviz.Digraph(
            "entities",
            engine="neato",
            graph_attr={
                "overlap": "false",
                "sep": "+18",
                "splines": "true",
                "outputorder": "edgesfirst",
                "size": "20,20",
                "bgcolor": "#ffffff",
                "dpi": GRAPH_DPI,
            },
            node_attr={"shape": "circle", "style": "filled", "fixedsize": "true", "width": "3.5", "penwidth": "4.6",
                       "fontname": GRAPH_FONT, "fontsize": "32"},
            edge_attr={"fontname": GRAPH_FONT, "fontsize": "30", "fontcolor": "#1a1a1a", "color": "#2b2b2b",
                       "penwidth": "3.6", "arrowsize": "1.8"},
        )
        for node_id, (kind, label) in self._nodes.items():
            graph.node(node_id, label=circle_label(text=label), **NODE_STYLES[kind])
        relations: dict[tuple[str, str], list[str]] = {}
        for (source, target, kind), count in self._edges.items():
            name = RELATION_NAMES[kind]
            relations.setdefault((source, target), []).append(name if count == 1 else f"{name} ×{count}")
        for (source, target), names in relations.items():
            if names == [RELATION_NAMES["in"]]:
                graph.edge(source, target, style="dashed", len=IN_EDGE_LENGTH)
                continue
            middle = f"label_{source}_{target}"
            half = str(float(USER_EDGE_LENGTH) / 2)
            graph.node(middle, label=edge_label(names=names), shape="plaintext", fixedsize="false", style="")
            graph.edge(source, middle, arrowhead="none", len=half)
            graph.edge(middle, target, len=half)
        logger.info("Entity graph: %d nodes, %d edges", len(self._nodes), len(self._edges))
        return graph

    def _resolve_users(self, *, github_ids: Sequence[int]) -> pd.Index:
        mapping = self._database.github_to_user_ids()
        missing = [github_id for github_id in github_ids if github_id not in mapping.index]
        if missing:
            raise ValueError(f"GitHub users not in Users.parquet: {missing}")
        user_ids = pd.Index([int(mapping[github_id]) for github_id in github_ids])
        for user_id in user_ids:
            self._user_node(user_id=user_id)
        return user_ids

    def _user_node(self, *, user_id: object) -> str | None:
        if pd.isna(user_id):
            return None
        row = self._users.loc[int(user_id)]
        kind = "cohort" if bool(row["is_cohort"]) else "user"
        node_id = f"u{int(user_id)}"
        self._nodes.setdefault(node_id, (kind, str(row["login"])))
        return node_id

    def _repo_node(self, *, repo_id: object) -> str | None:
        if pd.isna(repo_id):
            return None
        row = self._repositories.loc[int(repo_id)]
        node_id = f"r{int(repo_id)}"
        self._nodes.setdefault(node_id, ("repo", str(row["name"])))
        return node_id

    def _pr_node(self, *, pr_id: object) -> str | None:
        if pd.isna(pr_id):
            return None
        row = self._pull_requests.loc[int(pr_id)]
        node_id = f"p{int(pr_id)}"
        if node_id not in self._nodes:
            self._nodes[node_id] = ("pr", f"PR #{row['number']}")
            self._edge(source=node_id, target=self._repo_node(repo_id=row["repo_id"]), kind="in")
            self._edge(source=self._user_node(user_id=row["author_id"]), target=node_id, kind="author")
        return node_id

    def _issue_node(self, *, issue_id: object) -> str | None:
        if pd.isna(issue_id):
            return None
        row = self._issues.loc[int(issue_id)]
        if bool(row["is_pull_request"]) and not pd.isna(row["pull_request_id"]):
            return self._pr_node(pr_id=row["pull_request_id"])
        node_id = f"i{int(issue_id)}"
        if node_id not in self._nodes:
            label = "PR conv." if bool(row["is_pull_request"]) else "Issue"
            self._nodes[node_id] = ("issue", f"{label} #{row['number']}")
            self._edge(source=node_id, target=self._repo_node(repo_id=row["repo_id"]), kind="in")
            self._edge(source=self._user_node(user_id=row["author_id"]), target=node_id, kind="author")
        return node_id

    def _drop_redundant_authorship(self) -> None:
        opened = {(source, target) for source, target, kind in self._edges if kind in ("open_pr", "open_issue")}
        for source, target in opened:
            self._edges.pop((source, target, "author"), None)

    def _edge(
        self,
        *,
        source: str | None,
        target: str | None,
        kind: str,
    ) -> None:
        if source is None or target is None:
            return
        key = (source, target, kind)
        self._edges[key] = self._edges.get(key, 0) + 1

    @staticmethod
    def _window(
        *,
        frame: pd.DataFrame,
        actor: str,
        time: str,
        user_ids: pd.Index,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        return frame[frame[actor].isin(user_ids) & (frame[time] >= start) & (frame[time] < end)]

    def _add_pushes(self, *, user_ids: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> None:
        rows = self._window(frame=self._database.pushes, actor="user_id", time="created_at", user_ids=user_ids, start=start, end=end)
        for row in rows.itertuples(index=False):
            self._edge(source=self._user_node(user_id=row.user_id), target=self._repo_node(repo_id=row.repo_id), kind="push")

    def _add_pull_requests(self, *, user_ids: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> None:
        rows = self._window(frame=self._database.pull_requests, actor="author_id", time="created_at", user_ids=user_ids, start=start, end=end)
        for row in rows.itertuples(index=False):
            self._edge(source=self._user_node(user_id=row.author_id), target=self._pr_node(pr_id=row.pr_id), kind="open_pr")

    def _add_issues(self, *, user_ids: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> None:
        issues = self._database.issues
        plain = issues[~issues["is_pull_request"].fillna(False).astype(bool)]
        rows = self._window(frame=plain, actor="author_id", time="created_at", user_ids=user_ids, start=start, end=end)
        for row in rows.itertuples(index=False):
            self._edge(source=self._user_node(user_id=row.author_id), target=self._issue_node(issue_id=row.issue_id), kind="open_issue")

    def _add_issue_comments(self, *, user_ids: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> None:
        rows = self._window(frame=self._database.issue_comments, actor="author_id", time="created_at", user_ids=user_ids, start=start, end=end)
        for row in rows.itertuples(index=False):
            self._edge(source=self._user_node(user_id=row.author_id), target=self._issue_node(issue_id=row.issue_id), kind="issue_comment")

    def _add_reviews(self, *, user_ids: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> None:
        rows = self._window(frame=self._database.pr_reviews, actor="reviewer_id", time="submitted_at", user_ids=user_ids, start=start, end=end)
        for row in rows.itertuples(index=False):
            self._edge(source=self._user_node(user_id=row.reviewer_id), target=self._pr_node(pr_id=row.pr_id), kind="pr_review")

    def _add_review_comments(self, *, user_ids: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> None:
        rows = self._window(frame=self._database.pr_review_comments, actor="author_id", time="created_at", user_ids=user_ids, start=start, end=end)
        for row in rows.itertuples(index=False):
            self._edge(source=self._user_node(user_id=row.author_id), target=self._pr_node(pr_id=row.pr_id), kind="pr_review_comment")


class GraphReport:
    def __init__(
        self,
        *,
        settings: GraphSettings,
        database: RelationalDatabase,
        output_dir: Path,
    ) -> None:
        self._settings = settings
        self._database = database
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
        paths += render(
            graph=EntityGraphBuilder(
                settings=self._settings,
                database=self._database,
            ).build(),
            output_dir=self._output_dir,
            name="entity_graph",
        )
        for path in paths:
            logger.info("💾 %s", path)
        return paths
